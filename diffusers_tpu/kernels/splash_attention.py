# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Splash attention over short key/value sequences.

Wraps ``jax.experimental.pallas.ops.tpu.splash_attention`` for one
device-local call: the query is padded to a whole number of query blocks,
the key/value sequence (a text prompt, never long) is attended as is.
Callers own the ``shard_map`` around it.
"""

import math
import os

import jax
from jax.experimental.pallas.ops.tpu import splash_attention as pallas_splash
import jax.numpy as jnp


_PROBE_PAD_KV = os.environ.get("DT_PROBE_PAD_KV") == "1"


def attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
) -> jax.Array:
  """Runs splash attention on device-local JAX arrays.

  Args:
    query: ``(B, heads, seq_q, dim)``.
    key: ``(B, heads, seq_kv, dim)``.
    value: ``(B, heads, seq_kv, dim)``.
    scale: Softmax scale; ``None`` means ``1/sqrt(dim)``.
    block_q: Query block rows; the query is padded to a multiple of it.
    block_kv: Key/value block rows (clamped to ``seq_kv``).
    block_kv_compute: Key/value compute rows (clamped to ``seq_kv``).

  Returns:
    Output with the query's shape.
  """
  scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
  query = query * scale_factor

  def _kernel_3d(q_3d, k_3d, v_3d):
    # Pad the query's sequence axis (1 once vmap strips the batch) up to
    # a whole number of query blocks; the output is sliced back below.
    q_orig_len = q_3d.shape[1]
    pad_len = (block_q - q_orig_len % block_q) % block_q
    if pad_len:
      q_3d = jnp.pad(q_3d, [(0, 0), (0, pad_len), (0, 0)])

    # Timing probe only (DT_PROBE_PAD_KV=1): zero-pads k/v to a whole
    # number of kv blocks to measure the cost of a partial last block
    # (57,344 tokens ran 5.05 s/step vs 5.44 s at 57,290). Output is wrong
    # because the zero keys take softmax weight; read it only for timing.
    if _PROBE_PAD_KV:
      kv_pad = (block_kv - k_3d.shape[1] % block_kv) % block_kv
      if kv_pad:
        k_3d = jnp.pad(k_3d, [(0, 0), (0, kv_pad), (0, 0)])
        v_3d = jnp.pad(v_3d, [(0, 0), (0, kv_pad), (0, 0)])

    padded_q_seq_len = q_3d.shape[1]
    kv_seq_len = k_3d.shape[1]
    mask = pallas_splash.MultiHeadMask(
        [
            pallas_splash.FullMask((padded_q_seq_len, kv_seq_len))
            for _ in range(q_3d.shape[0])
        ]
    )
    bsizes = pallas_splash.BlockSizes(
        block_q=min(block_q, padded_q_seq_len),
        block_kv=min(block_kv, kv_seq_len),
        block_kv_compute=min(block_kv_compute, kv_seq_len),
    )
    splash_kernel = pallas_splash.make_splash_mha(
        mask=mask, block_sizes=bsizes, head_shards=1, q_seq_shards=1
    )
    out = splash_kernel(q_3d, k_3d, v_3d)
    return out[:, :q_orig_len, ...]

  return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(query, key, value)
