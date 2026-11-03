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

"""Attention on the torchax runtime: the custom splash kernel behind SDPA.

``configure`` returns a callable with
``torch.nn.functional.scaled_dot_product_attention``'s signature for a
model to hand its attention processors. The kernels themselves know
nothing of torch or a mesh; this file adds the tensor conversion through
``env``, the ``shard_map`` over the mesh, k-smoothing, the route between
self- and cross-attention, and block sizes autotuned per shape.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
import functools
import math

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import torch
import torchax

from . import block_sizes
from . import splash as custom_splash_attention
from . import splash_attention

# One SDPA call: (query, key, value, attn_mask=None, dropout_p=0.0,
# is_causal=False, scale=None, enable_gqa=False) -> output, [B, H, S, D].
Sdpa = Callable[..., torch.Tensor]

KERNELS = ("custom_splash",)

# A key/value sequence longer than this is video tokens (e.g. 81 frames x
# 1024 = 82944) and runs the custom kernel; a shorter one is a text prompt
# (e.g. 512) and runs splash.
_MAX_SPLASH_KV_LEN = 10000

# Modules with this many heads or fewer (the VAE's) run plain attention:
# the Pallas kernels' VMEM model is for transformer shapes.
_MAX_PLAIN_ATTENTION_HEADS = 8


def _plain_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float | None,
) -> jax.Array:
  """Dot-product attention with an fp32 softmax, for few-head modules."""
  s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
  attn = jnp.einsum("bhsd,bhtd->bhst", query * s, key)
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)
  return jnp.einsum("bhst,bhtd->bhsd", attn, value)


def _partition_specs(
    mesh: jax.sharding.Mesh,
    query_shape: tuple[int, ...],
    self_attention: bool,
) -> tuple[P, P, P]:
  """Places a ``(B, H, S, D)`` attention call on the mesh.

  Self-attention shards the heads (axis 1), with key and value alike.
  Cross-attention shards the query's sequence (axis 2) and replicates the
  short key/value side. The output is constrained to the sequence layout
  in both cases.

  Args:
    mesh: The runner's mesh.
    query_shape: ``(B, H, S, D)``.
    self_attention: Whether the key/value sequence is the video tokens.

  Returns:
    ``(query_spec, key_value_spec, output_constraint)``.
  """
  batch_size, num_heads = query_shape[0], query_shape[1]
  if num_heads < mesh.size:
    # Fewer heads than devices (e.g. a VAE): replicate.
    return P(), P(), P()
  names = mesh.axis_names
  if len(names) >= 3:
    # Three or more axes: heads on the second, sequence on the third.
    first, heads, seq = names[:3]
    if self_attention:
      return (
          P(first, heads, seq, None),
          P(first, heads, None, None),
          P(first, None, (heads, seq), None),
      )
    spec = P(first, None, (heads, seq), None)
    return spec, P(first, None, None, None), spec
  if len(names) == 1:
    # Pure tensor parallel.
    batch, shard = None, names[0]
  elif batch_size >= mesh.shape[names[0]]:
    # Batched CFG: the batch on the first axis, heads on the second.
    batch, shard = names[0], names[1]
  else:
    # Batch smaller than the first axis (e.g. sequential CFG): heads
    # across all devices.
    batch, shard = None, tuple(names)
  if self_attention:
    spec = P(batch, shard, None, None)
    return spec, spec, P(batch, None, shard, None)
  spec = P(batch, None, shard, None)
  return spec, P(batch, None, None, None), spec


def _on_mesh(
    local_attention: Callable[..., jax.Array],
    mesh: jax.sharding.Mesh,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    self_attention: bool,
) -> jax.Array:
  """Runs a device-local kernel over the mesh with ``shard_map``."""
  query_spec, key_value_spec, output_constraint = _partition_specs(
      mesh, query.shape, self_attention
  )
  sharded = shard_map(
      local_attention,
      mesh=mesh,
      in_specs=(query_spec, key_value_spec, key_value_spec),
      out_specs=query_spec,
      check_rep=False,
  )
  out = sharded(query, key, value)
  return jax.lax.with_sharding_constraint(out, output_constraint)


def configure(
    mesh: jax.sharding.Mesh,
    env: torchax.tensor.Environment,
    kernel: str = "custom_splash",
    block_q: int | None = None,
    block_kv: int | None = None,
    block_kv_compute: int | None = None,
    use_fixed_m: bool = False,
    qk_fp8: bool = False,
) -> Sdpa:
  """Returns the SDPA callable for the configured attention.

  Block sizes are resolved once per ``(seq, kv_seq, head_dim)``, on the
  first call (or trace) with that shape: a field the config pins wins,
  else the autotuner's pick for the detected TPU, else a default.

  Args:
    mesh: The mesh the kernel runs over.
    env: The torchax environment, for the torch/JAX conversion.
    kernel: One of ``KERNELS``.
    block_q: Pinned query block, or None.
    block_kv: Pinned key/value block, or None.
    block_kv_compute: Pinned key/value compute block, or None.
    use_fixed_m: Hybrid fixed-m softmax in self-attention.
    qk_fp8: E4M3 Q/K into the QK dot.

  Returns:
    ``(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
    scale=None, enable_gqa=False) -> output``, each ``[B, H, S, D]``.

  Raises:
    ValueError: If ``kernel`` names no kernel on this runtime.
  """
  if kernel not in KERNELS:
    raise ValueError(
        f"attention {kernel!r} is not a torchax kernel; use one of"
        f" {list(KERNELS)}"
    )
  pinned = {
      "block_q": block_q,
      "block_kv": block_kv,
      "block_kv_compute": block_kv_compute,
  }
  # The softmax variant moves the optimum (fixed-m halves the VMEM budget
  # on gens with a fixed-m derate; fp8 Q/K halve their tile bytes), so the
  # search and its measured calibrations key on it. Without it, v6e fixed-m
  # picks a DMA-cliff tile (9472/2048, 2.4x slower).
  kernel_mode = "residual-fixedm" if use_fixed_m else "residual"
  if qk_fp8:
    kernel_mode = "residual-qkfp8"
  spec = block_sizes.detect_spec()
  resolved: dict[tuple[int, int, int], Mapping[str, int]] = {}

  def _blocks(
      seq_len: int, kv_seq_len: int, head_dim: int
  ) -> Mapping[str, int]:
    shape = (seq_len, kv_seq_len, head_dim)
    if shape in resolved:
      return resolved[shape]
    blocks = block_sizes.resolve_block_sizes(
        seq_len, kv_seq_len, head_dim, pinned, kernel_mode, spec
    )
    print(
        f"  [splash autotune] seq={seq_len} kv_seq={kv_seq_len} "
        f"head_dim={head_dim} -> {blocks}",
        flush=True,
    )
    resolved[shape] = blocks
    return blocks

  def _sdpa(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      attn_mask: torch.Tensor | None = None,
      dropout_p: float = 0.0,
      is_causal: bool = False,
      scale: float | None = None,
      enable_gqa: bool = False,
  ) -> torch.Tensor:
    del attn_mask, dropout_p, is_causal, enable_gqa  # Not supported.
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    if jquery.shape[1] <= _MAX_PLAIN_ATTENTION_HEADS:
      return env.j2t_iso(_plain_attention(jquery, jkey, jvalue, scale))
    # (B, H, S, D): axis 2 is the sequence.
    blocks = _blocks(jquery.shape[2], jkey.shape[2], jquery.shape[-1])
    # k-smoothing: zero-mean K rows make every row's mean logit exactly 0,
    # hence rowmax >= 0, which the fixed-m no-flush guarantee rests on.
    # Softmax is invariant to it.
    jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
    if jkey.shape[2] > _MAX_SPLASH_KV_LEN:
      local_attention = functools.partial(
          custom_splash_attention.attention,
          scale=scale,
          use_fixed_m=use_fixed_m,
          qk_fp8=qk_fp8,
          **blocks,
      )
      self_attention = True
    else:
      local_attention = functools.partial(
          splash_attention.attention, scale=scale, **blocks
      )
      self_attention = False
    return env.j2t_iso(
        _on_mesh(local_attention, mesh, jquery, jkey, jvalue, self_attention)
    )

  return _sdpa
