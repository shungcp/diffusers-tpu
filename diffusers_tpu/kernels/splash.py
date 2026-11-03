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

"""Custom Pallas flash attention kernel for TPU.

A hand-written Pallas kernel using ``exp2`` fused softmax, ``fori_loop`` KV
tiling, and a hybrid per-head fixed-m / online softmax. One kernel body
(``_flash_attention_kernel``) serves both the normalized production path
and the ring-attention residual path via a compile-time ``emit_residuals``
flag. Two device-local entry points wrap it: ``attention`` returns the
normalized output, ``residual_attention`` the unnormalized partials a ring
merges across KV shards. Neither knows about a mesh or a runtime: callers
own the ``shard_map`` and the collectives around them.

The host-side derivation of the fixed-m bound lives in
``fixed_m_bound``; this module only consumes the
``(2, heads)`` prefetch operand it produces.
"""

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from . import fixed_m_bound
from . import fp8_cast

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))


@functools.cache
def _splash_vmem_limit_bytes() -> int:
  """Mosaic scoped-VMEM ceiling per pallas_call, adapted to the device.

  Set to 2x physical core VMEM: 720p tiles need ~57 MiB, above the
  32 MiB Mosaic default on libtpu >= 0.0.41. Scales across generations.

  The effective ceiling for a `pallas_call` is its own
  `vmem_limit_bytes` if set, else `xla_tpu_scoped_vmem_limit_kib` if set,
  else 32 MiB (verified on v6e). The per-op value always wins.

  Returns:
    The scoped-VMEM limit in bytes.
  """
  devices = jax.devices()
  if devices and devices[0].platform == "tpu":
    vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
  else:
    vmem_bytes = 64 * 2**20  # v7-sized fallback for CPU-only tracing.
  return 2 * vmem_bytes


def _exp2(x: jax.Array) -> jax.Array:
  return jnp.exp2(x)


import os as _os
EXP_BF16 = _os.environ.get("DT_EXP_BF16") == "1"


# Tried: folding RoPE into this kernel. Slower (23.20 vs 22.04 ms per call
# on v7x-8): K is re-rotated once per q-block, so the fold repeats work that
# a single pass outside the kernel does once.


# pylint: disable=too-few-public-methods
class BlockSizes:
  """Pallas grid tiling for one splash-attention call.

  Attributes:
    block_q: Query rows per grid step.
    block_kv: Key/value rows per grid step.
    block_kv_compute: Key/value rows per inner compute step; defaults
      to ``block_kv``, i.e. one compute step per KV tile.
  """

  __slots__ = ("block_q", "block_kv", "block_kv_compute")

  def __init__(
      self, block_q: int, block_kv: int, block_kv_compute: int | None = None
  ):
    self.block_q = block_q
    self.block_kv = block_kv
    self.block_kv_compute = (
        block_kv_compute if block_kv_compute is not None else block_kv
    )


# pylint: disable=too-many-statements
def _flash_attention_kernel(
    mk_ref: Any,  # (2, num_heads) f32 prefetch: [0]=ceil(CS bound), [1]=flag
    q_ref: Any,  # (bq, head_dim_qk) bf16 (or e4m3 when quantized outside)
    k_ref: Any,  # (bkv, head_dim_qk) bf16 (ditto)
    v_ref: Any,  # (bkv, head_dim_v) bf16
    *refs: Any,  # (m, l, o) cross-j scratch then the real outputs
    mask_value: float,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    kv_seq_len: int,
    use_fixed_m: bool,
    emit_residuals: bool,
    dynamic_kv: bool = False,
) -> None:
  """Hybrid fixed-m / online-softmax flash attention over one (h, i) tile.

  One body serves both output modes; ``emit_residuals`` is compile-time
  (bound via ``functools.partial``), so the epilogue branch resolves at
  trace with no runtime cost:

    - False: normalize ``o / l`` and write the attention output.
    - True (ring): emit unnormalized ``o`` and running ``(m, l)`` at
      ``log2`` scale for an outer LSE-merge across KV shards.

  With ``use_fixed_m``, per-head flag ``mk_ref[1]`` routes in-window heads
  onto the fixed-m fast path (``m`` = integral CS bound, no reduce_max /
  alpha-rescale); outliers keep online softmax. Under ``emit_residuals`` a
  fixed head's ``m`` is an upper BOUND not the rowmax, so the merge must be
  m-invariant (LSE space). Matmuls: bf16 x bf16, f32 accumulate.
  """
  float32 = jnp.float32
  # Pallas passes inputs then outputs positionally. The first three
  # "outputs" are the cross-j (m, l, o) scratch, declared as outputs with
  # constant index maps so they persist across the j sweep; the rest are
  # the real outputs, whose count depends on ``emit_residuals``.
  m_scratch_ref, l_scratch_ref, o_scratch_ref, *out_refs = refs
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(
        f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}"
    )
  sv_dims = (((0,), (0,)), ((), ()))  # V . s contracts over the kv sub-block.

  h = pl.program_id(0)
  j = pl.program_id(2)

  # Per-head dispatch via scalar prefetch; branch at body level only
  # (per-step conditionals are far more expensive).
  is_fixed = mk_ref[1, h] > 0.5 if use_fixed_m else False

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    if use_fixed_m:

      @pl.when(is_fixed)
      def init_fixed():
        # m = the head's integral Cauchy-Schwarz bound, prefetched as a
        # scalar. Deriving it per q-row in-kernel would expand the whole
        # (bq, head_dim) tile to f32 and reduce it on every (head,
        # q-block) — costly in bf16 and far worse when q_ref is e4m3.
        # The per-head max is a valid m for every row in the tile: the
        # eligibility gate already bounds it inside the no-flush window
        # (see FIXED_M_SAFE_BOUND), and m is a common scale that
        # cancels in o/l. Stored as (m - C) so the hot loop stays
        # exp2(qk - m).
        m_scratch_ref[...] = jnp.full_like(
            m_scratch_ref, mk_ref[0, h] - fixed_m_bound.FIXED_M_RECENTER
        )

      @pl.when(jnp.logical_not(is_fixed))
      def init_online():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

    else:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

  def _update(slice_k, fixed: bool, mask_start=None):
    """One KV sub-block update; ``fixed`` selects the softmax variant.

    Bit-identical arithmetic for the normalized and residual paths — the
    two output modes diverge only in the epilogue, never here.

    ``mask_start`` (dynamic_kv only): the sub-block's first kv position.
    Rows at or past the runtime kv length (``mk_ref[2, 0]``) score
    ``mask_value``, so a padded sequence attends exactly like the
    unpadded one.
    """
    q = q_ref[...]  # (bq, head_dim) bf16
    k_chunk = k_ref[slice_k, :]  # (bkv_compute, head_dim) bf16
    v_chunk = v_ref[slice_k, :]  # (bkv_compute, head_dim_v) bf16
    # QK matmul: bf16 x bf16 -> f32 accumulate, (bkv_compute, bq).
    qk = lax.dot_general(
        k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32
    )
    if mask_start is not None:
      pos = mask_start + lax.broadcasted_iota(jnp.int32, qk.shape, 0)
      qk = jnp.where(pos < mk_ref[2, 0].astype(jnp.int32), qk, mask_value)

    if fixed:
      # Fixed m is a per-head constant: no reduce_max and no alpha
      # rescale — plain accumulate at scale 2^-m. Read it from scalar
      # prefetch, not m_scratch: the scratch tile is (NUM_SUBLANES, bq)
      # f32 and only row 0 carries information.
      if EXP_BF16:
        # Lossy option (DT_EXP_BF16=1): compute exp2 in bf16. At ~1e11 exps
        # per block the EUP can outlast the matmuls, and s is cast to bf16
        # for the PV matmul anyway; only the l sum loses precision.
        s_cast = _exp2((qk - (mk_ref[0, h] - fixed_m_bound.FIXED_M_RECENTER))
                       .astype(jnp.bfloat16))
        s_curr = s_cast.astype(float32)
      else:
        s_curr = _exp2(qk - (mk_ref[0, h] - fixed_m_bound.FIXED_M_RECENTER))
        s_cast = s_curr.astype(jnp.bfloat16)
      # l comes from an f32 VPU reduce, not the textbook "append a ones
      # row to V and read l out of the PV matmul": s never leaves VREGs,
      # so the reduce is register work rather than a pass over the tile,
      # while the ones row pushes head_dim_v 128 -> 129 into a worse MXU
      # tile. Measured 21.87 ms (this) vs 22.61 (matmul) vs 23.02 (bf16).
      l_curr = s_curr.sum(axis=0, keepdims=True).astype(float32)
      o_curr = lax.dot_general(
          v_chunk, s_cast, sv_dims, preferred_element_type=float32
      )
      # o_prev read here, not before the matmuls: a longer live range
      # costs VMEM pressure across both dots.
      o_scratch_ref[:] = o_scratch_ref[:] + o_curr
      l_scratch_ref[...] = l_scratch_ref[...] + l_curr
      return

    # Online softmax update (all f32).
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = _exp2(qk - m_next[0:1])
    # bf16 explicitly: the q ref may be e4m3, but PV stays bf16.
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v_chunk,
        s_cast,
        sv_dims,
        preferred_element_type=float32,
    )
    # alpha[0:1] rescales the running o to the new max before adding.
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  def _compute_body(fixed: bool):
    def compute_body(kv_compute_index, _):
      _update(pl.ds(kv_compute_index * bkv_compute, bkv_compute), fixed)

    return compute_body

  compute_online = _compute_body(False)

  def compute_masked(kv_compute_index, _):
    start = kv_compute_index * bkv_compute
    _update(pl.ds(start, bkv_compute), False, mask_start=j * bkv + start)

  def last_compute_body(kv_compute_index):
    # Final partial sub-block when kv_seq_len % bkv_compute != 0.
    slice_len = kv_seq_len % bkv_compute
    start = kv_compute_index * bkv_compute
    _update(pl.ds(start, slice_len), False,
            mask_start=(j * bkv + start) if dynamic_kv else None)

  assert bkv % bkv_compute == 0

  def _run_main(body_fn):
    lax.fori_loop(0, bkv // bkv_compute, body_fn, None, unroll=True)

  def _run_last():
    # The last KV block always runs the online body (see the note below).
    body = compute_masked if dynamic_kv else compute_online
    if kv_seq_len % bkv == 0:
      lax.fori_loop(0, bkv // bkv_compute, body, None, unroll=True)
    else:
      # The last iter may contain padding; handle it separately.
      remain = kv_seq_len % bkv
      iters = (remain + bkv_compute - 1) // bkv_compute
      if remain % bkv_compute == 0:
        lax.fori_loop(0, iters, body, None, unroll=True)
      else:
        lax.fori_loop(0, iters - 1, body, None, unroll=True)
        last_compute_body(iters - 1)

  # dynamic_kv: a non-last block that reaches past the runtime kv length
  # runs the masked online body for every head, like the last block does
  # (exact for fixed heads too, see below). Blocks wholly before it are
  # untouched, so an unpadded call compiles to the same schedule.
  if dynamic_kv:
    has_pad = (j + 1) * bkv > mk_ref[2, 0].astype(jnp.int32)
    clean = jnp.logical_not(has_pad)

    @pl.when((j != grid_width - 1) & has_pad)
    def body_masked():
      _run_main(compute_masked)
  else:
    clean = True

  if use_fixed_m:
    compute_fixed = _compute_body(True)

    @pl.when((j != grid_width - 1) & clean & is_fixed)
    def body_fixed():
      _run_main(compute_fixed)

    @pl.when((j != grid_width - 1) & clean & jnp.logical_not(is_fixed))
    def body_online():
      _run_main(compute_online)

  else:

    @pl.when((j != grid_width - 1) & clean)
    def body():
      _run_main(compute_online)

  # Last KV block runs online for EVERY head (exact: fixed heads arrive at
  # scale 2^-m with m_scratch=m, which alpha-rescale absorbs). Must be a
  # single body: a dual fixed+online last branch degrades the Mosaic
  # schedule ~2.5x at most KV edge widths.
  @pl.when(j == grid_width - 1)
  def last_body():
    _run_last()

  @pl.when(j == grid_width - 1)
  def end():
    if emit_residuals:
      # Emit unnormalized o and running (m, l); caller slices m/l row 0.
      o_out_ref, m_out_ref, l_out_ref = out_refs
      o_out_ref[...] = o_scratch_ref[...]
      m_out_ref[...] = m_scratch_ref[...]
      l_out_ref[...] = l_scratch_ref[...]
    else:
      # Normalize by l. The scratch tile is (NUM_SUBLANES, bq) and the
      # output is (head_dim_v, bq), so the inverse is stacked
      # ``head_dim_v_repeats`` times. ``pltpu.repeat`` did this until jax
      # 0.11; a concatenate of the same copies is what it lowered to.
      (o_ref,) = out_refs
      l = l_scratch_ref[...]
      l_inv = jnp.concatenate([1.0 / l] * head_dim_v_repeats, axis=0)
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mk: jax.Array,
    block_sizes: BlockSizes,
    *,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    dynamic_kv: bool = False,
    use_fixed_m: bool = False,
    emit_residuals: bool = False,
    qk_fp8: bool = False,
):
  """Runs the fused splash kernel over ``(heads, seq, dim)`` q/k/v.

  Args:
      q: ``(num_heads, seq_q, head_dim)`` — already scaled by
          ``scale * log2(e)`` and cast to bf16.
      k, v: ``(num_kv_heads, seq_kv, head_dim)`` bf16.
      mk: ``(2, num_heads)`` f32 fixed-m prefetch operand (row 0 =
          integral CS bound used as m, row 1 = eligibility flag); a
          ``(2, 1)`` zeros placeholder when ``use_fixed_m`` is off.
      block_sizes: kernel block sizes (clamped to seq by the caller).
      q_seq_len, kv_seq_len: unpadded lengths (default to the padded
          shapes).
      use_fixed_m: hybrid fixed-m softmax (see the kernel docstring).
      emit_residuals: when True return unnormalized ``(o, m, l)`` for an
          outer ring LSE-merge instead of the normalized output.
      qk_fp8: E4M3 Q/K into the QK dot.

  Returns:
      If ``emit_residuals``: ``(o_unnorm, m, l)`` where ``o_unnorm`` is
      ``(num_heads, head_dim_v, seq_q)`` f32 and ``m``, ``l`` are
      ``(num_heads, NUM_SUBLANES, seq_q)`` f32. Otherwise the normalized
      output ``(num_heads, head_dim_v, seq_q)`` in ``q.dtype``.
  """
  # Output keeps the caller's dtype: Q is re-cast to E4M3 below and
  # inheriting that would halve output precision.
  compute_dtype = q.dtype
  if qk_fp8:
    # E4M3 Q/K into the QK dot; casts sit at tile-load granularity and
    # halve both tiles' bytes. Saturating: post-qk-norm rows are O(1).
    q = fp8_cast.quantize_saturating(q)
    k = fp8_cast.quantize_saturating(k)
  num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  num_kv_heads = k.shape[0]
  padded_kv_seq_len = k.shape[1]
  # Use original (unpadded) lengths for padding handling inside the kernel.
  actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
  actual_kv_seq_len = (
      kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
  )
  # GQA broadcast: each q-head reads the kv-head it maps to (MHA has
  # q_heads_per_kv_head == 1, so h // 1 == h — the ring path is MHA).
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def kv_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)

  def out_index_map(h, i, j, *_):
    return (h, 0, i)

  in_specs = [
      pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((None, bkv, head_dim_qk), kv_index_map),
      pl.BlockSpec((None, bkv, head_dim_v), kv_index_map),
  ]
  operands = [mk, q, k, v]
  # First three outputs are cross-j scratch (a single block reused per
  # head); the rest are the real per-head outputs. Normalized o keeps
  # q.dtype; residual o is f32 (it is merged before any downcast).
  o_out_dtype = jnp.float32 if emit_residuals else compute_dtype
  out_shapes = [
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
  ]
  out_specs = [
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
      pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
  ]
  # Index of the first REAL output, after the scratch block above.
  first_out = len(out_shapes)
  out_shapes.append(
      jax.ShapeDtypeStruct(
          (num_q_heads, head_dim_v, actual_q_seq_len), o_out_dtype
      )
  )
  out_specs.append(pl.BlockSpec((None, head_dim_v, bq), out_index_map))
  if emit_residuals:
    # Extra per-head running max / sum, broadcast over the sublane axis.
    out_shapes += [
        jax.ShapeDtypeStruct(
            (num_q_heads, NUM_SUBLANES, actual_q_seq_len), jnp.float32
        ),
        jax.ShapeDtypeStruct(
            (num_q_heads, NUM_SUBLANES, actual_q_seq_len), jnp.float32
        ),
    ]
    out_specs += [
        pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
        pl.BlockSpec((None, NUM_SUBLANES, bq), out_index_map),
    ]
  # Use actual (unpadded) seq lens with ceiling division.
  grid_width = (actual_kv_seq_len + bkv - 1) // bkv
  grid_height = (actual_q_seq_len + bq - 1) // bq
  grid = (num_q_heads, grid_height, grid_width)

  all_out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel,
          mask_value=DEFAULT_MASK_VALUE,
          grid_width=grid_width,
          bkv=bkv,
          bkv_compute=bkv_compute,
          head_dim_v=head_dim_v,
          kv_seq_len=actual_kv_seq_len,
          use_fixed_m=use_fixed_m,
          emit_residuals=emit_residuals,
          dynamic_kv=dynamic_kv,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_splash_vmem_limit_bytes(),
      ),
      out_shape=out_shapes,
  )(*operands)
  if emit_residuals:
    return all_out[first_out], all_out[first_out + 1], all_out[first_out + 2]
  return all_out[first_out]


def _local_blocks(
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
) -> BlockSizes:
  """Clamps the requested blocks to one device-local call's extents.

  No physical pad: Mosaic masks partial edge blocks and the kernel slices
  the last sub-block itself (padding cost ~20 ms/step). Clamping keeps an
  oversized block from allocating a full tile; a block equal to the array
  dimension is legal regardless of 8/128 alignment.

  Args:
    q_seq_len: Unpadded query length.
    kv_seq_len: Unpadded key/value length.
    block_q: Requested query block.
    block_kv: Requested key/value block.
    block_kv_compute: Requested key/value compute block.

  Returns:
    The clamped block sizes.
  """
  bkv = min(block_kv, kv_seq_len)
  bkv_compute = min(block_kv_compute, kv_seq_len)
  if bkv % bkv_compute:
    # A short kv (e.g. a few hundred text tokens) can clamp bkv to a
    # length bkv_compute does not divide; fall back to one compute step.
    bkv_compute = bkv
  return BlockSizes(
      block_q=min(block_q, q_seq_len),
      block_kv=bkv,
      block_kv_compute=bkv_compute,
  )


def attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    use_fixed_m: bool = False,
    qk_fp8: bool = False,
    kv_len: jax.Array | None = None,
) -> jax.Array:
  """Device-local custom splash attention, normalized.

  ``kv_len`` (optional, a traced int scalar): keys at or past it are
  masked inside the kernel. Lets one compiled program serve every true
  length up to the padded shape -- the padded keys should be zero so the
  fixed-m bound is unaffected.

  Args:
    query: ``(B, heads, seq_q, dim)`` device-local.
    key: ``(B, heads, seq_kv, dim)`` device-local.
    value: ``(B, heads, seq_kv, dim)`` device-local.
    scale: Softmax scale; ``None`` means ``1/sqrt(dim)``.
    block_q: Query block rows (clamped to ``seq_q``).
    block_kv: Key/value block rows (clamped to ``seq_kv``).
    block_kv_compute: MXU compute sub-block over K/V (clamped).
    use_fixed_m: Hybrid fixed-m softmax. The per-head eligibility gate is
      ``fixed_m_bound.FIXED_M_SAFE_BOUND`` over this call's Q and K, whose
      no-flush guarantee presumes the caller k-smoothed ``key``.
    qk_fp8: E4M3 Q/K into the QK dot.

  Returns:
    ``(B, heads, seq_q, dim)`` in ``query``'s dtype.
  """
  scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
  # Fuse scale + log2(e) for exp2-based softmax
  query = query * scale_factor * fixed_m_bound.LOG2_E

  # Per-head fixed-m operands over the local shard. Computed
  # unconditionally; XLA drops it when unused.
  mk_arr = fixed_m_bound.fixed_m_operands(
      query, key, gate=fixed_m_bound.FIXED_M_SAFE_BOUND
  )
  if kv_len is not None:
    # Row 2 of the scalar-prefetch operand carries the runtime kv length
    # (float32 holds every length below 2^24 exactly).
    mk_arr = jnp.concatenate(
        [mk_arr, jnp.full((1, mk_arr.shape[1]), kv_len, mk_arr.dtype)], axis=0)

  def _kernel_3d(q_3d, k_3d, v_3d):
    # vmap strips the batch axis: q_3d is (heads, seq, dim), shape[1]=seq.
    q_orig_len = q_3d.shape[1]
    kv_orig_len = k_3d.shape[1]
    bsizes = _local_blocks(
        q_orig_len, kv_orig_len, block_q, block_kv, block_kv_compute
    )
    # Q cast back to bf16 for the 2x MXU path (K/V already bf16; kernel
    # accumulates f32). mk_arr is closed over, so it stays unmapped
    # under the vmap — matching the kernel's (2, heads) prefetch operand.
    out = _splash_attention_forward(
        q_3d.astype(jnp.bfloat16),
        k_3d,
        v_3d,
        mk_arr,
        bsizes,
        q_seq_len=q_orig_len,
        kv_seq_len=kv_orig_len,
        use_fixed_m=use_fixed_m,
        qk_fp8=qk_fp8,
        dynamic_kv=kv_len is not None,
    )
    # Output is (heads, dim, seq) — transpose back to (heads, seq, dim)
    out = jnp.swapaxes(out, 1, 2)
    return out[:, :q_orig_len, ...]

  return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(query, key, value)


# ---------------------------------------------------------------------------
# Ring residual path
# ---------------------------------------------------------------------------
#
# Ring attention drives ``_flash_attention_kernel`` with
# ``emit_residuals=True`` via ``residual_attention`` below, so an outer loop
# can merge unnormalized ``(o, m, l)`` across KV shards. Two ring rules: the
# local K shard is UN-smoothed (gate halved to ``FIXED_M_RING_SAFE_BOUND``),
# and a fixed head's ``m`` is an upper bound, so the merge must be
# m-invariant (LSE space).


def residual_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    use_fixed_m: bool = False,
    qk_fp8: bool = False,
    mk: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Device-local residual attention over one KV shard for ring merge.

  Args:
    query: ``(B, heads, seq_q, dim)`` device-local.
    key: ``(B, heads, seq_kv, dim)`` device-local (one ring shard).
    value: ``(B, heads, seq_kv, dim)`` device-local.
    scale: Softmax scale; ``None`` means ``1/sqrt(dim)``.
    block_q: Query block rows (clamped to ``seq_q``).
    block_kv: Key/value block rows (clamped to ``seq_kv``).
    block_kv_compute: MXU compute sub-block over K/V (clamped).
    use_fixed_m: Hybrid fixed-m softmax on the UN-smoothed local K shard,
      against the HALVED gate ``FIXED_M_RING_SAFE_BOUND``. A fixed head's
      returned ``m`` is the CS upper bound, not the rowmax, so callers
      must merge in LSE space (``lse = m + log2(l)`` is m-invariant).
    qk_fp8: E4M3 Q/K into the QK dot.
    mk: The ``(2, heads)`` prefetch operand, required with
      ``use_fixed_m``: a CALIBRATED operand that arrives as a graph input
      (``fixed_m_bound.static_fixed_m_operands``) or the per-step runtime
      bound (``fixed_m_bound.ring_fixed_m_operands``). Build it outside
      any vmap so it stays unmapped (a mapped operand slows the
      pallas_call ~1.6x at batch 2).

  Returns:
    ``(o_unnorm, m, l)`` where ``o_unnorm`` is ``(B, heads, seq_q, dim)``
    f32 (unnormalized, at the block's ``log2`` max scale) and ``m, l``
    are ``(B, heads, seq_q, 1)`` f32 running max / sum.

  Raises:
    ValueError: If ``use_fixed_m`` is set without ``mk``.
  """
  scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale

  if not use_fixed_m:
    mk_arr = jnp.zeros((2, 1), jnp.float32)  # unused placeholder
  elif mk is None:
    raise ValueError("use_fixed_m needs the (2, heads) operand mk.")
  else:
    mk_arr = mk

  def _kernel_3d(q_3d, k_3d, v_3d):
    q_orig_len = q_3d.shape[1]
    kv_orig_len = k_3d.shape[1]
    bsizes = _local_blocks(
        q_orig_len, kv_orig_len, block_q, block_kv, block_kv_compute
    )
    q_scaled = q_3d * scale_factor * fixed_m_bound.LOG2_E
    o_un, m, l = _splash_attention_forward(
        q_scaled.astype(jnp.bfloat16),
        k_3d,
        v_3d,
        mk_arr,
        bsizes,
        q_seq_len=q_orig_len,
        kv_seq_len=kv_orig_len,
        use_fixed_m=use_fixed_m,
        emit_residuals=True,
        qk_fp8=qk_fp8,
    )
    # o_un (h, dim, seq) -> (h, seq, dim); m, l (h, NUM_SUBLANES, seq) -> row 0.
    o_un = jnp.swapaxes(o_un, 1, 2)[:, :q_orig_len, :]
    m = m[:, 0, :q_orig_len, None]
    l = l[:, 0, :q_orig_len, None]
    return o_un, m, l

  return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(query, key, value)
