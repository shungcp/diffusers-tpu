"""Custom TPU Pallas/JAX ops and ``_c10d_functional`` hooks for ``torch_tpu`` MAGI-2-preview.

All ``torch.tpu.pallas.jax_op`` kernels are **strictly single-chip** (zero
collectives inside ``jax_op``) and attach explicit ``@op.register_fake`` shape
functions so Dynamo never triggers uncached ``jax.export`` during tracing.

Cross-chip collectives (Ulysses CP=8 ``all_to_all_single`` and MH-MoE EP=8
``all_gather_into_tensor`` / ``reduce_scatter_tensor``) execute through native
PyTorch ``torch.ops._c10d_functional`` under ``torch.tpu.distributed.spmd_safe``
and ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``.

Registered single-chip Pallas kernels:
1. ``pallas::magi2_grouped_linear_m3`` — ``jax.lax.ragged_dot`` for ``M=3``
   modality-grouped projections (1.0x FLOPs vs 3.0x masked matmul).
2. ``pallas::magi2_splash_sink_s{seq}`` — in-tree Pallas
   ``splash.residual_attention`` + exact base-2 1-token attention sink.
3. ``pallas::magi2_moe_head_{bf16,fp8}_cap{cap}`` — zero-cumsum
   ``argsort[:cap]`` + MegaCore Dual-TC ``gmm_v2`` (``fuse_act="swigluoai"``)
   for one head's local 32 routed experts (supports BF16 and FP8-W8A8).
4. ``pallas::magi2_mhc_{pre,post,fused_post_pre}`` — Tokamax Pallas 4-stream
   mHC kernels with exact MAGI-2 global-max Sinkhorn-Knopp epilogue (with
   automatic fallback to compiled PyTorch if tuple-return ``jax_op`` is
   unavailable or ``DT_MAGI2_TT_MHC=torch``).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # pragma: no cover
  jax = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]

from ..models.magi2.upstream_compat import Magi2TpuHooks


os.environ.setdefault("DT_FORCE_TPU_TRACE", "1")

_REG: dict[tuple[Any, ...], Any] = {}


# ---------------------------------------------------------------------------
# 1. Modality GroupedLinear (M=3) via jax.lax.ragged_dot
# ---------------------------------------------------------------------------


def get_grouped_linear_op():
  """Register and return ``pallas::magi2_grouped_linear_m3`` backed by Pallas ``gmm_v2``.

  Uses in-tree ``diffusers_tpu.kernels.gmm_v2`` (lowers to ``tpu_custom_call``)
  because ``torch_tpu``'s ``cHLO -> StableHLO`` pass marks ``mhlo.ragged_dot``
  illegal inside ``jax_op``.
  """
  key = ("grouped_linear_m3",)
  if key in _REG:
    return _REG[key]

  from ..kernels import gmm_v2 as _gmm_v2

  def fn(x: jax.Array, w: jax.Array, group_sizes: jax.Array) -> jax.Array:
    orig_m = x.shape[0]
    pad_m = (-orig_m) % 16
    x_in = jnp.pad(x, ((0, pad_m), (0, 0))) if pad_m else x
    y = _gmm_v2.gmm_v2(
        x_in.astype(jnp.bfloat16),
        w.astype(jnp.bfloat16),
        group_sizes,
        preferred_element_type=jnp.dtype(jnp.bfloat16),
        acc_dtype=jnp.dtype(jnp.float32),
        zero_initialize=True,
    )
    return (y[:orig_m] if pad_m else y).astype(x.dtype)

  op = torch.tpu.pallas.jax_op("pallas::magi2_grouped_linear_m3", fn)

  @op.register_fake
  def _fake(x, w, group_sizes):
    del group_sizes
    return torch.empty(
        (x.shape[0], w.shape[2]),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.contiguous_format,
    )

  _REG[key] = op
  return op


# ---------------------------------------------------------------------------
# 2. In-Tree Pallas Splash Attention + Exact Base-2 Sink
# ---------------------------------------------------------------------------


def get_splash_sink_op(seq_len: int):
  """Register and return ``pallas::magi2_splash_sink_s{seq_len}``."""
  key = ("splash_sink", int(seq_len))
  if key in _REG:
    return _REG[key]

  from ..kernels import block_sizes as _block_sizes
  from ..kernels import splash as _splash
  from ..models.magi2.jax_preview_dit import apply_residual_attention_sink

  try:
    blocks = _block_sizes.resolve_block_sizes(
        int(seq_len), int(seq_len), 128, {}, "residual", _block_sizes.V7_SPEC
    )
    bq = int(blocks["block_q"])
    bkv = int(blocks["block_kv"])
    bkv_c = int(blocks["block_kv_compute"])
  except Exception:
    bq = min(1024, int(seq_len))
    bkv = min(1024, int(seq_len))
    bkv_c = min(512, bkv)

  def fn(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      sinks_loc: jax.Array,
  ) -> jax.Array:
    # q, k, v: (S, H_loc, D) -> (1, H_loc, S, D)
    out_dtype = q.dtype
    scale = q.shape[-1] ** -0.5
    q_b = jnp.swapaxes(q, 0, 1)[None, :, :, :]
    k_b = jnp.swapaxes(k, 0, 1)[None, :, :, :]
    v_b = jnp.swapaxes(v, 0, 1)[None, :, :, :]
    o_unnorm, m_b2, l_b2 = _splash.residual_attention(
        q_b,
        k_b,
        v_b,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv_c,
        use_fixed_m=False,
    )
    o_b = apply_residual_attention_sink(o_unnorm, m_b2, l_b2, sinks_loc).astype(out_dtype)
    return jnp.swapaxes(o_b[0], 0, 1)

  op = torch.tpu.pallas.jax_op(f"pallas::magi2_splash_sink_s{int(seq_len)}_q{bq}_kv{bkv}", fn)

  @op.register_fake
  def _fake(q, k, v, sinks_loc):
    del k, v, sinks_loc
    return torch.empty_like(q, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


def get_sparse_splash_sink_op(
    seq_len: int,
    n_vid: int,
    rows_per_frame: int,
    window: int,
    *,
    block_q: int = 1024,
    block_kv: int = 1024,
    anchor_first_frame: bool = True,
):
  """Register and return ``pallas::magi2_sparse_splash_sink_...`` with exact Attention Sink."""
  bq = min(int(block_q), int(seq_len))
  bkv = min(int(block_kv), int(seq_len))
  bkv_c = min(512, bkv)
  key = (
      "sparse_splash_sink",
      int(seq_len),
      int(n_vid),
      int(rows_per_frame),
      int(window),
      bq,
      bkv,
      bool(anchor_first_frame),
  )
  if key in _REG:
    return _REG[key]

  from ..kernels import sparse_splash as _sparse_splash
  from ..models.magi2.jax_preview_dit import (
      apply_residual_attention_sink,
      build_magi2_sparse_kv_table,
  )

  kv_table = build_magi2_sparse_kv_table(
      int(seq_len),
      int(n_vid),
      int(rows_per_frame),
      int(window),
      block_q=bq,
      block_kv=bkv,
      anchor_first_frame=anchor_first_frame,
  )

  def fn(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      sinks_loc: jax.Array,
  ) -> jax.Array:
    out_dtype = q.dtype
    scale = q.shape[-1] ** -0.5
    q_b = jnp.swapaxes(q, 0, 1)[None, :, :, :]
    k_b = jnp.swapaxes(k, 0, 1)[None, :, :, :]
    v_b = jnp.swapaxes(v, 0, 1)[None, :, :, :]
    o_unnorm, m_b2, l_b2 = _sparse_splash.residual_attention(
        q_b,
        k_b,
        v_b,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv_c,
        use_fixed_m=False,
        kv_table=kv_table,
    )
    o_b = apply_residual_attention_sink(o_unnorm, m_b2, l_b2, sinks_loc).astype(out_dtype)
    return jnp.swapaxes(o_b[0], 0, 1)

  aff = 1 if anchor_first_frame else 0
  op_name = (
      f"pallas::magi2_sparse_splash_sink_s{int(seq_len)}_v{int(n_vid)}"
      f"_r{int(rows_per_frame)}_w{int(window)}_a{aff}_q{bq}_kv{bkv}"
  )
  op = torch.tpu.pallas.jax_op(op_name, fn)

  @op.register_fake
  def _fake(q, k, v, sinks_loc):
    del k, v, sinks_loc
    return torch.empty_like(q, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


# ---------------------------------------------------------------------------
# 3. Single-Head Local-32-Expert Zero-Cumsum Sliced-AG + Dual-TC gmm_v2
# ---------------------------------------------------------------------------


def compute_moe_head_cap(s_all: int, top_k: int, ep_size: int, cap_factor: float) -> int:
  total_routes = int(s_all) * int(top_k)
  if cap_factor <= 0.0:
    return total_routes
  raw = math.ceil((total_routes / max(1, int(ep_size))) * float(cap_factor))
  # Round up to 128-row alignment (MXU tile friendly) and cap at total_routes.
  return min(total_routes, max(128, ((raw + 127) // 128) * 128))


def get_moe_local_head_op(
    s_all: int,
    *,
    e_local: int = 32,
    top_k: int = 6,
    ep_size: int = 8,
    cap_factor: float = 2.0,
    fp8: bool = False,
):
  """Register single-head local-expert Dual-TC ``gmm_v2`` kernel (BF16 or FP8)."""
  cap = compute_moe_head_cap(s_all, top_k, ep_size, cap_factor)
  key = ("moe_head", int(s_all), int(e_local), int(top_k), int(cap), bool(fp8))
  if key in _REG:
    return _REG[key]

  from ..models.magi2.jax_mh_moe import _grouped_swiglu_ffn

  if not fp8:
    def fn_bf16(
        x_all: jax.Array,       # (S_all, d_h) bf16
        idx_all: jax.Array,     # (S_all, K) i32
        probs_all: jax.Array,   # (S_all, K) f32
        w_gu_h: jax.Array,      # (E_loc, d_h, 2*d_e) bf16
        w_dn_h: jax.Array,      # (E_loc, d_e, d_h) bf16
        e_lo_arr: jax.Array,    # (1,) i32
    ) -> jax.Array:
      e_lo = e_lo_arr[0]
      flat_idx = idx_all.reshape(-1)
      flat_probs = probs_all.reshape(-1).astype(jnp.float32)
      local_e = flat_idx - e_lo
      is_local = (local_e >= 0) & (local_e < e_local)
      safe_e = jnp.where(is_local, local_e, jnp.int32(e_local))

      order = jnp.argsort(safe_e, stable=False)[:cap]
      # CRITICAL: JAX jnp.bincount clamps negative indices to bin 0!
      # Always count safe_e in [0..e_local] with length=e_local+1 and slice [:e_local].
      gs_raw = jnp.bincount(safe_e, length=e_local + 1)[:e_local].astype(jnp.int32)
      # Defensive 32-element clamp ensuring sum(group_sizes) <= cap.
      gs_end = jnp.minimum(jnp.cumsum(gs_raw), jnp.int32(cap))
      gs_start = jnp.concatenate([jnp.zeros((1,), jnp.int32), gs_end[:-1]])
      gs = (gs_end - gs_start).astype(jnp.int32)
      total_local = gs_end[-1]

      tok_ids = order // jnp.int32(top_k)
      x_sorted = x_all[tok_ids]
      out_sorted = _grouped_swiglu_ffn(
          x_sorted,
          w_gu_h,
          w_dn_h,
          gs,
          zero_initialize=False,
          out_dtype=x_all.dtype,
      )
      # IEEE select MUST wrap the product so uninitialized tail rows from
      # gmm_v2(zero_initialize=False) never compute NaN * 0.0 = NaN.
      valid = (jnp.arange(cap, dtype=jnp.int32) < total_local)[:, None]
      ys_w = jnp.where(
          valid,
          out_sorted.astype(jnp.float32) * flat_probs[order, None],
          jnp.float32(0.0),
      )
      y_f32 = jnp.zeros(x_all.shape, dtype=jnp.float32).at[tok_ids].add(ys_w)
      return y_f32.astype(x_all.dtype)

    name = f"pallas::magi2_moe_head_bf16_v2_s{int(s_all)}_c{int(cap)}"
    op = torch.tpu.pallas.jax_op(name, fn_bf16)

    @op.register_fake
    def _fake_bf16(x_all, idx_all, probs_all, w_gu_h, w_dn_h, e_lo_arr):
      del idx_all, probs_all, w_gu_h, w_dn_h, e_lo_arr
      return torch.empty_like(x_all, memory_format=torch.contiguous_format)

    _REG[key] = op
    return op

  def fn_fp8(
      x_all: jax.Array,         # (S_all, d_h) bf16
      idx_all: jax.Array,       # (S_all, K) i32
      probs_all: jax.Array,     # (S_all, K) f32
      w_gu_h: jax.Array,        # (E_loc, d_h, 2*d_e) f8e4m3fn
      w_dn_h: jax.Array,        # (E_loc, d_e, d_h) f8e4m3fn
      w_gu_s_h: jax.Array,      # (E_loc, 1, 1, 2*d_e) f32
      w_dn_s_h: jax.Array,      # (E_loc, 1, 1, d_h) f32
      e_lo_arr: jax.Array,      # (1,) i32
  ) -> jax.Array:
    e_lo = e_lo_arr[0]
    flat_idx = idx_all.reshape(-1)
    flat_probs = probs_all.reshape(-1).astype(jnp.float32)
    local_e = flat_idx - e_lo
    is_local = (local_e >= 0) & (local_e < e_local)
    safe_e = jnp.where(is_local, local_e, jnp.int32(e_local))

    order = jnp.argsort(safe_e, stable=False)[:cap]
    gs_raw = jnp.bincount(safe_e, length=e_local + 1)[:e_local].astype(jnp.int32)
    gs_end = jnp.minimum(jnp.cumsum(gs_raw), jnp.int32(cap))
    gs_start = jnp.concatenate([jnp.zeros((1,), jnp.int32), gs_end[:-1]])
    gs = (gs_end - gs_start).astype(jnp.int32)
    total_local = gs_end[-1]

    tok_ids = order // jnp.int32(top_k)
    x_sorted = x_all[tok_ids]
    out_sorted = _grouped_swiglu_ffn(
        x_sorted,
        w_gu_h,
        w_dn_h,
        gs,
        w_gu_scale=w_gu_s_h,
        w_down_scale=w_dn_s_h,
        zero_initialize=False,
        out_dtype=x_all.dtype,
    )
    valid = (jnp.arange(cap, dtype=jnp.int32) < total_local)[:, None]
    ys_w = jnp.where(
        valid,
        out_sorted.astype(jnp.float32) * flat_probs[order, None],
        jnp.float32(0.0),
    )
    y_f32 = jnp.zeros(x_all.shape, dtype=jnp.float32).at[tok_ids].add(ys_w)
    return y_f32.astype(x_all.dtype)

  name = f"pallas::magi2_moe_head_fp8_v2_s{int(s_all)}_c{int(cap)}"
  op = torch.tpu.pallas.jax_op(name, fn_fp8)

  @op.register_fake
  def _fake_fp8(x_all, idx_all, probs_all, w_gu_h, w_dn_h, w_gu_s_h, w_dn_s_h, e_lo_arr):
    del idx_all, probs_all, w_gu_h, w_dn_h, w_gu_s_h, w_dn_s_h, e_lo_arr
    return torch.empty_like(x_all, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


# ---------------------------------------------------------------------------
# 4. Tokamax Pallas 4-Stream mHC (Exact MAGI-2 Global-Max Sinkhorn)
# ---------------------------------------------------------------------------


def get_mhc_ops():
  """Attempt to register Tokamax Pallas mHC ``jax_op`` triple ``(pre, post, fpp)``.

  Returns ``(pre_op, post_op, fpp_op)`` or ``(None, None, None)`` if disabled
  via ``DT_MAGI2_TT_MHC=torch`` or if Tokamax / tuple-return ``jax_op`` is not
  available in the active ``torch_tpu`` build.
  """
  if os.environ.get("DT_MAGI2_TT_MHC", "tokamax").strip().lower() == "torch":
    return None, None, None
  key = ("mhc_tokamax",)
  if key in _REG:
    return _REG[key]

  try:
    from ..models.magi2.rope_and_ops import (
        mhc_fused_post_pre_tokamax,
        mhc_post_tokamax,
        mhc_pre_tokamax,
        use_tokamax_mhc,
    )

    if not use_tokamax_mhc():
      _REG[key] = (None, None, None)
      return _REG[key]

    def fn_pre(
        s_flat: jax.Array,
        norm_w: jax.Array,
        phi: jax.Array,
        a_pre: jax.Array,
        a_post: jax.Array,
        a_res: jax.Array,
        b_pre: jax.Array,
        b_post: jax.Array,
        b_res: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      return mhc_pre_tokamax(
          s_flat, norm_w, phi, a_pre, b_pre, a_post, b_post, a_res, b_res
      )

    def fn_post(
        sub_out: jax.Array,
        s_flat: jax.Array,
        post_mix: jax.Array,
        comb_mix: jax.Array,
    ) -> jax.Array:
      return mhc_post_tokamax(sub_out, s_flat, post_mix, comb_mix)

    def fn_fpp(
        sub_out: jax.Array,
        s_flat: jax.Array,
        post_mix: jax.Array,
        comb_mix: jax.Array,
        norm_w: jax.Array,
        phi: jax.Array,
        a_pre: jax.Array,
        a_post: jax.Array,
        a_res: jax.Array,
        b_pre: jax.Array,
        b_post: jax.Array,
        b_res: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
      return mhc_fused_post_pre_tokamax(
          sub_out, s_flat, post_mix, comb_mix,
          norm_w, phi, a_pre, b_pre, a_post, b_post, a_res, b_res,
      )

    op_pre = torch.tpu.pallas.jax_op("pallas::magi2_mhc_pre", fn_pre)
    op_post = torch.tpu.pallas.jax_op("pallas::magi2_mhc_post", fn_post)
    op_fpp = torch.tpu.pallas.jax_op("pallas::magi2_mhc_fpp", fn_fpp)

    @op_pre.register_fake
    def _fake_pre(s_flat, norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res):
      del norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res
      S, ad = s_flat.shape
      return (
          torch.empty((S, ad // 4), dtype=s_flat.dtype, device=s_flat.device),
          torch.empty((S, 4), dtype=torch.float32, device=s_flat.device),
          torch.empty((S, 4, 4), dtype=torch.float32, device=s_flat.device),
      )

    @op_post.register_fake
    def _fake_post(sub_out, s_flat, post_mix, comb_mix):
      del sub_out, post_mix, comb_mix
      return torch.empty_like(s_flat, memory_format=torch.contiguous_format)

    @op_fpp.register_fake
    def _fake_fpp(
        sub_out, s_flat, post_mix, comb_mix,
        norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res,
    ):
      del sub_out, post_mix, comb_mix, norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res
      S, ad = s_flat.shape
      return (
          torch.empty_like(s_flat, memory_format=torch.contiguous_format),
          torch.empty((S, ad // 4), dtype=s_flat.dtype, device=s_flat.device),
          torch.empty((S, 4), dtype=torch.float32, device=s_flat.device),
          torch.empty((S, 4, 4), dtype=torch.float32, device=s_flat.device),
      )

    _REG[key] = (op_pre, op_post, op_fpp)
  except Exception as exc:  # pragma: no cover
    print(f"[magi2_kernels_tt] Tokamax mHC jax_op unavailable ({exc}); using compiled PyTorch mHC.", flush=True)
    _REG[key] = (None, None, None)

  return _REG[key]


# ---------------------------------------------------------------------------
# 4b. Exact fp32 Precision.HIGHEST Matmul & MoE Router (No BF16 Truncation!)
# ---------------------------------------------------------------------------


def get_fp32_matmul_highest_op():
  """Register ``pallas::magi2_fp32_mm_highest`` for exact fp32 mHC/Adapter matmuls."""
  key = ("fp32_mm_highest",)
  if key in _REG:
    return _REG[key]

  from jax import lax

  def fn(a: jax.Array, b: jax.Array) -> jax.Array:
    return lax.dot_general(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        (((a.ndim - 1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST,
    )

  op = torch.tpu.pallas.jax_op("pallas::magi2_fp32_mm_highest", fn)

  @op.register_fake
  def _fake(a, b):
    return torch.empty(
        (*a.shape[:-1], b.shape[1]),
        dtype=torch.float32,
        device=a.device,
        memory_format=torch.contiguous_format,
    )

  _REG[key] = op
  return op


def get_route_mh_moe_op(top_k: int = 6, route_scale: float = 4.9):
  """Register ``pallas::magi2_route_mh_moe`` matching ``jax_mh_moe.route_mh_moe`` (HIGHEST)."""
  key = ("route_mh_moe", int(top_k), float(route_scale))
  if key in _REG:
    return _REG[key]

  from ..models.magi2.jax_mh_moe import route_mh_moe

  k_val = int(top_k)
  rs_val = float(route_scale)

  def fn(
      x_heads: jax.Array,
      gate: jax.Array,
      expert_bias: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    probs, idx = route_mh_moe(
        x_heads,
        gate,
        expert_bias,
        top_k=k_val,
        route_norm=True,
        route_scale=rs_val,
    )
    return idx.astype(jnp.int32), probs.astype(jnp.float32)

  try:
    op = torch.tpu.pallas.jax_op(f"pallas::magi2_route_mh_moe_k{k_val}", fn)

    @op.register_fake
    def _fake(x_heads, gate, expert_bias):
      del gate, expert_bias
      S, H, _ = x_heads.shape
      return (
          torch.empty((H, S, k_val), dtype=torch.int32, device=x_heads.device),
          torch.empty((H, S, k_val), dtype=torch.float32, device=x_heads.device),
      )

    _REG[key] = op
  except Exception as exc:  # pragma: no cover
    print(f"[magi2_kernels_tt] route_mh_moe jax_op unavailable ({exc}); using torch router.", flush=True)
    _REG[key] = None
  return _REG[key]


# ---------------------------------------------------------------------------
# 5. Build Complete Magi2TpuHooks for 8-Rank torch_tpu Execution
# ---------------------------------------------------------------------------


def build_torchtpu_hooks(
    *,
    rank: int,
    world: int,
    group_name: str,
    s_total: int,
    sample_lens: tuple[int, ...],
    device: torch.device,
    num_heads_q: int = 24,
    e_local: int = 32,
    top_k: int = 6,
    route_scale: float = 4.9,
    cap_factor: float | None = None,
    fp8_moe: bool = False,
    use_splash: bool = True,
    sparse_spec: tuple[int, int, int] | None = None,
) -> Magi2TpuHooks:
  """Pre-register all ``jax_op`` kernels and build :class:`Magi2TpuHooks` for ``torch_tpu``."""
  if cap_factor is None:
    cap_factor = float(os.environ.get("DT_MAGI2_AG_CAP_FACTOR", "2.0"))

  # Pre-register ALL jax_ops BEFORE Dynamo tracing!
  use_native_ragged = (
      os.environ.get("DT_MAGI2_TT_GROUPED_IMPL", "native").strip().lower() == "native"
      and hasattr(torch, "tpu")
      and hasattr(torch.tpu, "ops")
      and hasattr(torch.tpu.ops, "ragged_dot")
  )
  grouped_op = None if use_native_ragged else get_grouped_linear_op()
  if use_splash:
    if sparse_spec is not None:
      sp_n_vid, sp_rpf, sp_win = sparse_spec
      splash_ops = {
          int(sl): get_sparse_splash_sink_op(int(sl), sp_n_vid, sp_rpf, sp_win)
          for sl in set(sample_lens)
      }
    else:
      splash_ops = {int(sl): get_splash_sink_op(int(sl)) for sl in set(sample_lens)}
  else:
    splash_ops = {}
  moe_op = get_moe_local_head_op(
      s_total,
      e_local=e_local,
      top_k=top_k,
      ep_size=world,
      cap_factor=cap_factor,
      fp8=fp8_moe,
  )
  mhc_pre_op, mhc_post_op, mhc_fpp_op = get_mhc_ops()
  fp32_mm_op = get_fp32_matmul_highest_op()
  route_op = get_route_mh_moe_op(top_k=top_k, route_scale=route_scale)

  if rank == 0:
    sp_tag = f"sparse(w={sparse_spec[2]})" if sparse_spec is not None else "dense"
    print(
        f"[magi2_kernels_tt] hooks: grouped={'native_ragged' if use_native_ragged else 'gmm_v2'} "
        f"splash={use_splash}({sp_tag}) moe={'fp8' if fp8_moe else 'bf16'} "
        f"mhc_tokamax={mhc_pre_op is not None} route_highest={route_op is not None} "
        f"fp32_mm_highest={fp32_mm_op is not None}",
        flush=True,
    )

  # Device tensors holding per-rank indices so all 8 ranks emit 100% identical SPMD HLO!
  e_lo_t = torch.tensor([rank * e_local], dtype=torch.int32, device=device)
  h_q_loc = num_heads_q // max(1, world)
  sink_idx_t = torch.arange(rank * h_q_loc, (rank + 1) * h_q_loc, dtype=torch.long, device=device)

  def _grouped_linear(x, w, group_sizes, mod_ids):
    if use_native_ragged:
      return torch.tpu.ops.ragged_dot(
          x.contiguous(), w.contiguous(), group_sizes.to(torch.int32).contiguous()
      )
    if w.shape[1] % 128 == 0 and w.shape[2] % 128 == 0:
      return grouped_op(x.contiguous(), w.contiguous(), group_sizes.to(torch.int32).contiguous())
    out = torch.zeros((x.shape[0], w.shape[2]), device=x.device, dtype=x.dtype)
    for m in range(w.shape[0]):
      mask = (mod_ids == m).unsqueeze(-1)
      out = torch.where(mask, torch.matmul(x, w[m]), out)
    return out

  def _ulysses_s2h(x):
    # (S_loc, H, D) -> (S_total, H_loc, D)
    s_loc, h_tot, d = x.shape
    h_loc = h_tot // world
    send = x.reshape(s_loc, world, h_loc, d).permute(1, 0, 2, 3).contiguous()
    splits = [1] * world
    recv = torch.ops._c10d_functional.all_to_all_single(send, splits, splits, group_name)
    recv = torch.ops._c10d_functional.wait_tensor(recv)
    return recv.reshape(world * s_loc, h_loc, d)

  def _ulysses_h2s(x):
    # (S_total, H_loc, D) -> (S_loc, H, D)
    s_tot, h_loc, d = x.shape
    s_loc = s_tot // world
    send = x.reshape(world, s_loc, h_loc, d).contiguous()
    splits = [1] * world
    recv = torch.ops._c10d_functional.all_to_all_single(send, splits, splits, group_name)
    recv = torch.ops._c10d_functional.wait_tensor(recv)
    return recv.permute(1, 0, 2, 3).contiguous().reshape(s_loc, world * h_loc, d)

  def _ep_all_gather(x):
    out = torch.ops._c10d_functional.all_gather_into_tensor(x.contiguous(), world, group_name)
    return torch.ops._c10d_functional.wait_tensor(out)

  def _ep_reduce_scatter(x):
    out = torch.ops._c10d_functional.reduce_scatter_tensor(x.contiguous(), "sum", world, group_name)
    return torch.ops._c10d_functional.wait_tensor(out)

  def _attn_with_sink(q, k, v, sinks, slens, scale):
    s_tot, h_loc, d = q.shape
    sinks_loc = sinks.index_select(1, sink_idx_t).contiguous() if world > 1 else sinks.contiguous()
    if not use_splash:
      from ..models.magi2.torch_preview_dit import sdpa_with_sink_ref_torch
      return sdpa_with_sink_ref_torch(q, k, v, sinks_loc, slens, scale)
    outs = []
    offset = 0
    for sl in slens:
      op = splash_ops[int(sl)]
      outs.append(
          op(
              q[offset : offset + sl].contiguous(),
              k[offset : offset + sl].contiguous(),
              v[offset : offset + sl].contiguous(),
              sinks_loc,
          )
      )
      offset += sl
    if offset < s_tot:
      outs.append(torch.zeros((s_tot - offset, h_loc, d), device=q.device, dtype=q.dtype))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

  def _moe_local_head(x_all, idx_all, probs_all, w_gu_h, w_dn_h, w_gu_s_h, w_dn_s_h):
    if fp8_moe:
      return moe_op(
          x_all.contiguous(),
          idx_all.contiguous(),
          probs_all.contiguous(),
          w_gu_h.contiguous(),
          w_dn_h.contiguous(),
          w_gu_s_h.contiguous(),
          w_dn_s_h.contiguous(),
          e_lo_t,
      )
    return moe_op(
        x_all.contiguous(),
        idx_all.contiguous(),
        probs_all.contiguous(),
        w_gu_h.contiguous(),
        w_dn_h.contiguous(),
        e_lo_t,
    )

  def _fp32_mm_highest(a, b):
    return fp32_mm_op(a.float().contiguous(), b.float().contiguous())

  def _route_mh_moe(x_heads, gate, expert_bias):
    return route_op(x_heads.contiguous(), gate.float().contiguous(), expert_bias.float().contiguous())

  def _mhc_pre(s_flat, norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res):
    return mhc_pre_op(
        s_flat.contiguous(), norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res
    )

  def _mhc_post(s_flat, sub_out, pm, cm):
    return mhc_post_op(sub_out.contiguous(), s_flat.contiguous(), pm.contiguous(), cm.contiguous())

  def _mhc_fpp(
      s_flat, sub_out, pm, cm,
      norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res,
  ):
    return mhc_fpp_op(
        sub_out.contiguous(), s_flat.contiguous(), pm.contiguous(), cm.contiguous(),
        norm_w, phi, a_pre, a_post, a_res, b_pre, b_post, b_res,
    )

  return Magi2TpuHooks(
      attn_with_sink=_attn_with_sink,
      grouped_linear=_grouped_linear,
      moe_local_head=_moe_local_head,
      route_mh_moe=_route_mh_moe if route_op is not None else None,
      fp32_matmul_highest=_fp32_mm_highest if fp32_mm_op is not None else None,
      mhc_pre=_mhc_pre if mhc_pre_op is not None else None,
      mhc_post=_mhc_post if mhc_post_op is not None else None,
      mhc_fused_post_pre=_mhc_fpp if mhc_fpp_op is not None else None,
      ulysses_scatter_head_gather_seq=_ulysses_s2h if world > 1 else None,
      ulysses_scatter_seq_gather_head=_ulysses_h2s if world > 1 else None,
      ep_all_gather=_ep_all_gather if world > 1 else None,
      ep_reduce_scatter=_ep_reduce_scatter if world > 1 else None,
      ep_rank=rank,
      ep_size=world,
  )


# ---------------------------------------------------------------------------
# 8. Stage 2 1080p Refiner Single-Chip Pallas Block-Grid GQA Attention & Hooks
# ---------------------------------------------------------------------------


def get_refiner_block_grid_attn_op(layout: Any, *, qk_fp8: bool = False):
  """Register and return ``pallas::magi2_refiner_gqa_wcoal_...`` wrapping the exact JAX kernel."""
  nt, nh, nw = layout.block_grid_thw
  rt, rh, rw = layout.radius
  key = (
      "refiner_gqa_wcoal",
      (nt, nh, nw),
      (rt, rh, rw),
      int(layout.block_vol),
      int(layout.n_vid_pad),
      int(layout.n_tail_pad),
      bool(qk_fp8),
  )
  if key in _REG:
    return _REG[key]

  from ..models.magi2.jax_refiner_dit import _gqa_wcoal_block_grid_attention

  def fn(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      kv_blen: jax.Array,
  ) -> jax.Array:
    scale = q.shape[-1] ** -0.5
    return _gqa_wcoal_block_grid_attention(
        q,
        k,
        v,
        kv_blen,
        layout=layout,
        scale=scale,
        qk_fp8=qk_fp8,
    )

  fp8_tag = "fp8" if qk_fp8 else "bf16"
  op_name = (
      f"pallas::magi2_refiner_gqa_wcoal_{fp8_tag}_g{nt}x{nh}x{nw}"
      f"_r{rt}x{rh}x{rw}_v{int(layout.n_vid_pad)}_t{int(layout.n_tail_pad)}"
  )
  op = torch.tpu.pallas.jax_op(op_name, fn)

  @op.register_fake
  def _fake(q, k, v, kv_blen):
    del k, v, kv_blen
    return torch.empty_like(q, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


def build_refiner_torchtpu_hooks(
    *,
    rank: int,
    cp_size: int,
    cp_group_name: str,
    layout: Any,
    qk_fp8: bool = False,
):
  """Pre-register Stage 2 Refiner Pallas ops and build :class:`Magi2RefinerTpuHooks`."""
  from ..models.magi2.torch_refiner_dit import Magi2RefinerTpuHooks

  use_native_ragged = (
      os.environ.get("DT_MAGI2_TT_GROUPED_IMPL", "native").strip().lower() == "native"
      and hasattr(torch, "tpu")
      and hasattr(torch.tpu, "ops")
      and hasattr(torch.tpu.ops, "ragged_dot")
  )
  grouped_op = None if use_native_ragged else get_grouped_linear_op()
  attn_op = get_refiner_block_grid_attn_op(layout, qk_fp8=qk_fp8)
  fp32_mm_op = get_fp32_matmul_highest_op()

  if rank == 0:
    print(
        f"[magi2_kernels_tt] refiner hooks: cp={cp_size} "
        f"grouped={'native_ragged' if use_native_ragged else 'gmm_v2'} "
        f"attn=gqa_wcoal({'fp8' if qk_fp8 else 'bf16'}) radius={layout.radius}",
        flush=True,
    )

  def _grouped_linear(x, w, group_sizes, mod_ids):
    if use_native_ragged:
      return torch.tpu.ops.ragged_dot(
          x.contiguous(), w.contiguous(), group_sizes.to(torch.int32).contiguous()
      )
    if w.shape[1] % 128 == 0 and w.shape[2] % 128 == 0:
      return grouped_op(x.contiguous(), w.contiguous(), group_sizes.to(torch.int32).contiguous())
    out = torch.zeros((x.shape[0], w.shape[2]), device=x.device, dtype=x.dtype)
    for m in range(w.shape[0]):
      mask = (mod_ids == m).unsqueeze(-1)
      out = torch.where(mask, torch.matmul(x, w[m]), out)
    return out

  def _ulysses_s2h(x):
    # (S_loc, H, D) -> (S_total, H_loc, D) across cp_group_name (CP=4 -> 32Q->8Q, 8KV->2KV)
    s_loc, h_tot, d = x.shape
    h_loc = h_tot // cp_size
    send = x.reshape(s_loc, cp_size, h_loc, d).permute(1, 0, 2, 3).contiguous()
    splits = [1] * cp_size
    recv = torch.ops._c10d_functional.all_to_all_single(send, splits, splits, cp_group_name)
    recv = torch.ops._c10d_functional.wait_tensor(recv)
    return recv.reshape(cp_size * s_loc, h_loc, d)

  def _ulysses_h2s(x):
    # (S_total, H_loc, D) -> (S_loc, H, D)
    s_tot, h_loc, d = x.shape
    s_loc = s_tot // cp_size
    send = x.reshape(cp_size, s_loc, h_loc, d).contiguous()
    splits = [1] * cp_size
    recv = torch.ops._c10d_functional.all_to_all_single(send, splits, splits, cp_group_name)
    recv = torch.ops._c10d_functional.wait_tensor(recv)
    return recv.permute(1, 0, 2, 3).contiguous().reshape(s_loc, cp_size * h_loc, d)

  def _block_grid_attn(q, k, v, kv_blen):
    return attn_op(q.contiguous(), k.contiguous(), v.contiguous(), kv_blen.to(torch.int32).contiguous())

  def _fp32_mm_highest(a, b):
    return fp32_mm_op(a.float().contiguous(), b.float().contiguous())

  return Magi2RefinerTpuHooks(
      grouped_linear=_grouped_linear,
      ulysses_s2h=_ulysses_s2h if cp_size > 1 else None,
      ulysses_h2s=_ulysses_h2s if cp_size > 1 else None,
      block_grid_attn=_block_grid_attn,
      fp32_matmul_highest=_fp32_mm_highest if fp32_mm_op is not None else None,
      cp_size=cp_size,
      qk_fp8=qk_fp8,
  )
