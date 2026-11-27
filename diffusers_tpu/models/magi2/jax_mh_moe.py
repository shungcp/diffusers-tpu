"""Multi-Head Latent MoE for Sand.ai MAGI-2 Preview (113.93B) on TPU v7x-8.

Implements MAGI-2's ``CoreMultiHeadMoE`` (used on layers 2..37 of the 40-layer
Stage 1 Preview DiT) in JAX / Flax NNX, matching the upstream PyTorch/Triton
reference (``inference/model/magi2_preview.py``, ``inference/flash_mh_moe/``):

* Input ``x: (S, H * d_head)`` (3072 = 12 heads x 256) is split into ``H = 12``
  independent latent heads, each with its own disjoint bank of ``E = 256``
  experts (``12 * 256 = 3072`` experts/layer, ``d_expert = 1280``, 108.72B
  routed params across 36 layers = 202.50 GiB BF16).
* Routing (fp32):
  ``scores = sigmoid(einsum("shd,hed->hse", x_heads, gate))``;
  ``idx = topk(scores + expert_bias, k=6)`` (using ``expert_bias_ema`` loaded
  from checkpoint); unbiased ``probs = gather(scores, idx)``, L1-normalized
  with ``eps=1e-12``, scaled by ``route_scale=4.9``.
* Expert FFN: ``SwiGLU7(x @ W_gate, x @ W_up) @ W_down`` where ``SwiGLU7`` is
  ``clip(g, max=7) * sigmoid(1.702 * g) * (clip(u, -7, 7) + 1)`` — identical to
  ``gmm_v2(..., fuse_act="swigluoai")``.

TPU v7x-8 Sharding Strategy (100% balanced, 0 idle chips):
  Rather than sharding over ``H = 12`` (which forces padding to 16 on 8 chips
  and leaves 2 of 8 chips idle), we shard along the expert axis ``E = 256``
  (``E_local = 32`` experts/head/chip across all 12 heads = 384 local
  experts/chip -> 25.31 GiB/chip BF16 or 12.66 GiB/chip FP8, zero idle chips).
  Routed weights are stored as:
    - ``w_gate_up``: ``(ep, H, E_local, d_head, 2 * d_expert)``  ``P("tp", ...)``
    - ``w_down``:    ``(ep, H, E_local, d_expert, d_head)``      ``P("tp", ...)``

Three selectable execution modes (``DT_MAGI2_MOE_MODE``):
  - ``"ref"``: Single-device exact JAX reference (CPU parity tests / ``ep=1``).
  - ``"ag"``:  100% mathematically zero-drop ``EP=8`` via per-head
    ``lax.all_gather`` + local ``gmm_v2`` over ``E_local=32`` experts +
    ``lax.psum_scatter``. Immune to arbitrary routing skew.
  - ``"ep"``:  Fast ``EP=8`` via per-head ``lax.ragged_all_to_all`` dispatch /
    combine inside ``lax.scan`` over ``H=12`` heads + ``gmm_v2``
    (``zero_initialize=False``) with configurable capacity factor
    (``DT_MAGI2_MOE_CAPACITY_FACTOR``, default ``2.0``) and optional W8A8 FP8
    (``DT_MAGI2_FP8=1``).
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any

import numpy as np

# Top-level JAX / Pallas kernel imports (never import Pallas inside jax.jit!).
try:
  import jax
  from jax import lax
  import jax.numpy as jnp
  from ...kernels import gmm_v2 as _gmm_v2
except ImportError:  # pragma: no cover - allows syntax/module inspection without JAX
  jax = None  # type: ignore[assignment]
  lax = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]
  _gmm_v2 = None  # type: ignore[assignment]

from .rope_and_ops import swiglu7


# ---------------------------------------------------------------------------
# Runtime configuration helpers
# ---------------------------------------------------------------------------


def resolve_moe_mode(mode: str | None, ep_size: int) -> str:
  """Resolve execution mode ('ref', 'ag', or 'ep') from explicit arg or env."""
  raw = (mode or os.environ.get("DT_MAGI2_MOE_MODE", "auto")).strip().lower()
  if raw == "auto":
    return "ref" if ep_size <= 1 else "ag"
  if raw not in ("ref", "ag", "ep"):
    raise ValueError(f"Unsupported MAGI-2 MoE mode {raw!r}; expected 'ref', 'ag', 'ep', or 'auto'")
  if ep_size <= 1 and raw != "ref":
    return "ref"
  return raw


def resolve_capacity_factor(capacity_factor: float | None = None) -> float:
  if capacity_factor is not None:
    return float(capacity_factor)
  return float(os.environ.get("DT_MAGI2_MOE_CAPACITY_FACTOR", "2.0"))


def use_fp8_env() -> bool:
  return os.environ.get("DT_MAGI2_FP8", "0").strip() == "1"


def _is_tpu() -> bool:
  if jax is None:
    return False
  if os.environ.get("DT_FORCE_TPU_TRACE", "").strip() == "1":
    return True
  try:
    devices = jax.devices()
    return bool(devices) and devices[0].platform == "tpu"
  except Exception:
    return False


def _can_use_gmm_v2(
    w_gu: Any,
    w_down: Any,
    out_dtype: Any,
) -> bool:
  """Return True iff TPU Pallas megablox gmm_v2 is valid for these shapes."""
  if _gmm_v2 is None or not _is_tpu():
    return False
  if os.environ.get("DT_MAGI2_MOE_GMM", "1").strip() != "1":
    return False
  if jnp.dtype(out_dtype) != jnp.dtype(jnp.bfloat16):
    return False
  # gmm_v2 requires K aligned to 128, and 2*d_expert aligned to 2*128=256 when
  # fuse_act="swigluoai" is active, and d_head aligned to 128.
  _, d_head, two_d_exp = w_gu.shape
  _, d_exp, d_head_out = w_down.shape
  return (
      d_head % 128 == 0
      and two_d_exp % 256 == 0
      and d_exp % 128 == 0
      and d_head_out % 128 == 0
  )


def _magi2_gmm_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
  """MegaCore Dual-TC tile heuristic for MAGI-2 (d_head=256, d_expert=1280) on TPU v7x.

  Splitting ``out_n`` into 2 tiles (``tile_n=640`` for GMM1 gate/up, ``128`` for
  GMM2 down) engages both TPU7x MegaCore TensorCores simultaneously.
  """
  tile_m_env = int(os.environ.get("DT_MAGI2_GMM_TILE_M", "512"))
  if tile_m_env <= 0:
    return _gmm_v2.calculate_tiling(
        dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=fuse_act
    )
  out_n = dims.size_n // (2 if fuse_act is not None else 1)
  dual_tc = os.environ.get("DT_MAGI2_GMM_DUAL_TC", "1").strip() != "0"
  if dual_tc and out_n >= 256:
    tile_n = max(128, ((out_n // 2) // 128) * 128)
  else:
    tile_n = min(1280, out_n)
  return _gmm_v2.TileSizes(
      tile_m=min(tile_m_env, dims.size_m),
      tile_k=min(1280, dims.size_k),
      tile_n=tile_n,
  )


# ---------------------------------------------------------------------------
# FP8 W8A8 quantization & low-RAM streaming sharding helpers
# ---------------------------------------------------------------------------


def _quantize_fp8_impl(w: Any) -> tuple[Any, Any]:
  """Per-output-channel symmetric E4M3 quantization for grouped weights.

  Args:
    w: Grouped weight array of shape ``(..., E_local, K, N)`` in bf16/fp32.

  Returns:
    ``(qw, scale)`` where ``qw`` has shape ``(..., E_local, K, N)`` in
    ``float8_e4m3fn`` and ``scale`` has shape ``(..., E_local, 1, 1, N)`` in
    ``float32`` (matching ``gmm_v2``'s ``[size_group, num_blocks=1, 1, N]``).
  """
  w32 = w.astype(jnp.float32)
  amax = jnp.maximum(jnp.max(jnp.abs(w32), axis=-2, keepdims=True), jnp.float32(1e-8))
  scale = amax / jnp.float32(448.0)
  qw = jnp.clip(w32 / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
  # Expand (..., E_local, 1, N) -> (..., E_local, 1, 1, N) for gmm_v2 rhs_scale.
  return qw, scale[..., None, :, :]


quantize_moe_fp8 = jax.jit(_quantize_fp8_impl) if jax is not None else _quantize_fp8_impl


def shard_routed_gate_up(
    w_gate_np: np.ndarray,
    w_up_np: np.ndarray,
    *,
    num_heads: int,
    num_experts: int,
    mesh: Any,
    axis_name: str = "tp",
    fp8: bool = False,
    dtype: Any = None,
) -> tuple[Any, Any | None]:
  """Fuse and shard ``W_gate`` & ``W_up`` directly onto devices with minimal host RAM.

  Checkpoint layout: ``W_gate``, ``W_up`` each ``[H * E, d_head, d_expert]``
  where dim 0 is ``h * E + e``.
  Target layout: ``(ep, H, E_local, d_head, 2 * d_expert)`` sharded along
  axis 0 with ``PartitionSpec(axis_name, None, None, None, None)``.

  Slices one chip at a time via ``addressable_devices_indices_map`` so we never
  allocate a full fused 3.75 GiB host buffer.
  """
  from jax.sharding import NamedSharding, PartitionSpec as P

  ep_size = int(dict(mesh.shape).get(axis_name, 1)) if mesh is not None else 1
  if num_experts % ep_size != 0:
    raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
  e_local = num_experts // ep_size
  d_head, d_expert = int(w_gate_np.shape[1]), int(w_gate_np.shape[2])

  g_view = w_gate_np.reshape(num_heads, ep_size, e_local, d_head, d_expert)
  u_view = w_up_np.reshape(num_heads, ep_size, e_local, d_head, d_expert)
  global_shape = (ep_size, num_heads, e_local, d_head, 2 * d_expert)

  if mesh is None:
    arr_np = np.concatenate(
        [np.transpose(g_view, (1, 0, 2, 3, 4)), np.transpose(u_view, (1, 0, 2, 3, 4))],
        axis=-1,
    )
    arr = jnp.asarray(arr_np, dtype=dtype) if dtype is not None else jnp.asarray(arr_np)
    del arr_np
  else:
    sharding = NamedSharding(mesh, P(axis_name, None, None, None, None))
    idx_map = sharding.addressable_devices_indices_map(global_shape)
    shards = []
    for dev, idx in idx_map.items():
      ep_sl = idx[0]
      g_s = np.transpose(g_view[:, ep_sl, :, :, :], (1, 0, 2, 3, 4))
      u_s = np.transpose(u_view[:, ep_sl, :, :, :], (1, 0, 2, 3, 4))
      gu_s = np.ascontiguousarray(np.concatenate([g_s, u_s], axis=-1))
      if dtype is not None and gu_s.dtype != np.dtype(jnp.dtype(dtype)):
        gu_s = gu_s.astype(jnp.dtype(dtype))
      shards.append(jax.device_put(gu_s, dev))
      del g_s, u_s, gu_s
    arr = jax.make_array_from_single_device_arrays(global_shape, sharding, shards)

  if fp8:
    qw, qsc = quantize_moe_fp8(arr)
    del arr
    return qw, qsc
  return arr, None


def shard_routed_down(
    w_down_np: np.ndarray,
    *,
    num_heads: int,
    num_experts: int,
    mesh: Any,
    axis_name: str = "tp",
    fp8: bool = False,
    dtype: Any = None,
) -> tuple[Any, Any | None]:
  """Shard ``W_down`` ``[H * E, d_expert, d_head]`` -> ``(ep, H, E_local, d_expert, d_head)``."""
  from jax.sharding import NamedSharding, PartitionSpec as P

  ep_size = int(dict(mesh.shape).get(axis_name, 1)) if mesh is not None else 1
  if num_experts % ep_size != 0:
    raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
  e_local = num_experts // ep_size
  d_expert, d_head = int(w_down_np.shape[1]), int(w_down_np.shape[2])

  d_view = w_down_np.reshape(num_heads, ep_size, e_local, d_expert, d_head)
  global_shape = (ep_size, num_heads, e_local, d_expert, d_head)

  if mesh is None:
    arr_np = np.ascontiguousarray(np.transpose(d_view, (1, 0, 2, 3, 4)))
    arr = jnp.asarray(arr_np, dtype=dtype) if dtype is not None else jnp.asarray(arr_np)
    del arr_np
  else:
    sharding = NamedSharding(mesh, P(axis_name, None, None, None, None))
    idx_map = sharding.addressable_devices_indices_map(global_shape)
    shards = []
    for dev, idx in idx_map.items():
      ep_sl = idx[0]
      d_s = np.ascontiguousarray(np.transpose(d_view[:, ep_sl, :, :, :], (1, 0, 2, 3, 4)))
      if dtype is not None and d_s.dtype != np.dtype(jnp.dtype(dtype)):
        d_s = d_s.astype(jnp.dtype(dtype))
      shards.append(jax.device_put(d_s, dev))
      del d_s
    arr = jax.make_array_from_single_device_arrays(global_shape, sharding, shards)

  if fp8:
    qw, qsc = quantize_moe_fp8(arr)
    del arr
    return qw, qsc
  return arr, None


# ---------------------------------------------------------------------------
# Core routing & grouped SwiGLU7 FFN primitives
# ---------------------------------------------------------------------------


def route_mh_moe(
    x_heads: Any,
    gate: Any,
    expert_bias: Any | None,
    *,
    top_k: int,
    route_norm: bool = True,
    route_scale: float = 4.9,
    norm_eps: float = 1e-12,
) -> tuple[Any, Any]:
  """Per-head sigmoid top-k router matching ``flash_mh_moe/route.py``.

  Args:
    x_heads: ``(S, H, d_head)`` input tokens split into heads.
    gate: ``(H, E, d_head)`` float32 router weights.
    expert_bias: Optional ``(H, E)`` float32 aux-free routing bias
      (loaded from ``expert_bias_ema``). Added ONLY for top-k selection;
      routing probabilities are gathered from the UNBIASED sigmoid scores!
    top_k: Number of active experts per head per token (6 in MAGI-2 Preview).
    route_norm: L1-normalize gathered top-k probabilities (``eps=1e-12``).
    route_scale: Post-normalization probability multiplier (``4.9``).
    norm_eps: Floor for L1 normalization denominator.

  Returns:
    ``(topk_probs, topk_idx)`` each of shape ``(H, S, top_k)`` in
    ``(float32, int32)``.
  """
  # HIGHEST precision prevents TPU bf16 truncation from flipping borderline top-k.
  logits = jnp.einsum(
      "shd,hed->hse",
      x_heads.astype(jnp.float32),
      gate.astype(jnp.float32),
      precision=lax.Precision.HIGHEST,
  )
  scores = jax.nn.sigmoid(logits)
  topk_scores = scores
  if expert_bias is not None:
    topk_scores = topk_scores + expert_bias[:, None, :].astype(jnp.float32)

  _, topk_idx = lax.top_k(topk_scores, top_k)  # (H, S, K) int32
  # Gather UNBIASED sigmoid scores for routing weights.
  topk_probs = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if route_norm:
    denom = jnp.maximum(
        jnp.sum(jnp.abs(topk_probs), axis=-1, keepdims=True),
        jnp.float32(norm_eps),
    )
    topk_probs = topk_probs / denom
  topk_probs = topk_probs * jnp.float32(route_scale)
  return topk_probs, topk_idx


def _apply_swiglu7_typed(g: Any, u: Any, out_dtype: Any) -> Any:
  """Apply exact SwiGLU7 preserving ``out_dtype`` (bf16 in prod, fp32 in tests)."""
  return swiglu7(g, u, out_dtype=out_dtype)


def _grouped_swiglu_ffn(
    xs: Any,
    w_gu: Any,
    w_down: Any,
    group_sizes: Any,
    *,
    w_gu_scale: Any | None = None,
    w_down_scale: Any | None = None,
    zero_initialize: bool = True,
    out_dtype: Any | None = None,
) -> Any:
  """Grouped ``SwiGLU7(x @ W_gate, x @ W_up) @ W_down`` for one head.

  Uses ``gmm_v2(..., fuse_act="swigluoai")`` on TPU (exact match to MAGI-2
  ``swiglu7``, fusing gate+up+activation in VMEM and supporting W8A8 E4M3 FP8),
  falling back cleanly to ``jax.lax.ragged_dot`` on CPU or small test shapes.

  Args:
    xs: ``(M, d_head)`` tokens sorted by expert.
    w_gu: ``(G, d_head, 2 * d_expert)`` concatenated ``[W_gate, W_up]``.
    w_down: ``(G, d_expert, d_head)``.
    group_sizes: ``(G,)`` int32 active token counts per expert; ``sum <= M``.
    w_gu_scale: Optional ``(G, 1, 1, 2 * d_expert)`` fp32 FP8 scale.
    w_down_scale: Optional ``(G, 1, 1, d_head)`` fp32 FP8 scale.
    zero_initialize: Passed to ``gmm_v2``. Set ``False`` when caller masks tail.
    out_dtype: Output dtype (defaults to ``xs.dtype``).
  """
  out_dt = jnp.dtype(xs.dtype if out_dtype is None else out_dtype)

  if _can_use_gmm_v2(w_gu, w_down, out_dt):
    orig_m = xs.shape[0]
    # gmm_v2 requires M divisible by TPU sublane tiling (16 bf16, 32 fp8).
    align = 32 if w_gu_scale is not None else 16
    pad_m = (-orig_m) % align
    xs_in = jnp.pad(xs, ((0, pad_m), (0, 0))) if pad_m else xs
    xs_in = xs_in.astype(out_dt)

    h = _gmm_v2.gmm_v2(
        xs_in,
        w_gu,
        group_sizes,
        rhs_scale=w_gu_scale,
        fuse_act="swigluoai",
        preferred_element_type=out_dt,
        acc_dtype=jnp.dtype(jnp.float32),
        maybe_quantize_lhs=(w_gu_scale is not None),
        zero_initialize=zero_initialize,
        tile_info=_magi2_gmm_tile_fn,
    )
    y = _gmm_v2.gmm_v2(
        h,
        w_down,
        group_sizes,
        rhs_scale=w_down_scale,
        preferred_element_type=out_dt,
        acc_dtype=jnp.dtype(jnp.float32),
        maybe_quantize_lhs=(w_down_scale is not None),
        zero_initialize=zero_initialize,
        tile_info=_magi2_gmm_tile_fn,
    )
    return y[:orig_m] if pad_m else y

  # Portable fallback: jax.lax.ragged_dot (CPU parity, tiny configs, fp32 tests).
  if w_gu_scale is not None:
    w_gu_eff = (w_gu.astype(jnp.float32) * w_gu_scale[:, 0, :, :]).astype(xs.dtype)
  else:
    w_gu_eff = w_gu.astype(xs.dtype)
  if w_down_scale is not None:
    w_down_eff = (w_down.astype(jnp.float32) * w_down_scale[:, 0, :, :]).astype(out_dt)
  else:
    w_down_eff = w_down.astype(out_dt)

  prec = lax.Precision.HIGHEST if out_dt == jnp.dtype(jnp.float32) else lax.Precision.DEFAULT
  gu = lax.ragged_dot(
      xs,
      w_gu_eff,
      group_sizes,
      precision=prec,
      preferred_element_type=jnp.float32,
  )
  g, u = jnp.split(gu, 2, axis=-1)
  h = _apply_swiglu7_typed(g, u, out_dt)
  y = lax.ragged_dot(
      h,
      w_down_eff,
      group_sizes,
      precision=prec,
      preferred_element_type=jnp.float32,
  )
  return y.astype(out_dt)


# ---------------------------------------------------------------------------
# Mode 1: Single-device exact JAX reference ("ref")
# ---------------------------------------------------------------------------


def mh_moe_forward_ref(
    x: Any,
    gate: Any,
    expert_bias: Any | None,
    w_gate_up: Any,
    w_down: Any,
    *,
    num_heads: int,
    num_experts: int,
    top_k: int,
    route_norm: bool = True,
    route_scale: float = 4.9,
    w_gate_up_scale: Any | None = None,
    w_down_scale: Any | None = None,
) -> Any:
  """Single-device exact JAX reference for ``CoreMultiHeadMoE``.

  Args:
    x: ``(S, H * d_head)``.
    gate: ``(H, E, d_head)`` fp32.
    expert_bias: ``(H, E)`` fp32 or ``None``.
    w_gate_up: ``(H, E, d_head, 2 * d_expert)`` (unsharded ep=1 weights).
    w_down: ``(H, E, d_expert, d_head)``.
  """
  seq_len = x.shape[0]
  d_head = x.shape[1] // num_heads
  out_dtype = x.dtype
  x_heads = x.reshape(seq_len, num_heads, d_head)

  topk_probs, topk_idx = route_mh_moe(
      x_heads,
      gate,
      expert_bias,
      top_k=top_k,
      route_norm=route_norm,
      route_scale=route_scale,
  )  # (H, S, K)

  x_by_head = jnp.transpose(x_heads, (1, 0, 2))  # (H, S, d_head)
  tok_ids_base = jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32), top_k)

  def _head_body(_carry, head_inputs):
    x_h, idx_h, probs_h, w_gu_h, w_down_h, w_gu_sc_h, w_down_sc_h = head_inputs
    flat_e = idx_h.reshape(-1)
    flat_p = probs_h.reshape(-1)
    order = jnp.argsort(flat_e, stable=True)
    sorted_tok = tok_ids_base[order]
    sorted_p = flat_p[order]
    group_sizes = jnp.bincount(flat_e, length=num_experts).astype(jnp.int32)

    xs = x_h[sorted_tok]
    ys = _grouped_swiglu_ffn(
        xs,
        w_gu_h,
        w_down_h,
        group_sizes,
        w_gu_scale=w_gu_sc_h,
        w_down_scale=w_down_sc_h,
        zero_initialize=True,
        out_dtype=out_dtype,
    )
    ys_w = ys.astype(jnp.float32) * sorted_p[:, None]
    y_h = (
        jnp.zeros((seq_len, d_head), dtype=jnp.float32)
        .at[sorted_tok]
        .add(ys_w)
        .astype(out_dtype)
    )
    return None, y_h

  scan_inputs = (
      x_by_head,
      topk_idx,
      topk_probs,
      w_gate_up,
      w_down,
      w_gate_up_scale,
      w_down_scale,
  )
  _, y_by_head = lax.scan(_head_body, None, scan_inputs)  # (H, S, d_head)
  return jnp.transpose(y_by_head, (1, 0, 2)).reshape(seq_len, num_heads * d_head)


# ---------------------------------------------------------------------------
# Mode 2: 100% Zero-Drop All-Gather + Local Experts + Psum-Scatter ("ag")
# ---------------------------------------------------------------------------


def mh_moe_forward_ag(
    x_local: Any,
    gate: Any,
    expert_bias: Any | None,
    w_gate_up_local: Any,
    w_down_local: Any,
    *,
    num_heads: int,
    num_experts: int,
    top_k: int,
    axis_name: str = "tp",
    route_norm: bool = True,
    route_scale: float = 4.9,
    w_gate_up_scale_local: Any | None = None,
    w_down_scale_local: Any | None = None,
    loop_k: bool | None = None,
    capacity_factor: float | None = None,
) -> Any:
  """Fast EP over ``E`` experts via per-head ``all_gather`` + ``psum_scatter``.

  Must be executed inside ``jax.shard_map`` over ``axis_name``.
  Each chip owns ``E_local = E // ep_size`` contiguous experts per head
  (experts ``[rank * E_local, (rank + 1) * E_local)`` across ALL ``H`` heads).
  On large sequences (``S_total * top_k > 16384``), ``order = argsort(sort_key)[:cap]``
  slices ONLY the local routes (no ``cumsum`` needed!) cutting gather, GMM buffer,
  and scatter-add HBM traffic by ~5-6x (90.3 ms -> 51.4 ms/layer on TPU7x-8).
  Set ``DT_MAGI2_AG_CAP_FACTOR=0`` for unsliced mathematically zero-drop mode.

  Args:
    x_local: ``(S_local, H * d_head)`` sequence-sharded input on this chip.
    gate: ``(H, E, d_head)`` replicated fp32 router weights.
    expert_bias: ``(H, E)`` replicated fp32 router bias (from ``expert_bias_ema``).
    w_gate_up_local: ``(H, E_local, d_head, 2 * d_expert)`` local expert weights.
    w_down_local: ``(H, E_local, d_expert, d_head)`` local expert weights.
    loop_k: If True (or ``DT_MAGI2_AG_LOOP_K=1``), scans over ``k = 0..K-1``
      to cut peak activation buffer by ``K``x (useful on smaller-HBM chips);
      default False fuses all ``K`` slots into one ``gmm_v2`` per head so
      weights are read once per head instead of ``K`` times.
    capacity_factor: Local-route capacity multiplier for ``order[:cap]`` slicing
      (default ``DT_MAGI2_AG_CAP_FACTOR`` = ``1.5``; ``<= 0`` disables slicing).
  """
  ep_size = lax.axis_size(axis_name)
  rank = lax.axis_index(axis_name)
  e_local = num_experts // ep_size
  e_start = rank * e_local

  s_local = x_local.shape[0]
  s_total = s_local * ep_size
  d_head = x_local.shape[1] // num_heads
  out_dtype = x_local.dtype

  # Route on local sequence shard first (1/ep_size routing FLOPs).
  x_heads_local = x_local.reshape(s_local, num_heads, d_head)
  topk_probs_local, topk_idx_local = route_mh_moe(
      x_heads_local,
      gate,
      expert_bias,
      top_k=top_k,
      route_norm=route_norm,
      route_scale=route_scale,
  )  # (H, S_local, K)

  x_by_head_local = jnp.transpose(x_heads_local, (1, 0, 2))  # (H, S_local, d_head)
  do_loop_k = (
      loop_k
      if loop_k is not None
      else os.environ.get("DT_MAGI2_AG_LOOP_K", "0").strip() == "1"
  )
  ag_cap_factor = (
      float(capacity_factor)
      if capacity_factor is not None
      else float(os.environ.get("DT_MAGI2_AG_CAP_FACTOR", "2.0"))
  )

  def _head_body(_carry, head_inputs):
    x_hl, idx_hl, probs_hl, w_gu_h, w_down_h, w_gu_sc_h, w_down_sc_h = head_inputs
    # Gather one head at a time: only (S_total, 256) bf16 = ~61 MB at S=120k.
    x_h = lax.all_gather(x_hl, axis_name, axis=0, tiled=True)       # (S_total, d_head)
    idx_h = lax.all_gather(idx_hl, axis_name, axis=0, tiled=True)   # (S_total, K)
    probs_h = lax.all_gather(probs_hl, axis_name, axis=0, tiled=True)  # (S_total, K)

    if do_loop_k:
      # Scan over k=0..K-1 with static buffer size S_total (zero-drop guaranteed).
      idx_by_k = jnp.transpose(idx_h, (1, 0))      # (K, S_total)
      probs_by_k = jnp.transpose(probs_h, (1, 0))  # (K, S_total)
      iota_s = jnp.arange(s_total, dtype=jnp.int32)

      def _k_body(y_acc, k_inputs):
        ek, pk = k_inputs
        local_e = ek - e_start
        is_local = (local_e >= 0) & (local_e < e_local)
        sort_key = jnp.where(is_local, local_e, e_local)
        order = jnp.argsort(sort_key, stable=False)
        group_sizes = jnp.bincount(sort_key, length=e_local + 1)[:e_local].astype(
            jnp.int32
        )
        total_local = jnp.sum(group_sizes)
        xs = x_h[order]
        ys = _grouped_swiglu_ffn(
            xs,
            w_gu_h,
            w_down_h,
            group_sizes,
            w_gu_scale=w_gu_sc_h,
            w_down_scale=w_down_sc_h,
            zero_initialize=False,
            out_dtype=out_dtype,
        )
        valid = (iota_s < total_local)[:, None]
        # IEEE select does not propagate uninitialized tail values where valid=False.
        ys_w = jnp.where(
            valid,
            ys.astype(jnp.float32) * pk[order, None],
            jnp.float32(0.0),
        )
        inv = jnp.empty_like(order).at[order].set(iota_s)
        y_acc = y_acc + ys_w[inv]
        return y_acc, None

      y_partial_f32, _ = lax.scan(
          _k_body,
          jnp.zeros((s_total, d_head), dtype=jnp.float32),
          (idx_by_k, probs_by_k),
      )
    else:
      # Combined single-GMM over all K slots: reads expert weights only 1x/head!
      m_all = s_total * top_k
      flat_e = idx_h.reshape(m_all)
      flat_p = probs_h.reshape(m_all)
      flat_tok = jnp.repeat(jnp.arange(s_total, dtype=jnp.int32), top_k)

      local_e = flat_e - e_start
      is_local = (local_e >= 0) & (local_e < e_local)
      sort_key = jnp.where(is_local, local_e, e_local)
      group_sizes = jnp.bincount(sort_key, length=e_local + 1)[:e_local].astype(
          jnp.int32
      )

      if ag_cap_factor > 0.0 and ep_size > 1 and m_all > 16384:
        raw_cap = int(math.ceil((m_all / ep_size) * ag_cap_factor))
        cap = min(m_all, ((raw_cap + 31) // 32) * 32)
      else:
        cap = m_all

      # All local routes sort to [0, total_local); slicing [:cap] avoids cumsum
      # and cuts gather/GMM/scatter-add HBM volume by ~5-6x!
      order = jnp.argsort(sort_key, stable=False)[:cap]
      if cap < m_all:
        # Defensively clamp sum(group_sizes) <= cap (32 ints, <1 us) so gmm_v2
        # never touches out-of-bounds rows even under extreme router skew.
        gs_end = jnp.minimum(jnp.cumsum(group_sizes), jnp.int32(cap))
        gs_start = jnp.concatenate([jnp.zeros((1,), jnp.int32), gs_end[:-1]])
        group_sizes = gs_end - gs_start
        total_local = gs_end[-1]
      else:
        total_local = jnp.sum(group_sizes)

      sorted_tok = flat_tok[order]
      sorted_p = flat_p[order]

      xs = x_h[sorted_tok]
      ys = _grouped_swiglu_ffn(
          xs,
          w_gu_h,
          w_down_h,
          group_sizes,
          w_gu_scale=w_gu_sc_h,
          w_down_scale=w_down_sc_h,
          zero_initialize=False,
          out_dtype=out_dtype,
      )
      valid = (jnp.arange(cap, dtype=jnp.int32) < total_local)[:, None]
      ys_w = jnp.where(
          valid,
          ys.astype(jnp.float32) * sorted_p[:, None],
          jnp.float32(0.0),
      )
      y_partial_f32 = jnp.zeros((s_total, d_head), dtype=jnp.float32).at[sorted_tok].add(ys_w)

    y_partial = y_partial_f32.astype(out_dtype)
    y_h_local = lax.psum_scatter(y_partial, axis_name, scatter_dimension=0, tiled=True)
    return None, y_h_local

  scan_inputs = (
      x_by_head_local,
      topk_idx_local,
      topk_probs_local,
      w_gate_up_local,
      w_down_local,
      w_gate_up_scale_local,
      w_down_scale_local,
  )
  _, y_by_head_local = lax.scan(_head_body, None, scan_inputs)  # (H, S_local, d_head)
  return jnp.transpose(y_by_head_local, (1, 0, 2)).reshape(s_local, num_heads * d_head)


# ---------------------------------------------------------------------------
# Mode 3: Fast Expert-Parallel via ragged_all_to_all ("ep")
# ---------------------------------------------------------------------------


def mh_moe_forward_ep(
    x_local: Any,
    gate: Any,
    expert_bias: Any | None,
    w_gate_up_local: Any,
    w_down_local: Any,
    *,
    num_heads: int,
    num_experts: int,
    top_k: int,
    axis_name: str = "tp",
    capacity_factor: float = 2.0,
    route_norm: bool = True,
    route_scale: float = 4.9,
    w_gate_up_scale_local: Any | None = None,
    w_down_scale_local: Any | None = None,
) -> Any:
  """Fast ``EP=8`` Multi-Head MoE via per-head ``lax.ragged_all_to_all``.

  Must be executed inside ``jax.shard_map`` over ``axis_name``.
  Dispatches only routed token-expert pairs to the chip owning each expert
  (``dest = e // E_local``), sorts received tokens across the ``E_local = 32``
  local experts, runs ``gmm_v2(..., zero_initialize=False)`` (with optional
  W8A8 FP8), and combines via reverse ``ragged_all_to_all``.

  Capacity handling:
    Receive buffer capacity is ``cap = ceil(S_local * K * capacity_factor)``
    rounded up to a multiple of 32 (capped at the zero-drop maximum
    ``ep_size * S_local * K``). Set ``capacity_factor <= 0`` or ``>= ep_size``
    for guaranteed zero-drop EP. Whenever total received rows exceed ``cap``,
    excess tail counts are symmetrically clipped across all chips so
    ``ragged_all_to_all`` never writes out of bounds and dropped slots cleanly
    contribute ``0.0``.
  """
  ep_size = lax.axis_size(axis_name)
  rank = lax.axis_index(axis_name)
  e_local = num_experts // ep_size

  s_local = x_local.shape[0]
  d_head = x_local.shape[1] // num_heads
  out_dtype = x_local.dtype
  m_send = s_local * top_k
  max_cap = m_send * ep_size

  if capacity_factor <= 0.0 or capacity_factor >= float(ep_size):
    cap_raw = max_cap
  else:
    cap_raw = min(max_cap, max(m_send, int(math.ceil(m_send * capacity_factor))))
  # Round capacity up to a multiple of 32 for gmm_v2 bf16/fp8 sublane alignment.
  cap = ((cap_raw + 31) // 32) * 32

  x_heads_local = x_local.reshape(s_local, num_heads, d_head)
  topk_probs_local, topk_idx_local = route_mh_moe(
      x_heads_local,
      gate,
      expert_bias,
      top_k=top_k,
      route_norm=route_norm,
      route_scale=route_scale,
  )  # (H, S_local, K)

  x_by_head_local = jnp.transpose(x_heads_local, (1, 0, 2))  # (H, S_local, d_head)
  flat_tok_base = jnp.repeat(jnp.arange(s_local, dtype=jnp.int32), top_k)
  iota_cap = jnp.arange(cap, dtype=jnp.int32)

  def _head_body(_carry, head_inputs):
    x_hl, idx_hl, probs_hl, w_gu_h, w_down_h, w_gu_sc_h, w_down_sc_h = head_inputs
    flat_e = idx_hl.reshape(m_send)
    flat_p = probs_hl.reshape(m_send)
    dest_chip = flat_e // e_local
    local_e = flat_e - dest_chip * e_local

    # Sorting by flat_e groups by dest_chip AND orders by local_e inside each chunk.
    send_order = jnp.argsort(flat_e, stable=False)
    sorted_tok = flat_tok_base[send_order]
    sorted_p = flat_p[send_order]
    xs_send = x_hl[sorted_tok]
    e_send = local_e[send_order].astype(jnp.int32)

    send_sz = jnp.bincount(dest_chip, length=ep_size).astype(jnp.int32)  # (ep,)
    all_cnt = lax.all_gather(send_sz, axis_name, axis=0, tiled=False)    # (ep, ep)

    # Compute exact, capacity-clipped (src, dst) receive layout identically on all chips.
    col_end = jnp.cumsum(all_cnt, axis=0)
    col_start = col_end - all_cnt
    eff_end = jnp.minimum(col_end, jnp.int32(cap))
    eff_start_raw = jnp.minimum(col_start, jnp.int32(cap))
    eff_cnt = eff_end - eff_start_raw  # (ep, ep); sum over src <= cap
    eff_start = jnp.where(eff_cnt > 0, eff_start_raw, jnp.int32(0))

    all_in_off_raw = jnp.cumsum(all_cnt, axis=1) - all_cnt  # (ep, ep)
    my_send_sz = eff_cnt[rank, :]
    my_recv_sz = eff_cnt[:, rank]
    my_in_off = jnp.where(my_send_sz > 0, all_in_off_raw[rank, :], jnp.int32(0))
    my_out_off = eff_start[rank, :]

    # Forward ragged_all_to_all: dispatch tokens & their local expert IDs.
    recv_x = lax.ragged_all_to_all(
        xs_send,
        jnp.empty((cap, d_head), dtype=out_dtype),
        my_in_off,
        my_send_sz,
        my_out_off,
        my_recv_sz,
        axis_name=axis_name,
    )
    recv_e_raw = lax.ragged_all_to_all(
        e_send[:, None],
        jnp.full((cap, 1), e_local, dtype=jnp.int32),
        my_in_off,
        my_send_sz,
        my_out_off,
        my_recv_sz,
        axis_name=axis_name,
    )[:, 0]

    # Mask tail slots past tot_recv into dummy bucket e_local so they are skipped.
    tot_recv = jnp.sum(my_recv_sz).astype(jnp.int32)
    recv_e = jnp.where(iota_cap < tot_recv, recv_e_raw, jnp.int32(e_local))

    # Regroup tokens from all source chips by local expert [0..e_local).
    r_order = jnp.argsort(recv_e, stable=False)
    r_inv = jnp.empty_like(r_order).at[r_order].set(iota_cap)
    group_sizes = jnp.bincount(recv_e, length=e_local + 1)[:e_local].astype(jnp.int32)

    xs_local = recv_x[r_order]
    ys_local = _grouped_swiglu_ffn(
        xs_local,
        w_gu_h,
        w_down_h,
        group_sizes,
        w_gu_scale=w_gu_sc_h,
        w_down_scale=w_down_sc_h,
        zero_initialize=False,
        out_dtype=out_dtype,
    )
    out_recv = ys_local[r_inv]

    # Reverse ragged_all_to_all: return expert outputs to originating chips.
    # Zero-initialized back_buf ensures any capacity-dropped slots stay 0.0.
    rev_out_off = jnp.where(my_recv_sz > 0, all_in_off_raw[:, rank], jnp.int32(0))
    back = lax.ragged_all_to_all(
        out_recv,
        jnp.zeros((m_send, d_head), dtype=out_dtype),
        eff_start[:, rank],
        my_recv_sz,
        rev_out_off,
        my_send_sz,
        axis_name=axis_name,
    )
    back_w = back.astype(jnp.float32) * sorted_p[:, None]
    y_h_local = (
        jnp.zeros((s_local, d_head), dtype=jnp.float32)
        .at[sorted_tok]
        .add(back_w)
        .astype(out_dtype)
    )
    return None, y_h_local

  scan_inputs = (
      x_by_head_local,
      topk_idx_local,
      topk_probs_local,
      w_gate_up_local,
      w_down_local,
      w_gate_up_scale_local,
      w_down_scale_local,
  )
  _, y_by_head_local = lax.scan(_head_body, None, scan_inputs)  # (H, S_local, d_head)
  return jnp.transpose(y_by_head_local, (1, 0, 2)).reshape(s_local, num_heads * d_head)


# ---------------------------------------------------------------------------
# Unified dispatch helper (inside shard_map)
# ---------------------------------------------------------------------------


def mh_moe_forward_local(
    x_local: Any,
    gate: Any,
    expert_bias: Any | None,
    w_gate_up_local: Any,
    w_down_local: Any,
    *,
    num_heads: int,
    num_experts: int,
    top_k: int,
    ep_size: int = 1,
    axis_name: str = "tp",
    mode: str | None = None,
    capacity_factor: float | None = None,
    route_norm: bool = True,
    route_scale: float = 4.9,
    w_gate_up_scale_local: Any | None = None,
    w_down_scale_local: Any | None = None,
) -> Any:
  """Dispatch to 'ref', 'ag', or 'ep' given already-sliced per-chip weights.

  ``w_gate_up_local`` has shape ``(H, E_local, d_head, 2 * d_expert)`` and
  ``w_down_local`` has shape ``(H, E_local, d_expert, d_head)``.
  """
  resolved = resolve_moe_mode(mode, ep_size)
  if resolved == "ref":
    return mh_moe_forward_ref(
        x_local,
        gate,
        expert_bias,
        w_gate_up_local,
        w_down_local,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        route_norm=route_norm,
        route_scale=route_scale,
        w_gate_up_scale=w_gate_up_scale_local,
        w_down_scale=w_down_scale_local,
    )
  if resolved == "ag":
    return mh_moe_forward_ag(
        x_local,
        gate,
        expert_bias,
        w_gate_up_local,
        w_down_local,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        axis_name=axis_name,
        route_norm=route_norm,
        route_scale=route_scale,
        w_gate_up_scale_local=w_gate_up_scale_local,
        w_down_scale_local=w_down_scale_local,
    )
  return mh_moe_forward_ep(
      x_local,
      gate,
      expert_bias,
      w_gate_up_local,
      w_down_local,
      num_heads=num_heads,
      num_experts=num_experts,
      top_k=top_k,
      axis_name=axis_name,
      capacity_factor=resolve_capacity_factor(capacity_factor),
      route_norm=route_norm,
      route_scale=route_scale,
      w_gate_up_scale_local=w_gate_up_scale_local,
      w_down_scale_local=w_down_scale_local,
  )


# ---------------------------------------------------------------------------
# Flax NNX Module (lazy flax import following diffusers-tpu convention)
# ---------------------------------------------------------------------------


def _build_modules():
  """Lazily import Flax NNX and construct ``CoreMultiHeadMoE``."""
  from flax import nnx
  from jax.sharding import PartitionSpec as P

  class CoreMultiHeadMoE(nnx.Module):
    """Multi-Head Latent MoE core block sharded along ``E`` experts/head.

    Parameter layouts (global shapes):
      - ``gate``: ``(H, E, d_head)`` float32, replicated ``P()``.
      - ``expert_bias``: ``(H, E)`` float32 (from ``expert_bias_ema``), ``P()``.
      - ``w_gate_up``: ``(ep, H, E_local, d_head, 2 * d_expert)`` bf16 or e4m3,
        sharded ``P(axis_name, None, None, None, None)``.
      - ``w_down``: ``(ep, H, E_local, d_expert, d_head)`` bf16 or e4m3,
        sharded ``P(axis_name, None, None, None, None)``.
      - ``w_gate_up_scale`` / ``w_down_scale``: optional FP8 per-channel scales
        ``(ep, H, E_local, 1, 1, N)`` float32, sharded along axis 0.
    """

    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 12,
        num_experts: int = 256,
        top_k: int = 6,
        expert_intermediate_size: int = 1280,
        *,
        route_norm: bool = True,
        route_scale: float = 4.9,
        ep_size: int = 1,
        axis_name: str = "tp",
        dtype: Any = None,
        fp8: bool = False,
        mode: str | None = None,
        capacity_factor: float | None = None,
    ):
      if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
      if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
      self.hidden_size = hidden_size
      self.num_heads = num_heads
      self.num_experts = num_experts
      self.top_k = top_k
      self.d_head = hidden_size // num_heads
      self.d_expert = expert_intermediate_size
      self.route_norm = route_norm
      self.route_scale = float(route_scale)
      self.ep_size = ep_size
      self.e_local = num_experts // ep_size
      self.axis_name = axis_name
      self.fp8 = fp8
      self.mode = mode
      self.capacity_factor = capacity_factor

      w_dtype = jnp.float8_e4m3fn if fp8 else (dtype if dtype is not None else jnp.bfloat16)
      self.gate = nnx.Param(jnp.zeros((num_heads, num_experts, self.d_head), jnp.float32))
      self.expert_bias = nnx.Param(jnp.zeros((num_heads, num_experts), jnp.float32))
      self.w_gate_up = nnx.Param(
          jnp.zeros((ep_size, num_heads, self.e_local, self.d_head, 2 * self.d_expert), w_dtype)
      )
      self.w_down = nnx.Param(
          jnp.zeros((ep_size, num_heads, self.e_local, self.d_expert, self.d_head), w_dtype)
      )
      if fp8:
        self.w_gate_up_scale = nnx.Param(
            jnp.ones((ep_size, num_heads, self.e_local, 1, 1, 2 * self.d_expert), jnp.float32)
        )
        self.w_down_scale = nnx.Param(
            jnp.ones((ep_size, num_heads, self.e_local, 1, 1, self.d_head), jnp.float32)
        )
      else:
        self.w_gate_up_scale = None
        self.w_down_scale = None

    def call_in_shard_map(
        self,
        x_local: Any,
        *,
        mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      """Run inside an enclosing ``shard_map`` where axis 0 of weights is already 1."""
      w_gu_loc = self.w_gate_up[...][0]
      w_dn_loc = self.w_down[...][0]
      w_gu_sc_loc = self.w_gate_up_scale[...][0] if self.w_gate_up_scale is not None else None
      w_dn_sc_loc = self.w_down_scale[...][0] if self.w_down_scale is not None else None
      return mh_moe_forward_local(
          x_local,
          self.gate[...],
          self.expert_bias[...],
          w_gu_loc,
          w_dn_loc,
          num_heads=self.num_heads,
          num_experts=self.num_experts,
          top_k=self.top_k,
          ep_size=self.ep_size,
          axis_name=self.axis_name,
          mode=mode if mode is not None else self.mode,
          capacity_factor=(
              capacity_factor if capacity_factor is not None else self.capacity_factor
          ),
          route_norm=self.route_norm,
          route_scale=self.route_scale,
          w_gate_up_scale_local=w_gu_sc_loc,
          w_down_scale_local=w_dn_sc_loc,
      )

    def __call__(
        self,
        x: Any,
        *,
        mesh: Any = None,
        mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      """Standalone invocation (wraps in ``jax.shard_map`` when ``mesh`` & ``ep > 1``)."""
      eff_mode = mode if mode is not None else self.mode
      eff_cap = capacity_factor if capacity_factor is not None else self.capacity_factor
      if mesh is None or self.ep_size <= 1:
        return self.call_in_shard_map(x, mode="ref", capacity_factor=eff_cap)

      ax = self.axis_name
      w_spec = P(ax, None, None, None, None)
      sc_spec = P(ax, None, None, None, None, None)

      if self.w_gate_up_scale is not None:
        def _run_fp8(x_, gate_, bias_, wgu_, wdn_, wgusc_, wdnsc_):
          return mh_moe_forward_local(
              x_,
              gate_,
              bias_,
              wgu_[0],
              wdn_[0],
              num_heads=self.num_heads,
              num_experts=self.num_experts,
              top_k=self.top_k,
              ep_size=self.ep_size,
              axis_name=ax,
              mode=eff_mode,
              capacity_factor=eff_cap,
              route_norm=self.route_norm,
              route_scale=self.route_scale,
              w_gate_up_scale_local=wgusc_[0],
              w_down_scale_local=wdnsc_[0],
          )

        fn = jax.shard_map(
            _run_fp8,
            mesh=mesh,
            in_specs=(P(ax, None), P(), P(), w_spec, w_spec, sc_spec, sc_spec),
            out_specs=P(ax, None),
            check_vma=False,
        )
        return fn(
            x,
            self.gate[...],
            self.expert_bias[...],
            self.w_gate_up[...],
            self.w_down[...],
            self.w_gate_up_scale[...],
            self.w_down_scale[...],
        )

      def _run_bf16(x_, gate_, bias_, wgu_, wdn_):
        return mh_moe_forward_local(
            x_,
            gate_,
            bias_,
            wgu_[0],
            wdn_[0],
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=self.ep_size,
            axis_name=ax,
            mode=eff_mode,
            capacity_factor=eff_cap,
            route_norm=self.route_norm,
            route_scale=self.route_scale,
        )

      fn = jax.shard_map(
          _run_bf16,
          mesh=mesh,
          in_specs=(P(ax, None), P(), P(), w_spec, w_spec),
          out_specs=P(ax, None),
          check_vma=False,
      )
      return fn(
          x,
          self.gate[...],
          self.expert_bias[...],
          self.w_gate_up[...],
          self.w_down[...],
      )

  return CoreMultiHeadMoE


# ---------------------------------------------------------------------------
# Tiny self-test: ref vs exact NumPy/PyTorch logic + multi-device AG & EP
# ---------------------------------------------------------------------------


def _numpy_ref_mh_moe(
    x: np.ndarray,
    gate: np.ndarray,
    expert_bias: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    num_heads: int,
    num_experts: int,
    top_k: int,
    route_norm: bool = True,
    route_scale: float = 4.9,
) -> np.ndarray:
  """Exact NumPy reference mirroring ``magi2_preview.py`` + ``mh_moe_fwd.py``."""
  s, d_model = x.shape
  d_head = d_model // num_heads
  x_heads = x.reshape(s, num_heads, d_head).astype(np.float64)
  gate_h = gate.reshape(num_heads, num_experts, d_head).astype(np.float64)
  bias_h = expert_bias.reshape(num_heads, num_experts).astype(np.float64)
  wg_h = w_gate.reshape(num_heads, num_experts, d_head, -1).astype(np.float64)
  wu_h = w_up.reshape(num_heads, num_experts, d_head, -1).astype(np.float64)
  wd_h = w_down.reshape(num_heads, num_experts, -1, d_head).astype(np.float64)

  logits = np.einsum("shd,hed->hse", x_heads, gate_h)
  scores = 1.0 / (1.0 + np.exp(-logits))
  biased = scores + bias_h[:, None, :]

  out = np.zeros((s, num_heads, d_head), dtype=np.float64)
  for h in range(num_heads):
    for tok in range(s):
      idx = np.argsort(-biased[h, tok])[:top_k]
      probs = scores[h, tok, idx]
      if route_norm:
        probs = probs / max(np.abs(probs).sum(), 1e-12)
      probs = probs * route_scale
      for k in range(top_k):
        e = int(idx[k])
        g = x_heads[tok, h] @ wg_h[h, e]
        u = x_heads[tok, h] @ wu_h[h, e]
        g_c = np.minimum(g, 7.0)
        u_c = np.clip(u, -7.0, 7.0)
        hid = (g_c / (1.0 + np.exp(-1.702 * g_c))) * (u_c + 1.0)
        out[tok, h] += probs[k] * (hid @ wd_h[h, e])
  return out.reshape(s, d_model).astype(np.float32)


def test_tiny_parity(verbose: bool = True) -> bool:
  """Verify ``CoreMultiHeadMoE`` ('ref', 'ag', 'ep', and FP8) against NumPy reference."""
  from flax import nnx
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  rng = np.random.default_rng(42)
  seq_len = 16
  num_heads = 3
  num_experts = 16
  top_k = 4
  d_head = 8
  d_expert = 12
  hidden_size = num_heads * d_head

  x_np = rng.standard_normal((seq_len, hidden_size)).astype(np.float32) * 0.5
  gate_np = rng.standard_normal((num_heads * num_experts, d_head)).astype(np.float32) * 0.2
  bias_np = rng.standard_normal((num_heads * num_experts,)).astype(np.float32) * 0.05
  wg_np = rng.standard_normal((num_heads * num_experts, d_head, d_expert)).astype(np.float32) * 0.1
  wu_np = rng.standard_normal((num_heads * num_experts, d_head, d_expert)).astype(np.float32) * 0.1
  wd_np = rng.standard_normal((num_heads * num_experts, d_expert, d_head)).astype(np.float32) * 0.1

  want = _numpy_ref_mh_moe(
      x_np,
      gate_np,
      bias_np,
      wg_np,
      wu_np,
      wd_np,
      num_heads=num_heads,
      num_experts=num_experts,
      top_k=top_k,
      route_norm=True,
      route_scale=4.9,
  )

  CoreMultiHeadMoE = _build_modules()

  # 1. Single-device ref parity (fp32).
  moe_ref = CoreMultiHeadMoE(
      hidden_size=hidden_size,
      num_heads=num_heads,
      num_experts=num_experts,
      top_k=top_k,
      expert_intermediate_size=d_expert,
      ep_size=1,
      dtype=jnp.float32,
  )
  w_gu, _ = shard_routed_gate_up(
      wg_np, wu_np, num_heads=num_heads, num_experts=num_experts, mesh=None, dtype=jnp.float32
  )
  w_dn, _ = shard_routed_down(
      wd_np, num_heads=num_heads, num_experts=num_experts, mesh=None, dtype=jnp.float32
  )
  moe_ref.gate.value = jnp.asarray(gate_np.reshape(num_heads, num_experts, d_head))
  moe_ref.expert_bias.value = jnp.asarray(bias_np.reshape(num_heads, num_experts))
  moe_ref.w_gate_up.value = w_gu
  moe_ref.w_down.value = w_dn

  gdef, state = nnx.split(moe_ref)

  @jax.jit
  def _run_ref(st, x_):
    return nnx.merge(gdef, st)(x_)

  got_ref = np.asarray(_run_ref(state, jnp.asarray(x_np)))
  err_ref = float(np.max(np.abs(want - got_ref)))
  ok = err_ref < 1e-4
  if verbose:
    print(f"[jax_mh_moe] ref fp32 max|diff| = {err_ref:.3e} ({'PASS' if ok else 'FAIL'})")

  # 2. Multi-device EP/AG test when multiple JAX devices are available.
  devs = jax.devices()
  ep_candidates = [d for d in (len(devs), 4, 2) if d <= len(devs) and num_experts % d == 0 and seq_len % d == 0]
  if ep_candidates and ep_candidates[0] > 1:
    ep = ep_candidates[0]
    mesh = Mesh(np.array(devs[:ep]), ("tp",))
    x_sh = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh, P("tp", None)))
    for test_mode in ("ag", "ep"):
      moe_ep = CoreMultiHeadMoE(
          hidden_size=hidden_size,
          num_heads=num_heads,
          num_experts=num_experts,
          top_k=top_k,
          expert_intermediate_size=d_expert,
          ep_size=ep,
          dtype=jnp.float32,
          mode=test_mode,
          capacity_factor=float(ep),  # zero-drop parity check
      )
      w_gu_ep, _ = shard_routed_gate_up(
          wg_np, wu_np, num_heads=num_heads, num_experts=num_experts, mesh=mesh, dtype=jnp.float32
      )
      w_dn_ep, _ = shard_routed_down(
          wd_np, num_heads=num_heads, num_experts=num_experts, mesh=mesh, dtype=jnp.float32
      )
      moe_ep.gate.value = jax.device_put(
          jnp.asarray(gate_np.reshape(num_heads, num_experts, d_head)), NamedSharding(mesh, P())
      )
      moe_ep.expert_bias.value = jax.device_put(
          jnp.asarray(bias_np.reshape(num_heads, num_experts)), NamedSharding(mesh, P())
      )
      moe_ep.w_gate_up.value = w_gu_ep
      moe_ep.w_down.value = w_dn_ep
      gd_ep, st_ep = nnx.split(moe_ep)

      @jax.jit
      def _run_ep(st, x_, _gd=gd_ep, _mesh=mesh):
        return nnx.merge(_gd, st)(x_, mesh=_mesh)

      got_ep = np.asarray(_run_ep(st_ep, x_sh))
      err_ep = float(np.max(np.abs(want - got_ep)))
      pass_ep = err_ep < 1e-4
      ok &= pass_ep
      if verbose:
        print(
            f"[jax_mh_moe] {test_mode} (ep={ep}) fp32 max|diff| = {err_ep:.3e} "
            f"({'PASS' if pass_ep else 'FAIL'})"
        )

  # 3. FP8 quantization round-trip sanity check.
  w_gu_q, w_gu_sc = shard_routed_gate_up(
      wg_np, wu_np, num_heads=num_heads, num_experts=num_experts, mesh=None, fp8=True, dtype=jnp.bfloat16
  )
  w_dn_q, w_dn_sc = shard_routed_down(
      wd_np, num_heads=num_heads, num_experts=num_experts, mesh=None, fp8=True, dtype=jnp.bfloat16
  )
  moe_fp8 = CoreMultiHeadMoE(
      hidden_size=hidden_size,
      num_heads=num_heads,
      num_experts=num_experts,
      top_k=top_k,
      expert_intermediate_size=d_expert,
      ep_size=1,
      fp8=True,
  )
  moe_fp8.gate.value = jnp.asarray(gate_np.reshape(num_heads, num_experts, d_head))
  moe_fp8.expert_bias.value = jnp.asarray(bias_np.reshape(num_heads, num_experts))
  moe_fp8.w_gate_up.value = w_gu_q
  moe_fp8.w_down.value = w_dn_q
  moe_fp8.w_gate_up_scale.value = w_gu_sc
  moe_fp8.w_down_scale.value = w_dn_sc
  gd_f8, st_f8 = nnx.split(moe_fp8)

  @jax.jit
  def _run_f8(st, x_):
    return nnx.merge(gd_f8, st)(x_)

  got_f8 = np.asarray(_run_f8(st_f8, jnp.asarray(x_np, jnp.bfloat16)), dtype=np.float32)
  rel_f8 = float(np.linalg.norm(want - got_f8) / max(np.linalg.norm(want), 1e-8))
  pass_f8 = rel_f8 < 0.08
  ok &= pass_f8
  if verbose:
    print(f"[jax_mh_moe] fp8-w8a8 rel_l2 = {rel_f8:.3e} ({'PASS' if pass_f8 else 'FAIL'})")

  return ok


if __name__ == "__main__":
  passed = test_tiny_parity(verbose=True)
  raise SystemExit(0 if passed else 1)
