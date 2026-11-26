"""Core math ops, 3D partial RoPE, mHC, block-grid permutation, and schedulers for MAGI-2.

Native JAX / NumPy implementation of Sand.ai MAGI-2-preview (and Refiner) primitives,
matching the PyTorch reference in ``MAGI-2-preview/inference/`` to machine precision:

* ``swiglu7`` / ``swiglu7_interleaved``: Clamped QuickGELU-SwiGLU + 1 (GPT-OSS variant)
  computed in float32 and cast to bfloat16; upper-only clamp on gate (``max=7.0``),
  symmetric clamp on up (``[-7.0, 7.0]``).
* ``rms_norm_1pw`` / ``multi_modality_rms_norm``: Gemma-style ``(1 + w)`` RMSNorm in
  float32 supporting single-modality ``(D,)`` and 3-modality ``(3, D)`` / flat ``(3*D,)``
  weights over contiguous ``[VIDEO, AUDIO, TEXT]`` splits or explicit ``modality_ids``.
* ``sinkhorn_knopp`` / ``mhc_pre`` / ``mhc_post``: 4-stream Manifold-Constrained
  Hyper-Connections (mHC) with ``matmul_scale = 1 / sqrt(n * D)``, shared ``mhc_norm``,
  20-iteration column-first then row-first Sinkhorn-Knopp in float32 -> bfloat16, and
  ``2 * sigmoid`` post-scale.
* ``sinusoidal_embedding_1d``: Timestep embedding (cos first, sin second, ``pos = raw_t * 1000``).
* ``elementwise_fourier_embed`` / ``apply_3d_partial_rope`` / ``build_preview_rope_coords`` /
  ``build_refiner_rope_coords``: Exact 3D partial RoPE rotating channels ``[0:96]``
  (16 bands/axis, split-half ``(d, d+48)``) while passing ``[96:128]`` untouched.
* ``refiner_block_perm``: 3D ``(8, 4, 4)`` block-grid permutation matching
  ``Magi2RefinerDataProxy._build_block_order``.
* ``FlowUniPCMultistepScheduler``: Pure NumPy/JAX port of ``sampler.py``'s BH2
  2nd-order UniPC flow-matching scheduler.
* ``zero_snr_ddpm_sigmas`` / ``refiner_noise_augment`` / ``trilinear_upsample_latents``:
  Stage 1 -> Stage 2 exact ``align_corners=True`` trilinear latent upsampling and
  Zero-Terminal-SNR DDPM VP noise augmentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # pragma: no cover - allows importing pure-NumPy helpers without JAX
  jax = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]


__all__ = [
    "swiglu7",
    "swiglu7_interleaved",
    "rms_norm_1pw",
    "multi_modality_rms_norm",
    "sinkhorn_knopp",
    "mhc_pre",
    "mhc_post",
    "sinusoidal_embedding_1d",
    "elementwise_fourier_embed",
    "apply_3d_partial_rope",
    "build_preview_rope_coords",
    "build_refiner_rope_coords",
    "refiner_block_perm",
    "SchedulerOutput",
    "FlowUniPCMultistepScheduler",
    "zero_snr_ddpm_sigmas",
    "refiner_noise_augment",
    "trilinear_upsample_latents",
]


def _is_jax_array(x: Any) -> bool:
  return jax is not None and isinstance(x, jax.Array)


def _require_jax(fn_name: str) -> None:
  if jax is None or jnp is None:
    raise ImportError(f"{fn_name} requires JAX to be installed.")


# ---------------------------------------------------------------------------
# 1. SwiGLU7 activation (magi2_preview.py:L87-98, mh_moe_fwd.py:L19-30)
# ---------------------------------------------------------------------------


def swiglu7(
    g: Any,
    u: Any,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: Any = None,
) -> Any:
  """Clamped QuickGELU-SwiGLU + 1 (GPT-OSS variant), matching MAGI-2.

  Computes in float32 and casts to ``out_dtype`` (default ``jnp.bfloat16``):
    g_c = min(g, limit)                # upper clamp ONLY
    u_c = clip(u, -limit, limit)       # symmetric clamp
    out = (g_c * sigmoid(alpha * g_c)) * (u_c + 1.0)
  """
  _require_jax("swiglu7")
  dtype = jnp.bfloat16 if out_dtype is None else out_dtype
  g_f = jnp.minimum(g.astype(jnp.float32), jnp.float32(limit))
  u_f = jnp.clip(u.astype(jnp.float32), jnp.float32(-limit), jnp.float32(limit))
  out_glu = g_f * jax.nn.sigmoid(jnp.float32(alpha) * g_f)
  return (out_glu * (u_f + jnp.float32(1.0))).astype(dtype)


def swiglu7_interleaved(
    x: Any,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: Any = None,
) -> Any:
  """Interleaved SwiGLU7 for dense MLPs & shared experts (even=gate, odd=up)."""
  return swiglu7(
      x[..., ::2],
      x[..., 1::2],
      alpha=alpha,
      limit=limit,
      out_dtype=out_dtype,
  )


# ---------------------------------------------------------------------------
# 2-3. Gemma-style (1 + w) RMSNorm & MultiModalityRMSNorm
#      (magi2_preview.py:L497-624, L1448-1520)
# ---------------------------------------------------------------------------


def rms_norm_1pw(
    x: Any,
    weight: Any,
    eps: float = 1e-6,
    cast_back: bool = True,
) -> Any:
  """Gemma-style ``(1 + w)`` RMSNorm computed in float32.

  Args:
    x: Input array of shape ``(..., D)``.
    weight: Zero-centered scale parameter of shape ``(D,)`` (or broadcastable to ``x``).
    eps: Numerical stability epsilon (default ``1e-6``).
    cast_back: If True, cast output back to ``x.dtype``; if False, keep ``float32``
      (used for Preview QK-norm and mHC norm).
  """
  _require_jax("rms_norm_1pw")
  orig_dtype = x.dtype
  x_f = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(x_f * x_f, axis=-1, keepdims=True) + jnp.float32(eps)
  )
  w_f = jnp.asarray(weight, dtype=jnp.float32)
  y = x_f * inv_rms * (jnp.float32(1.0) + w_f)
  return y.astype(orig_dtype) if cast_back else y


def multi_modality_rms_norm(
    x: Any,
    weight: Any,
    modality_lens: Sequence[int] | Any | None = None,
    eps: float = 1e-6,
    cast_back: bool = True,
    *,
    modality_ids: Any | None = None,
) -> Any:
  """Multi-modality Gemma-style ``(1 + w)`` RMSNorm matching ``MultiModalityRMSNorm``.

  Supports:
  - Single-modality weights of shape ``(D,)`` or ``(1, D)`` (ignores
    ``modality_lens`` / ``modality_ids``).
  - Multi-modality weights of shape ``(M, D)`` or raw checkpoint flat ``(M * D,)``
    (with ``M = 3`` for ``[VIDEO, AUDIO, TEXT]``), applied along axis 0 via:
    * ``modality_lens``: contiguous per-modality counts ``[N_v, N_a, N_t]`` along
      axis 0 (matching ``ModalityDispatcher``); static Python ints use zero-gather
      slicing, while traced JAX arrays fall back to dynamic boundary indexing.
    * ``modality_ids``: explicit integer modality index per token, shape ``(S,)``
      (or ``(B, S)`` when ``x`` is batched).
  - Both 2D ``(S, D)`` (pre_norm / mhc_norm) and 3D ``(S, H, head_dim)``
    (q_norm / k_norm, where affine ``(head_dim,)`` is shared across heads).
  """
  _require_jax("multi_modality_rms_norm")
  d = x.shape[-1]
  w = jnp.asarray(weight, dtype=jnp.float32)
  if w.ndim == 1:
    if w.shape[0] == d:
      return rms_norm_1pw(x, w, eps=eps, cast_back=cast_back)
    if w.shape[0] % d == 0:
      w = w.reshape(w.shape[0] // d, d)
    else:
      raise ValueError(
          f"Incompatible RMSNorm weight shape {weight.shape} for hidden dim {d}."
      )
  elif w.ndim == 2:
    if w.shape[1] != d:
      raise ValueError(
          f"Incompatible RMSNorm weight shape {weight.shape} for hidden dim {d}."
      )
    if w.shape[0] == 1:
      return rms_norm_1pw(x, w[0], eps=eps, cast_back=cast_back)
  else:
    raise ValueError(f"Expected 1D or 2D RMSNorm weight, got shape {weight.shape}.")

  num_modalities = w.shape[0]
  orig_dtype = x.dtype
  x_f = x.astype(jnp.float32)
  t = x_f * jax.lax.rsqrt(
      jnp.mean(x_f * x_f, axis=-1, keepdims=True) + jnp.float32(eps)
  )

  if modality_ids is not None:
    ids = jnp.asarray(modality_ids, dtype=jnp.int32)
    w_sel = w[ids]  # (*ids.shape, D)
    extra_dims = x.ndim - ids.ndim - 1
    if extra_dims < 0:
      raise ValueError(
          f"modality_ids shape {ids.shape} has too many dims for x shape {x.shape}."
      )
    w_sel = w_sel.reshape(ids.shape + (1,) * extra_dims + (d,))
    y = t * (jnp.float32(1.0) + w_sel)
    return y.astype(orig_dtype) if cast_back else y

  if modality_lens is None:
    raise ValueError(
        "Either modality_lens or modality_ids must be provided when weight has "
        f"multiple modalities (shape={w.shape})."
    )

  # Static Python/NumPy integer lengths: exact zero-gather slice + concat.
  is_static = isinstance(modality_lens, (tuple, list, np.ndarray)) and all(
      isinstance(n, (int, np.integer)) for n in modality_lens
  )
  if is_static:
    lens = [int(n) for n in modality_lens]
    if len(lens) != num_modalities:
      raise ValueError(
          f"len(modality_lens)={len(lens)} does not match weight modalities={num_modalities}."
      )
    if sum(lens) != x.shape[0]:
      raise ValueError(
          f"sum(modality_lens)={sum(lens)} does not match x.shape[0]={x.shape[0]}."
      )
    if x.shape[0] == 0:
      return t.astype(orig_dtype) if cast_back else t
    bcast_shape = (1,) * (x.ndim - 1) + (d,)
    chunks = []
    offset = 0
    for m, n_m in enumerate(lens):
      if n_m > 0:
        w_m = w[m].reshape(bcast_shape)
        chunks.append(t[offset : offset + n_m] * (jnp.float32(1.0) + w_m))
        offset += n_m
    y = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)
    return y.astype(orig_dtype) if cast_back else y

  # Dynamic / traced modality_lens along axis 0: compute token -> modality index.
  lens_arr = jnp.asarray(modality_lens, dtype=jnp.int32)
  if lens_arr.shape != (num_modalities,):
    raise ValueError(
        f"modality_lens shape {lens_arr.shape} does not match weight modalities={num_modalities}."
    )
  bounds = jnp.cumsum(lens_arr)
  idx = jnp.arange(x.shape[0], dtype=jnp.int32)
  dyn_ids = jnp.sum(idx[:, None] >= bounds[None, :-1], axis=-1)
  w_sel = w[dyn_ids].reshape((x.shape[0],) + (1,) * (x.ndim - 2) + (d,))
  y = t * (jnp.float32(1.0) + w_sel)
  return y.astype(orig_dtype) if cast_back else y


# ---------------------------------------------------------------------------
# 4-6. Manifold-Constrained Hyper-Connections (mHC) & Sinkhorn-Knopp
#      (magi2_preview.py:L1586-2057, L2976-3108)
# ---------------------------------------------------------------------------


def sinkhorn_knopp(
    H: Any,
    iters: int = 20,
    eps: float = 1e-12,
    *,
    out_dtype: Any = None,
) -> Any:
  """Doubly-stochastic Sinkhorn-Knopp projection matching ``magi2_preview._sinkhorn_knopp``.

  Runs in float32, normalizing **columns first (``axis=-2``), then rows
  (``axis=-1``)** for ``iters`` iterations, and casts to ``out_dtype``
  (default ``jnp.bfloat16``).
  """
  _require_jax("sinkhorn_knopp")
  dtype = jnp.bfloat16 if out_dtype is None else out_dtype
  h_f = jnp.asarray(H, dtype=jnp.float32)
  eps_f = jnp.float32(eps)
  h_max = jnp.max(h_f, axis=(-2, -1), keepdims=True)
  m = jnp.exp(h_f - h_max)
  for _ in range(iters):
    m = m / (jnp.sum(m, axis=-2, keepdims=True) + eps_f)
    m = m / (jnp.sum(m, axis=-1, keepdims=True) + eps_f)
  return m.astype(dtype)


def mhc_pre(
    s_flat: Any,
    norm_w: Any,
    phi_fused: Any,
    alpha_pre: Any,
    bias_pre: Any,
    modality_lens: Sequence[int] | Any | None = None,
    eps: float = 1e-6,
    *,
    modality_ids: Any | None = None,
) -> tuple[Any, Any, Any]:
  """mHC pre-sublayer projection and stream collapse.

  Matches ``MHCHandler.apply_norm_and_compute_h_`` + ``compute_and_apply_hpre``
  (``magi2_preview.py:L1966-1998, L3044-3092``). Note that within each
  ``TransformerLayer``, ``mhc_norm`` (``norm_w``) is shared across both the
  attention and MLP sub-blocks, whereas ``phi_fused``, ``alpha_*``, and
  ``bias_*`` are separate per sub-block.

  Args:
    s_flat: Multi-stream residual state of shape ``(S, n * D)`` (or ``(S, n, D)``),
      typically bfloat16 (``n=4, D=3072 -> n*D=12288``).
    norm_w: ``mhc_norm.weight``, shape ``(3, n*D)``, ``(3*n*D,)``, or ``(n*D,)`` fp32.
    phi_fused: ``mhc_phi_fused_{attn,mlp}``, shape ``(n*D, 2*n + n*n)`` fp32.
    alpha_pre: ``mhc_alpha_pre_{attn,mlp}``, scalar or shape ``(1,)`` fp32.
    bias_pre: ``mhc_bias_pre_{attn,mlp}``, shape ``(n,)`` fp32.
    modality_lens: Contiguous per-modality lengths ``[N_v, N_a, N_t]`` along axis 0.
    eps: RMSNorm epsilon (default ``1e-6``).
    modality_ids: Optional per-token modality ids if tokens are not modality-grouped.

  Returns:
    ``(x_in, h_post_raw, h_res_raw)`` where:
      - ``x_in``: collapsed sublayer input of shape ``(S, D)``, dtype ``s_flat.dtype`` (bf16).
      - ``h_post_raw``: pre-affine post-gate logits of shape ``(S, n)``, ``float32``.
      - ``h_res_raw``: pre-affine residual mixing logits of shape ``(S, n, n)``, ``float32``.
  """
  _require_jax("mhc_pre")
  if s_flat.ndim == 3:
    seq_len, n, d = s_flat.shape
    s_flat_2d = s_flat.reshape(seq_len, n * d)
    s_streams = s_flat
  elif s_flat.ndim == 2:
    seq_len, flat_dim = s_flat.shape
    n = int(bias_pre.shape[0])
    if flat_dim % n != 0:
      raise ValueError(f"s_flat width {flat_dim} is not divisible by num_streams {n}.")
    d = flat_dim // n
    s_flat_2d = s_flat
    s_streams = s_flat.reshape(seq_len, n, d)
  else:
    raise ValueError(f"Expected 2D or 3D s_flat, got shape {s_flat.shape}.")

  state_dtype = s_flat.dtype
  matmul_scale = jnp.float32(1.0 / math.sqrt(float(n * d)))

  s_norm = multi_modality_rms_norm(
      s_flat_2d,
      norm_w,
      modality_lens=modality_lens,
      eps=eps,
      cast_back=False,
      modality_ids=modality_ids,
  )  # float32 (S, n*D)

  h_fused = jnp.matmul(
      s_norm,
      jnp.asarray(phi_fused, dtype=jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )  # float32 (S, 2*n + n*n)

  h_pre_raw = h_fused[:, :n]
  h_post_raw = h_fused[:, n : 2 * n]
  h_res_raw = h_fused[:, 2 * n :].reshape(seq_len, n, n)

  alpha_pre_f = jnp.asarray(alpha_pre, dtype=jnp.float32).reshape(())
  bias_pre_f = jnp.asarray(bias_pre, dtype=jnp.float32)
  h_pre = jax.nn.sigmoid(
      alpha_pre_f * matmul_scale * h_pre_raw + bias_pre_f[None, :]
  ).astype(state_dtype)

  # Match Triton _apply_hpre_fwd_kernel: accumulate in fp32 after bf16 h_pre cast.
  x_in = jnp.sum(
      h_pre.astype(jnp.float32)[:, :, None] * s_streams.astype(jnp.float32),
      axis=1,
  ).astype(state_dtype)
  return x_in, h_post_raw, h_res_raw


def mhc_post(
    s_streams: Any,
    sub_out: Any,
    h_post_raw: Any,
    h_res_raw: Any,
    alpha_post: Any,
    bias_post: Any,
    alpha_res: Any,
    bias_res: Any,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-12,
) -> Any:
  """mHC post-sublayer doubly-stochastic stream mixing and residual update.

  Matches ``MHCHandler.compute_hpost_and_hres`` + ``apply_hpost_and_hres``
  (``magi2_preview.py:L1999-2039, L3070-3108``).

  Args:
    s_streams: Previous multi-stream state, shape ``(S, n, D)`` (bf16).
    sub_out: Sub-layer output (attention or MLP), shape ``(S, D)`` (bf16).
    h_post_raw: Pre-affine post-gate logits from ``mhc_pre``, shape ``(S, n)`` fp32.
    h_res_raw: Pre-affine residual logits from ``mhc_pre``, shape ``(S, n, n)`` fp32.
    alpha_post: ``mhc_alpha_post_{attn,mlp}``, scalar or ``(1,)`` fp32.
    bias_post: ``mhc_bias_post_{attn,mlp}``, shape ``(n,)`` fp32.
    alpha_res: ``mhc_alpha_res_{attn,mlp}``, scalar or ``(1,)`` fp32.
    bias_res: ``mhc_bias_res_{attn,mlp}``, shape ``(n, n)`` fp32.
    sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default ``20``).
    sinkhorn_eps: Sinkhorn-Knopp denominator epsilon (default ``1e-12``).

  Returns:
    ``s_next_streams`` of shape ``(S, n, D)`` and dtype ``s_streams.dtype``.
  """
  _require_jax("mhc_post")
  if s_streams.ndim != 3:
    raise ValueError(f"Expected 3D s_streams (S, n, D), got shape {s_streams.shape}.")
  _, n, d = s_streams.shape
  state_dtype = s_streams.dtype
  matmul_scale = jnp.float32(1.0 / math.sqrt(float(n * d)))

  alpha_post_f = jnp.asarray(alpha_post, dtype=jnp.float32).reshape(())
  bias_post_f = jnp.asarray(bias_post, dtype=jnp.float32)
  h_post = (
      jnp.float32(2.0)
      * jax.nn.sigmoid(
          alpha_post_f * matmul_scale * jnp.asarray(h_post_raw, dtype=jnp.float32)
          + bias_post_f[None, :]
      )
  ).astype(state_dtype)

  alpha_res_f = jnp.asarray(alpha_res, dtype=jnp.float32).reshape(())
  bias_res_f = jnp.asarray(bias_res, dtype=jnp.float32)
  h_res_logits = (
      alpha_res_f * matmul_scale * jnp.asarray(h_res_raw, dtype=jnp.float32)
      + bias_res_f[None, :, :]
  )
  h_res = sinkhorn_knopp(
      h_res_logits,
      iters=sinkhorn_iters,
      eps=sinkhorn_eps,
      out_dtype=state_dtype,
  )

  # Match Triton _hyper_connect_fwd_kernel: bf16 h_res/h_post cast to fp32 and accumulated in fp32.
  mixed_res = jnp.matmul(
      h_res.astype(jnp.float32),
      s_streams.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )  # (S, n, D)
  out_mstream = (
      h_post.astype(jnp.float32)[:, :, None]
      * sub_out.astype(jnp.float32)[:, None, :]
  )  # (S, n, D)
  return (mixed_res + out_mstream).astype(state_dtype)


# ---------------------------------------------------------------------------
# 6b. Tokamax Fused Pallas TPU 4-Stream mHC (3.7x faster on TPU7x-8)
#     Calls Tokamax's Pallas memory-bound kernels directly (mhc_pre_mixes_collapse,
#     fused_post_pre_mixes, mhc_post_2d) with MAGI-2's EXACT global-max
#     col-then-row Sinkhorn-Knopp epilogue (avoiding Tokamax's per-row softmax
#     divergence when real MAGI-2 h_res_logits spread > 28, e.g. L7 spread=94!).
# ---------------------------------------------------------------------------

_TOKAMAX_MHC_KERNELS: Any = None
_TOKAMAX_MHC_CHECKED = False


def _get_tokamax_mhc_kernels() -> Any:
  global _TOKAMAX_MHC_KERNELS, _TOKAMAX_MHC_CHECKED
  if not _TOKAMAX_MHC_CHECKED:
    _TOKAMAX_MHC_CHECKED = True
    try:
      from tokamax._src.ops.mhc import fused_post_pre_kernel as _fpp  # type: ignore[import-not-found]
      from tokamax._src.ops.mhc import post_kernel as _post  # type: ignore[import-not-found]
      from tokamax._src.ops.mhc import pre_kernel as _pre  # type: ignore[import-not-found]
      _TOKAMAX_MHC_KERNELS = (_pre, _post, _fpp)
    except Exception:
      # Fallback: load standalone Pallas mHC kernel files bypassing tokamax/__init__.py
      # (which eagerly pulls in autotuning -> qwix -> flax and may fail on newer JAX).
      try:
        import sys
        import types
        from pathlib import Path

        mhc_dir = None
        for entry in sys.path:
          cand = Path(entry) / "tokamax" / "_src" / "ops" / "mhc"
          if (cand / "pre_kernel.py").is_file():
            mhc_dir = cand
            tok_root = Path(entry)
            break
        if mhc_dir is not None:
          for pkg, rel in (
              ("tokamax", "tokamax"),
              ("tokamax._src", "tokamax/_src"),
              ("tokamax._src.ops", "tokamax/_src/ops"),
              ("tokamax._src.ops.mhc", "tokamax/_src/ops/mhc"),
          ):
            ns = types.ModuleType(pkg)
            ns.__path__ = [str(tok_root / rel)]
            sys.modules[pkg] = ns
          from tokamax._src.ops.mhc import fused_post_pre_kernel as _fpp  # type: ignore[import-not-found]
          from tokamax._src.ops.mhc import post_kernel as _post  # type: ignore[import-not-found]
          from tokamax._src.ops.mhc import pre_kernel as _pre  # type: ignore[import-not-found]
          _TOKAMAX_MHC_KERNELS = (_pre, _post, _fpp)
        else:
          _TOKAMAX_MHC_KERNELS = None
      except Exception:
        _TOKAMAX_MHC_KERNELS = None
  return _TOKAMAX_MHC_KERNELS


def use_tokamax_mhc() -> bool:
  """Return True when Tokamax fused Pallas mHC should be used on TPU."""
  import os
  mode = os.environ.get("DT_MAGI2_MHC_BACKEND", "auto").strip().lower()
  if mode in ("jax", "ref", "off", "0", "false"):
    return False
  kerns = _get_tokamax_mhc_kernels()
  if kerns is None:
    if mode in ("tokamax", "pallas", "1", "true"):
      raise RuntimeError("DT_MAGI2_MHC_BACKEND=tokamax requested but tokamax is not importable")
    return False
  if jax is None:
    return False
  if os.environ.get("DT_FORCE_TPU_TRACE", "").strip() == "1":
    return True
  try:
    return any(d.platform == "tpu" for d in jax.devices())
  except Exception:
    return False


def mhc_pack_tokamax_params(
    norm_weight: Any,   # (n*D,) f32 — single-modality only
    phi_fused: Any,     # (n*D, 2n + n^2) f32
    alpha_pre: Any,     # (1,) or () f32
    bias_pre: Any,      # (n,) f32
    alpha_post: Any,    # (1,) or () f32
    bias_post: Any,     # (n,) f32
    alpha_res: Any,     # (1,) or () f32
    bias_res: Any,      # (n, n) f32
    *,
    n_stream: int = 4,
) -> tuple[Any, Any, Any]:
  """Fold MAGI-2 single-modality ``(1 + norm_weight)`` into ``fn`` (untransposed)."""
  flat_dim = phi_fused.shape[0]
  n = int(n_stream)
  nw = jnp.asarray(norm_weight, dtype=jnp.float32)
  phi = jnp.asarray(phi_fused, dtype=jnp.float32)
  # Fold (1 + nw) directly into phi; keep H_res untransposed so exact MAGI-2
  # global-max col-then-row Sinkhorn runs on raw H_res logits before transposing
  # comb_mix = H_res^T for Tokamax's sum_i comb[i, j] * s[i] post kernel.
  fn_tm = ((jnp.float32(1.0) + nw)[:, None] * phi).T  # (2n + n^2, flat_dim)
  inv_sqrt_c = jnp.float32(1.0 / math.sqrt(float(flat_dim)))
  hc_scale = (
      jnp.stack([
          jnp.asarray(alpha_pre, dtype=jnp.float32).reshape(()),
          jnp.asarray(alpha_post, dtype=jnp.float32).reshape(()),
          jnp.asarray(alpha_res, dtype=jnp.float32).reshape(()),
      ])
      * inv_sqrt_c
  )
  hc_base = jnp.concatenate([
      jnp.asarray(bias_pre, dtype=jnp.float32).reshape((n,)),
      jnp.asarray(bias_post, dtype=jnp.float32).reshape((n,)),
      jnp.asarray(bias_res, dtype=jnp.float32).reshape((n * n,)),
  ])
  return fn_tm, hc_scale, hc_base


def _magi2_mhc_gates_from_mixes(
    mixes: Any,        # (T, 2n + n^2) f32 raw s @ fn.T
    sqrsum: Any,       # (T, 1) f32 sum(s^2)
    hc_scale: Any,     # (3,) f32
    hc_base: Any,      # (2n + n^2,) f32
    *,
    flat_dim: int,
    n_stream: int,
    eps: float,
    sinkhorn_iters: int,
    sinkhorn_eps: float,
    state_dtype: Any,
) -> tuple[Any, Any]:
  """Exact MAGI-2 post & global-max Sinkhorn gates returning ``(post_mix, comb_mix)``."""
  t_len = mixes.shape[0]
  n = int(n_stream)
  scaled = mixes * jax.lax.rsqrt(
      sqrsum / jnp.float32(float(flat_dim)) + jnp.float32(eps)
  )
  post_logits = scaled[:, n : 2 * n] * hc_scale[1] + hc_base[None, n : 2 * n]
  # Match MAGI-2 Triton bf16 rounding of h_post and h_res before fp32 accumulation.
  post_mix = (
      (jnp.float32(2.0) * jax.nn.sigmoid(post_logits))
      .astype(state_dtype)
      .astype(jnp.float32)
  )
  h_res_logits = (
      scaled[:, 2 * n :].reshape(t_len, n, n) * hc_scale[2]
      + hc_base[2 * n :].reshape(1, n, n)
  )
  h_res = sinkhorn_knopp(
      h_res_logits,
      iters=int(sinkhorn_iters),
      eps=float(sinkhorn_eps),
      out_dtype=state_dtype,
  )
  # Tokamax post kernel computes out[j] = sum_i comb[i, j] * s[i] = (comb^T @ s)[j],
  # whereas MAGI-2 computes (h_res @ s)[j], so comb_mix = h_res^T.
  comb_mix = jnp.swapaxes(h_res, -2, -1).astype(jnp.float32)
  return post_mix, comb_mix


def mhc_pre_tokamax(
    s_flat: Any,        # (S, n*D) bf16
    norm_weight: Any,
    phi_fused: Any,
    alpha_pre: Any,
    bias_pre: Any,
    alpha_post: Any,
    bias_post: Any,
    alpha_res: Any,
    bias_res: Any,
    *,
    n_stream: int = 4,
    eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-12,
) -> tuple[Any, Any, Any]:
  """Tokamax Pallas ``mhc_pre_mixes_collapse`` + exact MAGI-2 global-max Sinkhorn."""
  pre_k, _, _ = _get_tokamax_mhc_kernels()
  fn_tm, hc_scale, hc_base = mhc_pack_tokamax_params(
      norm_weight, phi_fused, alpha_pre, bias_pre, alpha_post, bias_post,
      alpha_res, bias_res, n_stream=n_stream,
  )
  mixes, sqrsum, x_in = pre_k.mhc_pre_mixes_collapse(
      s_flat.astype(jnp.bfloat16),
      fn_tm,
      hc_scale,
      hc_base,
      float(eps),
      0.0,
      token_block_size=64,
  )
  post_mix, comb_mix = _magi2_mhc_gates_from_mixes(
      mixes,
      sqrsum,
      hc_scale,
      hc_base,
      flat_dim=s_flat.shape[-1],
      n_stream=n_stream,
      eps=eps,
      sinkhorn_iters=sinkhorn_iters,
      sinkhorn_eps=sinkhorn_eps,
      state_dtype=s_flat.dtype,
  )
  return x_in.astype(s_flat.dtype), post_mix, comb_mix


def mhc_post_tokamax(
    sub_out: Any,       # (S, D) bf16
    s_flat: Any,        # (S, n*D) bf16
    post_mix: Any,      # (S, n) f32
    comb_mix: Any,      # (S, n, n) f32 (already transposed H_res^T)
) -> Any:
  """Tokamax Pallas ``mhc_post_2d`` returning ``s_next_flat (S, n*D)``."""
  _, post_k, _ = _get_tokamax_mhc_kernels()
  out = post_k.mhc_post_2d(
      sub_out.astype(jnp.bfloat16),
      s_flat.astype(jnp.bfloat16),
      post_mix,
      comb_mix,
      token_block_size=64,
  )
  return out.astype(s_flat.dtype)


def mhc_fused_post_pre_tokamax(
    sub_out: Any,       # (S, D) bf16 — output of prev sublayer (e.g. attn)
    s_flat: Any,        # (S, n*D) bf16 — residual before prev sublayer
    post_mix: Any,      # (S, n) f32 — from prev mhc_pre
    comb_mix: Any,      # (S, n, n) f32 — from prev mhc_pre (H_res^T)
    norm_weight: Any,   # next sublayer (mlp) params
    phi_fused: Any,
    alpha_pre: Any,
    bias_pre: Any,
    alpha_post: Any,
    bias_post: Any,
    alpha_res: Any,
    bias_res: Any,
    *,
    n_stream: int = 4,
    eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-12,
) -> tuple[Any, Any, Any, Any]:
  """1-pass fused Tokamax ``fused_post_pre_mixes`` + exact MAGI-2 global-max Sinkhorn."""
  _, _, fpp_k = _get_tokamax_mhc_kernels()
  fn_tm, hc_scale, hc_base = mhc_pack_tokamax_params(
      norm_weight, phi_fused, alpha_pre, bias_pre, alpha_post, bias_post,
      alpha_res, bias_res, n_stream=n_stream,
  )
  t_len = s_flat.shape[0]
  n = int(n_stream)
  s_mid, mixes, sqrsum, x_next_in = fpp_k.fused_post_pre_mixes(
      sub_out.astype(jnp.bfloat16),
      s_flat.astype(jnp.bfloat16),
      post_mix,
      comb_mix.reshape(t_len, n * n),
      fn_tm,
      hc_scale,
      hc_base,
      float(eps),
      0.0,
      token_block_size=32,
  )
  post_next, comb_next = _magi2_mhc_gates_from_mixes(
      mixes,
      sqrsum,
      hc_scale,
      hc_base,
      flat_dim=s_flat.shape[-1],
      n_stream=n_stream,
      eps=eps,
      sinkhorn_iters=sinkhorn_iters,
      sinkhorn_eps=sinkhorn_eps,
      state_dtype=s_flat.dtype,
  )
  return s_mid.astype(s_flat.dtype), x_next_in.astype(s_flat.dtype), post_next, comb_next


# ---------------------------------------------------------------------------
# 7. Timestep sinusoidal embedding (magi2_preview.py:L2075-2087)
# ---------------------------------------------------------------------------


def sinusoidal_embedding_1d(raw_t: Any, dim: int = 64) -> Any:
  """1D sinusoidal timestep embedding: **cos first, sin second**, ``pos = raw_t * 1000``.

  Matches ``magi2_preview.sinusoidal_embedding_1d(dim, position)`` (note argument
  order ``(raw_t, dim=64)`` here). Works on both NumPy arrays/scalars and JAX arrays,
  always returning ``float32`` of shape ``(*raw_t.shape, dim)``.
  """
  xp = jnp if _is_jax_array(raw_t) else np
  pos = xp.asarray(raw_t, dtype=xp.float32) * xp.float32(1000.0)
  half = dim // 2
  freqs = xp.exp(
      xp.float32(-math.log(10000.0))
      * xp.arange(half, dtype=xp.float32)
      / xp.float32(half)
  )
  args = pos[..., None] * freqs
  emb = xp.concatenate([xp.cos(args), xp.sin(args)], axis=-1)
  if dim % 2:
    emb = xp.concatenate([emb, xp.zeros_like(emb[..., :1])], axis=-1)
  return emb


# ---------------------------------------------------------------------------
# 8-11. 3D Partial RoPE, Coord Construction & Refiner Block Permutation
#       (magi2_preview.py:L701-834, L903-916, L1127-1157;
#        preview_data_proxy.py:L259-335; refiner_data_proxy.py:L212-254,
#        L360-436, L555-642)
# ---------------------------------------------------------------------------


def _get_coords_np(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
  """NumPy equivalent of ``get_coords`` returning ``(T*H*W, 9)`` float32."""
  ori_t, ori_h, ori_w = (int(shape[0]), int(shape[1]), int(shape[2]))
  ref_t, ref_h, ref_w = (
      int(ref_feat_shape[0]),
      int(ref_feat_shape[1]),
      int(ref_feat_shape[2]),
  )
  n_tokens = ori_t * ori_h * ori_w
  if n_tokens == 0:
    return np.empty((0, 9), dtype=np.float32)

  offset_t, offset_h, offset_w = offset_thw
  time_rng = np.arange(ori_t, dtype=np.float32) + np.float32(offset_t)
  height_rng = np.arange(ori_h, dtype=np.float32) + np.float32(offset_h)
  width_rng = np.arange(ori_w, dtype=np.float32) + np.float32(offset_w)

  time_grid, height_grid, width_grid = np.meshgrid(
      time_rng, height_rng, width_rng, indexing="ij"
  )
  coords_flat = np.stack([time_grid, height_grid, width_grid], axis=-1).reshape(-1, 3)
  meta = np.broadcast_to(
      np.array(
          [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], dtype=np.float32
      ),
      (n_tokens, 6),
  )
  return np.concatenate([coords_flat, meta], axis=-1).astype(np.float32, copy=False)


def elementwise_fourier_embed(
    coords_9: np.ndarray,
    head_dim: int = 128,
    temperature: float = 10000.0,
) -> np.ndarray:
  """Exact NumPy port of ``ElementWiseFourierEmbed(dim=head_dim, in_pixels=False)``.

  Args:
    coords_9: Array of shape ``(L, 9)`` with columns
      ``(t, h, w, T, H, W, ref_T, ref_H, ref_W)``.
    head_dim: Attention head dimension (default ``128`` -> ``16`` bands/axis -> ``96`` dims).
    temperature: Inverse-frequency base temperature (default ``10000.0``).

  Returns:
    ``np.ndarray`` of shape ``(L, 6 * (head_dim // 8))`` (``[L, 96]`` for
    ``head_dim=128``) in ``float32``, packed as
    ``[sin_t(B), sin_h(B), sin_w(B), cos_t(B), cos_h(B), cos_w(B)]``.
  """
  coords = np.asarray(coords_9, dtype=np.float32)
  if coords.ndim != 2 or coords.shape[1] != 9:
    raise ValueError(f"Expected coords_9 of shape (L, 9), got {coords.shape}.")

  num_bands = head_dim // 8
  if coords.shape[0] == 0:
    return np.empty((0, 6 * num_bands), dtype=np.float32)

  exp = np.arange(num_bands, dtype=np.float32) / np.float32(num_bands)
  bands = np.float32(1.0) / (np.float32(temperature) ** exp)  # (B,)

  coords_xyz = coords[:, :3]
  sizes = coords[:, 3:6]
  refs = coords[:, 6:9]

  both_one = (refs == np.float32(1.0)) & (sizes == np.float32(1.0))
  with np.errstate(divide="ignore", invalid="ignore"):
    scales = (refs - np.float32(1.0)) / (sizes - np.float32(1.0))
  scales = np.where(both_one, np.float32(1.0), scales)
  if np.isnan(scales).any() or np.isinf(scales).any():
    raise ValueError("RoPE coordinate scales contain NaN or Inf.")

  # Spatial centering ONLY for (h, w); time dimension is NOT centered.
  centers = (sizes - np.float32(1.0)) / np.float32(2.0)
  centers[:, 0] = np.float32(0.0)
  coords_centered = coords_xyz - centers

  proj = coords_centered[:, :, None] * scales[:, :, None] * bands[None, None, :]
  sin_proj = np.sin(proj)  # (L, 3, B)
  cos_proj = np.cos(proj)  # (L, 3, B)
  return np.concatenate([sin_proj, cos_proj], axis=1).reshape(coords.shape[0], -1)


def apply_3d_partial_rope(x: Any, rope_96: Any) -> Any:
  """Apply split-half non-interleaved partial 3D RoPE to channels ``[0:96]``.

  Leaves trailing channels ``[96:128]`` untouched. Computes the rotation in
  float32 and casts back to ``x.dtype`` (preserving float32 in Preview after
  float32 QK-norm, and bfloat16 in Refiner after bfloat16 QK-norm).

  Args:
    x: Query or key tensor of shape ``(S, H, D)`` or ``(B, S, H, D)`` (``D=128``).
    rope_96: Packed ``[sin(48), cos(48)]`` array of shape ``(S, 96)`` or ``(B, S, 96)``.
  """
  use_jax = _is_jax_array(x) or _is_jax_array(rope_96)
  xp = jnp if use_jax else np
  orig_dtype = x.dtype

  rope = xp.asarray(rope_96, dtype=xp.float32)
  half_ro = rope.shape[-1] // 2
  ro_dim = half_ro * 2
  if ro_dim > x.shape[-1]:
    raise ValueError(
        f"Rotary dim {ro_dim} exceeds head dim {x.shape[-1]}."
    )

  # Insert singleton head axis at -2 so (S, 48) broadcasts over both (S, H, 48)
  # and (B, S, H, 48), and (B, S, 48) broadcasts over (B, S, H, 48).
  sin_b = rope[..., :half_ro][..., None, :]
  cos_b = rope[..., half_ro:][..., None, :]

  x_f = x.astype(xp.float32)
  x1 = x_f[..., :half_ro]
  x2 = x_f[..., half_ro:ro_dim]
  out1 = x1 * cos_b - x2 * sin_b
  out2 = x2 * cos_b + x1 * sin_b

  if ro_dim == x.shape[-1]:
    out = xp.concatenate([out1, out2], axis=-1)
    return out.astype(orig_dtype)
  out_rot = xp.concatenate([out1, out2], axis=-1).astype(orig_dtype)
  return xp.concatenate([out_rot, x[..., ro_dim:]], axis=-1)


def build_preview_rope_coords(
    video_thw: tuple[int, int, int],
    audio_len: int,
    txt_len: int,
    ref_img_hws: Sequence[tuple[int, int]] = (),
    *,
    spatial_rope_interpolation: str = "extra",
) -> np.ndarray:
  """Build Stage 1 Preview RoPE coordinate table ``[L, 9]`` in sequence order.

  Exact port of ``preview_data_proxy.SingleData.coords_mapping`` with
  ``patch_size=1, t_patch_size=1, add_time_token=False``:
  - Video: ``(T, H, W)`` with ``ref=(T, H, W)`` (``"extra"``) or ``(T, 32, 32)`` (``"inter"``).
  - Audio: ``(audio_len, 1, 1)`` with ``ref=((audio_len - 1)//8 + 1, 1, 1)``
    (maps ``0..249 -> 0..31`` for 10s audio).
  - Text: ``(txt_len, 1, 1)`` with ``ref=(1, 1, 1), offset=(-txt_len, 0, 0)`` ->
    produces exact ``(0, 0, 0)`` identity RoPE for all ``txt_len > 1``.
  - Ref image ``k`` of patch grid ``(h_k, w_k)`` at ``t_off = T + 2 + k``:
    1) Leading special token (1 row, TEXT): ``[t_off, -1, -1, 1, h_k, w_k, 1, h_k, w_k]``
       -> ``h_eff = -(h_k + 1)/2, w_eff = -(w_k + 1)/2``.
    2) Image patches (``h_k * w_k`` rows, VIDEO): ``(1, h_k, w_k)`` at ``offset=(t_off, 0, 0)``.
  """
  t_steps, h_steps, w_steps = (int(video_thw[0]), int(video_thw[1]), int(video_thw[2]))
  if spatial_rope_interpolation == "inter":
    video_h_ref, video_w_ref = 32, 32
  else:
    video_h_ref, video_w_ref = h_steps, w_steps

  parts: list[np.ndarray] = [
      _get_coords_np((t_steps, h_steps, w_steps), (t_steps, video_h_ref, video_w_ref)),
  ]

  audio_len = int(audio_len)
  if audio_len > 0:
    magic_audio_ref_t = (audio_len - 1) // 8 + 1
    parts.append(_get_coords_np((audio_len, 1, 1), (magic_audio_ref_t, 1, 1)))

  txt_len = int(txt_len)
  if txt_len > 0:
    parts.append(
        _get_coords_np((txt_len, 1, 1), (1, 1, 1), offset_thw=(-txt_len, 0, 0))
    )

  for k, (h_i, w_i) in enumerate(ref_img_hws):
    h_i, w_i = int(h_i), int(w_i)
    t_off = t_steps + 2 + k
    parts.append(
        np.array([[t_off, -1, -1, 1, h_i, w_i, 1, h_i, w_i]], dtype=np.float32)
    )
    parts.append(
        _get_coords_np((1, h_i, w_i), (1, h_i, w_i), offset_thw=(t_off, 0, 0))
    )

  if not parts:
    return np.empty((0, 9), dtype=np.float32)
  return np.concatenate(parts, axis=0)


def build_refiner_rope_coords(
    video_thw: tuple[int, int, int],
    txt_len: int,
    *,
    audio_len: int = 0,
    spatial_rope_interpolation: str = "extra",
) -> np.ndarray:
  """Build Stage 2 Refiner RoPE coordinate table ``[L, 9]`` in raster+tail order.

  Exact port of ``refiner_data_proxy.SingleData.coords_mapping`` for shipping
  video-only refiner (``audio_len=0``, no ref audio/video). Permute the video
  prefix ``coords[:T*H*W]`` with ``refiner_block_perm`` to obtain block-ordered
  coordinates matching ``Magi2RefinerDataProxy.process_input``.
  """
  t_steps, h_steps, w_steps = (int(video_thw[0]), int(video_thw[1]), int(video_thw[2]))
  if spatial_rope_interpolation == "inter":
    video_h_ref, video_w_ref = 32, 32
  else:
    video_h_ref, video_w_ref = h_steps, w_steps

  parts: list[np.ndarray] = [
      _get_coords_np((t_steps, h_steps, w_steps), (t_steps, video_h_ref, video_w_ref)),
  ]
  audio_len = int(audio_len)
  if audio_len > 0:
    magic_audio_ref_t = (audio_len - 1) // 8 + 1
    parts.append(_get_coords_np((audio_len, 1, 1), (magic_audio_ref_t, 1, 1)))
  txt_len = int(txt_len)
  if txt_len > 0:
    parts.append(
        _get_coords_np((txt_len, 1, 1), (1, 1, 1), offset_thw=(-txt_len, 0, 0))
    )
  return np.concatenate(parts, axis=0)


def refiner_block_perm(
    video_thw: tuple[int, int, int],
    bs_t: int = 8,
    bs_h: int = 4,
    bs_w: int = 4,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], int]:
  """3D ``(bs_t, bs_h, bs_w)`` block-grid permutation matching ``Magi2RefinerDataProxy``.

  Exact port of ``refiner_data_proxy.Magi2RefinerDataProxy._build_block_order``
  (lines 587-642) for the video token prefix ``[0, T*H*W)``.

  Args:
    video_thw: Video latent grid ``(T, H, W)`` (e.g. ``(63, 68, 120)`` at 1080p).
    bs_t: Temporal block size (default ``8``).
    bs_h: Height block size (default ``4``).
    bs_w: Width block size (default ``4``).

  Returns:
    ``(perm, inv_perm, block_grid_thw, pad_t)`` where:
      - ``perm``: ``int32[T*H*W]`` gather indices such that ``x_block = x_raster[perm]``.
      - ``inv_perm``: ``int32[T*H*W]`` inverse indices such that ``x_block[inv_perm] == x_raster``.
      - ``block_grid_thw``: ``(ceil(T/bs_t), ceil(H/bs_h), ceil(W/bs_w))`` (``(8, 17, 30)`` for 1080p).
      - ``pad_t``: Number of padded temporal frames needed to make every block a full
        ``bs_t * bs_h * bs_w`` tile (``1`` for ``T=63, bs_t=8`` -> ``64``).
  """
  t_num, h_num, w_num = (int(video_thw[0]), int(video_thw[1]), int(video_thw[2]))
  bs_t, bs_h, bs_w = int(bs_t), int(bs_h), int(bs_w)
  video_token_num = t_num * h_num * w_num
  if video_token_num == 0:
    empty = np.empty((0,), dtype=np.int32)
    return empty, empty, (0, 0, 0), 0

  gt = math.ceil(t_num / bs_t)
  gh = math.ceil(h_num / bs_h)
  gw = math.ceil(w_num / bs_w)
  pad_t = gt * bs_t - t_num

  padded = np.full((gt * bs_t, gh * bs_h, gw * bs_w), -1, dtype=np.int32)
  padded[:t_num, :h_num, :w_num] = np.arange(video_token_num, dtype=np.int32).reshape(
      t_num, h_num, w_num
  )
  flat = (
      padded.reshape(gt, bs_t, gh, bs_h, gw, bs_w)
      .transpose(0, 2, 4, 1, 3, 5)
      .reshape(-1)
  )
  perm = np.ascontiguousarray(flat[flat >= 0], dtype=np.int32)
  inv_perm = np.empty_like(perm)
  inv_perm[perm] = np.arange(video_token_num, dtype=np.int32)
  return perm, inv_perm, (gt, gh, gw), pad_t


# ---------------------------------------------------------------------------
# 12. FlowUniPCMultistepScheduler (sampler.py:L456-1311)
# ---------------------------------------------------------------------------


@dataclass
class SchedulerOutput:
  """Minimal diffusers-compatible scheduler step output."""

  prev_sample: Any

  def __getitem__(self, key: int | str) -> Any:
    if key == 0 or key == "prev_sample":
      return self.prev_sample
    raise KeyError(key)


class FlowUniPCMultistepScheduler:
  """Exact pure-NumPy/JAX port of ``sampler.FlowUniPCMultistepScheduler``.

  Computes all scalar schedule / log-SNR / Vandermonde BH1/BH2 coefficients in
  host ``np.float32`` (matching PyTorch bit-for-bit, with zero device syncs) and
  performs array arithmetic natively on either ``np.ndarray`` or ``jax.Array``.
  """

  def __init__(
      self,
      num_train_timesteps: int = 1000,
      solver_order: int = 2,
      prediction_type: str = "flow_prediction",
      shift: float = 1.0,
      use_dynamic_shifting: bool = False,
      thresholding: bool = False,
      dynamic_thresholding_ratio: float = 0.995,
      sample_max_value: float = 1.0,
      predict_x0: bool = True,
      solver_type: str = "bh2",
      lower_order_final: bool = True,
      disable_corrector: Sequence[int] = (),
      timestep_spacing: str = "linspace",
      steps_offset: int = 0,
      final_sigmas_type: str | None = "zero",
      *,
      order: int | None = None,
  ) -> None:
    if order is not None:
      solver_order = int(order)
    if solver_type not in ("bh1", "bh2"):
      if solver_type in ("midpoint", "heun", "logrho"):
        solver_type = "bh2"
      else:
        raise NotImplementedError(
            f"{solver_type} is not implemented for {self.__class__.__name__}"
        )

    self.num_train_timesteps = int(num_train_timesteps)
    self.solver_order = int(solver_order)
    self.order = self.solver_order
    self.prediction_type = prediction_type
    self.shift = float(shift)
    self.use_dynamic_shifting = bool(use_dynamic_shifting)
    self.thresholding = bool(thresholding)
    self.dynamic_thresholding_ratio = float(dynamic_thresholding_ratio)
    self.sample_max_value = float(sample_max_value)
    self.predict_x0 = bool(predict_x0)
    self.solver_type = solver_type
    self.lower_order_final = bool(lower_order_final)
    self.disable_corrector = list(disable_corrector)
    self.timestep_spacing = timestep_spacing
    self.steps_offset = int(steps_offset)
    self.final_sigmas_type = final_sigmas_type

    self.config = SimpleNamespace(
        num_train_timesteps=self.num_train_timesteps,
        solver_order=self.solver_order,
        prediction_type=self.prediction_type,
        shift=self.shift,
        use_dynamic_shifting=self.use_dynamic_shifting,
        thresholding=self.thresholding,
        dynamic_thresholding_ratio=self.dynamic_thresholding_ratio,
        sample_max_value=self.sample_max_value,
        predict_x0=self.predict_x0,
        solver_type=self.solver_type,
        lower_order_final=self.lower_order_final,
        disable_corrector=self.disable_corrector,
        timestep_spacing=self.timestep_spacing,
        steps_offset=self.steps_offset,
        final_sigmas_type=self.final_sigmas_type,
    )

    self.num_inference_steps: int | None = None
    alphas = np.linspace(
        1.0, 1.0 / self.num_train_timesteps, self.num_train_timesteps
    )[::-1].copy()
    sigmas = (1.0 - alphas).astype(np.float32)

    if not self.use_dynamic_shifting:
      s = np.float32(self.shift)
      sigmas = (s * sigmas / (np.float32(1.0) + (s - np.float32(1.0)) * sigmas)).astype(
          np.float32
      )

    self.sigmas: np.ndarray = sigmas
    self.timesteps: np.ndarray = sigmas * np.float32(self.num_train_timesteps)

    self.model_outputs: list[Any] = [None] * self.solver_order
    self.timestep_list: list[Any] = [None] * self.solver_order
    self.lower_order_nums: int = 0
    self.last_sample: Any = None
    self.this_order: int = 1
    self._step_index: int | None = None
    self._begin_index: int | None = None

    self.sigma_min: float = float(self.sigmas[-1])
    self.sigma_max: float = float(self.sigmas[0])

  @property
  def step_index(self) -> int | None:
    return self._step_index

  @property
  def begin_index(self) -> int | None:
    return self._begin_index

  def set_begin_index(self, begin_index: int = 0) -> None:
    self._begin_index = int(begin_index)

  def set_timesteps(
      self,
      num_inference_steps: int | None = None,
      device: Any = None,
      sigmas: Sequence[float] | np.ndarray | None = None,
      mu: float | None = None,
      shift: float | None = None,
  ) -> None:
    del device  # Host-side schedule; arrays stay on caller's device.
    if self.use_dynamic_shifting and mu is None:
      raise ValueError(
          "You have to pass a value for `mu` when `use_dynamic_shifting` is True."
      )

    if sigmas is None:
      if num_inference_steps is None:
        raise ValueError("Either `num_inference_steps` or `sigmas` must be provided.")
      sigmas_arr = np.linspace(
          self.sigma_max, self.sigma_min, int(num_inference_steps) + 1
      ).copy()[:-1]
    else:
      sigmas_arr = np.asarray(sigmas, dtype=np.float64).copy()

    if self.use_dynamic_shifting:
      assert mu is not None
      sigmas_arr = self.time_shift(mu, 1.0, sigmas_arr)
    else:
      shift_val = self.shift if shift is None else float(shift)
      sigmas_arr = shift_val * sigmas_arr / (1.0 + (shift_val - 1.0) * sigmas_arr)

    if self.final_sigmas_type == "zero":
      sigma_last = 0.0
    elif self.final_sigmas_type == "sigma_min":
      sigma_last = self.sigma_min
    else:
      raise ValueError(
          f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {self.final_sigmas_type!r}."
      )

    # IMPORTANT: match sampler.py:L638-644 — compute int64 timesteps from float64
    # sigmas BEFORE casting sigmas to float32!
    timesteps = (sigmas_arr * self.num_train_timesteps).astype(np.int64)
    self.sigmas = np.concatenate([sigmas_arr, [sigma_last]]).astype(np.float32)
    self.timesteps = timesteps
    self.num_inference_steps = len(timesteps)

    self.model_outputs = [None] * self.solver_order
    self.timestep_list = [None] * self.solver_order
    self.lower_order_nums = 0
    self.last_sample = None
    self._step_index = None
    self._begin_index = None

  def _threshold_sample(self, sample: Any) -> Any:
    xp = jnp if _is_jax_array(sample) else np
    dtype = sample.dtype
    batch_size = sample.shape[0]
    flat = sample.astype(xp.float32).reshape(batch_size, -1)
    s = xp.quantile(xp.abs(flat), self.dynamic_thresholding_ratio, axis=1)
    s = xp.clip(s, 1.0, self.sample_max_value)[:, None]
    flat = xp.clip(flat, -s, s) / s
    return flat.reshape(sample.shape).astype(dtype)

  def _sigma_to_t(self, sigma: Any) -> Any:
    return sigma * self.num_train_timesteps

  @staticmethod
  def _sigma_to_alpha_sigma_t(sigma: np.float32) -> tuple[np.float32, np.float32]:
    s = np.float32(sigma)
    return np.float32(np.float32(1.0) - s), s

  @staticmethod
  def time_shift(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)

  def convert_model_output(
      self, model_output: Any, *args: Any, sample: Any = None, **kwargs: Any
  ) -> Any:
    if sample is None:
      if len(args) > 1:
        sample = args[1]
      else:
        raise ValueError("Missing `sample` as a required keyword argument.")
    if self._step_index is None:
      raise RuntimeError("`step_index` is not initialized; call `step` first.")

    sigma_t = np.float32(self.sigmas[self._step_index])
    if self.prediction_type != "flow_prediction":
      raise ValueError(
          f"Unsupported prediction_type={self.prediction_type!r}; expected 'flow_prediction'."
      )

    if self.predict_x0:
      x0_pred = sample - sigma_t * model_output
      if self.thresholding:
        x0_pred = self._threshold_sample(x0_pred)
      return x0_pred

    epsilon = sample - np.float32(np.float32(1.0) - sigma_t) * model_output
    if self.thresholding:
      x0_pred = self._threshold_sample(sample - sigma_t * model_output)
      epsilon = model_output + x0_pred
    return epsilon

  def multistep_uni_p_bh_update(
      self,
      model_output: Any,
      *args: Any,
      sample: Any = None,
      order: int | None = None,
      **kwargs: Any,
  ) -> Any:
    del model_output  # Unused when solver_p is None; kept for signature parity.
    if sample is None:
      if len(args) > 1:
        sample = args[1]
      else:
        raise ValueError("Missing `sample` as a required keyword argument.")
    if order is None:
      if len(args) > 2:
        order = int(args[2])
      else:
        raise ValueError("Missing `order` as a required keyword argument.")
    assert self._step_index is not None

    m0 = self.model_outputs[-1]
    x = sample
    x_dtype = x.dtype

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(self.sigmas[self._step_index + 1])
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(self.sigmas[self._step_index])

    with np.errstate(divide="ignore", invalid="ignore"):
      lambda_t = np.float32(np.log(alpha_t) - np.log(sigma_t))
      lambda_s0 = np.float32(np.log(alpha_s0) - np.log(sigma_s0))
      h = np.float32(lambda_t - lambda_s0)

      rks_list: list[np.float32] = []
      d1s: list[Any] = []
      for i in range(1, order):
        si = self._step_index - i
        mi = self.model_outputs[-(i + 1)]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        lambda_si = np.float32(np.log(alpha_si) - np.log(sigma_si))
        rk = np.float32((lambda_si - lambda_s0) / h)
        rks_list.append(rk)
        d1s.append((mi - m0) / rk)

      rks_list.append(np.float32(1.0))
      rks = np.array(rks_list, dtype=np.float32)

      hh = np.float32(-h) if self.predict_x0 else h
      h_phi_1 = np.float32(np.expm1(hh))
      h_phi_k = np.float32(h_phi_1 / hh - np.float32(1.0))

      if self.solver_type == "bh1":
        B_h = hh
      elif self.solver_type == "bh2":
        B_h = np.float32(np.expm1(hh))
      else:
        raise NotImplementedError(f"Unsupported solver_type={self.solver_type!r}")

      R_rows = []
      b_vals = []
      factorial_i = 1
      for i in range(1, order + 1):
        R_rows.append(np.power(rks, i - 1))
        b_vals.append(np.float32(h_phi_k * np.float32(factorial_i) / B_h))
        factorial_i *= i + 1
        h_phi_k = np.float32(h_phi_k / hh - np.float32(1.0 / factorial_i))

      R = np.stack(R_rows, axis=0).astype(np.float32)
      b = np.array(b_vals, dtype=np.float32)

    xp = jnp if _is_jax_array(x) else np
    if len(d1s) > 0:
      if order == 2:
        rhos_p = np.array([0.5], dtype=np.float32)
      else:
        rhos_p = np.linalg.solve(R[:-1, :-1], b[:-1]).astype(np.float32)
      pred_res = xp.asarray(rhos_p[0], dtype=x_dtype) * d1s[0]
      for k in range(1, len(d1s)):
        pred_res = pred_res + xp.asarray(rhos_p[k], dtype=x_dtype) * d1s[k]
    else:
      pred_res = None

    if self.predict_x0:
      x_t_ = np.float32(sigma_t / sigma_s0) * x - np.float32(alpha_t * h_phi_1) * m0
      if pred_res is not None:
        x_t = x_t_ - np.float32(alpha_t * B_h) * pred_res
      else:
        x_t = x_t_
    else:
      x_t_ = np.float32(alpha_t / alpha_s0) * x - np.float32(sigma_t * h_phi_1) * m0
      if pred_res is not None:
        x_t = x_t_ - np.float32(sigma_t * B_h) * pred_res
      else:
        x_t = x_t_

    return x_t.astype(x_dtype)

  def multistep_uni_c_bh_update(
      self,
      this_model_output: Any,
      *args: Any,
      last_sample: Any = None,
      this_sample: Any = None,
      order: int | None = None,
      **kwargs: Any,
  ) -> Any:
    del this_sample  # Unused in UniC formula; kept for signature parity.
    if last_sample is None:
      if len(args) > 1:
        last_sample = args[1]
      else:
        raise ValueError("Missing `last_sample` as a required keyword argument.")
    if order is None:
      if len(args) > 3:
        order = int(args[3])
      else:
        raise ValueError("Missing `order` as a required keyword argument.")
    assert self._step_index is not None

    m0 = self.model_outputs[-1]
    x = last_sample
    x_dtype = x.dtype
    model_t = this_model_output

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(self.sigmas[self._step_index])
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(self.sigmas[self._step_index - 1])

    with np.errstate(divide="ignore", invalid="ignore"):
      lambda_t = np.float32(np.log(alpha_t) - np.log(sigma_t))
      lambda_s0 = np.float32(np.log(alpha_s0) - np.log(sigma_s0))
      h = np.float32(lambda_t - lambda_s0)

      rks_list: list[np.float32] = []
      d1s: list[Any] = []
      for i in range(1, order):
        si = self._step_index - (i + 1)
        mi = self.model_outputs[-(i + 1)]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        lambda_si = np.float32(np.log(alpha_si) - np.log(sigma_si))
        rk = np.float32((lambda_si - lambda_s0) / h)
        rks_list.append(rk)
        d1s.append((mi - m0) / rk)

      rks_list.append(np.float32(1.0))
      rks = np.array(rks_list, dtype=np.float32)

      hh = np.float32(-h) if self.predict_x0 else h
      h_phi_1 = np.float32(np.expm1(hh))
      h_phi_k = np.float32(h_phi_1 / hh - np.float32(1.0))

      if self.solver_type == "bh1":
        B_h = hh
      elif self.solver_type == "bh2":
        B_h = np.float32(np.expm1(hh))
      else:
        raise NotImplementedError(f"Unsupported solver_type={self.solver_type!r}")

      R_rows = []
      b_vals = []
      factorial_i = 1
      for i in range(1, order + 1):
        R_rows.append(np.power(rks, i - 1))
        b_vals.append(np.float32(h_phi_k * np.float32(factorial_i) / B_h))
        factorial_i *= i + 1
        h_phi_k = np.float32(h_phi_k / hh - np.float32(1.0 / factorial_i))

      R = np.stack(R_rows, axis=0).astype(np.float32)
      b = np.array(b_vals, dtype=np.float32)

    if order == 1:
      rhos_c = np.array([0.5], dtype=np.float32)
    else:
      rhos_c = np.linalg.solve(R, b).astype(np.float32)

    xp = jnp if _is_jax_array(x) else np
    d1_t = model_t - m0
    if len(d1s) > 0:
      corr_res = xp.asarray(rhos_c[0], dtype=x_dtype) * d1s[0]
      for k in range(1, len(d1s)):
        corr_res = corr_res + xp.asarray(rhos_c[k], dtype=x_dtype) * d1s[k]
      corr_term = corr_res + xp.asarray(rhos_c[-1], dtype=x_dtype) * d1_t
    else:
      corr_term = xp.asarray(rhos_c[-1], dtype=x_dtype) * d1_t

    if self.predict_x0:
      x_t_ = np.float32(sigma_t / sigma_s0) * x - np.float32(alpha_t * h_phi_1) * m0
      x_t = x_t_ - np.float32(alpha_t * B_h) * corr_term
    else:
      x_t_ = np.float32(alpha_t / alpha_s0) * x - np.float32(sigma_t * h_phi_1) * m0
      x_t = x_t_ - np.float32(sigma_t * B_h) * corr_term

    return x_t.astype(x_dtype)

  def index_for_timestep(
      self, timestep: Any, schedule_timesteps: np.ndarray | None = None
  ) -> int:
    if schedule_timesteps is None:
      schedule_timesteps = self.timesteps
    t_scalar = np.asarray(timestep).reshape(-1)[0]
    indices = np.nonzero(schedule_timesteps == t_scalar)[0]
    pos = 1 if len(indices) > 1 else 0
    return int(indices[pos])

  def _init_step_index(self, timestep: Any) -> None:
    if self._begin_index is None:
      self._step_index = self.index_for_timestep(timestep)
    else:
      self._step_index = self._begin_index

  def step(
      self,
      model_output: Any,
      timestep: Any,
      sample: Any,
      return_dict: bool = True,
      generator: Any = None,
  ) -> SchedulerOutput | tuple[Any]:
    del generator
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is None; call `set_timesteps` before `step`."
      )
    if self._step_index is None:
      self._init_step_index(timestep)
    assert self._step_index is not None

    use_corrector = (
        self._step_index > 0
        and (self._step_index - 1) not in self.disable_corrector
        and self.last_sample is not None
    )

    model_output_convert = self.convert_model_output(model_output, sample=sample)
    if use_corrector:
      sample = self.multistep_uni_c_bh_update(
          this_model_output=model_output_convert,
          last_sample=self.last_sample,
          this_sample=sample,
          order=self.this_order,
      )

    for i in range(self.solver_order - 1):
      self.model_outputs[i] = self.model_outputs[i + 1]
      self.timestep_list[i] = self.timestep_list[i + 1]

    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep

    if self.lower_order_final:
      this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
    else:
      this_order = self.solver_order

    self.this_order = min(this_order, self.lower_order_nums + 1)
    assert self.this_order > 0

    self.last_sample = sample
    prev_sample = self.multistep_uni_p_bh_update(
        model_output=model_output,
        sample=sample,
        order=self.this_order,
    )

    if self.lower_order_nums < self.solver_order:
      self.lower_order_nums += 1
    self._step_index += 1

    if not return_dict:
      return (prev_sample,)
    return SchedulerOutput(prev_sample=prev_sample)

  def scale_model_input(self, sample: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return sample

  def add_noise(
      self,
      original_samples: Any,
      noise: Any,
      timesteps: Any,
  ) -> Any:
    t_arr = np.asarray(timesteps).reshape(-1)
    if self._begin_index is None:
      step_indices = [self.index_for_timestep(t) for t in t_arr]
    elif self._step_index is not None:
      step_indices = [self._step_index] * len(t_arr)
    else:
      step_indices = [self._begin_index] * len(t_arr)

    sigma = self.sigmas[step_indices].reshape((-1,) + (1,) * (original_samples.ndim - 1))
    alpha_t = np.float32(1.0) - sigma
    if _is_jax_array(original_samples):
      alpha_t = jnp.asarray(alpha_t, dtype=original_samples.dtype)
      sigma = jnp.asarray(sigma, dtype=original_samples.dtype)
    else:
      alpha_t = alpha_t.astype(original_samples.dtype)
      sigma = sigma.astype(original_samples.dtype)
    return alpha_t * original_samples + sigma * noise

  def __len__(self) -> int:
    return self.num_train_timesteps


# ---------------------------------------------------------------------------
# 13. ZeroSNRDDPMDiscretization & Refiner VP noise augmentation
#     (inference_engine.py:L220-222, L588-592, L804-873)
# ---------------------------------------------------------------------------


def zero_snr_ddpm_sigmas(
    n: int = 1000,
    *,
    linear_start: float = 0.00085,
    linear_end: float = 0.0120,
    num_timesteps: int = 1000,
    shift_scale: float = 1.0,
    keep_start: bool = False,
    post_shift: bool = False,
    do_append_zero: bool = False,
    flip: bool = True,
) -> np.ndarray:
  """Zero-Terminal-SNR DDPM sigma schedule matching ``ZeroSNRDDPMDiscretization``.

  Default arguments match ``inference_engine.py:L220-222``:
  ``ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)``,
  returning decreasing float32 sigmas with ``sigmas[0] ~= 0.999575``,
  ``sigmas[220] ~= 0.838910``, and ``sigmas[999] == 0.0``.
  """
  if keep_start and not post_shift:
    linear_start = linear_start / (shift_scale + (1.0 - shift_scale) * linear_start)

  betas = (
      np.linspace(
          linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float64
      )
      ** 2
  )
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  if not post_shift:
    alphas_cumprod = alphas_cumprod / (
        shift_scale + (1.0 - shift_scale) * alphas_cumprod
    )

  if n < num_timesteps:
    timesteps = np.linspace(num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
    alphas_cumprod = alphas_cumprod[timesteps]
  elif n > num_timesteps:
    raise ValueError(f"n must be <= {num_timesteps}, got {n}.")

  ac_sqrt = np.sqrt(alphas_cumprod.astype(np.float32))
  s0 = ac_sqrt[0].copy()
  st = ac_sqrt[-1].copy()
  ac_sqrt = (ac_sqrt - st) * (s0 / (s0 - st))
  if post_shift:
    sq = ac_sqrt * ac_sqrt
    ac_sqrt = np.sqrt(
        sq / (np.float32(shift_scale) + np.float32(1.0 - shift_scale) * sq)
    )

  sigmas = ac_sqrt[::-1].copy()
  if do_append_zero:
    sigmas = np.concatenate([sigmas, np.zeros((1,), dtype=np.float32)])
  if flip:
    sigmas = sigmas[::-1].copy()
  return sigmas.astype(np.float32, copy=False)


def refiner_noise_augment(
    z_up: Any,
    noise: Any,
    noise_value: int = 220,
    *,
    sigmas: np.ndarray | None = None,
) -> Any:
  """VP noise augmentation for Stage 2 Refiner input (``inference_engine.py:L588-592``).

  At ``noise_value == 0``, returns ``z_up`` unchanged. Otherwise returns:
      ``z_up * sigma + noise * sqrt(1 - sigma**2)``
  where ``sigma = sigmas[noise_value]`` (``0.838910`` at ``noise_value=220``,
  with ``sqrt(1 - sigma**2) = 0.544270``).
  """
  if noise_value == 0:
    return z_up
  if sigmas is None:
    sigmas = zero_snr_ddpm_sigmas(1000)
  sigma = np.float32(sigmas[int(noise_value)])
  noise_coeff = np.float32(np.sqrt(np.float32(1.0) - sigma * sigma))

  orig_dtype = z_up.dtype
  use_jax = _is_jax_array(z_up) or _is_jax_array(noise)
  xp = jnp if use_jax else np
  out = (
      xp.asarray(z_up, dtype=xp.float32) * xp.float32(sigma)
      + xp.asarray(noise, dtype=xp.float32) * xp.float32(noise_coeff)
  )
  return out.astype(orig_dtype)


# ---------------------------------------------------------------------------
# 14. Exact align_corners=True trilinear upsampling
#     (inference_engine.py:L582-587)
# ---------------------------------------------------------------------------


def _align_corners_1d_weights(
    in_size: int, out_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Exact PyTorch ``area_pixel_compute_scale(..., align_corners=True)`` 1D weights."""
  if out_size <= 0 or in_size <= 0:
    raise ValueError(f"Sizes must be positive, got in_size={in_size}, out_size={out_size}.")
  if out_size == 1 or in_size == 1:
    zeros_i = np.zeros((out_size,), dtype=np.int32)
    zeros_f = np.zeros((out_size,), dtype=np.float32)
    return zeros_i, zeros_i, zeros_f

  scale = float(in_size - 1) / float(out_size - 1)
  pos = np.arange(out_size, dtype=np.float64) * scale
  pos = np.clip(pos, 0.0, float(in_size - 1))
  idx0 = np.floor(pos).astype(np.int32)
  idx1 = np.minimum(idx0 + 1, in_size - 1)
  w1 = (pos - idx0).astype(np.float32)
  return idx0, idx1, w1


def trilinear_upsample_latents(
    z_cthw: Any,
    target_thw: tuple[int, int, int],
) -> Any:
  """Exact ``align_corners=True`` trilinear upsampling matching PyTorch F.interpolate.

  NOTE: ``jax.image.resize(..., method="linear")`` uses ``align_corners=False``
  and does NOT match PyTorch's ``align_corners=True``! This separable 3-pass
  (W -> H -> T) implementation matches ``torch.nn.functional.interpolate(
  z, size=target_thw, mode="trilinear", align_corners=True)`` to float32 machine
  precision on both ``jax.Array`` and ``np.ndarray``.

  Args:
    z_cthw: Video latent of shape ``(C, T, H, W)`` or ``(B, C, T, H, W)``
      (any array with ``ndim >= 3`` whose trailing 3 axes are ``(T, H, W)``).
    target_thw: Target ``(T_out, H_out, W_out)`` (e.g. ``(63, 68, 120)``).
  """
  if z_cthw.ndim < 3:
    raise ValueError(
        f"Expected at least 3D array (..., T, H, W), got shape {z_cthw.shape}."
    )
  out_t, out_h, out_w = (int(target_thw[0]), int(target_thw[1]), int(target_thw[2]))
  use_jax = _is_jax_array(z_cthw)
  xp = jnp if use_jax else np
  orig_dtype = z_cthw.dtype
  x = xp.asarray(z_cthw, dtype=xp.float32)

  # Contract W -> H -> T to match PyTorch UpSampleTrilinear3d reduction order.
  for neg_axis, out_size in ((-1, out_w), (-2, out_h), (-3, out_t)):
    axis = x.ndim + neg_axis
    in_size = int(x.shape[axis])
    if in_size == out_size:
      continue
    idx0, idx1, w1 = _align_corners_1d_weights(in_size, out_size)
    if use_jax:
      i0 = jnp.asarray(idx0, dtype=jnp.int32)
      i1 = jnp.asarray(idx1, dtype=jnp.int32)
      w1_arr = jnp.asarray(w1, dtype=jnp.float32)
    else:
      i0, i1, w1_arr = idx0, idx1, w1
    shape = [1] * x.ndim
    shape[axis] = out_size
    w1_b = w1_arr.reshape(shape)
    w0_b = xp.float32(1.0) - w1_b
    v0 = xp.take(x, i0, axis=axis)
    v1 = xp.take(x, i1, axis=axis)
    x = w0_b * v0 + w1_b * v1

  return x.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Self-test suite (NumPy / JAX, plus optional PyTorch reference parity check)
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
  rng = np.random.default_rng(42)
  has_jax = jax is not None and jnp is not None
  try:
    import torch
    import torch.nn.functional as F

    has_torch = True
  except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    has_torch = False

  print(f"[self-test] jax={has_jax}, torch={has_torch}")

  # 1. zero_snr_ddpm_sigmas & refiner_noise_augment
  sigmas = zero_snr_ddpm_sigmas(1000)
  assert sigmas.shape == (1000,) and sigmas.dtype == np.float32
  assert abs(float(sigmas[0]) - 0.999575) < 1e-6, sigmas[0]
  assert abs(float(sigmas[220]) - 0.838910) < 1e-6, sigmas[220]
  assert abs(math.sqrt(1.0 - float(sigmas[220]) ** 2) - 0.544270) < 1e-6
  assert float(sigmas[999]) == 0.0
  z_small = rng.standard_normal((4, 6, 8, 10)).astype(np.float32)
  eps_small = rng.standard_normal((4, 6, 8, 10)).astype(np.float32)
  assert np.array_equal(refiner_noise_augment(z_small, eps_small, noise_value=0), z_small)
  z_aug = refiner_noise_augment(z_small, eps_small, noise_value=220)
  s220 = sigmas[220]
  np.testing.assert_allclose(
      z_aug, z_small * s220 + eps_small * np.sqrt(1.0 - s220 * s220), rtol=1e-6, atol=1e-6
  )
  print("  [PASS] zero_snr_ddpm_sigmas & refiner_noise_augment")

  # 2. sinusoidal_embedding_1d
  emb0 = sinusoidal_embedding_1d(np.array([0.0], dtype=np.float32), dim=64)
  assert emb0.shape == (1, 64) and emb0.dtype == np.float32
  np.testing.assert_allclose(emb0[0, :32], np.ones(32), atol=1e-7)
  np.testing.assert_allclose(emb0[0, 32:], np.zeros(32), atol=1e-7)
  raw_t = np.array([0.0, 0.25, 0.999], dtype=np.float32)
  emb_t = sinusoidal_embedding_1d(raw_t, dim=64)
  half = 32
  freqs_ref = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
  args_ref = (raw_t * 1000.0)[:, None] * freqs_ref[None, :]
  emb_ref = np.concatenate([np.cos(args_ref), np.sin(args_ref)], axis=-1)
  np.testing.assert_allclose(emb_t, emb_ref, rtol=1e-6, atol=1e-6)
  emb_odd = sinusoidal_embedding_1d(raw_t, dim=65)
  assert emb_odd.shape == (3, 65) and np.all(emb_odd[:, -1] == 0.0)
  print("  [PASS] sinusoidal_embedding_1d")

  # 3. Preview & Refiner RoPE coords + ElementWiseFourierEmbed + apply_3d_partial_rope
  coords_prev = build_preview_rope_coords(
      video_thw=(32, 4, 6), audio_len=250, txt_len=17, ref_img_hws=((4, 6), (2, 3))
  )
  expected_len = 32 * 4 * 6 + 250 + 17 + (1 + 24) + (1 + 6)
  assert coords_prev.shape == (expected_len, 9) and coords_prev.dtype == np.float32
  rope_prev = elementwise_fourier_embed(coords_prev, head_dim=128, temperature=10000.0)
  assert rope_prev.shape == (expected_len, 96) and rope_prev.dtype == np.float32
  # Verify text tokens get exact (0, 0, 0) identity RoPE (sin=0, cos=1)
  txt_start = 32 * 4 * 6 + 250
  txt_rope = rope_prev[txt_start : txt_start + 17]
  np.testing.assert_allclose(txt_rope[:, :48], 0.0, atol=1e-7)
  np.testing.assert_allclose(txt_rope[:, 48:], 1.0, atol=1e-7)
  # Verify audio maps 0..249 -> 0..31 on t-axis (band 0 has omega=1.0 -> sin(31), cos(31))
  audio_last = rope_prev[32 * 4 * 6 + 249]
  assert abs(float(audio_last[0]) - math.sin(31.0)) < 1e-5
  assert abs(float(audio_last[48]) - math.cos(31.0)) < 1e-5
  # Verify ref image 0 special token: t=34, h_eff=-(4+1)/2=-2.5, w_eff=-(6+1)/2=-3.5
  sp0 = rope_prev[txt_start + 17]
  assert abs(float(sp0[0]) - math.sin(34.0)) < 1e-5
  assert abs(float(sp0[16]) - math.sin(-2.5)) < 1e-5
  assert abs(float(sp0[32]) - math.sin(-3.5)) < 1e-5

  # Verify partial RoPE (3D & 4D), norm preservation on [0:96], pass-through on [96:128]
  q3 = rng.standard_normal((expected_len, 4, 128)).astype(np.float32)
  q3_rot = apply_3d_partial_rope(q3, rope_prev)
  assert q3_rot.shape == q3.shape and q3_rot.dtype == np.float32
  np.testing.assert_array_equal(q3_rot[..., 96:], q3[..., 96:])
  np.testing.assert_allclose(
      np.linalg.norm(q3_rot[..., :96], axis=-1),
      np.linalg.norm(q3[..., :96], axis=-1),
      rtol=1e-5,
      atol=1e-5,
  )
  # Text segment should be untouched everywhere because RoPE is identity
  np.testing.assert_allclose(
      q3_rot[txt_start : txt_start + 17], q3[txt_start : txt_start + 17], atol=1e-6
  )
  q4 = q3[None, ...]
  q4_rot = apply_3d_partial_rope(q4, rope_prev)
  np.testing.assert_allclose(q4_rot[0], q3_rot, atol=1e-6)
  print("  [PASS] preview RoPE coords, ElementWiseFourierEmbed, apply_3d_partial_rope")

  # 4. Refiner RoPE coords & 3D (8, 4, 4) block permutation
  coords_ref = build_refiner_rope_coords((9, 8, 8), txt_len=12)
  assert coords_ref.shape == (9 * 8 * 8 + 12, 9)
  perm, inv_perm, grid_thw, pad_t = refiner_block_perm((63, 68, 120), 8, 4, 4)
  assert grid_thw == (8, 17, 30) and pad_t == 1
  assert perm.shape == (63 * 68 * 120,) and inv_perm.shape == (63 * 68 * 120,)
  # Check exact bijection
  idx_check = np.arange(63 * 68 * 120, dtype=np.int32)
  assert np.array_equal(perm[inv_perm], idx_check)
  assert np.array_equal(inv_perm[perm], idx_check)
  # Check first block on small (9, 8, 8) grid contains t in 0..7, h in 0..3, w in 0..3
  p_s, ip_s, g_s, pt_s = refiner_block_perm((9, 8, 8), 8, 4, 4)
  assert g_s == (2, 2, 2) and pt_s == 7
  first_blk = p_s[:128]
  expected_blk0 = (
      np.arange(8)[:, None, None] * 64
      + np.arange(4)[None, :, None] * 8
      + np.arange(4)[None, None, :]
  ).reshape(-1)
  assert np.array_equal(first_blk, expected_blk0)
  # Last block (bt=1, bh=1, bw=1) has only 1 * 4 * 4 = 16 tokens
  last_blk = p_s[-16:]
  expected_last = (8 * 64 + np.arange(4, 8)[:, None] * 8 + np.arange(4, 8)[None, :]).reshape(-1)
  assert np.array_equal(last_blk, expected_last)
  assert np.array_equal(p_s[ip_s], np.arange(9 * 8 * 8, dtype=np.int32))
  print("  [PASS] build_refiner_rope_coords & refiner_block_perm")

  # 5. trilinear_upsample_latents
  # Exact reproduction of any linear function f(t, h, w) = a*t + b*h + c*w + d
  t_in, h_in, w_in = 5, 6, 7
  t_out, h_out, w_out = 9, 13, 15
  tg, hg, wg = np.meshgrid(
      np.linspace(0.0, 1.0, t_in, dtype=np.float32),
      np.linspace(0.0, 1.0, h_in, dtype=np.float32),
      np.linspace(0.0, 1.0, w_in, dtype=np.float32),
      indexing="ij",
  )
  lin_in = np.stack([2.0 * tg - 3.0 * hg + 5.0 * wg + 1.0, -tg + 4.0 * hg - wg], axis=0)
  lin_out = trilinear_upsample_latents(lin_in, (t_out, h_out, w_out))
  assert lin_out.shape == (2, t_out, h_out, w_out)
  tgo, hgo, wgo = np.meshgrid(
      np.linspace(0.0, 1.0, t_out, dtype=np.float32),
      np.linspace(0.0, 1.0, h_out, dtype=np.float32),
      np.linspace(0.0, 1.0, w_out, dtype=np.float32),
      indexing="ij",
  )
  lin_exp = np.stack([2.0 * tgo - 3.0 * hgo + 5.0 * wgo + 1.0, -tgo + 4.0 * hgo - wgo], axis=0)
  np.testing.assert_allclose(lin_out, lin_exp, rtol=1e-5, atol=1e-5)
  if has_torch:
    z_rand = rng.standard_normal((1, 4, 8, 10, 14)).astype(np.float32)
    z_np = trilinear_upsample_latents(z_rand, (15, 21, 30))
    z_pt = F.interpolate(
        torch.from_numpy(z_rand), size=(15, 21, 30), mode="trilinear", align_corners=True
    ).numpy()
    np.testing.assert_allclose(z_np, z_pt, rtol=1e-5, atol=1e-5)
  print("  [PASS] trilinear_upsample_latents (align_corners=True)")

  # 6. FlowUniPCMultistepScheduler: exact ODE recovery on constant velocity field
  for steps, shift in ((100, 7.0), (5, 5.0)):
    sched = FlowUniPCMultistepScheduler()
    sched.set_timesteps(steps, shift=shift)
    assert sched.sigmas.shape == (steps + 1,) and sched.sigmas.dtype == np.float32
    assert sched.timesteps.shape == (steps,) and sched.timesteps.dtype == np.int64
    assert float(sched.sigmas[-1]) == 0.0
    x0_true = rng.standard_normal((1, 4, 2, 3, 3)).astype(np.float32)
    eps_true = rng.standard_normal((1, 4, 2, 3, 3)).astype(np.float32)
    v_const = eps_true - x0_true
    x = (1.0 - sched.sigmas[0]) * x0_true + sched.sigmas[0] * eps_true
    for t in sched.timesteps:
      x = sched.step(v_const, t, x, return_dict=False)[0]
    np.testing.assert_allclose(x, x0_true, rtol=1e-4, atol=1e-4)
  print("  [PASS] FlowUniPCMultistepScheduler (100/shift=7 & 5/shift=5)")

  # 7. JAX ops: swiglu7, rms_norm_1pw, multi_modality_rms_norm, sinkhorn_knopp, mhc_pre/post
  if has_jax:
    g_np = np.array([-20.0, -7.0, 0.0, 3.5, 7.0, 20.0], dtype=np.float32)
    u_np = np.array([-20.0, -7.0, 0.0, 3.5, 7.0, 20.0], dtype=np.float32)
    g_c = np.minimum(g_np, 7.0)
    u_c = np.clip(u_np, -7.0, 7.0)
    sw_ref = (g_c * (1.0 / (1.0 + np.exp(-1.702 * g_c)))) * (u_c + 1.0)
    sw_jax = swiglu7(jnp.asarray(g_np), jnp.asarray(u_np))
    assert sw_jax.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(sw_jax, dtype=np.float32), sw_ref, rtol=1e-2, atol=1e-2)
    inter_np = np.stack([g_np, u_np], axis=-1).reshape(-1)
    sw_int = swiglu7_interleaved(jnp.asarray(inter_np))
    assert np.array_equal(np.asarray(sw_jax, dtype=np.float32), np.asarray(sw_int, dtype=np.float32))

    # rms_norm_1pw & multi_modality_rms_norm (2D & 3D, static & modality_ids & 0-len)
    x_np = rng.standard_normal((15, 16)).astype(np.float32)
    w_single = rng.standard_normal((16,)).astype(np.float32) * 0.1
    rms_ref = x_np / np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + 1e-6) * (1.0 + w_single)
    y_single = rms_norm_1pw(jnp.asarray(x_np), jnp.asarray(w_single), cast_back=False)
    np.testing.assert_allclose(np.asarray(y_single), rms_ref, rtol=1e-5, atol=1e-5)

    w_mm = rng.standard_normal((3, 16)).astype(np.float32) * 0.1
    lens = (7, 0, 8)
    ids = np.array([0] * 7 + [2] * 8, dtype=np.int32)
    y_lens = multi_modality_rms_norm(
        jnp.asarray(x_np), jnp.asarray(w_mm), modality_lens=lens, cast_back=False
    )
    y_flat = multi_modality_rms_norm(
        jnp.asarray(x_np), jnp.asarray(w_mm.reshape(-1)), modality_lens=lens, cast_back=False
    )
    y_ids = multi_modality_rms_norm(
        jnp.asarray(x_np), jnp.asarray(w_mm), modality_ids=jnp.asarray(ids), cast_back=False
    )
    y_dyn = jax.jit(
        lambda x_, w_, l_: multi_modality_rms_norm(x_, w_, modality_lens=l_, cast_back=False)
    )(jnp.asarray(x_np), jnp.asarray(w_mm), jnp.asarray(lens, dtype=jnp.int32))
    mm_ref = x_np / np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + 1e-6) * (1.0 + w_mm[ids])
    for arr in (y_lens, y_flat, y_ids, y_dyn):
      np.testing.assert_allclose(np.asarray(arr), mm_ref, rtol=1e-5, atol=1e-5)

    # 3D QK-norm (S, H, head_dim)
    q_np = rng.standard_normal((15, 4, 16)).astype(np.float32)
    y_qk = multi_modality_rms_norm(
        jnp.asarray(q_np), jnp.asarray(w_mm), modality_lens=lens, cast_back=False
    )
    qk_ref = q_np / np.sqrt(np.mean(q_np**2, axis=-1, keepdims=True) + 1e-6) * (
        1.0 + w_mm[ids][:, None, :]
    )
    np.testing.assert_allclose(np.asarray(y_qk), qk_ref, rtol=1e-5, atol=1e-5)

    # Sinkhorn-Knopp (col-first axis=-2, row-second axis=-1)
    h_np = rng.standard_normal((8, 4, 4)).astype(np.float32)
    m_ref = np.exp(h_np - np.max(h_np, axis=(-2, -1), keepdims=True))
    for _ in range(20):
      m_ref = m_ref / (np.sum(m_ref, axis=-2, keepdims=True) + 1e-12)
      m_ref = m_ref / (np.sum(m_ref, axis=-1, keepdims=True) + 1e-12)
    m_jax = sinkhorn_knopp(jnp.asarray(h_np), iters=20, eps=1e-12)
    assert m_jax.dtype == jnp.bfloat16
    m_jax_f = np.asarray(m_jax, dtype=np.float32)
    np.testing.assert_allclose(m_jax_f, m_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.sum(m_jax_f, axis=-1), 1.0, atol=2e-2)
    np.testing.assert_allclose(np.sum(m_jax_f, axis=-2), 1.0, atol=2e-2)
    # 1-iter test verifies exact col-first then row-second order (rows sum to 1, cols may not)
    m1 = np.asarray(
        sinkhorn_knopp(jnp.asarray(h_np), iters=1, eps=1e-12, out_dtype=jnp.float32)
    )
    np.testing.assert_allclose(np.sum(m1, axis=-1), 1.0, atol=1e-5)

    # mhc_pre & mhc_post round-trip & jit test
    s_np = rng.standard_normal((15, 4, 32)).astype(np.float32)
    s_jax = jnp.asarray(s_np, dtype=jnp.bfloat16)
    norm_w = jnp.asarray(rng.standard_normal((3, 128)).astype(np.float32) * 0.1)
    phi_fused = jnp.asarray(rng.standard_normal((128, 24)).astype(np.float32) * 0.1)
    a_pre = jnp.array([0.01], dtype=jnp.float32)
    b_pre = jnp.zeros((4,), dtype=jnp.float32)
    a_post = jnp.array([0.01], dtype=jnp.float32)
    b_post = jnp.zeros((4,), dtype=jnp.float32)
    a_res = jnp.array([0.01], dtype=jnp.float32)
    b_res = jnp.zeros((4, 4), dtype=jnp.float32)

    x_in, h_post_raw, h_res_raw = mhc_pre(
        s_jax.reshape(15, 128), norm_w, phi_fused, a_pre, b_pre, modality_lens=(7, 3, 5)
    )
    assert x_in.shape == (15, 32) and x_in.dtype == jnp.bfloat16
    assert h_post_raw.shape == (15, 4) and h_post_raw.dtype == jnp.float32
    assert h_res_raw.shape == (15, 4, 4) and h_res_raw.dtype == jnp.float32

    s_next = mhc_post(s_jax, x_in, h_post_raw, h_res_raw, a_post, b_post, a_res, b_res)
    assert s_next.shape == (15, 4, 32) and s_next.dtype == jnp.bfloat16

    # Verify apply_3d_partial_rope and trilinear_upsample_latents on JAX arrays
    q_jax = jnp.asarray(q3, dtype=jnp.bfloat16)
    q_jax_rot = apply_3d_partial_rope(q_jax, jnp.asarray(rope_prev))
    assert q_jax_rot.dtype == jnp.bfloat16
    z_jax_up = trilinear_upsample_latents(jnp.asarray(lin_in), (t_out, h_out, w_out))
    np.testing.assert_allclose(np.asarray(z_jax_up), lin_exp, rtol=1e-5, atol=1e-5)
    print("  [PASS] JAX ops (swiglu7, RMSNorm, Sinkhorn, mHC, RoPE, trilinear)")

  print("[self-test] ALL PASSED!")


if __name__ == "__main__":
  _run_self_tests()
