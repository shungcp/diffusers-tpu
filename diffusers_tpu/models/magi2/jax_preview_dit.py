"""Stage 1 MAGI-2 Preview (113.93B / 5.95B active) DiT in JAX / Flax NNX for TPU v7x-8.

Exact native JAX port of ``inference/model/magi2_preview.py`` and
``inference/infra/checkpoint/magi2_checkpointing.py`` following ``diffusers-tpu``
conventions (lazy Flax NNX import in ``_build_modules()``, ``nnx.eval_shape``,
``nnx.split``/``nnx.merge`` inside ``jax.jit``, top-level Pallas kernel imports,
and streaming ``diffusers_tpu.safetensors_io.safe_open`` loading).

Architecture Summary (``Magi2PreviewConfig``):
  * 40 layers, ``hidden_size = 3072``, 4-stream Manifold-Constrained
    Hyper-Connections (mHC, width ``12288``, 20-iter col-then-row Sinkhorn).
  * ``mm_layers = (0, 1, 38, 39)``: 3-modality attention + 3-modality mHC norm
    + 3-modality dense SwiGLU7 MLP (``3072 -> 16384 -> 8192 -> 3072``).
  * ``moe_layers = (2..37)``: single-stream attention & mHC norm + 3-modality
    ``MultiHeadMoELayer`` (``pre_norm`` M=3, ``split_linear``,
    ``CoreMultiHeadMoE`` with 12 heads x 256 experts top-6, ``merge_linear``,
    global shared expert + 3-modality-specific shared expert).
  * No AdaLN: timestep is injected solely by overwriting ``x[:, :64]`` of
    ``PreAdapter`` output (stream 0) with ``sinusoidal_embedding_1d`` (cos/sin).
  * Attention: 24 Q / 24 KV heads (``head_dim = 128``), fp32 QK-norm, 3D
    partial RoPE (``[0:96]`` rotated, ``[96:128]`` pass-through), Ulysses
    ``CP=8`` (3 heads/chip), exact varlen ``[cond, uncond]`` separation via
    ``splash.residual_attention`` (NO key mean-centering!), exact 1-token
    Attention Sink, and per-head scalar sigmoid output gate ``linear_g``.
  * Checkpoint parity: loads ``moe_mlp.router.expert_bias_ema`` into
    ``expert_bias`` by default (set ``MAGI2_ROUTER_BIAS_SOURCE=main`` to keep
    raw ``expert_bias``).
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import numpy as np

# Top-level JAX & Pallas kernel imports (never import Pallas inside jax.jit!).
try:
  import jax
  from jax import lax
  import jax.numpy as jnp
  from ...kernels import block_sizes as _block_sizes
  from ...kernels import fixed_m_bound as _fixed_m_bound
  from ...kernels import splash as _splash
except ImportError:  # pragma: no cover - allows syntax checks without JAX installed
  jax = None  # type: ignore[assignment]
  lax = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]
  _block_sizes = None  # type: ignore[assignment]
  _fixed_m_bound = None  # type: ignore[assignment]
  _splash = None  # type: ignore[assignment]

from .config import Magi2PreviewConfig
from .jax_mh_moe import (
    _build_modules as _build_moe_modules,
    shard_routed_down,
    shard_routed_gate_up,
    use_fp8_env,
)
from .rope_and_ops import (
    apply_3d_partial_rope,
    build_preview_rope_coords,
    elementwise_fourier_embed,
    mhc_fused_post_pre_tokamax,
    mhc_post,
    mhc_post_tokamax,
    mhc_pre,
    mhc_pre_tokamax,
    multi_modality_rms_norm,
    rms_norm_1pw,
    sinusoidal_embedding_1d,
    swiglu7,
    swiglu7_interleaved,
    use_tokamax_mhc,
)

# Exact modality IDs from inference/model/magi2_preview.py:L71-79.
MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2
MODALITY_TIME = 3  # Remapped to MODALITY_TEXT internally before dispatch.

_LOG2_E = 1.4426950408889634
_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"


# ---------------------------------------------------------------------------
# Pure-JAX helpers (RoPE, Splash/Ref Attention with 1-Token Sink)
# ---------------------------------------------------------------------------


def _is_tpu() -> bool:
  if jax is None:
    return False
  try:
    devs = jax.devices()
    return bool(devs) and devs[0].platform == "tpu"
  except Exception:
    return False


def _resolve_use_splash(use_splash: bool | None) -> bool:
  if use_splash is not None:
    return bool(use_splash) and _splash is not None and _is_tpu()
  env = os.environ.get("DT_MAGI2_ATTN_BACKEND", "auto").strip().lower()
  if env == "ref":
    return False
  if env == "splash":
    return _splash is not None
  return _splash is not None and _is_tpu()


def _elementwise_fourier_embed_jax(
    coords_9: Any,
    head_dim: int = 128,
    temperature: float = 10000.0,
) -> Any:
  """On-device pure-JAX equivalent of ``elementwise_fourier_embed`` for ``jax.jit``."""
  coords = coords_9.astype(jnp.float32)
  num_bands = head_dim // 8
  exp = jnp.arange(num_bands, dtype=jnp.float32) / jnp.float32(num_bands)
  bands = jnp.float32(1.0) / (jnp.float32(temperature) ** exp)

  coords_xyz = coords[:, :3]
  sizes = coords[:, 3:6]
  refs = coords[:, 6:9]

  both_one = (refs == jnp.float32(1.0)) & (sizes == jnp.float32(1.0))
  denom = jnp.where(both_one, jnp.float32(1.0), sizes - jnp.float32(1.0))
  scales = jnp.where(both_one, jnp.float32(1.0), (refs - jnp.float32(1.0)) / denom)

  centers = (sizes - jnp.float32(1.0)) * jnp.float32(0.5)
  centers = centers.at[:, 0].set(jnp.float32(0.0))
  coords_centered = coords_xyz - centers

  proj = coords_centered[:, :, None] * scales[:, :, None] * bands[None, None, :]
  sin_proj = jnp.sin(proj)
  cos_proj = jnp.cos(proj)
  return jnp.concatenate([sin_proj, cos_proj], axis=1).reshape(coords.shape[0], -1)


def apply_residual_attention_sink(
    o_unnorm: Any,
    m: Any,
    l: Any,
    sinks_local: Any,
) -> Any:
  """Apply exact Attention Sink to base-2 ``splash.residual_attention`` outputs.

  ``splash.residual_attention`` returns:
    ``o_unnorm = sum_j 2^(a_j * log2(e) - m) * v_j``  (shape ``(B, H, Q, D)``)
    ``l        = sum_j 2^(a_j * log2(e) - m)``        (shape ``(B, H, Q, 1)``)
    ``m``                                             (shape ``(B, H, Q, 1)``)

  Adding a virtual zero-value sink key with logit ``s_h`` gives:
    ``o = o_unnorm / (l + exp2(s_h * log2(e) - m))``
        ``== (o_unnorm / l) * sigmoid(lse_nat - s_h)``.

  We compute with ``m_eff = max(m, sink_b2)`` so both ``exp2`` exponents are
  ``<= 0`` — 100% overflow-free and NaN-free while remaining algebraically
  identical to ``o_unnorm / (l + exp2(sink_b2 - m))``.
  """
  sinks_f = sinks_local.astype(jnp.float32)
  if sinks_f.shape[0] == 1:
    sink_nat = sinks_f[0]  # (H_loc,)
  else:
    sink_nat = jax.scipy.special.logsumexp(sinks_f, axis=0)  # (H_loc,)
  sink_b2 = (sink_nat * jnp.float32(_LOG2_E))[None, :, None, None]  # (1, H_loc, 1, 1)

  m_eff = jnp.maximum(m, sink_b2)
  alpha = jnp.exp2(m - m_eff)
  sink_w = jnp.exp2(sink_b2 - m_eff)
  return (o_unnorm * alpha) / (l * alpha + sink_w)


def _sdpa_with_sink_single_segment(
    q_seg: Any,
    k_seg: Any,
    v_seg: Any,
    sinks_local: Any | None,
    *,
    use_splash: bool,
) -> Any:
  """Single-sample bidirectional SDPA + optional Attention Sink.

  Args:
    q_seg: ``(1, H_q_local, S, D)``
    k_seg: ``(1, H_kv_local, S, D)``
    v_seg: ``(1, H_kv_local, S, D)``
    sinks_local: ``(sink_num, H_q_local)`` float32 or ``None``.
    use_splash: If True, run TPU Pallas ``splash.residual_attention``
      (NEVER mean-centering K, since softmax with a fixed sink logit is not
      shift-invariant in K!). If False, run exact float32 reference attention.
  """
  hq = q_seg.shape[1]
  hkv = k_seg.shape[1]
  head_dim = q_seg.shape[-1]
  seq_len = q_seg.shape[2]
  out_dtype = q_seg.dtype

  if hkv != hq:
    if hq % hkv != 0:
      raise ValueError(f"num_heads_q={hq} must be divisible by num_heads_kv={hkv}")
    rep = hq // hkv
    k_seg = jnp.repeat(k_seg, rep, axis=1)
    v_seg = jnp.repeat(v_seg, rep, axis=1)

  scale = head_dim ** -0.5

  if use_splash and _splash is not None and head_dim % 128 == 0:
    try:
      blocks = _block_sizes.resolve_block_sizes(
          seq_len, seq_len, head_dim, {}, "residual", _block_sizes.detect_spec()
      )
    except Exception:
      blocks = {
          "block_q": min(1024, seq_len),
          "block_kv": min(1024, seq_len),
          "block_kv_compute": min(512, seq_len),
      }
    # IMPORTANT: Do NOT subtract mean(k) here! With an attention sink,
    # shifting K shifts QK^T relative to the fixed sink logit s_h.
    # Also keep use_fixed_m=False so m is exact online rowmax.
    o_unnorm, m, l = _splash.residual_attention(
        q_seg.astype(jnp.bfloat16),
        k_seg.astype(jnp.bfloat16),
        v_seg.astype(jnp.bfloat16),
        scale=scale,
        use_fixed_m=False,
        **blocks,
    )
    if sinks_local is not None:
      o_f32 = apply_residual_attention_sink(o_unnorm, m, l, sinks_local)
    else:
      o_f32 = o_unnorm / jnp.maximum(l, jnp.float32(1e-30))
    return o_f32.astype(out_dtype)

  # Exact float32 fallback (CPU parity tests & small head_dim).
  logits = (
      jnp.einsum(
          "bhqd,bhkd->bhqk",
          q_seg.astype(jnp.float32),
          k_seg.astype(jnp.float32),
          precision=lax.Precision.HIGHEST,
      )
      * jnp.float32(scale)
  )
  lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  probs = jnp.exp(logits - lse)
  o_std = jnp.einsum(
      "bhqk,bhkd->bhqd",
      probs,
      v_seg.astype(jnp.float32),
      precision=lax.Precision.HIGHEST,
  )
  if sinks_local is not None:
    sinks_f = sinks_local.astype(jnp.float32)
    if sinks_f.shape[0] == 1:
      sink_nat = sinks_f[0]
    else:
      sink_nat = jax.scipy.special.logsumexp(sinks_f, axis=0)
    o_std = o_std * jax.nn.sigmoid(lse - sink_nat[None, :, None, None])
  return o_std.astype(out_dtype)


# ---------------------------------------------------------------------------
# Sharding spec builder for nnx.split / jax.shard_map
# ---------------------------------------------------------------------------


def build_sharded_state_specs(state: Any, axis_name: str = "tp") -> Any:
  """Return a ``PartitionSpec`` pytree matching ``nnx.split(model)[1]``.

  Routed MoE weights ``w_gate_up`` / ``w_down`` (and optional FP8 scales) are
  sharded along axis 0 (``ep``) over ``axis_name``; all dense/attn/shared/mHC
  weights (~9.94 GiB total) are replicated ``P()``.
  """
  from jax.sharding import PartitionSpec as P

  def _leaf_spec(path, _leaf):
    keys = [
        str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k))))
        for k in path
    ]
    if keys and keys[-1] == "value":
      keys = keys[:-1]
    name = keys[-1] if keys else ""
    if name in ("w_gate_up", "w_down"):
      return P(axis_name, None, None, None, None)
    if name in ("w_gate_up_scale", "w_down_scale"):
      return P(axis_name, None, None, None, None, None)
    return P()

  return jax.tree_util.tree_map_with_path(_leaf_spec, state)


# ---------------------------------------------------------------------------
# Flax NNX Modules (lazy flax import inside _build_modules)
# ---------------------------------------------------------------------------


def _build_modules():
  """Lazily import Flax NNX and construct Stage 1 MAGI-2 Preview modules."""
  from flax import nnx
  from jax.sharding import PartitionSpec as P

  CoreMultiHeadMoE = _build_moe_modules()

  class GroupedLinear(nnx.Module):
    """Exact JAX equivalent of ``CustomGroupedLinear`` (no bias).

    Stores weights as ``(in_features, out_features)`` when ``num_modality == 1``
    or ``(num_modality, in_features, out_features)`` when ``num_modality > 1``,
    using ``jax.lax.ragged_dot`` for dynamic per-CP-rank modality counts inside
    ``shard_map`` with 100% static SPMD shapes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_modality: int = 1,
        dtype: Any = None,
    ):
      self.in_features = in_features
      self.out_features = out_features
      self.num_modality = num_modality
      dt = dtype if dtype is not None else jnp.bfloat16
      shape = (
          (in_features, out_features)
          if num_modality == 1
          else (num_modality, in_features, out_features)
      )
      self.weight = nnx.Param(jnp.zeros(shape, dt))

    def __call__(self, x: Any, group_sizes: Any | None = None) -> Any:
      w = self.weight[...]
      out_dt = x.dtype
      prec = (
          lax.Precision.HIGHEST
          if jnp.dtype(out_dt) == jnp.dtype(jnp.float32)
          else lax.Precision.DEFAULT
      )
      if self.num_modality == 1:
        y = lax.dot_general(
            x,
            w.astype(out_dt),
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
        return y.astype(out_dt)
      if group_sizes is None:
        raise ValueError("group_sizes is required for multi-modality GroupedLinear")
      y = lax.ragged_dot(
          x,
          w.astype(out_dt),
          group_sizes,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      return y.astype(out_dt)

  class Magi2PreAdapter(nnx.Module):
    """Float32 modality input projections (``PreAdapter``, ``bias=True``)."""

    def __init__(self, config: Magi2PreviewConfig):
      self.config = config
      ad = config.stream_hidden_size
      self.adapter_dim = ad
      self.video_embedder_weight = nnx.Param(
          jnp.zeros((config.video_in_channels, ad), jnp.float32)
      )
      self.video_embedder_bias = nnx.Param(jnp.zeros((ad,), jnp.float32))
      self.audio_embedder_weight = nnx.Param(
          jnp.zeros((config.audio_in_channels, ad), jnp.float32)
      )
      self.audio_embedder_bias = nnx.Param(jnp.zeros((ad,), jnp.float32))
      self.text_embedder_weight = nnx.Param(
          jnp.zeros((config.text_in_channels, ad), jnp.float32)
      )
      self.text_embedder_bias = nnx.Param(jnp.zeros((ad,), jnp.float32))

    def embed_video(self, x_video: Any) -> Any:
      return (
          lax.dot_general(
              x_video.astype(jnp.float32),
              self.video_embedder_weight[...],
              (((x_video.ndim - 1,), (0,)), ((), ())),
              precision=lax.Precision.HIGHEST,
          )
          + self.video_embedder_bias[...]
      )

    def embed_audio(self, x_audio: Any) -> Any:
      return (
          lax.dot_general(
              x_audio.astype(jnp.float32),
              self.audio_embedder_weight[...],
              (((x_audio.ndim - 1,), (0,)), ((), ())),
              precision=lax.Precision.HIGHEST,
          )
          + self.audio_embedder_bias[...]
      )

    def embed_text(self, x_text: Any) -> Any:
      return (
          lax.dot_general(
              x_text.astype(jnp.float32),
              self.text_embedder_weight[...],
              (((x_text.ndim - 1,), (0,)), ((), ())),
              precision=lax.Precision.HIGHEST,
          )
          + self.text_embedder_bias[...]
      )

    def __call__(self, x: Any, modality_ids: Any) -> Any:
      """Project packed ``(S, C_in)`` float32 tokens according to ``modality_ids``."""
      c = self.config
      x32 = x.astype(jnp.float32)
      out = jnp.zeros((x.shape[0], self.adapter_dim), dtype=jnp.float32)

      is_v = (modality_ids == MODALITY_VIDEO)[:, None]
      is_a = (modality_ids == MODALITY_AUDIO)[:, None]
      is_t = (modality_ids == MODALITY_TEXT)[:, None]

      out = jnp.where(is_v, self.embed_video(x32[:, : c.video_in_channels]), out)
      out = jnp.where(is_a, self.embed_audio(x32[:, : c.audio_in_channels]), out)
      out = jnp.where(is_t, self.embed_text(x32[:, : c.text_in_channels]), out)
      return out

  class Magi2PostAdapter(nnx.Module):
    """Float32 final RMSNorm + linear projections for video & audio (``bias=False``)."""

    def __init__(self, config: Magi2PreviewConfig):
      self.config = config
      ad = config.stream_hidden_size
      self.adapter_dim = ad
      self.final_out_dim = max(config.video_in_channels, config.audio_in_channels)
      self.final_norm_video_weight = nnx.Param(jnp.zeros((ad,), jnp.float32))
      self.final_norm_audio_weight = nnx.Param(jnp.zeros((ad,), jnp.float32))
      self.final_linear_video_weight = nnx.Param(
          jnp.zeros((ad, config.video_in_channels), jnp.float32)
      )
      self.final_linear_audio_weight = nnx.Param(
          jnp.zeros((ad, config.audio_in_channels), jnp.float32)
      )

    def __call__(self, x: Any, modality_ids: Any) -> Any:
      c = self.config
      x32 = x.astype(jnp.float32)
      v_norm = rms_norm_1pw(
          x32, self.final_norm_video_weight[...], eps=c.rms_norm_eps, cast_back=False
      )
      v_out = lax.dot_general(
          v_norm,
          self.final_linear_video_weight[...],
          (((1,), (0,)), ((), ())),
          precision=lax.Precision.HIGHEST,
      )
      if c.video_in_channels < self.final_out_dim:
        v_out = jnp.pad(
            v_out, ((0, 0), (0, self.final_out_dim - c.video_in_channels))
        )

      a_norm = rms_norm_1pw(
          x32, self.final_norm_audio_weight[...], eps=c.rms_norm_eps, cast_back=False
      )
      a_out = lax.dot_general(
          a_norm,
          self.final_linear_audio_weight[...],
          (((1,), (0,)), ((), ())),
          precision=lax.Precision.HIGHEST,
      )
      if c.audio_in_channels < self.final_out_dim:
        a_out = jnp.pad(
            a_out, ((0, 0), (0, self.final_out_dim - c.audio_in_channels))
        )

      out = jnp.zeros((x.shape[0], self.final_out_dim), dtype=jnp.float32)
      out = jnp.where((modality_ids == MODALITY_VIDEO)[:, None], v_out, out)
      out = jnp.where((modality_ids == MODALITY_AUDIO)[:, None], a_out, out)
      return out

  class Magi2PreviewAttention(nnx.Module):
    """Self-attention with QK-norm, 3D partial RoPE, Ulysses CP, Sink, & scalar gate."""

    def __init__(
        self,
        config: Magi2PreviewConfig,
        *,
        num_modality: int,
        dtype: Any = None,
    ):
      self.config = config
      self.num_modality = num_modality
      self.hidden_size = config.hidden_size
      self.head_dim = config.head_dim
      self.num_heads_q = config.num_heads_q
      self.num_heads_kv = config.num_heads_kv
      self.q_size = config.num_heads_q * config.head_dim
      self.kv_size = config.num_heads_kv * config.head_dim
      self.rms_norm_eps = config.rms_norm_eps
      dt = dtype if dtype is not None else jnp.bfloat16

      norm_shape = (
          (config.hidden_size,) if num_modality == 1 else (num_modality, config.hidden_size)
      )
      qk_norm_shape = (
          (config.head_dim,) if num_modality == 1 else (num_modality, config.head_dim)
      )
      self.pre_norm_weight = nnx.Param(jnp.zeros(norm_shape, jnp.float32))
      self.linear_g = GroupedLinear(
          config.hidden_size, config.num_heads_q, num_modality=num_modality, dtype=dt
      )
      self.linear_qkv = GroupedLinear(
          config.hidden_size,
          self.q_size + 2 * self.kv_size,
          num_modality=num_modality,
          dtype=dt,
      )
      self.linear_proj = GroupedLinear(
          self.q_size, config.hidden_size, num_modality=num_modality, dtype=dt
      )
      self.sinks = nnx.Param(
          jnp.zeros((max(1, config.sink_token_num), config.num_heads_q), jnp.float32)
      )
      self.q_norm_weight = nnx.Param(jnp.zeros(qk_norm_shape, jnp.float32))
      self.k_norm_weight = nnx.Param(jnp.zeros(qk_norm_shape, jnp.float32))

    def __call__(
        self,
        x_sorted: Any,
        rope_local: Any,
        perm_local: Any,
        inv_perm_local: Any,
        modality_ids_sorted: Any,
        group_sizes: Any,
        *,
        sample_lens: tuple[int, ...],
        cp_size: int = 1,
        axis_name: str = "tp",
        use_splash: bool = False,
    ) -> Any:
      s_local = x_sorted.shape[0]
      act_dtype = x_sorted.dtype

      x = multi_modality_rms_norm(
          x_sorted,
          self.pre_norm_weight[...],
          eps=self.rms_norm_eps,
          cast_back=True,
          modality_ids=modality_ids_sorted,
      )
      g = self.linear_g(x, group_sizes).reshape(s_local, self.num_heads_q, 1)
      qkv = self.linear_qkv(x, group_sizes)

      # Contiguous [Q, K, V] layout per modality (NOT head-interleaved).
      q = qkv[:, : self.q_size].reshape(s_local, self.num_heads_q, self.head_dim)
      k = qkv[:, self.q_size : self.q_size + self.kv_size].reshape(
          s_local, self.num_heads_kv, self.head_dim
      )
      v = qkv[:, self.q_size + self.kv_size :].reshape(
          s_local, self.num_heads_kv, self.head_dim
      )

      # QK-norm keeps float32 output so 3D partial RoPE runs in float32.
      q = multi_modality_rms_norm(
          q,
          self.q_norm_weight[...],
          eps=self.rms_norm_eps,
          cast_back=False,
          modality_ids=modality_ids_sorted,
      )
      k = multi_modality_rms_norm(
          k,
          self.k_norm_weight[...],
          eps=self.rms_norm_eps,
          cast_back=False,
          modality_ids=modality_ids_sorted,
      )

      # Unpermute back to local sequence order for RoPE & Ulysses attention.
      q = q[inv_perm_local]
      k = k[inv_perm_local]
      v = v[inv_perm_local]

      q = apply_3d_partial_rope(q, rope_local).astype(act_dtype)
      k = apply_3d_partial_rope(k, rope_local).astype(act_dtype)
      v = v.astype(act_dtype)

      if cp_size > 1:
        if self.num_heads_q % cp_size != 0:
          raise ValueError(
              f"num_heads_q={self.num_heads_q} must be divisible by cp_size={cp_size}"
          )
        if self.num_heads_kv < cp_size:
          rep = self.num_heads_q // self.num_heads_kv
          k = jnp.repeat(k, rep, axis=1)
          v = jnp.repeat(v, rep, axis=1)
        elif self.num_heads_kv % cp_size != 0:
          raise ValueError(
              f"num_heads_kv={self.num_heads_kv} must be divisible by cp_size={cp_size}"
          )
        # Ulysses forward all-to-all: (S_local, H, D) -> (S_total, H_local, D)
        q_full = lax.all_to_all(q, axis_name, split_axis=1, concat_axis=0, tiled=True)
        k_full = lax.all_to_all(k, axis_name, split_axis=1, concat_axis=0, tiled=True)
        v_full = lax.all_to_all(v, axis_name, split_axis=1, concat_axis=0, tiled=True)
        h_q_local = self.num_heads_q // cp_size
        rank = lax.axis_index(axis_name)
        sinks_local = (
            lax.dynamic_slice_in_dim(self.sinks[...], rank * h_q_local, h_q_local, axis=1)
            if self.config.attn_sinks
            else None
        )
      else:
        q_full, k_full, v_full = q, k, v
        sinks_local = self.sinks[...] if self.config.attn_sinks else None

      # Exact varlen [cond, uncond] separation (+ isolated CP padding tail).
      s_total = q_full.shape[0]
      h_q_local = q_full.shape[1]
      seg_outs = []
      offset = 0
      for seg_len in sample_lens:
        if seg_len <= 0:
          continue
        end = offset + seg_len
        if end > s_total:
          raise ValueError(
              f"Sum of sample_lens {sample_lens} exceeds S_total={s_total}"
          )
        q_s = jnp.transpose(q_full[offset:end], (1, 0, 2))[None, ...]  # (1, H_loc, seg, D)
        k_s = jnp.transpose(k_full[offset:end], (1, 0, 2))[None, ...]
        v_s = jnp.transpose(v_full[offset:end], (1, 0, 2))[None, ...]
        o_s = _sdpa_with_sink_single_segment(
            q_s, k_s, v_s, sinks_local, use_splash=use_splash
        )
        seg_outs.append(jnp.transpose(o_s[0], (1, 0, 2)))  # (seg, H_loc, D)
        offset = end

      if offset < s_total:
        # Trailing CP alignment pad tokens are isolated and discarded post-adapter.
        seg_outs.append(jnp.zeros((s_total - offset, h_q_local, self.head_dim), dtype=act_dtype))

      out_full = seg_outs[0] if len(seg_outs) == 1 else jnp.concatenate(seg_outs, axis=0)

      if cp_size > 1:
        # Ulysses reverse all-to-all: (S_total, H_local, D) -> (S_local, H_q, D)
        out = lax.all_to_all(out_full, axis_name, split_axis=0, concat_axis=1, tiled=True)
      else:
        out = out_full

      # Re-permute back to modality-sorted order, apply scalar head gate, project.
      out = out[perm_local]
      if self.config.attn_gating:
        out = (out.astype(jnp.float32) * jax.nn.sigmoid(g.astype(jnp.float32))).astype(
            act_dtype
        )
      out = out.reshape(s_local, self.q_size)
      return self.linear_proj(out, group_sizes)

  class Magi2DenseMLP(nnx.Module):
    """3-modality dense SwiGLU7 MLP used on ``mm_layers = (0, 1, 38, 39)``."""

    def __init__(
        self,
        config: Magi2PreviewConfig,
        *,
        num_modality: int = 3,
        dtype: Any = None,
    ):
      self.num_modality = num_modality
      self.rms_norm_eps = config.rms_norm_eps
      dt = dtype if dtype is not None else jnp.bfloat16
      norm_shape = (
          (config.hidden_size,) if num_modality == 1 else (num_modality, config.hidden_size)
      )
      self.pre_norm_weight = nnx.Param(jnp.zeros(norm_shape, jnp.float32))
      self.up_gate_proj = GroupedLinear(
          config.hidden_size,
          2 * config.dense_intermediate_size,
          num_modality=num_modality,
          dtype=dt,
      )
      self.down_proj = GroupedLinear(
          config.dense_intermediate_size,
          config.hidden_size,
          num_modality=num_modality,
          dtype=dt,
      )

    def __call__(
        self,
        x_sorted: Any,
        modality_ids_sorted: Any,
        group_sizes: Any,
        **_unused,
    ) -> Any:
      x = multi_modality_rms_norm(
          x_sorted,
          self.pre_norm_weight[...],
          eps=self.rms_norm_eps,
          cast_back=True,
          modality_ids=modality_ids_sorted,
      )
      x = self.up_gate_proj(x, group_sizes)
      x = swiglu7_interleaved(x, out_dtype=x_sorted.dtype)
      return self.down_proj(x, group_sizes)

  class Magi2MultiHeadMoELayer(nnx.Module):
    """Multi-Head Latent MoE + Global & Modality-Specific Shared Experts (layers 2..37).

    Note: ``pre_norm`` and ``modality_specific_shared_expert_fc{1,2}`` ALWAYS use
    ``num_modality = 3`` even though attention/mHC on layers 2..37 use ``M = 1``.
    """

    def __init__(
        self,
        config: Magi2PreviewConfig,
        *,
        ep_size: int = 1,
        axis_name: str = "tp",
        dtype: Any = None,
        fp8: bool = False,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ):
      self.rms_norm_eps = config.rms_norm_eps
      dt = dtype if dtype is not None else jnp.bfloat16
      d = config.hidden_size

      self.pre_norm_weight = nnx.Param(jnp.zeros((3, d), jnp.float32))
      self.split_linear = GroupedLinear(d, d, num_modality=1, dtype=dt)
      self.merge_linear = GroupedLinear(d, d, num_modality=1, dtype=dt)
      self.moe_mlp = CoreMultiHeadMoE(
          hidden_size=d,
          num_heads=config.moe_num_heads,
          num_experts=config.moe_num_experts,
          top_k=config.moe_top_k,
          expert_intermediate_size=config.moe_expert_intermediate_size,
          route_norm=config.moe_route_norm,
          route_scale=config.moe_route_scale,
          ep_size=ep_size,
          axis_name=axis_name,
          dtype=dt,
          fp8=fp8,
          mode=moe_mode,
          capacity_factor=capacity_factor,
      )
      self.shared_expert_fc1 = GroupedLinear(
          d, 2 * config.moe_shared_intermediate_size, num_modality=1, dtype=dt
      )
      self.shared_expert_fc2 = GroupedLinear(
          config.moe_shared_intermediate_size, d, num_modality=1, dtype=dt
      )
      self.modality_specific_shared_expert_fc1 = GroupedLinear(
          d,
          2 * config.moe_modality_shared_intermediate_size,
          num_modality=3,
          dtype=dt,
      )
      self.modality_specific_shared_expert_fc2 = GroupedLinear(
          config.moe_modality_shared_intermediate_size,
          d,
          num_modality=3,
          dtype=dt,
      )

    def __call__(
        self,
        x_sorted: Any,
        modality_ids_sorted: Any,
        group_sizes: Any,
        *,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      act_dt = x_sorted.dtype
      norm_out = multi_modality_rms_norm(
          x_sorted,
          self.pre_norm_weight[...],
          eps=self.rms_norm_eps,
          cast_back=True,
          modality_ids=modality_ids_sorted,
      )
      moe_in = self.split_linear(norm_out)
      moe_out = self.moe_mlp.call_in_shard_map(
          moe_in, mode=moe_mode, capacity_factor=capacity_factor
      )
      moe_out = self.merge_linear(moe_out)

      # Global shared expert + 3-modality-specific shared expert.
      sh_global = swiglu7_interleaved(self.shared_expert_fc1(norm_out), out_dtype=act_dt)
      sh_global = self.shared_expert_fc2(sh_global)

      sh_mod = swiglu7_interleaved(
          self.modality_specific_shared_expert_fc1(norm_out, group_sizes),
          out_dtype=act_dt,
      )
      sh_mod = self.modality_specific_shared_expert_fc2(sh_mod, group_sizes)

      return moe_out + sh_global + sh_mod

  class Magi2PreviewLayer(nnx.Module):
    """Single 4-stream mHC DiT layer (either MM dense or MoE)."""

    def __init__(
        self,
        config: Magi2PreviewConfig,
        layer_idx: int,
        *,
        ep_size: int = 1,
        axis_name: str = "tp",
        dtype: Any = None,
        fp8: bool = False,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ):
      self.config = config
      self.layer_idx = layer_idx
      self.is_mm = layer_idx in config.mm_layers
      self.is_moe = layer_idx in config.moe_layers
      num_modality = 3 if self.is_mm else 1
      self.num_modality = num_modality
      self.n_stream = config.mhc_num_stream
      self.hidden_size = config.hidden_size

      self.attention = Magi2PreviewAttention(
          config, num_modality=num_modality, dtype=dtype
      )
      if self.is_moe:
        self.mlp = Magi2MultiHeadMoELayer(
            config,
            ep_size=ep_size,
            axis_name=axis_name,
            dtype=dtype,
            fp8=fp8,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )
      else:
        self.mlp = Magi2DenseMLP(config, num_modality=num_modality, dtype=dtype)

      # 4-stream mHC parameters (float32). mhc_norm is shared across attn & mlp.
      n = config.mhc_num_stream
      c = config.hidden_size
      flat_dim = n * c
      phi_out = n + n + n * n
      alpha_init = float(config.mhc_alpha_init)

      mhc_norm_shape = (flat_dim,) if num_modality == 1 else (num_modality, flat_dim)
      self.mhc_norm_weight = nnx.Param(jnp.zeros(mhc_norm_shape, jnp.float32))
      self.mhc_phi_fused_attn = nnx.Param(jnp.zeros((flat_dim, phi_out), jnp.float32))
      self.mhc_phi_fused_mlp = nnx.Param(jnp.zeros((flat_dim, phi_out), jnp.float32))
      self.mhc_alpha_pre_attn = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_alpha_post_attn = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_alpha_res_attn = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_alpha_pre_mlp = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_alpha_post_mlp = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_alpha_res_mlp = nnx.Param(jnp.full((1,), alpha_init, jnp.float32))
      self.mhc_bias_pre_attn = nnx.Param(jnp.zeros((n,), jnp.float32))
      self.mhc_bias_post_attn = nnx.Param(jnp.zeros((n,), jnp.float32))
      self.mhc_bias_pre_mlp = nnx.Param(jnp.zeros((n,), jnp.float32))
      self.mhc_bias_post_mlp = nnx.Param(jnp.zeros((n,), jnp.float32))
      self.mhc_bias_res_attn = nnx.Param(jnp.zeros((n, n), jnp.float32))
      self.mhc_bias_res_mlp = nnx.Param(jnp.zeros((n, n), jnp.float32))

    def __call__(
        self,
        hidden_states: Any,
        rope_local: Any,
        perm_local: Any,
        inv_perm_local: Any,
        modality_ids_sorted: Any,
        group_sizes: Any,
        *,
        sample_lens: tuple[int, ...],
        cp_size: int = 1,
        axis_name: str = "tp",
        use_splash: bool = False,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      c = self.config
      s_local = hidden_states.shape[0]
      if (
          self.num_modality == 1
          and self.n_stream == 4
          and self.hidden_size >= 128
          and s_local >= 64
          and use_tokamax_mhc()
      ):
        attn_in, post_mix_a, comb_mix_a = mhc_pre_tokamax(
            hidden_states,
            self.mhc_norm_weight[...],
            self.mhc_phi_fused_attn[...],
            self.mhc_alpha_pre_attn[...],
            self.mhc_bias_pre_attn[...],
            self.mhc_alpha_post_attn[...],
            self.mhc_bias_post_attn[...],
            self.mhc_alpha_res_attn[...],
            self.mhc_bias_res_attn[...],
            n_stream=self.n_stream,
            eps=c.rms_norm_eps,
            sinkhorn_iters=c.mhc_sinkhorn_iters,
            sinkhorn_eps=c.mhc_sinkhorn_eps,
        )
        attn_out = self.attention(
            attn_in,
            rope_local,
            perm_local,
            inv_perm_local,
            modality_ids_sorted,
            group_sizes,
            sample_lens=sample_lens,
            cp_size=cp_size,
            axis_name=axis_name,
            use_splash=use_splash,
        )
        s_mid_flat, mlp_in, post_mix_m, comb_mix_m = mhc_fused_post_pre_tokamax(
            attn_out,
            hidden_states,
            post_mix_a,
            comb_mix_a,
            self.mhc_norm_weight[...],
            self.mhc_phi_fused_mlp[...],
            self.mhc_alpha_pre_mlp[...],
            self.mhc_bias_pre_mlp[...],
            self.mhc_alpha_post_mlp[...],
            self.mhc_bias_post_mlp[...],
            self.mhc_alpha_res_mlp[...],
            self.mhc_bias_res_mlp[...],
            n_stream=self.n_stream,
            eps=c.rms_norm_eps,
            sinkhorn_iters=c.mhc_sinkhorn_iters,
            sinkhorn_eps=c.mhc_sinkhorn_eps,
        )
        mlp_out = self.mlp(
            mlp_in,
            modality_ids_sorted,
            group_sizes,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )
        return mhc_post_tokamax(mlp_out, s_mid_flat, post_mix_m, comb_mix_m)

      s = hidden_states.reshape(s_local, self.n_stream, self.hidden_size)

      attn_in, h_post_attn_, h_res_attn_ = mhc_pre(
          s,
          self.mhc_norm_weight[...],
          self.mhc_phi_fused_attn[...],
          self.mhc_alpha_pre_attn[...],
          self.mhc_bias_pre_attn[...],
          eps=c.rms_norm_eps,
          modality_ids=modality_ids_sorted,
      )
      attn_out = self.attention(
          attn_in,
          rope_local,
          perm_local,
          inv_perm_local,
          modality_ids_sorted,
          group_sizes,
          sample_lens=sample_lens,
          cp_size=cp_size,
          axis_name=axis_name,
          use_splash=use_splash,
      )
      s_mid = mhc_post(
          s,
          attn_out,
          h_post_attn_,
          h_res_attn_,
          self.mhc_alpha_post_attn[...],
          self.mhc_bias_post_attn[...],
          self.mhc_alpha_res_attn[...],
          self.mhc_bias_res_attn[...],
          sinkhorn_iters=c.mhc_sinkhorn_iters,
          sinkhorn_eps=c.mhc_sinkhorn_eps,
      )

      mlp_in, h_post_mlp_, h_res_mlp_ = mhc_pre(
          s_mid,
          self.mhc_norm_weight[...],
          self.mhc_phi_fused_mlp[...],
          self.mhc_alpha_pre_mlp[...],
          self.mhc_bias_pre_mlp[...],
          eps=c.rms_norm_eps,
          modality_ids=modality_ids_sorted,
      )
      mlp_out = self.mlp(
          mlp_in,
          modality_ids_sorted,
          group_sizes,
          moe_mode=moe_mode,
          capacity_factor=capacity_factor,
      )
      s_next = mhc_post(
          s_mid,
          mlp_out,
          h_post_mlp_,
          h_res_mlp_,
          self.mhc_alpha_post_mlp[...],
          self.mhc_bias_post_mlp[...],
          self.mhc_alpha_res_mlp[...],
          self.mhc_bias_res_mlp[...],
          sinkhorn_iters=c.mhc_sinkhorn_iters,
          sinkhorn_eps=c.mhc_sinkhorn_eps,
      )
      return s_next.reshape(s_local, self.n_stream * self.hidden_size)

  class Magi2PreviewTransformer(nnx.Module):
    """Full Stage 1 MAGI-2 Preview 113.93B DiT (``PreAdapter`` + 40 layers + ``PostAdapter``)."""

    def __init__(
        self,
        config: Magi2PreviewConfig | None = None,
        *,
        ep_size: int = 1,
        axis_name: str = "tp",
        dtype: Any = None,
        fp8: bool = False,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ):
      self.config = config or Magi2PreviewConfig()
      self.ep_size = ep_size
      self.axis_name = axis_name
      self.dtype = dtype if dtype is not None else jnp.bfloat16
      self.fp8 = fp8

      self.pre_adapter = Magi2PreAdapter(self.config)
      self.layers = nnx.List([
          Magi2PreviewLayer(
              self.config,
              i,
              ep_size=ep_size,
              axis_name=axis_name,
              dtype=self.dtype,
              fp8=fp8,
              moe_mode=moe_mode,
              capacity_factor=capacity_factor,
          )
          for i in range(self.config.num_layers)
      ])
      self.post_adapter = Magi2PostAdapter(self.config)

    def forward_embedded_local(
        self,
        x_embedded_local: Any,
        rope_or_coords_local: Any,
        modality_mapping_local: Any,
        *,
        sample_lens: tuple[int, ...],
        time_token_sequence_local: Any | None = None,
        cp_size: int = 1,
        use_splash: bool | None = None,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      """Run timestep overwrite + 40-layer block + PostAdapter on already-embedded tokens.

      Intended for execution inside ``jax.shard_map`` (when ``cp_size > 1``) or
      directly on a single device (when ``cp_size == 1``). Allows callers to
      pre-embed static text tokens once across all 100 denoising steps if desired.
      """
      c = self.config
      # Remap TIME (3) -> TEXT (2), matching magi2_preview.py:L3224-3226.
      mod_ids = jnp.where(
          modality_mapping_local == MODALITY_TIME,
          jnp.int32(MODALITY_TEXT),
          modality_mapping_local.astype(jnp.int32),
      )

      if rope_or_coords_local.shape[-1] == 9:
        rope_local = _elementwise_fourier_embed_jax(
            rope_or_coords_local, head_dim=c.head_dim, temperature=c.rope_temperature
        )
      else:
        rope_local = rope_or_coords_local.astype(jnp.float32)

      x = x_embedded_local.astype(jnp.float32)
      # Timestep injection: overwrite stream-0 channels [0:time_channel_dim] in fp32!
      if time_token_sequence_local is not None and time_token_sequence_local.shape[-1] > 0:
        t_dim = time_token_sequence_local.shape[-1]
        x = x.at[:, :t_dim].set(time_token_sequence_local.astype(x.dtype))

      # Cast to block compute dtype (bf16 in prod, fp32 in fp32 parity tests).
      x = x.astype(self.dtype)

      # Local modality sorting [VIDEO, AUDIO, TEXT] for MultiModalityRMSNorm & GroupedLinear.
      perm_local = jnp.argsort(mod_ids, stable=True)
      inv_perm_local = jnp.empty_like(perm_local).at[perm_local].set(
          jnp.arange(perm_local.shape[0], dtype=perm_local.dtype)
      )
      mod_ids_sorted = mod_ids[perm_local]
      group_sizes = jnp.bincount(mod_ids_sorted, length=3).astype(jnp.int32)

      x_sorted = x[perm_local]
      splash_flag = _resolve_use_splash(use_splash)

      for layer in self.layers:
        x_sorted = layer(
            x_sorted,
            rope_local,
            perm_local,
            inv_perm_local,
            mod_ids_sorted,
            group_sizes,
            sample_lens=sample_lens,
            cp_size=cp_size,
            axis_name=self.axis_name,
            use_splash=splash_flag,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )

      x_seq = x_sorted[inv_perm_local]
      return self.post_adapter(x_seq, mod_ids)

    def forward_local(
        self,
        x_local: Any,
        rope_or_coords_local: Any,
        modality_mapping_local: Any,
        *,
        sample_lens: tuple[int, ...],
        time_token_sequence_local: Any | None = None,
        cp_size: int = 1,
        use_splash: bool | None = None,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      """Full ``PreAdapter`` + block + ``PostAdapter`` on one device / inside ``shard_map``."""
      mod_ids = jnp.where(
          modality_mapping_local == MODALITY_TIME,
          jnp.int32(MODALITY_TEXT),
          modality_mapping_local.astype(jnp.int32),
      )
      x_emb = self.pre_adapter(x_local, mod_ids)
      return self.forward_embedded_local(
          x_emb,
          rope_or_coords_local,
          mod_ids,
          sample_lens=sample_lens,
          time_token_sequence_local=time_token_sequence_local,
          cp_size=cp_size,
          use_splash=use_splash,
          moe_mode=moe_mode,
          capacity_factor=capacity_factor,
      )

    def __call__(
        self,
        x: Any,
        rope_or_coords: Any,
        modality_mapping: Any,
        *,
        sample_lens: tuple[int, ...] | None = None,
        time_token_sequence: Any | None = None,
        raw_t: Any | None = None,
        mesh: Any = None,
        use_splash: bool | None = None,
        moe_mode: str | None = None,
        capacity_factor: float | None = None,
    ) -> Any:
      """Run Stage 1 MAGI-2 Preview DiT (single-device or Ulysses CP + EP on ``mesh``).

      Note: For compiled inference loops, prefer ``make_jitted_forward(model, ...)``
      which uses ``nnx.split``/``nnx.merge`` inside ``jax.jit`` (never ``nnx.jit``).
      """
      if time_token_sequence is None and raw_t is not None:
        time_token_sequence = sinusoidal_embedding_1d(
            raw_t, dim=self.config.time_channel_dim
        )
      s_total = x.shape[0]
      seg_lens = (s_total,) if sample_lens is None else tuple(int(v) for v in sample_lens)

      cp_size = (
          int(dict(mesh.shape).get(self.axis_name, 1)) if mesh is not None else 1
      )
      if mesh is None or cp_size <= 1:
        return self.forward_local(
            x,
            rope_or_coords,
            modality_mapping,
            sample_lens=seg_lens,
            time_token_sequence_local=time_token_sequence,
            cp_size=1,
            use_splash=use_splash,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )

      graphdef, state = nnx.split(self)
      state_specs = build_sharded_state_specs(state, axis_name=self.axis_name)
      ax = self.axis_name
      has_time = time_token_sequence is not None

      def _sharded_fn(st, x_l, rope_l, mod_l, time_l):
        mdl = nnx.merge(graphdef, st)
        return mdl.forward_local(
            x_l,
            rope_l,
            mod_l,
            sample_lens=seg_lens,
            time_token_sequence_local=time_l,
            cp_size=cp_size,
            use_splash=use_splash,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )

      fn = jax.shard_map(
          _sharded_fn,
          mesh=mesh,
          in_specs=(
              state_specs,
              P(ax, None),
              P(ax, None),
              P(ax,),
              P(ax, None) if has_time else None,
          ),
          out_specs=P(ax, None),
          check_vma=False,
      )
      return fn(state, x, rope_or_coords, modality_mapping, time_token_sequence)

  return (
      Magi2PreAdapter,
      Magi2PostAdapter,
      Magi2PreviewAttention,
      Magi2DenseMLP,
      Magi2MultiHeadMoELayer,
      Magi2PreviewLayer,
      Magi2PreviewTransformer,
  )


# ---------------------------------------------------------------------------
# Jitted runner (nnx.split / nnx.merge inside jax.jit — NEVER nnx.jit!)
# ---------------------------------------------------------------------------


def _stack_preview_layer_states(
    layer_states: list[Any],
    mesh: Any,
    axis_name: str,
    *,
    release_layers: list[Any] | None = None,
) -> tuple[Any, Any]:
  """Stack identical Preview layer states along axis 0 purely device-locally.

  Returns ``(stacked_state, stacked_specs)`` where EP-sharded MoE leaves get
  ``P(None, axis_name, ...)`` and replicated leaves get ``P()``. When
  ``release_layers`` is provided, source leaf references are cleared as each
  leaf path is stacked to keep peak HBM <= ~42 GiB/chip for 36 MoE layers.
  """
  from jax.sharding import NamedSharding, PartitionSpec as P

  single_specs = build_sharded_state_specs(layer_states[0], axis_name=axis_name)
  flat_spec_leaves, treedef = jax.tree_util.tree_flatten(single_specs)
  flat_per_layer = [jax.tree_util.tree_leaves(st) for st in layer_states]
  n_leaves = len(flat_spec_leaves)

  stacked_leaves = []
  stacked_specs = []
  for leaf_idx in range(n_leaves):
    orig_spec = flat_spec_leaves[leaf_idx]
    arrs = [per[leaf_idx] for per in flat_per_layer]
    if orig_spec == P():
      new_spec = P()
    else:
      new_spec = P(None, *orig_spec)
    stacked_specs.append(new_spec)

    if mesh is not None and mesh.size > 1:
      out_sh = NamedSharding(mesh, new_spec)
      st_arr = jax.jit(lambda *xs: jnp.stack(xs, axis=0), out_shardings=out_sh)(*arrs)
      st_arr.block_until_ready()
    else:
      st_arr = jnp.stack(arrs, axis=0)
    stacked_leaves.append(st_arr)
    # Free references in flat_per_layer immediately so memory can be reclaimed.
    for per in flat_per_layer:
      per[leaf_idx] = None
    del arrs

  if release_layers is not None:
    # Clear original NNX layer Param references to avoid holding 2x 202.5 GiB MoE weights.
    for lyr in release_layers:
      if hasattr(lyr, "mlp") and hasattr(lyr.mlp, "moe_mlp"):
        mm = lyr.mlp.moe_mlp
        for attr in ("w_gate_up", "w_down", "w_gate_up_scale", "w_down_scale"):
          v = getattr(mm, attr, None)
          if v is not None and hasattr(v, "set_value"):
            v.set_value(None)
    import gc
    gc.collect()

  return (
      jax.tree_util.tree_unflatten(treedef, stacked_leaves),
      jax.tree_util.tree_unflatten(treedef, stacked_specs),
  )


def make_jitted_forward(
    model: Any,
    *,
    sample_lens: tuple[int, ...],
    mesh: Any = None,
    has_time: bool = True,
    embedded_input: bool = False,
    use_splash: bool | None = None,
    moe_mode: str | None = None,
    capacity_factor: float | None = None,
    scan_layers: bool = True,
    release_model_moe: bool = False,
):
  """Compile MAGI-2 Preview forward with ``nnx.split``/``nnx.merge`` inside ``jax.jit``.

  When ``scan_layers=True`` (default), contiguous identical layer runs of length >= 3
  (all 36 MoE layers ``2..37`` in the 40-layer 113.93B Preview DiT) are stacked
  device-locally along axis 0 and executed via ``jax.lax.scan`` inside ``shard_map``,
  compiling only **1** MoE layer body instead of 36 (~10-20x faster compile!).

  Set ``release_model_moe=True`` on the full 113.93B checkpoint to drop the
  unstacked ``model.layers[2..37]`` MoE references as they are stacked so steady-state
  routed MoE HBM remains 1x (25.31 GiB/chip BF16 or 12.66 GiB/chip FP8).

  Returns ``(jitted_fn, state)`` where calling:
    ``jitted_fn(state, x, rope_or_coords, modality_mapping, time_token_sequence)``
  runs the full 40-layer Ulysses CP + EP=8 DiT in a single XLA executable.
  """
  from flax import nnx
  from jax.sharding import PartitionSpec as P

  seg_lens = tuple(int(v) for v in sample_lens)
  ax = model.axis_name
  cp_size = int(dict(mesh.shape).get(ax, 1)) if mesh is not None else 1
  layers = list(model.layers)

  # Use scanned path whenever there is a contiguous identical run of >= 3 layers.
  def _sig(layer):
    return (bool(layer.is_mm), bool(layer.is_moe))

  segments: list[tuple[bool, Any, list[int]]] = []
  has_scan = False
  if scan_layers and len(layers) >= 4:
    idx = 0
    while idx < len(layers):
      sig = _sig(layers[idx])
      j = idx + 1
      while j < len(layers) and _sig(layers[j]) == sig:
        j += 1
      run = list(range(idx, j))
      if len(run) >= 3:
        segments.append((True, nnx.split(layers[idx])[0], run))
        has_scan = True
      else:
        for k in run:
          segments.append((False, nnx.split(layers[k])[0], [k]))
      idx = j

  if not has_scan:
    graphdef, state = nnx.split(model)
    if mesh is not None and cp_size > 1:
      state_specs = build_sharded_state_specs(state, axis_name=ax)

      def _body(st, x_l, rope_l, mod_l, time_l):
        mdl = nnx.merge(graphdef, st)
        call = mdl.forward_embedded_local if embedded_input else mdl.forward_local
        return call(
            x_l,
            rope_l,
            mod_l,
            sample_lens=seg_lens,
            time_token_sequence_local=time_l,
            cp_size=cp_size,
            use_splash=use_splash,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )

      sharded = jax.shard_map(
          _body,
          mesh=mesh,
          in_specs=(
              state_specs,
              P(ax, None),
              P(ax, None),
              P(ax,),
              P(ax, None) if has_time else None,
          ),
          out_specs=P(ax, None),
          check_vma=False,
      )
      return jax.jit(sharded), state

    @jax.jit
    def _single(st, x_l, rope_l, mod_l, time_l):
      mdl = nnx.merge(graphdef, st)
      call = mdl.forward_embedded_local if embedded_input else mdl.forward_local
      return call(
          x_l,
          rope_l,
          mod_l,
          sample_lens=seg_lens,
          time_token_sequence_local=time_l,
          cp_size=1,
          use_splash=use_splash,
          moe_mode=moe_mode,
          capacity_factor=capacity_factor,
      )

    return _single, state

  # Scanned path: split pre_adapter, segments, post_adapter cleanly.
  pre_gd, pre_st = nnx.split(model.pre_adapter)
  post_gd, post_st = nnx.split(model.post_adapter)
  pre_specs = build_sharded_state_specs(pre_st, axis_name=ax)
  post_specs = build_sharded_state_specs(post_st, axis_name=ax)

  seg_states = []
  seg_specs = []
  for is_scan, _lgd, run in segments:
    if is_scan:
      st_list = [nnx.split(layers[k])[1] for k in run]
      rel = [layers[k] for k in run] if release_model_moe else None
      s_st, s_sp = _stack_preview_layer_states(st_list, mesh, ax, release_layers=rel)
      seg_states.append(s_st)
      seg_specs.append(s_sp)
    else:
      one_st = nnx.split(layers[run[0]])[1]
      seg_states.append(one_st)
      seg_specs.append(build_sharded_state_specs(one_st, axis_name=ax))

  packed_state = (pre_st, tuple(seg_states), post_st)
  packed_specs = (pre_specs, tuple(seg_specs), post_specs)
  c = model.config
  m_dtype = model.dtype
  splash_flag = _resolve_use_splash(use_splash)

  def _run_local(st, x_l, rope_l, mod_l, time_l, *, _cp: int):
    pre_s, layer_ss, post_s = st
    mod_ids = jnp.where(
        mod_l == MODALITY_TIME,
        jnp.int32(MODALITY_TEXT),
        mod_l.astype(jnp.int32),
    )
    if embedded_input:
      x_emb = x_l
    else:
      x_emb = nnx.merge(pre_gd, pre_s)(x_l, mod_ids)

    if rope_l.shape[-1] == 9:
      rope_96 = _elementwise_fourier_embed_jax(
          rope_l, head_dim=c.head_dim, temperature=c.rope_temperature
      )
    else:
      rope_96 = rope_l.astype(jnp.float32)

    x = x_emb.astype(jnp.float32)
    if time_l is not None and time_l.shape[-1] > 0:
      t_dim = time_l.shape[-1]
      x = x.at[:, :t_dim].set(time_l.astype(x.dtype))
    x = x.astype(m_dtype)

    perm_local = jnp.argsort(mod_ids, stable=True)
    inv_perm_local = jnp.empty_like(perm_local).at[perm_local].set(
        jnp.arange(perm_local.shape[0], dtype=perm_local.dtype)
    )
    mod_ids_sorted = mod_ids[perm_local]
    group_sizes = jnp.bincount(mod_ids_sorted, length=3).astype(jnp.int32)
    x_sorted = x[perm_local]

    for (is_scan, lgd, _run), lst in zip(segments, layer_ss):
      if is_scan:
        def _step(carry, one_st, _lgd=lgd):
          lyr = nnx.merge(_lgd, one_st)
          nxt = lyr(
              carry,
              rope_96,
              perm_local,
              inv_perm_local,
              mod_ids_sorted,
              group_sizes,
              sample_lens=seg_lens,
              cp_size=_cp,
              axis_name=ax,
              use_splash=splash_flag,
              moe_mode=moe_mode,
              capacity_factor=capacity_factor,
          )
          return nxt, None

        x_sorted, _ = jax.lax.scan(_step, x_sorted, lst)
      else:
        lyr = nnx.merge(lgd, lst)
        x_sorted = lyr(
            x_sorted,
            rope_96,
            perm_local,
            inv_perm_local,
            mod_ids_sorted,
            group_sizes,
            sample_lens=seg_lens,
            cp_size=_cp,
            axis_name=ax,
            use_splash=splash_flag,
            moe_mode=moe_mode,
            capacity_factor=capacity_factor,
        )

    x_seq = x_sorted[inv_perm_local]
    return nnx.merge(post_gd, post_s)(x_seq, mod_ids)

  if mesh is not None and cp_size > 1:
    sharded = jax.shard_map(
        lambda st, xl, rl, ml, tl: _run_local(st, xl, rl, ml, tl, _cp=cp_size),
        mesh=mesh,
        in_specs=(
            packed_specs,
            P(ax, None),
            P(ax, None),
            P(ax,),
            P(ax, None) if has_time else None,
        ),
        out_specs=P(ax, None),
        check_vma=False,
    )
    return jax.jit(sharded), packed_state

  @jax.jit
  def _single_scanned(st, x_l, rope_l, mod_l, time_l):
    return _run_local(st, x_l, rope_l, mod_l, time_l, _cp=1)

  return _single_scanned, packed_state


# ---------------------------------------------------------------------------
# Streaming Safetensors Checkpoint Loader (host RAM < 15 GiB for 212 GiB ckpt)
# ---------------------------------------------------------------------------


_LAYER_RE = re.compile(r"^block\.layers\.(\d+)\.(.+)$")


def _read_weight_map(checkpoint_dir: str | Path) -> dict[str, str]:
  """Read HF sharded ``model.safetensors.index.json`` or single ``model.safetensors``."""
  from ...safetensors_io import safe_open

  ckpt = Path(checkpoint_dir)
  index_path = ckpt / _INDEX_NAME
  if index_path.exists():
    wm = json.loads(index_path.read_text())["weight_map"]
    return {k: str(ckpt / shard) for k, shard in wm.items()}
  single_path = ckpt / _SINGLE_NAME
  if not single_path.exists():
    raise FileNotFoundError(f"No {_INDEX_NAME} or {_SINGLE_NAME} in {ckpt}")
  with safe_open(str(single_path), framework="np") as fh:
    return {k: str(single_path) for k in fh.keys()}


def _put_replicated(arr_np: np.ndarray, mesh: Any, target_dtype: Any) -> Any:
  from jax.sharding import NamedSharding, PartitionSpec as P

  dt = jnp.dtype(target_dtype)
  if arr_np.dtype != np.dtype(dt):
    arr_np = arr_np.astype(dt)
  arr_np = np.ascontiguousarray(arr_np)
  if mesh is None:
    return jnp.asarray(arr_np, dtype=dt)
  return jax.device_put(arr_np, NamedSharding(mesh, P()))


def _convert_grouped_linear_np(
    w_pt: np.ndarray,
    num_modality: int,
    in_features: int,
    out_features: int,
) -> np.ndarray:
  """Convert PyTorch ``CustomGroupedLinear.weight`` ``(M*out, in)`` to JAX layout."""
  expected = (num_modality * out_features, in_features)
  if tuple(w_pt.shape) != expected:
    raise ValueError(f"Expected GroupedLinear shape {expected}, got {tuple(w_pt.shape)}")
  if num_modality == 1:
    return np.ascontiguousarray(w_pt.T)
  return np.ascontiguousarray(
      np.transpose(w_pt.reshape(num_modality, out_features, in_features), (0, 2, 1))
  )


def _convert_norm_np(w_pt: np.ndarray, num_modality: int, dim: int) -> np.ndarray:
  expected = (num_modality * dim,)
  if tuple(w_pt.shape) != expected:
    raise ValueError(f"Expected RMSNorm shape {expected}, got {tuple(w_pt.shape)}")
  if num_modality == 1:
    return np.ascontiguousarray(w_pt)
  return np.ascontiguousarray(w_pt.reshape(num_modality, dim))


def load_magi2_preview_from_safetensors(
    checkpoint_dir: str | Path,
    *,
    config: Magi2PreviewConfig | None = None,
    mesh: Any = None,
    axis_name: str = "tp",
    dtype: Any = None,
    fp8: bool | None = None,
    moe_mode: str | None = None,
    capacity_factor: float | None = None,
    verbose: Any = print,
) -> Any:
  """Stream-load Stage 1 MAGI-2 Preview (212.44 GiB, 56 shards) straight onto TPU.

  Uses ``nnx.eval_shape`` so zero unsharded weights are ever allocated on device,
  opens each safetensors shard sequentially via ``diffusers_tpu.safetensors_io.safe_open``,
  slices ``W_gate`` / ``W_up`` / ``W_down`` directly into ``(ep, H, E_local, ...)``
  per chip (optionally TPU-JIT quantizing to E4M3 FP8), and swaps
  ``moe_mlp.router.expert_bias_ema -> expert_bias`` matching ``magi2_checkpointing.py``.
  """
  from flax import nnx
  from ...safetensors_io import safe_open

  t0 = time.perf_counter()
  cfg = config or Magi2PreviewConfig()
  dt = dtype if dtype is not None else jnp.bfloat16
  use_fp8 = use_fp8_env() if fp8 is None else bool(fp8)
  ep_size = int(dict(mesh.shape).get(axis_name, 1)) if mesh is not None else 1

  bias_src = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
  use_ema_bias = bias_src != "main"

  *_, Magi2PreviewTransformer = _build_modules()
  model = nnx.eval_shape(
      lambda: Magi2PreviewTransformer(
          cfg,
          ep_size=ep_size,
          axis_name=axis_name,
          dtype=dt,
          fp8=use_fp8,
          moe_mode=moe_mode,
          capacity_factor=capacity_factor,
      )
  )

  weight_map = _read_weight_map(checkpoint_dir)
  by_shard: dict[str, list[str]] = defaultdict(list)
  for key, shard_path in weight_map.items():
    by_shard[shard_path].append(key)

  # Staging dict holds at most one layer's W_gate until its paired W_up arrives.
  pending_gate_up: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
  ema_loaded = 0
  skipped_keys: list[str] = []
  unknown_keys: list[str] = []

  def _assign_linear(mod: Any, arr_np: np.ndarray) -> None:
    conv = _convert_grouped_linear_np(
        arr_np, mod.num_modality, mod.in_features, mod.out_features
    )
    mod.weight.value = _put_replicated(conv, mesh, dt)

  def _assign_one(key: str, arr_np: np.ndarray) -> None:
    nonlocal ema_loaded

    # Top-level adapters.
    if key == "pre_adapter.rope.bands":
      skipped_keys.append(key)
      return
    if key == "pre_adapter.video_embedder.weight":
      model.pre_adapter.video_embedder_weight.value = _put_replicated(arr_np.T, mesh, jnp.float32)
      return
    if key == "pre_adapter.video_embedder.bias":
      model.pre_adapter.video_embedder_bias.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if key == "pre_adapter.audio_embedder.weight":
      model.pre_adapter.audio_embedder_weight.value = _put_replicated(arr_np.T, mesh, jnp.float32)
      return
    if key == "pre_adapter.audio_embedder.bias":
      model.pre_adapter.audio_embedder_bias.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if key == "pre_adapter.text_embedder.weight":
      model.pre_adapter.text_embedder_weight.value = _put_replicated(arr_np.T, mesh, jnp.float32)
      return
    if key == "pre_adapter.text_embedder.bias":
      model.pre_adapter.text_embedder_bias.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if key == "post_adapter.final_norm_video.weight":
      model.post_adapter.final_norm_video_weight.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if key == "post_adapter.final_norm_audio.weight":
      model.post_adapter.final_norm_audio_weight.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if key == "post_adapter.final_linear_video.weight":
      model.post_adapter.final_linear_video_weight.value = _put_replicated(
          arr_np.T, mesh, jnp.float32
      )
      return
    if key == "post_adapter.final_linear_audio.weight":
      model.post_adapter.final_linear_audio_weight.value = _put_replicated(
          arr_np.T, mesh, jnp.float32
      )
      return

    m = _LAYER_RE.match(key)
    if not m:
      unknown_keys.append(key)
      return
    layer_idx = int(m.group(1))
    sub = m.group(2)
    layer = model.layers[layer_idx]
    num_mod = layer.num_modality

    # Attention weights.
    if sub == "attention.pre_norm.weight":
      layer.attention.pre_norm_weight.value = _put_replicated(
          _convert_norm_np(arr_np, num_mod, cfg.hidden_size), mesh, jnp.float32
      )
      return
    if sub == "attention.linear_g.weight":
      _assign_linear(layer.attention.linear_g, arr_np)
      return
    if sub == "attention.linear_qkv.weight":
      _assign_linear(layer.attention.linear_qkv, arr_np)
      return
    if sub == "attention.linear_proj.weight":
      _assign_linear(layer.attention.linear_proj, arr_np)
      return
    if sub == "attention.sinks":
      layer.attention.sinks.value = _put_replicated(arr_np, mesh, jnp.float32)
      return
    if sub == "attention.q_norm.weight":
      layer.attention.q_norm_weight.value = _put_replicated(
          _convert_norm_np(arr_np, num_mod, cfg.head_dim), mesh, jnp.float32
      )
      return
    if sub == "attention.k_norm.weight":
      layer.attention.k_norm_weight.value = _put_replicated(
          _convert_norm_np(arr_np, num_mod, cfg.head_dim), mesh, jnp.float32
      )
      return

    # mHC weights (all fp32).
    if sub == "mhc_norm.weight":
      layer.mhc_norm_weight.value = _put_replicated(
          _convert_norm_np(arr_np, num_mod, cfg.stream_hidden_size), mesh, jnp.float32
      )
      return
    if sub in (
        "mhc_phi_fused_attn",
        "mhc_phi_fused_mlp",
        "mhc_alpha_pre_attn",
        "mhc_alpha_post_attn",
        "mhc_alpha_res_attn",
        "mhc_alpha_pre_mlp",
        "mhc_alpha_post_mlp",
        "mhc_alpha_res_mlp",
        "mhc_bias_pre_attn",
        "mhc_bias_post_attn",
        "mhc_bias_pre_mlp",
        "mhc_bias_post_mlp",
        "mhc_bias_res_attn",
        "mhc_bias_res_mlp",
    ):
      getattr(layer, sub).value = _put_replicated(arr_np, mesh, jnp.float32)
      return

    # Dense MLP (MM layers).
    if not layer.is_moe:
      if sub == "mlp.pre_norm.weight":
        layer.mlp.pre_norm_weight.value = _put_replicated(
            _convert_norm_np(arr_np, layer.mlp.num_modality, cfg.hidden_size),
            mesh,
            jnp.float32,
        )
        return
      if sub == "mlp.up_gate_proj.weight":
        _assign_linear(layer.mlp.up_gate_proj, arr_np)
        return
      if sub == "mlp.down_proj.weight":
        _assign_linear(layer.mlp.down_proj, arr_np)
        return
      unknown_keys.append(key)
      return

    # MultiHeadMoELayer (MoE layers 2..37).
    mlp = layer.mlp
    if sub == "mlp.pre_norm.weight":
      # ALWAYS 3 modalities on MoE layers!
      mlp.pre_norm_weight.value = _put_replicated(
          _convert_norm_np(arr_np, 3, cfg.hidden_size), mesh, jnp.float32
      )
      return
    if sub == "mlp.split_linear.weight":
      _assign_linear(mlp.split_linear, arr_np)
      return
    if sub == "mlp.merge_linear.weight":
      _assign_linear(mlp.merge_linear, arr_np)
      return
    if sub == "mlp.shared_expert_fc1.weight":
      _assign_linear(mlp.shared_expert_fc1, arr_np)
      return
    if sub == "mlp.shared_expert_fc2.weight":
      _assign_linear(mlp.shared_expert_fc2, arr_np)
      return
    if sub == "mlp.modality_specific_shared_expert_fc1.weight":
      _assign_linear(mlp.modality_specific_shared_expert_fc1, arr_np)
      return
    if sub == "mlp.modality_specific_shared_expert_fc2.weight":
      _assign_linear(mlp.modality_specific_shared_expert_fc2, arr_np)
      return

    moe = mlp.moe_mlp
    if sub == "mlp.moe_mlp.gate":
      gate_r = arr_np.reshape(cfg.moe_num_heads, cfg.moe_num_experts, cfg.moe_head_dim)
      moe.gate.value = _put_replicated(gate_r, mesh, jnp.float32)
      return
    if sub == "mlp.moe_mlp.router.expert_bias_ema":
      if use_ema_bias:
        bias_r = arr_np.reshape(cfg.moe_num_heads, cfg.moe_num_experts)
        moe.expert_bias.value = _put_replicated(bias_r, mesh, jnp.float32)
        ema_loaded += 1
      else:
        skipped_keys.append(key)
      return
    if sub == "mlp.moe_mlp.router.expert_bias":
      if not use_ema_bias:
        bias_r = arr_np.reshape(cfg.moe_num_heads, cfg.moe_num_experts)
        moe.expert_bias.value = _put_replicated(bias_r, mesh, jnp.float32)
      else:
        skipped_keys.append(key)
      return
    if sub in ("mlp.moe_mlp.W_gate", "mlp.moe_mlp.W_up"):
      slot = "gate" if sub.endswith("W_gate") else "up"
      pair = None
      with _state_lock:
        pending_gate_up[layer_idx][slot] = arr_np
        if "gate" in pending_gate_up[layer_idx] and "up" in pending_gate_up[layer_idx]:
          pair = pending_gate_up.pop(layer_idx)
      if pair is not None:
        w_gu, w_gu_sc = shard_routed_gate_up(
            pair["gate"],
            pair["up"],
            num_heads=cfg.moe_num_heads,
            num_experts=cfg.moe_num_experts,
            mesh=mesh,
            axis_name=axis_name,
            fp8=use_fp8,
            dtype=dt,
        )
        del pair
        moe.w_gate_up.value = w_gu
        if use_fp8:
          moe.w_gate_up_scale.value = w_gu_sc
      return
    if sub == "mlp.moe_mlp.W_down":
      w_dn, w_dn_sc = shard_routed_down(
          arr_np,
          num_heads=cfg.moe_num_heads,
          num_experts=cfg.moe_num_experts,
          mesh=mesh,
          axis_name=axis_name,
          fp8=use_fp8,
          dtype=dt,
      )
      moe.w_down.value = w_dn
      if use_fp8:
        moe.w_down_scale.value = w_dn_sc
      return

    with _state_lock:
      unknown_keys.append(key)

  import concurrent.futures
  import threading

  _state_lock = threading.Lock()
  shard_items = sorted(by_shard.items())
  total_shards = len(shard_items)
  n_workers = max(1, min(int(os.environ.get("DT_MAGI2_LOAD_WORKERS", "8")), total_shards))
  done_shards = 0

  def _load_one_shard(item: tuple[str, list[str]]) -> None:
    nonlocal done_shards
    shard_path, keys = item
    wanted = set(keys)
    with safe_open(shard_path, framework="np") as fh:
      for key in fh.keys():
        if key not in wanted:
          continue
        arr_np = fh.get_tensor(key)
        _assign_one(key, arr_np)
        del arr_np
    with _state_lock:
      done_shards += 1
      if verbose and (done_shards % 8 == 0 or done_shards == total_shards):
        verbose(
            f"  [magi2-preview] loaded shard {done_shards}/{total_shards} "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

  if n_workers == 1:
    for item in shard_items:
      _load_one_shard(item)
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
      futs = [pool.submit(_load_one_shard, item) for item in shard_items]
      for fut in concurrent.futures.as_completed(futs):
        fut.result()

  if pending_gate_up:
    raise RuntimeError(
        f"Incomplete W_gate/W_up pairs left after reading all shards: "
        f"{ {k: list(v.keys()) for k, v in pending_gate_up.items()} }"
    )

  # Verify every nnx.Param was materialized (no leftover ShapeDtypeStruct).
  unloaded = []
  for path, leaf in nnx.to_flat_state(nnx.state(model, nnx.Param)):
    if isinstance(leaf.value, jax.ShapeDtypeStruct):
      unloaded.append(".".join(str(p) for p in path))
  if unloaded:
    raise RuntimeError(
        f"{len(unloaded)} MAGI-2 Preview parameter(s) were not loaded from "
        f"{checkpoint_dir}: {unloaded[:10]}"
    )

  if verbose:
    bias_msg = (
        f"expert_bias_ema -> expert_bias ({ema_loaded} layers)"
        if use_ema_bias
        else "expert_bias (raw, MAGI2_ROUTER_BIAS_SOURCE=main)"
    )
    fp8_msg = "FP8-W8A8" if use_fp8 else str(jnp.dtype(dt))
    verbose(
        f"  [magi2-preview] loaded {cfg.num_layers} layers ({fp8_msg}, EP/CP={ep_size}), "
        f"router bias: {bias_msg} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    if unknown_keys:
      verbose(
          f"  [magi2-preview] WARNING: ignored {len(unknown_keys)} unexpected key(s): "
          f"{unknown_keys[:5]}",
          flush=True,
      )

  return model


# ---------------------------------------------------------------------------
# Self-Test: Tiny Config Full-Model Parity vs Exact PyTorch Reference Logic
# ---------------------------------------------------------------------------


def _np_rms_norm_1pw(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  x64 = x.astype(np.float64)
  rrms = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)
  return (x64 * rrms * (1.0 + w.astype(np.float64))).astype(np.float32)


def _np_mm_rms_norm(
    x: np.ndarray, w: np.ndarray, mod_ids: np.ndarray, num_mod: int, eps: float = 1e-6
) -> np.ndarray:
  if num_mod == 1:
    return _np_rms_norm_1pw(x, w, eps=eps)
  w_m = w.reshape(num_mod, x.shape[-1])
  out = np.empty_like(x, dtype=np.float32)
  for m in range(num_mod):
    mask = mod_ids == m
    if np.any(mask):
      out[mask] = _np_rms_norm_1pw(x[mask], w_m[m], eps=eps)
  return out


def _np_grouped_linear(
    x: np.ndarray, w_pt: np.ndarray, mod_ids: np.ndarray, num_mod: int
) -> np.ndarray:
  in_f = x.shape[-1]
  out_f = w_pt.shape[0] // num_mod
  w = w_pt.reshape(num_mod, out_f, in_f).astype(np.float64)
  if num_mod == 1:
    return (x.astype(np.float64) @ w[0].T).astype(np.float32)
  out = np.empty((x.shape[0], out_f), dtype=np.float32)
  for m in range(num_mod):
    mask = mod_ids == m
    if np.any(mask):
      out[mask] = (x[mask].astype(np.float64) @ w[m].T).astype(np.float32)
  return out


def _np_swiglu7_interleaved(x: np.ndarray) -> np.ndarray:
  g = np.minimum(x[..., ::2].astype(np.float64), 7.0)
  u = np.clip(x[..., 1::2].astype(np.float64), -7.0, 7.0)
  return ((g / (1.0 + np.exp(-1.702 * g))) * (u + 1.0)).astype(np.float32)


def _np_sinkhorn(h: np.ndarray, iters: int = 20, eps: float = 1e-12) -> np.ndarray:
  h64 = h.astype(np.float64)
  h_max = np.max(h64, axis=(-2, -1), keepdims=True)
  m = np.exp(h64 - h_max)
  for _ in range(iters):
    m = m / (np.sum(m, axis=-2, keepdims=True) + eps)  # col first
    m = m / (np.sum(m, axis=-1, keepdims=True) + eps)  # row second
  return m.astype(np.float32)


def _np_preview_reference_forward(
    state_pt: dict[str, np.ndarray],
    cfg: Magi2PreviewConfig,
    x: np.ndarray,
    coords_9: np.ndarray,
    modality_mapping: np.ndarray,
    sample_lens: tuple[int, ...],
    time_token_sequence: np.ndarray,
) -> np.ndarray:
  """Exact pure-NumPy mirror of ``inference/model/magi2_preview.py``."""
  from .jax_mh_moe import _numpy_ref_mh_moe

  mod = modality_mapping.copy()
  mod[mod == MODALITY_TIME] = MODALITY_TEXT
  s_total = x.shape[0]
  ad = cfg.stream_hidden_size
  n = cfg.mhc_num_stream
  d = cfg.hidden_size
  mscale = 1.0 / math.sqrt(float(ad))

  # PreAdapter.
  rope = elementwise_fourier_embed(
      coords_9, head_dim=cfg.head_dim, temperature=cfg.rope_temperature
  )
  h = np.zeros((s_total, ad), dtype=np.float32)
  for m_id, name, cin in (
      (MODALITY_VIDEO, "video_embedder", cfg.video_in_channels),
      (MODALITY_AUDIO, "audio_embedder", cfg.audio_in_channels),
      (MODALITY_TEXT, "text_embedder", cfg.text_in_channels),
  ):
    mask = mod == m_id
    if np.any(mask):
      w = state_pt[f"pre_adapter.{name}.weight"].astype(np.float64)
      b = state_pt[f"pre_adapter.{name}.bias"].astype(np.float64)
      h[mask] = (x[mask, :cin].astype(np.float64) @ w.T + b).astype(np.float32)

  # Stream-0 timestep overwrite.
  h[:, : time_token_sequence.shape[-1]] = time_token_sequence.astype(np.float32)

  # Modality permutation.
  perm = np.argsort(mod, kind="stable")
  inv_perm = np.argsort(perm)
  mod_s = mod[perm]
  h_s = h[perm]

  for i in range(cfg.num_layers):
    pfx = f"block.layers.{i}"
    num_mod = 3 if i in cfg.mm_layers else 1
    s_streams = h_s.reshape(s_total, n, d)

    # mHC pre-attn.
    s_norm = _np_mm_rms_norm(h_s, state_pt[f"{pfx}.mhc_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps)
    hf = (s_norm.astype(np.float64) @ state_pt[f"{pfx}.mhc_phi_fused_attn"].astype(np.float64)).astype(np.float32)
    hp_raw, hpost_raw, hres_raw = hf[:, :n], hf[:, n : 2 * n], hf[:, 2 * n :].reshape(s_total, n, n)
    h_pre = 1.0 / (
        1.0 + np.exp(-(state_pt[f"{pfx}.mhc_alpha_pre_attn"] * mscale * hp_raw + state_pt[f"{pfx}.mhc_bias_pre_attn"]))
    )
    attn_in = np.einsum("tn,tnc->tc", h_pre, s_streams).astype(np.float32)

    # Attention.
    ax = _np_mm_rms_norm(attn_in, state_pt[f"{pfx}.attention.pre_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps)
    g = _np_grouped_linear(ax, state_pt[f"{pfx}.attention.linear_g.weight"], mod_s, num_mod).reshape(s_total, cfg.num_heads_q, 1)
    qkv = _np_grouped_linear(ax, state_pt[f"{pfx}.attention.linear_qkv.weight"], mod_s, num_mod)
    qs = cfg.num_heads_q * cfg.head_dim
    kvs = cfg.num_heads_kv * cfg.head_dim
    q = qkv[:, :qs].reshape(s_total, cfg.num_heads_q, cfg.head_dim)
    k = qkv[:, qs : qs + kvs].reshape(s_total, cfg.num_heads_kv, cfg.head_dim)
    v = qkv[:, qs + kvs :].reshape(s_total, cfg.num_heads_kv, cfg.head_dim)
    q = _np_mm_rms_norm(q, state_pt[f"{pfx}.attention.q_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps)
    k = _np_mm_rms_norm(k, state_pt[f"{pfx}.attention.k_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps)

    q, k, v = q[inv_perm], k[inv_perm], v[inv_perm]
    q = apply_3d_partial_rope(q, rope)
    k = apply_3d_partial_rope(k, rope)

    sinks = state_pt[f"{pfx}.attention.sinks"]  # (1, Hq)
    out_seq = np.zeros_like(q, dtype=np.float32)
    off = 0
    for seg_len in sample_lens:
      if seg_len <= 0:
        continue
      end = off + seg_len
      q_s = np.transpose(q[off:end], (1, 0, 2)).astype(np.float64)
      k_s = np.transpose(k[off:end], (1, 0, 2)).astype(np.float64)
      v_s = np.transpose(v[off:end], (1, 0, 2)).astype(np.float64)
      logits = np.einsum("hqd,hkd->hqk", q_s, k_s) * (cfg.head_dim ** -0.5)
      m_max = np.max(logits, axis=-1, keepdims=True)
      lse = m_max + np.log(np.sum(np.exp(logits - m_max), axis=-1, keepdims=True))
      probs = np.exp(logits - lse)
      o_std = np.einsum("hqk,hkd->hqd", probs, v_s)
      sink_sig = 1.0 / (1.0 + np.exp(-(lse - sinks[0, :, None, None].astype(np.float64))))
      out_seq[off:end] = np.transpose((o_std * sink_sig).astype(np.float32), (1, 0, 2))
      off = end

    out_s = out_seq[perm] * (1.0 / (1.0 + np.exp(-g.astype(np.float64)))).astype(np.float32)
    attn_out = _np_grouped_linear(
        out_s.reshape(s_total, qs), state_pt[f"{pfx}.attention.linear_proj.weight"], mod_s, num_mod
    )

    # mHC post-attn.
    h_post = 2.0 / (
        1.0 + np.exp(-(state_pt[f"{pfx}.mhc_alpha_post_attn"] * mscale * hpost_raw + state_pt[f"{pfx}.mhc_bias_post_attn"]))
    )
    h_res = _np_sinkhorn(
        state_pt[f"{pfx}.mhc_alpha_res_attn"] * mscale * hres_raw + state_pt[f"{pfx}.mhc_bias_res_attn"],
        iters=cfg.mhc_sinkhorn_iters,
        eps=cfg.mhc_sinkhorn_eps,
    )
    s_mid = (
        np.einsum("tij,tjc->tic", h_res, s_streams)
        + np.einsum("tn,tc->tnc", h_post, attn_out)
    ).astype(np.float32)

    # mHC pre-mlp.
    s_mid_norm = _np_mm_rms_norm(
        s_mid.reshape(s_total, ad), state_pt[f"{pfx}.mhc_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps
    )
    hf_m = (s_mid_norm.astype(np.float64) @ state_pt[f"{pfx}.mhc_phi_fused_mlp"].astype(np.float64)).astype(np.float32)
    hp_m, hpost_m, hres_m = hf_m[:, :n], hf_m[:, n : 2 * n], hf_m[:, 2 * n :].reshape(s_total, n, n)
    h_pre_m = 1.0 / (
        1.0 + np.exp(-(state_pt[f"{pfx}.mhc_alpha_pre_mlp"] * mscale * hp_m + state_pt[f"{pfx}.mhc_bias_pre_mlp"]))
    )
    mlp_in = np.einsum("tn,tnc->tc", h_pre_m, s_mid).astype(np.float32)

    # MLP or MultiHeadMoELayer.
    if i in cfg.moe_layers:
      mn = _np_mm_rms_norm(mlp_in, state_pt[f"{pfx}.mlp.pre_norm.weight"], mod_s, 3, cfg.rms_norm_eps)
      moe_in = _np_grouped_linear(mn, state_pt[f"{pfx}.mlp.split_linear.weight"], mod_s, 1)
      # CRITICAL: uses expert_bias_ema!
      moe_core = _numpy_ref_mh_moe(
          moe_in,
          state_pt[f"{pfx}.mlp.moe_mlp.gate"],
          state_pt[f"{pfx}.mlp.moe_mlp.router.expert_bias_ema"],
          state_pt[f"{pfx}.mlp.moe_mlp.W_gate"],
          state_pt[f"{pfx}.mlp.moe_mlp.W_up"],
          state_pt[f"{pfx}.mlp.moe_mlp.W_down"],
          num_heads=cfg.moe_num_heads,
          num_experts=cfg.moe_num_experts,
          top_k=cfg.moe_top_k,
          route_norm=cfg.moe_route_norm,
          route_scale=cfg.moe_route_scale,
      )
      moe_out = _np_grouped_linear(moe_core, state_pt[f"{pfx}.mlp.merge_linear.weight"], mod_s, 1)
      sh1 = _np_swiglu7_interleaved(_np_grouped_linear(mn, state_pt[f"{pfx}.mlp.shared_expert_fc1.weight"], mod_s, 1))
      sh1 = _np_grouped_linear(sh1, state_pt[f"{pfx}.mlp.shared_expert_fc2.weight"], mod_s, 1)
      sh2 = _np_swiglu7_interleaved(
          _np_grouped_linear(mn, state_pt[f"{pfx}.mlp.modality_specific_shared_expert_fc1.weight"], mod_s, 3)
      )
      sh2 = _np_grouped_linear(sh2, state_pt[f"{pfx}.mlp.modality_specific_shared_expert_fc2.weight"], mod_s, 3)
      mlp_out = moe_out + sh1 + sh2
    else:
      mn = _np_mm_rms_norm(mlp_in, state_pt[f"{pfx}.mlp.pre_norm.weight"], mod_s, num_mod, cfg.rms_norm_eps)
      ug = _np_grouped_linear(mn, state_pt[f"{pfx}.mlp.up_gate_proj.weight"], mod_s, num_mod)
      mlp_out = _np_grouped_linear(
          _np_swiglu7_interleaved(ug), state_pt[f"{pfx}.mlp.down_proj.weight"], mod_s, num_mod
      )

    # mHC post-mlp.
    h_post_m = 2.0 / (
        1.0 + np.exp(-(state_pt[f"{pfx}.mhc_alpha_post_mlp"] * mscale * hpost_m + state_pt[f"{pfx}.mhc_bias_post_mlp"]))
    )
    h_res_m = _np_sinkhorn(
        state_pt[f"{pfx}.mhc_alpha_res_mlp"] * mscale * hres_m + state_pt[f"{pfx}.mhc_bias_res_mlp"],
        iters=cfg.mhc_sinkhorn_iters,
        eps=cfg.mhc_sinkhorn_eps,
    )
    h_s = (
        np.einsum("tij,tjc->tic", h_res_m, s_mid)
        + np.einsum("tn,tc->tnc", h_post_m, mlp_out)
    ).reshape(s_total, ad).astype(np.float32)

  # Inv-permute & PostAdapter.
  h_out = h_s[inv_perm]
  out_dim = max(cfg.video_in_channels, cfg.audio_in_channels)
  y = np.zeros((s_total, out_dim), dtype=np.float32)
  v_mask = mod == MODALITY_VIDEO
  if np.any(v_mask):
    vn = _np_rms_norm_1pw(h_out[v_mask], state_pt["post_adapter.final_norm_video.weight"], cfg.rms_norm_eps)
    y[v_mask, : cfg.video_in_channels] = (
        vn.astype(np.float64) @ state_pt["post_adapter.final_linear_video.weight"].astype(np.float64).T
    ).astype(np.float32)
  a_mask = mod == MODALITY_AUDIO
  if np.any(a_mask):
    an = _np_rms_norm_1pw(h_out[a_mask], state_pt["post_adapter.final_norm_audio.weight"], cfg.rms_norm_eps)
    y[a_mask, : cfg.audio_in_channels] = (
        an.astype(np.float64) @ state_pt["post_adapter.final_linear_audio.weight"].astype(np.float64).T
    ).astype(np.float32)
  return y


def _make_tiny_checkpoint_dict(
    cfg: Magi2PreviewConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
  """Synthesize a complete checkpoint state_dict with exact HF safetensors keys."""
  sd: dict[str, np.ndarray] = {}
  ad = cfg.stream_hidden_size
  n = cfg.mhc_num_stream
  d = cfg.hidden_size

  def r32(shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)

  sd["pre_adapter.video_embedder.weight"] = r32((ad, cfg.video_in_channels))
  sd["pre_adapter.video_embedder.bias"] = r32((ad,))
  sd["pre_adapter.audio_embedder.weight"] = r32((ad, cfg.audio_in_channels))
  sd["pre_adapter.audio_embedder.bias"] = r32((ad,))
  sd["pre_adapter.text_embedder.weight"] = r32((ad, cfg.text_in_channels))
  sd["pre_adapter.text_embedder.bias"] = r32((ad,))
  sd["pre_adapter.rope.bands"] = np.float32(1.0) / (
      np.float32(cfg.rope_temperature) ** (np.arange(cfg.head_dim // 8, dtype=np.float32) / (cfg.head_dim // 8))
  )
  sd["post_adapter.final_norm_video.weight"] = r32((ad,))
  sd["post_adapter.final_norm_audio.weight"] = r32((ad,))
  sd["post_adapter.final_linear_video.weight"] = r32((cfg.video_in_channels, ad))
  sd["post_adapter.final_linear_audio.weight"] = r32((cfg.audio_in_channels, ad))

  qs = cfg.num_heads_q * cfg.head_dim
  kvs = cfg.num_heads_kv * cfg.head_dim
  for i in range(cfg.num_layers):
    pfx = f"block.layers.{i}"
    m = 3 if i in cfg.mm_layers else 1
    sd[f"{pfx}.attention.pre_norm.weight"] = r32((m * d,))
    sd[f"{pfx}.attention.linear_g.weight"] = r32((m * cfg.num_heads_q, d))
    sd[f"{pfx}.attention.linear_qkv.weight"] = r32((m * (qs + 2 * kvs), d))
    sd[f"{pfx}.attention.linear_proj.weight"] = r32((m * d, qs))
    sd[f"{pfx}.attention.sinks"] = r32((max(1, cfg.sink_token_num), cfg.num_heads_q), 0.3)
    sd[f"{pfx}.attention.q_norm.weight"] = r32((m * cfg.head_dim,))
    sd[f"{pfx}.attention.k_norm.weight"] = r32((m * cfg.head_dim,))

    sd[f"{pfx}.mhc_norm.weight"] = r32((m * ad,))
    for sub in ("attn", "mlp"):
      sd[f"{pfx}.mhc_phi_fused_{sub}"] = r32((ad, 2 * n + n * n))
      sd[f"{pfx}.mhc_alpha_pre_{sub}"] = np.array([cfg.mhc_alpha_init], np.float32)
      sd[f"{pfx}.mhc_alpha_post_{sub}"] = np.array([cfg.mhc_alpha_init], np.float32)
      sd[f"{pfx}.mhc_alpha_res_{sub}"] = np.array([cfg.mhc_alpha_init], np.float32)
      sd[f"{pfx}.mhc_bias_pre_{sub}"] = r32((n,))
      sd[f"{pfx}.mhc_bias_post_{sub}"] = r32((n,))
      sd[f"{pfx}.mhc_bias_res_{sub}"] = r32((n, n))

    if i in cfg.moe_layers:
      sd[f"{pfx}.mlp.pre_norm.weight"] = r32((3 * d,))
      sd[f"{pfx}.mlp.split_linear.weight"] = r32((d, d))
      sd[f"{pfx}.mlp.merge_linear.weight"] = r32((d, d))
      sd[f"{pfx}.mlp.shared_expert_fc1.weight"] = r32((2 * cfg.moe_shared_intermediate_size, d))
      sd[f"{pfx}.mlp.shared_expert_fc2.weight"] = r32((d, cfg.moe_shared_intermediate_size))
      sd[f"{pfx}.mlp.modality_specific_shared_expert_fc1.weight"] = r32(
          (3 * 2 * cfg.moe_modality_shared_intermediate_size, d)
      )
      sd[f"{pfx}.mlp.modality_specific_shared_expert_fc2.weight"] = r32(
          (3 * d, cfg.moe_modality_shared_intermediate_size)
      )
      te = cfg.moe_total_experts
      sd[f"{pfx}.mlp.moe_mlp.gate"] = r32((te, cfg.moe_head_dim), 0.2)
      # Deliberately make raw expert_bias wildly different from expert_bias_ema
      # to verify the loader strictly picks expert_bias_ema!
      sd[f"{pfx}.mlp.moe_mlp.router.expert_bias"] = np.full((te,), 10.0, np.float32)
      sd[f"{pfx}.mlp.moe_mlp.router.expert_bias_ema"] = r32((te,), 0.05)
      sd[f"{pfx}.mlp.moe_mlp.W_gate"] = r32((te, cfg.moe_head_dim, cfg.moe_expert_intermediate_size))
      sd[f"{pfx}.mlp.moe_mlp.W_up"] = r32((te, cfg.moe_head_dim, cfg.moe_expert_intermediate_size))
      sd[f"{pfx}.mlp.moe_mlp.W_down"] = r32((te, cfg.moe_expert_intermediate_size, cfg.moe_head_dim))
    else:
      sd[f"{pfx}.mlp.pre_norm.weight"] = r32((m * d,))
      sd[f"{pfx}.mlp.up_gate_proj.weight"] = r32((m * 2 * cfg.dense_intermediate_size, d))
      sd[f"{pfx}.mlp.down_proj.weight"] = r32((m * d, cfg.dense_intermediate_size))
  return sd


def _write_minimal_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
  """Write a standard safetensors file with pure Python/NumPy (no extra deps)."""
  import struct

  header: dict[str, Any] = {}
  chunks: list[bytes] = []
  offset = 0
  for name, arr in tensors.items():
    arr_c = np.ascontiguousarray(arr)
    if arr_c.dtype == np.float32:
      dt_str = "F32"
    elif arr_c.dtype.name == "bfloat16":
      dt_str = "BF16"
    else:
      raise ValueError(f"Unsupported test dtype {arr_c.dtype}")
    raw = arr_c.tobytes()
    header[name] = {
        "dtype": dt_str,
        "shape": list(arr_c.shape),
        "data_offsets": [offset, offset + len(raw)],
    }
    chunks.append(raw)
    offset += len(raw)
  hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr_bytes)))
    f.write(hdr_bytes)
    for c in chunks:
      f.write(c)


def test_tiny_parity(verbose: bool = True) -> bool:
  """Verify tiny 4-layer Preview DiT (2 MM + 2 MoE) + sink + loader against reference."""
  import tempfile
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  rng = np.random.default_rng(123)
  cfg = Magi2PreviewConfig(
      num_layers=4,
      hidden_size=64,
      head_dim=16,
      num_heads_q=4,
      num_heads_kv=4,
      video_in_channels=8,
      audio_in_channels=12,
      text_in_channels=24,
      dense_intermediate_size=32,
      mm_layers=(0, 3),
      moe_layers=(1, 2),
      mhc_num_stream=4,
      moe_num_heads=4,
      moe_num_experts=8,
      moe_top_k=2,
      moe_expert_intermediate_size=16,
      moe_shared_intermediate_size=16,
      moe_modality_shared_intermediate_size=16,
      time_channel_dim=16,
  )

  # 1. Verify base-2 residual_attention sink equivalence against natural-log formula.
  q_t = rng.standard_normal((1, 4, 10, 16)).astype(np.float32)
  k_t = rng.standard_normal((1, 4, 10, 16)).astype(np.float32)
  v_t = rng.standard_normal((1, 4, 10, 16)).astype(np.float32)
  sinks_t = rng.standard_normal((1, 4)).astype(np.float32) * 2.0
  logits_b2 = np.einsum("bhqd,bhkd->bhqk", q_t, k_t) * (16 ** -0.5) * _LOG2_E
  m_b2 = np.max(logits_b2, axis=-1, keepdims=True)
  w_b2 = np.exp2(logits_b2 - m_b2)
  l_b2 = np.sum(w_b2, axis=-1, keepdims=True)
  o_un = np.einsum("bhqk,bhkd->bhqd", w_b2, v_t)
  got_sink = np.asarray(
      apply_residual_attention_sink(
          jnp.asarray(o_un), jnp.asarray(m_b2), jnp.asarray(l_b2), jnp.asarray(sinks_t)
      )
  )
  want_sink = np.asarray(
      _sdpa_with_sink_single_segment(
          jnp.asarray(q_t), jnp.asarray(k_t), jnp.asarray(v_t), jnp.asarray(sinks_t), use_splash=False
      )
  )
  sink_err = float(np.max(np.abs(got_sink - want_sink)))
  ok = sink_err < 1e-5
  if verbose:
    print(f"[jax_preview_dit] base-2 sink formula max|diff| = {sink_err:.3e} ({'PASS' if ok else 'FAIL'})")

  # 2. Build packed B=2 CFG [cond (10 tok), uncond (12 tok), pad (2 tok)] = 24 tokens.
  # Cond: video (2,2,2)=8? Wait: use (2,1,2)=4 video + 2 audio + 4 text = 10;
  # Uncond: (2,1,2)=4 video + 2 audio + 6 text = 12; pad = 2 -> total = 24 (divisible by 2, 4, 8).
  coords_cond = build_preview_rope_coords((2, 1, 2), audio_len=2, txt_len=4)
  coords_uncond = build_preview_rope_coords((2, 1, 2), audio_len=2, txt_len=6)
  coords_pad = np.tile(np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1]], np.float32), (2, 1))
  coords_all = np.concatenate([coords_cond, coords_uncond, coords_pad], axis=0)

  mod_cond = np.array([0] * 4 + [1] * 2 + [2] * 4, dtype=np.int32)
  mod_uncond = np.array([0] * 4 + [1] * 2 + [2] * 6, dtype=np.int32)
  mod_pad = np.array([2, 2], dtype=np.int32)
  mod_all = np.concatenate([mod_cond, mod_uncond, mod_pad], axis=0)
  sample_lens = (len(mod_cond), len(mod_uncond))

  x_np = rng.standard_normal((len(mod_all), cfg.text_in_channels)).astype(np.float32) * 0.5
  raw_t_np = np.where((mod_all == MODALITY_VIDEO) | (mod_all == MODALITY_AUDIO), 0.75, 0.0).astype(
      np.float32
  )
  time_seq_np = sinusoidal_embedding_1d(raw_t_np, dim=cfg.time_channel_dim)
  rope_np = elementwise_fourier_embed(
      coords_all, head_dim=cfg.head_dim, temperature=cfg.rope_temperature
  )

  state_pt = _make_tiny_checkpoint_dict(cfg, rng)
  want_out = _np_preview_reference_forward(
      state_pt, cfg, x_np, coords_all, mod_all, sample_lens, time_seq_np
  )

  # 3. Write 2-shard safetensors checkpoint & test streaming loader + single-device JIT parity.
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    keys = sorted(state_pt.keys())
    half = len(keys) // 2
    s1 = {k: state_pt[k] for k in keys[:half]}
    s2 = {k: state_pt[k] for k in keys[half:]}
    _write_minimal_safetensors(tmp / "model-00001-of-00002.safetensors", s1)
    _write_minimal_safetensors(tmp / "model-00002-of-00002.safetensors", s2)
    index = {
        "metadata": {},
        "weight_map": {
            **{k: "model-00001-of-00002.safetensors" for k in s1},
            **{k: "model-00002-of-00002.safetensors" for k in s2},
        },
    }
    (tmp / _INDEX_NAME).write_text(json.dumps(index))

    model = load_magi2_preview_from_safetensors(
        tmp, config=cfg, mesh=None, dtype=jnp.float32, fp8=False, verbose=None
    )
    jit_fn, state = make_jitted_forward(
        model, sample_lens=sample_lens, mesh=None, has_time=True, use_splash=False
    )
    got_out = np.asarray(
        jit_fn(
            state,
            jnp.asarray(x_np),
            jnp.asarray(rope_np),
            jnp.asarray(mod_all),
            jnp.asarray(time_seq_np),
        )
    )

    real_len = sum(sample_lens)
    err_real = float(np.max(np.abs(want_out[:real_len] - got_out[:real_len])))
    txt_max = float(np.max(np.abs(got_out[mod_all == MODALITY_TEXT])))
    pass_1dev = err_real < 2e-4 and txt_max == 0.0
    ok &= pass_1dev
    if verbose:
      print(
          f"[jax_preview_dit] 1-dev fp32 max|diff| = {err_real:.3e}, "
          f"text_out_max = {txt_max:.1e} ({'PASS' if pass_1dev else 'FAIL'})"
      )

    # 4. Multi-device Ulysses CP + EP test when >= 2 devices exist.
    devs = jax.devices()
    cp_cands = [
        d
        for d in (len(devs), 4, 2)
        if d <= len(devs)
        and cfg.num_heads_q % d == 0
        and cfg.moe_num_experts % d == 0
        and len(mod_all) % d == 0
    ]
    if cp_cands and cp_cands[0] > 1:
      cp = cp_cands[0]
      mesh = Mesh(np.array(devs[:cp]), ("tp",))
      for mode in ("ag", "ep"):
        m_cp = load_magi2_preview_from_safetensors(
            tmp,
            config=cfg,
            mesh=mesh,
            dtype=jnp.float32,
            fp8=False,
            moe_mode=mode,
            capacity_factor=float(cp),
            verbose=None,
        )
        jit_cp, st_cp = make_jitted_forward(
            m_cp,
            sample_lens=sample_lens,
            mesh=mesh,
            has_time=True,
            use_splash=False,
            moe_mode=mode,
            capacity_factor=float(cp),
        )
        x_sh = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh, P("tp", None)))
        rope_sh = jax.device_put(jnp.asarray(rope_np), NamedSharding(mesh, P("tp", None)))
        mod_sh = jax.device_put(jnp.asarray(mod_all), NamedSharding(mesh, P("tp",)))
        t_sh = jax.device_put(jnp.asarray(time_seq_np), NamedSharding(mesh, P("tp", None)))
        got_cp = np.asarray(jit_cp(st_cp, x_sh, rope_sh, mod_sh, t_sh))
        err_cp = float(np.max(np.abs(want_out[:real_len] - got_cp[:real_len])))
        pass_cp = err_cp < 2e-4
        ok &= pass_cp
        if verbose:
          print(
              f"[jax_preview_dit] cp/ep={cp} mode={mode} fp32 max|diff| = {err_cp:.3e} "
              f"({'PASS' if pass_cp else 'FAIL'})"
          )

  return ok


if __name__ == "__main__":
  passed = test_tiny_parity(verbose=True)
  raise SystemExit(0 if passed else 1)
