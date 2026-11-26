"""Pure-PyTorch ``MAGI-2-preview`` 113.93B Multi-Head Latent MoE DiT.

Designed to execute identically across three modes via
:class:`diffusers_tpu.models.magi2.upstream_compat.Magi2TpuHooks`:

1. **Priority 2 (`torchax`)**: Traced into JAX/XLA via ``torchax`` with
   ``jax.shard_map`` + ``jax.lax.scan`` over layers 2..37, delegating mHC
   (Tokamax Pallas + global-max Sinkhorn), Attention+Sink (in-tree Pallas
   Splash), GroupedLinear (``jax.lax.ragged_dot``), and MH-MoE (zero-cumsum
   sliced-AG + MegaCore Dual-TC ``gmm_v2``) through ``env.t2j_iso``/``j2t_iso``.
2. **Priority 3 (`torch_tpu`)**: Executed on 8 TPU processes via ``torchrun``
   and ``torch.compile(backend="tpu", fullgraph=True)`` (``AotBlocks``),
   using ``torch.ops._c10d_functional`` for Ulysses CP=8 and EP=8 collectives
   and single-chip ``torch.tpu.pallas.jax_op`` kernels for mHC, Splash+Sink,
   ``GroupedLinear``, and local-32-expert Dual-TC ``gmm_v2``.
3. **Pure-PyTorch CPU/Reference**: Works with zero hooks installed for exact
   numerical verification against :mod:`jax_preview_dit`.

All mathematical formulas match :mod:`jax_preview_dit` and upstream
``SandAI-org/MAGI-2-preview`` exactly:
- Gemma ``(1 + w)`` RMSNorm everywhere (``cast_back=False`` on ``q_norm``,
  ``k_norm``, ``mhc_norm``, and ``post_adapter.final_norm_*``).
- Sinusoidal 64-d timestep injection ``[cos, sin]`` directly overwriting stream
  0 ``x[:, :64]`` in fp32 right after ``PreAdapter`` (no AdaLN).
- 3D spatially-centered partial RoPE (96/128 dims rotated in fp32, split-half).
- 4-stream mHC with **global-max** Sinkhorn-Knopp (20 iters, col-then-row).
- SwiGLU7 (upper-clamp ``g <= 7.0``, clamp ``u in [-7.0, 7.0]``,
  ``g * sigmoid(1.702 * g) * (u + 1.0)``): interleaved on dense/shared MLPs,
  split-half on routed MoE experts.
- Unbiased sigmoid Top-6 routing with ``expert_bias_ema`` selection bias,
  L1 normalization, and ``route_scale = 4.9``.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MODALITY_AUDIO,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    Magi2PreviewConfig,
)
from .upstream_compat import (
    Magi2TpuHooks,
    get_tpu_hooks,
    load_safetensors_weight_map,
)


# ---------------------------------------------------------------------------
# Pure-PyTorch Exact Reference Primitives
# ---------------------------------------------------------------------------


def rms_norm_1pw(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    cast_back: bool = True,
) -> torch.Tensor:
  """Gemma-style ``(1 + w)`` RMSNorm matching ``jax_preview_dit``."""
  x_f32 = x.float()
  inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
  y = x_f32 * inv_rms * (1.0 + weight.float())
  return y.to(x.dtype) if cast_back else y


def mm_rms_norm_1pw(
    x: torch.Tensor,
    weight: torch.Tensor,
    mod_ids: torch.Tensor | None = None,
    eps: float = 1e-6,
    cast_back: bool = True,
) -> torch.Tensor:
  """Multi-modality ``(1 + w)`` RMSNorm.

  Args:
    x: ``(S, D)`` or ``(S, H, D)``.
    weight: ``(D,)`` when ``M == 1``, or ``(M, D)`` when ``M > 1``.
    mod_ids: ``(S,)`` int32/int64 modality IDs in ``[0, M)``.
  """
  if weight.ndim == 1:
    return rms_norm_1pw(x, weight, eps=eps, cast_back=cast_back)
  assert mod_ids is not None
  w_sel = weight.index_select(0, mod_ids)
  if x.ndim == 3:
    w_sel = w_sel.unsqueeze(1)
  return rms_norm_1pw(x, w_sel, eps=eps, cast_back=cast_back)


def swiglu7_interleaved(x: torch.Tensor) -> torch.Tensor:
  """Interleaved SwiGLU7 used by dense MLPs and shared experts."""
  g = x[..., 0::2].float().clamp(max=7.0)
  u = x[..., 1::2].float().clamp(min=-7.0, max=7.0)
  return (g * torch.sigmoid(1.702 * g) * (u + 1.0)).to(x.dtype)


def swiglu7_split(gu: torch.Tensor) -> torch.Tensor:
  """Split-half SwiGLU7 used by routed MoE ``[W_gate, W_up]``."""
  g, u = gu.chunk(2, dim=-1)
  g = g.float().clamp(max=7.0)
  u = u.float().clamp(min=-7.0, max=7.0)
  return (g * torch.sigmoid(1.702 * g) * (u + 1.0)).to(gu.dtype)


def sinusoidal_embedding_1d_torch(
    raw_t: torch.Tensor,
    dim: int = 64,
    max_period: float = 10000.0,
) -> torch.Tensor:
  """Elementwise sinusoidal timestep embedding (``[cos, sin]``, fp32)."""
  half = dim // 2
  freqs = torch.exp(
      -math.log(max_period)
      * torch.arange(half, device=raw_t.device, dtype=torch.float32)
      / float(half)
  )
  args = (raw_t.float() * 1000.0).unsqueeze(-1) * freqs
  return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


def compute_3d_rope_96_torch(
    coords_9: torch.Tensor,
    temperature: float = 10000.0,
) -> torch.Tensor:
  """Compute packed ``(S, 96)`` fp32 ``[sin_t, sin_h, sin_w, cos_t, cos_h, cos_w]``."""
  c = coords_9.float()
  xyz = c[:, 0:3]
  sizes = c[:, 3:6]
  refs = c[:, 6:9]

  ones_mask = (refs == 1.0) & (sizes == 1.0)
  denom = torch.where(sizes == 1.0, torch.ones_like(sizes), sizes - 1.0)
  scales = torch.where(ones_mask, torch.ones_like(sizes), (refs - 1.0) / denom)

  centers = (sizes - 1.0) * 0.5
  # Do NOT center time (dim 0); center only spatial (h, w).
  t_zero = torch.zeros_like(centers[:, 0:1])
  centers = torch.cat((t_zero, centers[:, 1:3]), dim=-1)

  bands = 1.0 / (
      temperature
      ** (torch.arange(16, device=c.device, dtype=torch.float32) / 16.0)
  )
  proj = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * bands[None, None, :]
  sin_p = torch.sin(proj).reshape(c.shape[0], 48)
  cos_p = torch.cos(proj).reshape(c.shape[0], 48)
  return torch.cat((sin_p, cos_p), dim=-1)


def apply_3d_partial_rope_torch(
    x: torch.Tensor,
    rope_96: torch.Tensor,
) -> torch.Tensor:
  """Apply split-half 3D partial RoPE in fp32 to ``x[..., :96]``."""
  sin_48 = rope_96[:, :48].float().unsqueeze(1)
  cos_48 = rope_96[:, 48:96].float().unsqueeze(1)
  x_f32 = x.float()
  x1 = x_f32[..., :48]
  x2 = x_f32[..., 48:96]
  x_pass = x_f32[..., 96:]
  out1 = x1 * cos_48 - x2 * sin_48
  out2 = x2 * cos_48 + x1 * sin_48
  return torch.cat((out1, out2, x_pass), dim=-1)


def sinkhorn_knopp_global_max_torch(
    h_res_logits: torch.Tensor,
    iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
  """Exact MAGI-2 global-max Sinkhorn-Knopp (col-then-row, fp32)."""
  m = torch.exp(h_res_logits - h_res_logits.amax(dim=(-2, -1), keepdim=True))
  for _ in range(iters):
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    m = m / (m.sum(dim=-1, keepdim=True) + eps)
  return m


def mhc_pre_ref_torch(
    s_flat: torch.Tensor,
    norm_w: torch.Tensor,
    phi_fused: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    b_pre: torch.Tensor,
    b_post: torch.Tensor,
    b_res: torch.Tensor,
    mod_ids: torch.Tensor | None = None,
    *,
    num_streams: int = 4,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-12,
    rms_eps: float = 1e-6,
    fp32_matmul_fn: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pure-PyTorch 4-stream mHC pre-mixes + collapse."""
  S, flat_dim = s_flat.shape
  hidden_dim = flat_dim // num_streams
  scale = 1.0 / math.sqrt(float(flat_dim))

  s_norm = mm_rms_norm_1pw(s_flat, norm_w, mod_ids=mod_ids, eps=rms_eps, cast_back=False)
  if fp32_matmul_fn is not None:
    h_fused = fp32_matmul_fn(s_norm, phi_fused.float())
  else:
    h_fused = torch.matmul(s_norm, phi_fused.float())
  h_pre_raw = h_fused[:, :num_streams]
  h_post_raw = h_fused[:, num_streams : 2 * num_streams]
  h_res_raw = h_fused[:, 2 * num_streams :].reshape(S, num_streams, num_streams)

  h_pre = torch.sigmoid(alpha_pre.float() * scale * h_pre_raw + b_pre.float()).to(s_flat.dtype)
  h_post = (2.0 * torch.sigmoid(alpha_post.float() * scale * h_post_raw + b_post.float())).to(
      s_flat.dtype
  )
  h_res_logits = alpha_res.float() * scale * h_res_raw + b_res.float()
  h_res = sinkhorn_knopp_global_max_torch(
      h_res_logits, iters=sinkhorn_iters, eps=sinkhorn_eps
  ).to(s_flat.dtype)

  s_streams = s_flat.reshape(S, num_streams, hidden_dim).float()
  x_in = (h_pre.float().unsqueeze(-1) * s_streams).sum(dim=1).to(s_flat.dtype)
  return x_in, h_post, h_res


def mhc_post_ref_torch(
    s_flat: torch.Tensor,
    sub_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    *,
    num_streams: int = 4,
) -> torch.Tensor:
  """Pure-PyTorch 4-stream mHC residual combine."""
  S, flat_dim = s_flat.shape
  hidden_dim = flat_dim // num_streams
  s_streams = s_flat.reshape(S, num_streams, hidden_dim).float()
  s_next = torch.matmul(h_res.float(), s_streams) + h_post.float().unsqueeze(-1) * sub_out.float().unsqueeze(1)
  return s_next.to(s_flat.dtype).reshape(S, flat_dim)


def sdpa_with_sink_ref_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    sample_lens: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
  """Pure-PyTorch exact fp32 varlen attention with 1-token sink logit."""
  # q, k, v: (S_total, H_loc, D); sinks: (1, H_loc)
  S_total, H_loc, D = q.shape
  out_parts: list[torch.Tensor] = []
  offset = 0
  for slen in sample_lens:
    q_s = q[offset : offset + slen].float().transpose(0, 1)  # (H, slen, D)
    k_s = k[offset : offset + slen].float().transpose(0, 1)
    v_s = v[offset : offset + slen].float().transpose(0, 1)
    logits = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale  # (H, slen, slen)
    sink_col = sinks.float().transpose(0, 1).unsqueeze(-1).expand(H_loc, slen, 1)
    logits_aug = torch.cat((logits, sink_col), dim=-1)
    probs = torch.softmax(logits_aug, dim=-1)[..., :-1]
    o_s = torch.matmul(probs, v_s).transpose(0, 1).to(q.dtype)
    out_parts.append(o_s)
    offset += slen
  if offset < S_total:
    out_parts.append(torch.zeros((S_total - offset, H_loc, D), device=q.device, dtype=q.dtype))
  return torch.cat(out_parts, dim=0)


# ---------------------------------------------------------------------------
# PyTorch Module Building Blocks
# ---------------------------------------------------------------------------


class MultiModalityRMSNormTorch(nn.Module):
  def __init__(
      self,
      dim: int,
      num_modality: int = 1,
      eps: float = 1e-6,
      cast_back: bool = True,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_modality = num_modality
    self.eps = eps
    self.cast_back = cast_back
    shape = (dim,) if num_modality == 1 else (num_modality, dim)
    self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)

  def forward(self, x: torch.Tensor, mod_ids: torch.Tensor | None = None) -> torch.Tensor:
    return mm_rms_norm_1pw(
        x,
        self.weight,
        mod_ids=mod_ids if self.num_modality > 1 else None,
        eps=self.eps,
        cast_back=self.cast_back,
    )


class GroupedLinearTorch(nn.Module):
  """Modality-grouped linear projection matching ``jax_preview_dit.GroupedLinear``.

  - When ``num_modality == 1``: standard ``(out_features, in_features)`` weight
    dispatched with ``F.linear``.
  - When ``num_modality > 1``: ``(num_modality, in_features, out_features)``
    weight matching ``jax.lax.ragged_dot`` layout directly. Delegates to
    ``hooks.grouped_linear`` when installed, else runs Dynamo-safe masked matmul.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      num_modality: int = 1,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_modality = num_modality
    if num_modality == 1:
      self.weight = nn.Parameter(
          torch.empty((out_features, in_features), dtype=dtype),
          requires_grad=False,
      )
    else:
      self.weight = nn.Parameter(
          torch.empty((num_modality, in_features, out_features), dtype=dtype),
          requires_grad=False,
      )

  def forward(
      self,
      x: torch.Tensor,
      group_sizes: torch.Tensor | None = None,
      mod_ids: torch.Tensor | None = None,
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    if self.num_modality == 1:
      return F.linear(x, self.weight)
    h = hooks if hooks is not None else get_tpu_hooks()
    if h.grouped_linear is not None and group_sizes is not None:
      return h.grouped_linear(x, self.weight, group_sizes, mod_ids)
    assert mod_ids is not None
    out = torch.zeros((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
    for m in range(self.num_modality):
      mask = (mod_ids == m).unsqueeze(-1)
      out = torch.where(mask, torch.matmul(x, self.weight[m]), out)
    return out


class Magi2PreAdapterTorch(nn.Module):
  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    ad = config.stream_hidden_size
    self.video_embedder = nn.Linear(config.video_in_channels, ad, bias=True, dtype=torch.float32)
    self.audio_embedder = nn.Linear(config.audio_in_channels, ad, bias=True, dtype=torch.float32)
    self.text_embedder = nn.Linear(config.text_in_channels, ad, bias=True, dtype=torch.float32)
    for p in self.parameters():
      p.requires_grad_(False)

  def forward(self, x: torch.Tensor, mod_ids: torch.Tensor) -> torch.Tensor:
    # x: (S_loc, 5120) fp32 (video in [:48], audio in [:64], text in [:5120])
    x_f32 = x.float()
    v_emb = self.video_embedder(x_f32[:, : self.video_embedder.in_features])
    a_emb = self.audio_embedder(x_f32[:, : self.audio_embedder.in_features])
    t_emb = self.text_embedder(x_f32[:, : self.text_embedder.in_features])
    is_v = (mod_ids == MODALITY_VIDEO).unsqueeze(-1)
    is_a = (mod_ids == MODALITY_AUDIO).unsqueeze(-1)
    return torch.where(is_v, v_emb, torch.where(is_a, a_emb, t_emb))


class Magi2PostAdapterTorch(nn.Module):
  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    ad = config.stream_hidden_size
    self.v_ch = config.video_in_channels
    self.a_ch = config.audio_in_channels
    self.out_ch = max(self.v_ch, self.a_ch)
    self.final_norm_video = MultiModalityRMSNormTorch(ad, 1, eps=config.rms_norm_eps, cast_back=False)
    self.final_norm_audio = MultiModalityRMSNormTorch(ad, 1, eps=config.rms_norm_eps, cast_back=False)
    self.final_linear_video = nn.Linear(ad, self.v_ch, bias=False, dtype=torch.float32)
    self.final_linear_audio = nn.Linear(ad, self.a_ch, bias=False, dtype=torch.float32)
    for p in self.parameters():
      p.requires_grad_(False)

  def forward(
      self,
      x: torch.Tensor,
      mod_ids: torch.Tensor,
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    h = hooks if hooks is not None else get_tpu_hooks()
    v_n = self.final_norm_video(x)
    a_n = self.final_norm_audio(x)
    if h.fp32_matmul_highest is not None:
      v_out = h.fp32_matmul_highest(v_n, self.final_linear_video.weight.t())
      a_out = h.fp32_matmul_highest(a_n, self.final_linear_audio.weight.t())
    else:
      v_out = self.final_linear_video(v_n)
      a_out = self.final_linear_audio(a_n)
    if self.v_ch < self.out_ch:
      v_pad = torch.zeros(
          (v_out.shape[0], self.out_ch - self.v_ch),
          device=v_out.device,
          dtype=v_out.dtype,
      )
      v_out = torch.cat((v_out, v_pad), dim=-1)
    if self.a_ch < self.out_ch:
      a_pad = torch.zeros(
          (a_out.shape[0], self.out_ch - self.a_ch),
          device=a_out.device,
          dtype=a_out.dtype,
      )
      a_out = torch.cat((a_out, a_pad), dim=-1)
    zeros = torch.zeros_like(v_out)
    is_v = (mod_ids == MODALITY_VIDEO).unsqueeze(-1)
    is_a = (mod_ids == MODALITY_AUDIO).unsqueeze(-1)
    return torch.where(is_v, v_out, torch.where(is_a, a_out, zeros))


class Magi2PreviewAttentionTorch(nn.Module):
  def __init__(
      self,
      config: Magi2PreviewConfig,
      num_modality: int,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.num_heads_q = config.num_heads_q
    self.num_heads_kv = config.num_heads_kv
    self.head_dim = config.head_dim
    self.q_size = self.num_heads_q * self.head_dim
    self.kv_size = self.num_heads_kv * self.head_dim
    self.scale = float(self.head_dim) ** -0.5

    self.pre_norm = MultiModalityRMSNormTorch(
        config.hidden_size, num_modality, eps=config.rms_norm_eps, cast_back=True
    )
    self.linear_g = GroupedLinearTorch(
        config.hidden_size, self.num_heads_q, num_modality, dtype=dtype
    )
    self.linear_qkv = GroupedLinearTorch(
        config.hidden_size,
        self.q_size + 2 * self.kv_size,
        num_modality,
        dtype=dtype,
    )
    self.linear_proj = GroupedLinearTorch(
        self.q_size, config.hidden_size, num_modality, dtype=dtype
    )
    self.q_norm = MultiModalityRMSNormTorch(
        self.head_dim, num_modality, eps=config.rms_norm_eps, cast_back=False
    )
    self.k_norm = MultiModalityRMSNormTorch(
        self.head_dim, num_modality, eps=config.rms_norm_eps, cast_back=False
    )
    self.sinks = nn.Parameter(
        torch.zeros((config.sink_token_num, self.num_heads_q), dtype=torch.float32),
        requires_grad=False,
    )

  def forward(
      self,
      x_sorted: torch.Tensor,
      rope_96: torch.Tensor,
      mod_ids_sorted: torch.Tensor,
      group_sizes: torch.Tensor,
      perm_local: torch.Tensor,
      inv_perm_local: torch.Tensor,
      sample_lens: tuple[int, ...],
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    h = hooks if hooks is not None else get_tpu_hooks()
    S_loc = x_sorted.shape[0]
    act_dtype = x_sorted.dtype

    x_norm = self.pre_norm(x_sorted, mod_ids_sorted)
    g = self.linear_g(x_norm, group_sizes, mod_ids_sorted, hooks=h).reshape(
        S_loc, self.num_heads_q, 1
    )
    qkv = self.linear_qkv(x_norm, group_sizes, mod_ids_sorted, hooks=h)
    q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
    q = q.reshape(S_loc, self.num_heads_q, self.head_dim)
    k = k.reshape(S_loc, self.num_heads_kv, self.head_dim)
    v = v.reshape(S_loc, self.num_heads_kv, self.head_dim)

    q = self.q_norm(q, mod_ids_sorted)  # fp32
    k = self.k_norm(k, mod_ids_sorted)  # fp32

    # Unpermute back to original packed token order for RoPE + Ulysses + Attention.
    q = q.index_select(0, inv_perm_local)
    k = k.index_select(0, inv_perm_local)
    v = v.index_select(0, inv_perm_local)

    q = apply_3d_partial_rope_torch(q, rope_96).to(act_dtype)
    k = apply_3d_partial_rope_torch(k, rope_96).to(act_dtype)

    if h.ulysses_scatter_head_gather_seq is not None:
      q = h.ulysses_scatter_head_gather_seq(q)
      k = h.ulysses_scatter_head_gather_seq(k)
      v = h.ulysses_scatter_head_gather_seq(v)

    if h.attn_with_sink is not None:
      out = h.attn_with_sink(q, k, v, self.sinks, sample_lens, self.scale)
    else:
      if h.ulysses_scatter_head_gather_seq is not None:
        h_loc = q.shape[1]
        sinks_loc = self.sinks[:, h.ep_rank * h_loc : (h.ep_rank + 1) * h_loc]
      else:
        sinks_loc = self.sinks
      out = sdpa_with_sink_ref_torch(q, k, v, sinks_loc, sample_lens, self.scale)

    if h.ulysses_scatter_seq_gather_head is not None:
      out = h.ulysses_scatter_seq_gather_head(out)

    # Re-permute to modality-sorted order for gating + linear_proj.
    out = out.index_select(0, perm_local)
    out = (out.float() * torch.sigmoid(g.float())).to(act_dtype).reshape(S_loc, self.q_size)
    return self.linear_proj(out, group_sizes, mod_ids_sorted, hooks=h)


class Magi2DenseMLPTorch(nn.Module):
  def __init__(
      self,
      config: Magi2PreviewConfig,
      num_modality: int = 3,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.pre_norm = MultiModalityRMSNormTorch(
        config.hidden_size, num_modality, eps=config.rms_norm_eps, cast_back=True
    )
    self.up_gate_proj = GroupedLinearTorch(
        config.hidden_size,
        2 * config.dense_intermediate_size,
        num_modality,
        dtype=dtype,
    )
    self.down_proj = GroupedLinearTorch(
        config.dense_intermediate_size,
        config.hidden_size,
        num_modality,
        dtype=dtype,
    )

  def forward(
      self,
      x_sorted: torch.Tensor,
      mod_ids_sorted: torch.Tensor,
      group_sizes: torch.Tensor,
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    h = hooks if hooks is not None else get_tpu_hooks()
    x_norm = self.pre_norm(x_sorted, mod_ids_sorted)
    gu = self.up_gate_proj(x_norm, group_sizes, mod_ids_sorted, hooks=h)
    act = swiglu7_interleaved(gu)
    return self.down_proj(act, group_sizes, mod_ids_sorted, hooks=h)


class Magi2MultiHeadMoELayerTorch(nn.Module):
  def __init__(
      self,
      config: Magi2PreviewConfig,
      *,
      ep_size: int = 1,
      dtype: torch.dtype = torch.bfloat16,
      fp8_moe: bool = False,
  ) -> None:
    super().__init__()
    self.num_heads = config.moe_num_heads
    self.num_experts = config.moe_num_experts
    self.ep_size = ep_size
    self.e_local = self.num_experts // ep_size
    self.top_k = config.moe_top_k
    self.route_scale = float(config.moe_route_scale)
    self.head_dim = config.moe_head_dim
    self.expert_inter = config.moe_expert_intermediate_size
    self.fp8_moe = fp8_moe

    self.pre_norm = MultiModalityRMSNormTorch(
        config.hidden_size, 3, eps=config.rms_norm_eps, cast_back=True
    )
    self.split_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
    self.merge_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
    self.shared_expert_fc1 = nn.Linear(
        config.hidden_size, 2 * config.moe_shared_intermediate_size, bias=False, dtype=dtype
    )
    self.shared_expert_fc2 = nn.Linear(
        config.moe_shared_intermediate_size, config.hidden_size, bias=False, dtype=dtype
    )
    self.modality_specific_shared_expert_fc1 = GroupedLinearTorch(
        config.hidden_size, 2 * config.moe_modality_shared_intermediate_size, 3, dtype=dtype
    )
    self.modality_specific_shared_expert_fc2 = GroupedLinearTorch(
        config.moe_modality_shared_intermediate_size, config.hidden_size, 3, dtype=dtype
    )

    self.gate = nn.Parameter(
        torch.empty((self.num_heads, self.num_experts, self.head_dim), dtype=torch.float32),
        requires_grad=False,
    )
    self.expert_bias = nn.Parameter(
        torch.zeros((self.num_heads, self.num_experts), dtype=torch.float32),
        requires_grad=False,
    )

    moe_w_dtype = torch.float8_e4m3fn if fp8_moe else dtype
    self.w_gate_up = nn.Parameter(
        torch.empty(
            (self.num_heads, self.e_local, self.head_dim, 2 * self.expert_inter),
            dtype=moe_w_dtype,
        ),
        requires_grad=False,
    )
    self.w_down = nn.Parameter(
        torch.empty(
            (self.num_heads, self.e_local, self.expert_inter, self.head_dim),
            dtype=moe_w_dtype,
        ),
        requires_grad=False,
    )
    if fp8_moe:
      self.w_gate_up_scale = nn.Parameter(
          torch.ones((self.num_heads, self.e_local, 1, 1, 2 * self.expert_inter), dtype=torch.float32),
          requires_grad=False,
      )
      self.w_down_scale = nn.Parameter(
          torch.ones((self.num_heads, self.e_local, 1, 1, self.head_dim), dtype=torch.float32),
          requires_grad=False,
      )
    else:
      self.register_parameter("w_gate_up_scale", None)
      self.register_parameter("w_down_scale", None)

    for p in self.parameters():
      p.requires_grad_(False)

  def _route_local(
      self, x_heads: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # x_heads: (S_loc, H, d_h) -> idx (H, S_loc, K) int32, probs (H, S_loc, K) fp32
    logits = torch.einsum("shd,hed->hse", x_heads.float(), self.gate.float())
    scores = torch.sigmoid(logits)
    biased = scores + self.expert_bias.float().unsqueeze(1)
    _, top_idx = torch.topk(biased, k=self.top_k, dim=-1, largest=True, sorted=True)
    probs = torch.gather(scores, dim=-1, index=top_idx)
    denom = probs.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    probs = (probs / denom) * self.route_scale
    return top_idx.to(torch.int32), probs

  def _routed_ref_single_head(
      self,
      x_h: torch.Tensor,
      idx_h: torch.Tensor,
      probs_h: torch.Tensor,
      w_gu_h: torch.Tensor,
      w_dn_h: torch.Tensor,
      ep_rank: int,
  ) -> torch.Tensor:
    # Pure-PyTorch reference for a single head (used on CPU unit tests).
    S_all, d_h = x_h.shape
    e_lo = ep_rank * self.e_local
    e_hi = e_lo + self.e_local
    out = torch.zeros((S_all, d_h), device=x_h.device, dtype=torch.float32)
    x_bf = x_h.to(w_gu_h.dtype)
    for e_loc in range(self.e_local):
      e_glob = e_lo + e_loc
      mask = idx_h == e_glob  # (S_all, K)
      tok_w = (probs_h * mask.float()).sum(dim=-1)  # (S_all,)
      if not bool(mask.any()):
        continue
      tok_ids = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
      gu = torch.matmul(x_bf.index_select(0, tok_ids), w_gu_h[e_loc])
      act = swiglu7_split(gu)
      dn = torch.matmul(act, w_dn_h[e_loc]).float()
      out.index_add_(0, tok_ids, dn * tok_w.index_select(0, tok_ids).unsqueeze(-1))
    return out.to(x_h.dtype)

  def forward(
      self,
      x_sorted: torch.Tensor,
      mod_ids_sorted: torch.Tensor,
      group_sizes: torch.Tensor,
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    h = hooks if hooks is not None else get_tpu_hooks()
    S_loc = x_sorted.shape[0]
    x_norm = self.pre_norm(x_sorted, mod_ids_sorted)

    # 1. Routed Multi-Head MoE
    moe_in = self.split_linear(x_norm)
    if h.mh_moe_routed is not None:
      moe_core = h.mh_moe_routed(
          moe_in,
          self.gate,
          self.expert_bias,
          self.w_gate_up,
          self.w_down,
          self.w_gate_up_scale,
          self.w_down_scale,
      )
    else:
      x_heads = moe_in.reshape(S_loc, self.num_heads, self.head_dim)
      if h.route_mh_moe is not None:
        idx_all_h, probs_all_h = h.route_mh_moe(x_heads, self.gate, self.expert_bias)
      else:
        idx_all_h, probs_all_h = self._route_local(x_heads)
      head_outs: list[torch.Tensor] = []
      for hi in range(self.num_heads):
        xh = x_heads[:, hi, :].contiguous()
        ih = idx_all_h[hi].contiguous()
        ph = probs_all_h[hi].contiguous()
        if h.ep_size > 1 and h.ep_all_gather is not None:
          xh_g = h.ep_all_gather(xh)
          ih_g = h.ep_all_gather(ih)
          ph_g = h.ep_all_gather(ph)
        else:
          xh_g, ih_g, ph_g = xh, ih, ph

        if h.moe_local_head is not None:
          wgu_s = self.w_gate_up_scale[hi] if self.w_gate_up_scale is not None else None
          wd_s = self.w_down_scale[hi] if self.w_down_scale is not None else None
          yh_g = h.moe_local_head(
              xh_g, ih_g, ph_g, self.w_gate_up[hi], self.w_down[hi], wgu_s, wd_s
          )
        else:
          yh_g = self._routed_ref_single_head(
              xh_g, ih_g, ph_g, self.w_gate_up[hi], self.w_down[hi], h.ep_rank
          )

        if h.ep_size > 1 and h.ep_reduce_scatter is not None:
          yh = h.ep_reduce_scatter(yh_g)
        else:
          yh = yh_g
        head_outs.append(yh)
      moe_core = torch.stack(head_outs, dim=1).reshape(S_loc, self.num_heads * self.head_dim)

    moe_out = self.merge_linear(moe_core)

    # 2. Shared Expert (M=1)
    sh_act = swiglu7_interleaved(self.shared_expert_fc1(x_norm))
    sh_out = self.shared_expert_fc2(sh_act)

    # 3. Modality-Specific Shared Expert (M=3)
    ms_gu = self.modality_specific_shared_expert_fc1(x_norm, group_sizes, mod_ids_sorted, hooks=h)
    ms_act = swiglu7_interleaved(ms_gu)
    ms_out = self.modality_specific_shared_expert_fc2(ms_act, group_sizes, mod_ids_sorted, hooks=h)

    return moe_out + sh_out + ms_out


class Magi2PreviewLayerTorch(nn.Module):
  def __init__(
      self,
      config: Magi2PreviewConfig,
      layer_idx: int,
      *,
      ep_size: int = 1,
      dtype: torch.dtype = torch.bfloat16,
      fp8_moe: bool = False,
  ) -> None:
    super().__init__()
    self.layer_idx = layer_idx
    self.is_moe = config.is_moe_layer(layer_idx)
    self.num_modality = config.num_modality_for_layer(layer_idx)
    self.num_streams = config.mhc_num_stream
    self.sinkhorn_iters = config.mhc_sinkhorn_iters
    self.sinkhorn_eps = config.mhc_sinkhorn_eps
    self.rms_eps = config.rms_norm_eps

    self.attention = Magi2PreviewAttentionTorch(config, self.num_modality, dtype=dtype)
    if self.is_moe:
      self.mlp: nn.Module = Magi2MultiHeadMoELayerTorch(
          config, ep_size=ep_size, dtype=dtype, fp8_moe=fp8_moe
      )
    else:
      self.mlp = Magi2DenseMLPTorch(config, num_modality=self.num_modality, dtype=dtype)

    n = self.num_streams
    ad = config.stream_hidden_size
    self.mhc_norm = MultiModalityRMSNormTorch(ad, self.num_modality, eps=config.rms_norm_eps, cast_back=False)

    self.mhc_phi_fused_attn = nn.Parameter(torch.zeros((ad, n + n + n * n), dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_pre_attn = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_post_attn = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_res_attn = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_bias_pre_attn = nn.Parameter(torch.zeros((n,), dtype=torch.float32), requires_grad=False)
    self.mhc_bias_post_attn = nn.Parameter(torch.zeros((n,), dtype=torch.float32), requires_grad=False)
    self.mhc_bias_res_attn = nn.Parameter(torch.zeros((n, n), dtype=torch.float32), requires_grad=False)

    self.mhc_phi_fused_mlp = nn.Parameter(torch.zeros((ad, n + n + n * n), dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_pre_mlp = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_post_mlp = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_alpha_res_mlp = nn.Parameter(torch.full((1,), config.mhc_alpha_init, dtype=torch.float32), requires_grad=False)
    self.mhc_bias_pre_mlp = nn.Parameter(torch.zeros((n,), dtype=torch.float32), requires_grad=False)
    self.mhc_bias_post_mlp = nn.Parameter(torch.zeros((n,), dtype=torch.float32), requires_grad=False)
    self.mhc_bias_res_mlp = nn.Parameter(torch.zeros((n, n), dtype=torch.float32), requires_grad=False)

  def forward(
      self,
      s_flat: torch.Tensor,
      rope_96: torch.Tensor,
      mod_ids_sorted: torch.Tensor,
      group_sizes: torch.Tensor,
      perm_local: torch.Tensor,
      inv_perm_local: torch.Tensor,
      sample_lens: tuple[int, ...],
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    h = hooks if hooks is not None else get_tpu_hooks()
    mhc_mod = mod_ids_sorted if self.num_modality > 1 else None

    # 1. Attention mHC Pre
    if h.mhc_pre is not None and self.num_modality == 1:
      x_attn_in, h_post_a, h_res_a = h.mhc_pre(
          s_flat,
          self.mhc_norm.weight,
          self.mhc_phi_fused_attn,
          self.mhc_alpha_pre_attn,
          self.mhc_alpha_post_attn,
          self.mhc_alpha_res_attn,
          self.mhc_bias_pre_attn,
          self.mhc_bias_post_attn,
          self.mhc_bias_res_attn,
      )
    else:
      x_attn_in, h_post_a, h_res_a = mhc_pre_ref_torch(
          s_flat,
          self.mhc_norm.weight,
          self.mhc_phi_fused_attn,
          self.mhc_alpha_pre_attn,
          self.mhc_alpha_post_attn,
          self.mhc_alpha_res_attn,
          self.mhc_bias_pre_attn,
          self.mhc_bias_post_attn,
          self.mhc_bias_res_attn,
          mod_ids=mhc_mod,
          num_streams=self.num_streams,
          sinkhorn_iters=self.sinkhorn_iters,
          sinkhorn_eps=self.sinkhorn_eps,
          rms_eps=self.rms_eps,
          fp32_matmul_fn=h.fp32_matmul_highest,
      )

    # 2. Attention Sublayer
    attn_out = self.attention(
        x_attn_in,
        rope_96,
        mod_ids_sorted,
        group_sizes,
        perm_local,
        inv_perm_local,
        sample_lens,
        hooks=h,
    )

    # 3. Fused Attn-mHC-Post + MLP-mHC-Pre Seam (1-pass HBM when hook active!)
    if h.mhc_fused_post_pre is not None and self.num_modality == 1:
      s_mid, x_mlp_in, h_post_m, h_res_m = h.mhc_fused_post_pre(
          s_flat,
          attn_out,
          h_post_a,
          h_res_a,
          self.mhc_norm.weight,
          self.mhc_phi_fused_mlp,
          self.mhc_alpha_pre_mlp,
          self.mhc_alpha_post_mlp,
          self.mhc_alpha_res_mlp,
          self.mhc_bias_pre_mlp,
          self.mhc_bias_post_mlp,
          self.mhc_bias_res_mlp,
      )
    else:
      if h.mhc_post is not None and self.num_modality == 1:
        s_mid = h.mhc_post(s_flat, attn_out, h_post_a, h_res_a)
      else:
        s_mid = mhc_post_ref_torch(s_flat, attn_out, h_post_a, h_res_a, num_streams=self.num_streams)
      if h.mhc_pre is not None and self.num_modality == 1:
        x_mlp_in, h_post_m, h_res_m = h.mhc_pre(
            s_mid,
            self.mhc_norm.weight,
            self.mhc_phi_fused_mlp,
            self.mhc_alpha_pre_mlp,
            self.mhc_alpha_post_mlp,
            self.mhc_alpha_res_mlp,
            self.mhc_bias_pre_mlp,
            self.mhc_bias_post_mlp,
            self.mhc_bias_res_mlp,
        )
      else:
        x_mlp_in, h_post_m, h_res_m = mhc_pre_ref_torch(
            s_mid,
            self.mhc_norm.weight,
            self.mhc_phi_fused_mlp,
            self.mhc_alpha_pre_mlp,
            self.mhc_alpha_post_mlp,
            self.mhc_alpha_res_mlp,
            self.mhc_bias_pre_mlp,
            self.mhc_bias_post_mlp,
            self.mhc_bias_res_mlp,
            mod_ids=mhc_mod,
            num_streams=self.num_streams,
            sinkhorn_iters=self.sinkhorn_iters,
            sinkhorn_eps=self.sinkhorn_eps,
            rms_eps=self.rms_eps,
            fp32_matmul_fn=h.fp32_matmul_highest,
        )

    # 4. MLP / Multi-Head MoE Sublayer
    mlp_out = self.mlp(x_mlp_in, mod_ids_sorted, group_sizes, hooks=h)

    # 5. MLP mHC Post
    if h.mhc_post is not None and self.num_modality == 1:
      return h.mhc_post(s_mid, mlp_out, h_post_m, h_res_m)
    return mhc_post_ref_torch(s_mid, mlp_out, h_post_m, h_res_m, num_streams=self.num_streams)


class Magi2PreviewTransformerTorch(nn.Module):
  def __init__(
      self,
      config: Magi2PreviewConfig,
      *,
      ep_size: int = 1,
      dtype: torch.dtype = torch.bfloat16,
      fp8_moe: bool = False,
  ) -> None:
    super().__init__()
    self.config = config
    self.ep_size = ep_size
    self.dtype = dtype
    self.fp8_moe = fp8_moe

    self.pre_adapter = Magi2PreAdapterTorch(config)
    self.layers = nn.ModuleList([
        Magi2PreviewLayerTorch(config, i, ep_size=ep_size, dtype=dtype, fp8_moe=fp8_moe)
        for i in range(config.num_layers)
    ])
    self.post_adapter = Magi2PostAdapterTorch(config)

  def embed_and_sort(
      self,
      x: torch.Tensor,
      rope_or_coords: torch.Tensor,
      modality_mapping: torch.Tensor,
      time_token_sequence: torch.Tensor | None = None,
      *,
      embedded_input: bool = False,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    """Run PreAdapter, timestep injection, 3D RoPE, and modality sort."""
    mod_ids = torch.where(
        modality_mapping == 3,
        torch.full_like(modality_mapping, MODALITY_TEXT),
        modality_mapping,
    ).to(torch.int64)

    if rope_or_coords.shape[-1] == 9:
      rope_96 = compute_3d_rope_96_torch(rope_or_coords, self.config.rope_temperature)
    else:
      rope_96 = rope_or_coords.float()

    if embedded_input:
      x_emb = x.float()
    else:
      x_emb = self.pre_adapter(x, mod_ids)

    if time_token_sequence is not None:
      t_dim = time_token_sequence.shape[-1]
      x_emb = torch.cat((time_token_sequence.float(), x_emb[:, t_dim:]), dim=-1)
    x_act = x_emb.to(self.dtype)

    perm_local = torch.argsort(mod_ids, stable=True)
    inv_perm_local = torch.zeros_like(perm_local).scatter(
        0,
        perm_local,
        torch.arange(perm_local.shape[0], device=perm_local.device, dtype=perm_local.dtype),
    )
    mod_ids_sorted = mod_ids.index_select(0, perm_local)
    group_sizes = torch.stack([
        (mod_ids_sorted == 0).sum(),
        (mod_ids_sorted == 1).sum(),
        (mod_ids_sorted == 2).sum(),
    ]).to(torch.int32)
    x_sorted = x_act.index_select(0, perm_local)
    return x_sorted, rope_96, mod_ids, mod_ids_sorted, group_sizes, perm_local, inv_perm_local

  def forward(
      self,
      x: torch.Tensor,
      rope_or_coords: torch.Tensor,
      modality_mapping: torch.Tensor,
      *,
      sample_lens: tuple[int, ...],
      time_token_sequence: torch.Tensor | None = None,
      embedded_input: bool = False,
      hooks: Magi2TpuHooks | None = None,
  ) -> torch.Tensor:
    (
        x_sorted,
        rope_96,
        mod_ids,
        mod_ids_sorted,
        group_sizes,
        perm_local,
        inv_perm_local,
    ) = self.embed_and_sort(
        x,
        rope_or_coords,
        modality_mapping,
        time_token_sequence,
        embedded_input=embedded_input,
    )
    for layer in self.layers:
      x_sorted = layer(
          x_sorted,
          rope_96,
          mod_ids_sorted,
          group_sizes,
          perm_local,
          inv_perm_local,
          sample_lens,
          hooks=hooks,
      )
    x_seq = x_sorted.index_select(0, inv_perm_local)
    return self.post_adapter(x_seq, mod_ids)


# ---------------------------------------------------------------------------
# Fast 8-Thread Parallel Safetensors Loader (Matches jax_preview_dit Exactly)
# ---------------------------------------------------------------------------


def _np_to_torch(arr: np.ndarray, target_dtype: torch.dtype) -> torch.Tensor:
  """Zero-copy / fast conversion from safetensors NumPy (incl. ml_dtypes.bfloat16)."""
  if str(arr.dtype) == "bfloat16":
    u16 = np.ascontiguousarray(arr).view(np.uint16)
    t = torch.from_numpy(u16).view(torch.bfloat16)
  else:
    t = torch.from_numpy(np.ascontiguousarray(arr))
  return t.to(target_dtype) if t.dtype != target_dtype else t


def _quantize_fp8_per_col_torch(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-output-channel symmetric E4M3 quantization matching ``jax_mh_moe``."""
  # w_bf16: (H, E_loc, in_d, out_d)
  w_f32 = w_bf16.float()
  max_abs = w_f32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12)  # (H, E_loc, 1, out_d)
  scale = max_abs / 448.0
  w_q = (w_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
  return w_q, scale.unsqueeze(-3)  # (H, E_loc, 1, 1, out_d)


def _read_moe_expert_slice(
    f: Any,
    key: str,
    *,
    num_heads: int,
    num_experts: int,
    e_lo: int,
    e_hi: int,
    dtype: torch.dtype,
) -> torch.Tensor:
  """Read ONLY ``[e_lo:e_hi]`` experts across all ``num_heads`` via direct file seek.

  Reads ~240 MB instead of 2.0 GiB per routed MoE tensor (8x less disk I/O and
  8x less peak host RAM in 8-process ``torch_tpu`` launches).
  """
  from ...safetensors_io import _np_dtype

  if hasattr(f, "_meta") and hasattr(f, "_f") and key in f._meta:
    m = f._meta[key]
    shape = m["shape"]
    if len(shape) == 3 and shape[0] == num_heads * num_experts:
      d1, d2 = int(shape[1]), int(shape[2])
      dt = _np_dtype(m["dtype"])
      e_loc = e_hi - e_lo
      row_bytes = d1 * d2 * dt.itemsize
      chunk_bytes = e_loc * row_bytes
      buf = np.empty(num_heads * chunk_bytes, dtype=np.uint8)
      mv = memoryview(buf)
      base_off = f._base + int(m["data_offsets"][0])
      for h in range(num_heads):
        off = base_off + (h * num_experts + e_lo) * row_bytes
        f._f.seek(off)
        dst = mv[h * chunk_bytes : (h + 1) * chunk_bytes]
        got = 0
        while got < chunk_bytes:
          k = f._f.readinto(dst[got:])
          if not k:
            raise IOError(f"short read for {key} head={h}: {got}/{chunk_bytes}")
          got += k
      arr_loc = buf.view(dt).reshape(num_heads, e_loc, d1, d2)
      return _np_to_torch(arr_loc, dtype)

  # Fallback for standard safetensors mmap reader
  arr = f.get_tensor(key)
  if arr.ndim == 3 and arr.shape[0] == num_heads * num_experts:
    arr_4d = arr.reshape(num_heads, num_experts, arr.shape[1], arr.shape[2])
    arr_loc = np.ascontiguousarray(arr_4d[:, e_lo:e_hi, :, :])
  elif arr.ndim == 4 and arr.shape[0] == num_heads and arr.shape[1] == num_experts:
    arr_loc = np.ascontiguousarray(arr[:, e_lo:e_hi, :, :])
  else:
    raise ValueError(f"Unexpected MoE tensor shape {arr.shape}")
  return _np_to_torch(arr_loc, dtype)


def load_torch_preview_from_safetensors(
    checkpoint_dir: str | Path,
    config: Magi2PreviewConfig | None = None,
    *,
    ep_rank: int = 0,
    ep_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    fp8_moe: bool = False,
    max_workers: int = 1,
    device: str | torch.device = "cpu",
    h2d_fn: Any = None,
    sync_fn: Any = None,
    verbose: bool = False,
) -> Magi2PreviewTransformerTorch:
  """Low-RAM streaming loader for :class:`Magi2PreviewTransformerTorch`.

  Keeps the model on ``meta`` and streams each parameter directly to ``device``
  (optionally via ``h2d_fn(tensor, device)`` such as ``_stage_h2d``) using
  byte-range seek slicing on routed MoE weights (reading ONLY ``E_local = 32``
  experts/head per rank -> < 1 GiB peak host RAM per rank in 8-process runs!).
  """
  import time
  from ...safetensors_io import safe_open

  t0 = time.perf_counter()
  if config is None:
    config = Magi2PreviewConfig.from_checkpoint(checkpoint_dir)

  ckpt, wm = load_safetensors_weight_map(checkpoint_dir)
  with torch.device("meta"):
    model = Magi2PreviewTransformerTorch(
        config, ep_size=ep_size, dtype=dtype, fp8_moe=fp8_moe
    )

  target_dev = torch.device(device)
  H = config.moe_num_heads
  E = config.moe_num_experts
  e_local = E // ep_size
  e_lo = ep_rank * e_local
  e_hi = e_lo + e_local

  bias_src = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
  use_ema_bias = bias_src != "main"

  by_file: dict[str, list[str]] = {}
  for k, fn in wm.items():
    by_file.setdefault(fn, []).append(k)

  pending_gate: dict[int, torch.Tensor] = {}
  pending_up: dict[int, torch.Tensor] = {}

  def _push(t_cpu: torch.Tensor) -> torch.Tensor:
    if target_dev.type == "cpu":
      return t_cpu.contiguous()
    if h2d_fn is not None:
      return h2d_fn(t_cpu, target_dev)
    return t_cpu.contiguous().to(target_dev)

  def _set_p(mod: nn.Module, name: str, t_cpu: torch.Tensor) -> None:
    mod._parameters[name] = nn.Parameter(_push(t_cpu), requires_grad=False)

  def _assign_grouped_linear(mod: GroupedLinearTorch, arr: np.ndarray) -> None:
    t = _np_to_torch(arr, dtype)
    if mod.num_modality == 1:
      _set_p(mod, "weight", t)
    else:
      m_out, in_f = t.shape
      out_f = m_out // mod.num_modality
      _set_p(mod, "weight", t.reshape(mod.num_modality, out_f, in_f).transpose(1, 2).contiguous())

  def _assign_norm(mod: MultiModalityRMSNormTorch, arr: np.ndarray) -> None:
    t = _np_to_torch(arr, torch.float32)
    if mod.num_modality == 1:
      _set_p(mod, "weight", t.reshape(mod.dim))
    else:
      _set_p(mod, "weight", t.reshape(mod.num_modality, mod.dim))

  def _load_one_shard(item: tuple[str, list[str]]) -> list[tuple[str, Any]]:
    fn, keys = item
    loaded: list[tuple[str, Any]] = []
    with safe_open(str(ckpt / fn)) as f:
      for k in keys:
        if k == "pre_adapter.rope.bands":
          continue
        if k.endswith(".router.expert_bias_ema") and not use_ema_bias:
          continue
        if k.endswith(".router.expert_bias") and use_ema_bias:
          continue
        if k.endswith((".moe_mlp.W_gate", ".moe_mlp.W_up", ".moe_mlp.W_down")):
          loaded.append((
              k,
              _read_moe_expert_slice(
                  f, k, num_heads=H, num_experts=E, e_lo=e_lo, e_hi=e_hi, dtype=dtype
              ),
          ))
        else:
          loaded.append((k, f.get_tensor(k)))
    return loaded

  def _assign_one(key: str, val: Any) -> None:
    # PreAdapter
    if key == "pre_adapter.video_embedder.weight":
      _set_p(model.pre_adapter.video_embedder, "weight", _np_to_torch(val, torch.float32))
      return
    if key == "pre_adapter.video_embedder.bias":
      _set_p(model.pre_adapter.video_embedder, "bias", _np_to_torch(val, torch.float32))
      return
    if key == "pre_adapter.audio_embedder.weight":
      _set_p(model.pre_adapter.audio_embedder, "weight", _np_to_torch(val, torch.float32))
      return
    if key == "pre_adapter.audio_embedder.bias":
      _set_p(model.pre_adapter.audio_embedder, "bias", _np_to_torch(val, torch.float32))
      return
    if key == "pre_adapter.text_embedder.weight":
      _set_p(model.pre_adapter.text_embedder, "weight", _np_to_torch(val, torch.float32))
      return
    if key == "pre_adapter.text_embedder.bias":
      _set_p(model.pre_adapter.text_embedder, "bias", _np_to_torch(val, torch.float32))
      return
    # PostAdapter
    if key == "post_adapter.final_norm_video.weight":
      _assign_norm(model.post_adapter.final_norm_video, val)
      return
    if key == "post_adapter.final_norm_audio.weight":
      _assign_norm(model.post_adapter.final_norm_audio, val)
      return
    if key == "post_adapter.final_linear_video.weight":
      _set_p(model.post_adapter.final_linear_video, "weight", _np_to_torch(val, torch.float32))
      return
    if key == "post_adapter.final_linear_audio.weight":
      _set_p(model.post_adapter.final_linear_audio, "weight", _np_to_torch(val, torch.float32))
      return

    if not key.startswith("block.layers."):
      raise KeyError(f"Unrecognized MAGI-2 Preview checkpoint key: {key}")

    parts = key.split(".")
    li = int(parts[2])
    sub = ".".join(parts[3:])
    layer: Magi2PreviewLayerTorch = model.layers[li]  # type: ignore[assignment]

    # Attention
    if sub == "attention.pre_norm.weight":
      _assign_norm(layer.attention.pre_norm, val)
    elif sub == "attention.linear_g.weight":
      _assign_grouped_linear(layer.attention.linear_g, val)
    elif sub == "attention.linear_qkv.weight":
      _assign_grouped_linear(layer.attention.linear_qkv, val)
    elif sub == "attention.linear_proj.weight":
      _assign_grouped_linear(layer.attention.linear_proj, val)
    elif sub == "attention.sinks":
      _set_p(layer.attention, "sinks", _np_to_torch(val, torch.float32))
    elif sub == "attention.q_norm.weight":
      _assign_norm(layer.attention.q_norm, val)
    elif sub == "attention.k_norm.weight":
      _assign_norm(layer.attention.k_norm, val)
    # mHC
    elif sub == "mhc_norm.weight":
      _assign_norm(layer.mhc_norm, val)
    elif sub in (
        "mhc_phi_fused_attn",
        "mhc_alpha_pre_attn",
        "mhc_alpha_post_attn",
        "mhc_alpha_res_attn",
        "mhc_bias_pre_attn",
        "mhc_bias_post_attn",
        "mhc_bias_res_attn",
        "mhc_phi_fused_mlp",
        "mhc_alpha_pre_mlp",
        "mhc_alpha_post_mlp",
        "mhc_alpha_res_mlp",
        "mhc_bias_pre_mlp",
        "mhc_bias_post_mlp",
        "mhc_bias_res_mlp",
    ):
      _set_p(layer, sub, _np_to_torch(val, torch.float32))
    # Dense MLP
    elif not layer.is_moe:
      mlp: Magi2DenseMLPTorch = layer.mlp  # type: ignore[assignment]
      if sub == "mlp.pre_norm.weight":
        _assign_norm(mlp.pre_norm, val)
      elif sub == "mlp.up_gate_proj.weight":
        _assign_grouped_linear(mlp.up_gate_proj, val)
      elif sub == "mlp.down_proj.weight":
        _assign_grouped_linear(mlp.down_proj, val)
      else:
        raise KeyError(f"Unrecognized Dense MLP key: {key}")
    # Multi-Head MoE
    else:
      moe: Magi2MultiHeadMoELayerTorch = layer.mlp  # type: ignore[assignment]
      def _set_moe_w(w_name: str, s_name: str, w_cpu: torch.Tensor) -> None:
        if not fp8_moe:
          _set_p(moe, w_name, w_cpu)
          return
        if target_dev.type != "cpu":
          w_dev = _push(w_cpu)
          wq, ws = _quantize_fp8_per_col_torch(w_dev)
          del w_dev
          moe._parameters[w_name] = nn.Parameter(wq, requires_grad=False)
          moe._parameters[s_name] = nn.Parameter(ws, requires_grad=False)
        else:
          wq, ws = _quantize_fp8_per_col_torch(w_cpu)
          _set_p(moe, w_name, wq)
          _set_p(moe, s_name, ws)

      if sub == "mlp.pre_norm.weight":
        _assign_norm(moe.pre_norm, val)
      elif sub == "mlp.split_linear.weight":
        _set_p(moe.split_linear, "weight", _np_to_torch(val, dtype))
      elif sub == "mlp.merge_linear.weight":
        _set_p(moe.merge_linear, "weight", _np_to_torch(val, dtype))
      elif sub == "mlp.shared_expert_fc1.weight":
        _set_p(moe.shared_expert_fc1, "weight", _np_to_torch(val, dtype))
      elif sub == "mlp.shared_expert_fc2.weight":
        _set_p(moe.shared_expert_fc2, "weight", _np_to_torch(val, dtype))
      elif sub == "mlp.modality_specific_shared_expert_fc1.weight":
        _assign_grouped_linear(moe.modality_specific_shared_expert_fc1, val)
      elif sub == "mlp.modality_specific_shared_expert_fc2.weight":
        _assign_grouped_linear(moe.modality_specific_shared_expert_fc2, val)
      elif sub == "mlp.moe_mlp.gate":
        _set_p(moe, "gate", _np_to_torch(val, torch.float32).reshape(H, E, config.moe_head_dim))
      elif sub in ("mlp.moe_mlp.router.expert_bias_ema", "mlp.moe_mlp.router.expert_bias"):
        _set_p(moe, "expert_bias", _np_to_torch(val, torch.float32).reshape(H, E))
      elif sub == "mlp.moe_mlp.W_gate":
        pending_gate[li] = val
        if li in pending_up:
          w_gu = torch.cat((pending_gate.pop(li), pending_up.pop(li)), dim=-1)
          _set_moe_w("w_gate_up", "w_gate_up_scale", w_gu)
          del w_gu
      elif sub == "mlp.moe_mlp.W_up":
        pending_up[li] = val
        if li in pending_gate:
          w_gu = torch.cat((pending_gate.pop(li), pending_up.pop(li)), dim=-1)
          _set_moe_w("w_gate_up", "w_gate_up_scale", w_gu)
          del w_gu
      elif sub == "mlp.moe_mlp.W_down":
        _set_moe_w("w_down", "w_down_scale", val)
      else:
        raise KeyError(f"Unrecognized MoE key: {key}")

  import gc
  import resource

  items = sorted(by_file.items())
  total_shards = len(items)
  done = 0

  def _after_shard() -> None:
    nonlocal done
    if sync_fn is not None:
      sync_fn()
    done += 1
    if done % 8 == 0:
      gc.collect()
    if verbose and (done % 14 == 0 or done == total_shards):
      rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
      print(
          f"  [torch-preview] streamed shard {done}/{total_shards} "
          f"({time.perf_counter()-t0:.1f}s, max_rss={rss_mib:.0f} MiB)",
          flush=True,
      )

  if max_workers <= 1:
    for item in items:
      shard_items = _load_one_shard(item)
      for k, v in shard_items:
        _assign_one(k, v)
      del shard_items
      _after_shard()
  else:
    # Bounded in-flight futures (at most max_workers shards in CPU RAM at once!)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      it = iter(items)
      inflight = []
      for _ in range(min(max_workers, total_shards)):
        inflight.append(pool.submit(_load_one_shard, next(it)))
      while inflight:
        fut = inflight.pop(0)
        try:
          nxt = next(it)
          inflight.append(pool.submit(_load_one_shard, nxt))
        except StopIteration:
          pass
        shard_items = fut.result()
        for k, v in shard_items:
          _assign_one(k, v)
        del shard_items
        _after_shard()

  if pending_gate or pending_up:
    raise RuntimeError(
        f"Unpaired W_gate/W_up layers: gate={sorted(pending_gate)}, up={sorted(pending_up)}"
    )

  model.eval()
  return model
