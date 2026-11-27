"""Pure-PyTorch VideoGym-LTX-2.3 (22B Audio-Video DiT) for ``torch_tpu`` and CPU parity.

Exact mathematical port of :mod:`diffusers_tpu.models.ltx2.jax_dit`:
  * 48 ``LTX2VideoTransformerBlock`` layers (`D_v = 4096 = 32 x 128`,
    `D_a = 2048 = 32 x 64`, `audio_cross_attention_dim = 2048 = 32 x 64`).
  * 6 attentions per block: ``attn1``, ``audio_attn1``, ``attn2``, ``audio_attn2``,
    ``audio_to_video_attn``, ``video_to_audio_attn`` — each with per-head
    ``2 * sigmoid(gate)``, full-width Q/K RMSNorm, float32 split-RoPE, optional
    STG ``value_only`` override on block 28, and ``~iso`` modality isolation gate.
  * Plain 4x GELU-tanh FFN (``net.0.proj -> gelu(approximate='tanh') -> net.2``).
  * Optional W8A8 E4M3 FP8 block linears (28 large 2D linears/block with
    ``min(shape) >= 512``) quantized directly on TPU device during streaming load
    (``~18.4 GiB`` resident vs ``~35.6 GiB`` BF16).
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import math
import os
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...safetensors_io import safe_open


@dataclass
class LTX2TpuHooks:
  """Injectable TPU Pallas & intra-chip ``sp=2`` hooks for :class:`LTX2TransformerTorch`."""

  video_self_attn: Callable | None = None
  v2a_attn: Callable | None = None
  fp8_linear: Callable | None = None
  sp_size: int = 1
  qk_fp8: bool = False


def rms_norm_torch(x: torch.Tensor, eps: float = 1e-6, weight: torch.Tensor | None = None) -> torch.Tensor:
  xf = x.float()
  y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))
  if weight is not None:
    y = y * weight.float()
  return y.to(x.dtype)


def layer_norm_no_affine_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  xf = x.float()
  m = xf.mean(dim=-1, keepdim=True)
  y = (xf - m) * torch.rsqrt((xf - m).pow(2).mean(dim=-1, keepdim=True) + float(eps))
  return y.to(x.dtype)


def sinusoid_ltx2_torch(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
  """diffusers ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
  half = dim // 2
  freqs = torch.exp(
      -math.log(10000.0)
      * torch.arange(half, device=t.device, dtype=torch.float32)
      / float(half)
  )
  arg = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
  return torch.cat((torch.cos(arg), torch.sin(arg)), dim=-1)


def apply_split_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """``x`` ``[B, S, H, D]``; ``cos, sin`` ``[S, H, D/2]`` -> split-half float32 rotation."""
  xf = x.float()
  r = x.shape[-1] // 2
  a, b = xf[..., :r], xf[..., r:]
  c = cos.float().unsqueeze(0)
  s = sin.float().unsqueeze(0)
  return torch.cat((a * c - b * s, b * c + a * s), dim=-1).to(x.dtype)


def dense_attention_ltx2_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """``[B, S, H, D]`` float32-score dense attention (exact match to ``jax_dit.dense_attention``)."""
  scale = q.shape[-1] ** -0.5
  s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
  p = torch.softmax(s, dim=-1).to(v.dtype)
  return torch.einsum("bhqk,bkhd->bqhd", p.float(), v.float()).to(v.dtype)


class LTX2LinearTorch(nn.Module):
  """Linear layer supporting BF16/FP32 or W8A8 E4M3 FP8 (stored as ``(in, out)``)."""

  def __init__(self, in_features: int, out_features: int, *, dtype: torch.dtype = torch.bfloat16) -> None:
    super().__init__()
    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype), requires_grad=False)
    self.bias = nn.Parameter(torch.empty((out_features,), dtype=dtype), requires_grad=False)
    self.register_parameter("scale", None)

  def forward(self, x: torch.Tensor, hooks: LTX2TpuHooks | None = None) -> torch.Tensor:
    if self.scale is not None:
      # W8A8 E4M3 FP8 path: self.weight is (in, out) float8_e4m3fn, self.scale is (1, out) float32.
      if hooks is not None and hooks.fp8_linear is not None:
        return hooks.fp8_linear(x, self.weight, self.scale, self.bias)
      # Exact CPU/eager fallback matching jax_dit._linear L144-151.
      f8_max = 448.0
      x32 = x.float()
      x_scale = torch.clamp(x32.abs().amax(dim=-1, keepdim=True), min=1e-12) / f8_max
      qx = torch.clamp(x32 / x_scale, -f8_max, f8_max).to(torch.float8_e4m3fn).float()
      acc = torch.matmul(qx, self.weight.float())
      y = (acc * x_scale * self.scale).to(x.dtype)
      return y + self.bias.to(x.dtype)
    return F.linear(x.to(self.weight.dtype), self.weight, self.bias.to(self.weight.dtype))


class LTX2AdaLNEmbedderTorch(nn.Module):
  def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dtype: torch.dtype) -> None:
    super().__init__()
    self.linear_1 = LTX2LinearTorch(in_dim, hidden_dim, dtype=dtype)
    self.linear_2 = LTX2LinearTorch(hidden_dim, hidden_dim, dtype=dtype)
    self.linear = LTX2LinearTorch(hidden_dim, out_dim, dtype=dtype)

  def forward(self, t_flat: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    e = sinusoid_ltx2_torch(t_flat, dim=256).to(dtype)
    e = self.linear_1(e)
    e = self.linear_2(F.silu(e))
    return self.linear(F.silu(e)), e


class LTX2AttentionTorch(nn.Module):
  """Exact ``LTX2Attention`` (full-width Q/K RMSNorm, split RoPE, 2*sigmoid gate, STG)."""

  def __init__(
      self,
      q_dim: int,
      kv_dim: int,
      inner_dim: int,
      heads: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.heads = int(heads)
    self.head_dim = inner_dim // heads
    self.to_q = LTX2LinearTorch(q_dim, inner_dim, dtype=dtype)
    self.to_k = LTX2LinearTorch(kv_dim, inner_dim, dtype=dtype)
    self.to_v = LTX2LinearTorch(kv_dim, inner_dim, dtype=dtype)
    self.to_gate_logits = LTX2LinearTorch(q_dim, heads, dtype=dtype)
    self.to_out = LTX2LinearTorch(inner_dim, q_dim, dtype=dtype)
    self.norm_q_weight = nn.Parameter(torch.ones((inner_dim,), dtype=dtype), requires_grad=False)
    self.norm_k_weight = nn.Parameter(torch.ones((inner_dim,), dtype=dtype), requires_grad=False)

  def forward(
      self,
      x: torch.Tensor,
      ctx: torch.Tensor | None,
      attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
      *,
      q_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
      k_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
      stg_flag: torch.Tensor | None = None,
      hooks: LTX2TpuHooks | None = None,
  ) -> torch.Tensor:
    b, s, _ = x.shape
    src = x if ctx is None else ctx
    gate = 2.0 * torch.sigmoid(self.to_gate_logits(x, hooks=None).float()).to(x.dtype)  # [B, S, H]
    v = self.to_v(src, hooks=hooks)
    q = rms_norm_torch(self.to_q(x, hooks=hooks), weight=self.norm_q_weight)
    k = rms_norm_torch(self.to_k(src, hooks=hooks), weight=self.norm_k_weight)
    q = q.reshape(b, s, self.heads, self.head_dim)
    k = k.reshape(src.shape[0], src.shape[1], self.heads, self.head_dim)
    vh = v.reshape(src.shape[0], src.shape[1], self.heads, self.head_dim)
    if q_rope is not None:
      q = apply_split_rope_torch(q, q_rope[0], q_rope[1])
      kr = q_rope if k_rope is None else k_rope
      k = apply_split_rope_torch(k, kr[0], kr[1])
    o = attn_fn(q, k, vh)
    if stg_flag is not None:
      # stg_flag is [1] or [B] bool/bf16 device tensor (SPMD-safe: identical HLO across all ranks!).
      mask = stg_flag.to(torch.bool).reshape(-1, 1, 1, 1)
      o = torch.where(mask, vh, o)
    o = (o * gate.unsqueeze(-1).to(o.dtype)).reshape(b, s, -1)
    return self.to_out(o, hooks=hooks)


class LTX2FFNTorch(nn.Module):
  """Plain 4x GELU-tanh FFN (``net.0.proj -> gelu(tanh) -> net.2``)."""

  def __init__(self, dim: int, mult: int = 4, *, dtype: torch.dtype = torch.bfloat16) -> None:
    super().__init__()
    self.proj = LTX2LinearTorch(dim, dim * mult, dtype=dtype)
    self.out = LTX2LinearTorch(dim * mult, dim, dtype=dtype)

  def forward(self, x: torch.Tensor, hooks: LTX2TpuHooks | None = None) -> torch.Tensor:
    h = self.proj(x, hooks=hooks)
    return self.out(F.gelu(h, approximate="tanh"), hooks=hooks)


def _mod_torch(table: torch.Tensor, temb: torch.Tensor) -> list[torch.Tensor]:
  n, d = table.shape
  vals = table.to(temb.dtype).unsqueeze(0).unsqueeze(0) + temb.reshape(temb.shape[0], temb.shape[1], n, d)
  return [vals[:, :, i] for i in range(n)]


class LTX2BlockTorch(nn.Module):
  """One ``LTX2VideoTransformerBlock`` (matching ``jax_dit.block`` line-for-line)."""

  def __init__(self, cfg: dict[str, Any], *, dtype: torch.dtype = torch.bfloat16) -> None:
    super().__init__()
    vh = int(cfg["num_attention_heads"])
    vd = int(cfg["attention_head_dim"])
    ah = int(cfg["audio_num_attention_heads"])
    ad = int(cfg["audio_attention_head_dim"])
    v_dim = vh * vd  # 4096
    a_dim = ah * ad  # 2048
    ca_dim = int(cfg["audio_cross_attention_dim"])  # 2048

    self.scale_shift_table = nn.Parameter(torch.empty((9, v_dim), dtype=dtype), requires_grad=False)
    self.audio_scale_shift_table = nn.Parameter(torch.empty((9, a_dim), dtype=dtype), requires_grad=False)
    self.prompt_scale_shift_table = nn.Parameter(torch.empty((2, v_dim), dtype=dtype), requires_grad=False)
    self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty((2, a_dim), dtype=dtype), requires_grad=False)
    self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.empty((5, v_dim), dtype=dtype), requires_grad=False)
    self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.empty((5, a_dim), dtype=dtype), requires_grad=False)

    self.attn1 = LTX2AttentionTorch(v_dim, v_dim, v_dim, vh, dtype=dtype)
    self.audio_attn1 = LTX2AttentionTorch(a_dim, a_dim, a_dim, ah, dtype=dtype)
    self.attn2 = LTX2AttentionTorch(v_dim, v_dim, v_dim, vh, dtype=dtype)
    self.audio_attn2 = LTX2AttentionTorch(a_dim, a_dim, a_dim, ah, dtype=dtype)
    self.audio_to_video_attn = LTX2AttentionTorch(v_dim, a_dim, ca_dim, ah, dtype=dtype)
    self.video_to_audio_attn = LTX2AttentionTorch(a_dim, v_dim, ca_dim, ah, dtype=dtype)
    self.ff = LTX2FFNTorch(v_dim, 4, dtype=dtype)
    self.audio_ff = LTX2FFNTorch(a_dim, 4, dtype=dtype)

  def forward(
      self,
      x: torch.Tensor,
      a: torch.Tensor,
      ctx: torch.Tensor,
      actx: torch.Tensor,
      temb: torch.Tensor,
      temb_a: torch.Tensor,
      tp: torch.Tensor,
      tpa: torch.Tensor,
      cav: torch.Tensor,
      cavg: torch.Tensor,
      caa: torch.Tensor,
      caag: torch.Tensor,
      token_t: torch.Tensor,
      rope_v_c: torch.Tensor,
      rope_v_s: torch.Tensor,
      rope_a_c: torch.Tensor,
      rope_a_s: torch.Tensor,
      rope_cv_c: torch.Tensor,
      rope_cv_s: torch.Tensor,
      rope_ca_c: torch.Tensor,
      rope_ca_s: torch.Tensor,
      stg_flag: torch.Tensor | None,
      iso_keep: torch.Tensor,
      *,
      hooks: LTX2TpuHooks | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    v_mods = _mod_torch(self.scale_shift_table, temb)
    sh, sc, g, sh_m, sc_m, g_m, sh_q, sc_q, g_q = [m.index_select(1, token_t) for m in v_mods]
    ash, asc, ag, ash_m, asc_m, ag_m, ash_q, asc_q, ag_q = _mod_torch(self.audio_scale_shift_table, temb_a)

    v_self_fn = (hooks.video_self_attn if (hooks is not None and hooks.video_self_attn is not None) else dense_attention_ltx2_torch)
    v2a_fn = (hooks.v2a_attn if (hooks is not None and hooks.v2a_attn is not None) else dense_attention_ltx2_torch)

    # 1. Self-attention
    h = rms_norm_torch(x) * (1.0 + sc) + sh
    x = x + self.attn1(h, None, v_self_fn, q_rope=(rope_v_c, rope_v_s), stg_flag=stg_flag, hooks=hooks) * g
    h = rms_norm_torch(a) * (1.0 + asc) + ash
    a = a + self.audio_attn1(h, None, dense_attention_ltx2_torch, q_rope=(rope_a_c, rope_a_s), stg_flag=stg_flag, hooks=hooks) * ag

    # 2. Prompt cross-attention
    sh_kv, sc_kv = _mod_torch(self.prompt_scale_shift_table, tp)
    ash_kv, asc_kv = _mod_torch(self.audio_prompt_scale_shift_table, tpa)
    h = rms_norm_torch(x) * (1.0 + sc_q) + sh_q
    c = ctx * (1.0 + sc_kv) + sh_kv
    x = x + self.attn2(h, c, dense_attention_ltx2_torch, hooks=hooks) * g_q
    h = rms_norm_torch(a) * (1.0 + asc_q) + ash_q
    c = actx * (1.0 + asc_kv) + ash_kv
    a = a + self.audio_attn2(h, c, dense_attention_ltx2_torch, hooks=hooks) * ag_q

    # 3. Bidirectional audio <-> video cross-attention
    nx, na = rms_norm_torch(x), rms_norm_torch(a)
    vt = self.video_a2v_cross_attn_scale_shift_table
    at = self.audio_a2v_cross_attn_scale_shift_table
    v_a2v_sc, v_a2v_sh, v_v2a_sc, v_v2a_sh = _mod_torch(vt[:4], cav)
    (a2v_gate,) = _mod_torch(vt[4:], cavg)
    a_a2v_sc, a_a2v_sh, a_v2a_sc, a_v2a_sh = _mod_torch(at[:4], caa)
    (v2a_gate,) = _mod_torch(at[4:], caag)

    keep = iso_keep.to(x.dtype)
    hq = nx * (1.0 + v_a2v_sc) + v_a2v_sh
    hk = na * (1.0 + a_a2v_sc) + a_a2v_sh
    x = x + keep * (a2v_gate * self.audio_to_video_attn(
        hq, hk, dense_attention_ltx2_torch,
        q_rope=(rope_cv_c, rope_cv_s), k_rope=(rope_ca_c, rope_ca_s), hooks=hooks,
    ))
    hq = na * (1.0 + a_v2a_sc) + a_v2a_sh
    hk = nx * (1.0 + v_v2a_sc) + v_v2a_sh
    a = a + keep * (v2a_gate * self.video_to_audio_attn(
        hq, hk, v2a_fn,
        q_rope=(rope_ca_c, rope_ca_s), k_rope=(rope_cv_c, rope_cv_s), hooks=hooks,
    ))

    # 4. Feed-forward
    x = x + self.ff(rms_norm_torch(x) * (1.0 + sc_m) + sh_m, hooks=hooks) * g_m
    a = a + self.audio_ff(rms_norm_torch(a) * (1.0 + asc_m) + ash_m, hooks=hooks) * ag_m
    return x, a


class LTX2TransformerTorch(nn.Module):
  """Top-level 48-block LTX-2.3 DiT in pure PyTorch."""

  def __init__(self, cfg: dict[str, Any], *, dtype: torch.dtype = torch.bfloat16) -> None:
    super().__init__()
    self.cfg = dict(cfg)
    self.dtype = dtype
    v_dim = int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"])
    a_dim = int(cfg["audio_num_attention_heads"]) * int(cfg["audio_attention_head_dim"])
    v_in = int(cfg["in_channels"])
    a_in = int(cfg["audio_in_channels"])
    v_out = int(cfg["out_channels"])
    a_out = int(cfg["audio_out_channels"])
    n_layers = int(cfg["num_layers"])

    self.proj_in = LTX2LinearTorch(v_in, v_dim, dtype=dtype)
    self.audio_proj_in = LTX2LinearTorch(a_in, a_dim, dtype=dtype)
    self.proj_out = LTX2LinearTorch(v_dim, v_out, dtype=dtype)
    self.audio_proj_out = LTX2LinearTorch(a_dim, a_out, dtype=dtype)
    self.scale_shift_table = nn.Parameter(torch.empty((2, v_dim), dtype=dtype), requires_grad=False)
    self.audio_scale_shift_table = nn.Parameter(torch.empty((2, a_dim), dtype=dtype), requires_grad=False)

    self.time_embed = LTX2AdaLNEmbedderTorch(256, v_dim, 9 * v_dim, dtype=dtype)
    self.audio_time_embed = LTX2AdaLNEmbedderTorch(256, a_dim, 9 * a_dim, dtype=dtype)
    self.prompt_adaln = LTX2AdaLNEmbedderTorch(256, v_dim, 2 * v_dim, dtype=dtype)
    self.audio_prompt_adaln = LTX2AdaLNEmbedderTorch(256, a_dim, 2 * a_dim, dtype=dtype)
    self.av_cross_attn_video_scale_shift = LTX2AdaLNEmbedderTorch(256, v_dim, 4 * v_dim, dtype=dtype)
    self.av_cross_attn_video_a2v_gate = LTX2AdaLNEmbedderTorch(256, v_dim, v_dim, dtype=dtype)
    self.av_cross_attn_audio_scale_shift = LTX2AdaLNEmbedderTorch(256, a_dim, 4 * a_dim, dtype=dtype)
    self.av_cross_attn_audio_v2a_gate = LTX2AdaLNEmbedderTorch(256, a_dim, a_dim, dtype=dtype)

    self.blocks = nn.ModuleList([LTX2BlockTorch(cfg, dtype=dtype) for _ in range(n_layers)])

  def conditioning(
      self,
      t_video: torch.Tensor,  # [B, K]
      t_audio: torch.Tensor,  # [B]
      sigma: torch.Tensor,  # [B]
  ) -> dict[str, torch.Tensor]:
    b = t_video.shape[0]
    dt = self.dtype
    temb, emb = self.time_embed(t_video.reshape(-1), dt)
    temb_a, emb_a = self.audio_time_embed(t_audio, dt)
    tp, _ = self.prompt_adaln(sigma, dt)
    tpa, _ = self.audio_prompt_adaln(sigma, dt)
    cav, _ = self.av_cross_attn_video_scale_shift(sigma, dt)
    cavg, _ = self.av_cross_attn_video_a2v_gate(sigma, dt)
    caa, _ = self.av_cross_attn_audio_scale_shift(sigma, dt)
    caag, _ = self.av_cross_attn_audio_v2a_gate(sigma, dt)
    r = lambda z: z.reshape(b, -1, z.shape[-1])
    return {
        "temb": r(temb), "emb": r(emb),
        "temb_a": r(temb_a), "emb_a": r(emb_a),
        "tp": r(tp), "tpa": r(tpa),
        "cav": r(cav), "cavg": r(cavg),
        "caa": r(caa), "caag": r(caag),
    }

  def output_heads(
      self,
      x: torch.Tensor,
      a: torch.Tensor,
      emb: torch.Tensor,
      emb_a: torch.Tensor,
      token_t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    emb_tok = emb.index_select(1, token_t)
    sh_v = self.scale_shift_table[0].to(emb_tok.dtype) + emb_tok
    sc_v = self.scale_shift_table[1].to(emb_tok.dtype) + emb_tok
    px = self.proj_out(layer_norm_no_affine_torch(x) * (1.0 + sc_v) + sh_v)

    sh_a = self.audio_scale_shift_table[0].to(emb_a.dtype) + emb_a
    sc_a = self.audio_scale_shift_table[1].to(emb_a.dtype) + emb_a
    pa = self.audio_proj_out(layer_norm_no_affine_torch(a) * (1.0 + sc_a) + sh_a)
    return px, pa


def _quantize_w_fp8_torch(w_out_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-output-channel symmetric E4M3 quantization matching ``jax_dit._quantize_w_fp8_tpu``.

  Input ``w_out_in`` has shape ``(out, in)``; returns ``(qw_in_out (in, out) f8e4m3fn, scale (1, out) f32)``.
  """
  w_io = w_out_in.t().contiguous()  # (in, out)
  w32 = w_io.float()
  scale = torch.clamp(w32.abs().amax(dim=0, keepdim=True), min=1e-12) / 448.0
  qw = torch.clamp(w32 / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
  return qw, scale


def _set_ltx2_param(model: LTX2TransformerTorch, key: str, tensor: torch.Tensor, *, fp8: bool, device: torch.device | None, h2d_fn: Any) -> bool:
  """Map one diffusers ``transformer/*.safetensors`` key into ``model``."""
  def _place(t: torch.Tensor) -> torch.Tensor:
    if device is None:
      return t
    return h2d_fn(t, device) if h2d_fn is not None else t.to(device)

  def _assign_linear(lin: LTX2LinearTorch, suffix: str, t: torch.Tensor, allow_fp8: bool) -> None:
    if suffix == "weight":
      if allow_fp8 and t.ndim == 2 and min(t.shape) >= 512:
        # Quantize directly on target device (4.95x faster on TPU vs host CPU!).
        t_dev = _place(t.to(model.dtype))
        qw, sc = _quantize_w_fp8_torch(t_dev)
        del t_dev
        lin.weight = nn.Parameter(qw, requires_grad=False)
        lin.scale = nn.Parameter(sc, requires_grad=False)
      else:
        lin.weight = nn.Parameter(_place(t.to(model.dtype)), requires_grad=False)
        lin.scale = None
    elif suffix == "bias":
      lin.bias = nn.Parameter(_place(t.to(model.dtype)), requires_grad=False)
    else:
      raise KeyError(f"Unknown linear suffix {suffix!r} for key {key}")

  if key.startswith("transformer_blocks."):
    _, bi_s, rest = key.split(".", 2)
    blk = model.blocks[int(bi_s)]
    if rest in (
        "scale_shift_table",
        "audio_scale_shift_table",
        "prompt_scale_shift_table",
        "audio_prompt_scale_shift_table",
        "video_a2v_cross_attn_scale_shift_table",
        "audio_a2v_cross_attn_scale_shift_table",
    ):
      setattr(blk, rest, nn.Parameter(_place(tensor.to(model.dtype)), requires_grad=False))
      return True
    sub_name, sub_rest = rest.split(".", 1)
    if sub_name in ("ff", "audio_ff"):
      ff = getattr(blk, sub_name)
      if sub_rest.startswith("net.0.proj."):
        _assign_linear(ff.proj, sub_rest.split(".")[-1], tensor, allow_fp8=fp8)
        return True
      if sub_rest.startswith("net.2."):
        _assign_linear(ff.out, sub_rest.split(".")[-1], tensor, allow_fp8=fp8)
        return True
    elif sub_name in (
        "attn1", "audio_attn1", "attn2", "audio_attn2",
        "audio_to_video_attn", "video_to_audio_attn",
    ):
      att = getattr(blk, sub_name)
      if sub_rest == "norm_q.weight":
        att.norm_q_weight = nn.Parameter(_place(tensor.to(model.dtype)), requires_grad=False)
        return True
      if sub_rest == "norm_k.weight":
        att.norm_k_weight = nn.Parameter(_place(tensor.to(model.dtype)), requires_grad=False)
        return True
      if sub_rest.startswith("to_out.0."):
        _assign_linear(att.to_out, sub_rest.split(".")[-1], tensor, allow_fp8=fp8)
        return True
      lin_name, suf = sub_rest.split(".", 1)
      _assign_linear(getattr(att, lin_name), suf, tensor, allow_fp8=fp8)
      return True
    return False

  # Top-level parameters (always kept BF16/dtype, never FP8).
  if key in ("scale_shift_table", "audio_scale_shift_table"):
    setattr(model, key, nn.Parameter(_place(tensor.to(model.dtype)), requires_grad=False))
    return True
  top_name, top_rest = key.split(".", 1)
  if top_name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out"):
    _assign_linear(getattr(model, top_name), top_rest, tensor, allow_fp8=False)
    return True
  if top_name in (
      "time_embed", "audio_time_embed", "prompt_adaln", "audio_prompt_adaln",
      "av_cross_attn_video_scale_shift", "av_cross_attn_video_a2v_gate",
      "av_cross_attn_audio_scale_shift", "av_cross_attn_audio_v2a_gate",
  ):
    emb_mod = getattr(model, top_name)
    if top_rest.startswith("emb.timestep_embedder.linear_1."):
      _assign_linear(emb_mod.linear_1, top_rest.split(".")[-1], tensor, allow_fp8=False)
      return True
    if top_rest.startswith("emb.timestep_embedder.linear_2."):
      _assign_linear(emb_mod.linear_2, top_rest.split(".")[-1], tensor, allow_fp8=False)
      return True
    if top_rest.startswith("linear."):
      _assign_linear(emb_mod.linear, top_rest.split(".")[-1], tensor, allow_fp8=False)
      return True
  return False


def load_torch_ltx2_dit(
    snapshot_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    fp8: bool = False,
    device: torch.device | None = None,
    h2d_fn: Callable[[torch.Tensor, torch.device], torch.Tensor] | None = None,
    sync_fn: Callable[[], None] | None = None,
    verbose: bool = True,
) -> LTX2TransformerTorch:
  """Stream-load LTX-2.3 48-block DiT directly onto ``device`` (with optional on-device W8A8 FP8)."""
  tf_dir = os.path.join(snapshot_dir, "transformer")
  with open(os.path.join(tf_dir, "config.json")) as f:
    cfg = json.load(f)

  with torch.device("meta"):
    model = LTX2TransformerTorch(cfg, dtype=dtype)

  t0 = time.perf_counter()
  loaded = 0
  skipped = 0
  shards = sorted(glob.glob(os.path.join(tf_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"No transformer safetensors found in {tf_dir}")

  for shard_path in shards:
    with safe_open(shard_path, framework="pt") as reader:
      for key in reader.keys():
        t = reader.get_tensor(key)
        if _set_ltx2_param(model, key, t, fp8=fp8, device=device, h2d_fn=h2d_fn):
          loaded += 1
        else:
          skipped += 1
    if sync_fn is not None:
      sync_fn()

  if verbose:
    print(
        f"[torch_ltx2_dit] Loaded {loaded} tensors (skipped {skipped}, dtype={dtype}, fp8={fp8}) "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
  return model
