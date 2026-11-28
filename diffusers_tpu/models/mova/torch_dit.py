"""Pure-PyTorch Dual-Tower DiT for MOVA (40L video + 30L audio + 30L bidirectional bridge).

Mirrors :mod:`diffusers_tpu.models.mova.jax_dit` operation-for-operation so:
  * CPU FP32 parity against ``jax_dit.dual_step`` is exact (``max|d| < 1e-5``).
  * ``mova/torchax_dit.py`` traces this module through ``torchax`` into the fused
    40-layer ``shard_map`` + Pallas ``splash`` / ``sparse_splash`` program.
  * ``backends/torchtpu_mova.py`` executes this module across 8 ``torch_tpu`` ranks
    (``2x4`` CFG × Ulysses SP) with pre-transposed BF16 / W8A8 E4M3 FP8 linears and
    ``DT_AOT_PERSIST`` grouped-block AOT compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
import time
from typing import Any, Callable

from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MovaTpuHooks:
  """Optional runtime hooks injected by ``torchax`` or ``torch_tpu``."""

  video_self_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  video_cross_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  a2v_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  v2a_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  audio_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  fp8_linear: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
  sp_size: int = 1
  qk_fp8: bool = False


def _layer_norm_torch(
    x: torch.Tensor,
    eps: float = 1e-6,
    w: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
  xf = x.float()
  mean = xf.mean(dim=-1, keepdim=True)
  diff = xf - mean
  y = diff * torch.rsqrt((diff * diff).mean(dim=-1, keepdim=True) + eps)
  if w is not None:
    y = y * w.float() + b.float()
  return y.to(x.dtype)


def _rms_torch(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  xf = x.float()
  y = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
  return (y * w.float()).to(x.dtype)


def _rope_pairs_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Interleaved-pair RoPE of ``x [B, S, H, D]`` in float32 (exact match to ``jax_dit._rope_pairs``)."""
  xf = x.float().reshape(*x.shape[:-1], -1, 2)
  even, odd = xf[..., 0], xf[..., 1]
  c = cos.float().unsqueeze(0).unsqueeze(2)
  s = sin.float().unsqueeze(0).unsqueeze(2)
  return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).reshape(x.shape).to(x.dtype)


def _rope_half_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Rotate-half RoPE of ``x [B, S, H, D]`` in ``x.dtype`` (exact match to ``jax_dit._rope_half``)."""
  c = cos.to(x.dtype).unsqueeze(0).unsqueeze(2)
  s = sin.to(x.dtype).unsqueeze(0).unsqueeze(2)
  r = x.shape[-1] // 2
  x1, x2 = x[..., :r], x[..., r:]
  return x * c + torch.cat((-x2, x1), dim=-1) * s


def dense_attention_mova_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """``[B, S, H, D]`` dense attention (FP32 exact on CPU; BF16 MXU matmul + FP32 softmax on TPU)."""
  scale = q.shape[-1] ** -0.5
  qt = q.transpose(1, 2)
  kt = k.transpose(1, 2)
  vt = v.transpose(1, 2)
  if q.dtype == torch.float32:
    s = torch.matmul(qt, kt.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vt).transpose(1, 2)
  s = torch.matmul(qt, kt.transpose(-1, -2)).float() * scale
  p = torch.softmax(s, dim=-1).to(v.dtype)
  return torch.matmul(p, vt).transpose(1, 2)


def patchify_video_torch(latents: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
  """``[B, C, T, H, W] -> [B, (T/pt)*(H/ph)*(W/pw), C*pt*ph*pw]``."""
  b, c, t, hh, ww = latents.shape
  pt, ph, pw = patch
  x = latents.reshape(b, c, t // pt, pt, hh // ph, ph, ww // pw, pw)
  x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
  return x.reshape(b, (t // pt) * (hh // ph) * (ww // pw), c * pt * ph * pw)


def unpatchify_video_torch(
    x: torch.Tensor,
    grid: tuple[int, int, int],
    patch: tuple[int, int, int],
    out_dim: int,
) -> torch.Tensor:
  """``'b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)'``."""
  b = x.shape[0]
  f, h, w = grid
  pt, ph, pw = patch
  x = x.reshape(b, f, h, w, pt, ph, pw, out_dim)
  return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, out_dim, f * pt, h * ph, w * pw)


class MovaLinearTorch(nn.Module):
  """Linear layer supporting BF16/FP32 (optional pre-transposed ``(in, out)``) or W8A8 E4M3 FP8."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
      pretranspose: bool = False,
  ) -> None:
    super().__init__()
    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.pretranspose = bool(pretranspose)
    w_shape = (in_features, out_features) if self.pretranspose else (out_features, in_features)
    self.weight = nn.Parameter(torch.empty(w_shape, dtype=dtype), requires_grad=False)
    self.bias = nn.Parameter(torch.empty((out_features,), dtype=dtype), requires_grad=False)
    self.register_parameter("scale", None)

  def forward(self, x: torch.Tensor, hooks: MovaTpuHooks | None = None) -> torch.Tensor:
    if self.scale is not None:
      if hooks is not None and hooks.fp8_linear is not None:
        return hooks.fp8_linear(x, self.weight, self.scale, self.bias)
      f8_max = 448.0
      x32 = x.float()
      x_scale = torch.clamp(x32.abs().amax(dim=-1, keepdim=True), min=1e-12) / f8_max
      qx = torch.clamp(x32 / x_scale, -f8_max, f8_max).to(torch.float8_e4m3fn).float()
      acc = torch.matmul(qx, self.weight.float())
      y = (acc * x_scale * self.scale).to(x.dtype)
      return y + self.bias.to(x.dtype)
    if self.pretranspose:
      return torch.matmul(x.to(self.weight.dtype), self.weight) + self.bias.to(self.weight.dtype)
    return F.linear(x.to(self.weight.dtype), self.weight, self.bias.to(self.weight.dtype))


class MovaBlockTorch(nn.Module):
  """One Wan/MOVA DiT block (self-attn + RoPE, text cross-attn, GELU-tanh FFN)."""

  def __init__(
      self,
      dim: int,
      num_heads: int,
      ffn_dim: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
      pretranspose: bool = False,
  ) -> None:
    super().__init__()
    self.dim = int(dim)
    self.num_heads = int(num_heads)
    self.head_dim = self.dim // self.num_heads
    self.modulation = nn.Parameter(torch.empty((6, dim), dtype=dtype), requires_grad=False)

    self.self_attn_q = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.self_attn_k = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.self_attn_v = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.self_attn_o = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.self_attn_norm_q = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)
    self.self_attn_norm_k = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)

    self.norm3_weight = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)
    self.norm3_bias = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)

    self.cross_attn_q = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.cross_attn_k = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.cross_attn_v = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.cross_attn_o = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.cross_attn_norm_q = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)
    self.cross_attn_norm_k = nn.Parameter(torch.empty((dim,), dtype=dtype), requires_grad=False)

    self.ffn_0 = MovaLinearTorch(dim, ffn_dim, dtype=dtype, pretranspose=pretranspose)
    self.ffn_2 = MovaLinearTorch(ffn_dim, dim, dtype=dtype, pretranspose=pretranspose)

  def forward(
      self,
      x: torch.Tensor,
      ctx: torch.Tensor,
      t_mod: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      *,
      self_attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
      cross_attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
      hooks: MovaTpuHooks | None = None,
  ) -> torch.Tensor:
    sa_fn = self_attn_fn or dense_attention_mova_torch
    ca_fn = cross_attn_fn or dense_attention_mova_torch
    mod = self.modulation.unsqueeze(0) + t_mod  # [B, 6, D]
    shift_a = mod[:, 0:1]
    scale_a = mod[:, 1:2]
    gate_a = mod[:, 2:3]
    shift_f = mod[:, 3:4]
    scale_f = mod[:, 4:5]
    gate_f = mod[:, 5:6]

    b, s, d = x.shape
    h, hd = self.num_heads, self.head_dim

    # 1. Self-attention with interleaved-pair RoPE
    y = _layer_norm_torch(x) * (1.0 + scale_a) + shift_a
    q = _rms_torch(self.self_attn_q(y, hooks=hooks), self.self_attn_norm_q)
    k = _rms_torch(self.self_attn_k(y, hooks=hooks), self.self_attn_norm_k)
    v = self.self_attn_v(y, hooks=hooks)
    q = _rope_pairs_torch(q.reshape(b, s, h, hd), cos, sin)
    k = _rope_pairs_torch(k.reshape(b, s, h, hd), cos, sin)
    out = sa_fn(q, k, v.reshape(b, s, h, hd)).reshape(b, s, d)
    x = x + gate_a * self.self_attn_o(out, hooks=hooks)

    # 2. Text cross-attention
    y = _layer_norm_torch(x, w=self.norm3_weight, b=self.norm3_bias)
    q = _rms_torch(self.cross_attn_q(y, hooks=hooks), self.cross_attn_norm_q)
    k = _rms_torch(self.cross_attn_k(ctx, hooks=hooks), self.cross_attn_norm_k)
    v = self.cross_attn_v(ctx, hooks=hooks)
    out = ca_fn(
        q.reshape(b, s, h, hd),
        k.reshape(b, ctx.shape[1], h, hd),
        v.reshape(b, ctx.shape[1], h, hd),
    ).reshape(b, s, d)
    x = x + self.cross_attn_o(out, hooks=hooks)

    # 3. Feed-forward (GELU-tanh in FP32)
    y = _layer_norm_torch(x) * (1.0 + scale_f) + shift_f
    hh = self.ffn_0(y, hooks=hooks)
    hh = F.gelu(hh.float(), approximate="tanh").to(hh.dtype)
    return x + gate_f * self.ffn_2(hh, hooks=hooks)


class MovaTowerTorch(nn.Module):
  """Wan-family Video (40L) or Audio (30L) DiT tower."""

  def __init__(
      self,
      config: dict[str, Any],
      *,
      dtype: torch.dtype = torch.bfloat16,
      pretranspose: bool = False,
  ) -> None:
    super().__init__()
    self.config = dict(config)
    self.dtype = dtype
    dim = int(config["dim"])
    num_heads = int(config["num_heads"])
    num_layers = int(config["num_layers"])
    ffn_dim = int(config.get("ffn_dim", 4 * dim))
    freq_dim = int(config.get("freq_dim", 256))
    text_dim = int(config.get("text_dim", 4096))
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    patch_prod = 1
    for p in patch_size:
      patch_prod *= int(p)
    in_dim = int(config.get("in_dim", 36))
    out_dim = int(config.get("out_dim", 16))

    # Time embedders always kept in FP32 (matching jax_dit.load_tower)
    self.time_embedding_0 = MovaLinearTorch(freq_dim, dim, dtype=torch.float32, pretranspose=pretranspose)
    self.time_embedding_2 = MovaLinearTorch(dim, dim, dtype=torch.float32, pretranspose=pretranspose)
    self.time_projection_1 = MovaLinearTorch(dim, 6 * dim, dtype=torch.float32, pretranspose=pretranspose)

    self.text_embedding_0 = MovaLinearTorch(text_dim, dim, dtype=dtype, pretranspose=pretranspose)
    self.text_embedding_2 = MovaLinearTorch(dim, dim, dtype=dtype, pretranspose=pretranspose)

    self.patch_embedding = MovaLinearTorch(in_dim * patch_prod, dim, dtype=dtype, pretranspose=pretranspose)
    self.blocks = nn.ModuleList([
        MovaBlockTorch(dim, num_heads, ffn_dim, dtype=dtype, pretranspose=pretranspose)
        for _ in range(num_layers)
    ])
    self.head_modulation = nn.Parameter(torch.empty((2, dim), dtype=dtype), requires_grad=False)
    self.head_head = MovaLinearTorch(dim, out_dim * patch_prod, dtype=dtype, pretranspose=pretranspose)

  def embed_time(self, sinusoid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = int(self.config["dim"])
    te = self.time_embedding_0(sinusoid.float()).float()
    te = self.time_embedding_2(F.silu(te)).float()
    tm = self.time_projection_1(F.silu(te)).float().reshape(-1, 6, dim)
    dt = self.head_head.weight.dtype
    return te.to(dt), tm.to(dt)

  def embed_text(self, context: torch.Tensor) -> torch.Tensor:
    y = self.text_embedding_0(context)
    y = F.gelu(y.float(), approximate="tanh").to(y.dtype)
    return self.text_embedding_2(y)

  def patch_embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
    """Apply ``patch_embedding`` with float32 accumulation (exact match to ``jax_dit.dual_step``)."""
    dt = self.head_head.weight.dtype
    w = self.patch_embedding.weight
    b = self.patch_embedding.bias
    if self.patch_embedding.pretranspose:
      y = torch.matmul(tokens.to(dt).float(), w.float()) + b.float()
    else:
      y = F.linear(tokens.to(dt).float(), w.float(), b.float())
    return y.to(dt)

  def head(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    m = self.head_modulation.unsqueeze(0) + t.unsqueeze(1)  # [B, 2, dim]
    y = _layer_norm_torch(x) * (1.0 + m[:, 1:2]) + m[:, 0:1]
    return self.head_head(y)


class MovaBridgeCondTorch(nn.Module):
  """One ``ConditionalCrossAttentionBlock`` (a2v or v2a)."""

  def __init__(
      self,
      q_dim: int,
      kv_dim: int,
      num_heads: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
      pretranspose: bool = False,
  ) -> None:
    super().__init__()
    self.num_heads = int(num_heads)
    self.y_norm_weight = nn.Parameter(torch.empty((kv_dim,), dtype=dtype), requires_grad=False)
    self.y_norm_bias = nn.Parameter(torch.empty((kv_dim,), dtype=dtype), requires_grad=False)
    self.q = MovaLinearTorch(q_dim, q_dim, dtype=dtype, pretranspose=pretranspose)
    self.k = MovaLinearTorch(kv_dim, q_dim, dtype=dtype, pretranspose=pretranspose)
    self.v = MovaLinearTorch(kv_dim, q_dim, dtype=dtype, pretranspose=pretranspose)
    self.o = MovaLinearTorch(q_dim, q_dim, dtype=dtype, pretranspose=pretranspose)
    self.norm_q = nn.Parameter(torch.empty((q_dim,), dtype=dtype), requires_grad=False)
    self.norm_k = nn.Parameter(torch.empty((q_dim,), dtype=dtype), requires_grad=False)

  def forward(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      x_rope: tuple[torch.Tensor, torch.Tensor],
      y_rope: tuple[torch.Tensor, torch.Tensor],
      *,
      attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
      hooks: MovaTpuHooks | None = None,
  ) -> torch.Tensor:
    af = attn_fn or dense_attention_mova_torch
    yn = _layer_norm_torch(y, w=self.y_norm_weight, b=self.y_norm_bias)
    q = _rms_torch(self.q(x, hooks=hooks), self.norm_q)
    k = _rms_torch(self.k(yn, hooks=hooks), self.norm_k)
    v = self.v(yn, hooks=hooks)
    b, sx, d = q.shape
    h = self.num_heads
    q = _rope_half_torch(q.reshape(b, sx, h, -1), x_rope[0], x_rope[1])
    k = _rope_half_torch(k.reshape(b, k.shape[1], h, -1), y_rope[0], y_rope[1])
    out = af(q, k, v.reshape(b, v.shape[1], h, -1)).reshape(b, sx, d)
    return self.o(out, hooks=hooks)


class MovaBridgeTorch(nn.Module):
  """30-layer bidirectional audio <-> video cross-attention bridge."""

  def __init__(
      self,
      bconf: dict[str, Any],
      num_shared: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
      pretranspose: bool = False,
  ) -> None:
    super().__init__()
    self.config = dict(bconf)
    self.dtype = dtype
    vdim = int(bconf["visual_hidden_dim"])
    adim = int(bconf["audio_hidden_dim"])
    hdim = int(bconf["head_dim"])
    bvh = vdim // hdim
    bah = adim // hdim
    self.a2v = nn.ModuleList([
        MovaBridgeCondTorch(vdim, adim, bvh, dtype=dtype, pretranspose=pretranspose)
        for _ in range(num_shared)
    ])
    self.v2a = nn.ModuleList([
        MovaBridgeCondTorch(adim, vdim, bah, dtype=dtype, pretranspose=pretranspose)
        for _ in range(num_shared)
    ])


def dual_step_torch(
    video: MovaTowerTorch,
    audio: MovaTowerTorch,
    bridge: MovaBridgeTorch,
    visual_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    t_sin: torch.Tensor,
    audio_t_sin: torch.Tensor,
    ropes: tuple[Any, ...],
    *,
    hooks: MovaTpuHooks | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Pure-PyTorch full single step matching ``jax_dit.dual_step``."""
  dt = video.head_head.weight.dtype
  vt, vtm = video.embed_time(t_sin)
  at, atm = audio.embed_time(audio_t_sin)
  vctx = video.embed_text(context.to(dt))
  actx = audio.embed_text(context.to(dt))

  patch = tuple(int(p) for p in video.config["patch_size"])
  _, _, tt, hh, ww = visual_latents.shape
  grid = (tt // patch[0], hh // patch[1], ww // patch[2])
  vx = video.patch_embed_tokens(patchify_video_torch(visual_latents, patch))
  ax = audio.patch_embed_tokens(audio_latents.transpose(1, 2))

  vcos, vsin, acos, asin, vrope, arope = ropes
  shared = min(len(video.blocks), len(audio.blocks))

  v_self = hooks.video_self_attn if (hooks and hooks.video_self_attn) else dense_attention_mova_torch
  v_cross = hooks.video_cross_attn if (hooks and hooks.video_cross_attn) else dense_attention_mova_torch
  a2v_fn = hooks.a2v_attn if (hooks and hooks.a2v_attn) else dense_attention_mova_torch
  v2a_fn = hooks.v2a_attn if (hooks and hooks.v2a_attn) else dense_attention_mova_torch
  a_fn = hooks.audio_attn if (hooks and hooks.audio_attn) else dense_attention_mova_torch

  for i in range(shared):
    v2a_out = bridge.v2a[i](ax, vx, arope, vrope, attn_fn=v2a_fn, hooks=hooks)
    a2v_out = bridge.a2v[i](vx, ax, vrope, arope, attn_fn=a2v_fn, hooks=hooks)
    vx = video.blocks[i](vx + a2v_out, vctx, vtm, vcos, vsin,
                         self_attn_fn=v_self, cross_attn_fn=v_cross, hooks=hooks)
    ax = audio.blocks[i](ax + v2a_out, actx, atm, acos, asin,
                         self_attn_fn=a_fn, cross_attn_fn=a_fn, hooks=hooks)
  for i in range(shared, len(video.blocks)):
    vx = video.blocks[i](vx, vctx, vtm, vcos, vsin,
                         self_attn_fn=v_self, cross_attn_fn=v_cross, hooks=hooks)

  vout = unpatchify_video_torch(video.head(vx, vt), grid, patch, int(video.config["out_dim"]))
  aout = audio.head(ax, at).transpose(1, 2)
  return vout, aout


# ---------------------------------------------------------------------------
# Streaming safetensors loaders (CPU / TPU with optional on-device W8A8 FP8)
# ---------------------------------------------------------------------------


def _quantize_w_fp8_torch(w_out_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-output-channel E4M3 quantization: ``(out, in) -> ((in, out) f8, (1, out) f32)``."""
  w_io = w_out_in.t().contiguous()
  w32 = w_io.float()
  scale = torch.clamp(w32.abs().amax(dim=0, keepdim=True), min=1e-12) / 448.0
  qw = torch.clamp(w32 / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
  return qw, scale


def _assign_mova_linear(
    lin: MovaLinearTorch,
    suffix: str,
    t: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    allow_fp8: bool,
    pretranspose_bf16: bool,
    place_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
  if suffix == "weight":
    if t.ndim > 2:
      t = t.reshape(t.shape[0], -1)
    if allow_fp8 and t.ndim == 2 and min(t.shape) >= 512:
      t_dev = place_fn(t.to(target_dtype))
      qw, sc = _quantize_w_fp8_torch(t_dev)
      del t_dev
      lin.weight = nn.Parameter(qw, requires_grad=False)
      lin.scale = nn.Parameter(sc, requires_grad=False)
      lin.pretranspose = True
    else:
      t_dev = place_fn(t.to(target_dtype))
      if pretranspose_bf16 and t_dev.ndim == 2:
        t_dev = t_dev.t().contiguous()
        lin.pretranspose = True
      else:
        lin.pretranspose = False
      lin.weight = nn.Parameter(t_dev, requires_grad=False)
      lin.scale = None
  elif suffix == "bias":
    lin.bias = nn.Parameter(place_fn(t.to(target_dtype)), requires_grad=False)
  else:
    raise KeyError(f"Unknown linear suffix {suffix!r}")


def _set_mova_tower_param(
    model: MovaTowerTorch,
    key: str,
    tensor: torch.Tensor,
    *,
    fp8: bool,
    pretranspose_bf16: bool = False,
    device: torch.device | None = None,
    h2d_fn: Any = None,
) -> bool:
  def _place(t: torch.Tensor) -> torch.Tensor:
    if device is None:
      return t
    return h2d_fn(t, device) if h2d_fn is not None else t.to(device)

  dt = model.dtype
  if key.startswith("blocks."):
    _, bi_s, rest = key.split(".", 2)
    blk = model.blocks[int(bi_s)]
    if rest == "modulation":
      blk.modulation = nn.Parameter(
          _place(tensor.reshape(tensor.shape[-2], tensor.shape[-1]).to(dt)), requires_grad=False
      )
      return True
    if rest == "norm3.weight":
      blk.norm3_weight = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
      return True
    if rest == "norm3.bias":
      blk.norm3_bias = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
      return True
    if rest.startswith("self_attn."):
      sub = rest[len("self_attn."):]
      if sub == "norm_q.weight":
        blk.self_attn_norm_q = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
        return True
      if sub == "norm_k.weight":
        blk.self_attn_norm_k = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
        return True
      nm, suf = sub.split(".", 1)
      _assign_mova_linear(getattr(blk, f"self_attn_{nm}"), suf, tensor, target_dtype=dt,
                          allow_fp8=fp8, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
      return True
    if rest.startswith("cross_attn."):
      sub = rest[len("cross_attn."):]
      if sub == "norm_q.weight":
        blk.cross_attn_norm_q = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
        return True
      if sub == "norm_k.weight":
        blk.cross_attn_norm_k = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
        return True
      nm, suf = sub.split(".", 1)
      _assign_mova_linear(getattr(blk, f"cross_attn_{nm}"), suf, tensor, target_dtype=dt,
                          allow_fp8=fp8, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
      return True
    if rest.startswith("ffn."):
      _, idx, suf = rest.split(".", 2)
      _assign_mova_linear(getattr(blk, f"ffn_{idx}"), suf, tensor, target_dtype=dt,
                          allow_fp8=fp8, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
      return True
    return False

  if key.startswith("time_embedding."):
    _, idx, suf = key.split(".", 2)
    _assign_mova_linear(getattr(model, f"time_embedding_{idx}"), suf, tensor,
                        target_dtype=torch.float32, allow_fp8=False,
                        pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  if key.startswith("time_projection."):
    _, idx, suf = key.split(".", 2)
    _assign_mova_linear(getattr(model, f"time_projection_{idx}"), suf, tensor,
                        target_dtype=torch.float32, allow_fp8=False,
                        pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  if key.startswith("text_embedding."):
    _, idx, suf = key.split(".", 2)
    _assign_mova_linear(getattr(model, f"text_embedding_{idx}"), suf, tensor,
                        target_dtype=dt, allow_fp8=False,
                        pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  if key.startswith("patch_embedding."):
    suf = key.split(".", 1)[1]
    _assign_mova_linear(model.patch_embedding, suf, tensor, target_dtype=dt,
                        allow_fp8=False, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  if key == "head.modulation":
    model.head_modulation = nn.Parameter(
        _place(tensor.reshape(tensor.shape[-2], tensor.shape[-1]).to(dt)), requires_grad=False
    )
    return True
  if key.startswith("head.head."):
    suf = key[len("head.head."):]
    _assign_mova_linear(model.head_head, suf, tensor, target_dtype=dt,
                        allow_fp8=False, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  return False


def _set_mova_bridge_param(
    model: MovaBridgeTorch,
    key: str,
    tensor: torch.Tensor,
    *,
    fp8: bool,
    pretranspose_bf16: bool = False,
    device: torch.device | None = None,
    h2d_fn: Any = None,
) -> bool:
  def _place(t: torch.Tensor) -> torch.Tensor:
    if device is None:
      return t
    return h2d_fn(t, device) if h2d_fn is not None else t.to(device)

  dt = model.dtype
  if key.startswith("audio_to_video_conditioners."):
    _, bi_s, rest = key.split(".", 2)
    cond = model.a2v[int(bi_s)]
  elif key.startswith("video_to_audio_conditioners."):
    _, bi_s, rest = key.split(".", 2)
    cond = model.v2a[int(bi_s)]
  else:
    return False

  if rest == "y_norm.weight":
    cond.y_norm_weight = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
    return True
  if rest == "y_norm.bias":
    cond.y_norm_bias = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
    return True
  if rest.startswith("inner."):
    sub = rest[len("inner."):]
    if sub == "norm_q.weight":
      cond.norm_q = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
      return True
    if sub == "norm_k.weight":
      cond.norm_k = nn.Parameter(_place(tensor.to(dt)), requires_grad=False)
      return True
    nm, suf = sub.split(".", 1)
    _assign_mova_linear(getattr(cond, nm), suf, tensor, target_dtype=dt,
                        allow_fp8=fp8, pretranspose_bf16=pretranspose_bf16, place_fn=_place)
    return True
  return False


def _iter_safetensors(folder: str):
  index = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
  if os.path.exists(index):
    with open(index) as f:
      files = sorted(set(json.load(f)["weight_map"].values()))
    shards = [os.path.join(folder, fn) for fn in files]
  else:
    shards = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"No safetensors found in {folder}")
  return shards


def load_torch_mova_tower(
    folder: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    fp8: bool = False,
    pretranspose_bf16: bool = False,
    device: torch.device | None = None,
    h2d_fn: Any = None,
    sync_fn: Callable[[], None] | None = None,
    verbose: bool = True,
) -> MovaTowerTorch:
  with open(os.path.join(folder, "config.json")) as f:
    cfg = json.load(f)
  with torch.device("meta"):
    model = MovaTowerTorch(cfg, dtype=dtype, pretranspose=pretranspose_bf16)
  t0 = time.perf_counter()
  loaded = skipped = 0
  for sp in _iter_safetensors(folder):
    with safe_open(sp, framework="pt") as r:
      for k in r.keys():
        t = r.get_tensor(k)
        if _set_mova_tower_param(model, k, t, fp8=fp8, pretranspose_bf16=pretranspose_bf16,
                                 device=device, h2d_fn=h2d_fn):
          loaded += 1
        else:
          skipped += 1
    if sync_fn is not None:
      sync_fn()
  if verbose:
    print(f"[torch_mova_dit] Loaded {os.path.basename(folder)}: {loaded} tensors "
          f"(skipped {skipped}, fp8={fp8}, preT={pretranspose_bf16}) in {time.perf_counter() - t0:.2f}s",
          flush=True)
  return model


def load_torch_mova_bridge(
    folder: str,
    num_shared: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    fp8: bool = False,
    pretranspose_bf16: bool = False,
    device: torch.device | None = None,
    h2d_fn: Any = None,
    sync_fn: Callable[[], None] | None = None,
    verbose: bool = True,
) -> MovaBridgeTorch:
  with open(os.path.join(folder, "config.json")) as f:
    bconf = json.load(f)
  with torch.device("meta"):
    model = MovaBridgeTorch(bconf, num_shared, dtype=dtype, pretranspose=pretranspose_bf16)
  t0 = time.perf_counter()
  loaded = skipped = 0
  for sp in _iter_safetensors(folder):
    with safe_open(sp, framework="pt") as r:
      for k in r.keys():
        t = r.get_tensor(k)
        if _set_mova_bridge_param(model, k, t, fp8=fp8, pretranspose_bf16=pretranspose_bf16,
                                  device=device, h2d_fn=h2d_fn):
          loaded += 1
        else:
          skipped += 1
    if sync_fn is not None:
      sync_fn()
  if verbose:
    print(f"[torch_mova_dit] Loaded bridge: {loaded} tensors "
          f"(skipped {skipped}, fp8={fp8}, preT={pretranspose_bf16}) in {time.perf_counter() - t0:.2f}s",
          flush=True)
  return model
