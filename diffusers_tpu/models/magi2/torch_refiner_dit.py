"""Pure-PyTorch Stage 2 MAGI-2 1080p Refiner DiT (6.72B) for ``torch_tpu`` and CPU parity.

Exact mathematical port of :mod:`diffusers_tpu.models.magi2.jax_refiner_dit`:
  * 30 dense pre-norm residual layers (no mHC, no Attention Sink, no MoE).
  * ``mm_layers = (0, 1, 28, 29)`` use 3-modality RMSNorm + 3-modality Linear
    (accelerated on TPU via ``torch.tpu.ops.ragged_dot`` since Refiner tokens
    are already packed monotonically as ``[video, audio?, text]``).
  * ``layers 2..27`` use single-modality RMSNorm + plain ``nn.Linear``.
  * GQA ``32Q / 8KV`` (``head_dim = 128``), fp32 QK-Norm, 3D partial RoPE
    (``[0:96]`` rotated in fp32), per-head scalar sigmoid output gate ``linear_g``,
    and 3D ``(8,4,4)`` block-grid local window + full text-tail attention.
  * Dense interleaved ``swiglu7`` MLP (``4096 -> 21840 -> 10920 -> 4096``),
    promoting the residual stream to ``float32`` after layer 0 (matching upstream
    ``magi2_refiner.py:L1848``).
  * Streaming safetensors loader (sequential offset order, direct-to-TPU H2D).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Magi2RefinerConfig
from .torch_preview_dit import (
    apply_3d_partial_rope_torch,
    rms_norm_1pw as rms_norm_1pw_torch,
    swiglu7_interleaved as swiglu7_interleaved_torch,
)

_ROPE_BANDS_KEY = "pre_adapter.rope.bands"


@dataclass
class Magi2RefinerTpuHooks:
  """Injectable TPU/Pallas/distributed hooks for :class:`Magi2RefinerTransformerTorch`."""

  grouped_linear: Callable | None = None
  ulysses_s2h: Callable | None = None
  ulysses_h2s: Callable | None = None
  block_grid_attn: Callable | None = None
  fp32_matmul_highest: Callable | None = None
  cp_size: int = 1
  qk_fp8: bool = False


def multi_modality_rms_norm_refiner_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    modality_ids: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    cast_back: bool = True,
) -> torch.Tensor:
  """Gemma ``1 + w`` RMSNorm supporting both ``(D,)`` and ``(3, D)`` weights."""
  if weight.ndim == 1 or modality_ids is None:
    return rms_norm_1pw_torch(x, weight, eps=eps, cast_back=cast_back)
  x_f = x.float()
  rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
  w_tok = weight.float().index_select(0, modality_ids)
  if x_f.ndim == 3:
    w_tok = w_tok.unsqueeze(1)
  y = x_f * rstd * (1.0 + w_tok)
  return y.to(x.dtype) if cast_back else y


def refiner_block_grid_attn_ref_torch(
    q: torch.Tensor,  # (S_total, H_q, D)
    k: torch.Tensor,  # (S_total, H_kv, D)
    v: torch.Tensor,  # (S_total, H_kv, D)
    kv_blen: torch.Tensor,  # (num_kv_blocks,) int32
    layout: Any,
) -> torch.Tensor:
  """Exact float32 CPU reference for 3D block-grid sparse + full text-tail attention."""
  s_tot, h_q, d = q.shape
  h_kv = k.shape[1]
  scale = d ** -0.5
  out_dtype = q.dtype

  if h_kv != h_q:
    rep = h_q // h_kv
    k_f = k.float().repeat_interleave(rep, dim=1)
    v_f = v.float().repeat_interleave(rep, dim=1)
  else:
    k_f = k.float()
    v_f = v.float()
  q_f = q.float()

  bvol = int(layout.block_vol)
  nt, nh, nw = layout.block_grid_thw
  rt, rh, rw = layout.radius
  n_vid_blocks = int(layout.n_vid_blocks)
  n_vid_pad = int(layout.n_vid_pad)
  n_tail_blocks = int(layout.n_tail_blocks)
  n_tail_pad = int(layout.n_tail_pad)
  blen_cpu = kv_blen.detach().cpu().numpy().astype(np.int64)

  # Build per-token validity mask across all KV tokens.
  valid_kv = np.zeros((s_tot,), dtype=np.bool_)
  for bi in range(n_vid_blocks + n_tail_blocks):
    vl = int(blen_cpu[bi])
    if vl > 0:
      valid_kv[bi * bvol : bi * bvol + vl] = True
  valid_kv_t = torch.from_numpy(valid_kv).to(device=q.device)

  out = torch.zeros((s_tot, h_q, d), device=q.device, dtype=torch.float32)

  # 1. Video query blocks: attend to 3D neighbor blocks + full text tail.
  for qt in range(nt):
    t0, t1 = max(0, qt - rt), min(nt, qt + rt + 1)
    for qh in range(nh):
      h0, h1 = max(0, qh - rh), min(nh, qh + rh + 1)
      for qw in range(nw):
        w0, w1 = max(0, qw - rw), min(nw, qw + rw + 1)
        qb = (qt * nh + qh) * nw + qw
        qs, qe = qb * bvol, (qb + 1) * bvol

        idx_chunks = []
        for kt in range(t0, t1):
          for kh in range(h0, h1):
            kb0 = (kt * nh + kh) * nw + w0
            kb1 = (kt * nh + kh) * nw + w1
            idx_chunks.append(torch.arange(kb0 * bvol, kb1 * bvol, device=q.device))
        if n_tail_pad > 0:
          idx_chunks.append(torch.arange(n_vid_pad, n_vid_pad + n_tail_pad, device=q.device))
        kv_idx = torch.cat(idx_chunks, dim=0)
        kv_valid = valid_kv_t.index_select(0, kv_idx)

        q_b = q_f[qs:qe].permute(1, 0, 2)  # (H, bvol, D)
        k_b = k_f.index_select(0, kv_idx).permute(1, 0, 2)  # (H, K, D)
        v_b = v_f.index_select(0, kv_idx).permute(1, 0, 2)
        logits = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale
        logits = logits.masked_fill(~kv_valid[None, None, :], -1e9)
        probs = torch.softmax(logits, dim=-1)
        o_b = torch.matmul(probs, v_b).permute(1, 0, 2)
        out[qs:qe] = o_b

  # 2. Tail (text) query tokens: attend to all valid video + tail tokens.
  if n_tail_pad > 0:
    q_t = q_f[n_vid_pad:s_tot].permute(1, 0, 2)
    k_all = k_f.permute(1, 0, 2)
    v_all = v_f.permute(1, 0, 2)
    logits_t = torch.matmul(q_t, k_all.transpose(-1, -2)) * scale
    logits_t = logits_t.masked_fill(~valid_kv_t[None, None, :], -1e9)
    probs_t = torch.softmax(logits_t, dim=-1)
    out[n_vid_pad:s_tot] = torch.matmul(probs_t, v_all).permute(1, 0, 2)

  return out.to(out_dtype)


class ModalityRMSNormTorch(nn.Module):
  """Exact ``ModalityRMSNorm`` (Gemma ``1 + w``, fp32 zero-init weights)."""

  def __init__(self, dim: int, *, num_modality: int = 1, eps: float = 1e-6) -> None:
    super().__init__()
    self.dim = int(dim)
    self.num_modality = int(num_modality)
    self.eps = float(eps)
    shape = (dim,) if num_modality == 1 else (num_modality, dim)
    self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

  def forward(
      self,
      x: torch.Tensor,
      modality_ids: torch.Tensor | None = None,
      *,
      cast_back: bool = True,
  ) -> torch.Tensor:
    return multi_modality_rms_norm_refiner_torch(
        x,
        self.weight,
        modality_ids if self.num_modality > 1 else None,
        eps=self.eps,
        cast_back=cast_back,
    )


class ModalityLinearTorch(nn.Module):
  """Exact ``ModalityLinear`` (``m=1`` plain matmul, ``m=3`` ragged grouped matmul).

  Stores ``weight`` as ``(out, in)`` for ``m=1`` and ``(3, in, out)`` for ``m=3``
  so ``torch.tpu.ops.ragged_dot`` runs directly without runtime transpose!
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      num_modality: int = 1,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.num_modality = int(num_modality)
    shape = (
        (out_features, in_features)
        if num_modality == 1
        else (num_modality, in_features, out_features)
    )
    self.weight = nn.Parameter(torch.empty(shape, dtype=dtype))

  def forward(
      self,
      x: torch.Tensor,
      group_sizes: torch.Tensor | None = None,
      modality_ids: torch.Tensor | None = None,
      *,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    if self.num_modality == 1:
      return F.linear(x.to(self.weight.dtype), self.weight)
    x_c = x.to(self.weight.dtype)
    if hooks is not None and hooks.grouped_linear is not None and group_sizes is not None:
      return hooks.grouped_linear(x_c, self.weight, group_sizes, modality_ids)
    # Exact CPU fallback (monotonically sorted or arbitrary modality_ids)
    out = torch.zeros((x_c.shape[0], self.out_features), device=x_c.device, dtype=x_c.dtype)
    if modality_ids is not None:
      for m in range(self.num_modality):
        mask = (modality_ids == m).unsqueeze(-1)
        out = torch.where(mask, torch.matmul(x_c, self.weight[m]), out)
    elif group_sizes is not None:
      off = 0
      for m in range(self.num_modality):
        cnt = int(group_sizes[m].item())
        if cnt > 0:
          out[off : off + cnt] = torch.matmul(x_c[off : off + cnt], self.weight[m])
          off += cnt
    return out


class RefinerPreAdapterTorch(nn.Module):
  """Exact ``magi2_refiner.Adapter`` (fp32, ``bias=True``)."""

  def __init__(self, config: Magi2RefinerConfig) -> None:
    super().__init__()
    self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
    self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
    self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=torch.float32)

  def _lin(self, lin: nn.Linear, x: torch.Tensor, hooks: Magi2RefinerTpuHooks | None) -> torch.Tensor:
    xf = x.float()
    if hooks is not None and hooks.fp32_matmul_highest is not None:
      return hooks.fp32_matmul_highest(xf, lin.weight.t()) + lin.bias
    return lin(xf)

  def forward(
      self,
      video_x: torch.Tensor,
      audio_x: torch.Tensor | None,
      text_x: torch.Tensor | None,
      *,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    parts = [self._lin(self.video_embedder, video_x, hooks)]
    if audio_x is not None and audio_x.shape[0] > 0:
      parts.append(self._lin(self.audio_embedder, audio_x, hooks))
    if text_x is not None and text_x.shape[0] > 0:
      parts.append(self._lin(self.text_embedder, text_x, hooks))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


class RefinerPostAdapterTorch(nn.Module):
  """Exact ``magi2_refiner.PostAdapter`` (fp32, ``bias=False``)."""

  def __init__(self, config: Magi2RefinerConfig) -> None:
    super().__init__()
    self.final_norm_video = ModalityRMSNormTorch(config.hidden_size, num_modality=1, eps=config.rms_norm_eps)
    self.final_norm_audio = ModalityRMSNormTorch(config.hidden_size, num_modality=1, eps=config.rms_norm_eps)
    self.final_linear_video = nn.Linear(config.hidden_size, config.video_in_channels, bias=False, dtype=torch.float32)
    self.final_linear_audio = nn.Linear(config.hidden_size, config.audio_in_channels, bias=False, dtype=torch.float32)

  def _lin(self, lin: nn.Linear, x: torch.Tensor, hooks: Magi2RefinerTpuHooks | None) -> torch.Tensor:
    xf = x.float()
    if hooks is not None and hooks.fp32_matmul_highest is not None:
      return hooks.fp32_matmul_highest(xf, lin.weight.t())
    return lin(xf)

  def forward(
      self,
      x_video: torch.Tensor,
      *,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    xv = self.final_norm_video(x_video.float(), cast_back=False)
    return self._lin(self.final_linear_video, xv, hooks)


class RefinerAttentionTorch(nn.Module):
  """Exact ``magi2_refiner.Attention`` (GQA 32Q/8KV, QK-Norm, Partial RoPE, Gate)."""

  def __init__(
      self,
      config: Magi2RefinerConfig,
      *,
      num_modality: int,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.config = config
    self.num_modality = int(num_modality)
    self.num_heads_q = config.num_heads_q
    self.num_heads_kv = config.num_heads_kv
    self.head_dim = config.head_dim
    self.q_size = config.num_heads_q * config.head_dim
    self.kv_size = config.num_heads_kv * config.head_dim
    self.dtype = dtype

    self.pre_norm = ModalityRMSNormTorch(config.hidden_size, num_modality=num_modality, eps=config.rms_norm_eps)
    self.linear_qkv = ModalityLinearTorch(
        config.hidden_size, self.q_size + 2 * self.kv_size, num_modality=num_modality, dtype=dtype
    )
    self.linear_g = (
        ModalityLinearTorch(config.hidden_size, config.num_heads_q, num_modality=num_modality, dtype=dtype)
        if config.enable_attn_gating
        else None
    )
    self.linear_proj = ModalityLinearTorch(self.q_size, config.hidden_size, num_modality=num_modality, dtype=dtype)
    self.q_norm = ModalityRMSNormTorch(config.head_dim, num_modality=num_modality, eps=config.rms_norm_eps)
    self.k_norm = ModalityRMSNormTorch(config.head_dim, num_modality=num_modality, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,  # (S_loc, D)
      rope_96: torch.Tensor,  # (S_loc, 96) fp32
      kv_blen: torch.Tensor,  # (num_kv_blocks,) int32
      modality_ids: torch.Tensor | None,
      group_sizes: torch.Tensor | None,
      *,
      layout: Any = None,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    seq_len = hidden_states.shape[0]
    compute_dtype = self.linear_qkv.weight.dtype

    h = self.pre_norm(hidden_states, modality_ids, cast_back=False).to(compute_dtype)
    qkv = self.linear_qkv(h, group_sizes, modality_ids, hooks=hooks).float()
    q = qkv[:, : self.q_size].reshape(seq_len, self.num_heads_q, self.head_dim)
    k = qkv[:, self.q_size : self.q_size + self.kv_size].reshape(seq_len, self.num_heads_kv, self.head_dim)
    v = qkv[:, self.q_size + self.kv_size :].reshape(seq_len, self.num_heads_kv, self.head_dim)

    g = None
    if self.linear_g is not None:
      g = self.linear_g(h, group_sizes, modality_ids, hooks=hooks).float().unsqueeze(-1)

    q = self.q_norm(q, modality_ids, cast_back=False)
    k = self.k_norm(k, modality_ids, cast_back=False)
    q = apply_3d_partial_rope_torch(q, rope_96)
    k = apply_3d_partial_rope_torch(k, rope_96)

    q_in = q.to(compute_dtype)
    k_in = k.to(compute_dtype)
    v_in = v.to(compute_dtype)

    if hooks is not None and hooks.cp_size > 1 and hooks.ulysses_s2h is not None:
      q_full = hooks.ulysses_s2h(q_in)
      k_full = hooks.ulysses_s2h(k_in)
      v_full = hooks.ulysses_s2h(v_in)
    else:
      q_full, k_full, v_full = q_in, k_in, v_in

    if hooks is not None and hooks.block_grid_attn is not None:
      out_full = hooks.block_grid_attn(q_full, k_full, v_full, kv_blen)
    else:
      assert layout is not None, "CPU reference attention requires layout"
      out_full = refiner_block_grid_attn_ref_torch(q_full, k_full, v_full, kv_blen, layout)

    if hooks is not None and hooks.cp_size > 1 and hooks.ulysses_h2s is not None:
      attn_out = hooks.ulysses_h2s(out_full)
    else:
      attn_out = out_full

    if g is not None:
      attn_out = attn_out.float() * torch.sigmoid(g)

    attn_flat = attn_out.reshape(seq_len, self.q_size).to(compute_dtype)
    return self.linear_proj(attn_flat, group_sizes, modality_ids, hooks=hooks)


class RefinerMLPTorch(nn.Module):
  """Exact ``magi2_refiner.MLP`` (dense interleaved ``swiglu7``, fp32 residual)."""

  def __init__(
      self,
      config: Magi2RefinerConfig,
      *,
      num_modality: int,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.num_modality = int(num_modality)
    self.pre_norm = ModalityRMSNormTorch(config.hidden_size, num_modality=num_modality, eps=config.rms_norm_eps)
    self.up_gate_proj = ModalityLinearTorch(
        config.hidden_size, config.intermediate_size * 2, num_modality=num_modality, dtype=dtype
    )
    self.down_proj = ModalityLinearTorch(
        config.intermediate_size, config.hidden_size, num_modality=num_modality, dtype=dtype
    )

  def forward(
      self,
      x: torch.Tensor,
      modality_ids: torch.Tensor | None,
      group_sizes: torch.Tensor | None,
      *,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    compute_dtype = self.up_gate_proj.weight.dtype
    h = self.pre_norm(x, modality_ids, cast_back=False).to(compute_dtype)
    u = self.up_gate_proj(h, group_sizes, modality_ids, hooks=hooks).float()
    u = swiglu7_interleaved_torch(u).to(compute_dtype)
    return self.down_proj(u, group_sizes, modality_ids, hooks=hooks).float()


class RefinerLayerTorch(nn.Module):
  """Exact ``magi2_refiner.TransformerLayer`` (plain pre-norm residual, no mHC)."""

  def __init__(
      self,
      config: Magi2RefinerConfig,
      layer_idx: int,
      *,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.layer_idx = int(layer_idx)
    self.num_modality = 3 if layer_idx in config.mm_layers else 1
    self.attention = RefinerAttentionTorch(config, num_modality=self.num_modality, dtype=dtype)
    self.mlp = RefinerMLPTorch(config, num_modality=self.num_modality, dtype=dtype)

  def forward(
      self,
      hidden_states: torch.Tensor,
      rope_96: torch.Tensor,
      kv_blen: torch.Tensor,
      modality_ids: torch.Tensor | None,
      group_sizes: torch.Tensor | None,
      *,
      layout: Any = None,
      hooks: Magi2RefinerTpuHooks | None = None,
  ) -> torch.Tensor:
    attn_out = self.attention(
        hidden_states, rope_96, kv_blen, modality_ids, group_sizes, layout=layout, hooks=hooks
    )
    hidden_states = hidden_states + attn_out
    mlp_out = self.mlp(hidden_states, modality_ids, group_sizes, hooks=hooks)
    return hidden_states + mlp_out


class Magi2RefinerTransformerTorch(nn.Module):
  """Top-level 6.72B Stage 2 MAGI-2 Refiner DiT in pure PyTorch."""

  def __init__(
      self,
      config: Magi2RefinerConfig | None = None,
      *,
      dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.config = config or Magi2RefinerConfig()
    self.dtype = dtype
    self.pre_adapter = RefinerPreAdapterTorch(self.config)
    self.layers = nn.ModuleList([
        RefinerLayerTorch(self.config, i, dtype=dtype)
        for i in range(self.config.num_layers)
    ])
    self.post_adapter = RefinerPostAdapterTorch(self.config)


def _resolve_torch_refiner_param(model: Magi2RefinerTransformerTorch, key: str) -> tuple[nn.Parameter, bool]:
  """Map checkpoint key to ``(nn.Parameter, transpose_last2)``."""
  if key.startswith("block.layers."):
    parts = key.split(".")
    li = int(parts[2])
    sub: Any = model.layers[li]
    for p in parts[3:-1]:
      sub = getattr(sub, p)
    param = getattr(sub, parts[-1])
    # ModalityLinearTorch stores m=3 weights as (3, in, out) for ragged_dot.
    need_t = isinstance(sub, ModalityLinearTorch) and sub.num_modality > 1 and parts[-1] == "weight"
    return param, need_t
  parts = key.split(".")
  sub = model
  for p in parts[:-1]:
    sub = getattr(sub, p)
  return getattr(sub, parts[-1]), False


def load_torch_refiner_from_safetensors(
    checkpoint_dir: str | Path,
    *,
    config: Magi2RefinerConfig | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    h2d_fn: Callable[[torch.Tensor, torch.device], torch.Tensor] | None = None,
    sync_fn: Callable[[], None] | None = None,
    verbose: bool = True,
) -> Magi2RefinerTransformerTorch:
  """Stream-load 6.72B MAGI-2 Refiner weights directly onto ``device`` (zero peak host copy)."""
  from ...safetensors_io import safe_open

  ckpt = Path(checkpoint_dir).expanduser()
  cfg = config or Magi2RefinerConfig()
  idx_path = ckpt / "model.safetensors.index.json"
  if idx_path.exists():
    wm = json.loads(idx_path.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, sh in wm.items():
      by_shard.setdefault(str(ckpt / sh), []).append(k)
  else:
    single = ckpt / "model.safetensors"
    with safe_open(str(single), framework="pt") as r:
      by_shard = {str(single): list(r.keys())}

  with torch.device("meta"):
    model = Magi2RefinerTransformerTorch(cfg, dtype=dtype)

  t0 = time.perf_counter()
  loaded = 0
  for shard_path, keys in sorted(by_shard.items()):
    key_set = set(keys)
    with safe_open(shard_path, framework="pt") as reader:
      ordered = [k for k in reader.keys() if k in key_set]
      for key in ordered:
        if key == _ROPE_BANDS_KEY:
          continue
        param_meta, need_t = _resolve_torch_refiner_param(model, key)
        t = reader.get_tensor(key)
        if t.dtype != param_meta.dtype:
          t = t.to(param_meta.dtype)
        if need_t:
          # Checkpoint stores (3 * out, in) -> reshape (3, out, in) -> transpose (3, in, out)
          m, in_f, out_f = param_meta.shape
          t = t.reshape(m, out_f, in_f).transpose(1, 2).contiguous()
        elif tuple(t.shape) != tuple(param_meta.shape):
          t = t.reshape(param_meta.shape).contiguous()
        if device is not None:
          t = h2d_fn(t, device) if h2d_fn is not None else t.to(device)
        # Replace meta parameter on owning module cleanly.
        if key.startswith("block.layers."):
          parts = key.split(".")
          owner: Any = model.layers[int(parts[2])]
          for p in parts[3:-1]:
            owner = getattr(owner, p)
          setattr(owner, parts[-1], nn.Parameter(t, requires_grad=False))
        else:
          parts = key.split(".")
          owner = model
          for p in parts[:-1]:
            owner = getattr(owner, p)
          setattr(owner, parts[-1], nn.Parameter(t, requires_grad=False))
        loaded += 1
    if sync_fn is not None:
      sync_fn()

  expected = sum(1 for _ in model.parameters())
  if loaded != expected:
    raise ValueError(f"Loaded {loaded} Refiner params, expected {expected}")
  if verbose:
    print(
        f"[torch_refiner_dit] Streamed {loaded} Refiner tensors ({dtype}) in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
  return model
