"""Native 8-rank ``torch_tpu`` backend for MOVA (360p / 720p Dual-Tower Audio-Video DiT).

Runs pure-PyTorch :class:`MovaTowerTorch` (high-noise + low-noise 40L video experts
and 30L audio tower) + :class:`MovaBridgeTorch` (30L bidirectional bridge) across
8 ``torch_tpu`` processes in a 2D ``(cfg=2, sp=4)`` mesh:
  * Rank mapping: ``cfg_rank = rank // sp`` (0=pos, 1=neg), ``sp_rank = rank % sp``.
  * Both video experts + audio + bridge stay whole-resident on TPU (bf16 ~62 GB,
    on-TPU W8A8 E4M3 FP8 ~31 GB — fits comfortably in v7x 95 GiB HBM).
  * Fused Q/K/V Ulysses ``all_to_all`` (2 collectives/block instead of 4) +
    fixed-m zero-mean-key Pallas ``splash`` / ``sparse_splash`` (``5888x2048``
    tiles on v7x) wrapped in ``AbstractDevice('TPU7x', 1, 'tpu')`` ``jax_op``.
  * Patchified local latent tokens stay 100% resident on-TPU across all 50 steps
    (zero per-step D2H/H2D, zero ``sp`` latent all-gather inside the loop).
  * Supports ``--fp8``, ``--qk-fp8``, ``--sparse-window``, ``--step-cache-thresh``,
    ``--lossy``, and ``DT_AOT_PERSIST`` cross-process AOT artifact persistence.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "False")

import argparse
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ..models.mova import jax_dit as _jd  # noqa: E402
from ..models.mova.torch_dit import (  # noqa: E402
    MovaBridgeTorch,
    MovaTowerTorch,
    MovaTpuHooks,
    dense_attention_mova_torch,
    load_torch_mova_bridge,
    load_torch_mova_tower,
    patchify_video_torch,
)

_REG: dict[Any, Any] = {}


def _sync() -> None:
  if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
    torch.accelerator.synchronize()
  elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
    torch.tpu.synchronize()


def _fast_h2d_no_pin(t: torch.Tensor, device: torch.device) -> torch.Tensor:
  if not t.is_contiguous():
    t = t.contiguous()
  return t.to(device)


def _c10d_all_to_all(x: torch.Tensor, group_name: str, sp: int) -> torch.Tensor:
  n = x.shape[0]
  splits = [n // sp] * sp
  out = torch.ops._c10d_functional.all_to_all_single(x.contiguous(), splits, splits, group_name)
  return torch.ops._c10d_functional.wait_tensor(out)


def _c10d_all_gather(x: torch.Tensor, group_name: str, sp: int) -> torch.Tensor:
  out = torch.ops._c10d_functional.all_gather_into_tensor(x.contiguous(), sp, group_name)
  return torch.ops._c10d_functional.wait_tensor(out)


# ---------------------------------------------------------------------------
# Pallas / JAX custom ops (fp8 linear + dense/sparse SplashAttention)
# ---------------------------------------------------------------------------


def _get_or_register_mova_fp8_linear():
  key = "mova_fp8_linear_v1"
  if key in _REG:
    return _REG[key]

  def fn(x: jax.Array, qw: jax.Array, sc: jax.Array, bias: jax.Array) -> jax.Array:
    f8_max = jnp.float32(448.0)
    x32 = x.astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True), jnp.float32(1e-12)) / f8_max
    qx = jnp.clip(x32 / x_scale, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        qx, qw, (((x.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    y = (acc * x_scale * sc).astype(x.dtype)
    return y + bias.astype(x.dtype)

  op = torch.tpu.pallas.jax_op("pallas::mova_fp8_linear_v1", fn)
  _REG[key] = op
  return op


def _resolve_v7_blocks(sq: int, sk: int, head_dim: int, block_kv: int | None = None) -> tuple[int, int, int]:
  from diffusers_tpu.kernels import block_sizes

  blocks = block_sizes.resolve_block_sizes(int(sq), int(sk), int(head_dim), {}, "residual-fixedm", block_sizes.V7_SPEC)
  bq = int(blocks["block_q"])
  bkv = int(blocks["block_kv"])
  bkvc = int(blocks["block_kv_compute"])
  if block_kv is not None:
    bkv = min(int(block_kv), int(sk))
    bkvc = min(bkvc, bkv)
  return bq, bkv, bkvc


def _get_or_register_mova_splash(
    tag: str,
    sq: int,
    sk: int,
    heads_local: int,
    head_dim: int,
    *,
    qk_fp8: bool = False,
    block_kv: int | None = None,
    kv_table: list[list[int]] | None = None,
):
  sp_tag = f"sp{len(kv_table)}x{sum(len(r) for r in kv_table)}" if kv_table is not None else "dense"
  bq, bkv, bkvc = _resolve_v7_blocks(sq, sk, head_dim, block_kv=block_kv)
  key = ("mova_splash", tag, int(sq), int(sk), int(heads_local), int(head_dim), bool(qk_fp8), bq, bkv, bkvc, sp_tag)
  if key in _REG:
    return _REG[key]

  from diffusers_tpu.kernels import sparse_splash, splash

  tpu_mesh = jax.sharding.AbstractMesh(
      (1,), ("_tpu",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
  )
  _qk_fp8 = bool(qk_fp8)
  _kv_table = kv_table

  def fn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    # Exact match to mova/jax_dit._splash_local: zero-mean keys + fixed-m softmax.
    k = k - jnp.mean(k, axis=2, keepdims=True)
    with jax.sharding.use_abstract_mesh(tpu_mesh):
      if _kv_table is not None:
        return sparse_splash.attention(
            q, k, v, scale=None, block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
            use_fixed_m=True, qk_fp8=_qk_fp8, kv_table=_kv_table,
        )
      return splash.attention(
          q, k, v, scale=None, block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
          use_fixed_m=True, qk_fp8=_qk_fp8,
      )

  op_name = f"pallas::mova_{tag}_{sp_tag}_{'fp8' if _qk_fp8 else 'bf16'}_q{sq}_k{sk}_h{heads_local}_bq{bq}_bkv{bkv}"
  op = torch.tpu.pallas.jax_op(op_name, fn)
  _REG[key] = op
  return op


def build_mova_torchtpu_hooks(
    *,
    rank: int,
    sp_size: int,
    sp_group_name: str,
    n_vid: int,
    n_aud: int,
    n_txt: int,
    vheads: int,
    head_dim: int,
    bvh: int,
    fp8: bool = False,
    qk_fp8: bool = False,
    block_kv: int | None = None,
    sparse_kv_table: list[list[int]] | None = None,
) -> MovaTpuHooks:
  assert vheads % sp_size == 0
  assert n_vid % sp_size == 0
  s_loc = n_vid // sp_size
  h_loc = vheads // sp_size

  vid_self_op = _get_or_register_mova_splash(
      "vself", n_vid, n_vid, h_loc, head_dim,
      qk_fp8=qk_fp8, block_kv=block_kv, kv_table=sparse_kv_table,
  )
  vid_cross_op = _get_or_register_mova_splash("vcross", s_loc, n_txt, vheads, head_dim, qk_fp8=False)
  a2v_op = _get_or_register_mova_splash("a2v", s_loc, n_aud, bvh, head_dim, qk_fp8=False)
  fp8_op = _get_or_register_mova_fp8_linear() if fp8 else None

  if rank == 0:
    sp_desc = f"sparse({sum(len(r) for r in sparse_kv_table)} tiles)" if sparse_kv_table else "dense"
    print(f"[torchtpu_mova] hooks({sp_desc}): sp={sp_size} n_vid={n_vid} (s_loc={s_loc}, h_loc={h_loc}) "
          f"fp8={fp8} qk_fp8={qk_fp8}", flush=True)

  def _video_self_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, sl, h, d = q.shape
    if sp_size > 1:
      # Fused Q+K+V Ulysses all_to_all (1 collective instead of 3!)
      qkv = torch.cat((q, k, v), dim=0)  # [3B, S_loc, H, D]
      qkv_s = qkv.view(3 * b, sl, sp_size, h_loc, d).permute(2, 0, 1, 3, 4).contiguous()
      qkv_g = _c10d_all_to_all(qkv_s, sp_group_name, sp_size)  # [sp, 3B, S_loc, H_loc, D]
      qkv_h = qkv_g.permute(1, 3, 0, 2, 4).contiguous().view(3 * b, h_loc, n_vid, d)
      qh, kh, vh = qkv_h[:b].contiguous(), qkv_h[b : 2 * b].contiguous(), qkv_h[2 * b :].contiguous()
    else:
      qh = q.transpose(1, 2).contiguous()
      kh = k.transpose(1, 2).contiguous()
      vh = v.transpose(1, 2).contiguous()

    oh = vid_self_op(qh, kh, vh)  # [B, H_loc, N_vid, D]
    if sp_size > 1:
      o_s = oh.view(b, h_loc, sp_size, sl, d).permute(2, 0, 3, 1, 4).contiguous()
      o_g = _c10d_all_to_all(o_s, sp_group_name, sp_size)  # [sp, B, S_loc, H_loc, D]
      return o_g.permute(1, 2, 0, 3, 4).contiguous().view(b, sl, h, d)
    return oh.transpose(1, 2).contiguous()

  def _video_cross_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    oh = vid_cross_op(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
    return oh.transpose(1, 2).contiguous()

  def _a2v_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    oh = a2v_op(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
    return oh.transpose(1, 2).contiguous()

  def _v2a_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Replicated audio queries [B, 400, H, D], local video keys/values [B, S_loc, H, D]
    b, sq, h, d = q.shape
    scale = 1.0 / math.sqrt(d)
    qt = q.transpose(1, 2)  # [B, H, 400, D]
    kt = k.transpose(1, 2)  # [B, H, S_loc, D]
    vt = v.transpose(1, 2)  # [B, H, S_loc, D]
    sc = torch.matmul(qt, kt.transpose(-1, -2)).float() * scale
    if sp_size <= 1:
      p = torch.softmax(sc, dim=-1).to(v.dtype)
      return torch.matmul(p, vt).transpose(1, 2).contiguous()

    m_loc = sc.amax(dim=-1, keepdim=True)  # [B, H, 400, 1]
    e = torch.exp(sc - m_loc)
    l_loc = e.sum(dim=-1, keepdim=True)  # [B, H, 400, 1]
    o_loc = torch.matmul(e.to(v.dtype), vt).float()  # [B, H, 400, D]
    # Pack [o_loc (D), l_loc (1), m_loc (1)] -> single all_gather (1 collective instead of 3!)
    pack = torch.cat((o_loc, l_loc, m_loc), dim=-1).permute(0, 2, 1, 3).contiguous()  # [B, 400, H, D+2]
    gathered = _c10d_all_gather(pack, sp_group_name, sp_size).view(sp_size, b, sq, h, d + 2)
    o_all = gathered[..., :d]
    l_all = gathered[..., d : d + 1]
    m_all = gathered[..., d + 1 : d + 2]
    m_max = m_all.amax(dim=0, keepdim=True)
    w = torch.exp(m_all - m_max)
    num = (o_all * w).sum(dim=0)
    den = (l_all * w).sum(dim=0)
    return (num / den).to(v.dtype).contiguous()

  def _fp8_lin(x: torch.Tensor, qw: torch.Tensor, sc: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return fp8_op(x.contiguous(), qw.contiguous(), sc.contiguous(), b.contiguous())

  return MovaTpuHooks(
      video_self_attn=_video_self_attn,
      video_cross_attn=_video_cross_attn,
      a2v_attn=_a2v_attn,
      v2a_attn=_v2a_attn,
      audio_attn=dense_attention_mova_torch,
      fp8_linear=_fp8_lin if fp8 else None,
      sp_size=sp_size,
      qk_fp8=qk_fp8,
  )


# ---------------------------------------------------------------------------
# Module-level compiled functions (closure-free for DT_AOT_PERSIST!)
# ---------------------------------------------------------------------------

_ACTIVE_MOVA_HOOKS: MovaTpuHooks | None = None


def _wrap_spmd_compile(fn, *, use_spmd_safe: bool = True):
  if use_spmd_safe:
    try:
      from torch.tpu.distributed import spmd_safe
      fn = spmd_safe(fn)
    except Exception:
      pass
  return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)


def _top_mova_pre(
    v_te0, v_te2, v_tp1, v_txt0, v_txt2, v_patch,
    a_te0, a_te2, a_tp1, a_txt0, a_txt2, a_patch,
    lat_tok_loc, cond_tok_loc, lat_a, ctx_row, t_sin, v_dim: int, a_dim: int,
):
  dt = v_patch.weight.dtype
  # Video & audio time embedders (FP32, cast to dt)
  vt = v_te0(t_sin.float()).float()
  vt = v_te2(torch.nn.functional.silu(vt)).float()
  vtm = v_tp1(torch.nn.functional.silu(vt)).float().reshape(-1, 6, v_dim)
  vt, vtm = vt.to(dt), vtm.to(dt)

  at = a_te0(t_sin.float()).float()
  at = a_te2(torch.nn.functional.silu(at)).float()
  atm = a_tp1(torch.nn.functional.silu(at)).float().reshape(-1, 6, a_dim)
  at, atm = at.to(dt), atm.to(dt)

  # Text embeddings
  vctx = v_txt0(ctx_row.to(dt))
  vctx = v_txt2(torch.nn.functional.gelu(vctx.float(), approximate="tanh").to(dt))
  actx = a_txt0(ctx_row.to(dt))
  actx = a_txt2(torch.nn.functional.gelu(actx.float(), approximate="tanh").to(dt))

  # Patch embeddings (FP32 accumulation matching jax_dit.dual_step)
  v_in = torch.cat((lat_tok_loc, cond_tok_loc), dim=-1).to(dt)
  if v_patch.pretranspose:
    vx = (torch.matmul(v_in.float(), v_patch.weight.float()) + v_patch.bias.float()).to(dt)
  else:
    vx = torch.nn.functional.linear(v_in.float(), v_patch.weight.float(), v_patch.bias.float()).to(dt)

  a_in = lat_a.transpose(1, 2).to(dt)
  if a_patch.pretranspose:
    ax = (torch.matmul(a_in.float(), a_patch.weight.float()) + a_patch.bias.float()).to(dt)
  else:
    ax = torch.nn.functional.linear(a_in.float(), a_patch.weight.float(), a_patch.bias.float()).to(dt)

  return vx, ax, vctx, actx, vt, vtm, at, atm


def _top_mova_shared_group(
    v_blks, a_blks, b_v2a, b_a2v,
    vx, ax, vctx, actx, vtm, atm,
    vcos, vsin, acos, asin, bvc, bvs, bac, bas,
):
  h = _ACTIVE_MOVA_HOOKS
  vrope = (bvc, bvs)
  arope = (bac, bas)
  for vb, ab, v2a_m, a2v_m in zip(v_blks, a_blks, b_v2a, b_a2v):
    v2a_out = type(v2a_m).forward(v2a_m, ax, vx, arope, vrope, attn_fn=h.v2a_attn, hooks=h)
    a2v_out = type(a2v_m).forward(a2v_m, vx, ax, vrope, arope, attn_fn=h.a2v_attn, hooks=h)
    vx = type(vb).forward(
        vb, vx + a2v_out, vctx, vtm, vcos, vsin,
        self_attn_fn=h.video_self_attn, cross_attn_fn=h.video_cross_attn, hooks=h,
    )
    ax = type(ab).forward(
        ab, ax + v2a_out, actx, atm, acos, asin,
        self_attn_fn=h.audio_attn, cross_attn_fn=h.audio_attn, hooks=h,
    )
  return vx, ax


def _top_mova_vid_group(v_blks, vx, vctx, vtm, vcos, vsin):
  h = _ACTIVE_MOVA_HOOKS
  for vb in v_blks:
    vx = type(vb).forward(
        vb, vx, vctx, vtm, vcos, vsin,
        self_attn_fn=h.video_self_attn, cross_attn_fn=h.video_cross_attn, hooks=h,
    )
  return vx


def _top_mova_post(v_hmod, v_hlin, a_hmod, a_hlin, vx, ax, vt, at, pt: int, ph: int, pw: int, out_dim: int):
  from ..models.mova.torch_dit import _layer_norm_torch

  mv = v_hmod.unsqueeze(0) + vt.unsqueeze(1)
  v_raw = v_hlin(_layer_norm_torch(vx) * (1.0 + mv[:, 1:2]) + mv[:, 0:1])
  # Reorder patch channels from (pt, ph, pw, out_dim) -> (out_dim, pt, ph, pw) to match patchified lat_v!
  b, s, _ = v_raw.shape
  v_tok = v_raw.view(b, s, pt, ph, pw, out_dim).permute(0, 1, 5, 2, 3, 4).contiguous().view(b, s, -1)

  ma = a_hmod.unsqueeze(0) + at.unsqueeze(1)
  a_out = a_hlin(_layer_norm_torch(ax) * (1.0 + ma[:, 1:2]) + ma[:, 0:1]).transpose(1, 2).contiguous()
  return v_tok, a_out


def _top_mova_cfg_euler(
    lat_tok_loc: torch.Tensor,
    lat_a: torch.Tensor,
    pv_2: torch.Tensor,
    pa_2: torch.Tensor,
    dt_t: torch.Tensor,
    guidance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  pv_f = pv_2.float()
  pa_f = pa_2.float()
  v_cfg = pv_f[1:2] + guidance * (pv_f[0:1] - pv_f[1:2])
  a_cfg = pa_f[1:2] + guidance * (pa_f[0:1] - pa_f[1:2])
  return lat_tok_loc + dt_t * v_cfg, lat_a + dt_t * a_cfg, v_cfg, a_cfg


def _top_mova_vel_sums(
    vc: torch.Tensor, vp: torch.Tensor, ac: torch.Tensor, ap: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  return (
      (vc - vp).abs().sum().reshape(1),
      vc.abs().sum().reshape(1),
      (ac - ap).abs().sum().reshape(1),
      ac.abs().sum().reshape(1),
  )


class CompiledMovaRunnerTT:
  """Grouped-block AOT runner for MOVA dual-tower DiT (dense + optional sparse)."""

  def __init__(
      self,
      v_experts: list[MovaTowerTorch],
      audio: MovaTowerTorch,
      bridge: MovaBridgeTorch,
      *,
      hooks_dense: MovaTpuHooks,
      hooks_sparse: MovaTpuHooks | None,
      cfg_rank: int,
      sp_rank: int,
      sp_size: int,
      n_vid: int,
      n_aud: int,
      aot_group: int = 5,
      fp8: bool = False,
      sparse_tag: str = "dense",
      use_compile: bool = True,
      use_spmd_safe: bool = True,
  ) -> None:
    self.v_experts = v_experts
    self.audio = audio
    self.bridge = bridge
    self.hooks_dense = hooks_dense
    self.hooks_sparse = hooks_sparse
    self.cfg_rank = int(cfg_rank)
    self.sp_rank = int(sp_rank)
    self.sp_size = int(sp_size)
    self.n_layers = len(v_experts[0].blocks)
    self.shared = min(self.n_layers, len(audio.blocks))
    self.aot_group = max(1, int(aot_group))
    assert self.shared % self.aot_group == 0, f"shared={self.shared} not divisible by aot_group={self.aot_group}"
    self.vid_only = self.n_layers - self.shared
    self.aot_group_vid = self.aot_group if (self.vid_only % self.aot_group == 0) else self.vid_only

    self.v_dim = int(v_experts[0].config["dim"])
    self.a_dim = int(audio.config["dim"])
    patch = tuple(int(p) for p in v_experts[0].config["patch_size"])
    self.pt, self.ph, self.pw = patch
    self.v_out_dim = int(v_experts[0].config["out_dim"])

    self._pre_mods = [
        (
            ve.time_embedding_0, ve.time_embedding_2, ve.time_projection_1,
            ve.text_embedding_0, ve.text_embedding_2, ve.patch_embedding,
            audio.time_embedding_0, audio.time_embedding_2, audio.time_projection_1,
            audio.text_embedding_0, audio.text_embedding_2, audio.patch_embedding,
        )
        for ve in v_experts
    ]
    self._post_mods = [
        (ve.head_modulation, ve.head_head, audio.head_modulation, audio.head_head)
        for ve in v_experts
    ]
    self._shared_chunks = [
        [
            (
                [ve.blocks[j] for j in range(s, s + self.aot_group)],
                [audio.blocks[j] for j in range(s, s + self.aot_group)],
                [bridge.v2a[j] for j in range(s, s + self.aot_group)],
                [bridge.a2v[j] for j in range(s, s + self.aot_group)],
            )
            for s in range(0, self.shared, self.aot_group)
        ]
        for ve in v_experts
    ]
    self._vid_chunks = [
        [
            [ve.blocks[j] for j in range(s, s + self.aot_group_vid)]
            for s in range(self.shared, self.n_layers, self.aot_group_vid)
        ]
        for ve in v_experts
    ]

    persist_dir = os.environ.get("DT_AOT_PERSIST", "").strip()
    if persist_dir == "0":
      persist_dir = ""

    if not use_compile:
      self._pre_fn = _top_mova_pre
      self._sh_dense_fn = _top_mova_shared_group
      self._vo_dense_fn = _top_mova_vid_group
      self._sh_sparse_fn = _top_mova_shared_group
      self._vo_sparse_fn = _top_mova_vid_group
      self._post_fn = _top_mova_post
      self._euler_fn = _top_mova_cfg_euler
      self._vel_fn = _top_mova_vel_sums
      return

    c_pre = _wrap_spmd_compile(_top_mova_pre, use_spmd_safe=use_spmd_safe)
    c_sh_d = _wrap_spmd_compile(_top_mova_shared_group, use_spmd_safe=use_spmd_safe)
    c_vo_d = _wrap_spmd_compile(_top_mova_vid_group, use_spmd_safe=use_spmd_safe)
    c_sh_s = _wrap_spmd_compile(_top_mova_shared_group, use_spmd_safe=use_spmd_safe) if hooks_sparse else None
    c_vo_s = _wrap_spmd_compile(_top_mova_vid_group, use_spmd_safe=use_spmd_safe) if hooks_sparse else None
    c_post = _wrap_spmd_compile(_top_mova_post, use_spmd_safe=use_spmd_safe)
    c_euler = _wrap_spmd_compile(_top_mova_cfg_euler, use_spmd_safe=use_spmd_safe)
    c_vel = _wrap_spmd_compile(_top_mova_vel_sums, use_spmd_safe=use_spmd_safe)

    if not persist_dir or not hasattr(c_sh_d, "aot_compile"):
      self._pre_fn = c_pre
      self._sh_dense_fn = c_sh_d
      self._vo_dense_fn = c_vo_d
      self._sh_sparse_fn = c_sh_s
      self._vo_sparse_fn = c_vo_s
      self._post_fn = c_post
      self._euler_fn = c_euler
      self._vel_fn = c_vel
      return

    os.makedirs(persist_dir, exist_ok=True)
    base_salt = (
        "mova_v1",
        self.aot_group,
        self.aot_group_vid,
        self.cfg_rank,
        self.sp_size,
        int(n_vid),
        int(n_aud),
        bool(fp8),
        bool(hooks_dense.qk_fp8),
    )

    def _wrap_persist(tag: str, compiled_fn: Any, extra: str = "") -> Any:
      if compiled_fn is None:
        return None
      h = hashlib.sha1(repr((tag, extra) + base_salt).encode("utf-8")).hexdigest()[:16]
      bin_path = os.path.join(persist_dir, f"mova_{tag}_g{self.aot_group}_{h}.bin")
      art_holder: list[Any] = [None]
      tried: list[bool] = [False]

      def _call(*args):
        if art_holder[0] is not None:
          return art_holder[0](*args)
        if not tried[0]:
          tried[0] = True
          if os.path.exists(bin_path) and hasattr(torch.compiler, "load_compiled_function"):
            t_ld = time.perf_counter()
            try:
              with open(bin_path, "rb") as f:
                art = torch.compiler.load_compiled_function(f, f_globals=globals())
              art_holder[0] = art
              if self.cfg_rank == 0 and self.sp_rank == 0:
                print(f"  [aot/mova] loaded {os.path.basename(bin_path)} in {time.perf_counter() - t_ld:.2f}s", flush=True)
              return art(*args)
            except Exception as exc:
              if self.cfg_rank == 0 and self.sp_rank == 0:
                print(f"  [aot/mova] cached {os.path.basename(bin_path)} unusable ({type(exc).__name__}); recompiling", flush=True)
          try:
            art = compiled_fn.aot_compile((args, {}))
            art_holder[0] = art
            if self.sp_rank == 0 and hasattr(art, "save_compiled_function"):
              tmp_p = f"{bin_path}.tmp.{os.getpid()}"
              art.save_compiled_function(tmp_p)
              os.replace(tmp_p, bin_path)
              if self.cfg_rank == 0:
                print(f"  [aot/mova] saved {os.path.basename(bin_path)}", flush=True)
            return art(*args)
          except Exception as exc:
            if self.cfg_rank == 0 and self.sp_rank == 0:
              print(f"  [aot/mova] {tag} fallback ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return compiled_fn(*args)

      return _call

    self._pre_fn = _wrap_persist("pre_v4", c_pre)
    self._sh_dense_fn = _wrap_persist("sh_dense", c_sh_d)
    self._vo_dense_fn = _wrap_persist("vo_dense", c_vo_d)
    self._sh_sparse_fn = _wrap_persist("sh_sparse", c_sh_s, extra=sparse_tag)
    self._vo_sparse_fn = _wrap_persist("vo_sparse", c_vo_s, extra=sparse_tag)
    self._post_fn = _wrap_persist("post", c_post)
    self._euler_fn = _wrap_persist("euler", c_euler)
    self._vel_fn = _wrap_persist("vel", c_vel)

  @torch.no_grad()
  def forward_local(
      self,
      expert_idx: int,
      lat_tok_loc: torch.Tensor,
      cond_tok_loc: torch.Tensor,
      lat_a: torch.Tensor,
      ctx_row: torch.Tensor,
      t_sin: torch.Tensor,
      ropes_local: tuple[torch.Tensor, ...],
      *,
      use_sparse: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    global _ACTIVE_MOVA_HOOKS
    _ACTIVE_MOVA_HOOKS = self.hooks_sparse if (use_sparse and self.hooks_sparse is not None) else self.hooks_dense
    sh_fn = self._sh_sparse_fn if (use_sparse and self._sh_sparse_fn is not None) else self._sh_dense_fn
    vo_fn = self._vo_sparse_fn if (use_sparse and self._vo_sparse_fn is not None) else self._vo_dense_fn

    vx, ax, vctx, actx, vt, vtm, at, atm = self._pre_fn(
        *self._pre_mods[expert_idx],
        lat_tok_loc, cond_tok_loc, lat_a, ctx_row, t_sin, self.v_dim, self.a_dim,
    )
    vcos, vsin, acos, asin, bvc, bvs, bac, bas = ropes_local
    for vb, ab, v2a_m, a2v_m in self._shared_chunks[expert_idx]:
      vx, ax = sh_fn(
          vb, ab, v2a_m, a2v_m,
          vx, ax, vctx, actx, vtm, atm,
          vcos, vsin, acos, asin, bvc, bvs, bac, bas,
      )
    for vb in self._vid_chunks[expert_idx]:
      vx = vo_fn(vb, vx, vctx, vtm, vcos, vsin)
    return self._post_fn(*self._post_mods[expert_idx], vx, ax, vt, at, self.pt, self.ph, self.pw, self.v_out_dim)


def main() -> None:
  p = argparse.ArgumentParser(description="8-rank native torch_tpu runner for MOVA DiT")
  p.add_argument("--repo", required=True, help="Local MOVA snapshot directory")
  p.add_argument("--stage0-npz", default="", help="Stage 0 conditioning .npz from mova/pipeline.py --encode-only")
  p.add_argument("--latents-out", default="mova_tt_latents.npz", help="Output denoised latents .npz")
  p.add_argument("--mode", choices=("denoise", "bench"), default="denoise")
  p.add_argument("--cfg-parallel", type=int, default=2, choices=(1, 2))
  p.add_argument("--fp8", action="store_true")
  p.add_argument("--qk-fp8", action="store_true")
  p.add_argument("--block-kv", type=int, default=None)
  p.add_argument("--sparse-window", type=int, default=0)
  p.add_argument("--sparse-dense-steps", type=int, default=8)
  p.add_argument("--step-cache-thresh", type=float, default=0.0)
  p.add_argument("--step-cache-max-skip", type=int, default=2)
  p.add_argument("--lossy", action="store_true")
  p.add_argument("--cfg-scale", type=float, default=5.0)
  p.add_argument("--aot-group", type=int, default=5)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--warmup", type=int, default=1)
  p.add_argument("--bench-iters", type=int, default=5)
  p.add_argument("--no-compile", action="store_true")
  p.add_argument("--no-spmd-safe", action="store_true")
  args = p.parse_args()

  if args.lossy:
    args.qk_fp8 = True
    if args.sparse_window == 0:
      args.sparse_window = 12
    if args.step_cache_thresh == 0.0:
      args.step_cache_thresh = 0.25

  rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    dist.init_process_group(backend="tpu")
  device = torch.device("tpu:0")

  cp = args.cfg_parallel
  assert world % cp == 0, f"world={world} must be divisible by cfg_parallel={cp}"
  sp = world // cp
  cfg_rank = rank // sp
  sp_rank = rank % sp

  sp_pg = cfg_pg = None
  sp_group_name = ""
  if world > 1:
    for cg in range(cp):
      ranks = [cg * sp + sr for sr in range(sp)]
      pg = dist.new_group(ranks=ranks)
      if cg == cfg_rank:
        sp_pg = pg
    if cp > 1:
      for sr in range(sp):
        ranks = [cg * sp + sr for cg in range(cp)]
        pg = dist.new_group(ranks=ranks)
        if sr == sp_rank:
          cfg_pg = pg
    sp_group_name = sp_pg.group_name if sp_pg is not None else ""

  def say(msg: str) -> None:
    if rank == 0:
      print(msg, flush=True)

  say(f"[torchtpu_mova] mode={args.mode} world={world} ({cp} cfg x {sp} sp) "
      f"fp8={args.fp8} qk_fp8={args.qk_fp8} sparse_w={args.sparse_window} "
      f"cache={args.step_cache_thresh} aot_group={args.aot_group}")

  # 1. Load Stage 0 conditioning (or synthetic 720p tensors in bench mode)
  if args.stage0_npz:
    s0 = np.load(args.stage0_npz, allow_pickle=False)
    context_2 = np.asarray(s0["context"], dtype=np.float32)       # [2, 512, 4096]
    condition_np = np.asarray(s0["condition"], dtype=np.float32)  # [1, 20, T, H/8, W/8]
    lat_v_init = np.asarray(s0["lat_v"], dtype=np.float32)        # [1, 16, T, H/8, W/8]
    lat_a_init = np.asarray(s0["lat_a"], dtype=np.float32)        # [1, 128, L_a]
    s_from = np.asarray(s0["s_from"], dtype=np.float32)
    s_to = np.asarray(s0["s_to"], dtype=np.float32)
    expert_arr = np.asarray(s0["expert"], dtype=np.int32)
    sinusoids = np.asarray(s0["sinusoids"], dtype=np.float32)
    fps = float(s0["fps"])
  else:
    rng = np.random.default_rng(42)
    context_2 = (rng.standard_normal((2, 512, 4096)) * 0.1).astype(np.float32)
    condition_np = np.zeros((1, 20, 49, 90, 160), dtype=np.float32)
    lat_v_init = rng.standard_normal((1, 16, 49, 90, 160)).astype(np.float32)
    lat_a_init = rng.standard_normal((1, 128, 400)).astype(np.float32)
    t_sc = np.linspace(1000.0, 20.0, 50, dtype=np.float32)
    s_from = np.linspace(1.0, 0.02, 50, dtype=np.float32)
    s_to = np.append(s_from[1:], np.float32(0.0))
    expert_arr = (t_sc < 900.0).astype(np.int32)
    sinusoids = _jd.time_sinusoid(t_sc)
    fps = 24.0

  n_steps = len(s_from)
  _, _, t_lat, h_lat, w_lat = lat_v_init.shape
  n_aud = int(lat_a_init.shape[2])
  n_txt = int(context_2.shape[1])

  # 2. Stream-load audio + bridge + both video experts directly onto TPU (pre-transposed BF16 / W8A8 FP8)
  t_ld = time.perf_counter()
  audio = load_torch_mova_tower(
      os.path.join(args.repo, "audio_dit"),
      dtype=torch.bfloat16, fp8=args.fp8, pretranspose_bf16=True,
      device=device, h2d_fn=_fast_h2d_no_pin, sync_fn=_sync, verbose=(rank == 0),
  )
  v_experts = [
      load_torch_mova_tower(
          os.path.join(args.repo, d),
          dtype=torch.bfloat16, fp8=args.fp8, pretranspose_bf16=True,
          device=device, h2d_fn=_fast_h2d_no_pin, sync_fn=_sync, verbose=(rank == 0),
      )
      for d in ("video_dit", "video_dit_2")
  ]
  shared = min(len(v_experts[0].blocks), len(audio.blocks))
  bridge = load_torch_mova_bridge(
      os.path.join(args.repo, "dual_tower_bridge"), shared,
      dtype=torch.bfloat16, fp8=args.fp8, pretranspose_bf16=True,
      device=device, h2d_fn=_fast_h2d_no_pin, sync_fn=_sync, verbose=(rank == 0),
  )
  _sync()
  load_s = time.perf_counter() - t_ld
  say(f"[torchtpu_mova] All 4 DiT modules loaded in {load_s:.2f}s.")

  patch = tuple(int(p) for p in v_experts[0].config["patch_size"])
  grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
  n_vid = grid[0] * grid[1] * grid[2]
  assert n_vid % sp == 0, f"n_vid={n_vid} must be divisible by sp={sp}"
  s_loc = n_vid // sp
  lo, hi = sp_rank * s_loc, (sp_rank + 1) * s_loc

  vheads = int(v_experts[0].config["num_heads"])
  vhd = int(v_experts[0].config["dim"]) // vheads
  ahd = int(audio.config["dim"]) // int(audio.config["num_heads"])
  bvh = int(bridge.config["visual_hidden_dim"]) // int(bridge.config["head_dim"])

  # 3. Build Pallas Splash + Sparse-Splash hooks & AOT runner
  hooks_dense = build_mova_torchtpu_hooks(
      rank=rank, sp_size=sp, sp_group_name=sp_group_name,
      n_vid=n_vid, n_aud=n_aud, n_txt=n_txt,
      vheads=vheads, head_dim=vhd, bvh=bvh,
      fp8=args.fp8, qk_fp8=args.qk_fp8, block_kv=args.block_kv,
      sparse_kv_table=None,
  )
  hooks_sparse = None
  sparse_tag = "dense"
  if args.sparse_window > 0:
    bq_sp, bkv_sp, _ = _resolve_v7_blocks(n_vid, n_vid, vhd, block_kv=args.block_kv)
    sp_table = _jd.build_mova_sparse_kv_table(grid, args.sparse_window, block_q=bq_sp, block_kv=bkv_sp)
    kept = sum(len(r) for r in sp_table)
    tot = len(sp_table) * (-(-n_vid // min(bkv_sp, n_vid)))
    sparse_tag = f"sw{args.sparse_window}_{bq_sp}x{bkv_sp}"
    say(f"[torchtpu_mova] Sparse window={args.sparse_window} ({bq_sp}x{bkv_sp}, density={kept / max(1, tot):.1%})")
    hooks_sparse = build_mova_torchtpu_hooks(
        rank=rank, sp_size=sp, sp_group_name=sp_group_name,
        n_vid=n_vid, n_aud=n_aud, n_txt=n_txt,
        vheads=vheads, head_dim=vhd, bvh=bvh,
        fp8=args.fp8, qk_fp8=args.qk_fp8, block_kv=args.block_kv,
        sparse_kv_table=sp_table,
    )

  runner = CompiledMovaRunnerTT(
      v_experts, audio, bridge,
      hooks_dense=hooks_dense, hooks_sparse=hooks_sparse,
      cfg_rank=cfg_rank, sp_rank=sp_rank, sp_size=sp,
      n_vid=n_vid, n_aud=n_aud, aot_group=args.aot_group,
      fp8=args.fp8, sparse_tag=sparse_tag,
      use_compile=not args.no_compile, use_spmd_safe=not args.no_spmd_safe,
  )

  # 4. Precompute local RoPE, local patchified condition & initial latents, and step schedules on TPU
  vcos_np, vsin_np = _jd.video_rope(*grid, vhd)
  acos_np, asin_np = _jd.audio_rope(n_aud, ahd)
  (bvc_np, bvs_np), (bac_np, bas_np) = _jd.bridge_rope(
      grid, n_aud, fps, bridge.config["audio_fps"], bridge.config["head_dim"]
  )
  ropes_local = (
      torch.from_numpy(np.ascontiguousarray(vcos_np[lo:hi])).to(device),
      torch.from_numpy(np.ascontiguousarray(vsin_np[lo:hi])).to(device),
      torch.from_numpy(np.ascontiguousarray(acos_np)).to(device),
      torch.from_numpy(np.ascontiguousarray(asin_np)).to(device),
      torch.from_numpy(np.ascontiguousarray(bvc_np[lo:hi])).to(device),
      torch.from_numpy(np.ascontiguousarray(bvs_np[lo:hi])).to(device),
      torch.from_numpy(np.ascontiguousarray(bac_np)).to(device),
      torch.from_numpy(np.ascontiguousarray(bas_np)).to(device),
  )

  # Patchify condition & lat_v once on host, slice local SP shard -> resident on TPU!
  cond_tok_full = patchify_video_torch(torch.from_numpy(condition_np), patch)  # [1, N_vid, 80]
  lat_tok_full = patchify_video_torch(torch.from_numpy(lat_v_init), patch)     # [1, N_vid, 64]
  cond_tok_loc = cond_tok_full[0, lo:hi, :].contiguous().clone().unsqueeze(0).to(device=device, dtype=torch.float32)
  lat_tok_init_t = lat_tok_full[0, lo:hi, :].contiguous().clone().unsqueeze(0).to(device=device, dtype=torch.float32)
  lat_a_init_t = torch.from_numpy(np.ascontiguousarray(lat_a_init)).to(device=device, dtype=torch.float32)

  rows_per_group = 2 // cp
  r_start = cfg_rank * rows_per_group
  ctx_row = torch.from_numpy(np.ascontiguousarray(context_2[r_start : r_start + rows_per_group])).to(
      device=device, dtype=torch.bfloat16
  )
  t_sin_list = [
      torch.from_numpy(np.ascontiguousarray(sinusoids[i : i + 1])).to(device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]
  dt_list = [
      torch.tensor([[[float(s_to[i] - s_from[i])]]], device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]

  v_tok_ch = lat_tok_init_t.shape[-1]  # 64
  pv_2_buf = torch.empty((2, s_loc, v_tok_ch), device=device, dtype=torch.bfloat16) if cp > 1 else None
  pa_2_buf = torch.empty((2, 128, n_aud), device=device, dtype=torch.bfloat16) if cp > 1 else None

  def _fwd_2rows_tpu(
      ex: int, lat_tok_t: torch.Tensor, lat_a_t: torch.Tensor, t_sin_t: torch.Tensor, use_sparse: bool
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if rows_per_group > 1:
      lt_in = lat_tok_t.repeat(rows_per_group, 1, 1)
      ct_in = cond_tok_loc.repeat(rows_per_group, 1, 1)
      la_in = lat_a_t.repeat(rows_per_group, 1, 1)
    else:
      lt_in, ct_in, la_in = lat_tok_t, cond_tok_loc, lat_a_t
    pv_loc, pa_loc = runner.forward_local(
        ex, lt_in, ct_in, la_in, ctx_row, t_sin_t, ropes_local, use_sparse=use_sparse
    )
    if cp > 1:
      dist.all_gather_into_tensor(pv_2_buf, pv_loc, group=cfg_pg)
      dist.all_gather_into_tensor(pa_2_buf, pa_loc, group=cfg_pg)
      return pv_2_buf, pa_2_buf
    return pv_loc, pa_loc

  if args.mode == "bench":
    use_sp_b = hooks_sparse is not None
    t_w = time.perf_counter()
    pv, pa = _fwd_2rows_tpu(0, lat_tok_init_t, lat_a_init_t, t_sin_list[0], use_sparse=use_sp_b)
    _sync()
    t_comp = time.perf_counter() - t_w
    say(f"[torchtpu_mova] Compile/warmup ({'sparse' if use_sp_b else 'dense'}) done in {t_comp:.2f}s "
        f"| pv std={float(pv.float().cpu().std()):.4f} pa std={float(pa.float().cpu().std()):.4f}")
    ts = []
    for it in range(args.bench_iters):
      t0 = time.perf_counter()
      _fwd_2rows_tpu(0, lat_tok_init_t, lat_a_init_t, t_sin_list[0], use_sparse=use_sp_b)
      _sync()
      dt = (time.perf_counter() - t0) * 1000.0
      ts.append(dt)
      say(f"  [iter {it}] {dt:.1f} ms/fwd")
    say(f"[torchtpu_mova] BENCH SUMMARY ({cp}x{sp}, fp8={args.fp8}, sparse={use_sp_b}, g={args.aot_group}): "
        f"load={load_s:.2f}s compile={t_comp:.2f}s median={float(np.median(ts)):.1f} ms")
  else:
    use_cache = args.step_cache_thresh > 0.0
    cache_start = max(6, int(round(n_steps * 0.14)))
    cache_end = n_steps - max(4, int(round(n_steps * 0.10)))

    for run_i in range(args.runs):
      lat_tok = lat_tok_init_t.clone()
      lat_a = lat_a_init_t.clone()
      prev_v = prev_a = None
      prev_expert = -1
      accum_err = 0.0
      last_rate = 1.0
      consec = 0
      n_cached = 0
      _sync()
      t_run0 = time.perf_counter()

      for i in range(n_steps):
        ex = int(expert_arr[i])
        same_expert = (ex == prev_expert)
        can_cache = (
            use_cache
            and prev_v is not None
            and same_expert
            and cache_start <= i < cache_end
            and consec < args.step_cache_max_skip
            and (accum_err + last_rate) < args.step_cache_thresh
        )
        if can_cache:
          lat_tok = lat_tok + dt_list[i] * prev_v
          lat_a = lat_a + dt_list[i] * prev_a
          accum_err += last_rate
          consec += 1
          n_cached += 1
          continue

        use_sp_i = (hooks_sparse is not None) and (i >= args.sparse_dense_steps)
        pv_2, pa_2 = _fwd_2rows_tpu(ex, lat_tok, lat_a, t_sin_list[i], use_sparse=use_sp_i)
        lat_tok, lat_a, v_cur, a_cur = runner._euler_fn(
            lat_tok, lat_a, pv_2, pa_2, dt_list[i], float(args.cfg_scale)
        )

        if use_cache:
          if prev_v is not None and same_expert:
            dv_s, v_s, da_s, a_s = runner._vel_fn(v_cur, prev_v, a_cur, prev_a)
            if sp > 1:
              v_pair = torch.cat((dv_s, v_s), dim=0)
              dist.all_reduce(v_pair, op=dist.ReduceOp.SUM, group=sp_pg)
              dv_glob, v_glob = float(v_pair[0].cpu()), float(v_pair[1].cpu())
            else:
              dv_glob, v_glob = float(dv_s.cpu()), float(v_s.cpu())
            rv = dv_glob / max(v_glob, 1e-6)
            ra = float(da_s.cpu()) / max(float(a_s.cpu()), 1e-6)
            last_rate = max(rv, 0.7 * ra) / float(consec + 1)
          else:
            last_rate = 1.0
          accum_err = 0.0
          consec = 0
          prev_v, prev_a = v_cur, a_cur
        prev_expert = ex

      _sync()
      tot_s = time.perf_counter() - t_run0
      # Gather sharded video tokens across sp_pg once at end of denoising -> unpatchify!
      if sp > 1:
        v_g = torch.empty((sp, 1, s_loc, v_tok_ch), device=device, dtype=torch.float32)
        dist.all_gather_into_tensor(v_g, lat_tok.contiguous(), group=sp_pg)
        lat_tok_full_t = v_g.permute(1, 0, 2, 3).contiguous().view(1, n_vid, v_tok_ch)
      else:
        lat_tok_full_t = lat_tok

      # Convert patchified (out_dim, pt, ph, pw) back to [1, 16, T, H/8, W/8]
      lt_np = lat_tok_full_t.cpu().numpy()
      f_g, h_g, w_g = grid
      pt, ph, pw = patch
      lat_v_out = (
          lt_np.reshape(1, f_g, h_g, w_g, 16, pt, ph, pw)
          .transpose(0, 4, 1, 5, 2, 6, 3, 7)
          .reshape(1, 16, t_lat, h_lat, w_lat)
      )
      lat_a_out = lat_a.cpu().numpy()
      say(
          f"[torchtpu_mova] run {run_i}: {n_steps} steps in {tot_s:.2f}s "
          f"(eval={n_steps - n_cached}, cached={n_cached}) | "
          f"lat_v std={float(lat_v_out.std()):.4f} lat_a std={float(lat_a_out.std()):.4f}"
      )

    if rank == 0:
      out_p = Path(args.latents_out)
      out_p.parent.mkdir(parents=True, exist_ok=True)
      np.savez(out_p, lat_v=lat_v_out.astype(np.float32), lat_a=lat_a_out.astype(np.float32))
      say(f"[torchtpu_mova] Saved denoised latents to {out_p}")

  if world > 1:
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
