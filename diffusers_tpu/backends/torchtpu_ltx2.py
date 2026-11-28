"""VideoGym-LTX-2.3 (22B Audio-Video DiT) native ``torch_tpu`` backend (8-rank, 4x2 CFG-parallel).

Runs the 48-block LTX-2.3 DiT natively via ``torchrun --nproc_per_node 8`` with:
  * **2D ``4x2`` (``cfg=4 x sp=2``) physical-device-aligned process groups**:
    Each ``sp=2`` pair lives on the 2 cores of one physical TPU ``v7x`` chip;
    each of the 4 ``cfg`` rows (``[neg, pos, pos+STG(blk 28), pos+iso]``) runs
    on its own 2-core group with **zero inter-chip collectives across all 48
    blocks**!
  * **3 intra-chip collectives per block** (144/step vs 240/step in JAX):
    fused QKV 1-A2A Ulysses + single-chip Pallas ``splash.attention`` ``jax_op``
    + reverse 1-A2A for ``attn1``, plus 1-``all_gather`` exact LSE merge for ``v2a``.
  * **100% resident weights** on every ``v7x`` core (``35.6 GiB`` BF16 or
    ``18.4 GiB`` on-TPU W8A8 E4M3 ``--fp8`` via ``pallas::ltx2_fp8_linear``).
  * **``spmd_safe`` + ``DT_AOT_PERSIST``** grouped block AOT compilation (zero
    closures, subgroup-salted persistent ``.bin`` cache).

Launch via ``scripts/run_ltx2_torchtpu.sh`` (handles Stage 0 Gemma-3/VAE encode
-> Stage 1 8-rank ``torch_tpu`` denoise -> Stage 3 on-TPU ``uint8`` MP4 decode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

# Prevent libtpu double-init when JAX exports single-chip Pallas jax_ops inside torch_tpu.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DT_FORCE_TPU_TRACE", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ..models.ltx2 import jax_dit as _jd  # noqa: E402
from ..models.ltx2.torch_dit import (  # noqa: E402
    LTX2TpuHooks,
    LTX2TransformerTorch,
    layer_norm_no_affine_torch,
    load_torch_ltx2_dit,
)

_REG: dict[Any, Any] = {}


def _sync() -> None:
  if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
    torch.accelerator.synchronize()
  elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
    torch.tpu.synchronize()


def _init_tpu_dist_2d(cfg_parallel: int = 4) -> dict[str, Any]:
  """Initialize 8-rank ``tpu_dist`` and build physical-chip-aligned 2D ``(cfg, sp)`` groups."""
  import torch_tpu  # noqa: F401

  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    dist.init_process_group(backend="tpu_dist", init_method="env://", rank=rank, world_size=world)

  device = torch.device("tpu")

  # On v7x-8, consecutive torchrun rank pairs (0,1), (2,3), (4,5), (6,7) drive
  # the 2 cores of each physical chip (devices (0,1), (4,5), (6,7), (2,3)).
  cp = max(1, min(int(cfg_parallel), world))
  assert world % cp == 0, f"cfg_parallel={cp} must divide world={world}"
  sp = world // cp
  cfg_rank = rank // sp
  sp_rank = rank % sp

  my_sp_pg = None
  my_sp_name = ""
  my_cfg_pg = None
  if world > 1:
    if sp > 1:
      for cr in range(cp):
        members = [cr * sp + sr for sr in range(sp)]
        pg = dist.new_group(members, backend="tpu_dist")
        if cr == cfg_rank:
          my_sp_pg = pg
          my_sp_name = pg.group_name
    if cp > 1:
      for sr in range(sp):
        members = [cr * sp + sr for cr in range(cp)]
        pg = dist.new_group(members, backend="tpu_dist")
        if sr == sp_rank:
          my_cfg_pg = pg

  return {
      "rank": rank,
      "world": world,
      "device": device,
      "cp": cp,
      "sp": sp,
      "cfg_rank": cfg_rank,
      "sp_rank": sp_rank,
      "sp_pg": my_sp_pg,
      "sp_group_name": my_sp_name,
      "cfg_pg": my_cfg_pg,
  }


# ---------------------------------------------------------------------------
# Single-chip Pallas jax_ops (registered BEFORE Dynamo tracing)
# ---------------------------------------------------------------------------


def get_ltx2_splash_op(seq_len: int, heads_local: int, head_dim: int = 128, *, qk_fp8: bool = False):
  """Register ``pallas::ltx2_splash_...`` wrapping ``kernels.splash.attention``."""
  key = ("ltx2_splash", int(seq_len), int(heads_local), int(head_dim), bool(qk_fp8))
  if key in _REG:
    return _REG[key]

  from ..kernels import block_sizes as _block_sizes
  from ..kernels import splash as _splash

  try:
    bs = _block_sizes.resolve_block_sizes(
        int(seq_len), int(seq_len), int(head_dim), {}, "residual-fixedm", _block_sizes.V7_SPEC
    )
    bq = int(bs["block_q"])
    bkv = int(bs["block_kv"])
    bkvc = int(bs["block_kv_compute"])
  except Exception:
    bq = min(1024, int(seq_len))
    bkv = min(2048, int(seq_len))
    bkvc = min(512, bkv)

  _tpu_mesh = (
      jax.sharding.AbstractMesh(
          (1,), ("_tpu",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
      )
      if hasattr(jax.sharding, "AbstractDevice")
      else None
  )

  def fn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    # q, k, v: (1, H_loc, S, D) bf16
    scale = q.shape[-1] ** -0.5
    k_c = k - jnp.mean(k.astype(jnp.float32), axis=2, keepdims=True).astype(k.dtype)
    if _tpu_mesh is not None:
      with jax.sharding.use_abstract_mesh(_tpu_mesh):
        return _splash.attention(
            q,
            k_c,
            v,
            scale=scale,
            block_q=bq,
            block_kv=bkv,
            block_kv_compute=bkvc,
            use_fixed_m=True,
            qk_fp8=qk_fp8,
        )
    return _splash.attention(
        q,
        k_c,
        v,
        scale=scale,
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkvc,
        use_fixed_m=True,
        qk_fp8=qk_fp8,
    )

  tag = "fp8" if qk_fp8 else "bf16"
  op = torch.tpu.pallas.jax_op(f"pallas::ltx2_splash_{tag}_s{seq_len}_h{heads_local}_q{bq}_kv{bkv}", fn)

  @op.register_fake
  def _fake(q, k, v):
    del k, v
    return torch.empty_like(q, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


def get_ltx2_fp8_linear_op():
  """Register ``pallas::ltx2_fp8_linear`` matching ``jax_dit._linear`` W8A8 E4M3 exact math."""
  key = ("ltx2_fp8_linear",)
  if key in _REG:
    return _REG[key]

  def fn(x: jax.Array, qw: jax.Array, w_scale: jax.Array, bias: jax.Array) -> jax.Array:
    # x: (B, S, K) bf16; qw: (K, N) float8_e4m3fn; w_scale: (1, N) f32; bias: (N,) bf16
    f8_max = jnp.float32(448.0)
    x32 = x.astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True), jnp.float32(1e-12)) / f8_max
    qx = jnp.clip(x32 / x_scale, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        qx, qw, (((x.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    y = (acc * x_scale * w_scale).astype(x.dtype)
    return y + bias.astype(x.dtype)

  op = torch.tpu.pallas.jax_op("pallas::ltx2_fp8_linear_v1", fn)

  @op.register_fake
  def _fake(x, qw, w_scale, bias):
    del w_scale, bias
    out_shape = (*x.shape[:-1], qw.shape[1])
    return torch.empty(out_shape, device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format)

  _REG[key] = op
  return op


def build_ltx2_torchtpu_hooks(
    *,
    rank: int,
    sp_size: int,
    sp_group_name: str,
    n_vid: int,
    num_video_heads: int = 32,
    head_dim: int = 128,
    fp8: bool = False,
    qk_fp8: bool = False,
) -> LTX2TpuHooks:
  """Pre-register single-chip Pallas ops and construct :class:`LTX2TpuHooks`."""
  h_loc = num_video_heads // sp_size
  splash_op = get_ltx2_splash_op(n_vid, h_loc, head_dim, qk_fp8=qk_fp8)
  fp8_op = get_ltx2_fp8_linear_op() if fp8 else None

  if rank == 0:
    print(
        f"[torchtpu_ltx2] hooks: sp={sp_size} splash(s={n_vid}, h_loc={h_loc}, qk_fp8={qk_fp8}) fp8_linear={fp8}",
        flush=True,
    )

  def _video_self_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q, k, v: (1, S_loc, H, D)
    b, s_loc, h_tot, d = q.shape
    if sp_size == 1:
      qb = q.permute(0, 2, 1, 3).contiguous()
      kb = k.permute(0, 2, 1, 3).contiguous()
      vb = v.permute(0, 2, 1, 3).contiguous()
      ob = splash_op(qb, kb, vb)
      return ob.permute(0, 2, 1, 3).contiguous()

    # Fused 1-A2A Ulysses over [q, k, v]: (3, S_loc, sp, h_loc, D) -> (sp, 3, S_loc, h_loc, D)
    qkv = torch.stack((q[0], k[0], v[0]), dim=0)
    send = qkv.reshape(3, s_loc, sp_size, h_loc, d).permute(2, 0, 1, 3, 4).contiguous()
    splits = [1] * sp_size
    recv = torch.ops._c10d_functional.all_to_all_single(send, splits, splits, sp_group_name)
    recv = torch.ops._c10d_functional.wait_tensor(recv)  # (sp, 3, S_loc, h_loc, D)
    # -> (3, h_loc, sp * S_loc, D)
    recv_bhsd = recv.permute(1, 3, 0, 2, 4).contiguous().reshape(3, h_loc, sp_size * s_loc, d)
    o_bhsd = splash_op(
        recv_bhsd[0:1].contiguous(),
        recv_bhsd[1:2].contiguous(),
        recv_bhsd[2:3].contiguous(),
    )  # (1, h_loc, S_tot, D)
    # Reverse 1-A2A: (h_loc, sp, S_loc, D) -> (sp, S_loc, h_loc, D)
    send_o = o_bhsd[0].reshape(h_loc, sp_size, s_loc, d).permute(1, 2, 0, 3).contiguous()
    recv_o = torch.ops._c10d_functional.all_to_all_single(send_o, splits, splits, sp_group_name)
    recv_o = torch.ops._c10d_functional.wait_tensor(recv_o)  # (sp, S_loc, h_loc, D)
    return recv_o.permute(1, 0, 2, 3).contiguous().reshape(b, s_loc, h_tot, d)

  def _v2a_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (1, N_a, 32, 64) replicated; k, v: (1, S_loc, 32, 64) sharded over sp_size
    scale = q.shape[-1] ** -0.5
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    if sp_size == 1:
      p = torch.softmax(s, dim=-1).to(v.dtype)
      return torch.einsum("bhqk,bkhd->bqhd", p.float(), v.float()).to(v.dtype)

    m_loc = s.amax(dim=-1, keepdim=True)  # (1, H, N_a, 1)
    p = torch.exp(s - m_loc)
    l_loc = p.sum(dim=-1, keepdim=True)  # (1, H, N_a, 1)
    o_loc = torch.einsum("bhqk,bkhd->bqhd", p.to(v.dtype).float(), v.float())  # (1, N_a, H, D) f32

    # Pack [o_loc (64), l_loc (1), m_loc (1)] -> single 1.13 MB all_gather!
    pack = torch.cat(
        (o_loc, l_loc.permute(0, 2, 1, 3), m_loc.permute(0, 2, 1, 3)), dim=-1
    ).contiguous()  # (1, N_a, H, D + 2)
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(pack, sp_size, sp_group_name)
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)  # (sp, N_a, H, D + 2)
    d = q.shape[-1]
    o_all = gathered[..., :d]  # (sp, N_a, H, D)
    l_all = gathered[..., d : d + 1]  # (sp, N_a, H, 1)
    m_all = gathered[..., d + 1 : d + 2]  # (sp, N_a, H, 1)
    m_max = m_all.amax(dim=0, keepdim=True)
    w = torch.exp(m_all - m_max)
    num = (o_all * w).sum(dim=0, keepdim=True)
    den = (l_all * w).sum(dim=0, keepdim=True)
    return (num / den).to(v.dtype)

  def _fp8_linear(x: torch.Tensor, qw: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert fp8_op is not None
    return fp8_op(x.contiguous(), qw.contiguous(), w_scale.contiguous(), bias.contiguous())

  return LTX2TpuHooks(
      video_self_attn=_video_self_attn,
      v2a_attn=_v2a_attn,
      fp8_linear=_fp8_linear if fp8 else None,
      sp_size=sp_size,
      qk_fp8=qk_fp8,
  )


# ---------------------------------------------------------------------------
# Module-level picklable AOT functions for DT_AOT_PERSIST
# ---------------------------------------------------------------------------

_ACTIVE_LTX2_HOOKS: LTX2TpuHooks | None = None


def _wrap_spmd_compile(fn, *, use_spmd_safe: bool = True):
  if use_spmd_safe and hasattr(torch, "tpu"):
    dist_mod = getattr(torch.tpu, "distributed", None)
    spmd_safe = getattr(dist_mod, "spmd_safe", None) if dist_mod is not None else None
    if spmd_safe is None:
      try:
        from torch_tpu._internal.distributed import spmd_safe  # type: ignore[import]
      except Exception:
        spmd_safe = None
    if spmd_safe is not None:
      fn = spmd_safe(fn)
  return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)


def _top_ltx2_pre(
    proj_in, audio_proj_in,
    time_embed, audio_time_embed, prompt_adaln, audio_prompt_adaln,
    av_v_ss, av_v_g, av_a_ss, av_a_g,
    x_full, a_full, sp_idx, tv, tt, rows_per_group: int,
):
  x_loc = x_full.index_select(1, sp_idx).to(torch.bfloat16)
  a_loc = a_full.to(torch.bfloat16)
  if rows_per_group > 1:
    x_loc = x_loc.repeat(rows_per_group, 1, 1)
    a_loc = a_loc.repeat(rows_per_group, 1, 1)
  b = tv.shape[0]
  dt = x_loc.dtype
  temb, emb = time_embed(tv.reshape(-1), dt)
  temb_a, emb_a = audio_time_embed(tt, dt)
  tp, _ = prompt_adaln(tt, dt)
  tpa, _ = audio_prompt_adaln(tt, dt)
  cav, _ = av_v_ss(tt, dt)
  cavg, _ = av_v_g(tt, dt)
  caa, _ = av_a_ss(tt, dt)
  caag, _ = av_a_g(tt, dt)
  x = proj_in(x_loc)
  a = audio_proj_in(a_loc)
  return (
      x, a,
      temb.reshape(b, -1, temb.shape[-1]),
      emb.reshape(b, -1, emb.shape[-1]),
      temb_a.reshape(b, -1, temb_a.shape[-1]),
      emb_a.reshape(b, -1, emb_a.shape[-1]),
      tp.reshape(b, -1, tp.shape[-1]),
      tpa.reshape(b, -1, tpa.shape[-1]),
      cav.reshape(b, -1, cav.shape[-1]),
      cavg.reshape(b, -1, cavg.shape[-1]),
      caa.reshape(b, -1, caa.shape[-1]),
      caag.reshape(b, -1, caag.shape[-1]),
  )


def _top_ltx2_group(
    mods,
    x, a, ctx, actx,
    temb, temb_a, tp, tpa, cav, cavg, caa, caag,
    token_t,
    rope_v_c, rope_v_s, rope_a_c, rope_a_s,
    rope_cv_c, rope_cv_s, rope_ca_c, rope_ca_s,
    stg_flags, iso_keep,
):
  for bi, m in enumerate(mods):
    x, a = type(m).forward(
        m,
        x, a, ctx, actx,
        temb, temb_a, tp, tpa, cav, cavg, caa, caag,
        token_t,
        rope_v_c, rope_v_s, rope_a_c, rope_a_s,
        rope_cv_c, rope_cv_s, rope_ca_c, rope_ca_s,
        stg_flags[bi], iso_keep,
        hooks=_ACTIVE_LTX2_HOOKS,
    )
  return x, a


def _top_ltx2_post(proj_out, audio_proj_out, ss_table, audio_ss_table, x, a, emb, emb_a, token_t):
  emb_tok = emb.index_select(1, token_t)
  sh_v = ss_table[0].to(emb_tok.dtype) + emb_tok
  sc_v = ss_table[1].to(emb_tok.dtype) + emb_tok
  px = proj_out(layer_norm_no_affine_torch(x) * (1.0 + sc_v) + sh_v)
  sh_a = audio_ss_table[0].to(emb_a.dtype) + emb_a
  sc_a = audio_ss_table[1].to(emb_a.dtype) + emb_a
  pa = audio_proj_out(layer_norm_no_affine_torch(a) * (1.0 + sc_a) + sh_a)
  return px, pa


def _guide_x0_torch(
    sample: torch.Tensor,
    out_4: torch.Tensor,
    s_cur: torch.Tensor,
    cfg: float,
    stg: float,
    mod: float,
    rescale: float,
) -> torch.Tensor:
  x0 = sample - out_4.float() * s_cur
  neg, pos, pst, iso = x0[0:1], x0[1:2], x0[2:3], x0[3:4]
  g = pos + (cfg - 1.0) * (pos - neg) + stg * (pos - pst)
  if mod != 1.0:
    g = g + (mod - 1.0) * (pos - iso)
  if rescale > 0.0:
    n_dof = float(pos.shape[1] * pos.shape[2] - 1)
    mp = pos.mean(dim=(1, 2), keepdim=True)
    mg = g.mean(dim=(1, 2), keepdim=True)
    std_p = torch.sqrt(((pos - mp) ** 2).sum(dim=(1, 2), keepdim=True) / n_dof)
    std_g = torch.sqrt(((g - mg) ** 2).sum(dim=(1, 2), keepdim=True) / n_dof)
    g = rescale * (g * (std_p / std_g)) + (1.0 - rescale) * g
  return (sample - g) / s_cur


def _top_ltx2_euler_step(
    x: torch.Tensor,
    a: torch.Tensor,
    pv_4: torch.Tensor,
    pa_4: torch.Tensor,
    first_t: torch.Tensor,
    s_cur: torch.Tensor,
    ds: torch.Tensor,
    cfg_v: float,
    stg_v: float,
    mod_v: float,
    cfg_a: float,
    stg_a: float,
    mod_a: float,
    rescale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  v = _guide_x0_torch(x, pv_4, s_cur, cfg_v, stg_v, mod_v, rescale)
  va = _guide_x0_torch(a, pa_4, s_cur, cfg_a, stg_a, mod_a, rescale)
  x_new = x + ds * v
  x_out = first_t * x + (1.0 - first_t) * x_new
  a_out = a + ds * va
  return x_out, a_out


class CompiledLTX2RunnerTT:
  """Grouped-block AOT runner for 48-block :class:`LTX2TransformerTorch`."""

  def __init__(
      self,
      model: LTX2TransformerTorch,
      *,
      hooks: LTX2TpuHooks,
      cfg_rank: int,
      sp_rank: int,
      sp_size: int,
      n_vid: int,
      n_aud: int,
      aot_group: int = 8,
      fp8: bool = False,
      use_compile: bool = True,
      use_spmd_safe: bool = True,
  ) -> None:
    self.model = model
    self.hooks = hooks
    self.cfg_rank = int(cfg_rank)
    self.sp_rank = int(sp_rank)
    self.sp_size = int(sp_size)
    self.aot_group = max(1, int(aot_group))
    assert len(model.blocks) % self.aot_group == 0, (
        f"num_layers={len(model.blocks)} must be divisible by aot_group={self.aot_group}"
    )

    self._blk_chunks = [
        [model.blocks[i] for i in range(s, s + self.aot_group)]
        for s in range(0, len(model.blocks), self.aot_group)
    ]
    self._pre_mods = (
        model.proj_in, model.audio_proj_in,
        model.time_embed, model.audio_time_embed,
        model.prompt_adaln, model.audio_prompt_adaln,
        model.av_cross_attn_video_scale_shift, model.av_cross_attn_video_a2v_gate,
        model.av_cross_attn_audio_scale_shift, model.av_cross_attn_audio_v2a_gate,
    )
    self._post_mods = (
        model.proj_out, model.audio_proj_out,
        model.scale_shift_table, model.audio_scale_shift_table,
    )

    persist_dir = os.environ.get("DT_AOT_PERSIST", "").strip()
    if persist_dir == "0":
      persist_dir = ""

    if not use_compile:
      self._pre_fn = _top_ltx2_pre
      self._group_fn = _top_ltx2_group
      self._post_fn = _top_ltx2_post
      self._euler_fn = _top_ltx2_euler_step
      return

    c_pre = _wrap_spmd_compile(_top_ltx2_pre, use_spmd_safe=use_spmd_safe)
    c_grp = _wrap_spmd_compile(_top_ltx2_group, use_spmd_safe=use_spmd_safe)
    c_post = _wrap_spmd_compile(_top_ltx2_post, use_spmd_safe=use_spmd_safe)
    c_euler = _wrap_spmd_compile(_top_ltx2_euler_step, use_spmd_safe=use_spmd_safe)

    if not persist_dir or not hasattr(c_grp, "aot_compile"):
      self._pre_fn = c_pre
      self._group_fn = c_grp
      self._post_fn = c_post
      self._euler_fn = c_euler
      return

    os.makedirs(persist_dir, exist_ok=True)
    base_salt = (
        "ltx2_v2",
        self.aot_group,
        self.cfg_rank,
        self.sp_size,
        int(n_vid),
        int(n_aud),
        str(model.dtype),
        bool(fp8),
        bool(hooks.qk_fp8),
    )

    def _wrap_persist(tag: str, compiled_fn: Any) -> Any:
      h = hashlib.sha1(repr((tag,) + base_salt).encode("utf-8")).hexdigest()[:16]
      bin_path = os.path.join(persist_dir, f"ltx2_{tag}_g{self.aot_group}_{h}.bin")
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
                print(f"  [aot/ltx2] loaded {os.path.basename(bin_path)} in {time.perf_counter() - t_ld:.2f}s", flush=True)
              return art(*args)
            except Exception as exc:
              if self.cfg_rank == 0 and self.sp_rank == 0:
                print(f"  [aot/ltx2] cached {os.path.basename(bin_path)} unusable ({type(exc).__name__}); recompiling", flush=True)
          try:
            art = compiled_fn.aot_compile((args, {}))
            art_holder[0] = art
            if self.sp_rank == 0 and hasattr(art, "save_compiled_function"):
              tmp_p = f"{bin_path}.tmp.{os.getpid()}"
              art.save_compiled_function(tmp_p)
              os.replace(tmp_p, bin_path)
              if self.cfg_rank == 0:
                print(f"  [aot/ltx2] saved {os.path.basename(bin_path)}", flush=True)
            return art(*args)
          except Exception as exc:
            if self.cfg_rank == 0 and self.sp_rank == 0:
              print(f"  [aot/ltx2] {tag} fallback to torch.compile ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return compiled_fn(*args)

      return _call

    self._pre_fn = _wrap_persist("pre", c_pre)
    self._group_fn = _wrap_persist("grp", c_grp)
    self._post_fn = _wrap_persist("post", c_post)
    self._euler_fn = _wrap_persist("euler", c_euler)

  @torch.no_grad()
  def forward_local(
      self,
      x_full: torch.Tensor,   # [1, N_vid, 128] f32/bf16 on TPU
      a_full: torch.Tensor,   # [1, N_aud, 128] f32/bf16 on TPU
      sp_idx: torch.Tensor,   # [S_loc] int64 on TPU
      ctx_row: torch.Tensor,  # [rows_per_group, 1024, 4096] bf16
      actx_row: torch.Tensor,  # [rows_per_group, 1024, 2048] bf16
      tv: torch.Tensor,       # [1, 2] f32 on TPU
      tt: torch.Tensor,       # [1] f32 on TPU
      rows_per_group: int,
      token_t_local: torch.Tensor,  # [S_loc] int64
      ropes_local: dict[str, tuple[torch.Tensor, torch.Tensor]],
      stg_chunks: list[torch.Tensor],
      iso_keep: torch.Tensor,  # [rows_per_group, 1, 1] bf16
  ) -> tuple[torch.Tensor, torch.Tensor]:
    global _ACTIVE_LTX2_HOOKS
    _ACTIVE_LTX2_HOOKS = self.hooks

    x, a, temb, emb, temb_a, emb_a, tp, tpa, cav, cavg, caa, caag = self._pre_fn(
        *self._pre_mods, x_full, a_full, sp_idx, tv, tt, rows_per_group,
    )

    rv_c, rv_s = ropes_local["video"]
    ra_c, ra_s = ropes_local["audio"]
    rcv_c, rcv_s = ropes_local["ca_video"]
    rca_c, rca_s = ropes_local["ca_audio"]

    for chunk, stg_c in zip(self._blk_chunks, stg_chunks):
      x, a = self._group_fn(
          chunk,
          x, a, ctx_row, actx_row,
          temb, temb_a, tp, tpa, cav, cavg, caa, caag,
          token_t_local,
          rv_c, rv_s, ra_c, ra_s, rcv_c, rcv_s, rca_c, rca_s,
          stg_c, iso_keep,
      )
    return self._post_fn(*self._post_mods, x, a, emb, emb_a, token_t_local)

  @torch.no_grad()
  def euler_step(
      self,
      x: torch.Tensor,
      a: torch.Tensor,
      pv_4: torch.Tensor,
      pa_4: torch.Tensor,
      first_t: torch.Tensor,
      s_cur: torch.Tensor,
      ds: torch.Tensor,
      cfg_v: float,
      stg_v: float,
      mod_v: float,
      cfg_a: float,
      stg_a: float,
      mod_a: float,
      rescale: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self._euler_fn(
        x, a, pv_4, pa_4, first_t, s_cur, ds,
        cfg_v, stg_v, mod_v, cfg_a, stg_a, mod_a, rescale,
    )


def _guide_x0_np(
    sample: np.ndarray,
    out_4: np.ndarray,
    sigma: float,
    cfg: float,
    stg: float,
    mod: float,
    rescale: float,
) -> np.ndarray:
  """Exact float32 x0-space guidance matching ``models/ltx2/pipeline.py:L210-220``."""
  x0 = sample - out_4.astype(np.float32) * float(sigma)
  neg, pos, pst = x0[0:1], x0[1:2], x0[2:3]
  g = pos + (cfg - 1.0) * (pos - neg) + stg * (pos - pst)
  if out_4.shape[0] == 4 and mod != 1.0:
    g = g + (mod - 1.0) * (pos - x0[3:4])
  if rescale > 0.0:
    axes = tuple(range(1, pos.ndim))
    std_p = np.std(pos, ddof=1, keepdims=True, axis=axes)
    std_g = np.std(g, ddof=1, keepdims=True, axis=axes)
    g = rescale * (g * (std_p / std_g)) + (1.0 - rescale) * g
  return (sample - g) / float(sigma)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="VideoGym-LTX-2.3 22B DiT on torch_tpu (8-rank, 4x2).")
  ap.add_argument("--mode", default="denoise", choices=("bench", "denoise"))
  ap.add_argument("--repo", default="/workspace/shun/ltx23_weights", help="Local snapshot dir of VideoGym-LTX-2.3.")
  ap.add_argument("--stage0-npz", default="", help="Path to Stage 0 conditioning & noise .npz.")
  ap.add_argument("--latents-out", default="ltx2_tt_latents.npz", help="Output path for denoised latents .npz.")
  ap.add_argument("--fp8", action="store_true", help="Enable on-TPU W8A8 E4M3 FP8 block linears.")
  ap.add_argument("--qk-fp8", action="store_true", help="Enable E4M3 FP8 QK^T inside Pallas SplashAttention.")
  ap.add_argument("--aot-group", type=int, default=8, help="DiT blocks per compiled AOT graph (divides 48; default 8).")
  ap.add_argument("--cfg-parallel", type=int, default=4, choices=(1, 2, 4), help="CFG groups (default 4 -> 4x2 on v7x-8).")
  ap.add_argument("--guidance", type=float, default=3.0)
  ap.add_argument("--stg", type=float, default=1.0)
  ap.add_argument("--modality", type=float, default=1.0)
  ap.add_argument("--audio-guidance", type=float, default=7.0)
  ap.add_argument("--audio-stg", type=float, default=1.0)
  ap.add_argument("--audio-modality", type=float, default=3.0)
  ap.add_argument("--rescale", type=float, default=0.7)
  ap.add_argument("--stg-block", type=int, default=28)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--bench-iters", type=int, default=5)
  ap.add_argument("--runs", type=int, default=1)
  ap.add_argument("--no-compile", action="store_true")
  ap.add_argument("--no-spmd-safe", action="store_true")
  return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv)
  topo = _init_tpu_dist_2d(cfg_parallel=args.cfg_parallel)
  rank, world, device = topo["rank"], topo["world"], topo["device"]
  cp, sp = topo["cp"], topo["sp"]
  cfg_rank, sp_rank = topo["cfg_rank"], topo["sp_rank"]
  sp_pg, sp_group_name, cfg_pg = topo["sp_pg"], topo["sp_group_name"], topo["cfg_pg"]

  def say(msg: str) -> None:
    if rank == 0:
      print(msg, flush=True)

  def _fast_h2d_no_pin(tensor: torch.Tensor, dev: torch.device) -> torch.Tensor:
    t_c = tensor.contiguous()
    flat = t_c.reshape(-1)
    if t_c.element_size() == 2 and (flat.numel() * 2) % 4 == 0:
      dt, sh = flat.dtype, t_c.shape
      return flat.view(torch.int32).to(dev).view(dt).reshape(sh)
    return t_c.to(dev)

  say(
      f"[torchtpu_ltx2] mode={args.mode} world={world} ({cp} cfg x {sp} sp) "
      f"fp8={args.fp8} qk_fp8={args.qk_fp8} aot_group={args.aot_group}"
  )

  # 1. Load Stage 0 conditioning / latents (or construct synthetic 768x512x129 inputs for bench).
  if args.stage0_npz and Path(args.stage0_npz).exists():
    s0 = np.load(args.stage0_npz, allow_pickle=False)
    ctx_4 = np.asarray(s0["ctx"], dtype=np.float32)  # [4, 1024, 4096]
    actx_4 = np.asarray(s0["actx"], dtype=np.float32)  # [4, 1024, 2048]
    x_init = np.asarray(s0["x"], dtype=np.float32)  # [1, 6528, 128]
    a_init = np.asarray(s0["a"], dtype=np.float32)  # [1, 134, 128]
    sigmas = np.asarray(s0["sigmas"], dtype=np.float32)  # [steps + 1]
    token_t_np = np.asarray(s0["token_t"], dtype=np.int64)  # [6528]
    first_np = np.asarray(s0["first"], dtype=np.float32)  # [1, 6528, 1]
    lf, lh, lw = int(s0["lf"]), int(s0["lh"]), int(s0["lw"])
    fps = float(s0["fps"])
    n_aud = int(a_init.shape[1])
  else:
    lf, lh, lw, fps, n_aud = 17, 16, 24, 24.0, 134
    n_vid_tmp = lf * lh * lw
    rng = np.random.default_rng(10)
    ctx_4 = (rng.standard_normal((4, 1024, 4096)) * 0.1).astype(np.float32)
    actx_4 = (rng.standard_normal((4, 1024, 2048)) * 0.1).astype(np.float32)
    x_init = rng.standard_normal((1, n_vid_tmp, 128)).astype(np.float32)
    a_init = rng.standard_normal((1, n_aud, 128)).astype(np.float32)
    sigmas = np.linspace(1.0, 0.0, 31, dtype=np.float32)
    token_t_np = np.ones((n_vid_tmp,), dtype=np.int64)
    token_t_np[: lh * lw] = 0
    first_np = (token_t_np == 0).astype(np.float32)[None, :, None]

  n_vid = lf * lh * lw
  assert n_vid % sp == 0, f"n_vid={n_vid} must be divisible by sp={sp}"
  s_loc = n_vid // sp
  lo, hi = sp_rank * s_loc, (sp_rank + 1) * s_loc

  # 2. Stream-load 48-block DiT directly onto TPU (with optional on-TPU W8A8 FP8 + pre-transposed BF16).
  t_ld = time.perf_counter()
  model = load_torch_ltx2_dit(
      args.repo,
      dtype=torch.bfloat16,
      fp8=args.fp8,
      pretranspose_bf16=True,
      device=device,
      h2d_fn=_fast_h2d_no_pin,
      sync_fn=_sync,
      verbose=(rank == 0),
  )
  _sync()
  load_s = time.perf_counter() - t_ld
  say(f"[torchtpu_ltx2] DiT loaded in {load_s:.2f}s.")

  # 3. Pre-register Pallas Splash & FP8 Linear ops and create AOT runner.
  hooks = build_ltx2_torchtpu_hooks(
      rank=rank,
      sp_size=sp,
      sp_group_name=sp_group_name,
      n_vid=n_vid,
      num_video_heads=int(model.cfg["num_attention_heads"]),
      head_dim=int(model.cfg["attention_head_dim"]),
      fp8=args.fp8,
      qk_fp8=args.qk_fp8,
  )
  runner = CompiledLTX2RunnerTT(
      model,
      hooks=hooks,
      cfg_rank=cfg_rank,
      sp_rank=sp_rank,
      sp_size=sp,
      n_vid=n_vid,
      n_aud=n_aud,
      aot_group=args.aot_group,
      fp8=args.fp8,
      use_compile=not args.no_compile,
      use_spmd_safe=not args.no_spmd_safe,
  )

  # 4. Prepare local static RoPE, token_t, STG, iso_keep, and CFG-row prompt tensors.
  rope_np = _jd.ropes(model.cfg, lf, lh, lw, fps, n_aud)
  ropes_local = {
      "video": (
          torch.from_numpy(np.ascontiguousarray(rope_np["video"][0][lo:hi])).to(device),
          torch.from_numpy(np.ascontiguousarray(rope_np["video"][1][lo:hi])).to(device),
      ),
      "audio": (
          torch.from_numpy(np.ascontiguousarray(rope_np["audio"][0])).to(device),
          torch.from_numpy(np.ascontiguousarray(rope_np["audio"][1])).to(device),
      ),
      "ca_video": (
          torch.from_numpy(np.ascontiguousarray(rope_np["ca_video"][0][lo:hi])).to(device),
          torch.from_numpy(np.ascontiguousarray(rope_np["ca_video"][1][lo:hi])).to(device),
      ),
      "ca_audio": (
          torch.from_numpy(np.ascontiguousarray(rope_np["ca_audio"][0])).to(device),
          torch.from_numpy(np.ascontiguousarray(rope_np["ca_audio"][1])).to(device),
      ),
  }
  token_t_local = torch.from_numpy(np.ascontiguousarray(token_t_np[lo:hi])).to(device)
  sp_idx_t = torch.arange(lo, hi, device=device, dtype=torch.int64)
  first_t = torch.from_numpy(np.ascontiguousarray(first_np)).to(device=device, dtype=torch.float32)

  # Map 4 CFG rows across cp groups (default cp=4: 1 row per group!).
  rows_per_group = 4 // cp
  r_start = cfg_rank * rows_per_group
  r_end = r_start + rows_per_group
  ctx_row = torch.from_numpy(np.ascontiguousarray(ctx_4[r_start:r_end])).to(device=device, dtype=torch.bfloat16)
  actx_row = torch.from_numpy(np.ascontiguousarray(actx_4[r_start:r_end])).to(device=device, dtype=torch.bfloat16)

  stg_all_np = np.zeros((len(model.blocks), rows_per_group), dtype=np.bool_)
  for local_r in range(rows_per_group):
    if r_start + local_r == 2:
      stg_all_np[args.stg_block, local_r] = True
  stg_flags_all = torch.from_numpy(stg_all_np).to(device)
  stg_chunks = [
      stg_flags_all[s : s + args.aot_group].contiguous()
      for s in range(0, len(model.blocks), args.aot_group)
  ]

  iso_keep_np = np.array(
      [0.0 if (r_start + lr == 3) else 1.0 for lr in range(rows_per_group)], dtype=np.float32
  ).reshape(rows_per_group, 1, 1)
  iso_keep = torch.from_numpy(iso_keep_np).to(device=device, dtype=torch.bfloat16)

  # Pre-place all per-step schedule scalars on TPU once (zero H2D inside the denoising loop).
  n_steps = len(sigmas) - 1
  tv_list = [
      torch.tensor([[0.0, float(sigmas[i]) * 1000.0]], device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]
  tt_list = [
      torch.tensor([float(sigmas[i]) * 1000.0], device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]
  s_list = [
      torch.tensor([[[float(sigmas[i])]]], device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]
  ds_list = [
      torch.tensor([[[float(sigmas[i + 1] - sigmas[i])]]], device=device, dtype=torch.float32)
      for i in range(n_steps)
  ]

  v_out_ch = int(model.cfg["out_channels"])
  a_out_ch = int(model.cfg["audio_out_channels"])
  pv_sp_buf = (
      torch.empty((sp, rows_per_group, s_loc, v_out_ch), device=device, dtype=torch.bfloat16)
      if sp > 1 else None
  )
  pv_4_buf = (
      torch.empty((4, n_vid, v_out_ch), device=device, dtype=torch.bfloat16)
      if cp > 1 else None
  )
  pa_4_buf = (
      torch.empty((4, n_aud, a_out_ch), device=device, dtype=torch.bfloat16)
      if cp > 1 else None
  )

  def _fwd_4rows_tpu(
      x_t: torch.Tensor, a_t: torch.Tensor, tv_t: torch.Tensor, tt_t: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    pv_loc, pa_loc = runner.forward_local(
        x_t, a_t, sp_idx_t, ctx_row, actx_row, tv_t, tt_t, rows_per_group,
        token_t_local, ropes_local, stg_chunks, iso_keep,
    )
    # Gather video tokens across sp_group -> [rows_per_group, n_vid, 128]
    if sp > 1:
      dist.all_gather_into_tensor(pv_sp_buf, pv_loc, group=sp_pg)
      if rows_per_group == 1:
        pv_grp = pv_sp_buf.view(1, n_vid, v_out_ch)
      else:
        pv_grp = pv_sp_buf.permute(1, 0, 2, 3).contiguous().view(rows_per_group, n_vid, v_out_ch)
    else:
      pv_grp = pv_loc

    # Gather 4 CFG rows across cfg_group -> [4, n_vid, 128] and [4, n_aud, 128]
    if cp > 1:
      dist.all_gather_into_tensor(pv_4_buf, pv_grp, group=cfg_pg)
      dist.all_gather_into_tensor(pa_4_buf, pa_loc, group=cfg_pg)
      return pv_4_buf, pa_4_buf
    return pv_grp, pa_loc

  x_init_t = torch.from_numpy(np.ascontiguousarray(x_init)).to(device=device, dtype=torch.float32)
  a_init_t = torch.from_numpy(np.ascontiguousarray(a_init)).to(device=device, dtype=torch.float32)

  if args.mode == "bench":
    say(f"[torchtpu_ltx2] Running {args.warmup} warmup forward(s) (includes AOT compile) ...")
    t_w = time.perf_counter()
    pv_t, pa_t = _fwd_4rows_tpu(x_init_t, a_init_t, tv_list[0], tt_list[0])
    _sync()
    t_comp = time.perf_counter() - t_w
    pv_np, pa_np = pv_t.float().cpu().numpy(), pa_t.float().cpu().numpy()
    say(f"[torchtpu_ltx2] Compile/warmup done in {t_comp:.2f}s | pv std={float(pv_np.std()):.4f} pa std={float(pa_np.std()):.4f}")
    for _ in range(max(0, args.warmup - 1)):
      _fwd_4rows_tpu(x_init_t, a_init_t, tv_list[0], tt_list[0])
      _sync()
    ts = []
    for it in range(args.bench_iters):
      t0 = time.perf_counter()
      _fwd_4rows_tpu(x_init_t, a_init_t, tv_list[0], tt_list[0])
      _sync()
      dt = (time.perf_counter() - t0) * 1000.0
      ts.append(dt)
      say(f"  [iter {it}] {dt:.1f} ms/fwd")
    say(
        f"[torchtpu_ltx2] BENCH SUMMARY ({cp}x{sp}, fp8={args.fp8}, qk_fp8={args.qk_fp8}, g={args.aot_group}): "
        f"load={load_s:.2f}s compile={t_comp:.2f}s median={float(np.median(ts)):.1f} ms min={min(ts):.1f} ms"
    )
  else:
    for run_i in range(args.runs):
      x_t = x_init_t.clone()
      a_t = a_init_t.clone()
      _sync()
      t_run0 = time.perf_counter()

      # Step 0 (synced to separate AOT compile / load time cleanly)
      pv_4, pa_4 = _fwd_4rows_tpu(x_t, a_t, tv_list[0], tt_list[0])
      x_t, a_t = runner.euler_step(
          x_t, a_t, pv_4, pa_4, first_t, s_list[0], ds_list[0],
          args.guidance, args.stg, args.modality,
          args.audio_guidance, args.audio_stg, args.audio_modality,
          args.rescale,
      )
      _sync()
      t_after0 = time.perf_counter()
      step0_ms = (t_after0 - t_run0) * 1000.0

      # Steps 1..N-1 pipelined 100% on-TPU (zero host sync / D2H / H2D)
      for i in range(1, n_steps):
        pv_4, pa_4 = _fwd_4rows_tpu(x_t, a_t, tv_list[i], tt_list[i])
        x_t, a_t = runner.euler_step(
            x_t, a_t, pv_4, pa_4, first_t, s_list[i], ds_list[i],
            args.guidance, args.stg, args.modality,
            args.audio_guidance, args.audio_stg, args.audio_modality,
            args.rescale,
        )
      _sync()
      t_end = time.perf_counter()
      tot_s = t_end - t_run0
      steady_ms = (t_end - t_after0) * 1000.0 / max(1, n_steps - 1)

      x = x_t.float().cpu().numpy()
      a = a_t.float().cpu().numpy()
      say(
          f"[torchtpu_ltx2] run {run_i}: {n_steps} steps in {tot_s:.2f}s "
          f"(step0={step0_ms:.1f}ms, steady={steady_ms:.1f}ms/step) | "
          f"x std={float(x.std()):.4f} a std={float(a.std()):.4f}"
      )

    if rank == 0:
      out_p = Path(args.latents_out)
      out_p.parent.mkdir(parents=True, exist_ok=True)
      np.savez(out_p, x=x.astype(np.float32), a=a.astype(np.float32))
      say(f"[torchtpu_ltx2] Saved denoised latents to {out_p}")

  if world > 1:
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
