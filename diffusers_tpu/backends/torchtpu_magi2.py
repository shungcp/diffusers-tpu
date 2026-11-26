"""Priority 3 (`torch_tpu`) 8-rank multi-process runner for ``MAGI-2-preview`` 113.93B DiT.

Launch via ``scripts/run_magi2_torchtpu.sh`` (``torch.distributed.run --nproc_per_node 8``):
- 1 process per TPU chip (``torch.device("tpu")``, ``tpu_dist`` backend).
- Each rank loads ONLY its local ``E_local = 32`` routed experts per head
  (25.3 GiB/rank BF16 or 13.0 GiB/rank FP8-W8A8) using 8-thread parallel
  safetensors streaming + fast pinned-int32 ``_stage_h2d`` upload.
- Pre-registers single-chip Pallas/JAX ops via :mod:`magi2_kernels_tt`
  (``pallas::magi2_grouped_linear_m3``, ``pallas::magi2_splash_sink_s*``,
  ``pallas::magi2_moe_head_{bf16,fp8}_*``, and optional Tokamax mHC) and
  cross-chip Ulysses CP=8 / EP=8 collectives via ``torch.ops._c10d_functional``.
- Compiles just 2 reusable AOT layer programs via
  ``torch.compile(spmd_safe(...), backend="tpu", fullgraph=True, dynamic=False)``:
  one serves all 4 MM Dense layers (0, 1, 38, 39) and one serves all 36 MoE
  layers (2..37) (or groups of ``--aot-group G`` consecutive MoE layers).

Modes:
- ``--mode bench``: 1-step numerical parity check against a Native JAX reference
  ``.npz`` (if ``--ref-npz`` provided) + warmup & steady-state ``s/step``
  benchmark on real 113.93B weights (``272p`` or ``540p``, BF16 or FP8).
- ``--mode denoise``: Full FlowUniPC CFG=2 Stage 1 denoising using precomputed
  ``--text-embeds-cache``, writing ``<out>/<tag>_stage1_latents.npz``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .layerwise_offload import _stage_h2d
from .magi2_kernels_tt import build_torchtpu_hooks
from ..models.magi2.config import (
    RESOLUTION_PRESETS,
    Magi2PreviewConfig,
)
from ..models.magi2.jax_pipeline import (
    build_preview_packing_plan,
    generate_initial_noise,
)
from ..models.magi2.rope_and_ops import FlowUniPCMultistepScheduler
from ..models.magi2.torch_preview_dit import (
    Magi2PreviewTransformerTorch,
    load_torch_preview_from_safetensors,
    sinusoidal_embedding_1d_torch,
)


def _init_tpu_dist() -> tuple[int, int, torch.device, str, Any]:
  """Initialize ``torch_tpu`` device and ``tpu_dist`` process group."""
  import torch_tpu  # noqa: F401 — registers torch.device("tpu")

  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    dist.init_process_group(
        backend="tpu_dist",
        init_method="env://",
        world_size=world,
        rank=rank,
    )
    pg = dist.new_group(list(range(world)), backend="tpu_dist")
    group_name = pg.group_name
  else:
    pg = None
    group_name = ""
  device = torch.device("tpu")
  return rank, world, device, group_name, pg


def _sync() -> None:
  if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
    torch.accelerator.synchronize()
  elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
    torch.tpu.synchronize()


def _move_model_fast_h2d(model: torch.nn.Module, device: torch.device) -> None:
  """Move all parameters & buffers to TPU via fast pinned-int32 ``_stage_h2d``."""
  for mod in model.modules():
    for name, p in list(mod._parameters.items()):
      if p is None:
        continue
      t_dev = _stage_h2d(p.data, device)
      mod._parameters[name] = torch.nn.Parameter(t_dev, requires_grad=False)
      del p
    for name, b in list(mod._buffers.items()):
      if b is None:
        continue
      mod._buffers[name] = _stage_h2d(b, device)
      del b
  gc.collect()
  _sync()


def _wrap_spmd_compile(fn, *, use_spmd_safe: bool = True):
  """Compile ``fn`` with ``torch.compile(backend="tpu", fullgraph=True)``."""
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


class CompiledMagi2PreviewTT:
  """AOT-compiled 40-layer ``Magi2PreviewTransformerTorch`` for ``torch_tpu``."""

  def __init__(
      self,
      model: Magi2PreviewTransformerTorch,
      *,
      sample_lens: tuple[int, ...],
      hooks: Any,
      aot_group: int = 1,
      use_compile: bool = True,
      use_spmd_safe: bool = True,
  ) -> None:
    self.model = model
    self.sample_lens = tuple(int(s) for s in sample_lens)
    self.hooks = hooks
    self.aot_group = max(1, int(aot_group))
    self.use_compile = use_compile

    def _run_one_layer(
        mod, x_sorted, rope_96, mod_ids_sorted, group_sizes, perm_local, inv_perm_local
    ):
      return type(mod).forward(
          mod,
          x_sorted,
          rope_96,
          mod_ids_sorted,
          group_sizes,
          perm_local,
          inv_perm_local,
          self.sample_lens,
          hooks=self.hooks,
      )

    def _run_group(
        mods, x_sorted, rope_96, mod_ids_sorted, group_sizes, perm_local, inv_perm_local
    ):
      for m in mods:
        x_sorted = type(m).forward(
            m,
            x_sorted,
            rope_96,
            mod_ids_sorted,
            group_sizes,
            perm_local,
            inv_perm_local,
            self.sample_lens,
            hooks=self.hooks,
        )
      return x_sorted

    def _run_post(post_mod, x_sorted, inv_perm_local, mod_ids):
      x_seq = x_sorted.index_select(0, inv_perm_local)
      return type(post_mod).forward(post_mod, x_seq, mod_ids, hooks=self.hooks)

    if use_compile:
      self._mm_fn = _wrap_spmd_compile(_run_one_layer, use_spmd_safe=use_spmd_safe)
      if self.aot_group == 1:
        self._moe_fn = _wrap_spmd_compile(_run_one_layer, use_spmd_safe=use_spmd_safe)
      else:
        self._moe_fn = _wrap_spmd_compile(_run_group, use_spmd_safe=use_spmd_safe)
      self._post_fn = _wrap_spmd_compile(_run_post, use_spmd_safe=use_spmd_safe)
    else:
      self._mm_fn = _run_one_layer
      self._moe_fn = _run_one_layer if self.aot_group == 1 else _run_group
      self._post_fn = _run_post

  @torch.no_grad()
  def forward_embedded_local(
      self,
      x_emb_local: torch.Tensor,
      rope_96_local: torch.Tensor,
      mod_ids_local: torch.Tensor,
      mod_ids_sorted: torch.Tensor,
      group_sizes: torch.Tensor,
      perm_local: torch.Tensor,
      inv_perm_local: torch.Tensor,
      time_local: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Run all 40 layers + PostAdapter on this rank's local sequence slice."""
    x_e = x_emb_local.float()
    if time_local is not None:
      t_dim = time_local.shape[-1]
      x_e = torch.cat((time_local.float(), x_e[:, t_dim:]), dim=-1)
    x_sorted = x_e.to(self.model.dtype).index_select(0, perm_local).clone()

    # Layers 0, 1 (MM Dense)
    for li in (0, 1):
      x_sorted = self._mm_fn(
          self.model.layers[li],
          x_sorted,
          rope_96_local,
          mod_ids_sorted,
          group_sizes,
          perm_local,
          inv_perm_local,
      )

    # Layers 2..37 (36 MoE layers, single or grouped AOT program)
    moe_layers = [self.model.layers[li] for li in range(2, 38)]
    if self.aot_group == 1:
      for ml in moe_layers:
        x_sorted = self._moe_fn(
            ml,
            x_sorted,
            rope_96_local,
            mod_ids_sorted,
            group_sizes,
            perm_local,
            inv_perm_local,
        )
    else:
      g = self.aot_group
      for start in range(0, len(moe_layers), g):
        chunk = tuple(moe_layers[start : start + g])
        if len(chunk) == g:
          x_sorted = self._moe_fn(
              chunk,
              x_sorted,
              rope_96_local,
              mod_ids_sorted,
              group_sizes,
              perm_local,
              inv_perm_local,
          )
        else:
          for ml in chunk:
            x_sorted = self._mm_fn(
                ml,
                x_sorted,
                rope_96_local,
                mod_ids_sorted,
                group_sizes,
                perm_local,
                inv_perm_local,
            )

    # Layers 38, 39 (MM Dense)
    for li in (38, 39):
      x_sorted = self._mm_fn(
          self.model.layers[li],
          x_sorted,
          rope_96_local,
          mod_ids_sorted,
          group_sizes,
          perm_local,
          inv_perm_local,
      )

    return self._post_fn(self.model.post_adapter, x_sorted, inv_perm_local, mod_ids_local)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="MAGI-2-preview 113.93B DiT on torch_tpu (8-rank).")
  ap.add_argument("--mode", default="bench", choices=("bench", "denoise"))
  ap.add_argument(
      "--model-dir",
      default=os.environ.get("MAGI2_CKPT_ROOT", "/workspace/shun/magi2_preview_weights"),
      help="Root dir containing preview/ checkpoint (or preview dir directly).",
  )
  ap.add_argument("--resolution", default="272p", choices=tuple(RESOLUTION_PRESETS.keys()))
  ap.add_argument("--seconds", type=float, default=10.0)
  ap.add_argument("--steps", type=int, default=100)
  ap.add_argument("--shift", type=float, default=7.0)
  ap.add_argument("--video-cfg", type=float, default=5.0)
  ap.add_argument("--audio-cfg", type=float, default=7.0)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--fp8", action="store_true", help="Enable W8A8 FP8 MoE weights.")
  ap.add_argument("--cap-factor", type=float, default=None, help="Sliced-AG MoE capacity factor (default 2.0).")
  ap.add_argument("--dense-attention", action="store_true", help="Disable Splash Pallas kernel.")
  ap.add_argument("--aot-group", type=int, default=1, help="Number of MoE layers per compiled graph (1, 2, 4, 6).")
  ap.add_argument("--no-compile", action="store_true", help="Run eager torch_tpu without torch.compile.")
  ap.add_argument("--no-spmd-safe", action="store_true", help="Disable torch.tpu.distributed.spmd_safe.")
  ap.add_argument("--warmup", type=int, default=1, help="Warmup passes in --mode bench.")
  ap.add_argument("--bench-iters", type=int, default=3, help="Timed steady passes in --mode bench.")
  ap.add_argument("--text-embeds-cache", default="", help="Path to precomputed prompt embeds .npz.")
  ap.add_argument("--ref-npz", default="", help="Optional Native JAX reference .npz for 1-step parity check.")
  ap.add_argument("--out", default="/workspace/shun/magi2_outputs")
  ap.add_argument("--tag", default="magi2_torchtpu")
  return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv)
  rank, world, device, group_name, pg = _init_tpu_dist()

  def say(msg: str) -> None:
    if rank == 0:
      print(msg, flush=True)

  model_root = Path(args.model_dir).expanduser()
  preview_dir = model_root / "preview" if (model_root / "preview").exists() else model_root
  cfg = Magi2PreviewConfig.from_checkpoint(preview_dir)

  preset = RESOLUTION_PRESETS[args.resolution]
  p_h, p_w = int(preset["height"]), int(preset["width"])
  vid_t = max(1, int(round(args.seconds * 12.5)) // 4 + 1)  # 32 for 10s
  if args.seconds == 10.0:
    vid_t = 32
  lat_h, lat_w = p_h // 16, p_w // 16
  aud_t = int(round(args.seconds * 25.0))  # 250 for 10s

  say(
      f"[torchtpu_magi2] mode={args.mode} world={world} res={args.resolution} "
      f"vid_thw=({vid_t},{lat_h},{lat_w}) aud_t={aud_t} fp8={args.fp8} "
      f"aot_group={args.aot_group} compile={not args.no_compile}"
  )

  # 1. Stream EP-sharded weights directly to TPU HBM via byte-range seek +
  #    non-pinning int32-bitcast H2D + per-shard sync (8 processes already read
  #    in parallel, keeping peak host RAM < 1.5 GiB/rank!).
  def _fast_h2d_no_pin(tensor: torch.Tensor, dev: torch.device) -> torch.Tensor:
    t_c = tensor.contiguous()
    flat = t_c.reshape(-1)
    if t_c.element_size() == 2 and (flat.numel() * 2) % 4 == 0:
      dt, sh = flat.dtype, t_c.shape
      return flat.view(torch.int32).to(dev).view(dt).reshape(sh)
    return t_c.to(dev)

  t0 = time.perf_counter()
  model = load_torch_preview_from_safetensors(
      preview_dir,
      config=cfg,
      ep_rank=rank,
      ep_size=world,
      dtype=torch.bfloat16,
      fp8_moe=args.fp8,
      max_workers=1,
      device=device,
      h2d_fn=_fast_h2d_no_pin,
      sync_fn=_sync,
      verbose=(rank == 0),
  )
  _sync()
  t_load = time.perf_counter() - t0
  t_cpu, t_h2d = t_load, 0.0
  say(f"[torchtpu_magi2] Direct-to-TPU streaming load done in {t_load:.2f}s.")

  # 2. Load text embeddings or construct deterministic dummy text embeddings for bench/parity.
  if args.text_embeds_cache and Path(args.text_embeds_cache).exists():
    tc = np.load(args.text_embeds_cache, allow_pickle=False)
    cond_txt_np = np.asarray(tc["cond_txt"], dtype=np.float32)
    uncond_txt_np = np.asarray(tc["uncond_txt"], dtype=np.float32)
    if cond_txt_np.ndim == 3:
      cond_txt_np = cond_txt_np[0]
    if uncond_txt_np.ndim == 3:
      uncond_txt_np = uncond_txt_np[0]
  else:
    rng = np.random.default_rng(1234)
    cond_txt_np = (rng.standard_normal((128, cfg.text_in_channels)) * 0.1).astype(np.float32)
    uncond_txt_np = (rng.standard_normal((128, cfg.text_in_channels)) * 0.1).astype(np.float32)

  # 3. Build exact sequence packing plan (matches jax_pipeline bit-for-bit!).
  plan = build_preview_packing_plan(
      video_thw=(vid_t, lat_h, lat_w),
      audio_len=aud_t,
      cond_txt_len=int(cond_txt_np.shape[0]),
      uncond_txt_len=int(uncond_txt_np.shape[0]),
      ref_img_hws=[],
      cp_size=world,
      head_dim=cfg.head_dim,
      rope_temperature=cfg.rope_temperature,
  )
  s_padded = int(plan["s_padded"])
  s_loc = s_padded // world
  s_cond, s_uncond = int(plan["sample_lens"][0]), int(plan["sample_lens"][1])
  n_vid = vid_t * lat_h * lat_w
  say(
      f"[torchtpu_magi2] Packed seq: S_cond={s_cond} S_uncond={s_uncond} "
      f"pad={plan['pad_size']} S_padded={s_padded} (S_local={s_loc}/rank)"
  )

  # 4. Pre-register all single-chip Pallas jax_ops and build Magi2TpuHooks!
  hooks = build_torchtpu_hooks(
      rank=rank,
      world=world,
      group_name=group_name,
      s_total=s_padded,
      sample_lens=(s_cond, s_uncond),
      device=device,
      num_heads_q=cfg.num_heads_q,
      e_local=cfg.moe_num_experts // world,
      top_k=cfg.moe_top_k,
      route_scale=cfg.moe_route_scale,
      cap_factor=args.cap_factor,
      fp8_moe=args.fp8,
      use_splash=not args.dense_attention,
  )

  runner = CompiledMagi2PreviewTT(
      model,
      sample_lens=(s_cond, s_uncond),
      hooks=hooks,
      aot_group=args.aot_group,
      use_compile=not args.no_compile,
      use_spmd_safe=not args.no_spmd_safe,
  )

  # Prepare local static RoPE / modality / sort permutation tensors on TPU.
  lo, hi = rank * s_loc, (rank + 1) * s_loc
  rope_local = torch.from_numpy(np.ascontiguousarray(plan["rope_all"][lo:hi])).to(device=device, dtype=torch.float32)
  mod_raw_np = plan["mod_all"][lo:hi]
  mod_ids_np = np.where(mod_raw_np == 3, 2, mod_raw_np).astype(np.int64)
  perm_np = np.argsort(mod_ids_np, kind="stable").astype(np.int64)
  inv_perm_np = np.empty_like(perm_np)
  inv_perm_np[perm_np] = np.arange(s_loc, dtype=np.int64)
  mod_sorted_np = mod_ids_np[perm_np]
  gs_np = np.array([
      int((mod_sorted_np == 0).sum()),
      int((mod_sorted_np == 1).sum()),
      int((mod_sorted_np == 2).sum()),
  ], dtype=np.int32)

  mod_ids_local = torch.from_numpy(mod_ids_np).to(device)
  mod_ids_sorted = torch.from_numpy(mod_sorted_np).to(device)
  perm_local = torch.from_numpy(perm_np).to(device)
  inv_perm_local = torch.from_numpy(inv_perm_np).to(device)
  group_sizes = torch.from_numpy(gs_np).to(device)
  time_kind_local = torch.from_numpy(np.ascontiguousarray(plan["time_kind_all"][lo:hi])).to(device)

  def _fp32_linear(lin: torch.nn.Linear, x_t: torch.Tensor) -> torch.Tensor:
    if hooks.fp32_matmul_highest is not None:
      return hooks.fp32_matmul_highest(x_t, lin.weight.t()) + lin.bias
    return lin(x_t)

  # Pre-embed static text & pad tokens once via PreAdapter on TPU (fp32 HIGHEST!).
  with torch.no_grad():
    txt_cond_emb = _fp32_linear(model.pre_adapter.text_embedder, torch.from_numpy(cond_txt_np).to(device)).to(torch.bfloat16)
    txt_uncond_emb = _fp32_linear(model.pre_adapter.text_embedder, torch.from_numpy(uncond_txt_np).to(device)).to(torch.bfloat16)
    pad_sz = int(plan["pad_size"])
    if pad_sz > 0:
      pad_emb = model.pre_adapter.text_embedder.bias.to(torch.bfloat16).unsqueeze(0).expand(pad_sz, -1).contiguous()
    else:
      pad_emb = torch.empty((0, cfg.stream_hidden_size), device=device, dtype=torch.bfloat16)

  def _run_step(latent_v_cpu: np.ndarray, latent_a_cpu: np.ndarray, raw_t: float) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
      v_flat = np.transpose(latent_v_cpu[0], (1, 2, 3, 0)).reshape(n_vid, cfg.video_in_channels)
      v_emb = _fp32_linear(model.pre_adapter.video_embedder, torch.from_numpy(np.ascontiguousarray(v_flat)).to(device)).to(torch.bfloat16)
      a_emb = _fp32_linear(model.pre_adapter.audio_embedder, torch.from_numpy(np.ascontiguousarray(latent_a_cpu[0])).to(device)).to(torch.bfloat16)

      parts = [v_emb, a_emb, txt_cond_emb, v_emb, a_emb, txt_uncond_emb]
      if pad_sz > 0:
        parts.append(pad_emb)
      x_emb_full = torch.cat(parts, dim=0)
      x_emb_local = x_emb_full[lo:hi].clone()

      emb_noisy = sinusoidal_embedding_1d_torch(torch.tensor(raw_t, device=device, dtype=torch.float32), dim=cfg.time_channel_dim)
      emb_clean = sinusoidal_embedding_1d_torch(torch.tensor(0.0, device=device, dtype=torch.float32), dim=cfg.time_channel_dim)
      emb_zero = torch.zeros((cfg.time_channel_dim,), device=device, dtype=torch.float32)
      time_local = torch.where(
          (time_kind_local == 2).unsqueeze(-1),
          emb_noisy.unsqueeze(0),
          torch.where((time_kind_local == 1).unsqueeze(-1), emb_clean.unsqueeze(0), emb_zero.unsqueeze(0)),
      )

      out_local = runner.forward_embedded_local(
          x_emb_local,
          rope_local,
          mod_ids_local,
          mod_ids_sorted,
          group_sizes,
          perm_local,
          inv_perm_local,
          time_local,
      )

      # Gather full sequence output on all ranks to extract CFG velocity cleanly.
      if world > 1:
        out_all = torch.empty(
            (world * out_local.shape[0], out_local.shape[1]),
            device=out_local.device,
            dtype=out_local.dtype,
        )
        dist.all_gather_into_tensor(out_all, out_local.contiguous(), group=pg)
      else:
        out_all = out_local

      cond_v = out_all[0:n_vid, : cfg.video_in_channels].reshape(vid_t, lat_h, lat_w, cfg.video_in_channels).permute(3, 0, 1, 2).unsqueeze(0)
      cond_a = out_all[n_vid : n_vid + aud_t, : cfg.audio_in_channels].unsqueeze(0)
      uncond_v = out_all[s_cond : s_cond + n_vid, : cfg.video_in_channels].reshape(vid_t, lat_h, lat_w, cfg.video_in_channels).permute(3, 0, 1, 2).unsqueeze(0)
      uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + aud_t, : cfg.audio_in_channels].unsqueeze(0)

      v_cfg = (uncond_v + args.video_cfg * (cond_v - uncond_v)).float().cpu().numpy()
      a_cfg = (uncond_a + args.audio_cfg * (cond_a - uncond_a)).float().cpu().numpy()
      return v_cfg, a_cfg

  # Initial deterministic latents (matching jax_pipeline --torch-noise!).
  init_v, init_a, _ = generate_initial_noise(
      video_shape=(1, cfg.video_in_channels, vid_t, lat_h, lat_w),
      audio_shape=(1, aud_t, cfg.audio_in_channels),
      refiner_shape=None,
      seed=args.seed,
      torch_noise=True,
  )

  if args.mode == "bench":
    say(f"[torchtpu_magi2] Running {args.warmup} warmup step(s) (includes AOT compile) ...")
    t_w = time.perf_counter()
    v_out, a_out = _run_step(init_v, init_a, 0.999)
    _sync()
    t_compile = time.perf_counter() - t_w
    say(
        f"[torchtpu_magi2] Warmup/compile done in {t_compile:.2f}s | "
        f"v_out std={float(v_out.std()):.4f} a_out std={float(a_out.std()):.4f}"
    )
    for _ in range(max(0, args.warmup - 1)):
      v_out, a_out = _run_step(init_v, init_a, 0.999)
      _sync()

    ts: list[float] = []
    for it in range(args.bench_iters):
      ts0 = time.perf_counter()
      v_out, a_out = _run_step(init_v, init_a, 0.999)
      _sync()
      dt_s = time.perf_counter() - ts0
      ts.append(dt_s)
      say(f"  [iter {it}] {dt_s:.3f} s/step")
    med = float(np.median(ts))
    say(
        f"[torchtpu_magi2] SUMMARY res={args.resolution} fp8={args.fp8}: "
        f"load={t_cpu+t_h2d:.2f}s compile={t_compile:.2f}s steady={med:.3f} s/step"
    )

    if args.ref_npz and Path(args.ref_npz).exists() and rank == 0:
      ref = np.load(args.ref_npz, allow_pickle=False)
      v_ref = np.asarray(ref["v_cfg"], dtype=np.float32)
      a_ref = np.asarray(ref["a_cfg"], dtype=np.float32)
      v_rel = float(np.linalg.norm(v_out - v_ref) / (np.linalg.norm(v_ref) + 1e-12))
      a_rel = float(np.linalg.norm(a_out - a_ref) / (np.linalg.norm(a_ref) + 1e-12))
      say(
          f"[torchtpu_magi2] PARITY vs Native JAX: "
          f"video rel_l2={v_rel:.4e} (std {v_out.std():.4f} vs {v_ref.std():.4f}), "
          f"audio rel_l2={a_rel:.4e} (std {a_out.std():.4f} vs {a_ref.std():.4f})"
      )

    if rank == 0:
      out_dir = Path(args.out)
      out_dir.mkdir(parents=True, exist_ok=True)
      summary = {
          "backend": "torch_tpu",
          "resolution": args.resolution,
          "fp8": args.fp8,
          "aot_group": args.aot_group,
          "load_cpu_s": t_cpu,
          "load_h2d_s": t_h2d,
          "compile_s": t_compile,
          "steady_step_s": med,
          "v_out_std": float(v_out.std()),
          "a_out_std": float(a_out.std()),
      }
      (out_dir / f"{args.tag}_bench.json").write_text(json.dumps(summary, indent=2))
  else:
    # Full FlowUniPC denoising loop
    v_sched = FlowUniPCMultistepScheduler()
    a_sched = FlowUniPCMultistepScheduler()
    v_sched.set_timesteps(args.steps, shift=args.shift)
    a_sched.set_timesteps(args.steps, shift=args.shift)

    lat_v = init_v.copy()
    lat_a = init_a.copy()

    # Warmup compile step
    t_w = time.perf_counter()
    _run_step(lat_v, lat_a, 0.999)
    _sync()
    t_compile = time.perf_counter() - t_w
    say(f"[torchtpu_magi2] Denoise compile done in {t_compile:.2f}s. Running {args.steps} steps ...")

    t_den = time.perf_counter()
    step_times: list[float] = []
    for si, t_int in enumerate(v_sched.timesteps):
      ts0 = time.perf_counter()
      raw_t = float(np.int64(t_int)) / 1000.0
      v_cfg, a_cfg = _run_step(lat_v, lat_a, raw_t)
      lat_v = np.asarray(v_sched.step(v_cfg, t_int, lat_v, return_dict=False)[0], dtype=np.float32)
      lat_a = np.asarray(a_sched.step(a_cfg, t_int, lat_a, return_dict=False)[0], dtype=np.float32)
      dt_s = time.perf_counter() - ts0
      step_times.append(dt_s)
      if si < 3 or (si + 1) % 10 == 0 or si == args.steps - 1:
        say(f"  step {si + 1:3d}/{args.steps} t={raw_t:.3f} ({dt_s:.3f}s)")

    total_den = time.perf_counter() - t_den
    say(
        f"[torchtpu_magi2] Denoise finished: {args.steps} steps in {total_den:.2f}s "
        f"(median={float(np.median(step_times)):.3f} s/step) | "
        f"video std={float(lat_v.std()):.4f} audio std={float(lat_a.std()):.4f}"
    )
    if rank == 0:
      out_dir = Path(args.out)
      out_dir.mkdir(parents=True, exist_ok=True)
      lat_path = out_dir / f"{args.tag}_stage1_latents.npz"
      np.savez(
          lat_path,
          video_latent=lat_v,
          audio_latent=lat_a,
          resolution=np.array(args.resolution),
          seed=np.array(args.seed, dtype=np.int64),
      )
      say(f"[torchtpu_magi2] Saved Stage 1 latents to {lat_path}")

  if world > 1:
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
