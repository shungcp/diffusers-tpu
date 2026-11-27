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


_ACTIVE_MAGI2_HOOKS: Any = None
_ACTIVE_SAMPLE_LENS: tuple[int, ...] = ()


def _top_run_mm_layer(
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
      _ACTIVE_SAMPLE_LENS,
      hooks=_ACTIVE_MAGI2_HOOKS,
  )


def _top_run_moe_layer(
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
      _ACTIVE_SAMPLE_LENS,
      hooks=_ACTIVE_MAGI2_HOOKS,
  )


def _top_run_group(
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
        _ACTIVE_SAMPLE_LENS,
        hooks=_ACTIVE_MAGI2_HOOKS,
    )
  return x_sorted


def _top_run_post(post_mod, x_sorted, inv_perm_local, mod_ids):
  x_seq = x_sorted.index_select(0, inv_perm_local)
  return type(post_mod).forward(post_mod, x_seq, mod_ids, hooks=_ACTIVE_MAGI2_HOOKS)


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
      sparse_spec: tuple[int, int, int] | None = None,
  ) -> None:
    self.model = model
    self.sample_lens = tuple(int(s) for s in sample_lens)
    self.hooks = hooks
    self.aot_group = max(1, int(aot_group))
    self.use_compile = use_compile
    self.sparse_spec = sparse_spec

    persist_dir = os.environ.get("DT_AOT_PERSIST", "").strip()
    if persist_dir == "0":
      persist_dir = ""

    def _make_fn(raw_fn, prog_tag: str):
      if not use_compile:
        return raw_fn
      compiled = _wrap_spmd_compile(raw_fn, use_spmd_safe=use_spmd_safe)
      if not persist_dir or not hasattr(compiled, "aot_compile"):
        return compiled
      import hashlib

      rank_str = os.environ.get("RANK", "0")
      world_str = os.environ.get("WORLD_SIZE", "1")
      salt = (
          prog_tag,
          self.sample_lens,
          self.aot_group,
          world_str,
          str(self.model.dtype),
          os.environ.get("DT_MAGI2_AG_CAP_FACTOR", ""),
          bool(getattr(self.hooks, "fp8_moe", False)),
          bool(getattr(self.hooks, "use_splash", True)),
          self.sparse_spec,
      )
      h = hashlib.sha1(repr(salt).encode("utf-8")).hexdigest()[:16]
      os.makedirs(persist_dir, exist_ok=True)
      bin_path = os.path.join(persist_dir, f"magi2_{prog_tag}_{h}.bin")
      art_holder: list[Any] = [None]
      tried: list[bool] = [False]

      def _call_persisted(*args):
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
              if rank_str == "0":
                print(
                    f"  [aot] loaded {os.path.basename(bin_path)} in {time.perf_counter() - t_ld:.2f}s",
                    flush=True,
                )
              return art(*args)
            except Exception as exc:
              if rank_str == "0":
                print(
                    f"  [aot] cached {os.path.basename(bin_path)} unusable ({type(exc).__name__}); recompiling",
                    flush=True,
                )
          try:
            art = compiled.aot_compile((args, {}))
            art_holder[0] = art
            if rank_str == "0" and hasattr(art, "save_compiled_function"):
              tmp_p = f"{bin_path}.tmp.{os.getpid()}"
              art.save_compiled_function(tmp_p)
              os.replace(tmp_p, bin_path)
              print(f"  [aot] saved {os.path.basename(bin_path)}", flush=True)
            return art(*args)
          except Exception as exc:
            if rank_str == "0":
              print(
                  f"  [aot] fallback to torch.compile for {prog_tag} ({type(exc).__name__}: {str(exc)[:120]})",
                  flush=True,
              )
        return compiled(*args)

      return _call_persisted

    self._mm_fn = _make_fn(_top_run_mm_layer, "mm")
    self._moe_fn = _make_fn(_top_run_moe_layer if self.aot_group == 1 else _top_run_group, "moe")
    self._post_fn = _make_fn(_top_run_post, "post")

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
    global _ACTIVE_MAGI2_HOOKS, _ACTIVE_SAMPLE_LENS
    _ACTIVE_MAGI2_HOOKS = self.hooks
    _ACTIVE_SAMPLE_LENS = self.sample_lens
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


# ---------------------------------------------------------------------------
# Stage 2 1080p Refiner (6.72B Dense GQA, DP=2 x CP=4, DT_AOT_PERSIST)
# ---------------------------------------------------------------------------

_ACTIVE_REFINER_HOOKS: Any = None


def _top_refiner_mm_layer(mod, x_local, rope_96_local, kv_blen, mod_ids_local, group_sizes_local):
  return type(mod).forward(
      mod,
      x_local,
      rope_96_local,
      kv_blen,
      mod_ids_local,
      group_sizes_local,
      hooks=_ACTIVE_REFINER_HOOKS,
  )


def _top_refiner_dense_layer(mod, x_local, rope_96_local, kv_blen):
  return type(mod).forward(
      mod,
      x_local,
      rope_96_local,
      kv_blen,
      None,
      None,
      hooks=_ACTIVE_REFINER_HOOKS,
  )


class CompiledMagi2RefinerTT:
  """AOT-compiled 30-layer 6.72B ``Magi2RefinerTransformerTorch`` (2 unique layer graphs)."""

  def __init__(
      self,
      model: Any,
      *,
      layout: Any,
      hooks: Any,
      cfg_rank: int,
      cp_rank: int,
      cp_size: int,
      use_compile: bool = True,
      use_spmd_safe: bool = True,
  ) -> None:
    self.model = model
    self.layout = layout
    self.hooks = hooks
    self.cfg_rank = int(cfg_rank)
    self.cp_rank = int(cp_rank)
    self.cp_size = int(cp_size)

    persist_dir = os.environ.get("DT_AOT_PERSIST", "").strip()
    if persist_dir == "0":
      persist_dir = ""

    def _make_fn(raw_fn, prog_tag: str):
      if not use_compile:
        return raw_fn
      compiled = _wrap_spmd_compile(raw_fn, use_spmd_safe=use_spmd_safe)
      if not persist_dir or not hasattr(compiled, "aot_compile"):
        return compiled
      import hashlib

      salt = (
          "refiner_v1",
          prog_tag,
          layout.block_grid_thw,
          layout.radius,
          int(layout.n_vid_pad),
          int(layout.n_tail_pad),
          self.cfg_rank,
          self.cp_size,
          str(self.model.dtype),
          bool(getattr(self.hooks, "qk_fp8", False)),
      )
      h = hashlib.sha1(repr(salt).encode("utf-8")).hexdigest()[:16]
      os.makedirs(persist_dir, exist_ok=True)
      bin_path = os.path.join(persist_dir, f"magi2_refiner_{prog_tag}_{h}.bin")
      art_holder: list[Any] = [None]
      tried: list[bool] = [False]

      def _call_persisted(*args):
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
              if self.cfg_rank == 0 and self.cp_rank == 0:
                print(
                    f"  [aot/refiner] loaded {os.path.basename(bin_path)} in {time.perf_counter() - t_ld:.2f}s",
                    flush=True,
                )
              return art(*args)
            except Exception as exc:
              if self.cfg_rank == 0 and self.cp_rank == 0:
                print(
                    f"  [aot/refiner] cached {os.path.basename(bin_path)} unusable ({type(exc).__name__}); recompiling",
                    flush=True,
                )
          try:
            art = compiled.aot_compile((args, {}))
            art_holder[0] = art
            if self.cp_rank == 0 and hasattr(art, "save_compiled_function"):
              tmp_p = f"{bin_path}.tmp.{os.getpid()}"
              art.save_compiled_function(tmp_p)
              os.replace(tmp_p, bin_path)
              if self.cfg_rank == 0:
                print(f"  [aot/refiner] saved {os.path.basename(bin_path)}", flush=True)
            return art(*args)
          except Exception as exc:
            if self.cfg_rank == 0 and self.cp_rank == 0:
              print(
                  f"  [aot/refiner] fallback to torch.compile for {prog_tag} ({type(exc).__name__}: {str(exc)[:120]})",
                  flush=True,
              )
        return compiled(*args)

      return _call_persisted

    self._mm_fn = _make_fn(_top_refiner_mm_layer, "mm")
    self._dense_fn = _make_fn(_top_refiner_dense_layer, "dense")

  @torch.no_grad()
  def forward_local(
      self,
      x_local: torch.Tensor,
      rope_96_local: torch.Tensor,
      kv_blen: torch.Tensor,
      mod_ids_local: torch.Tensor,
      group_sizes_local: torch.Tensor,
  ) -> torch.Tensor:
    global _ACTIVE_REFINER_HOOKS
    _ACTIVE_REFINER_HOOKS = self.hooks
    mm_set = set(self.model.config.mm_layers)
    # Quantize to model.dtype (bf16) matching pre_adapter cast, then promote to
    # fp32 so layer 0 shares the exact same AOT artifact dtype as layers 1/28/29.
    h = x_local.to(self.model.dtype).float()
    for li, layer in enumerate(self.model.layers):
      if li in mm_set:
        h = self._mm_fn(layer, h, rope_96_local, kv_blen, mod_ids_local, group_sizes_local)
      else:
        h = self._dense_fn(layer, h, rope_96_local, kv_blen)
    return h


def _run_torchtpu_stage2_refiner(
    *,
    args: argparse.Namespace,
    model_root: Path,
    stage1_video_np: np.ndarray,
    stage1_audio_np: np.ndarray,
    cond_txt_np: np.ndarray,
    uncond_txt_np: np.ndarray,
    refiner_noise_np: np.ndarray,
    rank: int,
    world: int,
    device: torch.device,
    h2d_fn: Any,
    say: Any,
) -> np.ndarray:
  """Run Stage 2 1080p 6.72B Refiner natively in ``torch_tpu`` (DP=2 x CP=4)."""
  import gc
  import math
  from ..models.magi2.config import Magi2RefinerConfig
  from ..models.magi2.jax_pipeline import (
      FlowUniPCMultistepScheduler,
      refiner_noise_augment,
      trilinear_upsample_latents,
  )
  from ..models.magi2.jax_refiner_dit import (
      build_refiner_attention_layout,
      pack_refiner_inputs_np,
  )
  from ..models.magi2.torch_refiner_dit import load_torch_refiner_from_safetensors
  from .magi2_kernels_tt import build_refiner_torchtpu_hooks

  refiner_dir = model_root / "refiner"
  if not refiner_dir.exists():
    raise FileNotFoundError(f"Stage 2 Refiner checkpoint dir not found: {refiner_dir}")

  ref_cfg_json = refiner_dir / "config.json"
  ref_cfg_obj = Magi2RefinerConfig.from_json(ref_cfg_json) if ref_cfg_json.exists() else Magi2RefinerConfig()
  preset_1080 = RESOLUTION_PRESETS["1080p"]
  ref_lat_h = int(preset_1080["refiner_height"]) // 16
  ref_lat_w = int(preset_1080["refiner_width"]) // 16
  ref_lat_t = int(stage1_video_np.shape[2]) * 2 - 1
  target_thw = (ref_lat_t, ref_lat_h, ref_lat_w)

  # Trilinear upsample + VP noise augment (bit-identical to jax_pipeline.py).
  z_up_np = trilinear_upsample_latents(stage1_video_np, target_thw)
  if args.refiner_noise_value != 0:
    z_ref_init_np = refiner_noise_augment(
        z_up_np, refiner_noise_np, noise_value=args.refiner_noise_value
    )
  else:
    z_ref_init_np = z_up_np

  ref_radius: tuple[int, int, int] | None = None
  if args.refiner_radius.strip():
    parts_r = [int(p.strip()) for p in args.refiner_radius.split(",")]
    if len(parts_r) != 3 or any(v < 0 for v in parts_r):
      raise ValueError(f"--refiner-radius must be 'rt,rh,rw' (got {args.refiner_radius!r})")
    ref_radius = (parts_r[0], parts_r[1], parts_r[2])

  # DP=2 x CP=4 CFG-parallel subgroups when world >= 2 and even (v7x-8: ranks [0..3] cond, [4..7] uncond).
  use_dp2 = (
      world >= 2
      and world % 2 == 0
      and ref_cfg_obj.num_heads_q % (world // 2) == 0
      and ref_cfg_obj.num_heads_kv % (world // 2) == 0
  )
  cp_size = (world // 2) if use_dp2 else world
  cfg_rank = (rank // cp_size) if use_dp2 else 0
  cp_rank = (rank % cp_size) if use_dp2 else rank

  if use_dp2:
    cp_pg_0 = dist.new_group(list(range(0, cp_size)), backend="tpu_dist")
    cp_pg_1 = dist.new_group(list(range(cp_size, world)), backend="tpu_dist")
    my_cp_pg = cp_pg_0 if cfg_rank == 0 else cp_pg_1
    my_cp_group_name = my_cp_pg.group_name
    my_cfg_pg = None
    for r0 in range(cp_size):
      pair_pg = dist.new_group([r0, r0 + cp_size], backend="tpu_dist")
      if rank in (r0, r0 + cp_size):
        my_cfg_pg = pair_pg
  else:
    my_cp_pg = dist.group.WORLD if world > 1 else None
    my_cp_group_name = my_cp_pg.group_name if my_cp_pg is not None else ""
    my_cfg_pg = None

  # Shared tail bucket so cond & uncond have identical total_seq_len.
  block_vol = ref_cfg_obj.block_t_size * ref_cfg_obj.block_h_size * ref_cfg_obj.block_w_size
  l_cond = int(cond_txt_np.shape[0])
  l_uncond = int(uncond_txt_np.shape[0])
  max_txt = max(l_cond, l_uncond)
  tail_bucket = math.ceil(max_txt / block_vol) * block_vol if max_txt > 0 else 0

  layout_cond = build_refiner_attention_layout(
      target_thw,
      n_text=l_cond,
      n_audio=0,
      config=ref_cfg_obj,
      radius=ref_radius,
      tail_bucket_len=tail_bucket if tail_bucket > 0 else None,
      cp_size=cp_size,
      build_flat_tables=False,
  )
  layout_uncond = build_refiner_attention_layout(
      target_thw,
      n_text=l_uncond,
      n_audio=0,
      config=ref_cfg_obj,
      radius=ref_radius,
      tail_bucket_len=tail_bucket if tail_bucket > 0 else None,
      cp_size=cp_size,
      build_flat_tables=False,
  )
  my_layout = layout_cond if cfg_rank == 0 else layout_uncond
  my_txt_raw = cond_txt_np if cfg_rank == 0 else uncond_txt_np

  say(
      f"[torchtpu_magi2] Stage 2 Refiner: grid={layout_cond.block_grid_thw} "
      f"radius={layout_cond.radius} total_seq={layout_cond.total_seq_len} "
      f"({'DP=2xCP=' + str(cp_size) if use_dp2 else 'CP=' + str(cp_size)}, "
      f"qk_fp8={args.refiner_qk_fp8}, steps={args.refiner_steps})"
  )

  t_ld = time.perf_counter()
  ref_model = load_torch_refiner_from_safetensors(
      refiner_dir,
      config=ref_cfg_obj,
      dtype=torch.bfloat16,
      device=device,
      h2d_fn=h2d_fn,
      sync_fn=_sync,
      verbose=(rank == 0),
  )
  _sync()
  say(f"[torchtpu_magi2] Stage 2 Refiner loaded in {time.perf_counter() - t_ld:.2f}s.")

  ref_hooks = build_refiner_torchtpu_hooks(
      rank=rank,
      cp_size=cp_size,
      cp_group_name=my_cp_group_name,
      layout=my_layout,
      qk_fp8=args.refiner_qk_fp8,
  )
  ref_runner = CompiledMagi2RefinerTT(
      ref_model,
      layout=my_layout,
      hooks=ref_hooks,
      cfg_rank=cfg_rank,
      cp_rank=cp_rank,
      cp_size=cp_size,
      use_compile=not args.no_compile,
      use_spmd_safe=not args.no_spmd_safe,
  )

  # Precompute local static slices for this CP rank.
  s_tot = int(my_layout.total_seq_len)
  s_loc = s_tot // cp_size
  lo_r, hi_r = cp_rank * s_loc, (cp_rank + 1) * s_loc
  n_vid_pad = int(my_layout.n_vid_pad)

  rope_loc_t = torch.from_numpy(np.ascontiguousarray(my_layout.rope_96[lo_r:hi_r])).to(
      device=device, dtype=torch.float32
  )
  kv_blen_t = torch.from_numpy(np.ascontiguousarray(my_layout.kv_block_valid_lens)).to(
      device=device, dtype=torch.int32
  )
  # Refiner packs [0..n_vid_pad) video (mod=0), [n_vid_pad..s_tot) text (mod=2) — already sorted!
  global_pos = np.arange(lo_r, hi_r, dtype=np.int64)
  mod_ids_loc_np = np.where(global_pos < n_vid_pad, 0, 2).astype(np.int64)
  gs_loc_np = np.array([
      int((mod_ids_loc_np == 0).sum()),
      0,
      int((mod_ids_loc_np == 2).sum()),
  ], dtype=np.int32)
  mod_ids_loc_t = torch.from_numpy(mod_ids_loc_np).to(device)
  gs_loc_t = torch.from_numpy(gs_loc_np).to(device)

  _, _, txt_packed_np = pack_refiner_inputs_np(z_ref_init_np, my_txt_raw[None, ...], my_layout)
  with torch.no_grad():
    txt_emb_t = ref_model.pre_adapter._lin(
        ref_model.pre_adapter.text_embedder,
        torch.from_numpy(np.ascontiguousarray(txt_packed_np)).to(device=device, dtype=torch.float32),
        ref_hooks,
    ).to(torch.bfloat16)

  # Precompute unpack mapping on host.
  vid_src_idx = my_layout.vid_src_idx  # (n_vid_pad,) int32 (-1 for pad slots)
  valid_slots = np.where(vid_src_idx >= 0)[0]
  valid_dst = vid_src_idx[valid_slots]
  n_vid_real = ref_lat_t * ref_lat_h * ref_lat_w

  def _pack_video_np(z_ncthw: np.ndarray) -> np.ndarray:
    c_v = z_ncthw.shape[1]
    v_raster = np.transpose(z_ncthw[0], (1, 2, 3, 0)).reshape(-1, c_v)
    out_p = np.zeros((n_vid_pad, c_v), dtype=np.float32)
    out_p[valid_slots] = v_raster[valid_dst]
    return out_p

  def _unpack_video_np(v_pad_np: np.ndarray) -> np.ndarray:
    c_v = v_pad_np.shape[1]
    v_raster = np.zeros((n_vid_real, c_v), dtype=np.float32)
    v_raster[valid_dst] = v_pad_np[valid_slots]
    return np.transpose(v_raster.reshape(ref_lat_t, ref_lat_h, ref_lat_w, c_v), (3, 0, 1, 2))[None, ...]

  def _refiner_branch_step(z_ncthw: np.ndarray) -> torch.Tensor:
    """Run this rank's CFG branch (cond on cfg_rank=0, uncond on cfg_rank=1), returning local post_adapter output."""
    with torch.no_grad():
      v_pad_np = _pack_video_np(z_ncthw)
      # Each CP rank embeds ONLY its local slice of video + text tokens!
      parts_loc = []
      v_lo = min(lo_r, n_vid_pad)
      v_hi = min(hi_r, n_vid_pad)
      if v_hi > v_lo:
        v_sub = torch.from_numpy(np.ascontiguousarray(v_pad_np[v_lo:v_hi])).to(
            device=device, dtype=torch.float32
        )
        parts_loc.append(
            ref_model.pre_adapter._lin(ref_model.pre_adapter.video_embedder, v_sub, ref_hooks).to(torch.bfloat16)
        )
      t_lo = max(0, lo_r - n_vid_pad)
      t_hi = max(0, hi_r - n_vid_pad)
      if t_hi > t_lo:
        parts_loc.append(txt_emb_t[t_lo:t_hi])
      x_loc = parts_loc[0] if len(parts_loc) == 1 else torch.cat(parts_loc, dim=0)

      h_loc = ref_runner.forward_local(x_loc, rope_loc_t, kv_blen_t, mod_ids_loc_t, gs_loc_t)
      # Apply PostAdapter locally on all S_loc rows (video rows are [0..v_cnt), tail rows discarded after gather).
      out_loc = ref_model.post_adapter(h_loc, hooks=ref_hooks)  # (S_loc, 48) fp32
      return out_loc

  ref_sched = FlowUniPCMultistepScheduler()
  ref_sched.set_timesteps(args.refiner_steps, shift=args.refiner_shift)
  lat_ref = z_ref_init_np.copy()

  t_ref0 = time.perf_counter()
  ref_times: list[float] = []
  for si, t_int in enumerate(ref_sched.timesteps):
    ts0 = time.perf_counter()
    if use_dp2:
      out_loc = _refiner_branch_step(lat_ref)
      # Exchange cond/uncond across 2-rank CFG pair first (tiny S_loc x 48 fp32!), then gather across CP=4.
      pair_buf = torch.empty((2, out_loc.shape[0], out_loc.shape[1]), device=device, dtype=torch.float32)
      dist.all_gather_into_tensor(pair_buf, out_loc.contiguous(), group=my_cfg_pg)
      cond_loc, uncond_loc = pair_buf[0], pair_buf[1]
      cfg_loc = uncond_loc + float(args.refiner_cfg) * (cond_loc - uncond_loc)
      full_buf = torch.empty((cp_size * cfg_loc.shape[0], cfg_loc.shape[1]), device=device, dtype=torch.float32)
      dist.all_gather_into_tensor(full_buf, cfg_loc.contiguous(), group=my_cp_pg)
      v_pad_out = full_buf[:n_vid_pad].cpu().numpy()
    else:
      # Fallback when world == 1 or odd: run cond then uncond sequentially.
      out_c = _refiner_branch_step(lat_ref)
      if my_cp_pg is not None:
        full_c = torch.empty((cp_size * out_c.shape[0], out_c.shape[1]), device=device, dtype=torch.float32)
        dist.all_gather_into_tensor(full_c, out_c.contiguous(), group=my_cp_pg)
      else:
        full_c = out_c
      v_pad_out = full_c[:n_vid_pad].cpu().numpy()

    v_cfg_video = _unpack_video_np(v_pad_out)
    lat_ref = np.asarray(ref_sched.step(v_cfg_video, t_int, lat_ref, return_dict=False)[0], dtype=np.float32)
    dt_s = time.perf_counter() - ts0
    ref_times.append(dt_s)
    say(f"  [torchtpu/refiner] step {si + 1}/{args.refiner_steps} t={int(t_int):3d} ({dt_s:.2f}s)")

  tot_ref = time.perf_counter() - t_ref0
  steady_ref = float(np.median(ref_times[1:])) if len(ref_times) > 1 else ref_times[0]
  say(
      f"[torchtpu_magi2] Stage 2 Refiner finished: {args.refiner_steps} steps in {tot_ref:.2f}s "
      f"(compile+s0={ref_times[0]:.2f}s, steady={steady_ref:.2f} s/step) | std={float(lat_ref.std()):.4f}"
  )

  if rank == 0:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / f"{args.tag}_refiner_latents.npz"
    np.savez(
        ref_path,
        video_latent=lat_ref,
        audio_latent=stage1_audio_np,
        resolution=np.array("1080p"),
        seed=np.array(args.seed, dtype=np.int64),
    )
    say(f"[torchtpu_magi2] Saved Stage 2 1080p Refiner latents to {ref_path}")

  del ref_model, ref_runner, ref_hooks, txt_emb_t, rope_loc_t, kv_blen_t
  gc.collect()
  _sync()
  return lat_ref


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
  ap.add_argument(
      "--lossy",
      action="store_true",
      help="Enable Stage 1 ~2.6x preset (step-cache-thresh=0.30 + late half-seq cfg-skip-after=0.65).",
  )
  ap.add_argument("--step-cache-thresh", type=float, default=None, help="Audio+video TeaCache threshold (0 disables; --lossy uses 0.30).")
  ap.add_argument("--step-cache-max-skip", type=int, default=2, help="Max consecutive cached steps (default: 2).")
  ap.add_argument("--step-cache-warmup", type=float, default=0.15, help="Initial dense warmup fraction (--lossy uses 0.12).")
  ap.add_argument("--step-cache-tail", type=float, default=0.10, help="Final dense tail fraction (--lossy uses 0.08).")
  ap.add_argument("--cfg-skip-after", type=float, default=None, help="Fraction after which to run half-seq cond-only DiT (--lossy uses 0.65).")
  ap.add_argument("--sparse-window", type=int, default=0, help="Stage 1 temporal-window block-sparse Splash+Sink radius in latent frames (0 disables).")
  ap.add_argument("--sparse-dense-steps", type=int, default=15, help="Initial dense Stage 1 steps before switching to --sparse-window (default: 15).")
  ap.add_argument("--refiner", action="store_true", help="Run native torch_tpu Stage 2 1080p 6.72B Refiner after Stage 1 (auto-on when --resolution 1080p).")
  ap.add_argument("--no-refiner", action="store_true", help="Skip Stage 2 Refiner even when --resolution 1080p.")
  ap.add_argument("--refiner-steps", type=int, default=5, help="Stage 2 Refiner denoising steps (default: 5).")
  ap.add_argument("--refiner-cfg", type=float, default=2.0, help="Stage 2 Refiner video CFG scale (default: 2.0).")
  ap.add_argument("--refiner-shift", type=float, default=5.0, help="Stage 2 Refiner FlowUniPC shift (default: 5.0).")
  ap.add_argument("--refiner-noise-value", type=int, default=220, help="Stage 2 Refiner VP noise augment level (default: 220).")
  ap.add_argument("--refiner-qk-fp8", action="store_true", help="Enable E4M3 FP8 QK^T in Stage 2 Refiner Pallas kernel.")
  ap.add_argument("--refiner-radius", default="", help="Optional Stage 2 Refiner 3D radius 'rt,rh,rw' (default '2,2,2'; e.g. '1,2,2').")
  ap.add_argument("--stage1-latents", default="", help="Optional pre-computed Stage 1 latents .npz to benchmark/run Stage 2 Refiner directly.")
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
  import gc

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

  run_refiner = (
      (args.refiner or bool(preset.get("use_refiner", False)))
      and not args.no_refiner
      and (args.mode == "denoise" or bool(args.stage1_latents))
  )

  # Direct-to-TPU non-pinning int32-bitcast H2D helper.
  def _fast_h2d_no_pin(tensor: torch.Tensor, dev: torch.device) -> torch.Tensor:
    t_c = tensor.contiguous()
    flat = t_c.reshape(-1)
    if t_c.element_size() == 2 and (flat.numel() * 2) % 4 == 0:
      dt, sh = flat.dtype, t_c.shape
      return flat.view(torch.int32).to(dev).view(dt).reshape(sh)
    return t_c.to(dev)

  # Load text embeddings (needed by both Stage 1 and Stage 2 Refiner).
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

  # Initial deterministic latents (matching jax_pipeline --torch-noise!).
  preset_1080 = RESOLUTION_PRESETS["1080p"]
  ref_shape = (
      (
          1,
          cfg.video_in_channels,
          vid_t * 2 - 1,
          int(preset_1080["refiner_height"]) // 16,
          int(preset_1080["refiner_width"]) // 16,
      )
      if run_refiner
      else None
  )
  init_v, init_a, ref_noise_np = generate_initial_noise(
      video_shape=(1, cfg.video_in_channels, vid_t, lat_h, lat_w),
      audio_shape=(1, aud_t, cfg.audio_in_channels),
      refiner_shape=ref_shape,
      seed=args.seed,
      torch_noise=True,
  )

  stage1_video_np: np.ndarray | None = None
  stage1_audio_np: np.ndarray | None = None

  if args.stage1_latents and Path(args.stage1_latents).exists():
    say(f"[torchtpu_magi2] Loading precomputed Stage 1 latents from {args.stage1_latents} (skipping Stage 1) ...")
    loaded_s1 = np.load(args.stage1_latents, allow_pickle=False)
    stage1_video_np = np.asarray(loaded_s1["video_latent"], dtype=np.float32)
    stage1_audio_np = (
        np.asarray(loaded_s1["audio_latent"], dtype=np.float32)
        if "audio_latent" in loaded_s1
        else np.zeros((1, aud_t, cfg.audio_in_channels), dtype=np.float32)
    )
    if stage1_video_np.ndim == 4:
      stage1_video_np = stage1_video_np[None, ...]
    if stage1_audio_np.ndim == 2:
      stage1_audio_np = stage1_audio_np[None, ...]
  else:
    say(
        f"[torchtpu_magi2] mode={args.mode} world={world} res={args.resolution} "
        f"vid_thw=({vid_t},{lat_h},{lat_w}) aud_t={aud_t} fp8={args.fp8} "
        f"lossy={args.lossy} sparse_w={args.sparse_window} refiner={run_refiner} "
        f"aot_group={args.aot_group} compile={not args.no_compile}"
    )

    # 1. Stream EP-sharded Stage 1 weights directly to TPU HBM.
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

    # 2. Build exact sequence packing plan (matches jax_pipeline bit-for-bit!).
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
    rpf = lat_h * lat_w
    say(
        f"[torchtpu_magi2] Packed seq: S_cond={s_cond} S_uncond={s_uncond} "
        f"pad={plan['pad_size']} S_padded={s_padded} (S_local={s_loc}/rank)"
    )

    # Resolve Stage 1 acceleration knobs (matching jax_pipeline.py).
    cache_thresh = (
        float(args.step_cache_thresh)
        if args.step_cache_thresh is not None
        else (0.30 if args.lossy else 0.0)
    )
    cfg_skip_after = (
        float(args.cfg_skip_after)
        if args.cfg_skip_after is not None
        else (0.65 if args.lossy else 1.0)
    )
    cache_warmup_frac = (
        0.12 if (args.lossy and args.step_cache_warmup == 0.15) else float(args.step_cache_warmup)
    )
    cache_tail_frac = (
        0.08 if (args.lossy and args.step_cache_tail == 0.10) else float(args.step_cache_tail)
    )
    use_step_cache = cache_thresh > 0.0
    use_cfg_skip = cfg_skip_after < 1.0
    use_sparse_attn = int(args.sparse_window) > 0 and (not args.dense_attention)
    sparse_dense_steps = max(0, int(args.sparse_dense_steps))
    sparse_spec = (n_vid, rpf, int(args.sparse_window)) if use_sparse_attn else None

    if use_step_cache or use_cfg_skip or use_sparse_attn:
      say(
          f"[torchtpu_magi2] Stage 1 accel: step_cache_thresh={cache_thresh:.3f} "
          f"(max_skip={args.step_cache_max_skip}, warmup={cache_warmup_frac:.2f}, tail={cache_tail_frac:.2f}), "
          f"cfg_skip_after={cfg_skip_after:.2f}"
          + (f", sparse_window=±{args.sparse_window} (dense_steps={sparse_dense_steps})" if use_sparse_attn else "")
      )

    # 3. Pre-register all single-chip Pallas jax_ops and build runners (all share `model`, 0 extra HBM!).
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

    sp_runner = None
    if use_sparse_attn:
      sp_hooks = build_torchtpu_hooks(
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
          use_splash=True,
          sparse_spec=sparse_spec,
      )
      sp_runner = CompiledMagi2PreviewTT(
          model,
          sample_lens=(s_cond, s_uncond),
          hooks=sp_hooks,
          aot_group=args.aot_group,
          use_compile=not args.no_compile,
          use_spmd_safe=not args.no_spmd_safe,
          sparse_spec=sparse_spec,
      )

    def _build_local_static_tensors(rope_all_np, mod_all_np, tk_all_np, s_tot_len: int):
      sl = s_tot_len // world
      l_i, h_i = rank * sl, (rank + 1) * sl
      r_t = torch.from_numpy(np.ascontiguousarray(rope_all_np[l_i:h_i])).to(device=device, dtype=torch.float32)
      m_raw = mod_all_np[l_i:h_i]
      m_ids = np.where(m_raw == 3, 2, m_raw).astype(np.int64)
      p_np = np.argsort(m_ids, kind="stable").astype(np.int64)
      ip_np = np.empty_like(p_np)
      ip_np[p_np] = np.arange(sl, dtype=np.int64)
      ms_np = m_ids[p_np]
      gs_n = np.array([
          int((ms_np == 0).sum()),
          int((ms_np == 1).sum()),
          int((ms_np == 2).sum()),
      ], dtype=np.int32)
      return (
          l_i,
          h_i,
          r_t,
          torch.from_numpy(m_ids).to(device),
          torch.from_numpy(ms_np).to(device),
          torch.from_numpy(p_np).to(device),
          torch.from_numpy(ip_np).to(device),
          torch.from_numpy(gs_n).to(device),
          torch.from_numpy(np.ascontiguousarray(tk_all_np[l_i:h_i])).to(device),
      )

    (
        lo, hi, rope_local, mod_ids_local, mod_ids_sorted,
        perm_local, inv_perm_local, group_sizes, time_kind_local,
    ) = _build_local_static_tensors(plan["rope_all"], plan["mod_all"], plan["time_kind_all"], s_padded)

    cond_runner = None
    sp_cond_runner = None
    if use_cfg_skip:
      pad_cond_len = (-s_cond) % world
      s_padded_cond = s_cond + pad_cond_len
      rope_c_np = plan["rope_all"][:s_cond]
      mod_c_np = plan["mod_all"][:s_cond]
      tk_c_np = plan["time_kind_all"][:s_cond]
      if pad_cond_len > 0:
        rope_c_np = np.concatenate([rope_c_np, np.zeros((pad_cond_len, rope_c_np.shape[1]), dtype=np.float32)], axis=0)
        mod_c_np = np.concatenate([mod_c_np, np.full((pad_cond_len,), 2, dtype=np.int32)], axis=0)
        tk_c_np = np.concatenate([tk_c_np, np.zeros((pad_cond_len,), dtype=np.int32)], axis=0)
      (
          lo_c, hi_c, rope_local_c, mod_ids_local_c, mod_ids_sorted_c,
          perm_local_c, inv_perm_local_c, group_sizes_c, time_kind_local_c,
      ) = _build_local_static_tensors(rope_c_np, mod_c_np, tk_c_np, s_padded_cond)

      cond_hooks = build_torchtpu_hooks(
          rank=rank,
          world=world,
          group_name=group_name,
          s_total=s_padded_cond,
          sample_lens=(s_cond,),
          device=device,
          num_heads_q=cfg.num_heads_q,
          e_local=cfg.moe_num_experts // world,
          top_k=cfg.moe_top_k,
          route_scale=cfg.moe_route_scale,
          cap_factor=args.cap_factor,
          fp8_moe=args.fp8,
          use_splash=not args.dense_attention,
      )
      cond_runner = CompiledMagi2PreviewTT(
          model,
          sample_lens=(s_cond,),
          hooks=cond_hooks,
          aot_group=args.aot_group,
          use_compile=not args.no_compile,
          use_spmd_safe=not args.no_spmd_safe,
      )
      if use_sparse_attn:
        sp_cond_hooks = build_torchtpu_hooks(
            rank=rank,
            world=world,
            group_name=group_name,
            s_total=s_padded_cond,
            sample_lens=(s_cond,),
            device=device,
            num_heads_q=cfg.num_heads_q,
            e_local=cfg.moe_num_experts // world,
            top_k=cfg.moe_top_k,
            route_scale=cfg.moe_route_scale,
            cap_factor=args.cap_factor,
            fp8_moe=args.fp8,
            use_splash=True,
            sparse_spec=sparse_spec,
        )
        sp_cond_runner = CompiledMagi2PreviewTT(
            model,
            sample_lens=(s_cond,),
            hooks=sp_cond_hooks,
            aot_group=args.aot_group,
            use_compile=not args.no_compile,
            use_spmd_safe=not args.no_spmd_safe,
            sparse_spec=sparse_spec,
        )

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
      if use_cfg_skip:
        if pad_cond_len > 0:
          pad_cond_emb = torch.zeros((pad_cond_len, cfg.stream_hidden_size), device=device, dtype=torch.bfloat16)
        else:
          pad_cond_emb = torch.empty((0, cfg.stream_hidden_size), device=device, dtype=torch.bfloat16)

    def _make_time_local(raw_t: float, tk_loc: torch.Tensor) -> torch.Tensor:
      emb_noisy = sinusoidal_embedding_1d_torch(torch.tensor(raw_t, device=device, dtype=torch.float32), dim=cfg.time_channel_dim)
      emb_clean = sinusoidal_embedding_1d_torch(torch.tensor(0.0, device=device, dtype=torch.float32), dim=cfg.time_channel_dim)
      emb_zero = torch.zeros((cfg.time_channel_dim,), device=device, dtype=torch.float32)
      return torch.where(
          (tk_loc == 2).unsqueeze(-1),
          emb_noisy.unsqueeze(0),
          torch.where((tk_loc == 1).unsqueeze(-1), emb_clean.unsqueeze(0), emb_zero.unsqueeze(0)),
      )

    def _run_step(
        latent_v_cpu: np.ndarray,
        latent_a_cpu: np.ndarray,
        raw_t: float,
        *,
        return_delta: bool = False,
        use_sparse: bool = False,
    ):
      with torch.no_grad():
        v_flat = np.transpose(latent_v_cpu[0], (1, 2, 3, 0)).reshape(n_vid, cfg.video_in_channels)
        v_emb = _fp32_linear(model.pre_adapter.video_embedder, torch.from_numpy(np.ascontiguousarray(v_flat)).to(device)).to(torch.bfloat16)
        a_emb = _fp32_linear(model.pre_adapter.audio_embedder, torch.from_numpy(np.ascontiguousarray(latent_a_cpu[0])).to(device)).to(torch.bfloat16)

        parts = [v_emb, a_emb, txt_cond_emb, v_emb, a_emb, txt_uncond_emb]
        if pad_sz > 0:
          parts.append(pad_emb)
        x_emb_local = torch.cat(parts, dim=0)[lo:hi].clone()
        time_local = _make_time_local(raw_t, time_kind_local)

        active_r = sp_runner if (use_sparse and sp_runner is not None) else runner
        out_local = active_r.forward_embedded_local(
            x_emb_local, rope_local, mod_ids_local, mod_ids_sorted,
            group_sizes, perm_local, inv_perm_local, time_local,
        )

        if world > 1:
          out_all = torch.empty((world * out_local.shape[0], out_local.shape[1]), device=out_local.device, dtype=out_local.dtype)
          dist.all_gather_into_tensor(out_all, out_local.contiguous(), group=pg)
        else:
          out_all = out_local

        cond_v = out_all[0:n_vid, : cfg.video_in_channels].reshape(vid_t, lat_h, lat_w, cfg.video_in_channels).permute(3, 0, 1, 2).unsqueeze(0)
        cond_a = out_all[n_vid : n_vid + aud_t, : cfg.audio_in_channels].unsqueeze(0)
        uncond_v = out_all[s_cond : s_cond + n_vid, : cfg.video_in_channels].reshape(vid_t, lat_h, lat_w, cfg.video_in_channels).permute(3, 0, 1, 2).unsqueeze(0)
        uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + aud_t, : cfg.audio_in_channels].unsqueeze(0)

        dv = (cond_v - uncond_v).float()
        da = (cond_a - uncond_a).float()
        v_cfg = (uncond_v.float() + args.video_cfg * dv).cpu().numpy()
        a_cfg = (uncond_a.float() + args.audio_cfg * da).cpu().numpy()
        if return_delta:
          return v_cfg, a_cfg, dv.cpu().numpy(), da.cpu().numpy()
        return v_cfg, a_cfg

    def _run_cond_step(
        latent_v_cpu: np.ndarray,
        latent_a_cpu: np.ndarray,
        raw_t: float,
        decay_val: float,
        cached_dv_cpu: np.ndarray,
        cached_da_cpu: np.ndarray,
        *,
        use_sparse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
      with torch.no_grad():
        v_flat = np.transpose(latent_v_cpu[0], (1, 2, 3, 0)).reshape(n_vid, cfg.video_in_channels)
        v_emb = _fp32_linear(model.pre_adapter.video_embedder, torch.from_numpy(np.ascontiguousarray(v_flat)).to(device)).to(torch.bfloat16)
        a_emb = _fp32_linear(model.pre_adapter.audio_embedder, torch.from_numpy(np.ascontiguousarray(latent_a_cpu[0])).to(device)).to(torch.bfloat16)

        parts = [v_emb, a_emb, txt_cond_emb]
        if pad_cond_len > 0:
          parts.append(pad_cond_emb)
        x_emb_local = torch.cat(parts, dim=0)[lo_c:hi_c].clone()
        time_local_c = _make_time_local(raw_t, time_kind_local_c)

        active_cr = sp_cond_runner if (use_sparse and sp_cond_runner is not None) else cond_runner
        assert active_cr is not None
        out_local = active_cr.forward_embedded_local(
            x_emb_local, rope_local_c, mod_ids_local_c, mod_ids_sorted_c,
            group_sizes_c, perm_local_c, inv_perm_local_c, time_local_c,
        )

        if world > 1:
          out_all = torch.empty((world * out_local.shape[0], out_local.shape[1]), device=out_local.device, dtype=out_local.dtype)
          dist.all_gather_into_tensor(out_all, out_local.contiguous(), group=pg)
        else:
          out_all = out_local

        cond_v = out_all[0:n_vid, : cfg.video_in_channels].reshape(vid_t, lat_h, lat_w, cfg.video_in_channels).permute(3, 0, 1, 2).unsqueeze(0).float().cpu().numpy()
        cond_a = out_all[n_vid : n_vid + aud_t, : cfg.audio_in_channels].unsqueeze(0).float().cpu().numpy()
        v_cfg = cond_v + (args.video_cfg - 1.0) * decay_val * cached_dv_cpu
        a_cfg = cond_a + (args.audio_cfg - 1.0) * decay_val * cached_da_cpu
        return v_cfg, a_cfg

    if args.mode == "bench":
      bench_sparse = use_sparse_attn
      say(f"[torchtpu_magi2] Running {args.warmup} warmup step(s) (includes AOT compile, sparse={bench_sparse}) ...")
      t_w = time.perf_counter()
      v_out, a_out = _run_step(init_v, init_a, 0.999, use_sparse=bench_sparse)
      _sync()
      t_compile = time.perf_counter() - t_w
      say(
          f"[torchtpu_magi2] Warmup/compile done in {t_compile:.2f}s | "
          f"v_out std={float(v_out.std()):.4f} a_out std={float(a_out.std()):.4f}"
      )
      for _ in range(max(0, args.warmup - 1)):
        v_out, a_out = _run_step(init_v, init_a, 0.999, use_sparse=bench_sparse)
        _sync()

      ts: list[float] = []
      for it in range(args.bench_iters):
        ts0 = time.perf_counter()
        v_out, a_out = _run_step(init_v, init_a, 0.999, use_sparse=bench_sparse)
        _sync()
        dt_s = time.perf_counter() - ts0
        ts.append(dt_s)
        say(f"  [iter {it}] {dt_s:.3f} s/step")
      med = float(np.median(ts))
      say(
          f"[torchtpu_magi2] SUMMARY res={args.resolution} fp8={args.fp8} sparse_w={args.sparse_window}: "
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
            "sparse_window": args.sparse_window,
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
      # Full FlowUniPC denoising loop (with optional TeaCache + half-seq CFG skip + sparse Splash+Sink).
      v_sched = FlowUniPCMultistepScheduler()
      a_sched = FlowUniPCMultistepScheduler()
      v_sched.set_timesteps(args.steps, shift=args.shift)
      a_sched.set_timesteps(args.steps, shift=args.shift)

      lat_v = init_v.copy()
      lat_a = init_a.copy()
      n_steps_tot = len(v_sched.timesteps)
      cache_start = int(round(n_steps_tot * cache_warmup_frac))
      cache_end = n_steps_tot - int(round(n_steps_tot * cache_tail_frac))
      cfg_skip_step = int(round(n_steps_tot * cfg_skip_after))
      max_consec_skip = max(0, int(args.step_cache_max_skip))

      def _vel_rel_np(v_cur: np.ndarray, v_prev: np.ndarray, a_cur: np.ndarray, a_prev: np.ndarray) -> tuple[float, float]:
        rv = float(np.sum(np.abs(v_cur - v_prev)) / max(float(np.sum(np.abs(v_cur))), 1e-6))
        ra = float(np.sum(np.abs(a_cur - a_prev)) / max(float(np.sum(np.abs(a_cur))), 1e-6))
        return rv, ra

      # Warmup compile step (dense full)
      t_w = time.perf_counter()
      _run_step(lat_v, lat_a, 0.999)
      _sync()
      t_compile = time.perf_counter() - t_w
      say(f"[torchtpu_magi2] Denoise compile done in {t_compile:.2f}s. Running {n_steps_tot} steps ...")

      prev_v_cfg: np.ndarray | None = None
      prev_a_cfg: np.ndarray | None = None
      cached_dv: np.ndarray | None = None
      cached_da: np.ndarray | None = None
      accum_rel_err = 0.0
      last_rel_rate = 1.0
      consec_skips = 0
      n_cached = 0
      n_cond_only = 0
      n_full_cfg = 0

      t_den = time.perf_counter()
      step_times: list[float] = []
      for si, t_int in enumerate(v_sched.timesteps):
        ts0 = time.perf_counter()
        raw_t = float(np.int64(t_int)) / 1000.0
        step_is_sparse = use_sparse_attn and si >= sparse_dense_steps
        can_cache = (
            use_step_cache
            and prev_v_cfg is not None
            and cache_start <= si < cache_end
            and consec_skips < max_consec_skip
            and (accum_rel_err + last_rel_rate) < cache_thresh
        )
        if can_cache:
          v_cfg, a_cfg = prev_v_cfg, prev_a_cfg
          accum_rel_err += last_rel_rate
          consec_skips += 1
          n_cached += 1
          tag_s = "cache"
        elif use_cfg_skip and si >= cfg_skip_step and cached_dv is not None and cached_da is not None:
          frac_late = (si - cfg_skip_step) / max(1, n_steps_tot - cfg_skip_step)
          decay_val = max(0.30, 1.0 - 0.70 * frac_late)
          v_cfg, a_cfg = _run_cond_step(
              lat_v, lat_a, raw_t, decay_val, cached_dv, cached_da, use_sparse=step_is_sparse
          )
          n_cond_only += 1
          if use_step_cache and prev_v_cfg is not None and prev_a_cfg is not None:
            rv_f, ra_f = _vel_rel_np(v_cfg, prev_v_cfg, a_cfg, prev_a_cfg)
            last_rel_rate = max(rv_f, 0.7 * ra_f) / float(consec_skips + 1)
          accum_rel_err = 0.0
          consec_skips = 0
          prev_v_cfg, prev_a_cfg = v_cfg, a_cfg
          tag_s = "cond_sp" if step_is_sparse else "cond"
        elif use_cfg_skip:
          v_cfg, a_cfg, cached_dv, cached_da = _run_step(
              lat_v, lat_a, raw_t, return_delta=True, use_sparse=step_is_sparse
          )
          n_full_cfg += 1
          if use_step_cache and prev_v_cfg is not None and prev_a_cfg is not None:
            rv_f, ra_f = _vel_rel_np(v_cfg, prev_v_cfg, a_cfg, prev_a_cfg)
            last_rel_rate = max(rv_f, 0.7 * ra_f) / float(consec_skips + 1)
          accum_rel_err = 0.0
          consec_skips = 0
          prev_v_cfg, prev_a_cfg = v_cfg, a_cfg
          tag_s = "full_sp" if step_is_sparse else "full"
        else:
          v_cfg, a_cfg = _run_step(lat_v, lat_a, raw_t, use_sparse=step_is_sparse)
          n_full_cfg += 1
          if use_step_cache:
            if prev_v_cfg is not None and prev_a_cfg is not None:
              rv_f, ra_f = _vel_rel_np(v_cfg, prev_v_cfg, a_cfg, prev_a_cfg)
              last_rel_rate = max(rv_f, 0.7 * ra_f) / float(consec_skips + 1)
            accum_rel_err = 0.0
            consec_skips = 0
            prev_v_cfg, prev_a_cfg = v_cfg, a_cfg
          tag_s = "full_sp" if step_is_sparse else "full"

        lat_v = np.asarray(v_sched.step(v_cfg, t_int, lat_v, return_dict=False)[0], dtype=np.float32)
        lat_a = np.asarray(a_sched.step(a_cfg, t_int, lat_a, return_dict=False)[0], dtype=np.float32)
        dt_s = time.perf_counter() - ts0
        step_times.append(dt_s)
        if si < 3 or (si + 1) % 10 == 0 or si == n_steps_tot - 1:
          say(f"  step {si + 1:3d}/{n_steps_tot} [{tag_s:7s}] t={raw_t:.3f} ({dt_s:.3f}s)")

      total_den = time.perf_counter() - t_den
      say(
          f"[torchtpu_magi2] Stage 1 Denoise finished: {n_steps_tot} steps in {total_den:.2f}s "
          f"(eff={total_den / max(1, n_steps_tot):.3f} s/step; full={n_full_cfg}, cond={n_cond_only}, cached={n_cached}) | "
          f"video std={float(lat_v.std()):.4f} audio std={float(lat_a.std()):.4f}"
      )
      stage1_video_np = lat_v
      stage1_audio_np = lat_a
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

    # Explicitly free 113.93B Stage 1 model before optional Stage 2 1080p Refiner!
    del model, runner, hooks, txt_cond_emb, txt_uncond_emb, pad_emb
    if sp_runner is not None:
      del sp_runner
    if cond_runner is not None:
      del cond_runner
    if sp_cond_runner is not None:
      del sp_cond_runner
    gc.collect()
    _sync()

  # Optional Stage 2 1080p Refiner natively in torch_tpu (DP=2 x CP=4).
  if run_refiner and stage1_video_np is not None and stage1_audio_np is not None:
    if ref_noise_np is None or ref_noise_np.shape[2] != stage1_video_np.shape[2] * 2 - 1:
      _, _, ref_noise_np = generate_initial_noise(
          video_shape=stage1_video_np.shape,
          audio_shape=stage1_audio_np.shape,
          refiner_shape=(
              1,
              cfg.video_in_channels,
              stage1_video_np.shape[2] * 2 - 1,
              int(preset_1080["refiner_height"]) // 16,
              int(preset_1080["refiner_width"]) // 16,
          ),
          seed=args.seed,
          torch_noise=True,
      )
    _run_torchtpu_stage2_refiner(
        args=args,
        model_root=model_root,
        stage1_video_np=stage1_video_np,
        stage1_audio_np=stage1_audio_np,
        cond_txt_np=cond_txt_np,
        uncond_txt_np=uncond_txt_np,
        refiner_noise_np=ref_noise_np,
        rank=rank,
        world=world,
        device=device,
        h2d_fn=_fast_h2d_no_pin,
        say=say,
    )

  if world > 1:
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
