"""End-to-end native JAX / TPU inference pipeline for Sand.ai MAGI-2-preview.

Orchestrates all four stages on TPU (v7x-8 target, single-chip / CPU compatible)
with staged HBM management so peak device memory stays <= ~45 GiB/chip:

  Stage 0  Text (+ optional I2V reference image) encoding:
           - ``Qwen35TextEncoder`` (27B, TP=8, ``skip_layer=2``, CJK split
             pre-tokenizer patch, no chat template, ``DEFAULT_NEGATIVE_PROMPT``).
           - Optional ``Wan2_2_VAEEncoder`` (fp32) for I2V reference images
             (letterbox ``resizepad`` + ``<Figure 1>`` token pooling).
           - Supports ``--text-embeds-cache <path.npz>`` to reuse precomputed
             text embeddings across repeated DiT benchmarks.
           - Explicitly frees device weights + GC before Stage 1.

  Stage 1  Preview DiT (113.93B / 5.95B active Multi-Head Latent MoE, 40 layers):
           - Ulysses CP=8 + EP=8 over ``mesh("tp")``, packed ``B=2`` CFG
             ``[cond, uncond]`` with exact upstream ``[video, audio, text,
             (ref_special, ref_patches)...]`` sequence layout and isolated CP
             alignment padding.
           - Pre-embeds static text / ref / pad tokens once via ``PreAdapter``
             and runs a single fused ``jax.jit`` + ``shard_map`` step per
             ``FlowUniPCMultistepScheduler`` iteration (separate video & audio
             schedulers, ``shift=7.0``, ``steps=100``, ``video_cfg=5.0``,
             ``audio_cfg=7.0``).
           - Supports ``--moe-mode {auto,ag,ep,ref}``, ``--fp8`` (W8A8 MoE),
             ``--warmup``, ``--runs N`` (resident steady-state benchmarking),
             and ``--profile-dir <dir>`` (TPU XProf trace on steady step).
           - Explicitly frees device weights + GC before Stage 2.

  Stage 2  Refiner DiT (6.72B dense GQA 32Q/8KV, 30 layers, 1080p only):
           - Exact ``align_corners=True`` trilinear upsampling
             ``(32, 32, 56) -> (63, 68, 120)`` + VP noise augmentation
             (``noise_value=220``: ``0.838910 * z_up + 0.544270 * eps``).
           - 3D Block-Grid sparse Pallas Splash Attention with shared tail
             bucket so cond & uncond reuse one compiled XLA graph across 5
             ``FlowUniPCMultistepScheduler`` steps (``shift=5.0``,
             ``video_cfg=2.0``, video-only).
           - Supports ``--refiner-sharding {replicated,fsdp,tp}`` and
             standalone testing via ``--load-stage1-latents <path.npz>``.
           - Explicitly frees device weights + GC before Stage 3.

  Stage 3  Video & Stereo Audio VAE Decode + ffmpeg MP4 Muxing:
           - ``TurboVAED`` (bf16, JIT-compiled sliding-window chunks with host
             offload) -> ``125`` frames @ ``12.5 fps`` (272p/540p) or ``249``
             frames @ ``25.0 fps`` (1080p).
           - ``AutoencoderOobleck`` (fp32, selective decoder-only safetensors
             load) + exact ``441/512`` sinc resampling -> 44.1 kHz stereo WAV.
           - Muxes RGB24 frames + stereo audio into ``<out>/<tag>.mp4`` via
             ``ffmpeg`` (``libx264``, ``yuv420p``, ``aac 192k``).

Example usage::

  python -u -m diffusers_tpu.models.magi2.jax_pipeline \\
      --resolution 540p --steps 100 --seed 42 --torch-noise \\
      --prompt "A red fox trotting through fresh snow." \\
      --out out --tag magi2_540p
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import wave
from typing import Any, Sequence

import numpy as np

from .config import (
    DEFAULT_NEGATIVE_PROMPT,
    Magi2PreviewConfig,
    Magi2RefinerConfig,
    Magi2TurboVAEDConfig,
    RESOLUTION_PRESETS,
)
from .rope_and_ops import (
    FlowUniPCMultistepScheduler,
    _get_coords_np,
    build_preview_rope_coords,
    elementwise_fourier_embed,
    refiner_noise_augment,
    trilinear_upsample_latents,
)

# Exact modality IDs shared with jax_preview_dit.py & upstream magi2_preview.py.
MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2

DEFAULT_REPO_ID = "sand-ai/MAGI-2-preview"
DEFAULT_PROMPT = (
    "A red fox trotting through fresh snow in a sunlit pine forest, "
    "soft crunching footsteps and gentle wind."
)
DEFAULT_LIBTPU_INIT_ARGS = (
    "--xla_tpu_dvfs_p_state=7 "
    "--xla_tpu_use_associative_scan_emitter=false "
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_async_all_to_all=true "
    "--xla_tpu_force_async_all_to_all=true "
    "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"
)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(
      description="End-to-end native JAX inference for Sand.ai MAGI-2-preview on TPU.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  # Checkpoint / model paths
  ap.add_argument(
      "--model-dir",
      default=os.environ.get("MAGI2_CKPT_ROOT", ""),
      help="Root directory containing preview/, refiner/, text_encoder/, vae/, "
      "turbo_vae/, stable-audio-open-1.0/ (defaults to $MAGI2_CKPT_ROOT or local "
      "HF snapshot of sand-ai/MAGI-2-preview).",
  )
  ap.add_argument(
      "--config",
      default="",
      help="Optional MAGI-2 JSON config (magi2_preview.json / magi2_refiner.json) "
      "for overriding architecture & sampling defaults.",
  )
  ap.add_argument("--preview-dir", default="", help="Override Stage 1 preview checkpoint dir.")
  ap.add_argument("--refiner-dir", default="", help="Override Stage 2 refiner checkpoint dir.")
  ap.add_argument("--text-encoder-dir", default="", help="Override Qwen3.5-27B text encoder dir.")
  ap.add_argument("--vae-dir", default="", help="Override Wan2.2 VAE encoder dir/pth (for I2V).")
  ap.add_argument("--turbo-vae-dir", default="", help="Override TurboVAED dir.")
  ap.add_argument("--turbo-vae-config", default="", help="Override TurboVAED JSON config path.")
  ap.add_argument("--turbo-vae-ckpt", default="", help="Override TurboVAED checkpoint path.")
  ap.add_argument("--audio-vae-dir", default="", help="Override Stable Audio Open 1.0 dir/safetensors.")

  # Prompt & conditioning
  ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Inline text prompt or JSON storyboard.")
  ap.add_argument("--prompt-file", default="", help="Read prompt from file (overrides --prompt).")
  ap.add_argument(
      "--image",
      default="",
      help="Reference image path (or JSON mapping '{\"<Figure 1>\": \"img.png\"}') for I2V.",
  )
  ap.add_argument(
      "--ref-image-type",
      default="original",
      choices=("original", "square"),
      help="I2V reference image resize policy matching upstream evaluation_config.ref_image_type.",
  )
  ap.add_argument(
      "--use-negative-prompt",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use negative prompt for unconditional CFG branch (upstream default: True).",
  )
  ap.add_argument(
      "--negative-prompt",
      default=DEFAULT_NEGATIVE_PROMPT,
      help="Negative prompt string when --use-negative-prompt is enabled.",
  )
  ap.add_argument(
      "--use-ref-for-uncond",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Keep reference image in the unconditional I2V CFG half (upstream default: True).",
  )
  ap.add_argument(
      "--text-embeds-cache",
      default="",
      help="Path to .npz file to load/save precomputed text embeddings (skips 27B encoder on hit).",
  )

  # Resolution & sampling
  ap.add_argument(
      "--resolution",
      default="540p",
      choices=tuple(RESOLUTION_PRESETS.keys()),
      help="Preset resolution: 272p (256x448), 540p (512x896), or 1080p (512x896 -> 1088x1920).",
  )
  ap.add_argument("--seconds", type=float, default=10.0, help="Clip duration in seconds.")
  ap.add_argument("--preview-height", type=int, default=0, help="Override Stage 1 pixel height.")
  ap.add_argument("--preview-width", type=int, default=0, help="Override Stage 1 pixel width.")
  ap.add_argument("--refiner-height", type=int, default=0, help="Override Stage 2 pixel height.")
  ap.add_argument("--refiner-width", type=int, default=0, help="Override Stage 2 pixel width.")
  ap.add_argument("--output-height", type=int, default=0, help="Optional final output resize height.")
  ap.add_argument("--output-width", type=int, default=0, help="Optional final output resize width.")

  ap.add_argument(
      "--steps",
      type=int,
      default=None,
      help="Stage 1 Preview denoising steps (default: config value, 100).",
  )
  ap.add_argument(
      "--shift",
      type=float,
      default=None,
      help="Stage 1 FlowUniPC shift (default: config value, 7.0).",
  )
  ap.add_argument(
      "--video-cfg",
      type=float,
      default=None,
      help="Stage 1 video CFG guidance scale (default: config value, 5.0).",
  )
  ap.add_argument(
      "--audio-cfg",
      type=float,
      default=None,
      help="Stage 1 audio CFG guidance scale (default: config value, 7.0).",
  )

  ap.add_argument(
      "--no-refiner",
      action="store_true",
      help="Disable Stage 2 Refiner even when --resolution 1080p.",
  )
  ap.add_argument(
      "--refiner-steps",
      type=int,
      default=None,
      help="Stage 2 Refiner denoising steps (default: config value, 5).",
  )
  ap.add_argument(
      "--refiner-shift",
      type=float,
      default=None,
      help="Stage 2 FlowUniPC shift (default: config value, 5.0).",
  )
  ap.add_argument(
      "--refiner-cfg",
      type=float,
      default=None,
      help="Stage 2 video CFG guidance scale (default: config value, 2.0).",
  )
  ap.add_argument(
      "--refiner-noise-value",
      type=int,
      default=None,
      help="Stage 2 ZeroSNR DDPM VP noise augmentation index (default: config value, 220; 0 disables).",
  )
  ap.add_argument(
      "--refiner-sharding",
      default="replicated",
      choices=("replicated", "fsdp", "tp"),
      help="Stage 2 Refiner weight sharding across TPU chips.",
  )
  ap.add_argument(
      "--refiner-attn-impl",
      default="",
      choices=("", "auto", "masked_pallas", "sparse_splash", "ref"),
      help="Override Stage 2 Refiner attention backend.",
  )

  # Noise & reproducibility
  ap.add_argument("--seed", type=int, default=42, help="PRNG seed for initial latents.")
  ap.add_argument(
      "--torch-noise",
      action="store_true",
      help="Draw initial noise using torch.Generator('cpu').manual_seed(seed) matching upstream order.",
  )
  ap.add_argument(
      "--init-latents",
      default="",
      help="Optional dir containing video_noise.npy, audio_noise.npy (and optional refiner_noise.npy).",
  )

  # TPU execution, quantization, benchmarking & profiling
  ap.add_argument("--chips", type=int, default=0, help="Number of TPU chips to use (0 = all).")
  ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
  ap.add_argument(
      "--preview-backend",
      default="jax",
      choices=("jax", "torchax"),
      help="Stage 1 Preview DiT execution backend ('jax' native NNX or 'torchax' PyTorch-on-JAX).",
  )
  ap.add_argument(
      "--moe-mode",
      default="auto",
      choices=("auto", "ag", "ep", "ref"),
      help="Stage 1 Multi-Head MoE dispatch mode ('ag' zero-drop all-gather, 'ep' ragged_all_to_all, 'ref').",
  )
  ap.add_argument(
      "--capacity-factor",
      type=float,
      default=None,
      help="Optional capacity factor for --moe-mode ep (default 2.0).",
  )
  ap.add_argument(
      "--fp8",
      action="store_true",
      help="Enable W8A8 E4M3 FP8 quantization for Stage 1 routed MoE weights on TPU.",
  )
  ap.add_argument(
      "--dense-attention",
      action="store_true",
      help="Disable TPU Pallas Splash attention in Stage 1 Preview (use exact fp32 fallback).",
  )
  ap.add_argument(
      "--warmup",
      action="store_true",
      help="Run a 1-step Stage 1 warmup before timed denoising to separate compile from steady state.",
  )
  ap.add_argument(
      "--runs",
      type=int,
      default=1,
      help="Number of Stage 1 runs keeping Preview DiT resident for steady-state benchmarking.",
  )
  ap.add_argument(
      "--sync-every-step",
      action="store_true",
      help="Block and log per-step latency on every denoising step.",
  )
  ap.add_argument(
      "--compile-cache",
      default="",
      help="Directory for JAX persistent compilation cache.",
  )
  ap.add_argument(
      "--profile-dir",
      default="",
      help="Optional TPU XProf trace directory (profiles --profile-step during Stage 1).",
  )
  ap.add_argument(
      "--profile-step",
      type=int,
      default=2,
      help="0-indexed Stage 1 denoising step to capture with jax.profiler.start_trace.",
  )

  # Output & stage checkpointing
  ap.add_argument("--out", default="out", help="Output directory for mp4, wav, and optional latents.")
  ap.add_argument("--tag", default="", help="Output filename stem (auto-derived if empty).")
  ap.add_argument("--no-audio", action="store_true", help="Skip Stage 3 stereo audio VAE decode.")
  ap.add_argument(
      "--adaptive-volume-scaling",
      action="store_true",
      help="Apply upstream adaptive_volume_scaling (-31 dB target RMS) to decoded audio.",
  )
  ap.add_argument("--skip-decode", action="store_true", help="Skip Stage 3 video & audio decode.")
  ap.add_argument(
      "--save-latents",
      action="store_true",
      help="Save Stage 1 (and Stage 2 if run) denoised latents to <out>/<tag>_*_latents.npz.",
  )
  ap.add_argument(
      "--load-stage1-latents",
      default="",
      help="Path to .npz (or .npy) Stage 1 latents to skip Stage 1 and run Stage 2/3 independently.",
  )
  ap.add_argument(
      "--save-frames-npy",
      action="store_true",
      help="Also save decoded uint8 frames as <out>/<tag>_frames.npy.",
  )
  ap.add_argument(
      "--decode-chunk-shard",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Shard TurboVAED independent temporal chunks across all TPU chips (bit-exact).",
  )
  ap.add_argument(
      "--uncond-cache",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Auto-cache negative-prompt Qwen3.5 embeddings on disk to skip redundant encodes.",
  )
  return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Checkpoint Path Resolution
# ---------------------------------------------------------------------------


def _first_existing(candidates: Sequence[Path]) -> Path | None:
  for p in candidates:
    if p.exists():
      return p
  return None


def resolve_checkpoint_paths(args: argparse.Namespace) -> dict[str, Path | None]:
  """Resolve all MAGI-2 component paths from CLI overrides, $MAGI2_CKPT_ROOT, or HF cache."""
  root_str = args.model_dir.strip()
  root: Path | None = Path(root_str).expanduser() if root_str else None
  if root is None:
    try:
      from ...hub import local_snapshot

      root = Path(local_snapshot(DEFAULT_REPO_ID))
    except Exception:
      root = None

  def _pick(override: str, rel_candidates: Sequence[str]) -> Path | None:
    if override:
      return Path(override).expanduser()
    if root is None:
      return None
    found = _first_existing([root / rel for rel in rel_candidates])
    return found if found is not None else (root / rel_candidates[0])

  turbo_dir = _pick(args.turbo_vae_dir, ("turbo_vae",))
  if args.turbo_vae_config:
    turbo_cfg: Path | None = Path(args.turbo_vae_config).expanduser()
  elif turbo_dir is not None:
    turbo_cfg = _first_existing([
        turbo_dir / "TurboV3-Wan22-TinyShallow_7_7.json",
        *sorted(turbo_dir.glob("*.json")),
    ])
  else:
    turbo_cfg = None

  if args.turbo_vae_ckpt:
    turbo_ckpt: Path | None = Path(args.turbo_vae_ckpt).expanduser()
  elif turbo_dir is not None:
    turbo_ckpt = _first_existing([
        turbo_dir / "checkpoint.ckpt",
        turbo_dir / "model.safetensors",
        *sorted(turbo_dir.glob("*.ckpt")),
        *sorted(turbo_dir.glob("*.safetensors")),
    ])
  else:
    turbo_ckpt = None

  return {
      "root": root,
      "preview": _pick(args.preview_dir, ("preview",)),
      "refiner": _pick(args.refiner_dir, ("refiner",)),
      "text_encoder": _pick(
          args.text_encoder_dir, ("text_encoder", "Qwen3.5-27B", "qwen35")
      ),
      "vae": _pick(args.vae_dir, ("vae",)),
      "turbo_vae_dir": turbo_dir,
      "turbo_vae_config": turbo_cfg,
      "turbo_vae_ckpt": turbo_ckpt,
      "audio_vae": _pick(
          args.audio_vae_dir, ("stable-audio-open-1.0", "audio_vae")
      ),
  }


# ---------------------------------------------------------------------------
# Upstream-Exact Prompt & I2V Image Helpers (inference_engine.py)
# ---------------------------------------------------------------------------


def parse_image_paths(image: str | None) -> dict[str, str]:
  """Normalize ``--image`` into ``{"<Figure N>": path}`` (``inference_engine.py:L259``)."""
  if not image:
    return {}
  value = image.strip()
  if not value:
    return {}
  if value.startswith("{"):
    parsed = json.loads(value)
    return {str(k): str(v) for k, v in parsed.items()}
  return {"<Figure 1>": value}


def parse_figure_id(token_key: str) -> int:
  m = re.search(r"\d+", token_key)
  return int(m.group()) if m else 1


def ensure_figure_tokens(prompt: str, figure_ids: Sequence[int]) -> str:
  """Attach ``reference_layer:The first frame refers to <Figure 1>`` if absent (``inference_engine.py:L283``)."""
  del figure_ids  # Upstream default injects the exact <Figure 1> reference_layer hint.
  try:
    prompt_obj = json.loads(prompt)
    if not isinstance(prompt_obj, dict):
      raise ValueError("prompt JSON is not an object")
    if not prompt_obj.get("reference_layer"):
      prompt_obj["reference_layer"] = ["The first frame refers to <Figure 1>"]
    return json.dumps(prompt_obj, ensure_ascii=False)
  except (json.JSONDecodeError, TypeError, ValueError):
    if "reference_layer:" in prompt and "<Figure 1>" in prompt:
      return prompt
    return prompt + "reference_layer:The first frame refers to <Figure 1>"


def resizepad(image: Any, th: int, tw: int) -> Any:
  """Letterbox-fit PIL image onto a white ``(tw, th)`` canvas (``inference_engine.py:L80``)."""
  from PIL import Image

  w, h = image.size
  if w <= 0 or h <= 0:
    raise ValueError(f"Invalid image size: width={w}, height={h}")
  scale = min(tw / w, th / h)
  target_w = max(1, int(round(w * scale)))
  target_h = max(1, int(round(h * scale)))
  resized = image.convert("RGB").resize(
      (target_w, target_h), resample=Image.Resampling.LANCZOS
  )
  canvas = Image.new("RGB", (tw, th), (255, 255, 255))
  canvas.paste(resized, ((tw - target_w) // 2, (th - target_h) // 2))
  return canvas


def preprocess_reference_image(
    image_path: str | Path,
    *,
    resize_type: str,
    preview_height: int,
    preview_width: int,
    vae_scale_factor: int = 32,
) -> np.ndarray:
  """Preprocess an I2V image to ``(1, 3, 1, H, W)`` float32 in ``[-1, 1]``.

  Exact match to ``inference_engine.py:L302-333`` (``resizepad`` +
  ``VideoProcessor(vae_scale_factor=32).preprocess``).
  """
  from PIL import Image

  pil_img = Image.open(image_path).convert("RGB")
  if resize_type == "square":
    target_h, target_w = 320, 320
  elif resize_type == "original":
    max_length = max(int(preview_height), int(preview_width))
    if pil_img.width > pil_img.height:
      target_w = max_length
      target_h = int(pil_img.height * max_length / pil_img.width)
    else:
      target_h = max_length
      target_w = int(pil_img.width * max_length / pil_img.height)
  else:
    raise ValueError(f"Invalid ref_image_type: {resize_type!r}")

  img = resizepad(pil_img, target_h, target_w)
  # Floor to multiple of vae_scale_factor (= 2 * vae_stride[1] = 32), matching VideoProcessor.
  proc_h = max(vae_scale_factor, (target_h // vae_scale_factor) * vae_scale_factor)
  proc_w = max(vae_scale_factor, (target_w // vae_scale_factor) * vae_scale_factor)
  if img.size != (proc_w, proc_h):
    img = img.resize((proc_w, proc_h), resample=Image.Resampling.LANCZOS)

  arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
  arr = arr * 2.0 - 1.0  # [-1, 1]
  # Match upstream: cast through bf16 -> float32 (inference_engine.py:L330,L333)
  # is tiny (<4e-3) and float32 input is exact for Wan2.2 VAE encoder.
  return np.transpose(arr, (2, 0, 1))[None, :, None, :, :]  # (1, 3, 1, H, W)


# ---------------------------------------------------------------------------
# Staged HBM & Mesh Helpers
# ---------------------------------------------------------------------------


def _mesh_context(mesh: Any):
  """Enter ``jax.set_mesh(mesh)`` (JAX >= 0.7) or ``mesh`` context manager."""
  if mesh is None:
    return nullcontext()
  import jax

  if hasattr(jax, "set_mesh"):
    return jax.set_mesh(mesh)
  return mesh


def _free_device_arrays(*objs: Any) -> None:
  """Explicitly delete underlying ``jax.Array`` buffers in objects/NNX modules + GC."""
  import jax
  from flax import nnx

  seen: set[int] = set()

  def _walk(obj: Any) -> None:
    if obj is None or id(obj) in seen:
      return
    seen.add(id(obj))
    if isinstance(obj, jax.Array):
      try:
        if not obj.is_deleted():
          obj.delete()
      except Exception:
        pass
      return
    if isinstance(obj, nnx.Module):
      try:
        for leaf in jax.tree.leaves(nnx.state(obj)):
          _walk(leaf)
      except Exception:
        pass
      return
    if isinstance(obj, dict):
      for v in obj.values():
        _walk(v)
      return
    if isinstance(obj, (list, tuple, set)):
      for v in obj:
        _walk(v)
      return
    if hasattr(obj, "__dict__"):
      for v in obj.__dict__.values():
        _walk(v)

  for o in objs:
    _walk(o)
  gc.collect()


def _hbm_stats_gib(devices: Sequence[Any]) -> tuple[float, float, float]:
  """Return ``(bytes_in_use, peak_bytes_in_use, bytes_limit)`` in GiB on chip 0."""
  try:
    stats = devices[0].memory_stats() or {}
    g = float(2**30)
    return (
        stats.get("bytes_in_use", 0) / g,
        stats.get("peak_bytes_in_use", 0) / g,
        stats.get("bytes_limit", 0) / g,
    )
  except Exception:
    return (0.0, 0.0, 0.0)


def _log_hbm(label: str, devices: Sequence[Any]) -> None:
  in_use, peak, limit = _hbm_stats_gib(devices)
  if limit > 0:
    print(
        f"  [hbm] {label}: in_use={in_use:.2f} GiB  peak={peak:.2f} / {limit:.2f} GiB",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Noise Generation (Exact Upstream Order & Shapes)
# ---------------------------------------------------------------------------


def generate_initial_noise(
    *,
    video_shape: tuple[int, int, int, int, int],
    audio_shape: tuple[int, int, int],
    refiner_shape: tuple[int, int, int, int, int] | None,
    seed: int,
    torch_noise: bool,
    init_latents_dir: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Generate (or load) initial fp32 noise matching upstream ``inference_engine.py`` order.

  Draw order:
    1. Stage 1 video noise: ``(1, 48, T, H/16, W/16)``
    2. Stage 1 audio noise: ``(1, 250, 64)``
    3. Stage 2 refiner noise (if enabled): ``(1, 48, 63, 68, 120)``
  """
  if init_latents_dir:
    d = Path(init_latents_dir).expanduser()
    v_noise = np.load(d / "video_noise.npy").astype(np.float32)
    a_noise = np.load(d / "audio_noise.npy").astype(np.float32)
    r_path = d / "refiner_noise.npy"
    r_noise = (
        np.load(r_path).astype(np.float32)
        if (refiner_shape is not None and r_path.exists())
        else None
    )
    if refiner_shape is not None and r_noise is None:
      _, _, r_noise = generate_initial_noise(
          video_shape=video_shape,
          audio_shape=audio_shape,
          refiner_shape=refiner_shape,
          seed=seed,
          torch_noise=torch_noise,
          init_latents_dir="",
      )
    return v_noise, a_noise, r_noise

  if torch_noise:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    v_noise = torch.randn(video_shape, generator=gen, dtype=torch.float32).numpy()
    a_noise = torch.randn(audio_shape, generator=gen, dtype=torch.float32).numpy()
    r_noise = (
        torch.randn(refiner_shape, generator=gen, dtype=torch.float32).numpy()
        if refiner_shape is not None
        else None
    )
    return v_noise, a_noise, r_noise

  import jax

  key = jax.random.PRNGKey(int(seed))
  k_v, k_a, k_r = jax.random.split(key, 3)
  v_noise = np.asarray(jax.random.normal(k_v, video_shape, dtype=np.float32))
  a_noise = np.asarray(jax.random.normal(k_a, audio_shape, dtype=np.float32))
  r_noise = (
      np.asarray(jax.random.normal(k_r, refiner_shape, dtype=np.float32))
      if refiner_shape is not None
      else None
  )
  return v_noise, a_noise, r_noise


# ---------------------------------------------------------------------------
# Stage 1 Preview Packing Metadata & Fused JIT Runner
# ---------------------------------------------------------------------------


def build_preview_packing_plan(
    *,
    video_thw: tuple[int, int, int],
    audio_len: int,
    cond_txt_len: int,
    uncond_txt_len: int,
    ref_img_hws: Sequence[tuple[int, int]],
    cp_size: int,
    head_dim: int = 128,
    rope_temperature: float = 10000.0,
    align_base: int = 48,
) -> dict[str, Any]:
  """Build host-side RoPE, modality IDs, and time-kind tags for packed B=2 CFG.

  Matches ``preview_data_proxy.py`` (`SingleData` -> `SimplePackedData` ->
  `_pad_for_ep_cp`) 100% exactly:
    - Segment 0 (cond): ``[video, audio, cond_text, (ref_special, ref_patches)...]``
    - Segment 1 (uncond): ``[video, audio, uncond_text, (ref_special, ref_patches)...]``
    - Trailing CP pad (`pad_size`): ``MODALITY_TEXT``, identity/range coords,
      zero time embedding, isolated from attention in ``Magi2PreviewAttention``.
  """
  t_v, h_v, w_v = (int(video_thw[0]), int(video_thw[1]), int(video_thw[2]))
  n_vid = t_v * h_v * w_v
  n_aud = int(audio_len)
  l_cond = int(cond_txt_len)
  l_uncond = int(uncond_txt_len)
  ref_hws = [(int(h), int(w)) for h, w in ref_img_hws]

  def _segment_mod_and_tkind(txt_len: int) -> tuple[np.ndarray, np.ndarray]:
    mods = [
        np.full((n_vid,), MODALITY_VIDEO, dtype=np.int32),
        np.full((n_aud,), MODALITY_AUDIO, dtype=np.int32),
        np.full((txt_len,), MODALITY_TEXT, dtype=np.int32),
    ]
    # time_kind: 2 = noisy video/audio (raw_t), 1 = clean text/ref (raw_t=0), 0 = CP pad (zeros).
    tkinds = [
        np.full((n_vid,), 2, dtype=np.int32),
        np.full((n_aud,), 2, dtype=np.int32),
        np.full((txt_len,), 1, dtype=np.int32),
    ]
    for h_k, w_k in ref_hws:
      mods.append(np.full((1,), MODALITY_TEXT, dtype=np.int32))
      mods.append(np.full((h_k * w_k,), MODALITY_VIDEO, dtype=np.int32))
      # Reference images are clean conditioning (raw_t = 0.0) for BOTH special token and patches!
      tkinds.append(np.full((1 + h_k * w_k,), 1, dtype=np.int32))
    return np.concatenate(mods, axis=0), np.concatenate(tkinds, axis=0)

  mod_cond, tkind_cond = _segment_mod_and_tkind(l_cond)
  mod_uncond, tkind_uncond = _segment_mod_and_tkind(l_uncond)
  s_cond = int(mod_cond.shape[0])
  s_uncond = int(mod_uncond.shape[0])
  s_real = s_cond + s_uncond

  align_to = math.lcm(align_base, max(1, int(cp_size))) if cp_size > 1 else 1
  s_padded = math.ceil(s_real / align_to) * align_to
  pad_size = s_padded - s_real

  coords_cond = build_preview_rope_coords(
      (t_v, h_v, w_v), n_aud, l_cond, ref_hws, spatial_rope_interpolation="extra"
  )
  coords_uncond = build_preview_rope_coords(
      (t_v, h_v, w_v), n_aud, l_uncond, ref_hws, spatial_rope_interpolation="extra"
  )
  coord_parts = [coords_cond, coords_uncond]
  mod_parts = [mod_cond, mod_uncond]
  tkind_parts = [tkind_cond, tkind_uncond]

  if pad_size > 0:
    coord_parts.append(_get_coords_np((pad_size, 1, 1), (pad_size, 1, 1)))
    mod_parts.append(np.full((pad_size,), MODALITY_TEXT, dtype=np.int32))
    tkind_parts.append(np.zeros((pad_size,), dtype=np.int32))

  coords_all = np.concatenate(coord_parts, axis=0)
  mod_all = np.concatenate(mod_parts, axis=0)
  time_kind_all = np.concatenate(tkind_parts, axis=0)
  rope_all = elementwise_fourier_embed(
      coords_all, head_dim=head_dim, temperature=rope_temperature
  )

  return {
      "sample_lens": (s_cond, s_uncond),
      "s_real": s_real,
      "s_padded": s_padded,
      "pad_size": pad_size,
      "n_vid": n_vid,
      "n_aud": n_aud,
      "rope_all": rope_all,
      "mod_all": mod_all,
      "time_kind_all": time_kind_all,
  }


def _make_preview_step_fn(
    model: Any,
    *,
    video_thw: tuple[int, int, int],
    n_aud: int,
    sample_lens: tuple[int, int],
    mesh: Any,
    use_splash: bool | None = None,
    moe_mode: str | None = None,
    capacity_factor: float | None = None,
):
  """Build a single fused ``jax.jit`` step around ``Magi2PreviewTransformer``.

  Pre-embeds static text/ref/pad tokens once outside the loop and embeds only
  noisy video (48->12288) & audio (64->12288) inside the step, avoiding any
  redundant 5120-wide text matmul on video tokens while remaining bit-exact
  with ``Magi2PreAdapter.__call__`` + ``forward_local``.
  """
  import jax
  from jax import lax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import NamedSharding, PartitionSpec as P

  from .jax_preview_dit import make_jitted_forward
  from .rope_and_ops import sinusoidal_embedding_1d

  t_v, h_v, w_v = video_thw
  n_vid = t_v * h_v * w_v
  s_cond, s_uncond = int(sample_lens[0]), int(sample_lens[1])
  seg_lens = (s_cond, s_uncond)
  ax = model.axis_name
  cp_size = int(dict(mesh.shape).get(ax, 1)) if mesh is not None else 1
  time_dim = model.config.time_channel_dim
  v_ch = model.config.video_in_channels
  a_ch = model.config.audio_in_channels
  model_dtype = model.dtype

  pre_gd, pre_st = nnx.split(model.pre_adapter)
  core_fn, core_state = make_jitted_forward(
      model,
      sample_lens=seg_lens,
      mesh=mesh,
      has_time=(time_dim > 0),
      embedded_input=True,
      use_splash=use_splash,
      moe_mode=moe_mode,
      capacity_factor=capacity_factor,
      scan_layers=True,
      release_model_moe=True,
  )
  state = (pre_st, core_state)

  if mesh is not None and cp_size > 1:
    seq_sharding = NamedSharding(mesh, P(ax, None))
    rep_sharding = NamedSharding(mesh, P())
  else:
    seq_sharding = None
    rep_sharding = None

  @jax.jit
  def _step_fn(
      st,
      latent_ncthw,
      audio_btc,
      raw_t_scalar,
      video_cfg,
      audio_cfg,
      txt_cond_emb,
      ref_cond_emb,
      txt_uncond_emb,
      ref_uncond_emb,
      pad_emb,
      rope_all,
      mod_all,
      time_kind_all,
  ):
    pre_s, core_s = st
    pre_mod = nnx.merge(pre_gd, pre_s)
    v_flat = jnp.transpose(latent_ncthw[0], (1, 2, 3, 0)).reshape(n_vid, v_ch)
    if seq_sharding is not None and (n_vid % cp_size == 0):
      v_flat = lax.with_sharding_constraint(v_flat, seq_sharding)
    v_emb = pre_mod.embed_video(v_flat).astype(model_dtype)
    a_emb = pre_mod.embed_audio(audio_btc[0]).astype(model_dtype)

    parts = [v_emb, a_emb, txt_cond_emb]
    if ref_cond_emb.shape[0] > 0:
      parts.append(ref_cond_emb)
    parts.extend([v_emb, a_emb, txt_uncond_emb])
    if ref_uncond_emb.shape[0] > 0:
      parts.append(ref_uncond_emb)
    if pad_emb.shape[0] > 0:
      parts.append(pad_emb)
    x_emb = jnp.concatenate(parts, axis=0)
    if seq_sharding is not None:
      x_emb = lax.with_sharding_constraint(x_emb, seq_sharding)

    if time_dim > 0:
      emb_noisy = sinusoidal_embedding_1d(raw_t_scalar, dim=time_dim)
      emb_clean = sinusoidal_embedding_1d(jnp.float32(0.0), dim=time_dim)
      emb_zero = jnp.zeros((time_dim,), dtype=jnp.float32)
      time_all = jnp.where(
          (time_kind_all == 2)[:, None],
          emb_noisy[None, :],
          jnp.where(
              (time_kind_all == 1)[:, None],
              emb_clean[None, :],
              emb_zero[None, :],
          ),
      )
      if seq_sharding is not None:
        time_all = lax.with_sharding_constraint(time_all, seq_sharding)
    else:
      time_all = None

    out_all = core_fn(core_s, x_emb, rope_all, mod_all, time_all)

    cond_v = jnp.transpose(
        out_all[0:n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2)
    )[None, ...]
    cond_a = out_all[n_vid : n_vid + n_aud, :a_ch][None, ...]

    uncond_v = jnp.transpose(
        out_all[s_cond : s_cond + n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch),
        (3, 0, 1, 2),
    )[None, ...]
    uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + n_aud, :a_ch][None, ...]

    v_cfg_video = uncond_v + video_cfg * (cond_v - uncond_v)
    v_cfg_audio = uncond_a + audio_cfg * (cond_a - uncond_a)
    if rep_sharding is not None:
      v_cfg_video = lax.with_sharding_constraint(v_cfg_video, rep_sharding)
      v_cfg_audio = lax.with_sharding_constraint(v_cfg_audio, rep_sharding)
    return v_cfg_video, v_cfg_audio

  return _step_fn, state


def _pre_embed_static_preview_tokens(
    model: Any,
    *,
    cond_txt_np: np.ndarray,
    uncond_txt_np: np.ndarray,
    ref_special_np: np.ndarray | None,
    ref_latents_np: list[np.ndarray],
    use_ref_for_uncond: bool,
    pad_size: int,
    mesh: Any,
) -> tuple[Any, Any, Any, Any, Any]:
  """Pre-embed static text, I2V reference, and CP pad tokens once via ``PreAdapter``."""
  import jax
  import jax.numpy as jnp
  from jax.sharding import NamedSharding, PartitionSpec as P

  ad = model.config.stream_hidden_size
  dt = model.dtype
  rep = NamedSharding(mesh, P()) if mesh is not None else None

  def _put(arr_np: np.ndarray) -> Any:
    j = jnp.asarray(arr_np, dtype=jnp.float32)
    return jax.device_put(j, rep) if rep is not None else j

  def _embed_txt(arr_np: np.ndarray) -> Any:
    if arr_np.shape[0] == 0:
      empty = jnp.zeros((0, ad), dtype=dt)
      return jax.device_put(empty, rep) if rep is not None else empty
    out = model.pre_adapter.embed_text(_put(arr_np)).astype(dt)
    return jax.device_put(out, rep) if rep is not None else out

  def _embed_vid(arr_np: np.ndarray) -> Any:
    out = model.pre_adapter.embed_video(_put(arr_np)).astype(dt)
    return jax.device_put(out, rep) if rep is not None else out

  cond_2d = cond_txt_np[0] if cond_txt_np.ndim == 3 else cond_txt_np
  uncond_2d = uncond_txt_np[0] if uncond_txt_np.ndim == 3 else uncond_txt_np
  txt_cond_emb = _embed_txt(cond_2d)
  txt_uncond_emb = _embed_txt(uncond_2d)

  ref_cond_parts = []
  ref_uncond_parts = []
  for k, ref_lat in enumerate(ref_latents_np):
    # ref_lat: (1, 48, 1, h_k, w_k) -> (h_k*w_k, 48) raster order
    rl = ref_lat[0] if ref_lat.ndim == 5 else ref_lat  # (48, 1, h_k, w_k)
    c_v, t_k, h_k, w_k = rl.shape
    patches_np = np.transpose(rl, (1, 2, 3, 0)).reshape(t_k * h_k * w_k, c_v)
    assert ref_special_np is not None
    spec_np = ref_special_np[k : k + 1]  # (1, 5120)
    spec_emb = _embed_txt(spec_np)
    patch_emb = _embed_vid(patches_np)
    ref_cond_parts.extend([spec_emb, patch_emb])
    if use_ref_for_uncond:
      ref_uncond_parts.extend([spec_emb, patch_emb])
    else:
      zero_patch_emb = _embed_vid(np.zeros_like(patches_np, dtype=np.float32))
      ref_uncond_parts.extend([spec_emb, zero_patch_emb])

  if ref_cond_parts:
    ref_cond_emb = jnp.concatenate(ref_cond_parts, axis=0)
    ref_uncond_emb = jnp.concatenate(ref_uncond_parts, axis=0)
    if rep is not None:
      ref_cond_emb = jax.device_put(ref_cond_emb, rep)
      ref_uncond_emb = jax.device_put(ref_uncond_emb, rep)
  else:
    empty = jnp.zeros((0, ad), dtype=dt)
    ref_cond_emb = jax.device_put(empty, rep) if rep is not None else empty
    ref_uncond_emb = jax.device_put(empty, rep) if rep is not None else empty

  if pad_size > 0:
    # Exact match to embed_text(zeros) = text_embedder_bias broadcast to (pad_size, ad).
    bias = model.pre_adapter.text_embedder_bias[...].astype(dt)
    pad_emb = jnp.broadcast_to(bias[None, :], (pad_size, ad))
    if rep is not None:
      pad_emb = jax.device_put(pad_emb, rep)
  else:
    empty = jnp.zeros((0, ad), dtype=dt)
    pad_emb = jax.device_put(empty, rep) if rep is not None else empty

  return txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb, pad_emb


# ---------------------------------------------------------------------------
# Stage 3 Output Helpers (WAV + ffmpeg MP4)
# ---------------------------------------------------------------------------


def save_stereo_wav(audio_2xN: np.ndarray, wav_path: Path, sample_rate: int = 44100) -> None:
  """Write stereo ``(2, N)`` float32 waveform to a 44.1 kHz WAV file."""
  wav = np.asarray(audio_2xN, dtype=np.float32)
  if wav.ndim == 2 and wav.shape[0] == 2:
    wav_nc = np.ascontiguousarray(wav.T, dtype=np.float32)
  elif wav.ndim == 2 and wav.shape[1] == 2:
    wav_nc = np.ascontiguousarray(wav, dtype=np.float32)
  else:
    raise ValueError(f"Expected stereo audio (2, N) or (N, 2), got {wav.shape}")
  wav_nc = np.clip(wav_nc, -1.0, 1.0)
  wav_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    from scipy.io import wavfile

    wavfile.write(str(wav_path), int(sample_rate), wav_nc)
  except ImportError:
    pcm16 = np.round(wav_nc * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as wf:
      wf.setnchannels(2)
      wf.setsampwidth(2)
      wf.setframerate(int(sample_rate))
      wf.writeframes(pcm16.tobytes())


def mux_video_and_audio_ffmpeg(
    frames_uint8: np.ndarray,
    mp4_path: Path,
    *,
    fps: float,
    wav_path: Path | None = None,
    output_width: int = 0,
    output_height: int = 0,
) -> bool:
  """Encode RGB24 uint8 ``(T, H, W, 3)`` frames (+ optional WAV) to H.264/AAC MP4 via ffmpeg."""
  ffmpeg_bin = shutil.which("ffmpeg")
  if ffmpeg_bin is None:
    try:
      import imageio_ffmpeg

      ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
      ffmpeg_bin = None
  if ffmpeg_bin is None:
    print("[magi2] WARNING: ffmpeg not found on PATH or imageio_ffmpeg; skipping MP4 muxing.", flush=True)
    return False

  t_num, h, w, c = frames_uint8.shape
  if c != 3 or frames_uint8.dtype != np.uint8:
    raise ValueError(f"Expected (T, H, W, 3) uint8 frames, got {frames_uint8.shape} {frames_uint8.dtype}")

  mp4_path.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
      ffmpeg_bin,
      "-y",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-vcodec",
      "rawvideo",
      "-s",
      f"{w}x{h}",
      "-pix_fmt",
      "rgb24",
      "-r",
      f"{fps:.6g}",
      "-i",
      "pipe:0",
  ]
  if wav_path is not None and wav_path.exists():
    cmd.extend(["-i", str(wav_path)])

  if output_width > 0 and output_height > 0 and (output_width != w or output_height != h):
    cmd.extend(["-vf", f"scale={output_width}:{output_height}:flags=bilinear"])

  cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
  if wav_path is not None and wav_path.exists():
    cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
  cmd.extend(["-movflags", "+faststart", str(mp4_path)])

  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
  assert proc.stdin is not None
  try:
    for i in range(t_num):
      proc.stdin.write(np.ascontiguousarray(frames_uint8[i]).tobytes())
    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0:
      err = stderr_bytes.decode("utf-8", errors="replace").strip()
      raise RuntimeError(f"ffmpeg exited with code {rc}: {err}")
  except Exception:
    proc.kill()
    proc.wait()
    raise
  return True


# ---------------------------------------------------------------------------
# Timing Summary Formatter
# ---------------------------------------------------------------------------


def print_timing_summary(timings: dict[str, Any], devices: Sequence[Any]) -> None:
  """Print structured wall-clock timing & HBM summary table."""
  _, peak_gib, limit_gib = _hbm_stats_gib(devices)
  lines = [
      "",
      "=" * 74,
      "  MAGI-2-preview Native JAX Pipeline Timing Summary",
      "=" * 74,
  ]

  def _row(name: str, val: str) -> None:
    lines.append(f"  {name:<36s} {val}")

  if "text_load_s" in timings:
    _row("Stage 0 Text Encoder load:", f"{timings['text_load_s']:8.2f} s")
  if "text_encode_s" in timings:
    src = " (cache)" if timings.get("text_from_cache") else ""
    _row(f"Stage 0 Text encode{src}:", f"{timings['text_encode_s']:8.2f} s")
  if timings.get("ref_img_encode_s", 0.0) > 0:
    _row("Stage 0 I2V ref image encode:", f"{timings['ref_img_encode_s']:8.2f} s")

  if "stage1_load_s" in timings:
    _row("Stage 1 Preview DiT load:", f"{timings['stage1_load_s']:8.2f} s")
  if "stage1_compile_s" in timings:
    _row("Stage 1 Preview compile/warmup:", f"{timings['stage1_compile_s']:8.2f} s")
  if "stage1_step_ms" in timings:
    _row(
        f"Stage 1 Preview per-step ({timings.get('stage1_steps', 0)} steps):",
        f"{timings['stage1_step_ms']:8.1f} ms/step",
    )
  if "stage1_denoise_s" in timings:
    _row("Stage 1 Preview denoise total:", f"{timings['stage1_denoise_s']:8.2f} s")
  if timings.get("stage1_run_times"):
    runs = timings["stage1_run_times"]
    if len(runs) > 1:
      steady_runs = runs[1:]
      _row(
          f"Stage 1 steady runs (2..{len(runs)}):",
          f"mean={np.mean(steady_runs):.2f} s  min={np.min(steady_runs):.2f} s",
      )

  if "stage2_load_s" in timings:
    _row("Stage 2 Refiner DiT load:", f"{timings['stage2_load_s']:8.2f} s")
  if "stage2_compile_s" in timings:
    _row("Stage 2 Refiner compile/step0:", f"{timings['stage2_compile_s']:8.2f} s")
  if "stage2_denoise_s" in timings:
    step_ms = timings.get("stage2_step_ms", 0.0)
    _row(
        f"Stage 2 Refiner denoise ({timings.get('stage2_steps', 0)} steps):",
        f"{timings['stage2_denoise_s']:8.2f} s  ({step_ms:.1f} ms/step)",
    )

  if "video_vae_load_s" in timings:
    _row("Stage 3 Video VAE (TurboVAED) load:", f"{timings['video_vae_load_s']:8.2f} s")
  if "video_vae_decode_s" in timings:
    _row("Stage 3 Video VAE decode:", f"{timings['video_vae_decode_s']:8.2f} s")
  if "audio_vae_load_s" in timings:
    _row("Stage 3 Audio VAE (Oobleck) load:", f"{timings['audio_vae_load_s']:8.2f} s")
  if "audio_vae_decode_s" in timings:
    _row("Stage 3 Audio VAE decode+resample:", f"{timings['audio_vae_decode_s']:8.2f} s")
  if "mux_s" in timings:
    _row("Stage 3 WAV + ffmpeg MP4 mux:", f"{timings['mux_s']:8.2f} s")

  lines.append("-" * 74)
  if "active_gen_s" in timings:
    _row("Active generation (excl. loads):", f"{timings['active_gen_s']:8.2f} s")
  _row("End-to-End wall time:", f"{timings['e2e_s']:8.2f} s")
  if limit_gib > 0:
    _row("Peak TPU HBM (chip 0):", f"{peak_gib:8.2f} / {limit_gib:.2f} GiB")
  lines.append("=" * 74)
  print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Main End-to-End Pipeline
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  t_e2e_start = time.perf_counter()

  # Set TPU XLA flags BEFORE importing JAX / libtpu.
  os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
  os.environ.setdefault("LIBTPU_INIT_ARGS", DEFAULT_LIBTPU_INIT_ARGS)

  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  if args.compile_cache:
    os.makedirs(args.compile_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", args.compile_cache)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

  # Resolve devices and 1D mesh ("tp") shared across TP (text), CP/EP (preview), CP/FSDP (refiner).
  all_devices = jax.devices()
  devices = all_devices[: args.chips] if args.chips > 0 else all_devices
  num_chips = len(devices)
  mesh = Mesh(np.array(devices), ("tp",))
  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

  # Load configs (optional JSON overrides) & resolve preset dimensions.
  preview_cfg = (
      Magi2PreviewConfig.from_json(args.config) if args.config else Magi2PreviewConfig()
  )
  refiner_cfg = (
      Magi2RefinerConfig.from_json(args.config) if args.config else Magi2RefinerConfig()
  )
  if args.steps is None:
    args.steps = preview_cfg.num_inference_steps
  if args.shift is None:
    args.shift = preview_cfg.shift
  if args.video_cfg is None:
    args.video_cfg = preview_cfg.video_cfg
  if args.audio_cfg is None:
    args.audio_cfg = preview_cfg.audio_cfg
  if args.refiner_steps is None:
    args.refiner_steps = refiner_cfg.num_inference_steps
  if args.refiner_shift is None:
    args.refiner_shift = refiner_cfg.shift
  if args.refiner_cfg is None:
    args.refiner_cfg = refiner_cfg.video_cfg
  if args.refiner_noise_value is None:
    args.refiner_noise_value = refiner_cfg.noise_value

  preset = RESOLUTION_PRESETS[args.resolution]
  preview_h = int(args.preview_height or preset["height"])
  preview_w = int(args.preview_width or preset["width"])
  # Match inference_engine.py:L230-236 & L411-418:
  video_frame_length = int(round(float(args.seconds) * 12.5 * 2.0))
  video_latent_t = (video_frame_length - 1) // 8 + 1
  audio_latent_t = int(round(float(args.seconds) * 25.0))
  preview_lat_h = preview_h // 16
  preview_lat_w = preview_w // 16
  preview_h = preview_lat_h * 16
  preview_w = preview_lat_w * 16

  use_refiner = bool(preset.get("use_refiner", False) or args.refiner_height or args.refiner_width)
  if args.no_refiner:
    use_refiner = False

  if use_refiner:
    refiner_h = int(args.refiner_height or preset.get("refiner_height", 1088))
    refiner_w = int(args.refiner_width or preset.get("refiner_width", 1920))
    refiner_lat_h = refiner_h // 16
    refiner_lat_w = refiner_w // 16
    refiner_h = refiner_lat_h * 16
    refiner_w = refiner_lat_w * 16
    refiner_lat_t = video_latent_t * 2 - 1
    output_fps = 25.0
  else:
    refiner_h = refiner_w = refiner_lat_h = refiner_lat_w = refiner_lat_t = 0
    output_fps = 12.5

  # Prompt & I2V image setup
  if args.prompt_file:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
  else:
    prompt = args.prompt

  image_dict = parse_image_paths(args.image)
  is_i2v = bool(image_dict)
  figure_ids: list[int] = []
  if is_i2v:
    sorted_fig_keys = sorted(image_dict.keys())
    figure_ids = [parse_figure_id(k) for k in sorted_fig_keys]
    prompt = ensure_figure_tokens(prompt, figure_ids)
  else:
    sorted_fig_keys = []

  tag = args.tag or (
      f"magi2_{'i2v_' if is_i2v else ''}{args.resolution}"
      + ("_norefiner" if (args.resolution == "1080p" and args.no_refiner) else "")
  )
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  paths = resolve_checkpoint_paths(args)

  print(
      f"[magi2] {devices[0].device_kind} x {num_chips}  dtype={args.dtype}  "
      f"res={args.resolution} ({preview_h}x{preview_w}"
      + (f" -> {refiner_h}x{refiner_w}" if use_refiner else "")
      + f")  steps={args.steps}"
      + (f"+{args.refiner_steps}" if use_refiner else "")
      + f"  moe={args.moe_mode}  fp8={args.fp8}  i2v={is_i2v}",
      flush=True,
  )
  print(f"[magi2] checkpoint root={paths['root']}  out={out_dir / tag}.mp4", flush=True)

  timings: dict[str, Any] = {}
  run_stage1 = not bool(args.load_stage1_latents)

  # =========================================================================
  # STAGE 0: Reference Image Encoding (I2V) & Qwen3.5-27B Text Encoding
  # =========================================================================
  ref_image_latents_np: list[np.ndarray] = []
  ref_image_hws: list[tuple[int, int]] = []
  if is_i2v and run_stage1:
    t0 = time.perf_counter()
    vae_dir = paths["vae"]
    if vae_dir is None or not vae_dir.exists():
      raise FileNotFoundError(
          f"I2V requires Wan2.2 VAE encoder checkpoint (--vae-dir or $MAGI2_CKPT_ROOT/vae), got: {vae_dir}"
      )
    from .jax_video_vae import load_wan22_vae_encoder

    print(f"[magi2] Stage 0a: Loading Wan2.2 VAE encoder from {vae_dir} ...", flush=True)
    wan_enc = load_wan22_vae_encoder(vae_dir, dtype=jnp.float32, strict=True, verbose=True)
    wan_gd, wan_st = nnx.split(wan_enc)
    wan_st = jax.tree.map(lambda x: jax.device_put(x, devices[0]), wan_st)

    @jax.jit
    def _jit_encode_ref(st, img):
      enc = nnx.merge(wan_gd, st)
      return enc.encode(img, input_layout="ncthw", output_layout="ncthw", normalize=True)

    for k in sorted_fig_keys:
      img_ncthw = preprocess_reference_image(
          image_dict[k],
          resize_type=args.ref_image_type,
          preview_height=preview_h,
          preview_width=preview_w,
      )
      img_tpu = jax.device_put(jnp.asarray(img_ncthw, dtype=jnp.float32), devices[0])
      lat = _jit_encode_ref(wan_st, img_tpu)
      lat_np = np.asarray(jax.block_until_ready(lat), dtype=np.float32)  # (1, 48, 1, h_k, w_k)
      ref_image_latents_np.append(lat_np)
      ref_image_hws.append((int(lat_np.shape[3]), int(lat_np.shape[4])))
      print(
          f"  [i2v] encoded {k} ({image_dict[k]}) -> latent {tuple(lat_np.shape)}",
          flush=True,
      )
      _free_device_arrays(img_tpu, lat)
    _free_device_arrays(wan_enc, wan_st)
    del wan_enc, wan_st, wan_gd, _jit_encode_ref
    gc.collect()
    timings["ref_img_encode_s"] = time.perf_counter() - t0
    _log_hbm("after Stage 0a (Wan2.2 encoder freed)", devices)

  # Text encoding (needed for Stage 1 and/or Stage 2 Refiner)
  need_text = run_stage1 or use_refiner
  cond_txt_np = np.zeros((1, 0, preview_cfg.text_in_channels), dtype=np.float32)
  uncond_txt_np = np.zeros((1, 0, preview_cfg.text_in_channels), dtype=np.float32)
  ref_special_np: np.ndarray | None = None

  if need_text:
    cache_path = Path(args.text_embeds_cache).expanduser() if args.text_embeds_cache else None
    if cache_path is not None and cache_path.exists():
      t0 = time.perf_counter()
      cached = np.load(cache_path, allow_pickle=False)
      cond_txt_np = np.asarray(cached["cond_txt"], dtype=np.float32)
      if args.use_negative_prompt:
        uncond_txt_np = np.asarray(cached["uncond_txt"], dtype=np.float32)
      else:
        uncond_txt_np = np.zeros((1, 0, preview_cfg.text_in_channels), dtype=np.float32)
      if is_i2v:
        if "ref_special_tokens" in cached:
          ref_special_np = np.asarray(cached["ref_special_tokens"], dtype=np.float32)
        else:
          # Pool <Figure N> spans from cached cond_txt using tokenizer only (no 27B weights!).
          txt_dir = paths["text_encoder"]
          if txt_dir is None or not txt_dir.exists():
            raise FileNotFoundError(
                "Cached text embeddings lack 'ref_special_tokens'; need --text-encoder-dir for tokenizer."
            )
          from .jax_text import load_tokenizer, pool_figure_tokens

          tok = load_tokenizer(txt_dir)
          ref_special_np = np.asarray(
              pool_figure_tokens(tok, prompt, figure_ids, cond_txt_np), dtype=np.float32
          )
      timings["text_encode_s"] = time.perf_counter() - t0
      timings["text_from_cache"] = True
      print(
          f"[magi2] Stage 0b: Loaded text embeddings from {cache_path}: "
          f"cond={tuple(cond_txt_np.shape)} uncond={tuple(uncond_txt_np.shape)} "
          f"({timings['text_encode_s']:.2f}s)",
          flush=True,
      )
    else:
      txt_dir = paths["text_encoder"]
      if txt_dir is None or not txt_dir.exists():
        raise FileNotFoundError(
            f"Text encoder dir not found (--text-encoder-dir or $MAGI2_CKPT_ROOT/text_encoder): {txt_dir}"
        )
      from .jax_text import Qwen35TextEncoder

      # Optional disk cache for static negative-prompt embeddings (skips redundant 512-tok compile+run).
      uncond_disk_path: Path | None = None
      uncond_cached_np: np.ndarray | None = None
      if args.use_negative_prompt and args.negative_prompt and args.uncond_cache:
        cc_root = (
            Path(args.compile_cache).expanduser()
            if args.compile_cache
            else Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/magi2")))
        )
        neg_key = hashlib.sha256(
            f"{txt_dir.resolve()}|skip=2|{args.dtype}|{args.negative_prompt}".encode("utf-8")
        ).hexdigest()[:16]
        uncond_disk_path = cc_root / "uncond_cache" / f"qwen35_uncond_{neg_key}.npy"
        if uncond_disk_path.exists():
          try:
            uncond_cached_np = np.asarray(np.load(uncond_disk_path, allow_pickle=False), dtype=np.float32)
          except Exception:
            uncond_cached_np = None

      print(f"[magi2] Stage 0b: Loading Qwen3.5-27B text encoder from {txt_dir} ...", flush=True)
      t_load = time.perf_counter()
      with _mesh_context(mesh):
        text_enc = Qwen35TextEncoder(
            txt_dir,
            mesh=mesh,
            axis_name="tp",
            dtype=dtype,
            skip_layer=2,
        )
      timings["text_load_s"] = time.perf_counter() - t_load
      _log_hbm("after Qwen3.5-27B load", devices)

      t_enc = time.perf_counter()
      with _mesh_context(mesh):
        cond_j = text_enc.encode(prompt)
        cond_j = jax.block_until_ready(cond_j)
        cond_txt_np = np.asarray(cond_j.astype(jnp.float32), dtype=np.float32)

        if args.use_negative_prompt and args.negative_prompt:
          if uncond_cached_np is not None:
            uncond_j = None
            uncond_txt_np = uncond_cached_np
            print(f"  [cache] Reused negative-prompt embeddings from {uncond_disk_path}", flush=True)
          else:
            uncond_j = text_enc.encode(args.negative_prompt)
            uncond_j = jax.block_until_ready(uncond_j)
            uncond_txt_np = np.asarray(uncond_j.astype(jnp.float32), dtype=np.float32)
            if uncond_disk_path is not None:
              uncond_disk_path.parent.mkdir(parents=True, exist_ok=True)
              np.save(uncond_disk_path, uncond_txt_np)
        else:
          uncond_j = None
          uncond_txt_np = np.zeros((1, 0, preview_cfg.text_in_channels), dtype=np.float32)

        if is_i2v:
          spec_j = text_enc.pool_figure_tokens(prompt, figure_ids, cond_j)
          ref_special_np = np.asarray(spec_j.astype(jnp.float32), dtype=np.float32)
        else:
          spec_j = None
      timings["text_encode_s"] = time.perf_counter() - t_enc
      timings["text_from_cache"] = False

      print(
          f"[magi2] Stage 0b: Text encoded in {timings['text_encode_s']:.2f}s "
          f"(load {timings['text_load_s']:.1f}s): "
          f"cond={tuple(cond_txt_np.shape)} uncond={tuple(uncond_txt_np.shape)}"
          + (f" ref_special={tuple(ref_special_np.shape)}" if ref_special_np is not None else ""),
          flush=True,
      )

      if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, Any] = {
            "cond_txt": cond_txt_np,
            "uncond_txt": uncond_txt_np,
        }
        if ref_special_np is not None:
          save_dict["ref_special_tokens"] = ref_special_np
        np.savez(cache_path, **save_dict)
        print(f"  [cache] Saved text embeddings to {cache_path}", flush=True)

      # Explicitly free Qwen3.5-27B (~50 GiB total / ~6.5 GiB/chip) before Stage 1!
      text_enc._compiled.clear()
      _free_device_arrays(cond_j, uncond_j, spec_j, text_enc)
      del cond_j, uncond_j, spec_j, text_enc
      gc.collect()
      _log_hbm("after Stage 0b (Qwen3.5-27B freed)", devices)

  # Draw initial noise on host in exact upstream order (video -> audio -> refiner).
  video_shape = (1, preview_cfg.video_in_channels, video_latent_t, preview_lat_h, preview_lat_w)
  audio_shape = (1, audio_latent_t, preview_cfg.audio_in_channels)
  refiner_shape = (
      (1, refiner_cfg.video_in_channels, refiner_lat_t, refiner_lat_h, refiner_lat_w)
      if use_refiner
      else None
  )
  init_video_np, init_audio_np, refiner_noise_np = generate_initial_noise(
      video_shape=video_shape,
      audio_shape=audio_shape,
      refiner_shape=refiner_shape,
      seed=args.seed,
      torch_noise=args.torch_noise,
      init_latents_dir=args.init_latents,
  )

  # =========================================================================
  # STAGE 1: Preview DiT (113.93B / 5.95B Active Multi-Head Latent MoE)
  # =========================================================================
  if run_stage1:
    preview_dir = paths["preview"]
    if preview_dir is None or not preview_dir.exists():
      raise FileNotFoundError(
          f"Stage 1 Preview checkpoint dir not found (--preview-dir or $MAGI2_CKPT_ROOT/preview): {preview_dir}"
      )
    preview_backend = getattr(args, "preview_backend", "jax")
    print(
        f"[magi2] Stage 1: Loading 114B Preview DiT from {preview_dir} "
        f"(backend={preview_backend}, moe_mode={args.moe_mode}, fp8={args.fp8}) ...",
        flush=True,
    )
    t_load1 = time.perf_counter()
    if preview_backend == "torchax":
      from .torchax_preview_dit import (
          TorchaxPreviewModel,
          load_torchax_preview_params_from_safetensors,
          make_torchax_preview_step_fn,
      )
      packed_tx, preview_cfg = load_torchax_preview_params_from_safetensors(
          preview_dir,
          config=preview_cfg,
          mesh=mesh,
          axis_name="tp",
          dtype=dtype,
          fp8_moe=args.fp8,
          verbose=True,
      )
      preview_model = TorchaxPreviewModel(
          packed_tx, preview_cfg, axis_name="tp", dtype=dtype, fp8_moe=args.fp8
      )
    else:
      from .jax_preview_dit import load_magi2_preview_from_safetensors
      preview_model = load_magi2_preview_from_safetensors(
          preview_dir,
          config=preview_cfg,
          mesh=mesh,
          axis_name="tp",
          dtype=dtype,
          fp8=args.fp8,
          moe_mode=args.moe_mode,
          capacity_factor=args.capacity_factor,
          verbose=print,
      )
    timings["stage1_load_s"] = time.perf_counter() - t_load1
    _log_hbm("after Stage 1 Preview load", devices)

    plan = build_preview_packing_plan(
        video_thw=(video_latent_t, preview_lat_h, preview_lat_w),
        audio_len=audio_latent_t,
        cond_txt_len=int(cond_txt_np.shape[1]),
        uncond_txt_len=int(uncond_txt_np.shape[1]),
        ref_img_hws=ref_image_hws,
        cp_size=num_chips,
        head_dim=preview_cfg.head_dim,
        rope_temperature=preview_cfg.rope_temperature,
    )
    print(
        f"[magi2] Stage 1 packed sequence: S_cond={plan['sample_lens'][0]} "
        f"S_uncond={plan['sample_lens'][1]} pad={plan['pad_size']} "
        f"S_padded={plan['s_padded']} (S_local={plan['s_padded'] // num_chips}/chip)",
        flush=True,
    )

    rep_sharding = NamedSharding(mesh, P())
    seq2d_sharding = NamedSharding(mesh, P("tp", None))
    seq1d_sharding = NamedSharding(mesh, P("tp",))

    with _mesh_context(mesh):
      (
          txt_cond_emb,
          ref_cond_emb,
          txt_uncond_emb,
          ref_uncond_emb,
          pad_emb,
      ) = _pre_embed_static_preview_tokens(
          preview_model,
          cond_txt_np=cond_txt_np,
          uncond_txt_np=uncond_txt_np,
          ref_special_np=ref_special_np,
          ref_latents_np=ref_image_latents_np,
          use_ref_for_uncond=args.use_ref_for_uncond,
          pad_size=plan["pad_size"],
          mesh=mesh,
      )
      rope_all = jax.device_put(jnp.asarray(plan["rope_all"], dtype=jnp.float32), seq2d_sharding)
      mod_all = jax.device_put(jnp.asarray(plan["mod_all"], dtype=jnp.int32), seq1d_sharding)
      time_kind_all = jax.device_put(
          jnp.asarray(plan["time_kind_all"], dtype=jnp.int32), seq1d_sharding
      )

      if preview_backend == "torchax":
        preview_step_fn, preview_state = make_torchax_preview_step_fn(
            preview_model,
            video_thw=(video_latent_t, preview_lat_h, preview_lat_w),
            n_aud=audio_latent_t,
            sample_lens=plan["sample_lens"],
            mesh=mesh,
            use_splash=(False if args.dense_attention else None),
            moe_mode=args.moe_mode,
            capacity_factor=args.capacity_factor,
        )
      else:
        preview_step_fn, preview_state = _make_preview_step_fn(
            preview_model,
            video_thw=(video_latent_t, preview_lat_h, preview_lat_w),
            n_aud=audio_latent_t,
            sample_lens=plan["sample_lens"],
            mesh=mesh,
            use_splash=(False if args.dense_attention else None),
            moe_mode=args.moe_mode,
            capacity_factor=args.capacity_factor,
        )

      video_cfg_j = jax.device_put(jnp.float32(args.video_cfg), rep_sharding)
      audio_cfg_j = jax.device_put(jnp.float32(args.audio_cfg), rep_sharding)

      # Optional explicit 1-step warmup to separate cold XLA compilation from steady-state loop.
      if args.warmup:
        t_warm = time.perf_counter()
        w_v = jax.device_put(jnp.asarray(init_video_np, dtype=jnp.float32), rep_sharding)
        w_a = jax.device_put(jnp.asarray(init_audio_np, dtype=jnp.float32), rep_sharding)
        w_t = jax.device_put(jnp.float32(0.999), rep_sharding)
        w_ov, w_oa = preview_step_fn(
            preview_state,
            w_v,
            w_a,
            w_t,
            video_cfg_j,
            audio_cfg_j,
            txt_cond_emb,
            ref_cond_emb,
            txt_uncond_emb,
            ref_uncond_emb,
            pad_emb,
            rope_all,
            mod_all,
            time_kind_all,
        )
        jax.block_until_ready((w_ov, w_oa))
        timings["stage1_compile_s"] = time.perf_counter() - t_warm
        print(f"  [stage1] warmup/compile done in {timings['stage1_compile_s']:.2f}s", flush=True)
        _free_device_arrays(w_v, w_a, w_t, w_ov, w_oa)
        del w_v, w_a, w_t, w_ov, w_oa

      # Denoising loop (supports --runs N keeping Stage 1 Preview resident).
      num_runs = max(1, int(args.runs))
      run_times: list[float] = []
      steady_step_ms_last = 0.0
      stage1_video_np: np.ndarray | None = None
      stage1_audio_np: np.ndarray | None = None

      for run_idx in range(num_runs):
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(args.steps, shift=args.shift)
        audio_scheduler.set_timesteps(args.steps, shift=args.shift)

        latent_v = jax.device_put(jnp.asarray(init_video_np, dtype=jnp.float32), rep_sharding)
        latent_a = jax.device_put(jnp.asarray(init_audio_np, dtype=jnp.float32), rep_sharding)

        prof_dir = args.profile_dir.strip() if run_idx == num_runs - 1 else ""
        prof_step = min(max(0, int(args.profile_step)), max(0, len(video_scheduler.timesteps) - 1))

        t_run_start = time.perf_counter()
        t_after_step0 = t_run_start
        per_step_ms: list[float] = []

        for step_idx, t_int in enumerate(video_scheduler.timesteps):
          # Exact upstream sampler.py:L162: t_int is int64(sigma * 1000), raw_t = t_int / 1000.0
          raw_t_val = float(np.int64(t_int)) / 1000.0
          raw_t_j = jax.device_put(jnp.float32(raw_t_val), rep_sharding)

          if prof_dir and step_idx == prof_step:
            jax.block_until_ready((latent_v, latent_a))
            os.makedirs(prof_dir, exist_ok=True)
            print(f"  [profiler] starting trace at step {step_idx} -> {prof_dir}", flush=True)
            jax.profiler.start_trace(prof_dir)

          t_s = time.perf_counter()
          v_cfg_video, v_cfg_audio = preview_step_fn(
              preview_state,
              latent_v,
              latent_a,
              raw_t_j,
              video_cfg_j,
              audio_cfg_j,
              txt_cond_emb,
              ref_cond_emb,
              txt_uncond_emb,
              ref_uncond_emb,
              pad_emb,
              rope_all,
              mod_all,
              time_kind_all,
          )
          latent_v = video_scheduler.step(v_cfg_video, t_int, latent_v, return_dict=False)[0]
          latent_a = audio_scheduler.step(v_cfg_audio, t_int, latent_a, return_dict=False)[0]

          if prof_dir and step_idx == prof_step:
            jax.block_until_ready((latent_v, latent_a))
            jax.profiler.stop_trace()
            print(f"  [profiler] stopped trace after step {step_idx}", flush=True)

          if step_idx == 0:
            jax.block_until_ready((latent_v, latent_a))
            t_after_step0 = time.perf_counter()
            s0_s = t_after_step0 - t_s
            if run_idx == 0 and not args.warmup:
              timings["stage1_compile_s"] = s0_s
            per_step_ms.append(s0_s * 1000.0)
            print(
                f"  [stage1{f' run {run_idx+1}/{num_runs}' if num_runs > 1 else ''}] "
                f"step 1/{len(video_scheduler.timesteps)} t={int(t_int):3d} "
                f"({s0_s*1000.0:.1f} ms)",
                flush=True,
            )
          elif args.sync_every_step or step_idx == len(video_scheduler.timesteps) - 1:
            jax.block_until_ready((latent_v, latent_a))
            dt_ms = (time.perf_counter() - t_s) * 1000.0
            per_step_ms.append(dt_ms)
            if args.sync_every_step or (step_idx + 1) % 10 == 0 or step_idx == len(video_scheduler.timesteps) - 1:
              print(
                  f"  [stage1{f' run {run_idx+1}/{num_runs}' if num_runs > 1 else ''}] "
                  f"step {step_idx+1}/{len(video_scheduler.timesteps)} t={int(t_int):3d} "
                  f"({dt_ms:.1f} ms)",
                  flush=True,
              )
          elif (step_idx + 1) % 20 == 0:
            print(
                f"  [stage1{f' run {run_idx+1}/{num_runs}' if num_runs > 1 else ''}] "
                f"dispatched step {step_idx+1}/{len(video_scheduler.timesteps)} t={int(t_int):3d}",
                flush=True,
            )

        jax.block_until_ready((latent_v, latent_a))
        t_run_end = time.perf_counter()
        run_wall = t_run_end - t_run_start
        run_times.append(run_wall)
        n_steps = len(video_scheduler.timesteps)
        if args.warmup or run_idx > 0 or n_steps <= 1:
          steady_step_ms_last = (run_wall / max(1, n_steps)) * 1000.0
        else:
          steady_step_ms_last = ((t_run_end - t_after_step0) / (n_steps - 1)) * 1000.0
        print(
            f"[magi2] Stage 1 run {run_idx+1}/{num_runs}: {run_wall:.2f}s "
            f"({steady_step_ms_last:.1f} ms/step steady)",
            flush=True,
        )

        if run_idx == num_runs - 1:
          stage1_video_np = np.asarray(latent_v, dtype=np.float32)
          stage1_audio_np = np.asarray(latent_a, dtype=np.float32)
        _free_device_arrays(latent_v, latent_a)
        del latent_v, latent_a

    assert stage1_video_np is not None and stage1_audio_np is not None
    timings["stage1_denoise_s"] = run_times[-1]
    timings["stage1_step_ms"] = steady_step_ms_last
    timings["stage1_steps"] = int(args.steps)
    timings["stage1_run_times"] = run_times

    if args.save_latents:
      lat_path = out_dir / f"{tag}_stage1_latents.npz"
      np.savez(
          lat_path,
          video_latent=stage1_video_np,
          audio_latent=stage1_audio_np,
          resolution=np.array(args.resolution),
          seed=np.array(args.seed, dtype=np.int64),
      )
      print(f"  [save] Stage 1 latents saved to {lat_path}", flush=True)

    # Explicitly free 114B Preview DiT (~26 GiB/chip BF16 or ~14 GiB/chip FP8) before Stage 2/3!
    _free_device_arrays(
        preview_model,
        preview_state,
        txt_cond_emb,
        ref_cond_emb,
        txt_uncond_emb,
        ref_uncond_emb,
        pad_emb,
        rope_all,
        mod_all,
        time_kind_all,
        video_cfg_j,
        audio_cfg_j,
    )
    del (
        preview_model,
        preview_state,
        preview_step_fn,
        txt_cond_emb,
        ref_cond_emb,
        txt_uncond_emb,
        ref_uncond_emb,
        pad_emb,
        rope_all,
        mod_all,
        time_kind_all,
        video_cfg_j,
        audio_cfg_j,
    )
    gc.collect()
    _log_hbm("after Stage 1 Preview freed", devices)
  else:
    # Load pre-saved Stage 1 latents for independent Stage 2 / Stage 3 testing.
    lat_src = Path(args.load_stage1_latents).expanduser()
    if lat_src.suffix == ".npy":
      stage1_video_np = np.load(lat_src).astype(np.float32)
      stage1_audio_np = np.zeros(audio_shape, dtype=np.float32)
    else:
      loaded = np.load(lat_src, allow_pickle=False)
      stage1_video_np = np.asarray(loaded["video_latent"], dtype=np.float32)
      stage1_audio_np = (
          np.asarray(loaded["audio_latent"], dtype=np.float32)
          if "audio_latent" in loaded
          else np.zeros(audio_shape, dtype=np.float32)
      )
    if stage1_video_np.ndim == 4:
      stage1_video_np = stage1_video_np[None, ...]
    if stage1_audio_np.ndim == 2:
      stage1_audio_np = stage1_audio_np[None, ...]
    print(
        f"[magi2] Loaded Stage 1 latents from {lat_src}: "
        f"video={tuple(stage1_video_np.shape)} audio={tuple(stage1_audio_np.shape)}",
        flush=True,
    )
    if use_refiner:
      refiner_lat_t = stage1_video_np.shape[2] * 2 - 1
      if refiner_noise_np is None or refiner_noise_np.shape[2:] != (
          refiner_lat_t,
          refiner_lat_h,
          refiner_lat_w,
      ):
        _, _, refiner_noise_np = generate_initial_noise(
            video_shape=stage1_video_np.shape,
            audio_shape=stage1_audio_np.shape,
            refiner_shape=(
                1,
                refiner_cfg.video_in_channels,
                refiner_lat_t,
                refiner_lat_h,
                refiner_lat_w,
            ),
            seed=args.seed,
            torch_noise=args.torch_noise,
        )

  final_video_latent_np = stage1_video_np
  final_audio_latent_np = stage1_audio_np

  # =========================================================================
  # STAGE 2: 1080p Refiner DiT (6.72B Dense GQA, 30 Layers, 5 Steps)
  # =========================================================================
  if use_refiner:
    refiner_dir = paths["refiner"]
    if refiner_dir is None or not refiner_dir.exists():
      raise FileNotFoundError(
          f"Stage 2 Refiner checkpoint dir not found (--refiner-dir or $MAGI2_CKPT_ROOT/refiner): {refiner_dir}"
      )
    from .jax_refiner_dit import (
        build_refiner_attention_layout,
        load_refiner,
        make_jitted_forward as make_refiner_jitted_forward,
        pack_refiner_inputs_np,
        unpack_refiner_video_jax,
    )

    target_thw = (refiner_lat_t, refiner_lat_h, refiner_lat_w)
    print(
        f"[magi2] Stage 2: Trilinear upsample {tuple(stage1_video_np.shape[2:])} -> {target_thw} "
        f"+ VP noise augment (noise_value={args.refiner_noise_value})",
        flush=True,
    )
    z_up_np = trilinear_upsample_latents(stage1_video_np, target_thw)
    if args.refiner_noise_value != 0:
      assert refiner_noise_np is not None
      z_ref_init_np = refiner_noise_augment(
          z_up_np, refiner_noise_np, noise_value=args.refiner_noise_value
      )
    else:
      z_ref_init_np = z_up_np

    print(
        f"[magi2] Stage 2: Loading 6.72B Refiner DiT from {refiner_dir} "
        f"(sharding={args.refiner_sharding}, CP={num_chips}) ...",
        flush=True,
    )
    t_load2 = time.perf_counter()
    refiner_model = load_refiner(
        refiner_dir,
        config=refiner_cfg,
        mesh=mesh,
        dtype=dtype,
        sharding=args.refiner_sharding,
        axis_name="tp",
        verbose=True,
    )
    timings["stage2_load_s"] = time.perf_counter() - t_load2
    _log_hbm("after Stage 2 Refiner load", devices)

    # Build shared-bucket layouts so cond and uncond reuse ONE compiled XLA graph!
    block_vol = refiner_cfg.block_t_size * refiner_cfg.block_h_size * refiner_cfg.block_w_size
    l_cond = int(cond_txt_np.shape[1])
    l_uncond = int(uncond_txt_np.shape[1])
    max_txt = max(l_cond, l_uncond)
    tail_bucket = (
        math.ceil(max_txt / block_vol) * block_vol if max_txt > 0 else 0
    )
    need_flat_tables = (args.refiner_attn_impl or "").lower() == "sparse_splash"

    layout_cond = build_refiner_attention_layout(
        target_thw,
        n_text=l_cond,
        n_audio=0,
        config=refiner_cfg,
        tail_bucket_len=tail_bucket if tail_bucket > 0 else None,
        cp_size=num_chips,
        build_flat_tables=need_flat_tables,
    )
    layout_uncond = build_refiner_attention_layout(
        target_thw,
        n_text=l_uncond,
        n_audio=0,
        config=refiner_cfg,
        tail_bucket_len=tail_bucket if tail_bucket > 0 else None,
        cp_size=num_chips,
        build_flat_tables=need_flat_tables,
    )
    print(
        f"[magi2] Stage 2 layout: grid={layout_cond.block_grid_thw} "
        f"n_vid_pad={layout_cond.n_vid_pad} n_tail_pad={layout_cond.n_tail_pad} "
        f"total_seq={layout_cond.total_seq_len} (cond_txt={l_cond}, uncond_txt={l_uncond})",
        flush=True,
    )

    _, _, txt_cond_packed_np = pack_refiner_inputs_np(z_ref_init_np, cond_txt_np, layout_cond)
    _, _, txt_uncond_packed_np = pack_refiner_inputs_np(z_ref_init_np, uncond_txt_np, layout_uncond)

    rep_sharding = NamedSharding(mesh, P())
    seq2d_sharding = NamedSharding(mesh, P("tp", None))
    vid_sharding = (
        seq2d_sharding if (layout_cond.n_vid_pad % num_chips == 0) else rep_sharding
    )

    with _mesh_context(mesh):
      jitted_refiner, _, refiner_state = make_refiner_jitted_forward(
          refiner_model,
          layout_cond,
          mesh=mesh,
          cp_axis="tp",
          parallel_mode="cp",
          compute_dtype=dtype,
          attn_impl=args.refiner_attn_impl or None,
      )
      # masked_pallas & ref consume dynamic kv_blen, so cond & uncond reuse ONE
      # compiled XLA graph; sparse_splash bakes static kv_table lists into layout.
      if need_flat_tables and not np.array_equal(
          layout_cond.kv_block_valid_lens > 0, layout_uncond.kv_block_valid_lens > 0
      ):
        jitted_refiner_uncond, _, _ = make_refiner_jitted_forward(
            refiner_model,
            layout_uncond,
            mesh=mesh,
            cp_axis="tp",
            parallel_mode="cp",
            compute_dtype=dtype,
            attn_impl=args.refiner_attn_impl or None,
        )
      else:
        jitted_refiner_uncond = jitted_refiner

      vid_src_idx_j = jax.device_put(
          jnp.asarray(layout_cond.vid_src_idx, dtype=jnp.int32), rep_sharding
      )
      txt_cond_j = jax.device_put(jnp.asarray(txt_cond_packed_np, dtype=jnp.float32), rep_sharding)
      txt_uncond_j = jax.device_put(
          jnp.asarray(txt_uncond_packed_np, dtype=jnp.float32), rep_sharding
      )
      rope_cond_j = jax.device_put(
          jnp.asarray(layout_cond.rope_96, dtype=jnp.float32), seq2d_sharding
      )
      rope_uncond_j = jax.device_put(
          jnp.asarray(layout_uncond.rope_96, dtype=jnp.float32), seq2d_sharding
      )
      kv_blen_cond_j = jax.device_put(
          jnp.asarray(layout_cond.kv_block_valid_lens, dtype=jnp.int32), rep_sharding
      )
      kv_blen_uncond_j = jax.device_put(
          jnp.asarray(layout_uncond.kv_block_valid_lens, dtype=jnp.int32), rep_sharding
      )

      @jax.jit
      def _pack_refiner_video(z_ncthw, src_idx):
        c_v = z_ncthw.shape[1]
        v_raster = jnp.transpose(z_ncthw[0], (1, 2, 3, 0)).reshape(-1, c_v)
        safe_src = jnp.maximum(src_idx, 0)
        packed = jnp.where((src_idx >= 0)[:, None], v_raster[safe_src], jnp.float32(0.0))
        return jax.lax.with_sharding_constraint(packed, vid_sharding)

      @jax.jit
      def _unpack_refiner_video(v_pad):
        out = unpack_refiner_video_jax(v_pad, layout_cond)[None, ...]
        return jax.lax.with_sharding_constraint(out, rep_sharding)

      refiner_scheduler = FlowUniPCMultistepScheduler()
      refiner_scheduler.set_timesteps(args.refiner_steps, shift=args.refiner_shift)

      latent_ref = jax.device_put(jnp.asarray(z_ref_init_np, dtype=jnp.float32), rep_sharding)
      ref_cfg = float(args.refiner_cfg)
      use_uncond_only = ref_cfg == 0.0
      use_cond_only = ref_cfg == 1.0

      t_ref_start = time.perf_counter()
      t_ref_after_s0 = t_ref_start
      for step_idx, t_int in enumerate(refiner_scheduler.timesteps):
        t_s = time.perf_counter()
        v_packed = _pack_refiner_video(latent_ref, vid_src_idx_j)
        if use_uncond_only:
          v_pred_pad, _ = jitted_refiner_uncond(
              refiner_state, v_packed, txt_uncond_j, rope_uncond_j, kv_blen_uncond_j, None
          )
        elif use_cond_only:
          v_pred_pad, _ = jitted_refiner(
              refiner_state, v_packed, txt_cond_j, rope_cond_j, kv_blen_cond_j, None
          )
        else:
          cond_pad, _ = jitted_refiner(
              refiner_state, v_packed, txt_cond_j, rope_cond_j, kv_blen_cond_j, None
          )
          uncond_pad, _ = jitted_refiner_uncond(
              refiner_state, v_packed, txt_uncond_j, rope_uncond_j, kv_blen_uncond_j, None
          )
          v_pred_pad = uncond_pad + jnp.float32(ref_cfg) * (cond_pad - uncond_pad)

        v_cfg_video = _unpack_refiner_video(v_pred_pad)
        latent_ref = refiner_scheduler.step(v_cfg_video, t_int, latent_ref, return_dict=False)[0]
        jax.block_until_ready(latent_ref)
        dt_s = time.perf_counter() - t_s
        if step_idx == 0:
          t_ref_after_s0 = time.perf_counter()
          timings["stage2_compile_s"] = dt_s
        print(
            f"  [stage2 refiner] step {step_idx+1}/{len(refiner_scheduler.timesteps)} "
            f"t={int(t_int):3d} ({dt_s:.2f}s)",
            flush=True,
        )

      t_ref_end = time.perf_counter()
      timings["stage2_denoise_s"] = t_ref_end - t_ref_start
      timings["stage2_steps"] = int(args.refiner_steps)
      timings["stage2_step_ms"] = (
          ((t_ref_end - t_ref_after_s0) / max(1, len(refiner_scheduler.timesteps) - 1)) * 1000.0
          if len(refiner_scheduler.timesteps) > 1
          else (t_ref_end - t_ref_start) * 1000.0
      )
      final_video_latent_np = np.asarray(latent_ref, dtype=np.float32)

    if args.save_latents:
      ref_lat_path = out_dir / f"{tag}_refiner_latents.npz"
      np.savez(
          ref_lat_path,
          video_latent=final_video_latent_np,
          audio_latent=final_audio_latent_np,
          resolution=np.array(args.resolution),
          seed=np.array(args.seed, dtype=np.int64),
      )
      print(f"  [save] Stage 2 Refiner latents saved to {ref_lat_path}", flush=True)

    # Explicitly free 6.72B Refiner DiT before Stage 3 VAEs!
    _free_device_arrays(
        refiner_model,
        refiner_state,
        latent_ref,
        vid_src_idx_j,
        txt_cond_j,
        txt_uncond_j,
        rope_cond_j,
        rope_uncond_j,
        kv_blen_cond_j,
        kv_blen_uncond_j,
    )
    del (
        refiner_model,
        refiner_state,
        jitted_refiner,
        jitted_refiner_uncond,
        latent_ref,
        vid_src_idx_j,
        txt_cond_j,
        txt_uncond_j,
        rope_cond_j,
        rope_uncond_j,
        kv_blen_cond_j,
        kv_blen_uncond_j,
    )
    gc.collect()
    _log_hbm("after Stage 2 Refiner freed", devices)

  # =========================================================================
  # STAGE 3: Video VAE (TurboVAED) & Stereo Audio VAE (AutoencoderOobleck)
  # =========================================================================
  if not args.skip_decode:
    # 3a. Video decode (TurboVAED, bf16, single-chip JIT sliding-window decode)
    turbo_cfg_path = paths["turbo_vae_config"]
    turbo_ckpt_path = paths["turbo_vae_ckpt"]
    if turbo_ckpt_path is None or not turbo_ckpt_path.exists():
      raise FileNotFoundError(
          f"TurboVAED checkpoint not found (--turbo-vae-dir / --turbo-vae-ckpt): {turbo_ckpt_path}"
      )
    from .jax_video_vae import load_turbo_vaed

    print(
        f"[magi2] Stage 3a: Loading TurboVAED from {turbo_ckpt_path} (bf16) ...",
        flush=True,
    )
    t_vload = time.perf_counter()
    turbo_vae = load_turbo_vaed(
        turbo_cfg_path if (turbo_cfg_path is not None and turbo_cfg_path.exists()) else Magi2TurboVAEDConfig(),
        turbo_ckpt_path,
        dtype=jnp.bfloat16,
        strict=True,
        verbose=True,
    )
    timings["video_vae_load_s"] = time.perf_counter() - t_vload

    t_vdec = time.perf_counter()
    if args.decode_chunk_shard and len(devices) > 1:
      with _mesh_context(mesh):
        frames_uint8 = turbo_vae.decode_sharded(
            final_video_latent_np,
            mesh=mesh,
            devices=devices,
            axis_name="tp",
            input_layout="ncthw",
            denormalize=True,
            to_uint8=True,
        )
      timings["video_vae_decode_s"] = time.perf_counter() - t_vdec
      print(
          f"[magi2] Stage 3a: Decoded video {tuple(frames_uint8.shape)} @ {output_fps:g} fps "
          f"in {timings['video_vae_decode_s']:.2f}s (chunk-sharded x{len(devices)})",
          flush=True,
      )
      _free_device_arrays(turbo_vae)
      del turbo_vae
      gc.collect()
    else:
      dec_graphdef, dec_state = nnx.split(turbo_vae.decoder)

      @functools.partial(jax.jit, static_argnames=("is_first_chunk",))
      def _jit_decode_chunk_uint8(st, z_chunk, is_first_chunk: bool):
        dec = nnx.merge(dec_graphdef, st)
        rgb = dec(z_chunk, is_first_chunk=is_first_chunk)
        # Exact match to inference_engine.py:L704-707:
        # float -> mul(0.5).add(0.5).clamp(0, 1).mul(255).astype(uint8) on TPU
        # (cuts PCIe D2H and host RAM 4x at 1080p!).
        rgb_u8 = (
            jnp.clip(rgb.astype(jnp.float32) * jnp.float32(0.5) + jnp.float32(0.5), 0.0, 1.0)
            * jnp.float32(255.0)
        ).astype(jnp.uint8)
        return rgb_u8

      z_vid_tpu = jax.device_put(jnp.asarray(final_video_latent_np, dtype=jnp.float32), devices[0])
      frames_bthwc = turbo_vae.decode(
          z_vid_tpu,
          input_layout="ncthw",
          output_layout="ndhwc",
          denormalize=True,
          output_offload=True,
          chunk_fn=lambda z_c, is_first_chunk=True: _jit_decode_chunk_uint8(
              dec_state, z_c, is_first_chunk=is_first_chunk
          ),
      )
      frames_uint8 = np.asarray(frames_bthwc[0], dtype=np.uint8)  # (T_frames, H, W, 3)
      timings["video_vae_decode_s"] = time.perf_counter() - t_vdec
      print(
          f"[magi2] Stage 3a: Decoded video {tuple(frames_uint8.shape)} @ {output_fps:g} fps "
          f"in {timings['video_vae_decode_s']:.2f}s",
          flush=True,
      )

      _free_device_arrays(turbo_vae, dec_state, z_vid_tpu)
      del turbo_vae, dec_state, dec_graphdef, z_vid_tpu, frames_bthwc
      gc.collect()

    # 3b. Stereo audio decode (AutoencoderOobleck, fp32 + 441/512 sinc resample)
    wav_path: Path | None = None
    if not args.no_audio and final_audio_latent_np.shape[1] > 0:
      audio_dir = paths["audio_vae"]
      if audio_dir is None or not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio VAE not found (--audio-vae-dir or $MAGI2_CKPT_ROOT/stable-audio-open-1.0): {audio_dir}"
        )
      from .jax_audio_vae import (
          adaptive_volume_scaling,
          load_audio_vae,
          resample_audio_25hz_to_44100,
      )

      print(f"[magi2] Stage 3b: Loading AutoencoderOobleck from {audio_dir} (fp32) ...", flush=True)
      t_aload = time.perf_counter()
      audio_vae = load_audio_vae(audio_dir, dtype=jnp.float32, strict=True, verbose=True)
      timings["audio_vae_load_s"] = time.perf_counter() - t_aload

      ae_graphdef, ae_state = nnx.split(audio_vae)

      @jax.jit
      def _jit_decode_audio(st, lat_btc):
        ae = nnx.merge(ae_graphdef, st)
        return ae.decode(lat_btc, input_layout="btc", output_layout="bct")

      t_adec = time.perf_counter()
      z_aud_tpu = jax.device_put(jnp.asarray(final_audio_latent_np, dtype=jnp.float32), devices[0])
      wav_raw = np.asarray(jax.block_until_ready(_jit_decode_audio(ae_state, z_aud_tpu)), dtype=np.float32)
      audio_44k = resample_audio_25hz_to_44100(wav_raw[0], axis=-1)  # (2, 441_000)
      if args.adaptive_volume_scaling:
        audio_44k = adaptive_volume_scaling(audio_44k)
      timings["audio_vae_decode_s"] = time.perf_counter() - t_adec

      rms = float(np.sqrt(np.mean(audio_44k**2)))
      dbfs = 20.0 * math.log10(rms + 1e-12)
      peak = float(np.max(np.abs(audio_44k)))
      print(
          f"[magi2] Stage 3b: Decoded stereo audio {tuple(audio_44k.shape)} @ 44.1 kHz "
          f"(RMS {dbfs:.1f} dBFS, peak {peak:.3f}) in {timings['audio_vae_decode_s']:.2f}s",
          flush=True,
      )

      wav_path = out_dir / f"{tag}_audio.wav"
      save_stereo_wav(audio_44k, wav_path, sample_rate=audio_vae.sample_rate)
      print(f"  [save] Stereo WAV saved to {wav_path}", flush=True)

      _free_device_arrays(audio_vae, ae_state, z_aud_tpu)
      del audio_vae, ae_state, ae_graphdef, z_aud_tpu, wav_raw
      gc.collect()

    # 3c. Mux video + stereo audio to <out>/<tag>.mp4
    t_mux = time.perf_counter()
    if args.save_frames_npy:
      frames_npy = out_dir / f"{tag}_frames.npy"
      np.save(frames_npy, frames_uint8)
      print(f"  [save] Raw frames saved to {frames_npy}", flush=True)

    mp4_path = out_dir / f"{tag}.mp4"
    muxed = mux_video_and_audio_ffmpeg(
        frames_uint8,
        mp4_path,
        fps=output_fps,
        wav_path=wav_path,
        output_width=args.output_width,
        output_height=args.output_height,
    )
    timings["mux_s"] = time.perf_counter() - t_mux
    if muxed:
      size_mb = mp4_path.stat().st_size / 1e6
      print(
          f"[magi2] Wrote {mp4_path} ({size_mb:.2f} MB, {frames_uint8.shape[0]} frames @ {output_fps:g} fps)",
          flush=True,
      )
    elif not args.save_frames_npy:
      fallback_npy = out_dir / f"{tag}_frames.npy"
      np.save(fallback_npy, frames_uint8)
      print(f"[magi2] Saved fallback raw frames to {fallback_npy}", flush=True)

  timings["active_gen_s"] = (
      timings.get("text_encode_s", 0.0)
      + timings.get("ref_img_encode_s", 0.0)
      + timings.get("stage1_denoise_s", 0.0)
      + timings.get("stage2_denoise_s", 0.0)
      + timings.get("video_vae_decode_s", 0.0)
      + timings.get("audio_vae_decode_s", 0.0)
  )
  timings["e2e_s"] = time.perf_counter() - t_e2e_start
  print_timing_summary(timings, devices)


if __name__ == "__main__":
  main()
