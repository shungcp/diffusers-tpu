"""Native JAX Resident Server for Sand.ai MAGI-2-preview on TPU v7x-8.

Keeps all MAGI-2 stages resident in TPU HBM simultaneously (~56.5 GiB / 94.74 GiB
per v7x logical device in BF16, or ~44 GiB in FP8), eliminating the 30-320s cold
model-load overhead between requests:
  - Stage 0a: Wan2_2_VAEEncoder (lazy-loaded on chip 0, @jax.jit ~23 ms warm)
  - Stage 0b: Qwen3.5-27B text encoder (TP=8, ~6.1 GiB/dev + disk --uncond-cache)
  - Stage 1:  113.93B Preview DiT (EP/CP=8, ~35.3 GiB BF16 / ~22.6 GiB FP8) with
              zero-extra-HBM ``bind_sample_lens`` caching across requests
  - Stage 2:  6.72B 1080p Refiner (optional, DP=2 x CP=4 GQA+W-coalesced, ~12.6 GiB)
  - Stage 3a: TurboVAED 8-chip chunk-parallel decoder (~0.87 GiB, 0.72s 540p / 3.88s 1080p)
  - Stage 3b: AutoencoderOobleck stereo audio decoder (~0.3 GiB, ~0.98s)

Usage:
  python -m diffusers_tpu.models.magi2.jax_server --requests reqs.jsonl
  python -m diffusers_tpu.models.magi2.jax_server --requests reqs.jsonl --lossy --fp8
  bash scripts/run_magi2_jax_server.sh --requests reqs.jsonl --lossy
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_NEGATIVE_PROMPT,
    Magi2PreviewConfig,
    Magi2RefinerConfig,
    Magi2TurboVAEDConfig,
    RESOLUTION_PRESETS,
)
from .jax_pipeline import (
    _free_device_arrays,
    _log_hbm,
    _make_preview_step_fn,
    _mesh_context,
    _pre_embed_static_preview_tokens,
    build_preview_packing_plan,
    ensure_figure_tokens,
    generate_initial_noise,
    mux_video_and_audio_ffmpeg,
    preprocess_reference_image,
    resolve_checkpoint_paths,
    save_stereo_wav,
)
from .prompt_enhancer import enhance_prompt_to_pe_json_str
from .rope_and_ops import (
    FlowUniPCMultistepScheduler,
    refiner_noise_augment,
    trilinear_upsample_latents,
)
from .jax_text import json_to_compact_markdown


class Magi2ResidentServer:
  """Multi-request TPU-resident Native JAX server for MAGI-2-preview."""

  def __init__(
      self,
      *,
      model_dir: str,
      resolution: str = "540p",
      dtype_str: str = "bfloat16",
      fp8: bool = False,
      moe_mode: str = "auto",
      capacity_factor: float | None = None,
      enable_refiner: bool = False,
      refiner_cfg_parallel: bool = True,
      refiner_qk_fp8: bool = False,
      compile_cache: str = "",
      uncond_cache: bool = True,
      decode_chunk_shard: bool = True,
      chips: int = 0,
  ):
    if compile_cache:
      cc_path = Path(compile_cache).expanduser()
      cc_path.mkdir(parents=True, exist_ok=True)
      os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cc_path)
      os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
      os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    self.compile_cache = compile_cache
    self.uncond_cache = uncond_cache
    self.decode_chunk_shard = decode_chunk_shard
    self.default_resolution = resolution
    self.dtype_str = dtype_str
    self.dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    self.fp8 = fp8
    self.moe_mode = moe_mode
    self.capacity_factor = capacity_factor
    self.refiner_cfg_parallel = refiner_cfg_parallel
    self.refiner_qk_fp8 = refiner_qk_fp8

    all_devices = jax.devices()
    self.devices = all_devices[:chips] if chips > 0 else all_devices
    self.num_chips = len(self.devices)
    self.mesh = Mesh(np.array(self.devices), ("tp",))
    self.rep_sharding = NamedSharding(self.mesh, P())
    self.seq2d_sharding = NamedSharding(self.mesh, P("tp", None))
    self.seq1d_sharding = NamedSharding(self.mesh, P("tp",))

    ns = argparse.Namespace(
        model_dir=model_dir,
        preview_dir="",
        refiner_dir="",
        text_encoder_dir="",
        vae_dir="",
        turbo_vae_dir="",
        turbo_vae_config="",
        turbo_vae_ckpt="",
        audio_vae_dir="",
    )
    self.paths = resolve_checkpoint_paths(ns)
    self.preview_cfg = Magi2PreviewConfig.from_checkpoint(self.paths["preview"])
    ref_cfg_json = self.paths["refiner"] / "config.json"
    self.refiner_cfg = (
        Magi2RefinerConfig.from_json(ref_cfg_json)
        if ref_cfg_json.exists()
        else Magi2RefinerConfig()
    )

    print(
        f"[magi2-server] Starting resident server on {self.num_chips}x {self.devices[0].device_kind} "
        f"(dtype={dtype_str}, fp8={fp8}, moe={moe_mode}, refiner={enable_refiner})",
        flush=True,
    )
    t0 = time.perf_counter()

    # 1. Stage 0b: Qwen3.5-27B Text Encoder (resident, ~6.1 GiB/dev)
    from .jax_text import Qwen35TextEncoder

    with _mesh_context(self.mesh):
      self.text_enc = Qwen35TextEncoder(
          self.paths["text_encoder"],
          mesh=self.mesh,
          axis_name="tp",
          dtype=self.dtype,
          skip_layer=2,
      )
    _log_hbm("server after Qwen3.5-27B load", self.devices)

    # 2. Stage 1: 113.93B Preview DiT (resident, ~35.3 GiB BF16 / ~22.6 GiB FP8)
    from .jax_preview_dit import load_magi2_preview_from_safetensors

    with _mesh_context(self.mesh):
      self.preview_model = load_magi2_preview_from_safetensors(
          self.paths["preview"],
          config=self.preview_cfg,
          mesh=self.mesh,
          axis_name="tp",
          dtype=self.dtype,
          fp8=self.fp8,
          moe_mode=self.moe_mode,
          capacity_factor=self.capacity_factor,
          verbose=print,
      )
    self.preview_state: Any = None
    self._base_core_fn: Any = None
    self._pre_gd: Any = None
    self._step_fn_cache: dict[tuple[Any, ...], Any] = {}
    _log_hbm("server after Stage 1 Preview load", self.devices)

    # 3. Stage 2: Optional 6.72B Refiner (resident if enabled, ~12.6 GiB)
    self.refiner_model: Any = None
    self.refiner_mesh: Any = None
    self.refiner_cfg_mode: bool = False
    self._refiner_fn_cache: dict[tuple[Any, ...], Any] = {}
    self.refiner_state: Any = None
    if enable_refiner:
      self._ensure_refiner_loaded()

    # 4. Stage 3a & 3b: TurboVAED + AutoencoderOobleck (resident, ~1.2 GiB)
    from flax import nnx
    from .jax_audio_vae import load_audio_vae
    from .jax_video_vae import load_turbo_vaed

    turbo_cfg_path = self.paths["turbo_vae_config"]
    self.turbo_vae = load_turbo_vaed(
        turbo_cfg_path
        if (turbo_cfg_path is not None and turbo_cfg_path.exists())
        else Magi2TurboVAEDConfig(),
        self.paths["turbo_vae_ckpt"],
        dtype=self.dtype,
        strict=True,
        verbose=True,
    )
    self.audio_vae = load_audio_vae(
        self.paths["audio_vae"],
        dtype=jnp.float32,
        strict=True,
        verbose=True,
    )
    ae_gd, ae_st = nnx.split(self.audio_vae)
    self._audio_state_d0 = jax.tree.map(lambda x: jax.device_put(x, self.devices[0]), ae_st)

    @jax.jit
    def _jit_decode_audio(st, lat_btc):
      return nnx.merge(ae_gd, st).decode(lat_btc, input_layout="btc", output_layout="bct")

    self._audio_jit_fn = _jit_decode_audio

    # 5. Stage 0a: Wan2.2 I2V Encoder (lazy loaded on first I2V request)
    self.wan_enc: Any = None
    self._wan_state_d0: Any = None
    self._wan_jit_fn: Any = None

    self._uncond_mem_cache: dict[str, np.ndarray] = {}
    self.startup_s = time.perf_counter() - t0
    _log_hbm("server ready (all models resident)", self.devices)
    print(f"[magi2-server] Startup complete in {self.startup_s:.2f}s — ready for requests!", flush=True)

  def _ensure_wan_loaded(self) -> None:
    if self.wan_enc is not None:
      return
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from .jax_video_vae import load_wan22_vae_encoder

    self.wan_enc = load_wan22_vae_encoder(
        self.paths["vae"], dtype=jnp.float32, strict=True, verbose=True
    )
    wan_gd, wan_st = nnx.split(self.wan_enc)
    self._wan_state_d0 = jax.tree.map(lambda x: jax.device_put(x, self.devices[0]), wan_st)

    @jax.jit
    def _jit_encode(st, img_tpu):
      return nnx.merge(wan_gd, st).encode(
          img_tpu, input_layout="ncthw", output_layout="ncthw", normalize=True
      )

    self._wan_jit_fn = _jit_encode

  def _ensure_refiner_loaded(self) -> None:
    if self.refiner_model is not None:
      return
    from jax.sharding import Mesh
    from .jax_refiner_dit import load_refiner

    use_cfg_par = (
        self.refiner_cfg_parallel
        and self.num_chips >= 2
        and self.num_chips % 2 == 0
        and self.refiner_cfg.num_heads_q % (self.num_chips // 2) == 0
        and self.refiner_cfg.num_heads_kv % (self.num_chips // 2) == 0
    )
    ref_cp = (self.num_chips // 2) if use_cfg_par else self.num_chips
    self.refiner_cfg_mode = use_cfg_par
    self.refiner_mesh = (
        Mesh(np.array(self.devices).reshape(2, ref_cp), ("cfg", "cp"))
        if use_cfg_par
        else self.mesh
    )
    ref_ax = "cp" if use_cfg_par else "tp"
    self.refiner_model = load_refiner(
        self.paths["refiner"],
        config=self.refiner_cfg,
        mesh=self.refiner_mesh,
        dtype=self.dtype,
        sharding="replicated",
        axis_name=ref_ax,
        verbose=True,
    )
    _log_hbm("server after Stage 2 Refiner load", self.devices)

  def _get_or_bind_preview_step(
      self,
      video_thw: tuple[int, int, int],
      n_aud: int,
      sample_lens: tuple[int, int],
  ) -> Any:
    """Return a compiled step_fn for ``(video_thw, n_aud, sample_lens)`` sharing 113.93B weights."""
    import jax
    from jax import lax
    import jax.numpy as jnp
    from flax import nnx
    from .rope_and_ops import sinusoidal_embedding_1d

    key = (video_thw, n_aud, sample_lens)
    cached = self._step_fn_cache.get(key)
    if cached is not None:
      return cached

    if self.preview_state is None:
      step_fn, state = _make_preview_step_fn(
          self.preview_model,
          video_thw=video_thw,
          n_aud=n_aud,
          sample_lens=sample_lens,
          mesh=self.mesh,
          use_splash=None,
          moe_mode=self.moe_mode,
          capacity_factor=self.capacity_factor,
      )
      self.preview_state = state
      self._base_core_fn = step_fn.core_fn
      self._pre_gd = nnx.split(self.preview_model.pre_adapter)[0]
      self._step_fn_cache[key] = step_fn
      return step_fn

    # Reuse the resident 113.93B packed_state with zero extra HBM weight allocation!
    t_v, h_v, w_v = video_thw
    n_vid = t_v * h_v * w_v
    s_cond, s_uncond = int(sample_lens[0]), int(sample_lens[1])
    cp_size = self.num_chips
    time_dim = self.preview_cfg.time_channel_dim
    v_ch = self.preview_cfg.video_in_channels
    a_ch = self.preview_cfg.audio_in_channels
    model_dtype = self.dtype
    seq_sharding = self.seq2d_sharding if cp_size > 1 else None
    rep_sharding = self.rep_sharding if cp_size > 1 else None
    pre_gd = self._pre_gd

    core_fn = self._base_core_fn.bind_sample_lens((s_cond, s_uncond))
    core_cond_fn = self._base_core_fn.bind_sample_lens((s_cond,))

    def _build_x_time(pre_s, latent_ncthw, audio_btc, raw_t_scalar, parts_tail, time_kind):
      pre_mod = nnx.merge(pre_gd, pre_s)
      v_flat = jnp.transpose(latent_ncthw[0], (1, 2, 3, 0)).reshape(n_vid, v_ch)
      if seq_sharding is not None and (n_vid % cp_size == 0):
        v_flat = lax.with_sharding_constraint(v_flat, seq_sharding)
      v_emb = pre_mod.embed_video(v_flat).astype(model_dtype)
      a_emb = pre_mod.embed_audio(audio_btc[0]).astype(model_dtype)
      parts = [v_emb, a_emb]
      for p in parts_tail:
        if p is None:
          parts.extend([v_emb, a_emb])
        elif p.shape[0] > 0:
          parts.append(p)
      x_emb = jnp.concatenate(parts, axis=0)
      if seq_sharding is not None:
        x_emb = lax.with_sharding_constraint(x_emb, seq_sharding)
      if time_dim > 0:
        emb_noisy = sinusoidal_embedding_1d(raw_t_scalar, dim=time_dim)
        emb_clean = sinusoidal_embedding_1d(jnp.float32(0.0), dim=time_dim)
        emb_zero = jnp.zeros((time_dim,), dtype=jnp.float32)
        time_all = jnp.where(
            (time_kind == 2)[:, None],
            emb_noisy[None, :],
            jnp.where((time_kind == 1)[:, None], emb_clean[None, :], emb_zero[None, :]),
        )
        if seq_sharding is not None:
          time_all = lax.with_sharding_constraint(time_all, seq_sharding)
      else:
        time_all = None
      return x_emb, time_all

    @jax.jit
    def _step_fn(
        st, latent_ncthw, audio_btc, raw_t_scalar, video_cfg, audio_cfg,
        txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb,
        pad_emb, rope_all, mod_all, time_kind_all,
    ):
      pre_s, core_s = st
      x_emb, time_all = _build_x_time(
          pre_s, latent_ncthw, audio_btc, raw_t_scalar,
          (txt_cond_emb, ref_cond_emb, None, txt_uncond_emb, ref_uncond_emb, pad_emb),
          time_kind_all,
      )
      out_all = core_fn(core_s, x_emb, rope_all, mod_all, time_all)
      cond_v = jnp.transpose(out_all[0:n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2))[None]
      cond_a = out_all[n_vid : n_vid + n_aud, :a_ch][None]
      uncond_v = jnp.transpose(out_all[s_cond : s_cond + n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2))[None]
      uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + n_aud, :a_ch][None]
      v_cfg = uncond_v + video_cfg * (cond_v - uncond_v)
      a_cfg = uncond_a + audio_cfg * (cond_a - uncond_a)
      if rep_sharding is not None:
        v_cfg = lax.with_sharding_constraint(v_cfg, rep_sharding)
        a_cfg = lax.with_sharding_constraint(a_cfg, rep_sharding)
      return v_cfg, a_cfg

    @jax.jit
    def _step_with_delta_fn(
        st, latent_ncthw, audio_btc, raw_t_scalar, video_cfg, audio_cfg,
        txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb,
        pad_emb, rope_all, mod_all, time_kind_all,
    ):
      pre_s, core_s = st
      x_emb, time_all = _build_x_time(
          pre_s, latent_ncthw, audio_btc, raw_t_scalar,
          (txt_cond_emb, ref_cond_emb, None, txt_uncond_emb, ref_uncond_emb, pad_emb),
          time_kind_all,
      )
      out_all = core_fn(core_s, x_emb, rope_all, mod_all, time_all)
      cond_v = jnp.transpose(out_all[0:n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2))[None]
      cond_a = out_all[n_vid : n_vid + n_aud, :a_ch][None]
      uncond_v = jnp.transpose(out_all[s_cond : s_cond + n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2))[None]
      uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + n_aud, :a_ch][None]
      dv, da = cond_v - uncond_v, cond_a - uncond_a
      v_cfg = uncond_v + video_cfg * dv
      a_cfg = uncond_a + audio_cfg * da
      if rep_sharding is not None:
        v_cfg = lax.with_sharding_constraint(v_cfg, rep_sharding)
        a_cfg = lax.with_sharding_constraint(a_cfg, rep_sharding)
        dv = lax.with_sharding_constraint(dv, rep_sharding)
        da = lax.with_sharding_constraint(da, rep_sharding)
      return v_cfg, a_cfg, dv, da

    @jax.jit
    def _cond_step_fn(
        st, latent_ncthw, audio_btc, raw_t_scalar, video_cfg, audio_cfg,
        delta_decay, cached_delta_v, cached_delta_a,
        txt_cond_emb, ref_cond_emb, pad_cond_emb, rope_cond, mod_cond, time_kind_cond,
    ):
      pre_s, core_s = st
      x_emb, time_all = _build_x_time(
          pre_s, latent_ncthw, audio_btc, raw_t_scalar,
          (txt_cond_emb, ref_cond_emb, pad_cond_emb),
          time_kind_cond,
      )
      out_c = core_cond_fn(core_s, x_emb, rope_cond, mod_cond, time_all)
      cond_v = jnp.transpose(out_c[0:n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2))[None]
      cond_a = out_c[n_vid : n_vid + n_aud, :a_ch][None]
      v_cfg = cond_v + (video_cfg - jnp.float32(1.0)) * delta_decay * cached_delta_v
      a_cfg = cond_a + (audio_cfg - jnp.float32(1.0)) * delta_decay * cached_delta_a
      if rep_sharding is not None:
        v_cfg = lax.with_sharding_constraint(v_cfg, rep_sharding)
        a_cfg = lax.with_sharding_constraint(a_cfg, rep_sharding)
      return v_cfg, a_cfg

    _step_fn.step_with_delta_fn = _step_with_delta_fn
    _step_fn.cond_step_fn = _cond_step_fn
    _step_fn.core_fn = core_fn
    self._step_fn_cache[key] = _step_fn
    return _step_fn

  def generate(self, req: dict[str, Any]) -> dict[str, Any]:
    """Run a single generation request using resident TPU models (0s weight load)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    t_req0 = time.perf_counter()
    res = str(req.get("resolution", self.default_resolution))
    preset = RESOLUTION_PRESETS[res]
    seconds = float(req.get("seconds", 10.0))
    steps = int(req.get("steps", 100))
    shift = float(req.get("shift", 7.0))
    video_cfg = float(req.get("video_cfg", 5.0))
    audio_cfg = float(req.get("audio_cfg", 7.0))
    seed = int(req.get("seed", 42))
    lossy = bool(req.get("lossy", False))
    cache_thresh = float(req["step_cache_thresh"]) if "step_cache_thresh" in req else (0.30 if lossy else 0.0)
    cfg_skip_after = float(req["cfg_skip_after"]) if "cfg_skip_after" in req else (0.65 if lossy else 1.0)
    max_consec_skip = int(req.get("step_cache_max_skip", 2))
    warmup_frac = float(req.get("step_cache_warmup", 0.12 if lossy else 0.15))
    tail_frac = float(req.get("step_cache_tail", 0.08 if lossy else 0.10))
    run_refiner = bool(req.get("refiner", preset["use_refiner"])) and preset["use_refiner"]
    use_enhance = bool(req.get("enhance", False))
    enhance_mode = str(req.get("enhance_mode", "builtin"))
    neg_prompt = str(req.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))

    raw_prompt = str(req.get("prompt", ""))
    prompt_file = str(req.get("prompt_file", ""))
    if prompt_file:
      raw_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()

    images: list[str] = []
    if req.get("image"):
      images.append(str(req["image"]))
    if req.get("images"):
      images.extend(str(p) for p in req["images"])
    is_i2v = len(images) > 0

    if use_enhance:
      raw_prompt = enhance_prompt_to_pe_json_str(
          raw_prompt, mode=enhance_mode, seconds=seconds, is_i2v=is_i2v, num_refs=len(images)
      )

    if is_i2v:
      figure_ids = list(range(1, len(images) + 1))
      raw_prompt = ensure_figure_tokens(raw_prompt, figure_ids)
    else:
      figure_ids = []
    prompt = json_to_compact_markdown(raw_prompt)

    out_path = Path(req.get("out", f"/workspace/shun/magi2_outputs/server_{seed}.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # Stage 0a: I2V ref image encode (resident Wan2.2 JIT)
    ref_image_latents_np: list[np.ndarray] = []
    preview_h, preview_w = int(preset["height"]), int(preset["width"])
    ref_resize_type = str(req.get("ref_image_type", "original"))
    if is_i2v:
      t0a = time.perf_counter()
      self._ensure_wan_loaded()
      for img_p in images:
        img_np = preprocess_reference_image(
            img_p,
            resize_type=ref_resize_type,
            preview_height=preview_h,
            preview_width=preview_w,
        )
        img_tpu = jax.device_put(jnp.asarray(img_np, dtype=jnp.float32), self.devices[0])
        lat = np.asarray(jax.block_until_ready(self._wan_jit_fn(self._wan_state_d0, img_tpu)), dtype=np.float32)
        ref_image_latents_np.append(lat)
        del img_tpu
      timings["stage0a_i2v_s"] = time.perf_counter() - t0a

    # Stage 0b: Qwen3.5-27B text encode (resident + uncond memory/disk cache!)
    t0b = time.perf_counter()
    with _mesh_context(self.mesh):
      cond_j = jax.block_until_ready(self.text_enc.encode(prompt))
      cond_txt_np = np.asarray(cond_j.astype(jnp.float32), dtype=np.float32)
      ref_special_np = None
      if is_i2v:
        spec_j = self.text_enc.pool_figure_tokens(prompt, figure_ids, cond_j)
        ref_special_np = np.asarray(spec_j.astype(jnp.float32), dtype=np.float32)

      uncond_txt_np = self._uncond_mem_cache.get(neg_prompt)
      if uncond_txt_np is None and self.uncond_cache and neg_prompt:
        cc_root = (
            Path(self.compile_cache).expanduser()
            if self.compile_cache
            else Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/magi2")))
        )
        neg_key = hashlib.sha256(
            f"{self.paths['text_encoder'].resolve()}|skip=2|{self.dtype_str}|{neg_prompt}".encode("utf-8")
        ).hexdigest()[:16]
        disk_p = cc_root / "uncond_cache" / f"qwen35_uncond_{neg_key}.npy"
        if disk_p.exists():
          uncond_txt_np = np.asarray(np.load(disk_p, allow_pickle=False), dtype=np.float32)
        else:
          uncond_j = jax.block_until_ready(self.text_enc.encode(neg_prompt))
          uncond_txt_np = np.asarray(uncond_j.astype(jnp.float32), dtype=np.float32)
          disk_p.parent.mkdir(parents=True, exist_ok=True)
          np.save(disk_p, uncond_txt_np)
        self._uncond_mem_cache[neg_prompt] = uncond_txt_np
      elif uncond_txt_np is None:
        uncond_txt_np = np.zeros((1, 0, self.preview_cfg.text_in_channels), dtype=np.float32)
    timings["stage0b_text_s"] = time.perf_counter() - t0b

    # Stage 1: Resident 113.93B Preview DiT
    t1 = time.perf_counter()
    video_latent_t = 32 if seconds == 10.0 else max(1, int(round(seconds * 12.5)) // 4 + 1)
    preview_lat_h, preview_lat_w = preview_h // 16, preview_w // 16
    audio_latent_t = int(round(seconds * 25.0))
    refiner_lat_t = int(preset.get("refiner_latent_t", 63))
    refiner_lat_h = int(preset.get("refiner_height", 1088)) // 16
    refiner_lat_w = int(preset.get("refiner_width", 1920)) // 16

    init_video_np, init_audio_np, refiner_noise_np = generate_initial_noise(
        video_shape=(1, self.preview_cfg.video_in_channels, video_latent_t, preview_lat_h, preview_lat_w),
        audio_shape=(1, audio_latent_t, self.preview_cfg.audio_in_channels),
        refiner_shape=(1, self.refiner_cfg.video_in_channels, refiner_lat_t, refiner_lat_h, refiner_lat_w)
        if run_refiner
        else None,
        seed=seed,
        torch_noise=True,
    )

    use_ref_uncond = bool(req.get("use_ref_for_uncond", True))
    ref_hws = [(int(r.shape[3]), int(r.shape[4])) for r in ref_image_latents_np]
    plan = build_preview_packing_plan(
        video_thw=(video_latent_t, preview_lat_h, preview_lat_w),
        audio_len=audio_latent_t,
        cond_txt_len=int(cond_txt_np.shape[1]),
        uncond_txt_len=int(uncond_txt_np.shape[1]),
        ref_img_hws=ref_hws,
        cp_size=self.num_chips,
        head_dim=self.preview_cfg.head_dim,
        rope_temperature=self.preview_cfg.rope_temperature,
    )

    with _mesh_context(self.mesh):
      txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb, pad_emb = _pre_embed_static_preview_tokens(
          self.preview_model,
          cond_txt_np=cond_txt_np,
          uncond_txt_np=uncond_txt_np,
          ref_special_np=ref_special_np,
          ref_latents_np=ref_image_latents_np,
          use_ref_for_uncond=use_ref_uncond,
          pad_size=plan["pad_size"],
          mesh=self.mesh,
      )
      rope_all = jax.device_put(jnp.asarray(plan["rope_all"], dtype=jnp.float32), self.seq2d_sharding)
      mod_all = jax.device_put(jnp.asarray(plan["mod_all"], dtype=jnp.int32), self.seq1d_sharding)
      time_kind_all = jax.device_put(jnp.asarray(plan["time_kind_all"], dtype=jnp.int32), self.seq1d_sharding)

      step_fn = self._get_or_bind_preview_step(
          (video_latent_t, preview_lat_h, preview_lat_w), audio_latent_t, plan["sample_lens"]
      )
      use_step_cache = cache_thresh > 0.0
      use_cfg_skip = cfg_skip_after < 1.0 and getattr(step_fn, "cond_step_fn", None) is not None
      cond_aux: list[Any] = []
      if use_cfg_skip:
        s_cond_len = int(plan["sample_lens"][0])
        pad_cond_len = (-s_cond_len) % self.num_chips
        pad_cond_emb = jax.device_put(
            jnp.zeros((pad_cond_len, self.preview_cfg.stream_hidden_size), dtype=self.dtype),
            self.rep_sharding,
        )
        rc, mc, tc = plan["rope_all"][:s_cond_len], plan["mod_all"][:s_cond_len], plan["time_kind_all"][:s_cond_len]
        if pad_cond_len > 0:
          rc = np.concatenate([rc, np.zeros((pad_cond_len, rc.shape[1]), dtype=np.float32)], axis=0)
          mc = np.concatenate([mc, np.full((pad_cond_len,), 2, dtype=np.int32)], axis=0)
          tc = np.concatenate([tc, np.zeros((pad_cond_len,), dtype=np.int32)], axis=0)
        rope_cond_only = jax.device_put(jnp.asarray(rc, dtype=jnp.float32), self.seq2d_sharding)
        mod_cond_only = jax.device_put(jnp.asarray(mc, dtype=jnp.int32), self.seq1d_sharding)
        tk_cond_only = jax.device_put(jnp.asarray(tc, dtype=jnp.int32), self.seq1d_sharding)
        cond_aux.extend([pad_cond_emb, rope_cond_only, mod_cond_only, tk_cond_only])

      if use_step_cache:
        @jax.jit
        def _vel_rel(vc, vp, ac, ap):
          rv = jnp.sum(jnp.abs(vc - vp)) / jnp.maximum(jnp.sum(jnp.abs(vc)), jnp.float32(1e-6))
          ra = jnp.sum(jnp.abs(ac - ap)) / jnp.maximum(jnp.sum(jnp.abs(ac)), jnp.float32(1e-6))
          return rv, ra

      video_cfg_j = jax.device_put(jnp.float32(video_cfg), self.rep_sharding)
      audio_cfg_j = jax.device_put(jnp.float32(audio_cfg), self.rep_sharding)
      vid_sched = FlowUniPCMultistepScheduler()
      aud_sched = FlowUniPCMultistepScheduler()
      vid_sched.set_timesteps(steps, shift=shift)
      aud_sched.set_timesteps(steps, shift=shift)

      latent_v = jax.device_put(jnp.asarray(init_video_np, dtype=jnp.float32), self.rep_sharding)
      latent_a = jax.device_put(jnp.asarray(init_audio_np, dtype=jnp.float32), self.rep_sharding)

      n_tot = len(vid_sched.timesteps)
      c_start = int(round(n_tot * warmup_frac))
      c_end = n_tot - int(round(n_tot * tail_frac))
      cfg_step = int(round(n_tot * cfg_skip_after))
      prev_v, prev_a, dv, da = None, None, None, None
      last_rate, accum_err, c_skips = 1.0, 0.0, 0
      n_full, n_cond, n_hit = 0, 0, 0

      for i, t_int in enumerate(vid_sched.timesteps):
        raw_t_j = jax.device_put(jnp.float32(float(np.int64(t_int)) / 1000.0), self.rep_sharding)
        if (
            use_step_cache
            and prev_v is not None
            and c_start <= i < c_end
            and c_skips < max_consec_skip
            and (accum_err + last_rate) < cache_thresh
        ):
          vv, va = prev_v, prev_a
          accum_err += last_rate
          c_skips += 1
          n_hit += 1
        elif use_cfg_skip and i >= cfg_step and dv is not None:
          decay = max(0.30, 1.0 - 0.70 * (i - cfg_step) / max(1, n_tot - cfg_step))
          vv, va = step_fn.cond_step_fn(
              self.preview_state, latent_v, latent_a, raw_t_j, video_cfg_j, audio_cfg_j,
              jax.device_put(jnp.float32(decay), self.rep_sharding), dv, da,
              txt_cond_emb, ref_cond_emb, pad_cond_emb, rope_cond_only, mod_cond_only, tk_cond_only,
          )
          n_cond += 1
          if use_step_cache and prev_v is not None:
            rv, ra = _vel_rel(vv, prev_v, va, prev_a)
            last_rate = max(float(rv), 0.7 * float(ra)) / float(c_skips + 1)
          accum_err, c_skips, prev_v, prev_a = 0.0, 0, vv, va
        elif use_cfg_skip:
          vv, va, dv, da = step_fn.step_with_delta_fn(
              self.preview_state, latent_v, latent_a, raw_t_j, video_cfg_j, audio_cfg_j,
              txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb, pad_emb,
              rope_all, mod_all, time_kind_all,
          )
          n_full += 1
          if use_step_cache and prev_v is not None:
            rv, ra = _vel_rel(vv, prev_v, va, prev_a)
            last_rate = max(float(rv), 0.7 * float(ra)) / float(c_skips + 1)
          accum_err, c_skips, prev_v, prev_a = 0.0, 0, vv, va
        else:
          vv, va = step_fn(
              self.preview_state, latent_v, latent_a, raw_t_j, video_cfg_j, audio_cfg_j,
              txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb, pad_emb,
              rope_all, mod_all, time_kind_all,
          )
          n_full += 1
          if use_step_cache:
            if prev_v is not None:
              rv, ra = _vel_rel(vv, prev_v, va, prev_a)
              last_rate = max(float(rv), 0.7 * float(ra)) / float(c_skips + 1)
            accum_err, c_skips, prev_v, prev_a = 0.0, 0, vv, va
        latent_v = vid_sched.step(vv, t_int, latent_v, return_dict=False)[0]
        latent_a = aud_sched.step(va, t_int, latent_a, return_dict=False)[0]

      jax.block_until_ready((latent_v, latent_a))
      stage1_video_np = np.asarray(latent_v, dtype=np.float32)
      stage1_audio_np = np.asarray(latent_a, dtype=np.float32)
      _free_device_arrays(
          latent_v, latent_a, prev_v, prev_a, dv, da,
          txt_cond_emb, ref_cond_emb, txt_uncond_emb, ref_uncond_emb, pad_emb,
          rope_all, mod_all, time_kind_all, *cond_aux,
      )
    timings["stage1_denoise_s"] = time.perf_counter() - t1

    # Stage 2: Optional 1080p Refiner (resident!)
    final_video_np = stage1_video_np
    fps = 25.0 if run_refiner else 12.5
    if run_refiner:
      t2 = time.perf_counter()
      self._ensure_refiner_loaded()
      from .jax_refiner_dit import (
          build_refiner_attention_layout,
          make_jitted_cfg_forward,
          make_jitted_forward as make_ref_fwd,
          pack_refiner_inputs_np,
          unpack_refiner_video_jax,
      )

      target_thw = (refiner_lat_t, refiner_lat_h, refiner_lat_w)
      z_up = trilinear_upsample_latents(stage1_video_np, target_thw)
      z_ref_init = refiner_noise_augment(z_up, refiner_noise_np, noise_value=self.refiner_cfg.noise_value)
      bvol = self.refiner_cfg.block_t_size * self.refiner_cfg.block_h_size * self.refiner_cfg.block_w_size
      l_c, l_u = int(cond_txt_np.shape[1]), int(uncond_txt_np.shape[1])
      t_bucket = math.ceil(max(l_c, l_u) / bvol) * bvol if max(l_c, l_u) > 0 else 0
      ref_cp = (self.num_chips // 2) if self.refiner_cfg_mode else self.num_chips
      lc = build_refiner_attention_layout(target_thw, n_text=l_c, n_audio=0, config=self.refiner_cfg, tail_bucket_len=t_bucket or None, cp_size=ref_cp)
      lu = build_refiner_attention_layout(target_thw, n_text=l_u, n_audio=0, config=self.refiner_cfg, tail_bucket_len=t_bucket or None, cp_size=ref_cp)
      _, _, tc_p = pack_refiner_inputs_np(z_ref_init, cond_txt_np, lc)
      _, _, tu_p = pack_refiner_inputs_np(z_ref_init, uncond_txt_np, lu)
      attn_impl = "masked_pallas_fp8" if self.refiner_qk_fp8 else None
      rkey = (target_thw, t_bucket, self.refiner_cfg_mode, attn_impl)
      ref_rep = NamedSharding(self.refiner_mesh, P())
      with _mesh_context(self.refiner_mesh):
        if rkey not in self._refiner_fn_cache:
          if self.refiner_cfg_mode:
            rfn, _, rst = make_jitted_cfg_forward(self.refiner_model, lc, mesh=self.refiner_mesh, cfg_axis="cfg", cp_axis="cp", compute_dtype=self.dtype, attn_impl=attn_impl)
          else:
            rfn, _, rst = make_ref_fwd(self.refiner_model, lc, mesh=self.refiner_mesh, cp_axis="tp", parallel_mode="cp", compute_dtype=self.dtype, attn_impl=attn_impl)
          if self.refiner_state is None:
            self.refiner_state = rst
          self._refiner_fn_cache[rkey] = rfn
        rfn = self._refiner_fn_cache[rkey]
        src_j = jax.device_put(jnp.asarray(lc.vid_src_idx, dtype=jnp.int32), ref_rep)
        lat_r = jax.device_put(jnp.asarray(z_ref_init, dtype=jnp.float32), ref_rep)
        rsched = FlowUniPCMultistepScheduler()
        rsched.set_timesteps(int(req.get("refiner_steps", self.refiner_cfg.num_inference_steps)), shift=self.refiner_cfg.shift)
        ref_cfg_val = float(req.get("refiner_cfg", self.refiner_cfg.video_cfg))
        if self.refiner_cfg_mode:
          t2_j = jax.device_put(jnp.asarray(np.stack([tc_p, tu_p], 0), dtype=jnp.float32), NamedSharding(self.refiner_mesh, P("cfg", None, None)))
          r2_j = jax.device_put(jnp.asarray(np.stack([lc.rope_96, lu.rope_96], 0), dtype=jnp.float32), NamedSharding(self.refiner_mesh, P("cfg", "cp", None)))
          b2_j = jax.device_put(jnp.asarray(np.stack([lc.kv_block_valid_lens, lu.kv_block_valid_lens], 0), dtype=jnp.int32), NamedSharding(self.refiner_mesh, P("cfg", None)))
          rcfg_j = jax.device_put(jnp.float32(ref_cfg_val), ref_rep)
          for t_int in rsched.timesteps:
            vr = jnp.transpose(lat_r[0], (1, 2, 3, 0)).reshape(-1, lat_r.shape[1])
            vp = jnp.where((src_j >= 0)[:, None], vr[jnp.maximum(src_j, 0)], jnp.float32(0.0))
            vpad = rfn(self.refiner_state, vp, t2_j, r2_j, b2_j, rcfg_j)
            lat_r = rsched.step(unpack_refiner_video_jax(vpad, lc)[None], t_int, lat_r, return_dict=False)[0]
          _free_device_arrays(t2_j, r2_j, b2_j, rcfg_j)
        else:
          seq2 = NamedSharding(self.refiner_mesh, P("tp", None))
          tc_j = jax.device_put(jnp.asarray(tc_p, dtype=jnp.float32), ref_rep)
          tu_j = jax.device_put(jnp.asarray(tu_p, dtype=jnp.float32), ref_rep)
          rc_j = jax.device_put(jnp.asarray(lc.rope_96, dtype=jnp.float32), seq2)
          ru_j = jax.device_put(jnp.asarray(lu.rope_96, dtype=jnp.float32), seq2)
          bc_j = jax.device_put(jnp.asarray(lc.kv_block_valid_lens, dtype=jnp.int32), ref_rep)
          bu_j = jax.device_put(jnp.asarray(lu.kv_block_valid_lens, dtype=jnp.int32), ref_rep)
          for t_int in rsched.timesteps:
            vr = jnp.transpose(lat_r[0], (1, 2, 3, 0)).reshape(-1, lat_r.shape[1])
            vp = jnp.where((src_j >= 0)[:, None], vr[jnp.maximum(src_j, 0)], jnp.float32(0.0))
            vc, _ = rfn(self.refiner_state, vp, tc_j, rc_j, bc_j, None)
            vu, _ = rfn(self.refiner_state, vp, tu_j, ru_j, bu_j, None)
            vpad = vu + jnp.float32(ref_cfg_val) * (vc - vu)
            lat_r = rsched.step(unpack_refiner_video_jax(vpad, lc)[None], t_int, lat_r, return_dict=False)[0]
          _free_device_arrays(tc_j, tu_j, rc_j, ru_j, bc_j, bu_j)
        jax.block_until_ready(lat_r)
        final_video_np = np.asarray(lat_r, dtype=np.float32)
        _free_device_arrays(src_j, lat_r)
      timings["stage2_refiner_s"] = time.perf_counter() - t2

    # Stage 3a & 3b: Resident 8-chip TurboVAED + Oobleck stereo audio
    t3v = time.perf_counter()
    if self.decode_chunk_shard and self.num_chips > 1:
      with _mesh_context(self.mesh):
        frames_u8 = self.turbo_vae.decode_sharded(
            final_video_np,
            mesh=self.mesh,
            devices=self.devices,
            axis_name="tp",
            input_layout="ncthw",
            denormalize=True,
            to_uint8=True,
        )
    else:
      z_d0 = jax.device_put(jnp.asarray(final_video_np, dtype=jnp.float32), self.devices[0])
      rgb = np.asarray(self.turbo_vae.decode(z_d0, input_layout="ncthw", output_layout="ndhwc", denormalize=True, output_offload=True)[0], dtype=np.float32)
      frames_u8 = (np.clip(rgb * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    timings["stage3a_video_s"] = time.perf_counter() - t3v

    t3a = time.perf_counter()
    from .jax_audio_vae import adaptive_volume_scaling, resample_audio_25hz_to_44100

    a_d0 = jax.device_put(jnp.asarray(stage1_audio_np, dtype=jnp.float32), self.devices[0])
    wav_raw = np.asarray(
        jax.block_until_ready(self._audio_jit_fn(self._audio_state_d0, a_d0)),
        dtype=np.float32,
    )
    del a_d0
    audio_44k = resample_audio_25hz_to_44100(wav_raw[0], axis=-1)
    if bool(req.get("adaptive_volume_scaling", False)):
      audio_44k = adaptive_volume_scaling(audio_44k)
    timings["stage3b_audio_s"] = time.perf_counter() - t3a

    t_mux = time.perf_counter()
    wav_path = out_path.with_suffix(".wav")
    save_stereo_wav(audio_44k, wav_path, sample_rate=self.audio_vae.sample_rate)
    mux_video_and_audio_ffmpeg(frames_u8, out_path, fps=fps, wav_path=wav_path)
    timings["stage3_mux_s"] = time.perf_counter() - t_mux
    timings["total_s"] = time.perf_counter() - t_req0
    timings["steps_breakdown"] = {"full_cfg": n_full, "cond_only": n_cond, "cached": n_hit}

    gc.collect()
    print(
        f"[magi2-server] Done {out_path} in {timings['total_s']:.2f}s "
        f"(0b={timings['stage0b_text_s']:.2f}s, s1={timings['stage1_denoise_s']:.2f}s "
        f"[full={n_full}/cond={n_cond}/hit={n_hit}], "
        f"s2={timings.get('stage2_refiner_s', 0.0):.2f}s, "
        f"s3v={timings['stage3a_video_s']:.2f}s, s3a={timings['stage3b_audio_s']:.2f}s)",
        flush=True,
    )
    return timings


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description="MAGI-2-preview TPU-Resident Native JAX Server")
  ap.add_argument("--model-dir", default=os.environ.get("MAGI2_CKPT_ROOT", "/workspace/shun/magi2_preview_weights"))
  ap.add_argument("--requests", required=True, help="Path to JSONL file with one request JSON object per line.")
  ap.add_argument("--resolution", default="540p", choices=tuple(RESOLUTION_PRESETS.keys()))
  ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
  ap.add_argument("--fp8", action="store_true")
  ap.add_argument("--moe-mode", default="auto", choices=("auto", "ag", "ep", "ref"))
  ap.add_argument("--capacity-factor", type=float, default=None)
  ap.add_argument("--enable-refiner", action="store_true", help="Preload 6.72B Stage 2 Refiner resident in HBM.")
  ap.add_argument("--refiner-qk-fp8", action="store_true")
  ap.add_argument("--lossy", action="store_true", help="Default --lossy for requests that don't specify 'lossy'.")
  ap.add_argument("--enhance", action="store_true", help="Default --enhance for requests that don't specify 'enhance'.")
  ap.add_argument("--compile-cache", default=os.environ.get("MAGI2_COMPILE_CACHE", ""))
  ap.add_argument("--chips", type=int, default=0)
  args = ap.parse_args(argv)

  reqs_path = Path(args.requests)
  reqs: list[dict[str, Any]] = []
  for line in reqs_path.read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
      continue
    obj = json.loads(s)
    obj.setdefault("resolution", args.resolution)
    obj.setdefault("lossy", args.lossy)
    obj.setdefault("enhance", args.enhance)
    reqs.append(obj)

  need_refiner = args.enable_refiner or any(r.get("resolution") == "1080p" for r in reqs)
  server = Magi2ResidentServer(
      model_dir=args.model_dir,
      resolution=args.resolution,
      dtype_str=args.dtype,
      fp8=args.fp8,
      moe_mode=args.moe_mode,
      capacity_factor=args.capacity_factor,
      enable_refiner=need_refiner,
      refiner_qk_fp8=args.refiner_qk_fp8,
      compile_cache=args.compile_cache,
      chips=args.chips,
  )

  all_results = []
  for idx, req in enumerate(reqs):
    print(f"\n[magi2-server] === Request {idx + 1}/{len(reqs)} ===", flush=True)
    res = server.generate(req)
    all_results.append({"index": idx + 1, "out": req.get("out", ""), **res})

  print("\n[magi2-server] === All Requests Summary ===", flush=True)
  for r in all_results:
    print(
        f"  req#{r['index']}: total={r['total_s']:.2f}s (s1={r['stage1_denoise_s']:.2f}s, "
        f"s3v={r['stage3a_video_s']:.2f}s, s3a={r['stage3b_audio_s']:.2f}s) -> {r['out']}",
        flush=True,
    )


if __name__ == "__main__":
  main()
