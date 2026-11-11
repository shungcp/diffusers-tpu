"""MiniMax-H3-Image (t2i and multi-reference edit) through torchax.

Executes diffusers' PyTorch `MiniMaxH3ImageTransformer2DModel` on JAX/TPU via
`torchax`, reusing `torchax_pipeline.py`'s 8-way TP sharding (`spec_for`),
Pallas FlashAttention (`install_attention`), and segmented AdaLN modulation
(`install_segmented_modulation`), paired with the native JAX Qwen3-VL text
tower and 2D ViT image decoder.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_dvfs_p_state=7 "
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_async_all_to_all=true "
    "--xla_tpu_enable_ici_ag_pipelining=true "
    "--xla_lhs_prioritize_async_depth_over_stall=ENABLED "
    "--xla_tpu_aggressive_opt_barrier_removal=ENABLED "
    "--xla_tpu_scoped_vmem_limit_kib=65536 "
    "--xla_tpu_enable_offloading_copy_to_sparsecore=false "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
)

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from .jax_image_pipeline import (
    DEFAULT_MIN_GENERATION_AREA,
    IMAGE_TAG,
    PDD_SAMPLE_SIGMAS,
    PIXEL_MEAN,
    PIXEL_STD,
    TEXT_ENCODER_LAYER,
    TEXT_TAG,
    _resolve_snapshots,
    bound_short_edge,
    build_image_packed_sequence,
    patchify_2d,
    prepare_canvas,
)
from .torchax_pipeline import (
    _to_jax,
    install_attention,
    install_segmented_modulation,
    segment_runs,
    spec_for,
)


class TorchaxImageTransformer(torch.nn.Module):
  """`MiniMaxH3ImageTransformer2DModel` compiled once across all 6 PDD steps via torchax."""

  def __init__(self, model: torch.nn.Module, mesh: jax.sharding.Mesh,
               scoped_vmem_kib: int = 65536):
    super().__init__()
    import torchax

    self._model = model
    self._mesh = mesh
    self._env = torchax.default_env()
    self._cache: dict = {}
    self._runs = None
    self.scoped_vmem_kib = scoped_vmem_kib
    self._device = {
        name: NamedSharding(mesh, spec_for(name))
        for name, _ in list(model.named_parameters()) + list(model.named_buffers())
    }
    self._params = {
        name: jax.device_put(_to_jax(tensor), self._device[name])
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
    }

  def forward_jax(self, inputs_jax: dict, token_tags_cpu: torch.Tensor,
                  timestep_indices_cpu: torch.Tensor, pdd_step_index: int):
    runs = segment_runs(token_tags_cpu, timestep_indices_cpu)
    if runs != self._runs:
      install_segmented_modulation(self._model, runs, reassoc=False)
      self._runs = runs
      print(f"  [segments] {runs}", flush=True)

    key = (
        tuple((k, tuple(v.shape), str(v.dtype)) for k, v in sorted(inputs_jax.items())),
        self._runs,
    )
    fn = self._cache.get(key)
    if fn is None:
      def call(params, inputs):
        with self._env:
          out = torch.func.functional_call(
              self._model,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              kwargs={
                  **{k: self._env.j2t_iso(v) for k, v in inputs.items()},
                  "pdd_step_index": 0,
                  "return_dict": False,
              },
          )
        return self._env.t2j_iso(out[0])

      opts = (
          {"xla_tpu_scoped_vmem_limit_kib": str(self.scoped_vmem_kib)}
          if self.scoped_vmem_kib
          else None
      )
      fn = jax.jit(call, compiler_options=opts)
      self._cache[key] = fn

    # Swap step `pdd_step_index`'s head into slot 0 so all 6 steps share 1 compiled program
    step_params = dict(self._params)
    if pdd_step_index != 0:
      step_params["pdd_proj_out.0.weight"] = self._params[
          f"pdd_proj_out.{pdd_step_index}.weight"
      ]
      step_params["pdd_proj_out.0.bias"] = self._params[
          f"pdd_proj_out.{pdd_step_index}.bias"
      ]
    with self._mesh:
      return fn(step_params, inputs_jax)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3-Image")
  p.add_argument("--text-model-id", default="MiniMaxAI/MiniMax-H3")
  p.add_argument("--prompt", default=(
      "A cinematic close-up portrait of a red fox in a snowy pine forest at "
      "dawn, crisp frost on whiskers, golden rim light, ultra-detailed 4K."))
  p.add_argument("--reference", action="append", default=[])
  p.add_argument("--size", default=None)
  p.add_argument("--width", type=int, default=2048)
  p.add_argument("--height", type=int, default=2048)
  p.add_argument("--min-generation-area", type=int, default=DEFAULT_MIN_GENERATION_AREA)
  p.add_argument("--steps", type=int, default=6)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--chips", type=int, default=0)
  p.add_argument("--block-q", type=int, default=2048)
  p.add_argument("--block-kv", type=int, default=4096)
  p.add_argument("--fixed-m", action="store_true", default=True)
  p.add_argument("--full-lane-rope", action="store_true", default=True)
  p.add_argument("--qk-fp8", action="store_true")
  p.add_argument("--scoped-vmem-kib", type=int, default=65536)
  p.add_argument("--warmup", action="store_true", default=True)
  p.add_argument("--runs", type=int, default=2)
  p.add_argument("--out", default="out")
  p.add_argument("--tag", default="h3_image_torchax")
  args = p.parse_args()
  if args.size:
    w_str, h_str = args.size.lower().split("x")
    args.width, args.height = int(w_str), int(h_str)
  return args


def main():
  args = parse_args()

  import torchax
  from PIL import Image, ImageOps
  from transformers import AutoProcessor, AutoTokenizer
  from diffusers import MiniMaxH3ImageTransformer2DModel
  from diffusers.image_processor import VaeImageProcessor
  from diffusers.utils.torch_utils import randn_tensor

  from . import jax_ref2va as r2v
  from . import jax_text as jtext
  from . import jax_vae as jvae
  from . import jax_vision as jvision
  from ..qwen3_vl import jax_text as qwen_text
  from ..qwen3_vl import jax_vision as _qv
  from ...safetensors_io import safe_open

  torchax.enable_globally()
  devices = jax.devices()[:args.chips or None]
  mesh = jax.make_mesh(
      (len(devices),), ("tp",),
      devices=devices,
      axis_types=(jax.sharding.AxisType.Auto,),
  )
  dtype = jnp.bfloat16
  replicated = NamedSharding(mesh, P())

  began = time.perf_counter()
  img_snap, text_dir = _resolve_snapshots(args.model_id, args.text_model_id)
  vae_dir = os.path.join(
      img_snap,
      "vae_diffusers" if os.path.isdir(os.path.join(img_snap, "vae_diffusers")) else "vae",
  )
  dit_sub = (
      "transformer_diffusers"
      if os.path.isdir(os.path.join(img_snap, "transformer_diffusers"))
      else "transformer"
  )
  with open(os.path.join(vae_dir, "config.json")) as f:
    vae_cfg = json.load(f)
  lat_mean_np = np.asarray(vae_cfg["latents_mean"], np.float32).reshape(1, -1, 1, 1)
  lat_std_np = np.asarray(vae_cfg["latents_std"], np.float32).reshape(1, -1, 1, 1)
  lat_mean_jnp = jnp.asarray(lat_mean_np)
  lat_std_jnp = jnp.asarray(lat_std_np)

  print(f"MiniMax-H3-Image torchax on {len(devices)} device(s) "
        f"({devices[0].device_kind}), workflow={'edit' if args.reference else 't2i'}",
        flush=True)

  # -- 1. Load Qwen3-VL Text & Vision Towers ---------------------------------
  t0 = time.perf_counter()
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(img_snap, "tokenizer"))
  processor = AutoProcessor.from_pretrained(os.path.join(img_snap, "processor"))
  encoder, cfg_text = jtext.build(text_dir, dtype=dtype, abstract=True)
  jtext.load(encoder, text_dir, mesh=mesh, spec_fn=jtext.param_spec)
  vision_model = None
  if args.reference:
    vision_model, _ = jvision.build(text_dir, dtype=dtype, abstract=True)
    jvision.load(vision_model, text_dir, mesh=mesh, spec_fn=lambda k, n: P())
    if len(devices) > 1:
      _qv.HEAD_SHARD_AXIS = "tp"
  print(f"  text encoder loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

  # -- 2. Load PyTorch MiniMaxH3ImageTransformer2DModel & Wrap in Torchax -----
  t0 = time.perf_counter()
  with torchax.disable_temporarily():
    pt_dit = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        img_snap, subfolder=dit_sub, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    pt_dit.requires_grad_(False)
  install_attention(
      pt_dit,
      mesh,
      fixed_m=args.fixed_m,
      block_q=args.block_q,
      block_kv=args.block_kv,
      full_lane_rope=args.full_lane_rope,
      qk_fp8=args.qk_fp8,
  )
  tx_dit = TorchaxImageTransformer(pt_dit, mesh, scoped_vmem_kib=args.scoped_vmem_kib)
  del pt_dit
  print(f"  DiT loaded & sharded via torchax ({time.perf_counter() - t0:.1f}s)", flush=True)

  # -- 3. Load JAX 2D Image VAE ----------------------------------------------
  t0 = time.perf_counter()
  decoder = nnx.eval_shape(
      lambda: jvae.ViTDecoder(**jvae.IMAGE_DECODER_CONFIG, dtype=dtype, rngs=nnx.Rngs(0))
  )
  post_q = jvae.PostQuantConv(jvae.IMAGE_DECODER_CONFIG["in_channels"],
                              dtype=dtype, rngs=nnx.Rngs(0))
  vae_encoder = quant_conv = None
  if args.reference:
    vae_encoder = nnx.eval_shape(lambda: jvae.Encoder3d(dtype=dtype, rngs=nnx.Rngs(0)))
    quant_conv = jvae.PostQuantConv(48, dtype=dtype, rngs=nnx.Rngs(0))
  dec_params, enc_params, qc_params = {}, {}, {}
  for shard in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))):
    with safe_open(shard, framework="flax") as reader:
      for k in reader.keys():
        if k.startswith("decoder."):
          dec_params[k] = reader.get_tensor(k)
        elif k.startswith("post_quant_conv."):
          leaf = getattr(post_q, k.split(".")[-1])
          leaf.value = jax.device_put(
              jnp.asarray(reader.get_tensor(k)).astype(leaf.value.dtype).reshape(leaf.value.shape),
              replicated,
          )
        elif args.reference and k.startswith("encoder."):
          enc_params[k] = reader.get_tensor(k)
        elif args.reference and k.startswith("quant_conv."):
          qc_params[k] = reader.get_tensor(k)
  jvae.load_decoder(decoder, dec_params, mesh=mesh, spec_for=lambda _n, _d: P())
  dec_graph, dec_state = nnx.split(decoder)
  pq_graph, pq_state = nnx.split(post_q)

  @functools.partial(jax.jit, out_shardings=replicated)
  def decode_tile_batch(state, z, pq=pq_state):
    with jax.default_matmul_precision("highest"):
      z = nnx.merge(pq_graph, pq)(z)
      return nnx.merge(dec_graph, state)(z, sdpa=None)

  @functools.partial(jax.jit, out_shardings=replicated)
  def to_uint8_rgb(pixels):
    mean = jnp.asarray(PIXEL_MEAN, jnp.float32).reshape(1, 3, 1, 1)
    std = jnp.asarray(PIXEL_STD, jnp.float32).reshape(1, 3, 1, 1)
    rgb = jnp.clip(pixels.astype(jnp.float32) * std + mean, 0.0, 1.0)
    u8 = jnp.clip(rgb * 255.0, 0.0, 255.0).astype(jnp.uint8)
    return jnp.transpose(u8[0], (1, 2, 0))

  enc_w = None
  if args.reference:
    jvae.load_encoder(vae_encoder, enc_params, mesh=mesh, spec_for=lambda _n, _d: P())
    jvae.load_quant_conv(quant_conv, qc_params, prefix="quant_conv.")
    _qc = nnx.state(quant_conv, nnx.Param)
    nnx.update(quant_conv, jax.tree.map(lambda a: jax.device_put(a, replicated), _qc))
    eg_vae, es_vae = nnx.split(vae_encoder)
    qg_vae, qs_vae = nnx.split(quant_conv)
    enc_w = (es_vae, qs_vae)

    def _enc_one_tile(w, tile):
      with jax.default_matmul_precision("highest"):
        e_mod = nnx.merge(eg_vae, w[0])
        q_mod = nnx.merge(qg_vae, w[1])
        return jvae.encode_video(e_mod, q_mod, tile)

    _ENC_ONE[0] = _enc_one_tile

  print(f"  2D Image VAE loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

  out_w, out_h, gen_w, gen_h, latent_h, latent_w = prepare_canvas(
      args.width, args.height, args.min_generation_area
  )
  timesteps_np = np.asarray([1.0 - s for s in PDD_SAMPLE_SIGMAS[:-1]], dtype=np.float32)
  sigmas_np = np.concatenate([1.0 - timesteps_np, np.zeros((1,), dtype=np.float32)])

  _text_graph, _text_state = nnx.split(encoder)
  _text_jit_raw = jax.jit(
      lambda st, i, c, s: nnx.merge(_text_graph, st)(
          i, c, s, stop_at_layer=TEXT_ENCODER_LAYER
      )
  )
  _text_jit = lambda _m, i, c, s: _text_jit_raw(_text_state, i, c, s)

  def encode_prompt_and_refs():
    if not args.reference:
      token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:4096]
      tags = np.ones((len(token_ids),), dtype=np.int64)
      ids_np = np.asarray([token_ids], dtype=np.int32)
      pos = np.broadcast_to(np.arange(ids_np.shape[1])[None, :], (3, ids_np.shape[1]))
      cos, sin = qwen_text.mrope_cos_sin(
          pos,
          head_dim=cfg_text["head_dim"],
          rope_theta=cfg_text["rope_theta"],
          mrope_section=cfg_text["mrope_section"],
      )
      cos_j = jax.device_put(jnp.asarray(cos, dtype), replicated)
      sin_j = jax.device_put(jnp.asarray(sin, dtype), replicated)
      ids_j = jax.device_put(jnp.asarray(ids_np, jnp.int32), replicated)
      p_embeds = _text_jit(encoder, ids_j, cos_j, sin_j)
      return p_embeds, tags, [], []

    raw_imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in args.reference]
    ref_imgs = [bound_short_edge(im, 1024, 2048) for im in raw_imgs]
    vis_imgs = [bound_short_edge(im, 1024, 1024) for im in raw_imgs]
    with torchax.disable_temporarily():
      vision = processor.image_processor(images=vis_imgs, return_tensors="pt")
      merge_size = processor.image_processor.merge_size ** 2
      v_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
      v_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
      img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
      token_ids, token_tags = [], []
      for idx, grid in enumerate(vision["image_grid_thw"]):
        label_ids = tokenizer(f"Picture {idx + 1}: ", add_special_tokens=False)["input_ids"][:4096]
        v_len = int(grid.prod().item()) // merge_size
        v_ids = [v_start] + [img_pad] * v_len + [v_end]
        token_ids.extend(label_ids + v_ids)
        token_tags.extend([TEXT_TAG] * len(label_ids) + [IMAGE_TAG] * len(v_ids))
      p_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:4096]
      token_ids.extend(p_ids)
      token_tags.extend([TEXT_TAG] * len(p_ids))
      mm_ids = np.asarray(processor.create_mm_token_type_ids([token_ids]), dtype=np.int64)
      pres = dict(
          ids=np.asarray(token_ids, dtype=np.int32),
          tags=np.asarray(token_tags, dtype=np.int64),
          pixel_values=vision["pixel_values"].float().numpy(),
          grid_thw=vision["image_grid_thw"].numpy(),
          mm_token_type_ids=mm_ids,
          image_pad_id=img_pad,
      )
      pos3 = r2v.text_positions(img_snap, pres["ids"], pres["grid_thw"], pres["mm_token_type_ids"])
    p_embeds = r2v.encode_text_with_vision(
        encoder, vision_model, pres, pos3, cfg_text, dtype,
        stop_at_layer=TEXT_ENCODER_LAYER, jit=True,
    )
    return p_embeds, pres["tags"], ref_imgs, [(im.height // 16 // 2 * 2, im.width // 16 // 2 * 2) for im in ref_imgs]

  def encode_ref_latents(ref_imgs, tgen):
    ref_latents = []
    mean5 = np.asarray(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    std5 = np.asarray(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    for im in ref_imgs:
      hc = im.height // 16 * 16
      wc = im.width // 16 * 16
      px = np.asarray(im, np.float32)[:hc, :wc].transpose(2, 0, 1)[None, :, None] / 255.0
      px = (px - mean5) / std5
      moments = jvae.tiled_encode(lambda t_: _ENC_ONE[0](enc_w, t_), jnp.asarray(px, dtype), mesh=mesh)
      moments_4d = np.asarray(moments[:, :, -1], np.float32)
      mu, logvar = np.split(moments_4d, 2, axis=1)
      sd = np.exp(0.5 * np.clip(logvar, -30.0, 20.0))
      with torchax.disable_temporarily():
        eps = randn_tensor(tuple(mu.shape), generator=tgen, dtype=torch.float32).numpy()
      lat = (mu + sd * eps).astype(np.float16).astype(np.float32)
      lat = lat[..., :lat.shape[-2] // 2 * 2, :lat.shape[-1] // 2 * 2]
      ref_latents.append((lat - lat_mean_np) / lat_std_np)
    return ref_latents

  with jax.set_mesh(mesh):
    prompt_embeds, text_tags, ref_imgs, ref_grids = encode_prompt_and_refs()
    jax.block_until_ready(prompt_embeds)

  (pos_np, tags_np, img_idx_np, txt_idx_np,
   n_ref, n_target) = build_image_packed_sequence(
      text_tags, ref_grids, latent_h, latent_w
  )
  seq = int(pos_np.shape[0])
  row_step_np = np.zeros((seq,), dtype=np.int64)
  if n_ref > 0:
    row_step_np[img_idx_np[:n_ref]] = 1

  position_ids_j = jax.device_put(jnp.asarray(pos_np, jnp.float32), replicated)
  token_tags_j = jax.device_put(jnp.asarray(tags_np, jnp.int64), replicated)
  image_idx_j = jax.device_put(jnp.asarray(img_idx_np, jnp.int64), replicated)
  text_idx_j = jax.device_put(jnp.asarray(txt_idx_np, jnp.int64), replicated)
  row_step_j = jax.device_put(jnp.asarray(row_step_np, jnp.int64), replicated)
  with torchax.disable_temporarily():
    token_tags_cpu = torch.from_numpy(tags_np.astype(np.int64))
    row_step_cpu = torch.from_numpy(row_step_np.astype(np.int64))

  @functools.partial(jax.jit, out_shardings=replicated)
  def euler_update(rows, dv, dt_scalar):
    gen = rows[:, n_ref:].astype(jnp.float32) + dt_scalar * dv[:, n_ref:].astype(jnp.float32)
    return jnp.concatenate([rows[:, :n_ref], gen], axis=1) if n_ref else gen

  @functools.partial(jax.jit, out_shardings=replicated)
  def unpatchify_and_denorm(img_rows):
    rows = img_rows[:, n_ref:] if n_ref else img_rows
    x = rows.reshape(1, latent_h // 2, latent_w // 2, 24, 2, 2)
    x = jnp.transpose(x, (0, 3, 1, 4, 2, 5)).reshape(1, 24, latent_h, latent_w)
    return (x.astype(jnp.float32) * lat_std_jnp + lat_mean_jnp).astype(dtype)

  def decode_image_2d(z_4d):
    px = jvae.tiled_decode(
        lambda t_: decode_tile_batch(dec_state, t_),
        z_4d, tile_px=256, overlap_px=64, mesh=mesh,
    )
    return to_uint8_rgb(px)

  def prepare_initial_rows():
    with torchax.disable_temporarily():
      tgen = torch.Generator().manual_seed(args.seed)
    ref_latents = encode_ref_latents(ref_imgs, tgen) if n_ref else []
    with torchax.disable_temporarily():
      target_noise = randn_tensor(
          (1, 24, 1, latent_h, latent_w), generator=tgen, dtype=torch.float32
      ).numpy()[:, :, 0]
    target_rows = patchify_2d(target_noise)
    if not n_ref:
      return jax.device_put(jnp.asarray(target_rows, jnp.float32), replicated)
    cond_list = []
    t0_val = np.float32(0.999)
    one_minus_t0 = np.float32(1.0) - t0_val
    for rlat in ref_latents:
      with torchax.disable_temporarily():
        c_noise = randn_tensor(tuple(rlat.shape), generator=tgen, dtype=torch.float32).numpy()
      noised = t0_val * rlat + one_minus_t0 * c_noise
      cond_list.append(patchify_2d(noised))
    all_rows = np.concatenate(cond_list + [target_rows], axis=1)
    return jax.device_put(jnp.asarray(all_rows, jnp.float32), replicated)

  t_vecs = [
      jax.device_put(
          jnp.asarray([timesteps_np[i], np.float32(0.999)] if n_ref else [timesteps_np[i]], jnp.float32),
          replicated,
      )
      for i in range(args.steps)
  ]
  dt_scalars = [
      jax.device_put(jnp.asarray(sigmas_np[i] - sigmas_np[i + 1], jnp.float32), replicated)
      for i in range(args.steps)
  ]

  with jax.set_mesh(mesh):
    if args.warmup:
      t_w = time.perf_counter()
      v0 = prepare_initial_rows()
      dv0 = tx_dit.forward_jax(
          {
              "hidden_states": v0,
              "encoder_hidden_states": prompt_embeds,
              "timestep": t_vecs[0],
              "timestep_indices": row_step_j,
              "token_tags": token_tags_j,
              "position_ids": position_ids_j,
              "image_indices": image_idx_j,
              "text_indices": text_idx_j,
          },
          token_tags_cpu,
          row_step_cpu,
          pdd_step_index=0,
      )
      v0 = euler_update(v0, dv0, dt_scalars[0])
      z0 = unpatchify_and_denorm(v0)
      u0 = decode_image_2d(z0)
      jax.block_until_ready(u0)
      print(f"  warmup {time.perf_counter() - t_w:.2f}s", flush=True)
      del v0, dv0, z0, u0

    u8_host = None
    for run_idx in range(args.runs):
      t_run = time.perf_counter()
      t_te0 = time.perf_counter()
      prompt_embeds, _, _, _ = encode_prompt_and_refs()
      jax.block_until_ready(prompt_embeds)
      t_te = time.perf_counter() - t_te0

      t_pipe0 = time.perf_counter()
      rows = prepare_initial_rows()
      jax.block_until_ready(rows)
      t_den0 = time.perf_counter()
      for i in range(args.steps):
        dv = tx_dit.forward_jax(
            {
                "hidden_states": rows,
                "encoder_hidden_states": prompt_embeds,
                "timestep": t_vecs[i],
                "timestep_indices": row_step_j,
                "token_tags": token_tags_j,
                "position_ids": position_ids_j,
                "image_indices": image_idx_j,
                "text_indices": text_idx_j,
            },
            token_tags_cpu,
            row_step_cpu,
            pdd_step_index=i,
        )
        rows = euler_update(rows, dv, dt_scalars[i])
      z_4d = unpatchify_and_denorm(rows)
      jax.block_until_ready(z_4d)
      t_den = time.perf_counter() - t_den0

      t_dec0 = time.perf_counter()
      u8_dev = decode_image_2d(z_4d)
      u8_host = np.asarray(u8_dev)
      t_dec = time.perf_counter() - t_dec0
      t_pipe = time.perf_counter() - t_pipe0
      t_total = time.perf_counter() - t_run
      print(
          f"run {run_idx}: e2e {t_total:.3f}s | pipeline (warm cond) {t_pipe:.3f}s "
          f"[text_encoder {t_te:.3f}s  denoise {t_den:.3f}s  vae_decode {t_dec:.3f}s]",
          flush=True,
      )

  if u8_host is not None:
    os.makedirs(args.out, exist_ok=True)
    with torchax.disable_temporarily():
      pil_img = Image.fromarray(u8_host)
      if (out_w, out_h) != (gen_w, gen_h):
        pil_img = VaeImageProcessor().resize(pil_img, height=out_h, width=out_w)
      out_path = os.path.join(args.out, f"{args.tag}.png")
      pil_img.save(out_path)
    np.save(os.path.join(args.out, f"{args.tag}.npy"), np.asarray(pil_img))
    print(f"wrote {out_path} ({pil_img.size[0]}x{pil_img.size[1]})", flush=True)


_ENC_ONE = [None]

if __name__ == "__main__":
  main()
