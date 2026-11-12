"""MiniMax-H3-Image (t2i and multi-reference edit) on torch_tpu.

Supports 8-way tensor parallelism across TPU devices (v7x-8 and v6e-8) for the
Qwen3-VL text encoder and the 50-layer MiniMaxH3ImageTransformer2DModel,
AOT-compiled 10-block DiT groups (`AotBlocks`), Pallas FlashAttention with
BHSD-first norm+RoPE (`ours_splash_tt`), fused segmented modulation
(`segmented_mod_tt`), and 8-rank data-parallel batched 2D ViT VAE tile
decoding over ICI.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time


def _setup_env():
  # Select the head runs of the segmented modulation with a mask instead of
  # slicing and `index_copy`ing them back: edit 2048x2048 denoise 1.43 ->
  # 1.14 s (a 2,079-row head). Video t2va stays sliced: no faster there (Pika
  # 768p 44.9 s either way) and its first request compiles 24 s longer.
  os.environ.setdefault("DT_SEGMOD_WHERE", "1")
  from .torchtpu_h3 import _auto_env
  _auto_env()
  _ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  _local = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1))))
  _premap = min(192, max(1, (_ram // 2 // _local) >> 30)) << 30
  os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(_premap))
  os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
  os.environ.setdefault("DT_SPMD_SAFE", "1")
  os.environ.setdefault("DT_OURS_BQ", "2048")
  os.environ.setdefault("DT_OURS_BKV", "4096")
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      "--xla_tpu_dvfs_p_state=7 "
      "--slicebuilder_use_insecure_grpc=true "
      "--xla_tpu_enable_latency_hiding_scheduler=true "
      "--xla_tpu_spmd_rng_bit_generator_unsafe=true "
      "--xla_tpu_enable_dot_strength_reduction=true",
  )


_setup_env()

PDD_SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
TEXT_TAG = 1
IMAGE_TAG = 0
CANVAS_MULTIPLE = 32
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3-Image")
  p.add_argument("--text-model-id", default="MiniMaxAI/MiniMax-H3")
  p.add_argument("--prompt", default=(
      "A cinematic close-up portrait of a red fox in a snowy pine forest at "
      "dawn, crisp frost on whiskers, golden rim light, ultra-detailed 4K."))
  p.add_argument("--reference", action="append", default=[],
                 help="Reference image path for editing; repeat for <Picture 2>, ...")
  p.add_argument("--size", default=None,
                 help="WIDTHxHEIGHT shorthand (e.g. 2048x2048, 4096x4096, 4736x3552)")
  p.add_argument("--width", type=int, default=2048)
  p.add_argument("--height", type=int, default=2048)
  p.add_argument("--min-generation-area", type=int,
                 default=DEFAULT_MIN_GENERATION_AREA)
  p.add_argument("--steps", type=int, default=6)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--text-layers", type=int, default=51)
  p.add_argument("--aot-group", type=int, default=10)
  p.add_argument("--aot-vae-group", type=int, default=12)
  p.add_argument("--vae-tile-batch", type=int, default=16)
  p.add_argument("--runs", type=int, default=2)
  p.add_argument("--profile", default="",
                 help="xprof directory: one trace per run after the first, to <dir>/run<N>/rank<R> "
                      "(rank 0 only unless DT_PROFILE_ALL_RANKS=1)")
  p.add_argument("--out", "--output", dest="out", default="out")
  p.add_argument("--tag", default="h3_image_torchtpu")
  args = p.parse_args()
  if args.size:
    w_str, h_str = args.size.lower().split("x")
    args.width, args.height = int(w_str), int(h_str)
  return args


def main():
  args = parse_args()

  import numpy as np
  import torch
  import torch_tpu  # noqa: F401
  from PIL import Image, ImageOps
  from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
  from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel
  from diffusers.image_processor import VaeImageProcessor
  from diffusers.utils.torch_utils import randn_tensor

  from . import aot_dit, ours_splash_tt, segmented_mod_tt, tp_h3
  from ..models.minimax_h3.jax_image_pipeline import (
      _resolve_snapshots,
      bound_short_edge,
      build_image_packed_sequence,
      patchify_2d,
      prepare_canvas,
  )

  device = torch.device("tpu")
  rank, world, all_reduce = tp_h3.init_parallel()
  if world > 1:
    device = torch.device("tpu", rank)
  say = (lambda *a, **k: print(*a, **k)) if rank == 0 else (lambda *a, **k: None)
  dtype = torch.bfloat16

  began = time.perf_counter()
  if rank == 0:
    img_snap, text_dir = _resolve_snapshots(args.model_id, args.text_model_id)
  if world > 1:
    import torch.distributed as dist
    dist.barrier()
  if rank != 0:
    img_snap, text_dir = _resolve_snapshots(args.model_id, args.text_model_id)

  vae_sub = "vae_diffusers" if os.path.isdir(os.path.join(img_snap, "vae_diffusers")) else "vae"
  dit_sub = (
      "transformer_diffusers"
      if os.path.isdir(os.path.join(img_snap, "transformer_diffusers"))
      else "transformer"
  )

  say(f"MiniMax-H3-Image torch_tpu on {world} rank(s), "
      f"workflow={'edit' if args.reference else 't2i'}", flush=True)

  # -- 1. Load & Shard Qwen3-VL Text Encoder (Resident in HBM) ---------------
  t0 = time.perf_counter()
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(img_snap, "tokenizer"))
  processor = AutoProcessor.from_pretrained(os.path.join(img_snap, "processor"))
  text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
      text_dir, torch_dtype=dtype, low_cpu_mem_usage=True
  ).eval()
  text_encoder.requires_grad_(False)
  # Truncate to 51 layers (layers 0..50) before sharding/moving to TPU
  for attr in ("model.language_model.layers", "language_model.layers", "model.layers"):
    node = text_encoder
    try:
      for part in attr.split("."):
        node = getattr(node, part)
      if len(node) > args.text_layers:
        del node[args.text_layers:]
      break
    except AttributeError:
      continue
  tp_h3.shard_text_encoder(text_encoder, rank, world, all_reduce, verbose=say)
  text_encoder.to(device)
  say(f"  text encoder loaded & sharded on TPU ({time.perf_counter() - t0:.1f}s)", flush=True)

  # -- 2. Load & Shard 50-layer DiT (Resident in HBM) ------------------------
  t0 = time.perf_counter()
  transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
      img_snap, subfolder=dit_sub, torch_dtype=dtype, low_cpu_mem_usage=True
  ).eval()
  transformer.requires_grad_(False)
  tp_h3.shard_transformer(
      transformer, rank, world, all_reduce,
      all_gather=getattr(tp_h3.init_parallel, "all_gather", None),
  )
  segmented_mod_tt.install(transformer, verbose=say)
  ours_splash_tt.install(verbose=say)
  transformer.to(device)
  dit_aot = aot_dit.AotBlocks(
      transformer.transformer_blocks, dynamic=False, group=args.aot_group, verbose=say
  ).install()
  say(f"  DiT loaded & sharded on TPU ({time.perf_counter() - t0:.1f}s)", flush=True)

  # -- 3. Load 2D Image VAE & Compile 24-layer ViT Decoder -------------------
  t0 = time.perf_counter()
  vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
      img_snap, subfolder=vae_sub, torch_dtype=dtype, low_cpu_mem_usage=True
  ).eval()
  vae.requires_grad_(False)
  vae.to(device)
  vae_aot = aot_dit.AotBlocks(
      vae.decoder.transformer_blocks, dynamic=False, group=args.aot_vae_group, verbose=say
  ).install()
  with open(os.path.join(img_snap, vae_sub, "config.json")) as f:
    vae_cfg = json.load(f)
  lat_mean = torch.tensor(vae_cfg["latents_mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
  lat_std = torch.tensor(vae_cfg["latents_std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
  px_mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
  px_std = torch.tensor(PIXEL_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
  say(f"  2D Image VAE loaded on TPU ({time.perf_counter() - t0:.1f}s)", flush=True)

  out_w, out_h, gen_w, gen_h, latent_h, latent_w = prepare_canvas(
      args.width, args.height, args.min_generation_area
  )
  say(f"  canvas: requested {out_w}x{out_h} -> generation {gen_w}x{gen_h} "
      f"(latent {latent_w}x{latent_h}, {(latent_h // 2) * (latent_w // 2)} image tokens)",
      flush=True)

  timesteps_np = np.asarray([1.0 - s for s in PDD_SAMPLE_SIGMAS[:-1]], dtype=np.float32)
  sigmas_np = np.concatenate([1.0 - timesteps_np, np.zeros((1,), dtype=np.float32)])

  from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

  # The text tower op by op cost 0.35-0.75 s a request. Compiled, with the
  # ids padded to a bucket of 256 so prompts of similar length share one
  # program: the tower is causal, so trailing pad tokens change nothing
  # before them, and the real rows are sliced back out.
  _text_fns = {}

  def compiled_text(token_ids):
    n = len(token_ids)
    bucket = -(-n // 256) * 256
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.tensor([token_ids + [pad_id] * (bucket - n)], dtype=torch.long, device=device)
    mm = torch.tensor(processor.create_mm_token_type_ids(ids.cpu().tolist()), dtype=torch.long,
                      device=device)
    fn = _text_fns.get(bucket)
    if fn is None:
      def text_fwd(ids_, mm_):
        out = text_encoder.model(input_ids=ids_, attention_mask=torch.ones_like(ids_),
                                 mm_token_type_ids=mm_, use_cache=False, output_hidden_states=True)
        return out.hidden_states[50]
      fn = torch.compile(text_fwd, backend="tpu", dynamic=False)
      _text_fns[bucket] = fn
    return fn(ids, mm)[:, :n].to(dtype)

  # Edit requests: the vision tower and the merge of its features into the
  # sequence scatter through boolean masks, which torch_tpu cannot trace, so
  # those run op by op. The 64-layer language model after them is compiled:
  # the deepstack features are scattered into dense (1, S, D) tensors here and
  # added inside the program instead of indexed in.
  _lm_fns = {}

  def _dense_deepstack(self_, hidden_states, visual_pos_masks, visual_embeds):
    if visual_embeds.shape == hidden_states.shape:
      return hidden_states + visual_embeds
    return _orig_deepstack(self_, hidden_states, visual_pos_masks, visual_embeds)

  lm = text_encoder.model.language_model
  _orig_deepstack = type(lm)._deepstack_process
  type(lm)._deepstack_process = _dense_deepstack

  def compiled_text_vision(token_ids, vision_kwargs, mark=lambda: None):
    m = text_encoder.model
    n = len(token_ids)
    bucket = -(-n // 256) * 256
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    pv = vision_kwargs["pixel_values"].to(device, text_encoder.dtype)
    grid = vision_kwargs["image_grid_thw"].to(device)
    embeds = m.get_input_embeddings()(ids)
    feats = m.get_image_features(pv, grid, return_dict=True)
    img = torch.cat(feats.pooler_output, dim=0).to(embeds.device, embeds.dtype)
    mark()
    mask, _ = m.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=img)
    embeds = embeds.masked_scatter(mask, img)
    pos = m.compute_3d_position_ids(input_ids=ids, image_grid_thw=grid, inputs_embeds=embeds,
                                    attention_mask=torch.ones_like(ids), mm_token_type_ids=mm)
    mask2 = mask[..., 0]
    deep = []
    for f in feats.deepstack_features:
      d = torch.zeros_like(embeds)
      d[mask2] = torch.cat(f, dim=0).to(d.dtype) if isinstance(f, (list, tuple)) else f.to(d.dtype)
      deep.append(d)
    pad = bucket - n
    if pad:
      embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad))
      deep = [torch.nn.functional.pad(d, (0, 0, 0, pad)) for d in deep]
      pos = torch.cat([pos, pos[..., -1:].expand(*pos.shape[:-1], pad)], dim=-1)
    mark()
    fn = _lm_fns.get((bucket, len(deep)))
    if fn is None:
      def lm_fwd(e_, p_, *d_):
        out = lm(input_ids=None, position_ids=p_, attention_mask=None, inputs_embeds=e_,
                 visual_pos_masks=None, deepstack_visual_embeds=list(d_),
                 use_cache=False, output_hidden_states=True)
        return out.hidden_states[50]
      fn = torch.compile(lm_fwd, backend="tpu", dynamic=False)
      _lm_fns[(bucket, len(deep))] = fn
    return fn(embeds, pos, *deep)[:, :n].to(dtype)

  def encode_prompt_and_refs():
    if not args.reference:
      token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:4096]
      tags = np.ones((len(token_ids),), dtype=np.int64)
      with torch.no_grad():
        if os.environ.get("DT_TEXT_COMPILE", "1") == "1":
          p_embeds = compiled_text(token_ids)
          if os.environ.get("DT_TEXT_CHECK") == "1":
            ref = get_qwen3vl_prompt_embeds(
                text_encoder, processor, token_ids, {},
                text_encoder_layer=50, device=device, dtype=dtype)
            d = (p_embeds.float() - ref.float()).abs()
            a_, b_ = p_embeds.float().flatten(), ref.float().flatten()
            say(f"  [text-check] cos {float((a_ @ b_) / (a_.norm() * b_.norm())):.6f}", flush=True)
            say(f"  [text-check] compiled vs eager: max|d| {float(d.max()):.3g}, "
                f"mean|d| {float(d.mean()):.3g}, |ref| {float(ref.float().abs().mean()):.3g}", flush=True)
        else:
          p_embeds = get_qwen3vl_prompt_embeds(
              text_encoder, processor, token_ids, {},
              text_encoder_layer=50, device=device, dtype=dtype
          )
      return p_embeds, tags, [], []

    _tt = [time.perf_counter()]
    def _mark():
      if os.environ.get("DT_TEXT_TIMING") == "1":
        torch.accelerator.synchronize()
      _tt.append(time.perf_counter())
    raw_imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in args.reference]
    ref_imgs = [bound_short_edge(im, 1024, 2048) for im in raw_imgs]
    vis_imgs = [bound_short_edge(im, 1024, 1024) for im in raw_imgs]
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
    _mark()
    vkw = {
        "pixel_values": vision["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw": vision["image_grid_thw"].to(device=device),
    }
    with torch.no_grad():
      if os.environ.get("DT_TEXT_COMPILE", "1") == "1":
        p_embeds = compiled_text_vision(token_ids, vkw, _mark)
        if os.environ.get("DT_TEXT_CHECK") == "1":
          ref = get_qwen3vl_prompt_embeds(
              text_encoder, processor, token_ids, vkw, text_encoder_layer=50, device=device, dtype=dtype)
          a_, b_ = p_embeds.float().flatten(), ref.float().flatten()
          say(f"  [text-check] cos {float((a_ @ b_) / (a_.norm() * b_.norm())):.6f}", flush=True)
      else:
        p_embeds = get_qwen3vl_prompt_embeds(
            text_encoder, processor, token_ids, vkw, text_encoder_layer=50, device=device, dtype=dtype)
    _mark()
    if os.environ.get("DT_TEXT_TIMING") == "1":
      say("  [text-timing] " + " ".join(f"{b - a:.3f}" for a, b in zip(_tt, _tt[1:])), flush=True)
    ref_grids = [(im.height // 16 // 2 * 2, im.width // 16 // 2 * 2) for im in ref_imgs]
    return p_embeds, np.asarray(token_tags, dtype=np.int64), ref_imgs, ref_grids

  # The tiled reference encode, compiled: all tiles (same size) as one batch
  # through one program instead of ~8,000 eager ops. GroupNorm and the
  # single-frame causal convs work per sample, so batching changes nothing.
  _enc_fns = {}

  def encode_moments(x):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    x = x.to(next(vae.encoder.parameters()).dtype)
    h, w = x.shape[-2:]
    ys, yl, yo = vae._split_tiles(h, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    xs, xl, xo = vae._split_tiles(w, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    if not vae.use_tiling or os.environ.get("DT_ENC_COMPILE", "1") != "1":
      return vae.encode(x, return_dict=False)[0]
    tiles = torch.cat([x[..., i:i + li, j:j + lj] for i, li in zip(ys, yl) for j, lj in zip(xs, xl)])
    key = tuple(tiles.shape)
    fn = _enc_fns.get(key)
    if fn is None:
      fn = torch.compile(lambda t: vae.quant_conv(vae.encoder(t[:, :, None]))[:, :, -1],
                         backend="tpu", dynamic=False)
      _enc_fns[key] = fn
    out = fn(tiles)
    ncol = len(xs)
    rows = [[out[r * ncol + c:r * ncol + c + 1] for c in range(ncol)] for r in range(len(ys))]
    ratio = vae.spatial_compression_ratio
    moments = vae._stitch_tiles(rows, [o // ratio for o in yo], [o // ratio for o in xo])
    return DiagonalGaussianDistribution(moments)

  def encode_ref_latents(ref_imgs, tgen):
    ref_latents = []
    for im in ref_imgs:
      hc = im.height // 16 * 16
      wc = im.width // 16 * 16
      px = torch.from_numpy(np.asarray(im, np.float32)[:hc, :wc]).to(device).permute(2, 0, 1)[None]
      px = (px.div(255.0) - px_mean) / px_std
      with torch.no_grad():
        posterior = encode_moments(px.to(dtype))
        if os.environ.get("DT_ENC_CHECK") == "1":
          ref_m = vae.encode(px.to(dtype), return_dict=False)[0].mean.float().flatten()
          got_m = posterior.mean.float().flatten()
          say(f"  [enc-check] compiled vs eager mean cos {float(got_m @ ref_m / got_m.norm() / ref_m.norm()):.6f} "
              f"max|d| {float((got_m - ref_m).abs().max()):.3g} |ref| {float(ref_m.abs().mean()):.3g}", flush=True)
      moments = posterior. parameters if hasattr(posterior, "parameters") else None
      # Draw eps on CPU generator for deterministic cross-rank noise
      mu = posterior.mean.detach().to("cpu", torch.float32)
      std = posterior.std.detach().to("cpu", torch.float32)
      eps = randn_tensor(mu.shape, generator=tgen, dtype=torch.float32)
      lat = (mu + std * eps).to(torch.float16).float().to(device)
      lat = lat[..., :lat.shape[-2] // 2 * 2, :lat.shape[-1] // 2 * 2]
      ref_latents.append(((lat - lat_mean) / lat_std).to("cpu").numpy())
    return ref_latents

  _stitch_cache = {}

  def sharded_decode_2d(z_4d: torch.Tensor) -> np.ndarray:
    """8-rank parallel batched 2D ViT tile decode + ICI all_gather + on-chip uint8."""
    ratio = vae.spatial_compression_ratio
    ys, yl, yo = vae._split_tiles(
        z_4d.shape[-2] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    xs, xl, xo = vae._split_tiles(
        z_4d.shape[-1] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    if not yo and not xo:
      with torch.no_grad():
        dec = vae._decode(z_4d)
    else:
      tiles = [
          z_4d[..., i_pos // ratio:i_pos // ratio + i_len // ratio,
               j_pos // ratio:j_pos // ratio + j_len // ratio]
          for i_pos, i_len in zip(ys, yl)
          for j_pos, j_len in zip(xs, xl)
      ]
      n_tiles = len(tiles)
      b_tile = args.vae_tile_batch
      per_rank = -(-n_tiles // world)
      # Round per_rank up to a multiple of b_tile (or keep per_rank if <= b_tile)
      if per_rank > b_tile:
        per_rank = -(-per_rank // b_tile) * b_tile
      total_padded = per_rank * world
      padded_tiles = tiles + [tiles[-1]] * (total_padded - n_tiles)
      my_tiles = torch.cat(padded_tiles[rank * per_rank:(rank + 1) * per_rank], dim=0)
      out_chunks = []
      with torch.no_grad():
        for s in range(0, per_rank, b_tile):
          chunk = my_tiles[s:s + b_tile]
          out_chunks.append(vae.decoder(vae.post_quant_conv(chunk)))
      my_dec = torch.cat(out_chunks, dim=0) if len(out_chunks) > 1 else out_chunks[0]
      if world > 1:
        all_dec = tp_h3.init_parallel.all_gather(my_dec, dim=0)[:n_tiles]
      else:
        all_dec = my_dec[:n_tiles]
      key = (tuple(all_dec.shape), tuple(yo), tuple(xo), len(ys), len(xs))
      fn = _stitch_cache.get(key)
      if fn is None:
        # Blending a few hundred tiles op by op cost 0.66 s at 4736x3552; the
        # stitch and the uint8 conversion are one program per canvas size.
        ncol, nrow = len(xs), len(ys)

        def stitch_u8(tiles_):
          rows = [[tiles_[i * ncol + j:i * ncol + j + 1] for j in range(ncol)]
                  for i in range(nrow)]
          return to_u8(vae._stitch_tiles(rows, yo, xo))
        fn = torch.compile(stitch_u8, backend="tpu", fullgraph=True, dynamic=False)
        _stitch_cache[key] = fn
      with torch.no_grad():
        u8 = fn(all_dec)
      torch.accelerator.synchronize()
      return u8.to("cpu").numpy()
    u8 = to_u8(dec)
    torch.accelerator.synchronize()
    return u8.to("cpu").numpy()

  def to_u8(dec):
    rgb = (dec.float() * px_std + px_mean).clamp(0.0, 1.0)
    return (rgb * 255.0).clamp(0.0, 255.0).to(torch.uint8)[0].permute(1, 2, 0).contiguous()

  # Initial prompt encode & sequence packing
  prompt_embeds, text_tags, ref_imgs, ref_grids = encode_prompt_and_refs()
  (pos_np, tags_np, img_idx_np, txt_idx_np,
   n_ref, n_target) = build_image_packed_sequence(
      text_tags, ref_grids, latent_h, latent_w
  )
  seq = int(pos_np.shape[0])
  row_step_np = np.zeros((seq,), dtype=np.int64)
  if n_ref > 0:
    row_step_np[img_idx_np[:n_ref]] = 1

  position_ids = torch.from_numpy(pos_np).to(device=device, dtype=torch.float32)
  token_tags = torch.from_numpy(tags_np.astype(np.int64)).to(device=device)
  image_indices = torch.from_numpy(img_idx_np.astype(np.int64)).to(device=device)
  text_indices = torch.from_numpy(txt_idx_np.astype(np.int64)).to(device=device)
  timestep_indices = torch.from_numpy(row_step_np).to(device=device)

  def prepare_initial_rows():
    tgen = torch.Generator().manual_seed(args.seed)
    ref_latents = encode_ref_latents(ref_imgs, tgen) if n_ref else []
    target_noise = randn_tensor(
        (1, 24, 1, latent_h, latent_w), generator=tgen, dtype=torch.float32
    ).numpy()[:, :, 0]
    target_rows = patchify_2d(target_noise)
    if not n_ref:
      return torch.from_numpy(target_rows).to(device=device, dtype=torch.float32)
    cond_list = []
    t0_val = np.float32(0.999)
    one_minus_t0 = np.float32(1.0) - t0_val
    for rlat in ref_latents:
      c_noise = randn_tensor(tuple(rlat.shape), generator=tgen, dtype=torch.float32).numpy()
      noised = t0_val * rlat + one_minus_t0 * c_noise
      cond_list.append(patchify_2d(noised))
    all_rows = np.concatenate(cond_list + [target_rows], axis=1)
    return torch.from_numpy(all_rows).to(device=device, dtype=torch.float32)

  t_vecs = [
      torch.tensor(
          [timesteps_np[i], np.float32(0.999)] if n_ref else [timesteps_np[i]],
          dtype=torch.float32,
          device=device,
      )
      for i in range(args.steps)
  ]
  dt_scalars = [
      float(sigmas_np[i] - sigmas_np[i + 1]) for i in range(args.steps)
  ]

  u8_host = None
  for run_idx in range(args.runs):
    prof_ctx = None
    if args.profile and (rank == 0 or os.environ.get("DT_PROFILE_ALL_RANKS") == "1") and run_idx >= 1:
      from torch_tpu._internal import profiler as tt_prof
      prof_ctx = tt_prof.profile(
          activities=[tt_prof.ProfilerActivity.CPU, tt_prof.ProfilerActivity.TPU],
          on_trace_ready=tt_prof.xprof_trace_handler(dir_name=f"{args.profile}/run{run_idx}/rank{rank}"),
      )
      prof_ctx.__enter__()
    torch.accelerator.synchronize()
    t_run = time.perf_counter()

    t_te0 = time.perf_counter()
    prompt_embeds, _, _, _ = encode_prompt_and_refs()
    torch.accelerator.synchronize()
    t_te = time.perf_counter() - t_te0

    t_pipe0 = time.perf_counter()
    rows = prepare_initial_rows()
    torch.accelerator.synchronize()
    t_prep = time.perf_counter() - t_pipe0

    t_den0 = time.perf_counter()
    with torch.no_grad():
      for i in range(args.steps):
        dv = transformer(
            hidden_states=rows,
            encoder_hidden_states=prompt_embeds,
            timestep=t_vecs[i],
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            image_indices=image_indices,
            text_indices=text_indices,
            pdd_step_index=i,
            return_dict=False,
        )[0]
        rows[:, n_ref:] = rows[:, n_ref:].float() + dt_scalars[i] * dv[:, n_ref:].float()
      gen_rows = rows[:, n_ref:] if n_ref else rows
      z_4d = (
          gen_rows.reshape(1, latent_h // 2, latent_w // 2, 24, 2, 2)
          .permute(0, 3, 1, 4, 2, 5)
          .reshape(1, 24, latent_h, latent_w)
      )
      z_4d = (z_4d.float() * lat_std + lat_mean).to(dtype)
    torch.accelerator.synchronize()
    t_den = time.perf_counter() - t_den0

    t_dec0 = time.perf_counter()
    u8_host = sharded_decode_2d(z_4d)
    t_dec = time.perf_counter() - t_dec0
    t_pipe = time.perf_counter() - t_pipe0
    t_total = time.perf_counter() - t_run
    say(
        f"run {run_idx}: e2e {t_total:.3f}s | pipeline (warm cond) {t_pipe:.3f}s "
        f"[text_encoder {t_te:.3f}s  prepare {t_prep:.3f}s  denoise {t_den:.3f}s  vae_decode {t_dec:.3f}s]",
        flush=True,
    )
    if prof_ctx is not None:
      prof_ctx.__exit__(None, None, None)

  if rank == 0 and u8_host is not None:
    os.makedirs(args.out, exist_ok=True)
    pil_img = Image.fromarray(u8_host)
    if (out_w, out_h) != (gen_w, gen_h):
      pil_img = VaeImageProcessor().resize(pil_img, height=out_h, width=out_w)
    out_path = os.path.join(args.out, f"{args.tag}.png")
    pil_img.save(out_path)
    np.save(os.path.join(args.out, f"{args.tag}.npy"), np.asarray(pil_img))
    say(f"wrote {out_path} ({pil_img.size[0]}x{pil_img.size[1]})", flush=True)


if __name__ == "__main__":
  main()
