"""MiniMax-H3-Image (t2i and multi-reference edit) end to end in native JAX.

Reuses the 50-layer MiniMax-H3 DiT (`jax_dit.py`, with `audio_in_channels=0`
and `num_pdd_heads=6`), the Qwen3-VL text and vision towers (`jax_text.py`,
`jax_vision.py`, `jax_ref2va.py`), and the 2D ViT image decoder + 3D causal
encoder in `jax_vae.py`.

All six PDD steps share one compiled XLA graph (with `pdd_step` passed as a
traced int32 scalar indexing the 6 output heads and the fused Euler velocity
update inside the jitted step), followed by an 8-way tile-parallel 2D ViT
decode and on-chip uint8 conversion.
"""

import argparse
import functools
import glob
import json
import math
import os
import time

import numpy as np

TEXT_ENCODER_LAYER = 51  # layers[:51] -> output of 50th decoder layer
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
  p.add_argument("--text-model-id", default="MiniMaxAI/MiniMax-H3",
                 help="Fallback repo for text_encoder/ if not in --model-id")
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
                 default=DEFAULT_MIN_GENERATION_AREA,
                 help="Minimum generation canvas area in pixels (default 2048^2; 0 disables)")
  p.add_argument("--steps", type=int, default=6)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--chips", type=int, default=0,
                 help="Number of TPU devices to use (default: all available)")
  p.add_argument("--offload", action="store_true",
                 help="Stream weights from pinned host memory (for 16 GiB v5e)")
  p.add_argument("--text-offload", action="store_true",
                 help="Keep text encoder in pinned host memory")
  p.add_argument("--adaln-host", action="store_true",
                 help="Precompute the 6 PDD timestep modulations on host and drop 26 GB of adaln weights")
  p.add_argument("--fp8", action="store_true",
                 help="Quantize the 6 DiT block linears to float8_e4m3fn on chip")
  p.add_argument("--qk-fp8", action="store_true",
                 help="Cast Q and K to float8_e4m3fn entering Pallas FlashAttention")
  p.add_argument("--int8-weights", action="store_true",
                 help="Quantize DiT and text encoder weights to int8 on load")
  p.add_argument("--free-text-after", action="store_true",
                 help="encode the prompt once, then free the text encoder before the DiT loads "
                      "(text-to-image; small hosts such as a v5e-1); later runs reuse the embedding")
  p.add_argument("--text-int8", action="store_true",
                 help="Quantize only the text encoder to int8 (and keep its first 50 layers), "
                      "so it stays on chip beside a bf16 DiT on 4 v6e chips")
  p.add_argument("--bf16-weights", action="store_true",
                 help="Round float32 input/output projections to bfloat16 on load")
  p.add_argument("--full-lane-rope", action="store_true", default=True,
                 help="Use 128-lane matrix multiply for RoPE")
  p.add_argument("--pad-sequence-to", type=int,
                 default=0 if os.environ.get("DT_STREAM_SHARD") == "whole" else 128,
                 help="Pad packed sequence length up to a multiple of this")
  p.add_argument("--block-q", type=int, default=2048)
  p.add_argument("--block-kv", type=int, default=4096)
  p.add_argument("--block-kv-compute", type=int, default=0)
  p.add_argument("--dense-attention", action="store_true")
  p.add_argument("--segments", action="store_true", default=True,
                 help="Use segment-wise AdaLN modulation in JAX DiT")
  p.add_argument("--vae-tile-batch", type=int, default=0,
                 help="Batch size for tiled 2D VAE decode when weights are TP-sharded")
  p.add_argument("--decode-tile-shard", action="store_true", default=True,
                 help="Replicate the 1.2 GB 2D ViT decoder and shard spatial tiles across devices")
  p.add_argument("--warmup", action="store_true",
                 help="Run one warmup pass before timed runs")
  p.add_argument("--runs", type=int, default=2,
                 help="Number of timed generation runs")
  p.add_argument("--requests", default="",
                 help="JSONL of text-to-image requests ({\"prompt\": ..., \"seed\": ...}) served "
                      "after the timed runs with the models resident, on the same canvas")
  p.add_argument("--text-bucket", type=int, default=64,
                 help="with --requests: pad the prompt to a multiple of this many tokens so "
                      "prompts in one bucket share one compiled program (attention masks the pads)")
  p.add_argument("--compile-cache", default="")
  p.add_argument("--cache-only", action="store_true")
  p.add_argument("--out", "--output", dest="out", default="out")
  p.add_argument("--tag", default="h3_image_jax")
  args = p.parse_args()
  if args.size:
    w_str, h_str = args.size.lower().split("x")
    args.width, args.height = int(w_str), int(h_str)
  return args


def _resolve_snapshots(model_id: str, text_model_id: str):
  """Locate or download MiniMax-H3-Image and the shared MiniMax-H3 text_encoder."""
  from huggingface_hub import snapshot_download
  from ...hub import local_snapshot

  if os.path.isdir(model_id):
    img_snap = model_id
  else:
    try:
      img_snap = local_snapshot(model_id)
    except FileNotFoundError:
      img_snap = snapshot_download(
          model_id,
          allow_patterns=[
              "*.json",
              "tokenizer/*",
              "processor/*",
              "scheduler/*",
              "transformer_diffusers/*",
              "vae_diffusers/*",
          ],
      )
  text_dir = os.path.join(img_snap, "text_encoder")
  if not os.path.isdir(text_dir) or not glob.glob(os.path.join(text_dir, "*.safetensors")):
    if os.path.isdir(text_model_id):
      base_snap = text_model_id
    else:
      try:
        base_snap = local_snapshot(text_model_id)
      except FileNotFoundError:
        base_snap = snapshot_download(
            text_model_id,
            allow_patterns=["text_encoder/*", "tokenizer/*", "processor/*"],
        )
    text_dir = os.path.join(base_snap, "text_encoder")
  return img_snap, text_dir


def bound_short_edge(image, min_short_edge: int, max_short_edge: int,
                     multiple: int = CANVAS_MULTIPLE):
  """Match `MiniMaxH3ImageEditResizeStep.bound_short_edge`."""
  from diffusers.image_processor import VaeImageProcessor

  width, height = image.size
  short = min(width, height)
  if min_short_edge <= short <= max_short_edge:
    return image
  target = min_short_edge if short < min_short_edge else max_short_edge
  scale = target / short
  tw = max(multiple, round(width * scale / multiple) * multiple)
  th = max(multiple, round(height * scale / multiple) * multiple)
  if (tw, th) != (width, height):
    image = VaeImageProcessor().resize(image, height=th, width=tw)
  return image


def prepare_canvas(width: int, height: int, min_area: int):
  """Match `MiniMaxH3ImagePrepareCanvasStep`."""
  out_w, out_h = width, height
  if min_area and min_area > 0:
    scale = max(1.0, math.sqrt(min_area / (width * height)))
  else:
    scale = 1.0
  gen_h = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
  gen_w = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
  return out_w, out_h, gen_w, gen_h, gen_h // 16, gen_w // 16


def _frame_position_grid(latent_h: int, latent_w: int,
                         patch_h: int = 2, patch_w: int = 2) -> np.ndarray:
  """Match `_frame_position_grid` in `minimax_h3_image/before_denoise.py`."""
  sqrt_area = np.sqrt(latent_h * latent_w)
  ranges = []
  for size, patch in ((latent_h, patch_h), (latent_w, patch_w)):
    ratio = size / sqrt_area
    left = (1.0 - ratio) / 2.0
    ranges.append(
        np.linspace(left, left + ratio, size // patch, endpoint=False, dtype=np.float64) * 32.0
    )
  grid = np.meshgrid(*ranges, indexing="ij")
  return np.stack(grid, axis=-1).reshape(-1, 2)


def build_image_packed_sequence(text_token_tags: np.ndarray,
                                reference_grids: list[tuple[int, int]],
                                latent_height: int,
                                latent_width: int):
  """Match `build_image_packed_sequence` in `minimax_h3_image/before_denoise.py`."""
  num_text = int(len(text_token_tags))
  ref_row_counts = [(h // 2) * (w // 2) for h, w in reference_grids]
  num_ref_rows = sum(ref_row_counts)
  num_target_rows = (latent_height // 2) * (latent_width // 2)
  seq_len = num_text + num_ref_rows + num_target_rows

  position_ids = np.zeros((seq_len, 3), dtype=np.float64)
  position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)

  cursor = num_text
  for k, ((h_k, w_k), n_k) in enumerate(zip(reference_grids, ref_row_counts)):
    position_ids[cursor:cursor + n_k, 0] = float(num_text + k)
    position_ids[cursor:cursor + n_k, 1:] = _frame_position_grid(h_k, w_k, 2, 2)
    cursor += n_k

  position_ids[cursor:, 0] = float(num_text + len(reference_grids))
  position_ids[cursor:, 1:] = _frame_position_grid(latent_height, latent_width, 2, 2)

  text_indices = np.arange(num_text, dtype=np.int32)
  image_indices = np.arange(num_text, seq_len, dtype=np.int32)
  token_tags = np.zeros((seq_len,), dtype=np.int32)
  token_tags[:num_text] = np.asarray(text_token_tags, dtype=np.int32)
  token_tags[num_text:] = IMAGE_TAG
  return (
      position_ids.astype(np.float32),
      token_tags,
      image_indices,
      text_indices,
      num_ref_rows,
      num_target_rows,
  )


def patchify_2d(latents: np.ndarray) -> np.ndarray:
  """`(1, 24, H_lat, W_lat)` -> `(1, (H_lat//2)*(W_lat//2), 96)`."""
  b, c, h, w = latents.shape
  x = latents.reshape(b, c, h // 2, 2, w // 2, 2)
  x = x.transpose(0, 2, 4, 1, 3, 5)
  return x.reshape(b, (h // 2) * (w // 2), c * 4)


def main():
  args = parse_args()
  # Modulate every head run (the text, an edit's reference rows) by selecting
  # it in place: the 53-row text head written back with a dynamic-update-slice
  # split the gated residual and norm into three full-sequence passes a block
  # (4736x3552 denoise 2.95 -> 2.68 s). jax_dit's default (512) is for video.
  os.environ.setdefault("DT_SEGMOD_SELECT_MIN", "1")
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      "--xla_tpu_dvfs_p_state=7 "
      "--xla_tpu_enable_latency_hiding_scheduler=true "
      "--xla_tpu_enable_async_collective_fusion=true "
      "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
      "--xla_tpu_overlap_compute_collective_tc=true "
      "--xla_enable_async_all_gather=true "
      "--xla_enable_async_collective_permute=true "
      "--xla_tpu_scoped_vmem_limit_kib=65536",
  )
  if args.chips and os.environ.get("DT_VISIBLE_CHIPS") == "1":
    os.environ.setdefault("TPU_VISIBLE_DEVICES",
                          ",".join(str(i) for i in range(args.chips)))

  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import NamedSharding, PartitionSpec as P

  if args.compile_cache:
    jax.config.update("jax_compilation_cache_dir", args.compile_cache)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

  os.environ["DT_PAD_SEQUENCE_TO"] = os.environ.get(
      "DT_PAD_SEQUENCE_TO", str(args.pad_sequence_to))
  os.environ["DT_FULL_LANE_ROPE"] = os.environ.get(
      "DT_FULL_LANE_ROPE", "1" if args.full_lane_rope else "0")

  from . import jax_dit as dit
  from . import jax_ref2va as r2v
  from . import jax_text as jtext
  from . import jax_vae as jvae
  from . import jax_vision as jvision
  from ..qwen3_vl import jax_text as qwen_text
  from ..qwen3_vl import jax_vision as _qv
  from ...safetensors_io import safe_open

  devices = jax.devices()[:args.chips or None]
  mesh = jax.make_mesh(
      (len(devices),), ("tp",),
      devices=devices,
      axis_types=(jax.sharding.AxisType.Auto,),
  )
  dtype = jnp.bfloat16
  replicated = NamedSharding(mesh, P())
  dit_host = "pinned_host" if args.offload else None
  text_host = "pinned_host" if (args.offload or args.text_offload) else None

  began = time.perf_counter()
  img_snap, text_dir = _resolve_snapshots(args.model_id, args.text_model_id)
  vae_dir = os.path.join(
      img_snap,
      "vae_diffusers" if os.path.isdir(os.path.join(img_snap, "vae_diffusers")) else "vae",
  )
  dit_dir = os.path.join(
      img_snap,
      "transformer_diffusers"
      if os.path.isdir(os.path.join(img_snap, "transformer_diffusers"))
      else "transformer",
  )
  with open(os.path.join(vae_dir, "config.json")) as f:
    vae_cfg = json.load(f)
  lat_mean_np = np.asarray(vae_cfg["latents_mean"], np.float32).reshape(1, -1, 1, 1)
  lat_std_np = np.asarray(vae_cfg["latents_std"], np.float32).reshape(1, -1, 1, 1)
  lat_mean_jnp = jnp.asarray(lat_mean_np)
  lat_std_jnp = jnp.asarray(lat_std_np)

  print(f"MiniMax-H3-Image JAX on {len(devices)} device(s) "
        f"({devices[0].device_kind}), workflow={'edit' if args.reference else 't2i'}",
        flush=True)

  # -- 1. Load Text & Vision Towers ------------------------------------------
  from transformers import AutoProcessor, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(os.path.join(img_snap, "tokenizer"))
  processor = AutoProcessor.from_pretrained(os.path.join(img_snap, "processor"))
  encoder, cfg_text = jtext.build(text_dir, dtype=dtype, abstract=True)
  _wc = os.environ.get("DT_WEIGHT_CACHE")
  _tc = os.path.join(_wc, "text" + ("_int8" if (args.int8_weights or args.text_int8) else "")) if _wc else None
  text_int8 = args.int8_weights or args.text_int8
  if _tc and os.path.exists(os.path.join(_tc, "index.json")):
    if text_int8:
      encoder.layers = nnx.List(list(encoder.layers)[:TEXT_ENCODER_LAYER])
    dit.load_param_cache(encoder, _tc, mesh,
                         lambda n, nd: jtext.param_spec(n.split("."), nd),
                         memory_kind=text_host)
    print(f"  [weight-cache] text tower from {_tc}", flush=True)
  else:
    if _tc:
      dit.CACHE_WRITE["dir"] = _tc
    jtext.load(encoder, text_dir, mesh=mesh, spec_fn=jtext.param_spec,
               memory_kind=text_host, int8=text_int8,
               keep_layers=TEXT_ENCODER_LAYER if text_int8 else None)
    if _tc:
      dit.cache_finish()
  print(f"  text encoder loaded ({time.perf_counter() - began:.1f}s)", flush=True)
  # Compile text-only encoder jit
  _text_graph, _text_state = nnx.split(encoder)
  # With the tower in pinned host memory (`--text-offload`), copy it in
  # inside the jit: a `pinned_host` operand makes `gather`/`dot_general`
  # raise rather than transfer, and inside the program XLA overlaps each
  # layer's copy with the previous layer's compute (as in `jax_pipeline`).
  _text_dev = (jax.tree.map(lambda a: a.sharding.with_memory_kind("device"), _text_state)
               if text_host is not None else None)

  def _text_fwd(st, i, c, s):
    if _text_dev is not None:
      st = jax.tree.map(jax.device_put, st, _text_dev)
    return nnx.merge(_text_graph, st)(i, c, s, stop_at_layer=TEXT_ENCODER_LAYER)

  _text_jit_raw = jax.jit(_text_fwd)
  _text_jit = lambda _m, i, c, s: _text_jit_raw(_text_state, i, c, s)

  def encode_t2i_prompt():
    token_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:4096]
    tags = np.ones((len(token_ids),), dtype=np.int64)
    ids_np = np.asarray([token_ids], dtype=np.int32)
    pos = np.broadcast_to(np.arange(ids_np.shape[1])[None, :], (3, ids_np.shape[1]))
    cos, sin = qwen_text.mrope_cos_sin(
        pos, head_dim=cfg_text["head_dim"], rope_theta=cfg_text["rope_theta"],
        mrope_section=cfg_text["mrope_section"])
    cos_j = jax.device_put(jnp.asarray(cos, dtype), replicated)
    sin_j = jax.device_put(jnp.asarray(sin, dtype), replicated)
    ids_j = jax.device_put(jnp.asarray(ids_np, jnp.int32), replicated)
    return _text_jit(encoder, ids_j, cos_j, sin_j), tags, [], []

  cached_text = None
  if args.free_text_after and not args.reference:
    # One prompt, then drop the tower: on a 47 GB host (v5e-1) the int8 text
    # tower and the int8 DiT do not fit in host memory together.
    cached_text = encode_t2i_prompt()
    jax.block_until_ready(cached_text[0])
    del encoder, _text_state, _text_jit_raw, _text_jit
    encoder = None
    jax.clear_caches()
    from .jax_pipeline import _release_host_memory
    _release_host_memory()
    print("  text encoder freed after encoding the prompt", flush=True)

  vision_model = None
  if args.reference:
    vision_model, _ = jvision.build(text_dir, dtype=dtype, abstract=True)
    jvision.load(vision_model, text_dir, mesh=mesh, spec_fn=lambda k, n: P(),
                 memory_kind=text_host if len(devices) == 1 else None)
    if os.environ.get("DT_VISION_HEAD_SHARD", "1") == "1" and len(devices) > 1:
      _qv.HEAD_SHARD_AXIS = "tp"

  # -- 2. Load 50-layer DiT --------------------------------------------------
  t_dit = time.perf_counter()
  timesteps_np = np.asarray(
      [1.0 - s for s in PDD_SAMPLE_SIGMAS[:-1]], dtype=np.float32
  )
  sigmas_np = np.concatenate([1.0 - timesteps_np, np.zeros((1,), dtype=np.float32)])

  _, Transformer = dit._build_modules()
  transformer = nnx.eval_shape(
      lambda: Transformer(**dit.IMAGE_CONFIG_FULL, dtype=dtype, rngs=nnx.Rngs(0))
  )
  _levels = None
  if args.adaln_host:
    if args.reference:
      _levels = [[float(t), float(np.float32(0.999))] for t in timesteps_np]
    else:
      _levels = [[float(t)] for t in timesteps_np]
  _dc = None
  if _wc:
    _dc = os.path.join(_wc, "dit_h3_image_" + "_".join(filter(None, [
        "edit" if args.reference else "t2i",
        "int8" if args.int8_weights else "",
        f"adaln{len(_levels)}" if _levels is not None else "",
        "bf16w" if args.bf16_weights else ""])))
  adaln_mods_host = None
  if _dc and os.path.exists(os.path.join(_dc, "index.json")):
    _extra = dit.load_param_cache(transformer, _dc, mesh,
                                  lambda n, nd: dit._spec_for(n),
                                  memory_kind=dit_host)
    adaln_mods_host = _extra.get("adaln")
    print(f"  [weight-cache] DiT from {_dc}", flush=True)
  else:
    if _dc:
      dit.CACHE_WRITE["dir"] = _dc
    adaln_temb = dit.host_temb(dit_dir, _levels) if _levels is not None else None
    dit.load_transformer(transformer, dit_dir, mesh=mesh,
                         memory_kind=dit_host, round_bf16=args.bf16_weights,
                         int8=args.int8_weights, adaln_temb=adaln_temb)
    if args.adaln_host:
      adaln_mods_host = dit.ADALN_MODS.pop("last")
    if _dc:
      dit.cache_finish(
          extra={"adaln": adaln_mods_host} if adaln_mods_host is not None else None
      )
  if args.fp8 and not args.int8_weights:
    with jax.set_mesh(mesh):
      dit.quantize_transformer_fp8(transformer, mesh=mesh)
    print("  [fp8] quantized DiT block linears to float8_e4m3fn", flush=True)
  if os.environ.get("DT_SWIGLU_INTERLEAVE", "1" if os.environ.get("DT_STREAM_SHARD") == "whole" else "0") == "1":
    # Each chip gets matching (value, gate) rows of `ff.proj`, so the SwiGLU
    # split is local instead of an exchange across the mesh every block.
    with jax.set_mesh(mesh):
      dit.interleave_swiglu_weights(transformer, mesh=mesh)
    print("  [swiglu] ff.proj rows interleaved per chip", flush=True)
  adaln_mods = None
  if adaln_mods_host is not None:
    # Put precomputed (6, 50, levels, 6*5376*3) modulations in HBM replicated
    adaln_mods = [
        jax.device_put(jnp.asarray(m, dtype), replicated)
        for m in adaln_mods_host
    ]
  print(f"  DiT loaded ({time.perf_counter() - t_dit:.1f}s)", flush=True)

  # -- 3. Load 2D Image VAE (Decoder + optional Encoder for edit) ------------
  t_vae = time.perf_counter()
  decoder = nnx.eval_shape(
      lambda: jvae.ViTDecoder(**jvae.IMAGE_DECODER_CONFIG, dtype=dtype, rngs=nnx.Rngs(0))
  )
  post_q = jvae.PostQuantConv(jvae.IMAGE_DECODER_CONFIG["in_channels"],
                              dtype=dtype, rngs=nnx.Rngs(0))
  vae_encoder = quant_conv = None
  if args.reference:
    vae_encoder = nnx.eval_shape(
        lambda: jvae.Encoder3d(dtype=dtype, rngs=nnx.Rngs(0))
    )
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
  # Replicate the 1.2 GB 2D ViT decoder so spatial tiles can be sharded 8-way across devices
  dec_spec = (lambda _n, _d: P()) if args.decode_tile_shard else jvae.decoder_spec
  jvae.load_decoder(decoder, dec_params, mesh=mesh, spec_for=dec_spec)
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

    @functools.partial(jax.jit, out_shardings=replicated)
    def _enc_one_tile(w, tile):
      with jax.default_matmul_precision("highest"):
        e_mod = nnx.merge(eg_vae, w[0])
        q_mod = nnx.merge(qg_vae, w[1])
        return jvae.encode_video(e_mod, q_mod, tile)

    _ENC_ONE[0] = _enc_one_tile

  print(f"  2D Image VAE loaded ({time.perf_counter() - t_vae:.1f}s)", flush=True)

  # -- 4. Configure Pallas FlashAttention & Compile DiT Step -----------------
  sdpa = None
  if not args.dense_attention:
    from ...kernels import mesh_attention
    from ...kernels import splash as csa

    local_attn = functools.partial(
        csa.attention,
        scale=None,
        block_q=args.block_q,
        block_kv=args.block_kv,
        block_kv_compute=args.block_kv_compute or args.block_kv,
        use_fixed_m=True,
        qk_fp8=args.qk_fp8 or os.environ.get("DT_QK_FP8") == "1",
    )

    # With the residual stream whole (`DT_STREAM_SHARD=whole`, the default in
    # run_h3_image_jax.sh), leave the attention output split by heads:
    # `to_out` contracts over the heads, so its partial sums meet in one
    # all-reduce. `_on_mesh` constrains the output to the sequence layout,
    # which costs a heads-to-sequence exchange and a gather on every block.
    keep_heads = os.environ.get("DT_STREAM_SHARD") == "whole" and os.environ.get(
        "DT_ATTN_KEEP_HEADS", "1") == "1"

    def sdpa(q, k, v, kv_len=None):
      k = k - jnp.mean(k, axis=2, keepdims=True)
      with jax.default_matmul_precision("default"):
        if keep_heads and q.shape[1] % mesh.size == 0:
          qs, kvs, _ = mesh_attention._partition_specs(mesh, q.shape, True)
          if kv_len is not None:
            # Bucketed prompts (`--requests`): keys past the true length masked.
            return jax.shard_map(lambda a, b, c, n: local_attn(a, b, c, kv_len=n), mesh=mesh,
                                 in_specs=(qs, kvs, kvs, P()), out_specs=qs,
                                 check_vma=False)(q, k, v, kv_len)
          return jax.shard_map(local_attn, mesh=mesh, in_specs=(qs, kvs, kvs),
                               out_specs=qs, check_vma=False)(q, k, v)
        if kv_len is not None:
          raise NotImplementedError("--requests needs DT_STREAM_SHARD=whole (keep-heads attention)")
        return mesh_attention._on_mesh(local_attn, mesh, q, k, v, True)

  out_w, out_h, gen_w, gen_h, latent_h, latent_w = prepare_canvas(
      args.width, args.height, args.min_generation_area
  )
  print(f"  canvas: requested {out_w}x{out_h} -> generation {gen_w}x{gen_h} "
        f"(latent {latent_w}x{latent_h}, {(latent_h // 2) * (latent_w // 2)} image tokens)",
        flush=True)


  def encode_prompt_and_refs(pad_to=0):
    """Run text (+ optional vision + reference VAE) encoding.

    `pad_to` (`--requests`): append pad tokens up to that many; the tower is
    causal, so the real rows are unchanged and the shape is the bucket's."""
    if not args.reference:
      return cached_text if cached_text is not None else encode_t2i_prompt()

    from PIL import Image, ImageOps
    raw_imgs = [
        ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        for p in args.reference
    ]
    ref_imgs = [bound_short_edge(im, 1024, 2048) for im in raw_imgs]
    vis_imgs = [bound_short_edge(im, 1024, 1024) for im in raw_imgs]
    vision = processor.image_processor(images=vis_imgs, return_tensors="pt")
    merge_size = processor.image_processor.merge_size ** 2
    v_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    token_ids, token_tags = [], []
    for idx, grid in enumerate(vision["image_grid_thw"]):
      label_ids = tokenizer(
          f"Picture {idx + 1}: ", add_special_tokens=False
      )["input_ids"][:4096]
      v_len = int(grid.prod().item()) // merge_size
      v_ids = [v_start] + [img_pad] * v_len + [v_end]
      token_ids.extend(label_ids + v_ids)
      token_tags.extend([TEXT_TAG] * len(label_ids) + [IMAGE_TAG] * len(v_ids))
    p_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][:4096]
    token_ids.extend(p_ids)
    token_tags.extend([TEXT_TAG] * len(p_ids))
    real_tags = np.asarray(token_tags, dtype=np.int64)
    if pad_to > len(token_ids):
      pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
      token_tags.extend([TEXT_TAG] * (pad_to - len(token_ids)))
      token_ids.extend([pad_id] * (pad_to - len(token_ids)))
    mm_ids = np.asarray(processor.create_mm_token_type_ids([token_ids]), dtype=np.int64)
    pres = dict(
        ids=np.asarray(token_ids, dtype=np.int32),
        tags=np.asarray(token_tags, dtype=np.int64),
        pixel_values=vision["pixel_values"].float().numpy(),
        grid_thw=vision["image_grid_thw"].numpy(),
        mm_token_type_ids=mm_ids,
        image_pad_id=img_pad,
    )
    pos3 = r2v.text_positions(
        img_snap, pres["ids"], pres["grid_thw"], pres["mm_token_type_ids"]
    )
    p_embeds = r2v.encode_text_with_vision(
        encoder, vision_model, pres, pos3, cfg_text, dtype,
        stop_at_layer=TEXT_ENCODER_LAYER, jit=True,
    )
    return p_embeds, real_tags, ref_imgs, [(im.height // 16 // 2 * 2, im.width // 16 // 2 * 2) for im in ref_imgs]

  # The tiled encode as one program. Op by op it dispatched ~1,300 small
  # programs (0.57 s for a 1024x1024 reference, ~0.18 s of it on the chip).
  def _eager_tiled_encode(w, x):
    return jvae.tiled_encode(lambda t_: _ENC_ONE[0](w, t_), x, mesh=mesh)

  _jit_tiled_encode = jax.jit(_eager_tiled_encode)

  def encode_ref_latents(ref_imgs, tgen):
    """Encode reference PIL images through the 3D VAE encoder using `tgen`."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    ref_latents = []
    mean5 = np.asarray(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    std5 = np.asarray(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    for im in ref_imgs:
      hc = im.height // 16 * 16
      wc = im.width // 16 * 16
      px = np.asarray(im, np.float32)[:hc, :wc].transpose(2, 0, 1)[None, :, None] / 255.0
      px = (px - mean5) / std5
      x_j = jnp.asarray(px, dtype)
      if os.environ.get("DT_ENC_JIT", "1") == "1":
        moments = _jit_tiled_encode(enc_w, x_j)
        if os.environ.get("DT_ENC_CHECK") == "1":
          ref = np.asarray(_eager_tiled_encode(enc_w, x_j)[:, :, -1], np.float32).ravel()
          got = np.asarray(moments[:, :, -1], np.float32).ravel()
          print(f"  [enc-check] jit vs eager cos {float(got @ ref / np.linalg.norm(got) / np.linalg.norm(ref)):.6f} "
                f"max|d| {float(np.abs(got - ref).max()):.3g}", flush=True)
      else:
        moments = _eager_tiled_encode(enc_w, x_j)
      moments_4d = np.asarray(moments[:, :, -1], np.float32)
      mu, logvar = np.split(moments_4d, 2, axis=1)
      sd = np.exp(0.5 * np.clip(logvar, -30.0, 20.0))
      eps = randn_tensor(
          tuple(mu.shape), generator=tgen, dtype=torch.float32
      ).numpy()
      lat = (mu + sd * eps).astype(np.float16).astype(np.float32)
      lat = lat[..., :lat.shape[-2] // 2 * 2, :lat.shape[-1] // 2 * 2]
      ref_latents.append((lat - lat_mean_np) / lat_std_np)
    return ref_latents

  # Initial prompt/vision encode to establish sequence layout
  with jax.set_mesh(mesh):
    prompt_embeds, text_tags, ref_imgs, ref_grids = encode_prompt_and_refs()
    jax.block_until_ready(prompt_embeds)

  (pos_np, tags_np, img_idx_np, txt_idx_np,
   n_ref, n_target) = build_image_packed_sequence(
      text_tags, ref_grids, latent_h, latent_w
  )
  seq = int(pos_np.shape[0])
  row_step_np = np.zeros((seq,), dtype=np.int32)
  if n_ref > 0:
    row_step_np[img_idx_np[:n_ref]] = 1

  position_ids = jax.device_put(jnp.asarray(pos_np), replicated)
  token_tags = jax.device_put(jnp.asarray(tags_np), replicated)
  image_idx = jax.device_put(jnp.asarray(img_idx_np), replicated)
  text_idx = jax.device_put(jnp.asarray(txt_idx_np), replicated)
  row_step = jax.device_put(jnp.asarray(row_step_np), replicated)

  _idx = (row_step_np * dit.MODALITY_NUM + tags_np).astype(np.int64)
  _edges = np.flatnonzero(_idx[1:] != _idx[:-1]) + 1
  _bounds = [0, *_edges.tolist(), int(_idx.shape[0])]
  segments = tuple((a, b, int(_idx[a])) for a, b in zip(_bounds, _bounds[1:]))
  if not args.segments:
    segments = None

  graphdef, dit_state = nnx.split(transformer)
  # `--offload`: the DiT sits in pinned host memory. A host operand makes
  # `dot_general` raise rather than transfer, so the copy is asked for inside
  # the program, where XLA overlaps each block's transfer with the previous
  # block's compute (as `jax_pipeline` does).
  dit_dev = (jax.tree.map(lambda v: NamedSharding(mesh, v.sharding.spec, memory_kind="device"),
                          dit_state)
             if dit_host is not None else None)

  @jax.jit
  def denoise_step(state, img_rows, text_emb, t_vec, dt_scalar, pdd_idx, mods=None):
    if dit_dev is not None:
      state = jax.tree.map(jax.device_put, state, dit_dev)
    model = nnx.merge(graphdef, state)
    dv = model(
        img_rows, None, text_emb, t_vec, row_step, token_tags, position_ids,
        image_idx, None, text_idx, sdpa=sdpa, segments=segments,
        adaln_mods=mods, pdd_step=pdd_idx,
    )
    gen = img_rows[:, n_ref:].astype(jnp.float32) + dt_scalar * dv[:, n_ref:].astype(jnp.float32)
    return jnp.concatenate([img_rows[:, :n_ref], gen], axis=1) if n_ref else gen

  @functools.partial(jax.jit, out_shardings=replicated)
  def unpatchify_and_denorm(img_rows):
    rows = img_rows[:, n_ref:] if n_ref else img_rows
    x = rows.reshape(1, latent_h // 2, latent_w // 2, 24, 2, 2)
    x = jnp.transpose(x, (0, 3, 1, 4, 2, 5)).reshape(1, 24, latent_h, latent_w)
    return (x.astype(jnp.float32) * lat_std_jnp + lat_mean_jnp).astype(dtype)

  def decode_image_2d(z_4d):
    if args.decode_tile_shard and len(devices) > 1:
      px = jvae.tiled_decode(
          lambda t_: decode_tile_batch(dec_state, t_),
          z_4d, tile_px=256, overlap_px=64, mesh=mesh,
      )
    else:
      px = jvae.tiled_decode(
          lambda t_: decode_tile_batch(dec_state, t_),
          z_4d, tile_px=256, overlap_px=64,
          batch=args.vae_tile_batch or 16,
      )
    return to_uint8_rgb(px)

  def prepare_initial_rows():
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    tgen = torch.Generator().manual_seed(args.seed)
    ref_latents = encode_ref_latents(ref_imgs, tgen) if n_ref else []
    target_noise = randn_tensor(
        (1, 24, 1, latent_h, latent_w), generator=tgen, dtype=torch.float32
    ).numpy()[:, :, 0]
    target_rows = patchify_2d(target_noise)
    if not n_ref:
      return jax.device_put(jnp.asarray(target_rows, jnp.float32), replicated)
    cond_list = []
    t0 = np.float32(0.999)
    one_minus_t0 = np.float32(1.0) - t0
    for rlat in ref_latents:
      c_noise = randn_tensor(
          tuple(rlat.shape), generator=tgen, dtype=torch.float32
      ).numpy()
      noised = t0 * rlat + one_minus_t0 * c_noise
      cond_list.append(patchify_2d(noised))
    all_rows = np.concatenate(cond_list + [target_rows], axis=1)
    return jax.device_put(jnp.asarray(all_rows, jnp.float32), replicated)

  def step_t_vec(i: int):
    if n_ref:
      return jax.device_put(
          jnp.asarray([timesteps_np[i], np.float32(0.999)], jnp.float32),
          replicated,
      )
    return jax.device_put(jnp.asarray([timesteps_np[i]], jnp.float32), replicated)

  t_vecs = [step_t_vec(i) for i in range(args.steps)]
  dt_scalars = [
      jax.device_put(jnp.asarray(sigmas_np[i] - sigmas_np[i + 1], jnp.float32), replicated)
      for i in range(args.steps)
  ]
  pdd_indices = [
      jax.device_put(jnp.asarray(i, jnp.int32), replicated)
      for i in range(args.steps)
  ]

  with jax.set_mesh(mesh):
    if args.warmup:
      t_w = time.perf_counter()
      v0 = prepare_initial_rows()
      v0 = denoise_step(
          dit_state, v0, prompt_embeds, t_vecs[0], dt_scalars[0], pdd_indices[0],
          None if adaln_mods is None else adaln_mods[0],
      )
      z0 = unpatchify_and_denorm(v0)
      u0 = decode_image_2d(z0)
      jax.block_until_ready(u0)
      print(f"  warmup {time.perf_counter() - t_w:.2f}s", flush=True)
      del v0, z0, u0

    u8_host = None
    whole = os.environ.get("H3_IMAGE_PROFILE_WHOLE") == "1"
    for run_idx in range(args.runs):
      prof = os.environ.get("H3_IMAGE_PROFILE_DIR") if run_idx == args.runs - 1 else None
      if prof and whole:
        jax.profiler.start_trace(prof)
      t_run = time.perf_counter()
      t_te0 = time.perf_counter()
      prompt_embeds, _, _, _ = encode_prompt_and_refs()
      jax.block_until_ready(prompt_embeds)
      t_te = time.perf_counter() - t_te0

      t_pipe0 = time.perf_counter()
      rows = prepare_initial_rows()
      jax.block_until_ready(rows)
      t_prep = time.perf_counter() - t_pipe0
      if prof and not whole:
        jax.profiler.start_trace(prof)
      t_den0 = time.perf_counter()
      for i in range(args.steps):
        rows = denoise_step(
            dit_state, rows, prompt_embeds, t_vecs[i], dt_scalars[i], pdd_indices[i],
            None if adaln_mods is None else adaln_mods[i],
        )
      z_4d = unpatchify_and_denorm(rows)
      jax.block_until_ready(z_4d)
      t_den = time.perf_counter() - t_den0
      if prof and not whole:
        jax.profiler.stop_trace()
        print(f"  [profile] denoise of the last request -> {prof}", flush=True)

      t_dec0 = time.perf_counter()
      u8_dev = decode_image_2d(z_4d)
      u8_host = np.asarray(u8_dev)
      t_dec = time.perf_counter() - t_dec0
      t_pipe = time.perf_counter() - t_pipe0
      t_total = time.perf_counter() - t_run
      if prof and whole:
        jax.profiler.stop_trace()
        print(f"  [profile] the whole last request -> {prof}", flush=True)
      print(
          f"run {run_idx}: e2e {t_total:.3f}s | pipeline (warm cond) {t_pipe:.3f}s "
          f"[text_encoder {t_te:.3f}s  prepare {t_prep:.3f}s  denoise {t_den:.3f}s  vae_decode {t_dec:.3f}s]",
          flush=True,
      )

  if args.requests:
    serve_requests(args, locals())

  if u8_host is not None:
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    os.makedirs(args.out, exist_ok=True)
    pil_img = Image.fromarray(u8_host)
    if (out_w, out_h) != (gen_w, gen_h):
      pil_img = VaeImageProcessor().resize(pil_img, height=out_h, width=out_w)
    out_path = os.path.join(args.out, f"{args.tag}.png")
    pil_img.save(out_path)
    np.save(os.path.join(args.out, f"{args.tag}.npy"), np.asarray(pil_img))
    print(f"wrote {out_path} ({pil_img.size[0]}x{pil_img.size[1]})", flush=True)


def serve_requests(args, env):
  """Requests with the models resident (`--requests`).

  Each JSONL line is `{"prompt": ..., "seed": ..., "reference": [paths]}`
  (`reference` optional; edit requests need the vision tower, i.e. the
  process started with `--reference`). Every per-request array (positions,
  tags, row indices, run bounds, true lengths) is an argument of one jitted
  step, and the text is padded to `--text-bucket` tokens: prompts of any
  length in a bucket reuse one compiled program (an edit also keys on its
  reference sizes). The padded text rows sit after the image rows; the text
  tower is causal, the token refiner masks them by `text_len` and the
  transformer's attention by `seq_len`, so a real row never sees them.
  """
  import json
  from concurrent.futures import ThreadPoolExecutor
  import torch
  from PIL import Image
  from diffusers.image_processor import VaeImageProcessor
  from diffusers.utils.torch_utils import randn_tensor
  import jax
  import jax.numpy as jnp

  e = env
  dit, nnx, mesh, replicated = e["dit"], e["nnx"], e["mesh"], e["replicated"]
  graphdef, dit_state, sdpa = e["graphdef"], e["dit_state"], e["sdpa"]
  tokenizer, encoder, _text_jit = e["tokenizer"], e["encoder"], e["_text_jit"]
  latent_h, latent_w = e["latent_h"], e["latent_w"]
  adaln_mods, steps = e["adaln_mods"], args.steps
  if encoder is None:
    raise SystemExit("--requests needs the text encoder resident (no --free-text-after)")
  bucket = max(1, args.text_bucket)
  text_val = int(TEXT_TAG)            # timestep index 0: MODALITY_NUM * 0 + tag

  @functools.partial(jax.jit, static_argnames=("run_vals", "n_ref"))
  def step(state, img_rows, text_emb, t_vec, dt_scalar, pdd_idx, mods, position_ids, token_tags,
           image_idx, text_idx, row_step, n_txt, seq_real, bounds, run_vals, n_ref):
    seg = tuple((bounds[2 * r], bounds[2 * r + 1], v) for r, v in enumerate(run_vals))
    model = nnx.merge(graphdef, state)
    dv = model(img_rows, None, text_emb, t_vec, row_step, token_tags, position_ids,
               image_idx, None, text_idx, sdpa=sdpa, segments=seg,
               seq_len=seq_real, text_len=n_txt, adaln_mods=mods, pdd_step=pdd_idx)
    gen = img_rows[:, n_ref:].astype(jnp.float32) + dt_scalar * dv[:, n_ref:].astype(jnp.float32)
    return jnp.concatenate([img_rows[:, :n_ref], gen], axis=1) if n_ref else gen

  put = lambda a, dt=None: jax.device_put(jnp.asarray(a, dt), replicated)
  lat_mean, lat_std, dtype = e["lat_mean_jnp"], e["lat_std_jnp"], e["dtype"]

  @functools.partial(jax.jit, out_shardings=replicated)
  def unpatch(rows):
    # `unpatchify_and_denorm` without the main request's reference count.
    x = rows.reshape(1, latent_h // 2, latent_w // 2, 24, 2, 2)
    x = jnp.transpose(x, (0, 3, 1, 4, 2, 5)).reshape(1, 24, latent_h, latent_w)
    return (x.astype(jnp.float32) * lat_std + lat_mean).astype(dtype)

  def encode_t2i(prompt):
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:4096]
    n = len(ids)
    tb = -(-n // bucket) * bucket
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids_np = np.asarray([ids + [pad_id] * (tb - n)], dtype=np.int32)
    pos = np.broadcast_to(np.arange(tb)[None, :], (3, tb))
    cos, sin = e["qwen_text"].mrope_cos_sin(
        pos, head_dim=e["cfg_text"]["head_dim"], rope_theta=e["cfg_text"]["rope_theta"],
        mrope_section=e["cfg_text"]["mrope_section"])
    emb = _text_jit(encoder, put(ids_np, jnp.int32), put(cos, e["dtype"]), put(sin, e["dtype"]))
    return emb, np.ones((n,), np.int64), [], []

  def encode_edit(prompt, refs):
    # The pipeline's own presentation, padded: measure the real length, then
    # encode once at the bucket's length.
    args.prompt, args.reference = prompt, refs
    from PIL import ImageOps
    vis = [bound_short_edge(ImageOps.exif_transpose(Image.open(r)).convert("RGB"), 1024, 1024)
           for r in refs]
    grid = e["processor"].image_processor(images=vis, return_tensors="pt")["image_grid_thw"]
    merge = e["processor"].image_processor.merge_size ** 2
    n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"][:4096])
    for idx, g in enumerate(grid):
      n += len(tokenizer(f"Picture {idx + 1}: ", add_special_tokens=False)["input_ids"]) + 2
      n += int(g.prod().item()) // merge
    return e["encode_prompt_and_refs"](pad_to=-(-n // bucket) * bucket)

  saver = ThreadPoolExecutor(max_workers=2)   # a 2048x2048 PNG is ~1.2 s of host time
  pending, seen = [], set()
  reqs = [json.loads(l) for l in open(args.requests) if l.strip()]
  for k, req in enumerate(reqs):
    t0 = time.perf_counter()
    refs = req.get("reference") or []
    if isinstance(refs, str):
      refs = [refs]
    with jax.set_mesh(mesh):
      emb, tags_real, ref_imgs, ref_grids = (encode_edit(req["prompt"], refs) if refs
                                             else encode_t2i(req["prompt"]))
      jax.block_until_ready(emb)
    t_text = time.perf_counter() - t0
    n, tb = len(tags_real), int(emb.shape[1])

    pos_np, tags_np, img_idx_np, txt_idx_np, n_ref, _ = build_image_packed_sequence(
        tags_real, ref_grids, latent_h, latent_w)
    seq_real = int(pos_np.shape[0])
    seq, extra = seq_real + tb - n, tb - n
    row_np = np.zeros((seq,), np.int32)
    if n_ref:
      row_np[img_idx_np[:n_ref]] = 1
    idx = (row_np[:seq_real] * dit.MODALITY_NUM + tags_np).astype(np.int64)
    edges = [0, *(np.flatnonzero(idx[1:] != idx[:-1]) + 1).tolist(), seq_real]
    runs = [(a, b, int(idx[a])) for a, b in zip(edges, edges[1:])] + [(seq_real, seq, text_val)]
    pos_np = np.concatenate([pos_np, np.repeat(pos_np[-1:], extra, axis=0)])
    tags_np = np.concatenate([tags_np, np.full((extra,), TEXT_TAG, np.int32)])
    txt_idx_np = np.concatenate([txt_idx_np, np.arange(seq_real, seq, dtype=np.int32)])
    lay = (put(pos_np), put(tags_np), put(img_idx_np), put(txt_idx_np), put(row_np),
           put(n, jnp.int32), put(seq_real, jnp.int32),
           put(np.asarray([b for r in runs for b in r[:2]], np.int32)))
    run_vals = tuple(v for _, _, v in runs)

    tgen = torch.Generator().manual_seed(int(req.get("seed", args.seed)))
    with jax.set_mesh(mesh):
      ref_lat = e["encode_ref_latents"](ref_imgs, tgen) if n_ref else []
    noise = randn_tensor((1, 24, 1, latent_h, latent_w), generator=tgen,
                         dtype=torch.float32).numpy()[:, :, 0]
    parts = []
    for rl in ref_lat:
      c = randn_tensor(tuple(rl.shape), generator=tgen, dtype=torch.float32).numpy()
      parts.append(patchify_2d(np.float32(0.999) * rl + (np.float32(1.0) - np.float32(0.999)) * c))
    rows = put(np.concatenate(parts + [patchify_2d(noise)], axis=1), jnp.float32)
    t_vecs = [put([e["timesteps_np"][i], 0.999] if n_ref else [e["timesteps_np"][i]], jnp.float32)
              for i in range(steps)]
    jax.block_until_ready(rows)
    t1 = time.perf_counter()
    t_prep = t1 - t0 - t_text
    with jax.set_mesh(mesh):
      for i in range(steps):
        rows = step(dit_state, rows, emb, t_vecs[i], e["dt_scalars"][i], e["pdd_indices"][i],
                    None if adaln_mods is None else adaln_mods[i], *lay,
                    run_vals=run_vals, n_ref=int(n_ref))
      z = unpatch(rows[:, n_ref:] if n_ref else rows)
      jax.block_until_ready(z)
      t_den = time.perf_counter() - t1
      t2 = time.perf_counter()
      u8 = np.asarray(e["decode_image_2d"](z))
    t_dec = time.perf_counter() - t2
    key = (tb, tuple(ref_grids))
    first = key not in seen
    seen.add(key)
    img = Image.fromarray(u8)
    if (e["out_w"], e["out_h"]) != (e["gen_w"], e["gen_h"]):
      img = VaeImageProcessor().resize(img, height=e["out_h"], width=e["out_w"])
    t_gen = time.perf_counter() - t0
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}_req{k}.png")
    pending.append(saver.submit(img.save, path))
    kind = f"edit, {len(refs)} ref" if refs else "t2i"
    print(f"[req {k}] {kind}, text {n}->{tb} tokens{' (new shape)' if first else ''}: "
          f"{t_gen:.3f}s = text {t_text:.3f} + prepare {t_prep:.3f} + denoise {t_den:.3f} "
          f"+ decode {t_dec:.3f}  -> {path}", flush=True)
  for f in pending:
    f.result()
  saver.shutdown()


_ENC_ONE = [None]

if __name__ == "__main__":
  main()
