"""Host-side preparation for ref2va: everything in the native JAX pipeline
that does not touch weights.

Reference image preprocessing, prompt tokens, vision tower inputs, the text
tower's mRoPE positions, and the packed layout. These are pure index
computations, so diffusers / transformers code is called directly wherever
possible rather than re-derived.

Run standalone to compare against a torch dump:

    JAX_PLATFORMS=cpu python -m diffusers_tpu.models.minimax_h3.jax_ref2va \\
        --reference ref.png --dump out/ref2va_host
"""

import argparse
import functools
import os

import numpy as np

# Match `MiniMaxH3ModularPipeline` attributes / ConfigSpec.
REFERENCE_SHORT_EDGE = 2048
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
ENCODE_SEED = 42
NOISE_AUG = 0.999
TEXT_TAG, VIDEO_TAG, AUDIO_TAG = 1, 0, 2
# `canvas_multiple` is a pipeline attribute (VAE stride 16 x patch 2).
CANVAS_MULTIPLE = 32


def load_reference(path, short_edge=REFERENCE_SHORT_EDGE):
  """Reference image -> resized PIL image, same algorithm as
  `MiniMaxH3Ref2VASetupStep`.

  Scales the short edge to `short_edge`, rounds both sides to
  `canvas_multiple`, and resizes with diffusers' `VaeImageProcessor.resize`
  (PIL LANCZOS). E.g. 1344x768 -> 3584x2048.
  """
  from PIL import Image
  from diffusers.image_processor import VaeImageProcessor
  image = Image.open(path).convert("RGB")
  width, height = image.size
  multiple = CANVAS_MULTIPLE
  scale = short_edge / min(width, height)
  th = max(multiple, round(height * scale / multiple) * multiple)
  tw = max(multiple, round(width * scale / multiple) * multiple)
  if image.size != (tw, th):
    image = VaeImageProcessor().resize(image, height=th, width=tw)
  return image


@functools.lru_cache(maxsize=4)
def _tok_proc(snapshot):
  """Load tokenizer and processor once per snapshot (reused per request)."""
  from transformers import AutoProcessor, AutoTokenizer
  return (AutoTokenizer.from_pretrained(os.path.join(snapshot, "tokenizer")),
          AutoProcessor.from_pretrained(os.path.join(snapshot, "processor")))


@functools.lru_cache(maxsize=4)
def _rope_shim(snapshot):
  """Meta-device shell model for `get_rope_index`, built once."""
  import torch
  from transformers import AutoConfig, Qwen3VLModel
  config = AutoConfig.from_pretrained(os.path.join(snapshot, "text_encoder"))
  with torch.device("meta"):
    shim = Qwen3VLModel(config)
  return shim


def presentation(snapshot, prompt, image):
  """Tokens, tags, vision tower inputs, and mRoPE inputs.

  Calls two `MiniMaxH3Ref2VATextEncoderStep` methods directly:
  `_gather_vision_features` yields pixel_values / grid, `_build_presentation`
  yields tokens and tags for `"<Picture 1>: " + vision block + prompt`.
  """
  import torch
  from transformers import AutoProcessor, AutoTokenizer
  from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
  from diffusers.modular_pipelines.minimax_h3.encoders import (
      MiniMaxH3Ref2VATextEncoderStep,
  )

  tokenizer, processor = _tok_proc(snapshot)
  refs = [MiniMaxH3ImageReference(image=image)]
  step = MiniMaxH3Ref2VATextEncoderStep.__new__(MiniMaxH3Ref2VATextEncoderStep)
  vision, img_counts, vid_counts, vid_ts = step._gather_vision_features(
      processor, refs, 24.0)
  ids, tags = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
      tokenizer, prompt, refs, img_counts, vid_counts, vid_ts,
      text_tag=TEXT_TAG, video_tag=VIDEO_TAG)
  mm = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
  return dict(ids=np.asarray(ids, np.int32), tags=np.asarray(tags, np.int64),
              pixel_values=vision["pixel_values"].float().numpy(),
              grid_thw=vision["image_grid_thw"].numpy(),
              mm_token_type_ids=mm.numpy(), image_token_counts=img_counts,
              image_pad_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
              processor=processor, references=refs)


def layout(text_tags, references, condition_shape, num_frames, height, width):
  """Packed layout, via `build_ref2va_packed_sequence` unchanged.

  It only reads the shape of the reference latent, so a meta tensor of the
  same shape is passed instead of running the VAE encode first. Returns
  (position_ids, token_tags, video_idx, audio_idx, text_idx,
  num_ref_video_rows, num_ref_audio_rows).
  """
  import torch
  from diffusers.modular_pipelines.minimax_h3.before_denoise import (
      MiniMaxH3Ref2VAPrepareLayoutStep,
  )
  from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
      MINIMAX_H3_AUDIO_CHANNELS,
      audio_latent_num_frames,
      video_latent_num_frames,
  )

  spatial, per_chunk, latents_per_chunk = 16, 17, 5
  return MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
      text_token_tags=torch.as_tensor(text_tags, dtype=torch.long),
      references=references,
      condition_latents=[torch.empty(condition_shape, device="meta")],
      audio_condition_latents=[],
      num_latent_frames=video_latent_num_frames(num_frames, per_chunk,
                                                latents_per_chunk),
      latent_height=height // spatial,
      latent_width=width // spatial,
      num_audio_latents=audio_latent_num_frames(num_frames),
      patch_size=(1, 2, 2),
      audio_channels=MINIMAX_H3_AUDIO_CHANNELS,
      audio_tag=AUDIO_TAG,
      video_tag=VIDEO_TAG,
  )


def text_positions(snapshot, ids, grid_thw, mm_token_type_ids):
  """Three-axis mRoPE positions (3, seq) for the text tower, via Qwen3-VL's
  own `get_rope_index`.

  For t2va (text only) all three axes are the same arange; in ref2va the
  vision block follows the image grid, so the axes differ. A meta-device
  shell model provides the index logic without weights.
  """
  import torch

  shim = _rope_shim(snapshot)
  shim.rope_deltas = None
  input_ids = torch.as_tensor(ids, dtype=torch.long)[None]
  position_ids, _ = shim.get_rope_index(
      input_ids, image_grid_thw=torch.as_tensor(grid_thw),
      attention_mask=torch.ones_like(input_ids),
      mm_token_type_ids=torch.as_tensor(mm_token_type_ids))
  return position_ids[:, 0].numpy()          # (3, seq)


def text_inputs(p, positions, cfg_text, dtype):
  """Host side of `encode_text_with_vision`: arrays that depend only on shapes and tokens."""
  import jax.numpy as jnp
  from ..qwen3_vl import jax_text as qwen_text
  from ..qwen3_vl import jax_vision as qv

  vmeta = qv.vision_metadata(p["grid_thw"])
  vcos, vsin = qv.vision_rope(vmeta["position_ids"])
  cos, sin = qwen_text.mrope_cos_sin(
      positions, head_dim=cfg_text["head_dim"],
      rope_theta=cfg_text["rope_theta"],
      mrope_section=cfg_text["mrope_section"])
  vpos = np.flatnonzero(p["ids"] == int(p["image_pad_id"]))
  return dict(
      pixels=jnp.asarray(p["pixel_values"], dtype),
      interp_indices=jnp.asarray(vmeta["interp_indices"]),
      interp_weights=jnp.asarray(vmeta["interp_weights"]),
      vcos=jnp.asarray(vcos, dtype), vsin=jnp.asarray(vsin, dtype),
      ids=jnp.asarray(p["ids"])[None],
      cos=jnp.asarray(cos, dtype), sin=jnp.asarray(sin, dtype),
      visual_positions=jnp.asarray(vpos))


def text_forward(encoder, vision_model, x, stop_at_layer):
  """Vision tower -> merged into the text stream -> text tower up to layer
  `stop_at_layer`. Pure function, jittable.

  Same wiring as the Qwen3-VL reference encoder: merged
  tokens fill the image_pad positions, and deepstack features are added back
  at those positions after the first 3 layers.
  """
  merged, deepstack = vision_model(
      x["pixels"], {"interp_indices": x["interp_indices"],
                    "interp_weights": x["interp_weights"]},
      x["vcos"], x["vsin"])
  if x["visual_positions"].shape[0] != merged.shape[0]:
    raise ValueError(f"{x['visual_positions'].shape[0]} image_pad tokens, "
                     f"but the vision tower produced {merged.shape[0]} tokens")
  embeds = encoder.embed_tokens[...][x["ids"]]
  inputs_embeds = embeds.at[:, x["visual_positions"]].set(
      merged.astype(embeds.dtype))
  # The 4th positional arg is the mask; pass everything else by keyword.
  return encoder(x["ids"], x["cos"], x["sin"], inputs_embeds=inputs_embeds,
                 deepstack=deepstack, visual_positions=x["visual_positions"],
                 stop_at_layer=stop_at_layer)


def encode_text_with_vision(encoder, vision_model, p, positions, cfg_text,
                            dtype, stop_at_layer, jit=False):
  """`text_inputs` + `text_forward`. `jit=True` compiles it as one program:
  op-by-op execution takes 72 s on 8 chips vs 38.8 s for the torch_tpu text tower."""
  x = text_inputs(p, positions, cfg_text, dtype)
  if not jit:
    return text_forward(encoder, vision_model, x, stop_at_layer)
  # Built once: `nnx.jit` returns a fresh callable with a fresh cache on
  # every call, so building it here each time would recompile each time.
  global _TEXT_JIT
  if _TEXT_JIT is None:
    from flax import nnx
    _TEXT_JIT = nnx.jit(text_forward, static_argnames=("stop_at_layer",))
  return _TEXT_JIT(encoder, vision_model, x, stop_at_layer=stop_at_layer)


_TEXT_JIT = None
_ENC_JIT = None


def encode_condition(encode_one, image, latents_mean, latents_std, batch=0,
                     dtype=None, mesh=None, weights=None):
  """Reference image -> normalized condition latent (1, 24, 1, H/16, W/16),
  float32 numpy.

  Native version of `encode_vae_condition`:
  1. uint8 / 255, normalized with ImageNet mean/std;
  2. spatially tiled encode (`jax_vae.tiled_encode`, 256 tiles, overlap 64)
     gives moments;
  3. split into mean / logvar (clamped to -30..20) and sample with a
     *separate* `Generator().manual_seed(42)` -- noise is drawn on the host
     with torch so it matches the reference bit for bit;
  4. round-trip through float16 (the reference rounds this way);
  5. normalize with latents_mean / latents_std.
  """
  import jax.numpy as jnp
  import torch
  from diffusers.utils.torch_utils import randn_tensor
  from . import jax_vae as jvae

  px = np.asarray(image, np.float32).transpose(2, 0, 1)[None, :, None] / 255.0
  mean = np.asarray(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
  std = np.asarray(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
  px = (px - mean) / std
  x = jnp.asarray(px, dtype or jnp.float32)
  if weights is not None:
    # Compile the whole tiled encode (tiling, vmap, blending) as one program;
    # op-by-op dispatch takes 2.4 s, dominated by small ops. Weights are
    # passed as arguments, not closed over (closures get baked in as constants).
    import jax
    global _ENC_JIT
    if _ENC_JIT is None:
      _ENC_JIT = jax.jit(
          lambda w, xx, fn, b, m: jvae.tiled_encode(lambda t: fn(w, t), xx,
                                                    batch=b, mesh=m),
          static_argnums=(2, 3, 4))
    moments = _ENC_JIT(weights, x, encode_one, batch, mesh)
  else:
    moments = jvae.tiled_encode(encode_one, x, batch=batch, mesh=mesh)
  moments = np.asarray(moments, np.float32)
  mu, logvar = np.split(moments, 2, axis=1)
  logvar = np.clip(logvar, -30.0, 20.0)
  sd = np.exp(0.5 * logvar)
  noise = randn_tensor(tuple(mu.shape),
                       generator=torch.Generator().manual_seed(ENCODE_SEED),
                       dtype=torch.float32).numpy()
  z = (mu + sd * noise).astype(np.float16).astype(np.float32)
  lm = np.asarray(latents_mean, np.float32).reshape(1, -1, 1, 1, 1)
  ls = np.asarray(latents_std, np.float32).reshape(1, -1, 1, 1, 1)
  return (z - lm) / ls


def condition_shape(image, latent_channels=24, spatial=16):
  """Latent shape of the encoded reference image (1, C, 1, H/16, W/16)."""
  w, h = image.size
  return (1, latent_channels, 1, h // spatial, w // spatial)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--reference", required=True)
  ap.add_argument("--prompt", default="a red fox trotting through snow")
  ap.add_argument("--num-frames", type=int, default=345)
  ap.add_argument("--height", type=int, default=768)
  ap.add_argument("--width", type=int, default=1344)
  ap.add_argument("--dump", default=None, help="write results as .npy files for comparison")
  args = ap.parse_args()

  from . import jax_dit as dit
  snapshot = dit.find_snapshot()
  image = load_reference(args.reference)
  print(f"reference -> {image.size[0]}x{image.size[1]}", flush=True)
  p = presentation(snapshot, args.prompt, image)
  print(f"tokens {p['ids'].shape[0]} (vision {p['image_token_counts']}), "
        f"pixel_values {p['pixel_values'].shape}, grid {p['grid_thw'].tolist()}",
        flush=True)
  pos3 = text_positions(snapshot, p["ids"], p["grid_thw"],
                        p["mm_token_type_ids"])
  print(f"text mRoPE positions {pos3.shape}, per-axis max {pos3.max(1).tolist()}",
        flush=True)
  shape = condition_shape(image)
  lay = layout(p["tags"], p["references"], shape, args.num_frames,
               args.height, args.width)
  pos, tags, vid, aud, txt, n_ref_v, n_ref_a = lay
  print(f"packed {pos.shape[0]} rows: text {txt.shape[0]}, audio {aud.shape[0]}, "
        f"video {vid.shape[0]} (reference {n_ref_v}), reference audio {n_ref_a}",
        flush=True)
  if args.dump:
    os.makedirs(args.dump, exist_ok=True)
    for k, v in dict(ids=p["ids"], text_tags=p["tags"], text_positions=pos3,
                     grid_thw=p["grid_thw"], position_ids=pos.numpy(),
                     token_tags=tags.numpy(), video_indices=vid.numpy(),
                     audio_indices=aud.numpy(),
                     text_indices=txt.numpy()).items():
      np.save(os.path.join(args.dump, f"{k}.npy"), np.asarray(v))
    print(f"wrote {args.dump}", flush=True)


if __name__ == "__main__":
  main()
