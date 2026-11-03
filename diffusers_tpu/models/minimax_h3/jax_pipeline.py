"""MiniMax-H3 text-to-video (and ref2va) in native JAX, end to end.

Every network runs in JAX here: the 64-layer Qwen3-VL text encoder
(`jax_text.py`), the 50-block DiT (`jax_dit.py`), and the 36-layer ViT
VAE decoder (`jax_vae.py`). What is *not* in JAX is the arithmetic
that never touches a weight -- tokenising the prompt, and the index
math that decides where text, audio and video rows sit in the packed
sequence. Those come from `diffusers` and `transformers` on the host
and cost microseconds.

**Single-chip support.** The DiT is 60 GiB in bfloat16 and the
language tower another 24, against 31 GiB on a v6e chip and 16 on a
v5e. Neither fits, so `--offload` keeps the weights in pinned host
memory and streams each one in per step; on four chips that costs 2.8%
of the request. `--chips` uses a subset of the host's chips.

`--reference` runs ref2va (vision tower + VAE-encoded reference +
`transformer_ref`).

Not implemented: **the audio track.** H3 emits audio latents alongside
video ones and this decodes only the video. t2va's audio measures -54 to
-64 dBFS in the torch pipeline too, i.e. silence.

    python -m diffusers_tpu.models.minimax_h3.jax_pipeline \\
        --prompt "a red fox trotting through snow" \\
        --num-frames 121 --height 480 --width 832 --steps 8 --offload
"""

import argparse
import functools
import os
import time

import numpy as np


def parse_args():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--prompt", default="a red fox trotting through snow")
  ap.add_argument("--sparse-window", type=int, default=0,
                  help="lossy: temporal-window block-sparse attention (same layout and "
                       "kernel as torchax --sparse-window); the first "
                       "--sparse-dense-steps calls stay dense")
  ap.add_argument("--sparse-dense-steps", type=int, default=6)
  ap.add_argument("--cache-threshold", type=float, default=0.0,
                  help="lossy: step-skip cache, same criterion as torchax / torch_tpu "
                       "--cache-threshold (reuse the previous output while the "
                       "accumulated relative drift of the time embedding is below "
                       "the threshold)")
  ap.add_argument("--torch-noise", action="store_true",
                  help="draw initial noise with torch.Generator(seed) on the host in the "
                       "reference order (condition -> video latent -> audio rows), giving "
                       "the same sample as the torch pipeline; otherwise jax.random gives "
                       "a different sample and DINO against the reference is much lower")
  ap.add_argument("--reference", default="",
                  help="reference image path; if given, runs ref2va (vision tower + "
                       "VAE-encoded reference + transformer_ref)")
  ap.add_argument("--negative-prompt", default="")
  ap.add_argument("--num-frames", type=int, default=121)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--width", type=int, default=832)
  ap.add_argument("--steps", type=int, default=8)
  ap.add_argument("--text-offload", action="store_true",
                  help="Stream only the text tower from host memory, keeping "
                       "the DiT resident (four chips: the 62 GiB tower does "
                       "not fit beside the step's temporaries).")
  ap.add_argument("--adaln-host", action="store_true",
                  help="Compute every block's AdaLN modulation for every step "
                       "on the host while loading, and never place adaln_proj "
                       "(26 of the DiT's 66 GB) on the chips. The modulation "
                       "depends on the timestep levels alone.")
  ap.add_argument("--int8-weights", action="store_true",
                  help="Weight-only int8 for the six big Linears of every "
                       "block (per-output-channel scales, quantized on the host "
                       "as the checkpoint is read): 66 -> 33 GB of DiT to hold "
                       "or stream. For hosts that cannot hold the bf16 DiT.")
  ap.add_argument("--pika-adapter", default=None,
                  help="A Pika-H3-Flex adapter directory (fl2va/ or ref2va/ with "
                       "manifest.json + adapter.safetensors). Merges the LoRA into "
                       "the DiT and samples its six-step re-noising ladder; "
                       "--steps, --shift and the step cache are ignored.")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--init-latents", default="",
                  help="Load `hidden_states.npy` / `audio_hidden_states.npy` "
                       "from this directory instead of drawing noise. The "
                       "two backends cannot share a generator, so this is "
                       "the only way to ask whether anything *but* the "
                       "noise still differs.")
  ap.add_argument("--attn-keep-heads", action="store_true",
                  help="drop the sharding constraint after the attention "
                       "kernel. Compiles faster but runs slower in steady "
                       "state (8 chips: 7.63 vs 6.28 s/step), so off by default.")
  ap.add_argument("--segments", action="store_true",
                  help="broadcast the modulation over runs of constant "
                       "AdaLN index instead of gathering per token. Much "
                       "slower here (658.1 vs 233.9 s), so off by default.")
  ap.add_argument("--block-q", type=int, default=2048)
  ap.add_argument("--block-kv", type=int, default=0,
                  help="attention kv block. 0 = auto (measured): ref2va 2048; "
                       "t2va 2048 up to 241 frames, 4096 beyond")
  # The following also honor their legacy environment variables.
  ap.add_argument("--pad-sequence-to", type=int, default=128,
                  help="pad the packed sequence to a multiple of this "
                       "(DT_PAD_SEQUENCE_TO); 0 disables. Without padding, "
                       "345-frame ref2va is 54 s slower")
  ap.add_argument("--full-lane-rope", action=argparse.BooleanOptionalAction,
                  default=True, help="full-lane RoPE (DT_FULL_LANE_ROPE)")
  ap.add_argument("--decode-tile-shard", action=argparse.BooleanOptionalAction,
                  default=True,
                  help="shard decode tiles across chips with replicated weights "
                       "(DT_DECODE_TILE_SHARD); --no-decode-tile-shard uses "
                       "tensor-parallel weights instead")
  ap.add_argument("--decode-chunk-batch", type=int, default=0,
                  help="temporal chunks decoded per call (DT_DECODE_CHUNK_BATCH); "
                       "0 = auto, see decode_chunk_batch(). 345 frames with "
                       "K=1/2/4/8/10 decode in 11.8/9.0/7.7/7.8/6.6 s; K=20 OOMs")
  ap.add_argument("--profile-dir", default="",
                  help="write a profile of one denoise step to this directory (H3_PROFILE_DIR)")
  ap.add_argument("--profile-step", type=int, default=8)
  ap.add_argument("--runs", type=int, default=1,
                  help="number of requests to run in one process; >1 keeps the DiT "
                       "resident and uses a tensor-parallel decoder")
  ap.add_argument("--block-kv-compute", type=int, default=0,
                  help="defaults to --block-kv")
  ap.add_argument("--vae-tile-px", type=int, default=256,
                  help="spatial tile for the VAE decode, in pixels; the "
                       "reference's default is 256 and the output depends "
                       "on it")
  ap.add_argument("--vae-tile-batch", type=int, default=0,
                  help="Decode this many spatial tiles per vmapped call. "
                       "0 is the original one-at-a-time loop.")
  ap.add_argument("--vae-tile-overlap-px", type=int, default=64,
                  help="minimum overlap between neighbouring tiles, in "
                       "pixels")
  ap.add_argument("--decode-dtype", default="", choices=("", "bfloat16",
                                                        "float32"),
                  help="dtype for the whole VAE decode -- the decoder's "
                       "own parameters, not just the latent handed to it.")
  ap.add_argument("--decode-npy", default="",
                  help="decode this latent instead of the one this run "
                       "produced; the denoise loop still runs")
  ap.add_argument("--bf16-weights", action="store_true",
                  help="round every float32 weight through bfloat16 on "
                       "load, the way the traced backend's "
                       "`torch_dtype=bfloat16` does")
  ap.add_argument("--init-text", default="",
                  help="take encoder_hidden_states from another backend's "
                       "H3_DUMP_INPUTS dir instead of running the text tower")
  ap.add_argument("--dump-latents", default="",
                  help="save the latent handed to the VAE decode, so it "
                       "can be diffed against the traced backend's")
  ap.add_argument("--shift", type=float, default=12.0)
  ap.add_argument("--audio-shift", type=float, default=3.0,
                  help="MiniMax-H3 ships a second scheduler for the audio "
                       "rows: scheduler/scheduler_config.json has shift 12, "
                       "audio_scheduler/ has 3. One forward serves rows at "
                       "two different noise levels.")
  ap.add_argument("--chips", type=int, default=0,
                  help="Use this many of the host's chips; 0 is all of them.")
  ap.add_argument("--offload-blocks", type=int, default=0,
                  help="Park only this many of the DiT's 50 transformer "
                       "blocks in host memory, counting from the first. "
                       "Offloading only what does not fit avoids paying the "
                       "transfer for every block on every step. "
                       "Implies --offload.")
  ap.add_argument("--offload", action="store_true",
                  help="Keep weights in pinned host memory and stream them "
                       "in. Required below about 80 GiB of total HBM.")
  ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
  ap.add_argument("--out", default="out")
  ap.add_argument("--tag", default="jax_t2va")
  ap.add_argument("--vae-clip", type=int, default=8,
                  help="Latent frames decoded per program. The decoder "
                       "is a transformer over one token per latent voxel, "
                       "so decoding a whole clip at once is very expensive.")
  ap.add_argument("--warmup", action="store_true",
                  help="Run the denoise loop once before timing it; "
                       "otherwise the timing is mostly cold compilation.")
  ap.add_argument("--compile-cache", default="",
                  help="Persist compiled programs here so a second "
                       "process does not pay for them again.")
  ap.add_argument("--fp8", action="store_true",
                  help="float8_e4m3fn weights and activations for DiT linears (native fp8 MXU "
                       "on v7x). Do not use on v6e (no fp8 MXU): lossless t2va 112.4 -> 120.7 s, "
                       "DINO 0.9709 -> 0.9377")
  ap.add_argument("--qk-fp8", action="store_true",
                  help="Cast Q and K to float8_e4m3fn inside Pallas splash attention.")
  ap.add_argument("--dense-attention", action="store_true",
                  help="Skip the Pallas splash kernel. Slower and much "
                       "hungrier, but it runs where the kernel does not.")
  return ap.parse_args()


# MiniMax-H3 conditions on `hidden_states[50]` of its 64-layer Qwen3-VL
# tower, not on the last one (`components.text_encoder_layer` in the
# reference: the final layer is post-norm and is not what the released
# weights were trained against).
TEXT_ENCODER_LAYER = 50


DECODE_CHUNK_CAP = 10


def decode_chunk_batch(frames_lat, requested=0, cap=DECODE_CHUNK_CAP):
  """Number of temporal chunks to decode per call.

  `requested` > 0 is used as is; 0 = auto: ceil(chunks / cap) batches, split
  evenly to minimize padding chunks. Time is roughly
  batches x fixed overhead + decoded chunks x per-chunk cost: 345 frames
  (17 chunks) decode in 7.7 s at K=4 vs 6.6 s at K=10, while 241 frames
  (12 chunks) at K=10 pads to 20 and ties K=4. cap 10 is the largest
  measured value that does not OOM (K=20 does).
  """
  if requested > 0:
    return requested
  token_drop, tokens_chunk = 3, 5
  num_tokens = frames_lat + token_drop
  n = (num_tokens + (-num_tokens) % tokens_chunk) // tokens_chunk - 1
  return -(-n // -(-n // cap))


def sigma_schedule(steps, shift):
  """The rectified-flow grid, `[1, 0]` compressed towards 1 by `shift`."""
  base = np.linspace(1.0, 0.0, int(steps), dtype=np.float32)
  sigmas = shift * base / (1 + (shift - 1) * base)
  # The shift can collide neighbouring float32 values near sigma = 1.
  keep = np.concatenate([[True], np.diff(sigmas) != 0])
  return sigmas[keep]


def _step_sigmas(args, pika=None):
  """Video and audio sigma grids, terminal 0 included.

  Pika's adapters fix their own ladder (`H3SigmaSchedule`): the knots
  / 1000 through each modality's shift, in float64, stored float32.
  """
  if pika is not None:
    q = np.asarray(pika["sampling"]["knots"], np.float64) / 1000.0
    shift = lambda q, s: (s * q / (1 + (s - 1) * q)).astype(np.float32)
    return (shift(q, pika["sampling"]["video_shift"]),
            shift(q, pika["sampling"]["audio_shift"]))
  return (sigma_schedule(args.steps, args.shift),
          sigma_schedule(args.steps, args.audio_shift))


def scheduler_step(sample, model_output, timestep, sigma, sigma_next):
  """One Euler step of H3's scheduler, in float32.

  `denoised` uses the sigma recovered from the *timestep* the
  transformer was conditioned on, while the Euler ratio uses the grid:
  below sigma 0.5 the round trip `1 - (1 - sigma)` is not exact and the
  reference keeps the two apart. Collapsing them changes the output.
  """
  import jax.numpy as jnp

  denoised = sample + (1.0 - timestep) * model_output
  ratio = sigma_next / sigma
  out = ratio * sample.astype(jnp.float32) + (1.0 - ratio) * denoised.astype(
      jnp.float32)
  return out.astype(sample.dtype)


def _ENC_W_FN(w, tile):
  """Per-tile reference encode with weights as an argument
  (see `jax_ref2va.encode_condition(weights=)`)."""
  return _ENC_ONE[0](w[0], w[1], tile)


_ENC_ONE = [None]


def main():
  args = parse_args()
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
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

  if args.compile_cache:
    jax.config.update("jax_compilation_cache_dir", args.compile_cache)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

  # `jax_dit` reads these at import time, so set them before importing it.
  os.environ["DT_PAD_SEQUENCE_TO"] = os.environ.get(
      "DT_PAD_SEQUENCE_TO", str(args.pad_sequence_to))
  os.environ["DT_FULL_LANE_ROPE"] = os.environ.get(
      "DT_FULL_LANE_ROPE", "1" if args.full_lane_rope else "0")
  # Branch on per-chip HBM: v6e has 32 GiB, v7x ~95 GiB per core (defaults
  # tuned on v7x regress on v6e).
  _hbm = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 2**30
  _small_hbm = _hbm < 48
  if (args.runs > 1 and (_small_hbm or args.chips == 1)
      and "DT_DECODE_TILE_SHARD" not in os.environ):
    args.decode_tile_shard = False      # replicated decoder does not fit beside a resident DiT on 32 GiB HBM
    if not args.vae_tile_batch and args.chips != 1:
      # The tensor-parallel decoder otherwise decodes one tile at a time:
      # batching 8 takes a resident decode from 3.4 to 2.1 s.
      args.vae_tile_batch = 8
  if not args.block_kv:
    # On v6e ref2va is 6 s faster with 2048 than 4096; on v7x 4096 is faster.
    args.block_kv = (2048 if args.num_frames <= 241 or (args.reference and _small_hbm)
                     else 4096)
    print(f"  [auto] --block-kv {args.block_kv}", flush=True)
  from . import jax_dit as dit
  from . import jax_text as jtext
  from . import jax_vae as jvae

  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
  dec_dtype = (dtype if not args.decode_dtype
               else (jnp.bfloat16 if args.decode_dtype == "bfloat16"
                     else jnp.float32))
  devices = jax.devices()[:args.chips] if args.chips else jax.devices()
  mesh = jax.make_mesh((len(devices),), ("tp",),
                       devices=devices,
                       axis_types=(jax.sharding.AxisType.Auto,))
  # Log every argument (prompt, seed, ...) so runs can be compared reliably.
  print("  [args] " + "  ".join(
      f"{k}={v}" for k, v in sorted(vars(args).items())), flush=True)
  print(f"{devices[0].device_kind} x {len(devices)}  dtype={args.dtype}"
        f"  offload={args.offload}", flush=True)

  snapshot = dit.find_snapshot()
  began = time.perf_counter()
  # Host-resident weights are the only way this fits a v6e-1 or a v5e.
  # Offloading and sharding are independent: weights keep their usual mesh
  # sharding and only the memory kind changes (a single-device host sharding
  # would be rejected by jit against a multi-device mesh).
  if args.offload_blocks:
    args.offload = True
  host = "pinned_host" if args.offload else None
  # On 1 chip (even a 95 GiB v7x), compiling the 62 GiB text tower resident
  # in HBM needs 119 GiB of temporaries; streaming it from pinned_host takes
  # ~9 s and leaves HBM free for a resident 31 GiB FP8 DiT.
  text_host = ("pinned_host" if (args.offload or args.text_offload
                                or len(devices) == 1) else None)

  def dit_host(name):
    """Which DiT weights live on the host. All of them, or the first N blocks."""
    if host is None or not args.offload_blocks:
      return host
    part = name.split(".")
    if part[0] == "transformer_blocks" and part[1].isdigit():
      return host if int(part[1]) < args.offload_blocks else "device"
    # The prologue and the heads are small and touched once a step;
    # leaving them resident costs little and saves a transfer.
    return "device"

  def to_device(tree):
    """Shardings that move a host-resident tree into HBM, in place.

    jax 0.11 has no `TransferToMemoryKind`, so the destination has to
    name a spec, and each weight has its own. Built out here from the
    loaded arrays and closed over: these are shardings, not weights, so
    capturing them does not bake 62 GiB into the program.
    """
    return jax.tree.map(
        lambda v: jax.sharding.NamedSharding(
            mesh, v.sharding.spec, memory_kind="device"),
        tree)

  # -- text ------------------------------------------------------------
  from transformers import AutoTokenizer

  text_dir = os.path.join(snapshot, "text_encoder")
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(snapshot, "tokenizer"))
  # Abstract, not concrete: a concrete build would allocate 24 GiB of zeros
  # on chip 0 before any weight is read.
  encoder, cfg_text = jtext.build(text_dir, dtype=dtype, abstract=True)
  jtext.load(encoder, text_dir, mesh=mesh, spec_fn=jtext.param_spec,
             memory_kind=text_host, int8=args.int8_weights,
             keep_layers=TEXT_ENCODER_LAYER if args.int8_weights else None)
  print(f"text encoder loaded  {time.perf_counter() - began:.0f}s", flush=True)

  if args.reference:
    # ref2va: `"<Picture 1>: " + 7168 image_pad tokens + prompt`. Vision tower
    # output fills the image_pad slots, deepstack features are added back after
    # the first 3 layers, and mRoPE follows the image grid. Host-side prep comes
    # from diffusers / transformers (`jax_ref2va`); DiT inputs match torchax
    # bit for bit.
    from . import jax_ref2va as r2v
    from . import jax_vision as jvision
    from ..qwen3_vl import jax_vision as _qv
    from jax.sharding import PartitionSpec as _P

    # When streaming from host (--offload / 1 chip) use small vision attention
    # chunks to save HBM; otherwise keep the module default (2048).
    _qv.ATTN_CHUNK = int(os.environ.get(
        "DT_VISION_ATTN_CHUNK", "512" if text_host is not None else str(_qv.ATTN_CHUNK)))
    began_r2v_text = time.perf_counter()
    ref_image = r2v.load_reference(args.reference)
    pres = r2v.presentation(snapshot, args.prompt, ref_image)
    pos3 = r2v.text_positions(snapshot, pres["ids"], pres["grid_thw"],
                              pres["mm_token_type_ids"])
    ids = pres["ids"][None]
    # Host-side per-image work (resize, processor, mRoPE positions): a new
    # reference image pays it on every request, so it is reported too.
    _t = time.perf_counter()
    _ri = r2v.load_reference(args.reference)
    _pr = r2v.presentation(snapshot, args.prompt, _ri)
    r2v.text_positions(snapshot, _pr["ids"], _pr["grid_thw"], _pr["mm_token_type_ids"])
    print(f"  [ref2va] host-side image preparation (warm) "
          f"{time.perf_counter() - _t:.2f}s", flush=True)
    del _ri, _pr
    vision, _ = jvision.build(text_dir, dtype=dtype, abstract=True)
    jvision.load(vision, text_dir, mesh=mesh, spec_fn=lambda k, n: _P(),
                 memory_kind=text_host)
    # Shard vision attention heads across chips (text stage 4.9 -> 1.4 s).
    if os.environ.get("DT_VISION_HEAD_SHARD", "1") == "1" and len(devices) > 1:
      _qv.HEAD_SHARD_AXIS = "tp"
    if text_host is None:
      with jax.set_mesh(mesh):
        prompt_embeds = r2v.encode_text_with_vision(
            encoder, vision, pres, pos3, cfg_text, dtype,
            stop_at_layer=TEXT_ENCODER_LAYER, jit=True)
        prompt_embeds = jax.block_until_ready(prompt_embeds)
        r2v_text_s = time.perf_counter() - began_r2v_text
        # Run again warm (what a serving process pays per request); the first
        # run includes host-side prep and compilation.
        _t = time.perf_counter()
        jax.block_until_ready(r2v.encode_text_with_vision(
            encoder, vision, pres, pos3, cfg_text, dtype,
            stop_at_layer=TEXT_ENCODER_LAYER, jit=True))
        r2v_text_warm = time.perf_counter() - _t
    else:
      eg, est = nnx.split(encoder)
      vg, vst = nnx.split(vision)
      est_dev, vst_dev = to_device(est), to_device(vst)
      x_in = r2v.text_inputs(pres, pos3, cfg_text, dtype)

      @jax.jit
      def _encode_r2v(est_, vst_, x_):
        live_enc = nnx.merge(eg, jax.tree.map(jax.device_put, est_, est_dev))
        live_vis = nnx.merge(vg, jax.tree.map(jax.device_put, vst_, vst_dev))
        return r2v.text_forward(live_enc, live_vis, x_, TEXT_ENCODER_LAYER)

      with jax.set_mesh(mesh):
        prompt_embeds = jax.block_until_ready(_encode_r2v(est, vst, x_in))
        r2v_text_s = time.perf_counter() - began_r2v_text
        _t = time.perf_counter()
        jax.block_until_ready(_encode_r2v(est, vst, x_in))
        r2v_text_warm = time.perf_counter() - _t
      del est, vst, eg, vg, _encode_r2v
    del vision
    print(f"  [ref2va] vision + text tower first {r2v_text_s:.1f}s (incl. host prep and compile), "
          f"warm {r2v_text_warm:.1f}s", flush=True)
    print(f"  [ref2va] reference {ref_image.size[0]}x{ref_image.size[1]}, "
          f"vision tokens {pres['image_token_counts']}", flush=True)
  else:
    began_t2va_text = time.perf_counter()
    ids = np.asarray(tokenizer(args.prompt, return_tensors="np")["input_ids"],
                     np.int32)
    # Text rows carry no spatial position: for a text-only prompt all three
    # mRoPE axes get the same arange, shape `(3, seq)`.
    pos = np.broadcast_to(np.arange(ids.shape[1])[None, :],
                          (3, ids.shape[1]))
    # The tower takes the rotary tables, not the positions; the caller builds
    # the interleaved mRoPE tables.
    from ..qwen3_vl import jax_text as qwen_text

    cos, sin = qwen_text.mrope_cos_sin(
        pos, head_dim=cfg_text["head_dim"],
        rope_theta=cfg_text["rope_theta"],
        mrope_section=cfg_text["mrope_section"])
    cos, sin = jnp.asarray(cos, dtype), jnp.asarray(sin, dtype)
    if text_host is None:
      # One compiled program, not op-by-op dispatch: eager takes 5.8 s for 7
      # tokens, almost all per-op launch overhead.
      _enc_jit = nnx.jit(
          lambda m, i, c, s_: m(i, c, s_, stop_at_layer=TEXT_ENCODER_LAYER))
      with jax.set_mesh(mesh):
        prompt_embeds = _enc_jit(encoder, jnp.asarray(ids), cos, sin)
        prompt_embeds = jax.block_until_ready(prompt_embeds)
      _t = time.perf_counter()
      with jax.set_mesh(mesh):
        jax.block_until_ready(_enc_jit(encoder, jnp.asarray(ids), cos, sin))
      print(f"  [t2va] text tower warm {time.perf_counter() - _t:.2f}s", flush=True)
      del _enc_jit
    else:
      # A `pinned_host` operand makes `dot_general` raise rather than
      # transfer, so the copy is requested explicitly *inside* the jit: done
      # eagerly it would move all 62 GiB at once; inside, XLA overlaps each
      # layer's transfer with the previous layer's compute.
      gd, st = nnx.split(encoder)
      st_dev = to_device(st)

      @jax.jit
      def encode(st, ids, cos, sin):
        live = jax.tree.map(jax.device_put, st, st_dev)
        return nnx.merge(gd, live)(ids, cos, sin,
                                   stop_at_layer=TEXT_ENCODER_LAYER)

      prompt_embeds = jax.block_until_ready(
          encode(st, jnp.asarray(ids), cos, sin))
      del st, gd
  # The 62 GiB tower (7.75 GiB per chip on 8 chips) must be freed before
  # compiling the DiT step or compilation OOMs. Dropping the name is not
  # enough while a jitted closure holds it, hence `clear_caches`.
  del encoder
  jax.clear_caches()
  prompt_embeds = jax.block_until_ready(prompt_embeds)
  if not args.reference:
    # Not included in `generate:` timing; printed for comparison with
    # backends that time the text tower inside it.
    print(f"  [t2va] text tower first {time.perf_counter() - began_t2va_text:.1f}s"
          f" (incl. compile; not part of generate)", flush=True)
  live = sum(x.nbytes for x in jax.live_arrays()) / 2**30
  print(f"  freed the text tower, {live:.1f} GiB still live", flush=True)

  print(f"prompt: {ids.shape[1]} tokens -> {tuple(prompt_embeds.shape)}",
        flush=True)

  if args.init_text:
    # Debug aid: substitute another backend's text embedding to separate
    # text-tower differences from everything downstream.
    ref = np.load(os.path.join(args.init_text, "encoder_hidden_states.npy"))
    mine = np.asarray(prompt_embeds, np.float64)
    if ref.shape != mine.shape:
      raise SystemExit(
          f"[text] shape mismatch: local {mine.shape} vs reference {ref.shape} "
          "(aborting)")
    cos = float((mine.ravel() * ref.astype(np.float64).ravel()).sum()
                / (np.linalg.norm(mine)
                   * np.linalg.norm(ref.astype(np.float64)) + 1e-30))
    print(f"  [text] loaded from {args.init_text} {ref.shape}, vs local "
          f"cos={cos:.6f}", flush=True)
    prompt_embeds = jax.device_put(
        jnp.asarray(ref, prompt_embeds.dtype), prompt_embeds.sharding)

  # -- ref2va: the reference's conditioning latents ---------------------
  # Encoded after the text tower is freed and before the DiT is loaded, so
  # they never share the chips. Same tiling, sampling seed and float16
  # rounding as the reference (99.4% of elements bit-identical, the rest
  # within one float16 ulp).
  cond_rows = None
  # One generator for every host-side draw, in the reference's order.
  import torch as _torch
  tgen = _torch.Generator().manual_seed(args.seed)
  if args.reference:
    import glob as _glob
    from safetensors import safe_open as _safe_open
    import torch as _torch

    began_enc = time.perf_counter()
    vae_dir = os.path.join(snapshot, "vae")
    _params = {}
    for shard in sorted(_glob.glob(os.path.join(vae_dir, "*.safetensors"))):
      with _safe_open(shard, framework="pt") as reader:
        for k in reader.keys():
          if k.startswith(("encoder.", "quant_conv.")):
            _params[k] = reader.get_tensor(k).to(_torch.float32).numpy()
    _ecfg = jvae.encoder_config(vae_dir)
    _enc = jvae.Encoder3d(dtype=jnp.float32, rngs=nnx.Rngs(0), **_ecfg)
    jvae.load_encoder(_enc, _params)
    _qc = jvae.PostQuantConv(_ecfg["latent_channels"] * 2, dtype=jnp.float32,
                             rngs=nnx.Rngs(0))
    jvae.load_quant_conv(_qc, _params, prefix="quant_conv.")
    del _params
    _eg, _es = nnx.split(_enc)
    _qg, _qs = nnx.split(_qc)
    # Replicated, so the tile axis can be split over the chips.
    from jax.sharding import NamedSharding as _NS, PartitionSpec as _PS
    _es = jax.device_put(_es, _NS(mesh, _PS()))
    _qs = jax.device_put(_qs, _NS(mesh, _PS()))

    # Weights as arguments, not closed over (see the decoder below).
    @jax.jit
    def _encode_one(es, qs, tile):  # noqa: E306
      with jax.default_matmul_precision(os.environ.get("DT_VAE_ENC_PRECISION", "high")):
        return nnx.merge(_qg, qs)(nnx.merge(_eg, es)(tile))

    import json as _json
    with open(os.path.join(vae_dir, "config.json")) as f:
      _vcfg = _json.load(f)
    _ENC_ONE[0] = _encode_one
    # Whole-encode jit (DT_ENC_JIT, on by default): 2.8 -> 0.5 s for a
    # 3648x2048 reference, tiles in groups (DT_ENC_GROUP) so it compiles.
    _jit_enc = os.environ.get("DT_ENC_JIT", "1") == "1"
    _enc_w = _ENC_W_FN if _jit_enc else (lambda t: _encode_one(_es, _qs, t))
    cond = r2v.encode_condition(_enc_w, ref_image,
                                _vcfg["latents_mean"], _vcfg["latents_std"],
                                batch=args.vae_tile_batch, mesh=mesh,
                                weights=(_es, _qs) if _jit_enc else None)
    _t = time.perf_counter()
    r2v.encode_condition(_enc_w, ref_image,
                         _vcfg["latents_mean"], _vcfg["latents_std"],
                         batch=args.vae_tile_batch, mesh=mesh,
                         weights=(_es, _qs) if _jit_enc else None)
    r2v_enc_warm = time.perf_counter() - _t
    if os.environ.get("DT_DUMP_COND"):
      np.save(os.environ["DT_DUMP_COND"], cond)
    del _enc, _qc, _es, _qs, _encode_one
    jax.clear_caches()
    # The anchor is noised: x_t = t*x0 + (1-t)*noise at t = 0.999, with noise
    # drawn from the request's generator before the target's
    # (`MiniMaxH3Ref2VAPrepareConditionStep`); drawn with torch on the host
    # to match the reference, then patchified the same way.
    from diffusers.utils.torch_utils import randn_tensor as _randn
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents as _patchify)
    if args.pika_adapter:
      # Pika's anchor noise: its own generator (seed + 7919), drawn in
      # frame-major [B, F, C, H, W] (`evaluation_noise_scope`).
      _c = cond.shape
      _noise = _torch.randn((_c[0], _c[2], _c[1], _c[3], _c[4]),
                            generator=_torch.Generator().manual_seed(args.seed + 7919),
                            dtype=_torch.float32).permute(0, 2, 1, 3, 4)
    else:
      _noise = _randn(tuple(cond.shape), generator=tgen, dtype=_torch.float32)
    _noised = r2v.NOISE_AUG * _torch.from_numpy(cond) + (1 - r2v.NOISE_AUG) * _noise
    cond_rows = _patchify(_noised, (1, 2, 2)).numpy().astype(np.float32)
    r2v_enc_s = time.perf_counter() - began_enc
    print(f"  [ref2va] reference condition {cond.shape} -> {cond_rows.shape[0]} rows, "
          f"first {r2v_enc_s:.1f}s (incl. weight load and compile), warm {r2v_enc_warm:.1f}s",
          flush=True)

  # -- layout ----------------------------------------------------------
  # Pure index arithmetic, no weights: where text, audio and video rows
  # sit in the packed sequence and what mRoPE coordinate each carries.
  # The reference's own routines are reused deliberately -- an off-by-one
  # here is a wrong video, not a crash.
  import torch
  from diffusers.modular_pipelines.minimax_h3.before_denoise import (
      MiniMaxH3PrepareLayoutStep,
  )
  # Import the frame / audio-length helpers rather than restating them.
  # Note `audio_channels` is how many rows an audio latent is packed into,
  # not the feature width (`audio_in_channels`).
  from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
      MINIMAX_H3_AUDIO_CHANNELS,
      align_num_frames,
      audio_latent_num_frames,
      video_latent_num_frames,
  )

  # From the VAE's own config, the way `components` reads them.
  spatial = 16                       # vae.spatial_compression_ratio
  frames_per_chunk = 17              # vae.config.clip_length
  latents_per_chunk = 5              # vae.tokens_chunk_size
  latent_h, latent_w = args.height // spatial, args.width // spatial
  aligned = align_num_frames(args.num_frames, frames_per_chunk,
                             latents_per_chunk)
  if aligned != args.num_frames:
    print(f"  {args.num_frames} frames aligned to {aligned} (the VAE only encodes "
          f"{frames_per_chunk}n+{latents_per_chunk})", flush=True)
    args.num_frames = aligned
  latent_frames = video_latent_num_frames(args.num_frames, frames_per_chunk,
                                          latents_per_chunk)
  audio_latents = audio_latent_num_frames(args.num_frames)
  audio_channels = MINIMAX_H3_AUDIO_CHANNELS
  text_tags = torch.ones(ids.shape[1], dtype=torch.long)

  n_ref = 0
  if args.reference:
    layout = r2v.layout(pres["tags"], pres["references"],
                        r2v.condition_shape(ref_image), args.num_frames,
                        args.height, args.width)
    n_ref = int(layout[5])
    if n_ref != cond_rows.shape[0]:
      raise ValueError(f"layout reserves {n_ref} reference rows, but the condition "
                       f"latent packs into {cond_rows.shape[0]} rows")
  else:
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=latent_frames,
        latent_height=latent_h,
        latent_width=latent_w,
        num_audio_latents=audio_latents,
        patch_size=(1, 2, 2),
        audio_channels=audio_channels,
        audio_tag=2,
        video_tag=0,
    )
  position_ids, token_tags, video_idx, audio_idx, text_idx = layout[:5]
  to_jax = lambda t: jnp.asarray(np.asarray(t))
  # float32, not int32: the reference builds continuous mRoPE coordinates
  # (e.g. `[10.0, 3.905, -5.166]`), and casting to int would truncate them.
  position_ids = to_jax(position_ids).astype(jnp.float32)
  token_tags = to_jax(token_tags).astype(jnp.int32)
  video_idx, audio_idx, text_idx = (to_jax(t).astype(jnp.int32)
                                    for t in (video_idx, audio_idx, text_idx))
  seq = position_ids.shape[0]
  print(f"packed sequence {seq}: text {text_idx.shape[0]}, "
        f"audio {audio_idx.shape[0]}, video {video_idx.shape[0]}", flush=True)

  # -- transformer -----------------------------------------------------
  _, Transformer = dit._build_modules()
  transformer = nnx.eval_shape(
      lambda: Transformer(**dit.CONFIG_FULL, dtype=dtype, rngs=nnx.Rngs(0)))
  # ref2va reads the `transformer_ref` partition: same config, 638 keys
  # by the same names, different weights.
  trunk = "transformer_ref" if args.reference else "transformer"
  lora = None
  if args.pika_adapter:
    lora, pika = dit.load_lora(args.pika_adapter)
    if pika["trunk"] != trunk:
      raise ValueError(f"adapter {pika['task']} wants {pika['trunk']}, "
                       f"this request reads {trunk}")
  adaln_temb = None
  if args.adaln_host:
    _sv, _sa = _step_sigmas(args, pika if args.pika_adapter else None)
    _levels = [[1.0 - _sv[i], 1.0 - _sa[i]]
               + ([max(1.0 - float(_sv[i]), 0.999)] if args.reference else [])
               for i in range(len(_sv) - 1)]
    adaln_temb = dit.host_temb(os.path.join(snapshot, trunk), _levels)
  dit.load_transformer(transformer, os.path.join(snapshot, trunk),
                       mesh=mesh, memory_kind=dit_host,
                       round_bf16=args.bf16_weights, lora=lora,
                       int8=args.int8_weights, adaln_temb=adaln_temb)
  adaln_mods = None
  if args.adaln_host:
    adaln_mods = [jax.device_put(m, jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())) for m in dit.ADALN_MODS.pop("last")]
  del lora
  if args.fp8:
    dit.quantize_transformer_fp8(transformer, mesh=mesh)
  if host is not None:
    _, _probe = nnx.split(transformer)
    _leaves = jax.tree.leaves(_probe)
    _on_host = sum(v.nbytes for v in _leaves
                   if v.sharding.memory_kind == "pinned_host") / 2**30
    _total = sum(v.nbytes for v in _leaves) / 2**30
    print(f"  DiT {_total:.1f} GiB: {_on_host:.1f} on host, "
          f"{_total - _on_host:.1f} resident "
          f"({(_total - _on_host) / len(devices):.2f} GiB a chip)",
          flush=True)
    del _probe, _leaves
  print(f"transformer loaded  {time.perf_counter() - began:.0f}s", flush=True)

  sdpa = None
  if not args.dense_attention:
    from ...kernels import mesh_attention
    from ...kernels import splash as csa

    # Attention dominates the step (~69% on the 345-frame t2va layout), so
    # block sizes are configurable rather than hardcoded.
    local = functools.partial(csa.attention, scale=None,
                              block_q=args.block_q,
                              block_kv=args.block_kv,
                              block_kv_compute=args.block_kv_compute
                              or args.block_kv,
                              use_fixed_m=True,
                              # lossy: Q/K enter the QK matmul as e4m3.
                              qk_fp8=args.qk_fp8 or os.environ.get("DT_QK_FP8") == "1")

    def sdpa(q, k, v):                                    # noqa: F811
      # The decode runs under `default_matmul_precision("highest")`, which
      # Mosaic rejects for the Pallas kernel ("Bad lhs type"); the kernel
      # does its own accumulation, so precision is reset for this call only.
      k = k - jnp.mean(k, axis=2, keepdims=True)
      with jax.default_matmul_precision("default"):
        if args.attn_keep_heads:
          # `_on_mesh` shards heads for the kernel and then constrains the
          # result back to the sequence layout; this drops the output
          # constraint and lets XLA choose the layout.
          from jax.experimental.shard_map import shard_map
          qs, kvs, _ = mesh_attention._partition_specs(mesh, q.shape, True)
          return shard_map(local, mesh=mesh, in_specs=(qs, kvs, kvs),
                           out_specs=qs, check_rep=False)(q, k, v)
        return mesh_attention._on_mesh(local, mesh, q, k, v, True)

  # -- block-sparse temporal window (lossy) ------------------------------
  # Same as torchax's `--sparse-window`: from call `--sparse-dense-steps` on,
  # a video row attends only to video rows within `window` latent frames
  # (text and audio rows still see everything). The table is built for the
  # padded query length against the true key length (padded keys dropped).
  sdpa_sparse = None
  if args.sparse_window and sdpa is not None:
    from ...kernels import sparse_splash
    # Only generated video rows are windowed. The ref2va packing is
    # [text][reference frame rows][audio][generated video]; all rows before
    # the generated video are global (attend to and are attended by all).
    # Including reference rows would make the window cover everything.
    _vi = np.asarray(video_idx)[n_ref:]
    _times = np.asarray(position_ids)[_vi, 0]
    _vs, _rpf = int(_vi.min()), int((_times == _times[0]).sum())
    assert np.all(np.diff(_vi) == 1) and _vi.max() == seq - 1, "generated video rows must be contiguous and at the end of the sequence"
    _qlen = (-(-seq // dit.PAD_SEQUENCE_TO) * dit.PAD_SEQUENCE_TO
             if dit.PAD_SEQUENCE_TO else seq)
    _bq, _bkv = min(args.block_q, _qlen), min(args.block_kv, seq)
    _reach = args.sparse_window * _rpf
    _table = []
    for _i in range(-(-_qlen // _bq)):
      _q0, _q1 = _i * _bq, min((_i + 1) * _bq, _qlen)
      _table.append([
          _j for _j in range(-(-seq // _bkv))
          if _q0 < _vs or _j * _bkv < _vs
          or (_j * _bkv < _q1 + _reach
              and min((_j + 1) * _bkv, seq) > _q0 - _reach)])
    _kept = sum(len(r) for r in _table)
    _full = len(_table) * (-(-seq // _bkv))
    print(f"  [sparse] layout ({_vs}, {_rpf}, {args.sparse_window}), "
          f"kept {_kept}/{_full} kv blocks ({100 * _kept / _full:.0f}%)",
          flush=True)
    if _kept < _full:
      _local_sp = functools.partial(
          sparse_splash.attention, scale=None, block_q=args.block_q,
          block_kv=args.block_kv,
          block_kv_compute=args.block_kv_compute or args.block_kv,
          use_fixed_m=True, qk_fp8=args.qk_fp8, kv_table=_table)

      def sdpa_sparse(q, k, v):
        # The token refiner also calls `sdpa`, over the text alone; the
        # table describes only the packed sequence, so fall back to dense.
        if q.shape[2] != _qlen or k.shape[2] != seq:
          return sdpa(q, k, v)
        k = k - jnp.mean(k, axis=2, keepdims=True)
        with jax.default_matmul_precision("default"):
          return mesh_attention._on_mesh(_local_sp, mesh, q, k, v, True)
    else:
      print("  [sparse] equivalent to dense (window covers the whole sequence), not applied", flush=True)

  # Draw in float32, as `randn_tensor(..., float32)` does in the reference.
  # `jax.random.normal` in bfloat16 samples the underlying uniform at
  # bfloat16 resolution and yields only ~128 distinct values. Latents stay
  # float32 through the loop like the reference; the DiT casts as needed.
  key = jax.random.key(args.seed)
  video = jax.random.normal(
      key, (1, video_idx.shape[0] - n_ref, dit.CONFIG_FULL["in_channels"] * 4),
      jnp.float32)
  audio = jax.random.normal(
      jax.random.fold_in(key, 1),
      (1, audio_idx.shape[0], dit.CONFIG_FULL["audio_in_channels"]),
      jnp.float32)
  if args.torch_noise:
    # The reference's own draws (`MiniMaxH3PrepareLatentsStep`): the video
    # as a latent tensor then patchified, the audio directly in rows, from
    # the request generator after any conditioning noise.
    from diffusers.utils.torch_utils import randn_tensor as _randn2
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents as _patchify2)
    _v = _randn2((1, dit.CONFIG_FULL["in_channels"], latent_frames, latent_h,
                  latent_w), generator=tgen, dtype=_torch.float32)
    _a = _randn2((audio_idx.shape[0], dit.CONFIG_FULL["audio_in_channels"]),
                 generator=tgen, dtype=_torch.float32)
    video = jnp.asarray(_patchify2(_v, (1, 2, 2)).numpy())[None]
    audio = jnp.asarray(_a.numpy())[None]
    if video.shape[1] != video_idx.shape[0] - n_ref:
      raise ValueError(f"torch noise {video.shape} does not match {video_idx.shape[0] - n_ref} rows")
  pika_eps = None
  if args.pika_adapter:
    # Pika's draws (`sample_evaluation_ladder`), on the host: video then
    # audio from seed, in its external layouts ([B, F, 24, H, W] and
    # [B, T, 64]); the re-noising eps from seed + 1, video then audio per
    # step. Same generators on CPU means a reference run on any backend
    # can see exactly these numbers.
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents as _patchify3)
    _vs = (1, latent_frames, dit.CONFIG_FULL["in_channels"], latent_h, latent_w)
    _T = audio_idx.shape[0] // 2
    _ach = dit.CONFIG_FULL["audio_in_channels"]

    def _vrows(x):
      return jnp.asarray(_patchify3(x.permute(0, 2, 1, 3, 4), (1, 2, 2))
                         .reshape(1, -1, 4 * _vs[2]).numpy())

    def _arows(x):
      return jnp.asarray(x.view(1, _T, 2, _ach).permute(0, 2, 1, 3)
                         .reshape(1, 2 * _T, _ach).numpy())

    _g = _torch.Generator().manual_seed(args.seed)
    video = _vrows(_torch.randn(_vs, generator=_g, dtype=_torch.float32))
    audio = _arows(_torch.randn((1, _T, 2 * _ach), generator=_g,
                                dtype=_torch.float32))
    _g = _torch.Generator().manual_seed(args.seed + 1)
    pika_eps = []
    for _ in range(len(pika["sampling"]["knots"]) - 2):
      _ev = _vrows(_torch.randn(_vs, generator=_g, dtype=_torch.float32))
      _ea = _arows(_torch.randn((1, _T, 2 * _ach), generator=_g,
                                dtype=_torch.float32))
      pika_eps.append((_ev, _ea))
  if n_ref:
    # ref2va: the reference's rows lead the video block, held at t = 0.999.
    video = jnp.concatenate([jnp.asarray(cond_rows)[None], video], axis=1)
  if args.init_latents:
    v = np.load(os.path.join(args.init_latents, "hidden_states.npy"))
    a = np.load(os.path.join(args.init_latents, "audio_hidden_states.npy"))
    if v.shape != video.shape or a.shape != audio.shape:
      raise ValueError(f"shape mismatch: video {v.shape} vs {video.shape}, "
                       f"audio {a.shape} vs {audio.shape}")
    video = jnp.asarray(v, jnp.float32)
    audio = jnp.asarray(a, jnp.float32)
    print(f"  [init] initial latents from {args.init_latents}", flush=True)

  # Video and audio rows follow different schedules. The reference builds a
  # per-row timestep (`build_row_timesteps`) and hands the transformer the
  # distinct values plus a per-row index.
  sigmas, a_sigmas = _step_sigmas(args, pika if args.pika_adapter else None)
  if args.pika_adapter:
    args.cache_threshold = 0.0
  timesteps = 1.0 - sigmas[:-1]
  a_timesteps = 1.0 - a_sigmas[:-1]
  if len(a_timesteps) != len(timesteps):
    raise ValueError(
        f"schedules differ in length: video {len(timesteps)}, audio {len(a_timesteps)}")
  # Row 0 of `timestep` is the video level, row 1 the audio one;
  # `adaln_indices = timestep_index * MODALITY_NUM + tag` picks the pair.
  # Only index and value must agree; `torch.unique` ordering is not needed.
  row_step = jnp.zeros((seq,), jnp.int32).at[audio_idx].set(1)
  if n_ref:
    # The reference rows' own level, `max(t, 0.999)`, as a third entry.
    # Fixed at shape (3,) so every step uses one program.
    row_step = row_step.at[video_idx[:n_ref]].set(2)

  def step_levels(i):
    lv = [timesteps[i], a_timesteps[i]]
    if n_ref:
      lv.append(max(float(timesteps[i]), r2v.NOISE_AUG))
    return jnp.asarray(lv, jnp.float32)

  def pika_step(x, v, sigma, sigma_next, eps):
    """Re-noising step: x0 = x + sigma v, then fresh noise at sigma_next."""
    x0 = x.astype(jnp.float32) + float(sigma) * v.astype(jnp.float32)
    if eps is None:
      return x0
    return (1.0 - float(sigma_next)) * x0 + float(sigma_next) * eps

  def video_step(video, dv, i):
    """Euler step on the generated rows only; reference rows are held."""
    if pika_eps is not None:
      eps = pika_eps[i][0] if i < len(pika_eps) else None
      gen = pika_step(video[:, n_ref:], dv[:, n_ref:], sigmas[i],
                      sigmas[i + 1], eps)
      return jnp.concatenate([video[:, :n_ref], gen], axis=1) if n_ref else gen
    if not n_ref:
      return scheduler_step(video, dv, float(timesteps[i]), float(sigmas[i]),
                            float(sigmas[i + 1]))
    gen = scheduler_step(video[:, n_ref:], dv[:, n_ref:], float(timesteps[i]),
                         float(sigmas[i]), float(sigmas[i + 1]))
    return jnp.concatenate([video[:, :n_ref], gen], axis=1)
  print(f"{len(timesteps)} denoise calls  video shift {args.shift} / "
        f"audio shift {args.audio_shift}", flush=True)

  graphdef, state = nnx.split(transformer)
  state_dev = to_device(state) if host is not None else None

  # Contiguous runs of a constant AdaLN index, so a block can broadcast one
  # modulation row per run instead of gathering per token. Off by default on
  # multiple chips: slicing short runs out of a sequence-sharded tensor and
  # scattering them back made the request 2.8x slower (658.1 vs 233.9 s).
  _idx = (np.asarray(row_step) * dit.MODALITY_NUM
          + np.asarray(token_tags)).astype(np.int64)
  _edges = np.flatnonzero(_idx[1:] != _idx[:-1]) + 1
  _bounds = [0, *_edges.tolist(), int(_idx.shape[0])]
  segments = tuple((a, b, int(_idx[a])) for a, b in zip(_bounds, _bounds[1:]))
  if not (args.segments or len(devices) == 1):
    segments = None
  else:
    print(f"  [segments] {segments}", flush=True)

  def make_call(attn):
    @functools.partial(jax.jit, static_argnums=())
    def call(state, video, audio, text, t, mods=None):
      if host is not None:
        state = jax.tree.map(jax.device_put, state, state_dev)
      model = nnx.merge(graphdef, state)
      return model(video, audio, text, t, row_step, token_tags, position_ids,
                   video_idx, audio_idx, text_idx, sdpa=attn,
                   segments=segments, adaln_mods=mods)
    return call

  call = make_call(sdpa)
  call_sparse = make_call(sdpa_sparse) if sdpa_sparse is not None else None

  # `with_sharding_constraint` in the DiT needs the mesh in context.
  mesh_ctx = jax.set_mesh(mesh)
  mesh_ctx.__enter__()

  if args.warmup:
    # One call of each program, outside the measurement. Shapes must match
    # the request's exactly, or the request recompiles on its first step.
    warm = time.perf_counter()
    tt0 = step_levels(0)
    v0, a0 = video, audio
    for _ in range(2):
      dv, da = call(state, v0, a0, prompt_embeds, tt0,
                    None if adaln_mods is None else adaln_mods[0])
      v0 = video_step(v0, dv, 0)
      a0 = scheduler_step(a0, da, float(timesteps[0]), float(sigmas[0]),
                          float(sigmas[1]))
      jax.block_until_ready(v0)
    if call_sparse is not None:
      # Use the *stepped* latents: they come out of a jitted step with a
      # different sharding than the initial ones, which is what the
      # request's sparse calls see (otherwise they recompile mid-request).
      jax.block_until_ready(call_sparse(
          state, v0, a0, prompt_embeds, tt0,
          None if adaln_mods is None else adaln_mods[0]))
    print(f"  warmup {time.perf_counter() - warm:.1f}s", flush=True)
    del v0, a0, dv, da

  # `generate:` times the request, not the process (no weight loading or saving).
  # -- step cache (lossy) ----------------------------------------------
  # Same indicator as the other two backends: the time embedding of the
  # call's distinct timestep levels, averaged, with its relative drift summed
  # since the last call that ran. Computed on the host (a few MFLOP) so it
  # stays out of the compiled step. `np.unique` matches the reference's
  # `torch.unique` of the per-row levels.
  cache = None
  if args.cache_threshold > 0:
    te = transformer.time_embedder
    _w = [np.asarray(a, np.float64) for a in (
        te.linear_1.weight[...], te.linear_1.bias[...],
        te.linear_2.weight[...], te.linear_2.bias[...])]
    _half = dit.CONFIG_FULL["freq_dim"] // 2
    _freqs = np.exp(-np.log(10000.0) * np.arange(_half) / _half)

    def indicator(levels):
      u = np.unique(np.asarray(levels, np.float64))
      e = u[:, None] * _freqs[None, :]
      e = np.concatenate([np.cos(e), np.sin(e)], -1)
      h = e @ _w[0].T + _w[1]
      h = h / (1 + np.exp(-h))
      return (h @ _w[2].T + _w[3]).mean(0)

    cache = {"last_mod": None, "last_out": None, "drift": 0.0, "skipped": 0}

  # --runs N: run N requests in one process (same prompt / reference; text
  # conditioning computed once). With N > 1 the DiT is kept through decode and
  # the decoder is tensor parallel (a replicated decoder does not fit beside
  # the DiT), built once. Runs 2+ show resident-server steady state.
  if args.runs > 1:
    os.environ["DT_KEEP_DIT"] = "1"
  video0, audio0 = video, audio
  cache_on = cache is not None
  for run in range(args.runs):
    if run:
      # A fresh context each time: re-entering the one exited at the end
      # of the previous decode leaves no mesh in context.
      mesh_ctx = jax.set_mesh(mesh)
      mesh_ctx.__enter__()
    video, audio = video0, audio0
    if cache_on:
      cache = {"last_mod": None, "last_out": None, "drift": 0.0, "skipped": 0}
    began_request = time.perf_counter()
    began_denoise = time.perf_counter()
    _prof = os.environ.get("H3_PROFILE_DIR") or args.profile_dir
    _prof_step = int(os.environ.get("H3_PROFILE_STEP", args.profile_step))
    for i, t in enumerate(timesteps):
      tt = step_levels(i)
      # `H3_PROFILE_DIR=<dir>`: profile a single step (default step 8, after
      # sparsity and step skipping have kicked in).
      if _prof and i == _prof_step:
        jax.profiler.start_trace(_prof)
      if os.environ.get("H3_DUMP_EVERY"):
        _e = os.environ["H3_DUMP_EVERY"]
        os.makedirs(_e, exist_ok=True)
        np.save(os.path.join(_e, f"hs_{i:03d}.npy"),
                np.asarray(video, np.float32))
      if i == int(os.environ.get("H3_DUMP_STEP", "0")) and os.environ.get(
          "H3_DUMP_INPUTS"):
        _d = os.environ["H3_DUMP_INPUTS"]
        os.makedirs(_d, exist_ok=True)
        for _k, _v in (("hidden_states", video), ("audio_hidden_states", audio),
                       ("encoder_hidden_states", prompt_embeds),
                       ("timestep", tt), ("timestep_indices", row_step),
                       ("token_tags", token_tags), ("position_ids", position_ids),
                       ("video_indices", video_idx),
                       ("audio_indices", audio_idx), ("text_indices", text_idx)):
          np.save(os.path.join(_d, f"{_k}.npy"), np.asarray(_v, np.float32))
        print(f"  [dump] saved 10 tensors to {_d}", flush=True)
      skip = False
      if cache is not None:
        cur = indicator(np.asarray(tt))
        if cache["last_mod"] is not None and cache["last_out"] is not None:
          cache["drift"] += float(np.abs(cur - cache["last_mod"]).mean()
                                  / (np.abs(cache["last_mod"]).mean() + 1e-9))
          if cache["drift"] < args.cache_threshold:
            skip = True
          else:
            cache["drift"] = 0.0
        if not skip:
          cache["last_mod"] = cur
      if skip:
        dv, da = cache["last_out"]
        cache["skipped"] += 1
      else:
        fn = (call_sparse if call_sparse is not None
              and i >= args.sparse_dense_steps else call)
        dv, da = fn(state, video, audio, prompt_embeds, tt,
                    None if adaln_mods is None else adaln_mods[i])
        if cache is not None:
          cache["last_out"] = (dv, da)
      video = video_step(video, dv, i)
      if pika_eps is not None:
        audio = pika_step(audio, da, a_sigmas[i], a_sigmas[i + 1],
                          pika_eps[i][1] if i < len(pika_eps) else None)
      else:
        audio = scheduler_step(audio, da, float(a_timesteps[i]),
                               float(a_sigmas[i]), float(a_sigmas[i + 1]))
      jax.block_until_ready(video)
      if _prof and i == _prof_step:
        jax.profiler.stop_trace()
        print(f"  [profile] step {i} -> {_prof}", flush=True)
      print(f"  step {i + 1}/{len(timesteps)}"
            f"  {time.perf_counter() - began_denoise:.1f}s", flush=True)

    denoise_wall = time.perf_counter() - began_denoise
    if cache is not None:
      print(f"  [step-cache] threshold {args.cache_threshold}: {len(timesteps)} calls, "
            f"skipped {cache['skipped']}", flush=True)
      cache = None
    # Free the DiT before building the decoder: the jitted function and its
    # graphdef hold references too, so delete all of them and clear caches.
    video = jax.device_get(video)
    if os.environ.get("DT_KEEP_DIT") == "1":
      # Keep the DiT resident through decode (needed to serve multiple requests).
      live = sum(x.nbytes for x in jax.live_arrays()) / 2**30
      print(f"  [keep-dit] DiT kept on chip for decode, {live:.1f} GiB live (all chips)",
            flush=True)
    else:
      del state, transformer, call, call_sparse, graphdef
      jax.clear_caches()
      live = sum(x.nbytes for x in jax.live_arrays()) / 2**30
      print(f"  freed the transformer, {live:.1f} GiB still live", flush=True)
    video = jnp.asarray(video)

    # -- decode ----------------------------------------------------------
    # Decoder setup (weight read + compile) happens here, after the DiT is
    # freed (both do not fit at 768p). It is measured and subtracted from the
    # request time rather than moved before `began_request`.
    began_setup = time.perf_counter()
    if run == 0:
      decoder = nnx.eval_shape(
          lambda: jvae.ViTDecoder(**jvae.CONFIG, dtype=dec_dtype,
                                  rngs=nnx.Rngs(0)))
      from safetensors import safe_open
      import glob as _glob

      # The reference decodes `decoder(post_quant_conv(z))`; skipping
      # post_quant_conv silently breaks real-weight output.
      post_q = jvae.PostQuantConv(jvae.CONFIG["in_channels"], dtype=dec_dtype,
                                  rngs=nnx.Rngs(0))
      params, pq_seen = {}, 0
      for shard in sorted(_glob.glob(os.path.join(snapshot, "vae",
                                                  "*.safetensors"))):
        with safe_open(shard, framework="flax") as reader:
          for k in reader.keys():
            if k.startswith("decoder."):
              params[k] = reader.get_tensor(k)
            elif k.startswith("post_quant_conv."):
              leaf = getattr(post_q, k.split(".")[-1])
              leaf.value = jnp.asarray(reader.get_tensor(k)).astype(
                  leaf.value.dtype).reshape(leaf.value.shape)
              pq_seen += 1
      if pq_seen != 2:
        raise ValueError(f"post_quant_conv should have 2 tensors, read {pq_seen}")
      # DT_DECODE_TILE_SHARD=1 (default): replicate decoder weights and shard
      # tiles across chips, as in the reference encode. Combined with batched
      # chunks this cuts a 345-frame decode from 12.8 to 7.7 s with identical
      # output. DT_DECODE_TILE_SHARD=0 uses tensor-parallel weights instead.
      tile_shard = os.environ.get(
          "DT_DECODE_TILE_SHARD", "1" if args.decode_tile_shard else "0") == "1"
      filled, skipped = jvae.load_decoder(decoder, params,
                                          mesh=None if tile_shard else mesh)
      print(f"vae decoder: {filled} tensors, skipped {skipped}", flush=True)

    # Inverse of the patchify, matching the reference
    # (`modular_pipelines/minimax_h3/decoders.py`): the latent row is outside
    # the patch row, `H index = h * 2 + ph`.
    # ref2va: drop the reference rows before unpatchifying.
    video = video[:, n_ref:]
    latents = video.reshape(1, latent_frames, latent_h // 2, latent_w // 2,
                            dit.CONFIG_FULL["in_channels"], 2, 2)
    latents = jnp.transpose(latents, (0, 4, 1, 2, 5, 3, 6)).reshape(
        1, dit.CONFIG_FULL["in_channels"], latent_frames, latent_h, latent_w)
    # Un-normalise the latents before decoding, as the reference does.
    import json as _json

    with open(os.path.join(snapshot, "vae", "config.json")) as f:
      vae_cfg = _json.load(f)
    lat_mean = jnp.asarray(vae_cfg["latents_mean"], jnp.float32).reshape(
        1, -1, 1, 1, 1)
    lat_std = jnp.asarray(vae_cfg["latents_std"], jnp.float32).reshape(
        1, -1, 1, 1, 1)
    latents = latents.astype(jnp.float32) * lat_std + lat_mean
    if args.dump_latents:
      # Saved before `--decode-npy` swaps the latent, so a run always records
      # its own. Same tensor the traced backend's `--dump-latents` saves (the
      # un-normalised input to `vae.decode`).
      np.save(args.dump_latents, np.asarray(latents, np.float32))
      print(f"  [latents] {tuple(latents.shape)} -> {args.dump_latents}",
            flush=True)
    if args.decode_npy:
      # Debug aid: decode a fixed latent so only the decoder varies.
      latents = jnp.asarray(np.load(args.decode_npy), jnp.float32)
      print(f"  [decode-npy] latent from {args.decode_npy} "
            f"{tuple(latents.shape)}, replacing the denoised one", flush=True)
    # The decoder uses the same attention kernel as the DiT: without `sdpa`
    # it falls back to a dense einsum over one token per latent voxel, whose
    # score matrix does not fit on any chip. The output is explicitly
    # replicated, since its leading dimension (1) cannot be split across chips.
    if run == 0:
      from jax.sharding import NamedSharding, PartitionSpec as P

      # Pass weights as arguments: a decoder captured from the enclosing scope
      # gets its 4.85 GB of weights baked into the program as constants (decode
      # went from 0.3 s of arithmetic to 119 s).
      dec_graph, dec_state = nnx.split(decoder)
      pq_graph, pq_state = nnx.split(post_q)
      if tile_shard:
        from jax.sharding import NamedSharding as _NS2, PartitionSpec as _PS2
        dec_state = jax.device_put(dec_state, _NS2(mesh, _PS2()))
        pq_state = jax.device_put(pq_state, _NS2(mesh, _PS2()))

      @functools.partial(jax.jit, out_shardings=NamedSharding(mesh, P()))
      def decode(state, z, pq=pq_state):
        # `decoder(post_quant_conv(z))`, as in the reference. `highest` makes
        # float32 matmuls run at full precision on TPU (otherwise bfloat16);
        # it has no effect on a bfloat16 decode.
        with jax.default_matmul_precision("highest"):
          z = nnx.merge(pq_graph, pq)(z)
          return nnx.merge(dec_graph, state)(z, sdpa=sdpa)
      # -- spatial tiling ---------------------------------------------------
      # The reference decodes each clip in 256x256 pixel tiles with a 64-pixel
      # minimum overlap and blends them; the released frames depend on this,
      # so whole-frame decoding would change the output. The geometry lives in
      # `jax_vae`. --vae-tile-batch N vmaps N tiles per call (same arithmetic).
      def decode_tiled(state, z):
        return jvae.tiled_decode(lambda t: decode(state, t), z,
                                 tile_px=args.vae_tile_px,
                                 overlap_px=args.vae_tile_overlap_px,
                                 batch=args.vae_tile_batch,
                                 mesh=mesh if tile_shard else None)

      # Temporal chunking bounds attention and intermediates (the decoder is a
      # 36-layer transformer over one token per latent voxel). The chunk size
      # is fixed by the VAE geometry at `tokens_chunk + token_overlap` latents;
      # `--vae-clip` is accepted only for backward compatibility.
      WARM_CHUNK = 7                       # tokens_chunk 5 + token_overlap 2
      if args.vae_clip and args.vae_clip != WARM_CHUNK:
        print(f"  --vae-clip {args.vae_clip} ignored: the VAE's temporal geometry fixes "
              f"{WARM_CHUNK} latents per chunk", flush=True)
      if args.warmup:
        # Warm the decode program too (it is built after the denoise loop).
        warm = time.perf_counter()
        # 1 chip decodes one chunk at a time to save HBM; multi-chip picks K
        # automatically (decode_chunk_batch).
        _cb = decode_chunk_batch(latents.shape[2], int(os.environ.get(
            "DT_DECODE_CHUNK_BATCH",
            args.decode_chunk_batch or (1 if mesh.devices.size == 1 else 0))))
        jax.block_until_ready(
            decode_tiled(dec_state,
                         jnp.concatenate([latents[:, :, :WARM_CHUNK]] * _cb,
                                         axis=0).astype(dec_dtype)))
        print(f"  decode warmup {time.perf_counter() - warm:.1f}s", flush=True)
    decoder_setup = time.perf_counter() - began_setup
    began_decode = time.perf_counter()
    # Follows `AutoencoderKLMiniMaxH3._decode`: the VAE encodes 17 pixel
    # frames per chunk and drops `token_drop = 3` trailing latents, so
    # consecutive decoded chunks overlap and are cross-faded, and each carries
    # an implicit leading pad that is cut.
    clip_length, token_drop, ratio = 17, 3, 4
    tokens_chunk = -(-clip_length // ratio)          # ceil -> 5
    pre_pad = (-clip_length) % ratio                 # 3
    tok_overlap = (-token_drop) % tokens_chunk       # 2
    frame_overlap = max(tok_overlap * ratio - pre_pad, 0)   # 5
    chunk_frames = tokens_chunk * ratio              # 20

    def blend(a, b, extent):
      """Linear cross-fade over the temporal axis, as `_blend` does."""
      extent = min(a.shape[2], b.shape[2], extent)
      if extent <= 0:
        return b
      w = jnp.arange(extent, dtype=b.dtype).reshape(1, 1, extent, 1, 1) / extent
      head = a[:, :, -extent:] * (1.0 - w) + b[:, :, :extent] * w
      return (head if extent == b.shape[2]
              else jnp.concatenate([head, b[:, :, extent:]], axis=2))

    frames_lat = latents.shape[2]
    num_tokens = frames_lat + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk - int(token_drop > 0)
    z = latents
    if pad_tokens:
      z = jnp.concatenate(
          [z, jnp.repeat(z[:, :, -1:], pad_tokens, axis=2)], axis=2)

    # DT_DECODE_TIMING=1 splits decode time between decode_tiled and the
    # slicing / blending. It blocks after each call, which removes async
    # overlap, so only the relative split is meaningful.
    dt_timing = os.environ.get("DT_DECODE_TIMING") == "1"
    t_dec = t_blend = 0.0
    per_call = []
    pieces, carry = [], None
    # DT_DECODE_CHUNK_BATCH=K stacks K temporal chunks along the batch axis per
    # decode call. Chunks are independent (cross-fading happens afterwards).
    # 1 chip decodes one chunk at a time to save HBM; multi-chip picks K
    # automatically (decode_chunk_batch).
    chunk_batch = decode_chunk_batch(frames_lat, int(os.environ.get(
        "DT_DECODE_CHUNK_BATCH",
        args.decode_chunk_batch or (1 if mesh.devices.size == 1 else 0))))
    print(f"  decode: {num_chunks} chunks, {chunk_batch} per call", flush=True)
    decoded = {}

    def chunk_out(i):
      if i not in decoded:
        group = list(range(i, min(i + chunk_batch, num_chunks)))
        zs = [z[:, :, g * tokens_chunk:g * tokens_chunk + tokens_chunk + tok_overlap]
              for g in group]
        zs += [zs[-1]] * (chunk_batch - len(zs))
        out_g = decode_tiled(dec_state,
                             jnp.concatenate(zs, axis=0).astype(dec_dtype))
        for k, g in enumerate(group):
          decoded[g] = out_g[k:k + 1]
      return decoded.pop(i)

    for i in range(num_chunks):
      s = i * tokens_chunk
      _t0 = time.perf_counter() if dt_timing else 0.0
      out = chunk_out(i)
      if dt_timing:
        out = jax.block_until_ready(out)
        _t1 = time.perf_counter(); t_dec += _t1 - _t0
        per_call.append(_t1 - _t0)
      for j in range(int(token_drop > 0) + 1):
        f = j * chunk_frames
        piece = out[:, :, f:f + chunk_frames][:, :, pre_pad:]
        if j == 0:
          pieces.append(piece if carry is None
                        else blend(carry, piece, frame_overlap))
        else:
          carry = piece
      if dt_timing:
        carry = jax.block_until_ready(carry) if carry is not None else carry
        if pieces:
          pieces[-1] = jax.block_until_ready(pieces[-1])
        t_blend += time.perf_counter() - _t1
    if carry is not None:
      pieces.append(carry)
    frames = jnp.concatenate(pieces, axis=2)
    if pad_tokens:
      intra_tail = clip_length % ratio
      before = z.shape[2] - pad_tokens
      pad_frames = sum(
          intra_tail if intra_tail and (before + k) % tokens_chunk == 0 else ratio
          for k in range(pad_tokens))
      frames = frames[:, :, :-pad_frames]
    frames = jax.block_until_ready(frames)
    del pieces, z
    decode_wall = time.perf_counter() - began_decode
    if dt_timing:
      print(f"  [decode timing] decode_tiled {t_dec:.2f}s  "
            f"slice+blend {t_blend:.2f}s  (serialized; sum >= normal wall time)",
            flush=True)
      # Per-call times separate fixed per-call overhead from arithmetic cost.
      if per_call:
        print("  [decode timing] per call "
              + " ".join(f"{x:.2f}" for x in per_call)
              + f"  (first {per_call[0]:.2f}s, median of rest "
              + f"{sorted(per_call[1:])[len(per_call[1:]) // 2]:.2f}s)"
              if len(per_call) > 1 else "", flush=True)
    print(f"  decode: {frames_lat} latents -> {frames.shape[2]} frames "
          f"({num_chunks} chunks of {tokens_chunk}+{tok_overlap} latents, "
          f"overlap {frame_overlap} frames)", flush=True)
    mesh_ctx.__exit__(None, None, None)

    # Back to pixels: undo ImageNet normalisation, then clamp.
    pixel_mean = jnp.asarray((0.485, 0.456, 0.406), jnp.float32).reshape(
        1, -1, 1, 1, 1)
    pixel_std = jnp.asarray((0.229, 0.224, 0.225), jnp.float32).reshape(
        1, -1, 1, 1, 1)
    if len(devices) == 1:
      # One small chip (v5e: 16 GB) cannot hold a float32 copy of the clip
      # beside the decoder; finish on the host, a few frames at a time.
      _f = np.asarray(frames)[0]
      _m = np.asarray(pixel_mean).reshape(-1, 1, 1, 1)
      _sd = np.asarray(pixel_std).reshape(-1, 1, 1, 1)
      out_frames = np.empty(_f.shape[1:] + (_f.shape[0],), np.uint8)
      for _a in range(0, _f.shape[1], 16):
        _c = np.clip(_f[:, _a:_a + 16].astype(np.float32) * _sd + _m, 0.0, 1.0)
        out_frames[_a:_a + 16] = np.round(
            np.transpose(_c, (1, 2, 3, 0)) * 255.0).astype(np.uint8)
      frames = out_frames
    else:
      frames = jnp.clip(frames.astype(jnp.float32) * pixel_std + pixel_mean,
                        0.0, 1.0)
      # uint8, matching the torch pipeline's output (and 4x smaller than float32).
      frames = np.asarray(
          jnp.round(jnp.transpose(frames[0], (1, 2, 3, 0)) * 255.0), np.uint8)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}_frames.npy")
    # `generate:` is the per-request time. One-off decoder setup is
    # subtracted and printed separately.
    wall = time.perf_counter() - began_request
    request = wall - decoder_setup
    np.save(path, frames)
    print(f"generate: {request:.1f}s" + (f"  [run {run}]" if args.runs > 1 else ""),
          flush=True)
    if args.reference:
      # The ref2va encodes run before the DiT is loaded, outside `generate`;
      # the traced backends count them inside, so print both.
      print(f"  [ref2va] request incl. encodes (warm): {request + r2v_text_warm + r2v_enc_warm:.1f}s"
            f" = generate {request:.1f} + text/vision {r2v_text_warm:.1f}"
            f" + reference encode {r2v_enc_warm:.1f}", flush=True)
    print(f"  denoise {denoise_wall:.1f}s, decode {decode_wall:.1f}s, "
          f"load+compile before the request {began_request - began:.1f}s",
          flush=True)
    print(f"  (wall {wall:.1f}s, of which one-off decoder setup {decoder_setup:.1f}s "
          f"is excluded from generate)", flush=True)
    print(f"wrote {path}  {tuple(frames.shape)}", flush=True)


if __name__ == "__main__":
  main()
