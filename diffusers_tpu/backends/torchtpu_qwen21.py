"""Qwen-Image-2.1 on TPU via torch_tpu.

Third backend for the same model, alongside ``backends/torchax_qwen21.py``
(torchax). The two reach the device very differently and the difference
decides how much work each one is:

* **torchax** traces the PyTorch model into one XLA graph. That is what makes
  it fast, and it is also why three parts of this model had to be
  restructured -- the DiT's prologue, Qwen3-VL's, and the vision tower's
  attention all derive shapes from tensor *contents*, which a trace cannot
  do.
* **torch_tpu** runs the model eagerly with deferred execution. Nothing is
  traced, so none of those restructurings are needed and the stock diffusers
  model runs as written. The price is that each operator becomes its own
  program.

So this script is short where the torchax one is long, and that is the
interesting part of the comparison rather than an accident.

**One chip, nothing sharded.** The DiT is 14.2 GB in bfloat16 and the VAE
0.7, which fits a v6e chip's 31 GB with the text encoder left on the host;
adding Qwen3-VL's 17.5 GB would not. Tensor-parallel across eight chips means
rewriting the DiT against column- and row-parallel linears, which is a
separate job -- see the notes at the bottom.

Needs diffusers main (the QwenImage21 classes are in no release) and
transformers >= 5.17.

    TORCH_TPU_TOPOLOGY=1,1,1 \\
    TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:<free port> \\
    python backends/torchtpu_qwen21.py --prompt "..."

Both environment variables are required even for a single chip: without them
PjRt refuses to start with "missing required environment variables for
distributed training". ``_auto_env`` fills them in by default; ``--no_auto_env`` opts out.
"""

import argparse
import os
import socket
import time


def _auto_env():
  """Fills in what torch_tpu's PjRt client insists on, for one chip."""
  os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = f"localhost:{port}"
  os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", str(time.time_ns()))
  # The default is 0, which halves host-to-device bandwidth; 16 GiB is where
  # it saturates on v6e.
  os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(16 << 30))
  # The stock LIBTPU_INIT_ARGS some launchers set includes
  # --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true, which
  # aborts at PjRt client creation on v6e ("platform ghostlite").
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      "--xla_tpu_dvfs_p_state=7 "
      "--xla_tpu_enable_latency_hiding_scheduler=true "
      "--xla_tpu_spmd_rng_bit_generator_unsafe=true "
      "--xla_tpu_enable_dot_strength_reduction=true",
  )


def shard_qwen21(transformer, rank, world, all_reduce):
  """Tensor-parallel the stock diffusers Qwen-Image-2.1 DiT in place, on the host.

  Per block: to_q/to_k/to_v and img_mlp.{proj,gate_layer} keep this rank's
  output slice (one head group), attn.to_out[0] and img_mlp.out keep the input
  slice and all-reduce their output. 32 heads -> 4 per rank on eight chips.
  """
  from .tp_h3 import _column, _divide, _row

  for block in transformer.transformer_blocks:
    attn = block.attn
    for name in ("to_q", "to_k", "to_v"):
      _column(getattr(attn, name), rank, world)
    _row(attn.to_out[0], rank, world, all_reduce)
    attn.heads = _divide(attn.heads, world)
    if hasattr(attn, "inner_dim"):
      attn.inner_dim = _divide(attn.inner_dim, world)
    _column(block.img_mlp.proj, rank, world)
    _column(block.img_mlp.gate_layer, rank, world)
    _row(block.img_mlp.out, rank, world, all_reduce)
  return len(transformer.transformer_blocks)


_CACHES = [None]


def install_ulysses(transformer, full_blocks, rank, world, group_name, verbose=print,
                    aot_prefill=False):
  """Cached steps: all 32 blocks in one AOT program, Ulysses sequence parallel
  (see `qwen21_core_tt.UlyssesGroup`). The prefill stays tensor parallel."""
  import torch
  from .aot_dit import AotBlocks
  from .qwen21_core_tt import UlyssesGroup

  unit = UlyssesGroup(full_blocks, rank, world, group_name)
  AotBlocks([unit]).install()
  blocks = list(transformer.transformer_blocks)
  extract_units = None
  if aot_prefill:
    from .qwen21_core_tt import ExtractCore
    extract_units = [ExtractCore(b) for b in blocks]
    AotBlocks(extract_units).install()
  for index, block in enumerate(blocks):
    original = block.forward

    def forward(hidden_states, modulation, rotary_emb=None, attention_mask=None,
                target_token_mask=None, layer_cache=None, kv_cache_mode=None,
                cache_write_slice=None, segments=None, key_valid=None,
                _original=original, _index=index):
      cached = (kv_cache_mode == "cached" and layer_cache is not None
                and attention_mask is None and key_valid is None
                and hidden_states.shape[1] % world == 0)
      if cached:
        if _index:
          return hidden_states          # block 0 already ran all of them
        # Contiguous copies: the inputs can be views into bigger buffers (the
        # target rows sliced off the prefix), and a compiled program that slices
        # them again rejects a base buffer of a different size.
        c = lambda t: None if t is None else t.clone()   # contiguous() keeps a contiguous view
        kvs = [c(t) for i in range(len(blocks)) for t in _CACHES[0].get_layer(i).get()]
        return unit(c(hidden_states), c(modulation), c(rotary_emb),
                    c(target_token_mask), *kvs)
      if (extract_units is not None and kv_cache_mode == "extract"
          and layer_cache is not None and attention_mask is None
          and key_valid is None and segments is not None and len(segments) == 1
          and segments[0][0] == 0 and segments[0][2]
          and cache_write_slice == slice(0, segments[0][1])):
        n = segments[0][1]
        text_mask = torch.tril(torch.ones(n, n, dtype=torch.bool,
                                          device=hidden_states.device))[None, None]
        out, k_p, v_p = extract_units[_index](
            hidden_states.clone(), modulation.clone(), rotary_emb.clone(),
            None if target_token_mask is None else target_token_mask.clone(), text_mask)
        layer_cache.store(k_p, v_p)
        return out
      return _original(hidden_states, modulation, rotary_emb=rotary_emb,
                       attention_mask=attention_mask,
                       target_token_mask=target_token_mask,
                       layer_cache=layer_cache, kv_cache_mode=kv_cache_mode,
                       cache_write_slice=cache_write_slice, segments=segments,
                       key_valid=key_valid)

    block.forward = forward
  t_forward = transformer.forward

  def transformer_forward(*a, **kw):
    _CACHES[0] = kw.get("kv_cache")
    return t_forward(*a, **kw)

  transformer.forward = transformer_forward
  verbose(f"  [ulysses] {len(blocks)} cached blocks in one program, "
          f"{world}-way sequence parallel", flush=True)


def install_cached_aot(transformer, verbose=print, group=1):
  """AOT-compile the cached (decode) steps of every block, once, reused for all 32.

  Eager torch_tpu pays a program launch per operator: on eight chips that is
  ~0.34 s of a ~0.35 s step. `AotBlocks` cannot take the stock block, whose
  `layer_cache` argument is a Python object holding the prefix keys/values (a
  different object per layer, so the compile key never repeats). Here the block
  is re-expressed for the cached path with the prefix keys/values as tensor
  arguments; the prefill step (and anything with a mask) keeps the stock eager
  forward.
  """
  import torch
  import diffusers.models.transformers.transformer_qwenimage21 as q21mod
  from .aot_dit import AotBlocks

  from .qwen21_core_tt import CachedCore as _CachedCore, CachedGroup

  blocks = list(transformer.transformer_blocks)
  if len(blocks) % group:
    raise ValueError(f"{len(blocks)} blocks do not split into groups of {group}")
  if group == 1:
    units = [_CachedCore(b) for b in blocks]
  else:
    units = [CachedGroup(blocks[i:i + group]) for i in range(0, len(blocks), group)]
  AotBlocks(units).install()
  for index, (block, unit) in enumerate(zip(blocks, [u for u in units for _ in range(group)])):
    original = block.forward
    first = index % group == 0
    group_blocks = blocks[index:index + group] if first else None

    def forward(hidden_states, modulation, rotary_emb=None, attention_mask=None,
                target_token_mask=None, layer_cache=None, kv_cache_mode=None,
                cache_write_slice=None, segments=None, key_valid=None,
                _unit=unit, _original=original, _first=first, _gb=group_blocks,
                _index=index):
      cached = (kv_cache_mode == "cached" and layer_cache is not None
                and attention_mask is None and key_valid is None)
      if cached:
        if not _first:
          return hidden_states          # already applied by its group's first block
        if group == 1:
          kvs = layer_cache.get()
        else:
          kvs = [t for i in range(group) for t in _CACHES[0].get_layer(_index + i).get()]
        return _unit(hidden_states, modulation, rotary_emb, target_token_mask, *kvs)
      return _original(hidden_states, modulation, rotary_emb=rotary_emb,
                       attention_mask=attention_mask,
                       target_token_mask=target_token_mask,
                       layer_cache=layer_cache, kv_cache_mode=kv_cache_mode,
                       cache_write_slice=cache_write_slice, segments=segments,
                       key_valid=key_valid)

    block.forward = forward
  if group > 1:
    # The group's first block needs the other blocks' prefix caches: stash the
    # whole cache object from the transformer's call.
    t_forward = transformer.forward

    def transformer_forward(*a, **kw):
      _CACHES[0] = kw.get("kv_cache")
      return t_forward(*a, **kw)

    transformer.forward = transformer_forward
  verbose(f"  [aot-cached] {len(blocks)} blocks, group {group}: cached steps "
          "AOT-compiled, prefill eager", flush=True)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-2.1")
  p.add_argument("--prompt", type=str, required=True)
  p.add_argument("--image", type=str, default=None,
                 help="Condition image; omit for text-to-image")
  p.add_argument("--height", type=int, default=1024)
  p.add_argument("--width", type=int, default=1024)
  p.add_argument("--num_inference_steps", type=int, default=20)
  p.add_argument("--true_cfg_scale", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--output", type=str, default="qwen_image_21_tt.png")
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--ours-attn", action="store_true",
                 help="route unmasked DiT attention through the repo's splash kernel")
  p.add_argument("--aot-dit", action="store_true",
                 help="compile one transformer block ahead of time, reuse it for all")
  p.add_argument("--text-on-host", action="store_true",
                 help="multi-chip: keep Qwen3-VL on the host CPU instead of sharding it on the chips")
  p.add_argument("--aot-cached", action="store_true",
                 help="AOT-compile the cached denoising steps per block (prefix K/V as tensors)")
  p.add_argument("--aot-prefill", action="store_true",
                 help="with --ulysses: also AOT-compile the prefill step (single text segment only)")
  p.add_argument("--ulysses", action="store_true",
                 help="multi-chip: cached steps sequence parallel with replicated weights (one program per step)")
  p.add_argument("--aot-vae", action="store_true",
                 help="AOT-compile the VAE decoder as one program (one image shape)")
  p.add_argument("--aot-group", type=int, default=1,
                 help="blocks per compiled graph with --aot-dit")
  p.add_argument("--no_auto_env", action="store_true",
                 help="Do not fill in the TORCH_TPU_* variables")
  return p.parse_args()


def main():
  args = parse_args()
  world = int(os.environ.get("WORLD_SIZE", 1))
  if not args.no_auto_env:
    if world > 1:
      from .torchtpu_h3 import _auto_env as _auto_env_multi
      _auto_env_multi()
    else:
      _auto_env()
  print(args)

  import torch
  import torch_tpu  # noqa: F401  registers the 'tpu' device
  from diffusers import QwenImage21Pipeline

  dev = torch.device("tpu")
  pipe = QwenImage21Pipeline.from_pretrained(args.model_id,
                                             torch_dtype=torch.bfloat16)
  for name in ("transformer", "vae"):
    n = sum(p.numel() for p in getattr(pipe, name).parameters())
    print(f"{name:12s} {n / 1e9:5.2f}B")

  # DiT and VAE to the chip; Qwen3-VL stays on the host, which is what keeps
  # this inside 31 GB. Unlike the torchax script there is no tracing, so the
  # text encoder needs no splitting -- it just runs.
  rank = 0
  full_blocks = None
  if world > 1 and args.ulysses:
    import copy
    full_blocks = [copy.deepcopy(b) for b in pipe.transformer.transformer_blocks]
  if world > 1:
    from .tp_h3 import init_parallel
    rank, world, all_reduce = init_parallel()
    n = shard_qwen21(pipe.transformer, rank, world, all_reduce)
    print(f"  [tp] rank {rank}/{world}: {n} blocks sharded", flush=True)
  pipe.transformer = pipe.transformer.to(dev)
  pipe.vae = pipe.vae.to(dev)
  # The pipeline takes its device from the transformer, so it would move the
  # text encoder's inputs to the chip while its weights are still on the host
  # ("embedding(): tensor is expected to be on tpu, got cpu"). Run the whole
  # encode on the host and move only the result.
  inner_encode = pipe.encode_prompt

  phase = {"text": 0.0, "decode": 0.0}

  text_on_tpu = world > 1 and not args.text_on_host
  if text_on_tpu:
    # Eight ranks each encoding on the host CPU was 3.5 s of a 5.1 s request.
    # Sharded eight ways (64/8 q heads, 8/8 kv heads) Qwen3-VL is 2.2 GB a chip.
    from .tp_h3 import shard_text_encoder
    shard_text_encoder(pipe.text_encoder, rank, world, all_reduce)
    pipe.text_encoder = pipe.text_encoder.to(dev)

  def encode_prompt(*a, **kw):
    if not text_on_tpu:
      kw["device"] = torch.device("cpu")
    _t = time.perf_counter()
    embeds, mask, image_pad_mask = inner_encode(*a, **kw)
    if text_on_tpu:
      torch.accelerator.synchronize()
    phase["text"] += time.perf_counter() - _t
    to_dev = lambda t: t.to(dev) if t is not None else None
    return to_dev(embeds), to_dev(mask), to_dev(image_pad_mask)

  pipe.encode_prompt = encode_prompt
  inner_decode = pipe.vae.decode

  def vae_decode(*a, **kw):
    _t = time.perf_counter()
    out = inner_decode(*a, **kw)
    torch.accelerator.synchronize()
    phase["decode"] += time.perf_counter() - _t
    return out

  pipe.vae.decode = vae_decode
  print("DiT + VAE on tpu, Qwen3-VL on host")

  if args.ours_attn:
    from diffusers_tpu.backends import ours_splash_tt
    import diffusers.models.transformers.transformer_qwenimage21 as q21mod
    from diffusers.models import attention_dispatch

    if ours_splash_tt.install():
      # The transformer module imported the function by name; point it at the
      # patched dispatcher too.
      q21mod.dispatch_attention_fn = attention_dispatch.dispatch_attention_fn
      print("  [ours] DiT attention -> splash kernel", flush=True)
  if args.aot_vae:
    from .aot_dit import AotBlocks
    AotBlocks([pipe.vae.decoder]).install()
    print("  [aot-vae] VAE decoder compiled as one program", flush=True)
  if full_blocks is not None:
    full_blocks = [b.to(dev) for b in full_blocks]
    from .tp_h3 import init_parallel as _ip
    install_ulysses(pipe.transformer, full_blocks, rank, world, _ip.group_name,
                    aot_prefill=args.aot_prefill)
  elif args.aot_cached:
    install_cached_aot(pipe.transformer, group=args.aot_group)
  if args.aot_dit:
    from diffusers_tpu.backends.aot_dit import AotBlocks

    AotBlocks(list(pipe.transformer.transformer_blocks),
              group=args.aot_group).install()
    print(f"  [aot] {len(pipe.transformer.transformer_blocks)} blocks, "
          f"group {args.aot_group}", flush=True)

  image = None
  if args.image:
    from PIL import Image

    image = Image.open(args.image).convert("RGB")

  for i in range(args.runs):
    t0 = time.perf_counter()
    out = pipe(
        prompt=args.prompt, image=image, height=args.height,
        width=args.width, num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    dt = time.perf_counter() - t0
    print(f"run {i}: {dt:.2f}s" + (" (includes compilation)" if i == 0 else "")
          + f"  [text {phase['text']:.2f}s, vae decode {phase['decode']:.2f}s]",
          flush=True)
    phase["text"] = phase["decode"] = 0.0

  if rank == 0:
    out.save(args.output)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()


# Not done, and what it would take:
#
# * **Tensor parallel.** torch_tpu models are usually sharded by rebuilding
#   them against column- and row-parallel linear layers rather than by
#   sharding the diffusers module, so eight-way TP here means a rewrite of
#   QwenImage21Transformer2DModel: 32 single-stream blocks, each
#   attn.{to_q,to_k,to_v} column-parallel, attn.to_out.0 row-parallel,
#   img_mlp.{proj,gate_layer} column-parallel and img_mlp.out row-parallel,
#   every linear bias-free. Simpler than WAN's double-stream MoE, but a
#   rewrite all the same, plus a weight mapping from the diffusers
#   checkpoint.
# * **Qwen3-VL on device.** It would not fit alongside an unsharded DiT, so
#   this waits on the TP work above.
# * **Layerwise host offload** would be the other way to fit more on one
#   chip: park the blocks in pinned host memory already in the device's tiled
#   layout and DMA one in at a time (26.3 GB/s against .to()'s 7.2 for
#   bfloat16, because the layout conversion is paid once at load).
