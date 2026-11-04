"""MiniMax-H3 on TPU via torch_tpu, with the blocks streamed from the host.

The ~62 GiB of weights do not fit one 31 GiB chip, so each block is kept in
pinned host memory already in the device's tiled layout (see
`layerwise_offload`), making every upload a raw DMA (~26 GB/s vs ~7 GB/s for
`.to()`). Fifty 1.2 GiB blocks per step is ~2.3 s of transfer, small against
a single-chip denoise step; `prefetch_size` overlaps the next block's upload.

By default the stock diffusers model runs eagerly, so every operator is its
own program; see `--aot-dit` for block-level AOT compilation.

    TORCH_TPU_TOPOLOGY=1,1,1 \\
    python -m diffusers_tpu.backends.torchtpu_h3 --workflow t2va \\
        --num-frames 362 --height 480 --width 832 --steps 8
"""

import argparse
import os
import socket
import time


def _auto_env():
  """Fills in what torch_tpu's PjRt client insists on, for one or more chips."""
  # Auto-detect topology for v7x-8 ("2,2,1,2" across 4 dual-core chips),
  # v6e-8 ("2,4,1" across 8 single-core chips), or 1-chip ("1,1,1,1" / "1,1,1").
  _world = int(os.environ.get("WORLD_SIZE", 1))
  if _world <= 1:
    # GKE TPU webhook injects 4-chip host bounds into the pod environment;
    # clear them and restrict to chip 0 for single-rank execution.
    os.environ.pop("TPU_CHIPS_PER_HOST_BOUNDS", None)
    os.environ.pop("TPU_HOST_BOUNDS", None)
    os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")
  _topo = os.environ.get("TORCH_TPU_TOPOLOGY") or os.environ.get("DT_TT_TOPOLOGY")
  try:
    from torch_tpu._internal.utils.hardware import get_tpu_topology
    _hw_topo = get_tpu_topology(_world)
  except Exception:
    import glob as _glob
    _vfio = _glob.glob("/dev/vfio/[0-9]*")
    _hw_topo = "2,2,1,2" if (_world == 8 and len(_vfio) == 4) else (
        "2,4,1" if _world == 8 else "1,1,1")
  if not _topo or (_topo == "2,4,1" and _hw_topo == "2,2,1,2"):
    _topo = _hw_topo
  os.environ["TORCH_TPU_TOPOLOGY"] = _topo
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = f"localhost:{port}"
  os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", str(time.time_ns()))
  # Premapped host buffer. The default (0) halves host-to-device bandwidth,
  # and offload streams all weights every step, so anything outside the
  # premapped region falls off the fast path. At 480p/121 on one chip:
  #
  #    16 GiB   DiT 5.43 GB/s  text 7.58   wall 182s
  #    64 GiB   DiT 26.11      text 6.15   wall 146s
  #   160 GiB   DiT 26.06      text 26.07  wall 109s
  #
  # Output is bitwise identical. Budget is half of host RAM split across
  # local ranks (the setting is per process), capped at 192 GiB where a
  # rank stops gaining.
  _ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  _local = max(1, int(os.environ.get("LOCAL_WORLD_SIZE",
                                     os.environ.get("WORLD_SIZE", 1))))
  # Clamp to [1, 192] GiB with no larger floor (a floor could overcommit a
  # small host), and round to a whole GiB: the runtime requires page alignment.
  _premap = min(192, max(1, (_ram // 2 // _local) >> 30)) << 30
  os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(_premap))
  # Disable AOTAutograd's disk cache: grouped-block graphs are large and the
  # cache can grow to tens of GB, while `AotBlocks` already caches by shape
  # within a process.
  os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      "--xla_tpu_dvfs_p_state=7 "
      "--slicebuilder_use_insecure_grpc=true "
      "--xla_tpu_enable_latency_hiding_scheduler=true "
      "--xla_tpu_spmd_rng_bit_generator_unsafe=true "
      "--xla_tpu_enable_dot_strength_reduction=true",
  )


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
  p.add_argument("--workflow", choices=("t2va", "ref2va"), default="t2va")
  p.add_argument("--reference", default=None, help="ref2va's condition image")
  p.add_argument("--prompt", default="a red fox trotting through snow")
  p.add_argument("--num-frames", type=int, default=362)
  p.add_argument("--max-duration", type=int, default=16)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--width", type=int, default=832)
  p.add_argument("--steps", type=int, default=8)
  p.add_argument("--pika-adapter", default=None,
                 help="Pika-H3-Flex adapter dir (fl2va/ or ref2va/): merge the "
                      "LoRA and sample its six-step re-noising ladder")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out", default="out")
  p.add_argument("--tag", default="tt_h3")
  p.add_argument("--prefetch", type=int, default=1,
                 help="Blocks kept in flight ahead of the one running.")
  p.add_argument("--no-offload", action="store_true",
                 help="Keep every block resident (no host streaming). Needs "
                      "enough HBM for all blocks.")
  p.add_argument("--vae-clip", type=int, default=0,
                 help="Decode this many latent frames at a time, each "
                      "handed to the host; one chip cannot hold a whole "
                      "clip's decode. 0 is off.")
  p.add_argument("--vae-clip-overlap", type=int, default=0,
                 help="Latent frames of temporal lead-in decoded before "
                      "each chunk and then discarded. Measured not to "
                      "help (see _chunked_decode). Default 0.")
  p.add_argument("--host-decode", action="store_true",
                 help="Keep the VAE's per-chunk decode results on the "
                      "host instead of the chip. Same arithmetic as "
                      "_decode, bounded peak.")
  p.add_argument("--decode-to-host", action="store_true",
                 help="Move the VAE's output to the host as it leaves the "
                      "decoder; the frame buffer downstream of it does not "
                      "fit one chip.")
  p.add_argument("--vae-tiling", action="store_true",
                 help="Tile the VAE decode; one chip cannot hold "
                      "a 362-frame float32 frame buffer.")
  p.add_argument("--compile-dit", action="store_true",
                 help="Run the DiT through torch.compile rather "
                      "than eager.")
  p.add_argument("--aot-release-between-runs", action="store_true",
                 help="Drop the compiled artifact between requests, to "
                      "avoid RuntimeProgramAllocationFailure on later "
                      "requests at 768p/345. Recompiling a block costs ~7 s.")
  p.add_argument("--ours-attn", action="store_true",
                 help="Route attention at this repo's own Pallas kernel "
                      "(kernels/splash.py) through torch_tpu's "
                      "`torch.tpu.pallas.jax_op`. Unlike torch_tpu's "
                      "own splash it does not require the block to "
                      "divide the sequence, which H3's 14,856 and "
                      "103,976 do not. Generic SDPA attention is ~47%% of "
                      "the denoise at 768p.")
  p.add_argument("--rank0-text", action="store_true",
                 help="Run Qwen3-VL on rank 0 only and broadcast the "
                      "embedding, instead of every rank streaming 58.1 GiB "
                      "of conditioner weights to compute the same result.")
  p.add_argument("--free-text-after", action="store_true",
                 help="Move Qwen3-VL back to the host as soon as the "
                      "prompt is encoded. Otherwise its non-layer modules "
                      "(vision tower) stay on the chip for the whole "
                      "request, which can make ref2va OOM when loading "
                      "the decode program.")
  p.add_argument("--dump-latents", default=None,
                 help="Save the latents handed to the VAE decode "
                      "to this path (rank 0 only), then decode as "
                      "usual. Lets decode experiments skip the "
                      "weight load and the denoise steps.")
  p.add_argument("--sparse-window", type=int, default=0,
                 help="Lossy. After --sparse-dense-steps DiT calls, each "
                      "video row attends only to video rows within this many "
                      "latent frames (text/audio rows stay dense). Ported "
                      "from torchax (sparse_attn_tt.py). 0 = dense.")
  p.add_argument("--sparse-dense-steps", type=int, default=6)
  p.add_argument("--cache-threshold", type=float, default=0.0,
                 help="Lossy. Skip a denoise step and reuse the previous "
                      "DiT output when the timestep embedding has drifted "
                      "less than this (accumulated) since the last computed "
                      "step. Ported from torchax's --cache-threshold "
                      "(torchax_pipeline.py); there 0.10 skips 12 of 19 "
                      "calls at 768p/345. 0 disables.")
  p.add_argument("--dit-park-for-decode", action="store_true",
                 help="With --no-offload: move the DiT blocks to the host "
                      "for the VAE decode and back afterwards. A resident "
                      "DiT speeds up the t2va denoise (169 -> 34 s) but "
                      "leaves too little HBM for the decode program.")
  p.add_argument("--vae-on-demand", action="store_true",
                 help="Keep the VAE on the host and move it to the "
                      "chip only for the decode. Otherwise it holds "
                      "9.70 GiB of HBM idle through the denoise, which "
                      "can make later ref2va requests OOM loading the "
                      "decode program.")
  p.add_argument("--hbm-trace", action="store_true",
                 help="Print what each module holds on the chip at "
                      "every stage boundary, plus the compilation "
                      "cache's own HBM.")
  p.add_argument("--aot-group", type=int, default=1,
                 help="Compile N consecutive blocks into one graph. "
                      "With one block per program, ~16%% of the denoise "
                      "is launch overhead (~25 ms per launch, 50 blocks "
                      "x steps) and each block's all-reduce is exposed.")
  p.add_argument("--seg-mod", action="store_true",
                 help="Modulate by run instead of per row. The "
                      "stock block does six "
                      "index_select(0, adaln_indices) a layer, each "
                      "materialising a full-sequence copy; the "
                      "the last run (video rows, ~98%% of the sequence) "
                      "broadcasts its row and only the short head is "
                      "rewritten. Bit-exact on CPU "
                      "(scripts/probes_aot/segmod_equiv.py).")
  p.add_argument("--shard-text", action="store_true",
                 help="Tensor-parallel Qwen3-VL's decoder layers. "
                      "Each rank otherwise streams 58.1 GiB of "
                      "conditioner weights at ~2 GB/s; sharded, it is "
                      "7.3 GiB a rank at 8 ranks.")
  p.add_argument("--aot-vae", action="store_true",
                 help="AOT-compile one decoder transformer layer and "
                      "reuse it for all 36, the way --aot-dit does "
                      "for the DiT. Compiling the whole stack is too "
                      "slow; one layer takes 4.5 s. "
                      "Decode 18.5 -> 14.4 s sharded, but max diff "
                      "7.60e-02 -- compilation reorders reductions, "
                      "so this is NOT bit-identical. Off by default.")
  p.add_argument("--shard-decode", action="store_true",
                 help="Decode the clip's chunks across the ranks "
                      "instead of all of them on one. The chunks are "
                      "independent (each carries its own overlap; the "
                      "blend happens on the host), so no tensor "
                      "sharding or device collective is needed. "
                      "Measured 90.8 -> 23.0 s, bit-identical. "
                      "Replaces --host-decode.")
  p.add_argument("--clear-before-decode", action="store_true",
                 help="Drop TpuBackend's cached executables before "
                      "the VAE decode, freeing HBM for loading the "
                      "decode program. Costs one block recompile, ~7 s.")
  p.add_argument("--phases", action="store_true",
                 help="Wall-clock the conditioner, the VAE decode "
                      "and the DiT forward, and print what is left "
                      "over (host-side time the device profile cannot see).")
  p.add_argument("--text-layers", type=int, default=0,
                 help="Run only the first N of Qwen3-VL's 64 decoder "
                      "layers. MiniMax-H3 reads hidden_states[50] and "
                      "never looks further, so later layers are wasted "
                      "work. 0 keeps all 64. Use 51, not 50: with "
                      "exactly 50, hidden_states[50] is the tuple's last "
                      "entry, which transformers returns post-norm "
                      "(DINO 0.9853 -> 0.9456).")
  p.add_argument("--cpu-text", action="store_true",
                 help="Run Qwen3-VL on the host and move only the "
                      "embedding to the chip. Streaming its 58.1 GiB "
                      "at ~2 GB/s costs ~40 s with the chip idle, while "
                      "the host handles the few prompt tokens in "
                      "seconds. Implies --rank0-text.")
  p.add_argument("--ours-no-fixed-m", action="store_true",
                 help="Turn OFF the hybrid fixed-m softmax and the "
                      "k-centering that goes with it. On by default, "
                      "matching torchax; turning it off makes the "
                      "kernel substantially slower.")
  p.add_argument("--splash-attn", action="store_true",
                 help="Route attention at torch_tpu's `splash_sdpa` "
                      "instead of the generic SDPA lowering (~47%% of "
                      "the denoise at 768p).")
  p.add_argument("--profile", default="",
                 help="Take an xprof trace of the last run into this "
                      "directory.")
  p.add_argument("--resident-blocks", type=int, default=0,
                 help="Keep this many of the fifty blocks on the chip "
                      "for the whole request instead of streaming "
                      "them. Full residency (`--no-offload`) fits at "
                      "480p/121 and does not at 768p/345, where it is "
                      "about 0.8 GiB short -- this dials the trade.")
  p.add_argument("--shard-adaln", action="store_true",
                 help="Also tensor-parallel `adaln_proj.linear`. It is "
                      "260 M parameters a block, 13.0 B over fifty; "
                      "without it eight-way TP leaves 16.3 B a rank "
                      "rather than 4.1 B. The output is gathered back, "
                      "so the arithmetic is unchanged.")
  p.add_argument("--aot-dit", action="store_true",
                 help="AOT-compile one transformer block and run all "
                      "fifty through that artifact, instead of "
                      "operator by operator. Works with the layer-wise "
                      "host streaming: the block is an argument, so "
                      "its weights are graph inputs.")
  p.add_argument("--aot-dynamic", action="store_true",
                 help="AOT-compile with dynamic shapes. Off by default: "
                      "the request's shapes are fixed.")
  p.add_argument("--frame-bucket", type=int, default=0,
                 help="Round every request up to a multiple of "
                      "this many frames. Bounds the number of "
                      "compiled programs without asking the "
                      "caller to change anything.")
  p.add_argument("--frame-sweep", default="",
                 help="Comma-separated frame counts to run in "
                      "one process, reporting host RSS and the "
                      "compile-cache counters around each.")
  p.add_argument("--evict-between", action="store_true",
                 help="Call torch.tpu._clear_cache() (which binds to "
                      "CompilationCache::EvictAll) plus gc and "
                      "malloc_trim after every shape, to measure how "
                      "much host memory eviction actually returns.")
  p.add_argument("--no-auto-env", action="store_true")
  p.add_argument("--qk-fp8", action="store_true",
                 help="Cast Q and K to float8_e4m3fn inside Pallas splash attention (DT_OURS_QK_FP8=1)")
  p.add_argument("--runs", type=int, default=1)
  return p.parse_args()


def main():
  args = parse_args()
  if args.qk_fp8:
    os.environ["DT_OURS_QK_FP8"] = "1"
  if not args.no_auto_env:
    _auto_env()
  print(args, flush=True)
  # `kill -USR1 <pid>` dumps every thread's Python stack for this rank, for
  # diagnosing stalls (py-spy cannot attach to Python 3.14 processes).
  import faulthandler
  import signal
  # One file per rank; ranks writing to a shared stderr interleave unreadably.
  _stk_dir = os.path.join(args.out, "stacks")
  os.makedirs(_stk_dir, exist_ok=True)
  _stk = open(os.path.join(
      _stk_dir, f"{args.tag}_rank{os.environ.get('RANK', 0)}.txt"), "a")
  faulthandler.register(signal.SIGUSR1, file=_stk, all_threads=True)

  import torch
  import torch_tpu  # noqa: F401  registers the 'tpu' device

  # torch_tpu defaults to `float32_matmul_precision = 'medium'`.
  # `DT_MATMUL_PRECISION` overrides it, to study per-step numerical drift
  # against torchax. Unchanged by default.
  _prec = os.environ.get("DT_MATMUL_PRECISION")
  if _prec:
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(_prec)
    print(f"  [precision] float32_matmul_precision {before} -> "
          f"{torch.get_float32_matmul_precision()}", flush=True)

  from diffusers_tpu.backends.layerwise_offload import LayerwiseOffloadManager

  dev = torch.device("tpu")
  begin = time.perf_counter()
  pipe = _load_pipeline(args)
  print(f"host load {time.perf_counter() - begin:.1f}s", flush=True)

  from diffusers_tpu.backends.tp_h3 import init_parallel, shard_transformer

  rank, world, all_reduce = init_parallel()
  transformer = getattr(pipe, pipe._dit_name)
  if args.pika_adapter:
    # Merged before any slicing: the adapter's matrices are full-width.
    from diffusers_tpu.models.minimax_h3.pika import merge_pika_lora, pika_scheduler
    pika = merge_pika_lora(transformer, args.pika_adapter)
    pipe.update_components(
        scheduler=pika_scheduler(pipe.scheduler, pika, "video_shift", args.seed),
        audio_scheduler=pika_scheduler(pipe.audio_scheduler, pika, "audio_shift",
                                       args.seed))
  if world > 1:
    # `--shard-adaln`: also shard the AdaLN projection (~43% of a block).
    # The gather restores the tensor exactly. Off by default.
    sliced = shard_transformer(
        transformer, rank, world, all_reduce,
        all_gather=getattr(init_parallel, "all_gather", None)
        if args.shard_adaln else None)
    print(f"tensor parallel: rank {rank} of {world}, {sliced} blocks sliced",
          flush=True)
    dev = torch.device("tpu", rank)
  blocks = transformer.transformer_blocks
  total = sum(p.numel() for p in transformer.parameters())
  resident = sum(
      p.numel() for b in blocks for p in b.parameters()
  )
  print(f"transformer {total / 1e9:.1f}B, of which the fifty blocks are "
        f"{resident / 1e9:.1f}B ({resident * 2 / 2**30:.0f} GiB bf16)",
        flush=True)

  if args.seg_mod:
    # Must run before AOT: it replaces the block class's forward, which
    # `aot_dit._flat_forward` calls via `type(module).forward`.
    from diffusers_tpu.backends import segmented_mod_tt

    segmented_mod_tt.install(transformer)

  _aot = None
  if args.ours_attn:
    from diffusers_tpu.backends import ours_splash_tt

    ours_splash_tt.install(fixed_m=not args.ours_no_fixed_m)

  if args.splash_attn:
    from diffusers_tpu.backends import splash_attn_tt

    splash_attn_tt.install()

  if args.aot_dit:
    if args.compile_dit:
      raise SystemExit("--aot-dit and --compile-dit are mutually exclusive")
    # Works with offload: the block is an argument of the compiled
    # artifact, so its weights are graph inputs and can be streamed from
    # pinned host memory (bit-exact).
    from diffusers_tpu.backends.aot_dit import AotBlocks

    _aot = AotBlocks(blocks, dynamic=args.aot_dynamic,
                     group=args.aot_group).install()
    print(f"DiT: {len(blocks)} blocks via aot_compile, "
          f"compiled once and reused (dynamic={args.aot_dynamic})", flush=True)

  if args.compile_dit:
    # Note: torch_tpu's `mark_embedded_constants` pass turns a compiled
    # graph's `get_attr` weights into MLIR constants (a weight copy per
    # graph), unlike eager mode where weights stay device buffers.
    # `backend="tpu"` is required: the default (inductor) builds an
    # executable that fails at run time, and under multi-rank TP that
    # looks like a collective deadlock.
    transformer.forward = torch.compile(transformer.forward,
                                        backend="tpu", dynamic=False)
    print("DiT via torch.compile(backend='tpu', dynamic=False)", flush=True)

  manager = None
  if not args.no_offload:
    # Only the blocks are host-resident; the rest of the transformer
    # (projections, time embedder, refiner, `norm_out`) is small and stays
    # on the chip.
    manager = LayerwiseOffloadManager(
        transformer,
        layers_attr_str="transformer_blocks",
        num_layers=len(blocks),
        enabled=True,
        prefetch_size=args.prefetch,
    )
    manager.pinned = max(0, min(args.resident_blocks, len(blocks)))
    manager.offload_all_layers(stage_from_host=True)
    if args.aot_group > 1 and _aot is not None:
      # Non-leading blocks in a group never run `_call_impl`, so hooks do
      # not fire; AotBlocks stages weights at group boundaries instead.
      _aot.set_manager(manager)
      print(f"  [aot] {args.aot_group} blocks per graph, weights staged per group by AOT",
            flush=True)
    else:
      manager.register_forward_hooks()
    print(f"streaming {len(blocks) - manager.pinned} blocks from pinned "
          f"host memory, {manager.pinned} resident, "
          f"prefetch {args.prefetch}", flush=True)
  else:
    # `_move_non_block_modules` moves everything except the blocks, so
    # the blocks must be moved explicitly here.
    for blk in blocks:
      blk.to(dev)
    print(f"{len(blocks)} blocks resident on chip (no offload)",
          flush=True)

  # Everything outside the streamed blocks lives on the chip.
  _move_non_block_modules(transformer, dev)

  # Report tensors left on the host by name, rather than failing later
  # with "Expected all tensors to be on the same device".
  _strays = [n for n, t in list(transformer.named_parameters())
             + list(transformer.named_buffers()) if t.device.type != "tpu"]
  if _strays:
    print(f"  [device] {len(_strays)} tensors still on the host, first few: "
          f"{_strays[:6]}", flush=True)
  else:
    print("  [device] transformer fully on chip", flush=True)
  _inputs_to_device(transformer, dev)
  # The VAEs go to the chip. The large decoded frame buffer
  # (`float32[frames, 3, h, w]`, 1.7 GB at 480p/362, 4.3 GB at 768p/345)
  # is handled by the decode options below.
  for name in ("vae", "audio_vae"):
    part = getattr(pipe, name, None)
    if part is not None:
      setattr(pipe, name, part.to(dev))
  if args.vae_clip:
    # The decode materialises the whole clip, so moving only its output
    # to the host is too late; decode in chunks and convert at the boundary.
    _chunked_decode(pipe, args.vae_clip, args.vae_clip_overlap)
    print(f"VAE decode in chunks of {args.vae_clip} latent frames, "
          f"{args.vae_clip_overlap} frames of lead-in", flush=True)
  if args.aot_vae:
    # Compile one decoder layer and reuse it for all 36, as for the DiT
    # (compiling the whole stack is too slow; one layer takes 4.5 s).
    # 18.5 -> 14.4 s on 8 ranks, but not bit-identical (max diff 7.6e-2,
    # reordered reductions), so off by default.
    from diffusers_tpu.backends.aot_dit import AotBlocks

    _vae_blocks = getattr(getattr(pipe.vae, "decoder", None),
                          "transformer_blocks", None)
    if _vae_blocks:
      AotBlocks(list(_vae_blocks)).install()
      print(f"  [aot-vae] {len(_vae_blocks)} decoder layers via AotBlocks",
            flush=True)
  if os.environ.get("DT_DECODE_TILE_BATCH"):
    _batch_decode_tiles(pipe.vae, int(os.environ["DT_DECODE_TILE_BATCH"]))
  if args.shard_decode:
    # Replaces --host-decode: same loop, chunks split across ranks and
    # gathered on the host. 90.8 s on one chip -> 23.0 s on 8 ranks,
    # bit-identical (`scripts/bench_decode.py`).
    _shard_decode(pipe.vae)
  elif args.host_decode:
    _host_decode(pipe.vae)
    print("VAE decode keeps chunks on the host", flush=True)
  if args.decode_to_host:
    # Only `decode`'s output is a problem: the processor widens it to
    # `float32[frames, 3, h, w]`. Moving it straight to the host keeps the
    # decode on the chip and the big buffer off it.
    _decode_to_host(pipe.vae)
    print("VAE decode returns to the host", flush=True)
  # 362 frames decode as `float32[362, 3, 480, 832]` (1.7 GB a tensor,
  # several live at once), too much for one chip; tiling bounds it.
  if args.vae_tiling:
    for name in ("vae", "audio_vae"):
      part = getattr(pipe, name, None)
      if part is not None and hasattr(part, "enable_tiling"):
        part.enable_tiling()
    print("VAE tiling on", flush=True)
  # Qwen3-VL (17.5 GB) does not fit beside the DiT's staging buffers, so
  # by default its layer stack streams by the same mechanism as the DiT.
  if (args.rank0_text and world > 1) or args.cpu_text:
    _rank0_text_encode(rank, all_reduce, on_cpu=args.cpu_text,
                       keep=args.text_layers, world=world)
  if args.cpu_text:
    # `_execution_device` is inferred from component parameters; with the
    # text encoder on the host it would resolve to cpu and latents would be
    # created there. Pin it to the chip.
    type(pipe)._execution_device = property(lambda _self, _d=dev: _d)
    print(f"  [cpu-text] execution device pinned to {dev}", flush=True)
  if args.free_text_after:
    _free_text_after_encode(pipe)
  if args.dump_latents and int(os.environ.get('RANK', 0)) == 0:
    _dump_latents(pipe, args.dump_latents)
  if args.vae_on_demand:
    _vae_on_demand(pipe)
  if args.dit_park_for_decode:
    _dit_park_for_decode(pipe)
  _step_cache_reset = (_install_step_cache(pipe, args.cache_threshold)
                       if args.cache_threshold > 0 else None)
  _sparse_reset = None
  if args.sparse_window > 0:
    from diffusers_tpu.backends import ours_splash_tt as _ot
    from diffusers_tpu.backends import sparse_attn_tt
    # Install after the step cache so skipped calls are counted too.
    _sparse_reset = sparse_attn_tt.install(
        pipe, args.sparse_window, args.sparse_dense_steps,
        _ot.BQ, _ot.BKV, fixed_m=not args.ours_no_fixed_m)
  if args.hbm_trace:
    _vae = getattr(pipe, 'vae', None)
    if _vae is not None:
      _inner_dec = _vae.decode
      def _traced_decode(*a, _f=_inner_dec, _p=pipe, **k):
        _hbm_snapshot('before decode', _p)
        try:
          return _f(*a, **k)
        finally:
          _hbm_snapshot('after decode', _p)
      _vae.decode = _traced_decode
  if args.clear_before_decode:
    _clear_before_decode(pipe)
  if args.phases:
    _install_phase_clocks(pipe)
  text_manager = _stream_text_encoder(pipe, args)
  if args.hbm_trace:
    _hbm_snapshot('loaded', pipe)
  # The schedulers hold `sigmas` and `timesteps` as CPU tensors built at
  # `set_timesteps`; move them to the chip to meet the latents.
  for name in ("scheduler", "audio_scheduler"):
    _tensor_attrs_to_device(getattr(pipe, name, None), dev)

  reference = None
  if args.workflow == "ref2va":
    _fix_reference_encode()
    if not args.reference:
      raise SystemExit("ref2va needs --reference")
    from PIL import Image

    reference = Image.open(args.reference).convert("RGB")

  os.makedirs(args.out, exist_ok=True)

  def host_gb():
    for line in open("/proc/self/status"):
      if line.startswith("VmRSS"):
        return int(line.split()[1]) / 1048576
    return -1.0

  # RSS alone does not say who holds memory. TPU VM images preload
  # gperftools tcmalloc (`LD_PRELOAD` in /etc/environment), so glibc's
  # `malloc_trim` and `mallinfo2` are no-ops here; use tcmalloc's counters:
  #
  #   live    current_allocated_bytes   still referenced; only an
  #                                     eviction API helps
  #   free    pageheap_free_bytes       freed, held by tcmalloc, not
  #                                     returned to the kernel yet --
  #                                     ReleaseFreeMemory() gets it
  #   heap    generic.heap_size         live + free + fragmentation
  #   other   RSS - heap                outside the allocator (mmap by
  #                                     MLIR/LLVM, the loaded .so's,
  #                                     the pinned host weights)
  import ctypes

  _tc = None
  try:
    _maps = open("/proc/self/maps").read()
  except OSError:
    _maps = ""
  if "/libtcmalloc.so.4" in _maps:
    for cand in ("libtcmalloc.so.4", "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"):
      try:
        probe = ctypes.CDLL(cand)
        probe.MallocExtension_GetNumericProperty
        _tc = probe
        break
      except (OSError, AttributeError):
        continue

  def tc(prop):
    if _tc is None:
      return 0.0
    out = ctypes.c_size_t(0)
    ok = _tc.MallocExtension_GetNumericProperty(
        prop.encode(), ctypes.byref(out))
    return out.value / 2**30 if ok else 0.0

  def report(tag):
    try:
      miss = torch.tpu._get_cache_misses()
      reqs = torch.tpu._get_cache_requests()
    except Exception:
      miss = reqs = -1
    rss = host_gb()
    heap = tc("generic.heap_size")
    live = tc("generic.current_allocated_bytes")
    free = tc("tcmalloc.pageheap_free_bytes")
    print(f"  [host] {tag:18s} RSS {rss:7.2f}  heap {heap:7.2f}  "
          f"live {live:7.2f}  free {free:7.2f}  other {rss - heap:7.2f} GB  "
          f"reqs {reqs} miss {miss}", flush=True)

  def evict():
    """Each release path separately, so the credit is attributable."""
    import gc
    steps = [
        ("_clear_cache", lambda: torch.tpu._clear_cache()),
        ("gc.collect", gc.collect),
        ("tcmalloc.Release",
         (lambda: _tc.MallocExtension_ReleaseFreeMemory())
         if _tc is not None else (lambda: None)),
    ]
    for name, fn in steps:
      before = host_gb()
      try:
        fn()
      except Exception as exc:
        print(f"  [evict] {name} failed: {exc}", flush=True)
        continue
      print(f"  [evict] {name:18s} returned {before - host_gb():6.2f} GB "
            f"(live {tc('generic.current_allocated_bytes'):.2f} "
            f"free {tc('tcmalloc.pageheap_free_bytes'):.2f})", flush=True)

  # `--frame-sweep` runs several frame counts in one process and prints
  # host RSS and compile-cache counters around each, to measure the host
  # memory cost of each new-shape compile.
  if args.frame_sweep:
    report("start")
    for frames in [int(x) for x in args.frame_sweep.split(",")]:
      # Round frame counts up to a bucket so different requests share a
      # compiled program; the tail can be cut off the result.
      if args.frame_bucket:
        bucketed = -(-frames // args.frame_bucket) * args.frame_bucket
        print(f"  [bucket] requested {frames} frames -> generating {bucketed}",
              flush=True)
        frames = bucketed
      args.num_frames = frames
      args.max_duration = max(5.0, frames / 24.0 + 0.5)
      began = time.perf_counter()
      try:
        # Default device must be tpu so the scheduler builds tensors on chip.
        with torch.device("tpu"):
          _generate(pipe, args, reference)
        note = f"{time.perf_counter() - began:.0f}s"
      except Exception as exc:
        note = f"FAILED {type(exc).__name__}"
      report(f"{frames}f {note}")
      if args.evict_between:
        evict()
        report(f"{frames}f evicted")
    return

  prof_ctx = None
  if args.profile:
    from torch_tpu._internal import profiler as tt_prof

    prof_ctx = tt_prof.profile(
        activities=[tt_prof.ProfilerActivity.TPU],
        on_trace_ready=tt_prof.xprof_trace_handler(dir_name=args.profile),
    )

  for run in range(args.runs):
    if manager is not None:
      manager.prepare_for_next_req()
    if run and args.aot_release_between_runs and _aot is not None:
      print(f"  [aot] released previous run's artifacts: {_aot.release()}", flush=True)
    if prof_ctx is not None and run == args.runs - 1:
      prof_ctx.__enter__()
    if args.hbm_trace:
      _hbm_snapshot(f'run {run} start', pipe)
    begin = time.perf_counter()
    # Tensors the pipeline builds here (timesteps, noise, index vectors)
    # land on the chip rather than the host.
    with torch.device("tpu"):
      if _step_cache_reset is not None:
        _step_cache_reset()
      if _sparse_reset is not None:
        _sparse_reset()
      result = _generate(pipe, args, reference)
    took = time.perf_counter() - begin
    if args.hbm_trace:
      _hbm_snapshot(f'run {run} end', pipe)
    print(f"run {run}: {took:.1f}s", flush=True)
    # Per-rank start/end times, to identify straggler ranks.
    print(f"  [run-rank] run {run} rank {os.environ.get('RANK', 0)} "
          f"start {begin % 10000:.1f} end {(begin + took) % 10000:.1f} "
          f"took {took:.1f}s", flush=True)
    if _STAGE:
      named = sum(_STAGE.values())
      bits = "  ".join(f"{k} {v:.1f}s" for k, v in _STAGE.items())
      # Remainder = request time minus the timed phases (mostly denoise).
      print(f"  [phases] {bits}  denoise/other {took - named:.1f}s", flush=True)
      _STAGE.clear()
    if prof_ctx is not None and run == args.runs - 1:
      prof_ctx.__exit__(None, None, None)
      print(f"  [profile] written to {args.profile}", flush=True)
  _write(result, args)


def _load_pipeline(args):
  """Loads H3 on the host. Nothing reaches the chip here.

  H3 is a modular pipeline, not a monolithic one, and the components
  have to be named: a bare `load_components()` leaves the processor
  unset and the text encoder step needs it. `ref2va` denoises against
  `transformer_ref`, a different partition, and naming the right one
  is also what stops `from_pretrained` pulling both.
  """
  import glob
  import torch
  from diffusers import ModularPipeline

  model_path = args.model_id
  if not os.path.isdir(model_path):
    # Local cache only: eight ranks asking the hub at once get rate limited.
    from ..hub import local_snapshot
    model_path = local_snapshot(model_path)
  _local = True
  pipe = ModularPipeline.from_pretrained(model_path,
                                         workflow=args.workflow,
                                         local_files_only=_local)
  if args.max_duration:
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MiniMaxH3ModularPipeline,
    )

    cap = args.max_duration
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: cap)
    type(pipe).max_duration = property(lambda self: cap)
  which = "transformer_ref" if args.workflow == "ref2va" else "transformer"
  if os.path.isdir(model_path):
    for _v in vars(pipe).values():
      _items = _v.values() if isinstance(_v, dict) else (_v if isinstance(_v, (list, tuple)) else ())
      for _spec in _items:
        if hasattr(_spec, "pretrained_model_name_or_path"):
          _spec.pretrained_model_name_or_path = model_path
  pipe.load_components(
      names=["text_encoder", "tokenizer", "processor", which, "vae",
             "audio_vae", "scheduler", "audio_scheduler"],
      torch_dtype=torch.bfloat16,
      local_files_only=_local,
  )
  pipe._dit_name = which
  return pipe


def _move_non_block_modules(transformer, dev, blocks_attr="transformer_blocks"):
  """Moves the transformer to the chip except the streamed blocks.

  `transformer.to(dev)` would move the blocks too. Filtering by qualified
  name (rather than tensor identity) also catches non-persistent buffers
  such as `rope.inv_freq`.
  """
  moved = held = 0
  for name, module in transformer.named_modules():
    if name == blocks_attr or name.startswith(blocks_attr + "."):
      held += sum(1 for _ in module.named_parameters(recurse=False))
      held += sum(1 for _ in module.named_buffers(recurse=False))
      continue
    for attr, param in list(module.named_parameters(recurse=False)):
      module._parameters[attr] = param.to(dev)
      moved += 1
    for attr, buf in list(module.named_buffers(recurse=False)):
      if buf is not None:
        module._buffers[attr] = buf.to(dev)
        moved += 1
  print(f"resident on chip: {moved} tensors; streamed: {held}", flush=True)


def _inputs_to_device(transformer, dev):
  """Moves the transformer's own arguments to the chip.

  The modular pipeline builds layout tensors (`position_ids`, index
  vectors, tags) on the host; move them to the chip on each call.
  """
  import functools

  import torch

  inner = transformer.forward

  # `functools.wraps` is required: the denoise step filters its arguments
  # by `inspect.signature(transformer.forward).parameters`, which follows
  # `__wrapped__`; a bare `*a, **kw` wrapper would receive none of them.
  @functools.wraps(inner)
  def forward(*a, **kw):
    move = lambda t: (t.to(dev) if torch.is_tensor(t)
                      and t.device.type != "tpu" else t)
    return inner(*[move(x) for x in a], **{k: move(v) for k, v in kw.items()})

  transformer.forward = forward


def _truncate_layers(encoder, keep: int):
  """Keep only the first `keep` decoder layers; returns a restore closure.

  H3 reads `outputs.hidden_states[50]`, so later layers are wasted compute
  and weight transfer. The config's layer count is left unchanged. Keep at
  least 51: with exactly 50, `hidden_states[50]` is the last entry, which
  transformers returns after the final norm.
  """
  import torch

  for attr in ("model.language_model.layers", "language_model.layers",
               "model.layers", "layers"):
    node = encoder
    try:
      for part in attr.split("."):
        node = getattr(node, part)
    except AttributeError:
      continue
    if len(node) <= keep:
      return None
    parent, last = encoder, attr.split(".")
    for part in last[:-1]:
      parent = getattr(parent, part)
    full = node
    setattr(parent, last[-1], torch.nn.ModuleList(list(node)[:keep]))
    return lambda: setattr(parent, last[-1], full)
  return None


_STAGE: dict[str, float] = {}


def _clock(label, owner, attr):
  """Wrap a method with a wall-clock timer accumulated into `_STAGE[label]`.

  Wall clock rather than a device profile: the chip is idle before the
  denoise, so xprof cannot show where that host-side time goes.
  """
  inner = getattr(owner, attr)

  def timed(*a, **k):
    begin = time.perf_counter()
    try:
      return inner(*a, **k)
    finally:
      _STAGE[label] = _STAGE.get(label, 0.0) + time.perf_counter() - begin

  setattr(owner, attr, timed)


def _install_phase_clocks(pipe, verbose=print):
  """Wall-clock the text encoder and VAE decode; the rest counts as denoise."""
  enc = getattr(pipe, "text_encoder", None)
  if enc is not None:
    # Wrap `enc.model.forward`, not `enc.forward`:
    # `get_qwen3vl_prompt_embeds` calls `text_encoder.model(...)`.
    inner_model = getattr(enc, "model", None)
    _clock("text_encoder", inner_model if inner_model is not None else enc,
           "forward")
  vae = getattr(pipe, "vae", None)
  if vae is not None:
    _clock("vae_decode", vae, "decode")
  # The DiT forward is not wrapped: some callers invoke it as an unbound
  # function, bypassing instance attributes. Denoise time is the remainder.
  verbose("  [phases] wall clocks installed: text_encoder / vae_decode (rest counted as denoise)",
          flush=True)



def _free_text_after_encode(pipe, verbose=print):
  """Move the text encoder back to the host right after encoding.

  Its non-layer modules (including the vision tower) otherwise stay on the
  chip for the whole request although ref2va only uses them once, which can
  leave too little HBM to load the decode program. Tensors are moved back
  before the next encode.
  """
  import torch

  enc = getattr(pipe, "text_encoder", None)
  if enc is None:
    return
  from diffusers.modular_pipelines.minimax_h3 import encoders

  inner = encoders.get_qwen3vl_prompt_embeds
  # Slots moved to the host, so they can be restored before the next encode.
  parked: list[tuple] = []
  #: Deduplicate by storage: a weight shared by two modules is reported
  #: once per holder and must be moved once, keeping the sharing intact.
  #: `tied` counts shared, restored-shared and skipped empty tensors.
  tied = {"park": 0, "restore": 0, "empty": 0}
  #: Host copies indexed by `gid`; each `parked` entry records its slot's gid.
  parked_obj: list = []

  def _key(t):
    s = t.untyped_storage()
    return (t.device.type, t.device.index, s.data_ptr(), s.nbytes())

  def _restore(dev):
    #: Sharing is determined when parking and replayed here; re-deriving it
    #: from host tensors is unreliable (e.g. data_ptr() collisions).
    by_gid: dict = {}
    for mod, attr, is_param, gid in parked:
      if gid in by_gid:
        tied["restore"] += 1
        setattr(mod, attr, by_gid[gid])   # reuse the same object, no second copy
        continue
      t = getattr(mod, attr)
      moved = (torch.nn.Parameter(t.to(dev), requires_grad=False)
               if is_param else t.to(dev))
      by_gid[gid] = moved
      setattr(mod, attr, moved)
    n = len(parked)
    parked.clear()
    parked_obj.clear()
    return n

  def encode(*a, **k):
    dev = torch.device("tpu", int(os.environ.get("RANK", 0)))
    if parked:
      verbose(f"  [free-text] restored {_restore(dev)} tensors before encode "
              f"({tied['restore']} shared reuses)", flush=True)
    out = inner(*a, **k)
    if True:
      n = 0
      # Move each storage once and record sharing as `gid` for `_restore`.
      seen: dict = {}
      for mod in enc.modules():
        for attr, t, is_param in (
            [(a, x, True) for a, x in mod.named_parameters(recurse=False)]
            + [(a, x, False) for a, x in mod.named_buffers(recurse=False)]):
          if t is None or t.device.type == "cpu":
            continue
          # Skip empty tensors: they are placeholders of offload-managed
          # layers (real data on the host) and parking them would pin the
          # empty shell in place.
          if t.numel() == 0:
            tied["empty"] += 1
            continue
          k = _key(t)
          if k in seen:
            tied["park"] += 1
            gid = seen[k]
            setattr(mod, attr, parked_obj[gid])
          else:
            moved = t.detach().to("cpu")
            if is_param:
              moved = torch.nn.Parameter(moved, requires_grad=False)
            gid = len(parked_obj)
            seen[k] = gid
            parked_obj.append(moved)
            setattr(mod, attr, moved)
            n += 1
          parked.append((mod, attr, is_param, gid))
      # Two more things, or the C++ layer aborts with a cpu-vs-tpu device
      # check: remove offload hooks on the parked modules, and pin
      # `_execution_device` (otherwise inferred as cpu from the encoder).
      hooks = 0
      # Only clear hooks on modules we actually parked; other layers still
      # rely on their hooks to stage weights.
      touched = {id(m) for m, _, _, _ in parked}
      for mod in enc.modules():
        if id(mod) not in touched:
          continue
        for d in (getattr(mod, "_forward_pre_hooks", None),
                  getattr(mod, "_forward_hooks", None)):
          if d:
            hooks += len(d)
            d.clear()
      type(pipe)._execution_device = property(
          lambda _self, _d=torch.device("tpu",
                                        int(os.environ.get("RANK", 0))): _d)
      try:
        from torch_tpu._internal import compile as tt_compile
        tt_compile.clear_cache()
      except Exception:  # noqa: BLE001
        pass
      verbose(f"  [free-text] encode done, {n} storages moved to host "
              f"({tied['park']} shared, {tied['empty']} empty "
              f"skipped), {hooks} hooks removed, execution device pinned",
              flush=True)
    return out

  encoders.get_qwen3vl_prompt_embeds = encode



def _dump_latents(pipe, path, verbose=print):
  """Save the latents passed to the VAE decoder, then decode as usual.

  The decode only needs latents, so a dump lets decode experiments skip the
  weight load and the denoise.
  """
  import numpy as np
  import torch

  vae = getattr(pipe, "vae", None)
  if vae is None:
    return
  inner = vae.decode
  done = []

  def decode(z, *a, **k):
    if not done:
      done.append(True)
      arr = z.detach().to(torch.float32).to("cpu").numpy()
      np.save(path, arr)
      verbose(f"  [latents] {tuple(z.shape)} {z.dtype} -> {path}",
              flush=True)
    return inner(z, *a, **k)

  vae.decode = decode



def _clear_before_decode(pipe, verbose=print):
  """Clear cached compiled executables before the VAE decode.

  Loading the decode program can fail for lack of HBM ("Attempting to
  reserve 3.12G at the bottom of memory"), while the DiT's executables are
  no longer needed. `clear_cache()` drops `TpuBackend._compiled_executables`;
  the next request recompiles a block (~7 s).
  """
  vae = getattr(pipe, "vae", None)
  if vae is None:
    return
  inner = vae.decode
  state = {"n": 0}

  def decode(*a, **k):
    if not state["n"]:
      state["n"] = 1
      try:
        from torch_tpu._internal import compile as tt_compile

        tt_compile.clear_cache()
        verbose("  [decode] cleared TpuBackend executables before decode", flush=True)
      except Exception as exc:  # noqa: BLE001
        verbose(f"  [decode] clear_cache unavailable: {type(exc).__name__}",
                flush=True)
    return inner(*a, **k)

  vae.decode = decode



def _rank0_text_encode(rank, all_reduce, on_cpu=False, keep=0, verbose=print,
                       world=1):
  """Run the text encoder on rank 0 only and broadcast the embedding.

  Otherwise every rank streams the full 58.1 GiB of (unsharded) Qwen3-VL
  weights to compute the same result for a few prompt tokens.

  The broadcast goes over a host (gloo) group, not the chip interconnect:
  device collectives require all ranks to issue the same op sequence, which
  no longer holds after rank 0 alone ran the encoder, so they hang. The
  embedding is only a few KB.
  """
  import torch
  import torch.distributed as dist

  from diffusers.modular_pipelines.minimax_h3 import encoders

  del all_reduce        # kept for the signature; device collectives are not used
  original = encoders.get_qwen3vl_prompt_embeds
  # Separate host group; the default group uses the tpu backend.
  host_group = dist.new_group(backend="gloo") if world > 1 else None

  def _to_cpu(obj):
    if torch.is_tensor(obj):
      return obj.to("cpu")
    if isinstance(obj, dict):
      return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
      return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
      return tuple(_to_cpu(v) for v in obj)
    if hasattr(obj, "to"):
      try:
        return obj.to("cpu")
      except Exception:
        return obj
    return obj

  def encode(text_encoder, processor, token_ids, vision_inputs=None,
             **kwargs):
    # Ignore `kwargs["device"]`: `_execution_device` is inferred from
    # parameter placement and resolves to cpu if the encoder is on the host.
    device = torch.device("tpu", rank)
    verbose(f"  [rank0-text] rank {rank} in encode, "
            f"{'computing' if rank == 0 else 'waiting for broadcast'}", flush=True)
    if rank == 0:
      if on_cpu:
        # Run the whole encoder on the host: a few tokens through the model
        # is ~0.4 TFLOP and reading 58 GiB at host memory bandwidth takes
        # seconds, versus ~40 s to stream it to the chip at ~2 GB/s.
        kwargs = dict(kwargs, device=torch.device("cpu"))
        token_ids = _to_cpu(token_ids)
        vision_inputs = _to_cpu(vision_inputs)
      # Time the encode itself.
      _t0 = time.perf_counter()
      _restore = _truncate_layers(text_encoder, keep) if keep else None
      try:
        if on_cpu:
          with torch.device("cpu"):
            emb = original(text_encoder, processor, token_ids, vision_inputs,
                           **kwargs)
        else:
          emb = original(text_encoder, processor, token_ids, vision_inputs,
                         **kwargs)
      finally:
        if _restore is not None:
          _restore()
      verbose(f"  [rank0-text] encode took "
              f"{time.perf_counter() - _t0:.1f}s"
              f" ({'host' if on_cpu else 'chip'})", flush=True)
      payload = [emb.to("cpu")]
    else:
      payload = [None]
    # Shape and dtype travel with the pickled object; no handshake needed.
    if host_group is not None:
      dist.broadcast_object_list(payload, src=0, group=host_group)
    out = payload[0].to(device)
    verbose(f"  [rank0-text] rank {rank} received embedding "
            f"{tuple(out.shape)} {out.dtype}", flush=True)
    return out

  encoders.get_qwen3vl_prompt_embeds = encode
  return encode


def _stream_text_encoder(pipe, args):
  """Streams Qwen3-VL's decoder layers, the same way the DiT's stream."""
  import torch

  from diffusers_tpu.backends.layerwise_offload import LayerwiseOffloadManager

  # With rank0-text, non-zero ranks never run the encoder, so its layers
  # need not be streamed. Non-layer modules still go to the chip so that
  # `_execution_device` (inferred from parameters) does not resolve to cpu.
  skip_layers = ((getattr(args, "rank0_text", False)
                  or getattr(args, "cpu_text", False))
                 and int(os.environ.get("RANK", 0)) != 0)
  # `--cpu-text`: the encoder runs entirely on the host, on every rank.
  if getattr(args, "cpu_text", False):
    print("text encoder: runs entirely on the host (cpu-text)", flush=True)
    return None

  encoder = getattr(pipe, "text_encoder", None)
  # The text encoder streams even with `--no-offload` (which only applies
  # to the DiT); leaving it on the host would yield CPU embeddings.
  if encoder is None:
    return None
  for attr in ("model.language_model.layers", "language_model.layers",
               "model.layers", "layers"):
    node = encoder
    try:
      for part in attr.split("."):
        node = getattr(node, part)
    except AttributeError:
      continue
    # Shard before offloading: `offload_all_layers` copies each layer's
    # weights to a host-resident copy, which should already be the shard.
    if getattr(args, "shard_text", False):
      from diffusers_tpu.backends.tp_h3 import init_parallel
      from diffusers_tpu.backends.tp_h3 import shard_text_encoder

      _r, _w, _ar = init_parallel()
      shard_text_encoder(encoder, _r, _w, _ar)

    manager = None
    if not skip_layers:
      manager = LayerwiseOffloadManager(
          encoder, layers_attr_str=attr, num_layers=len(node),
          enabled=True, prefetch_size=args.prefetch,
      )
      manager.offload_all_layers(stage_from_host=True)
      manager.register_forward_hooks()
    _move_non_block_modules(encoder, torch.device("tpu"), blocks_attr=attr)
    print(f"text encoder: {'non-layer modules on chip, layers not streamed (rank0-text)' if skip_layers else f'streaming {len(node)} layers at {attr}'}",
          flush=True)
    return manager
  print("text encoder: no layer stack found; left on the host", flush=True)
  return None


def _chunked_decode(pipe, clip, overlap=0):
  """Decodes the clip in temporal chunks, each handed to the host.

  Each chunk's intermediates are a fraction of the clip's, and each result
  moves to the host as soon as it exists, so the chip holds at most one
  chunk. Latents are `(b, c, frames, h, w)`; H3's video VAE compresses
  time by four.
  """
  import functools

  import torch

  vae = pipe.vae
  inner = vae.decode

  seen = []

  @functools.wraps(inner)
  def decode(latents, *a, **kw):
    if not seen:
      seen.append(True)
      print(f"  [decode] latents {tuple(latents.shape)}", flush=True)
    # Fall back to an unchunked decode for unexpected shapes.
    if clip <= 0 or latents.ndim != 5:
      return inner(latents, *a, **kw)
    frames = latents.shape[2]
    if frames <= clip or frames <= 0:
      return inner(latents, *a, **kw)
    # Chunking is lossy and `overlap` does not fix it: at 480p/121, clip 8,
    # outputs differ from the unchunked decode already inside the first
    # chunk (mean |d| ~13.5 for overlap 0, 1 and 2). Lead-in only costs
    # frames; it is kept behind a flag (default 0).
    pieces, tupled = [], False
    for begin in range(0, frames, clip):
      lead = min(overlap, begin)
      part = inner(latents[:, :, begin - lead:begin + clip], *a, **kw)
      tupled = isinstance(part, tuple)
      got = part[0] if tupled else getattr(part, "sample", part)
      if lead:
        # Trim at the marginal rate of four pixel frames per latent frame;
        # absolute chunk lengths are not clean multiples of four.
        if got.shape[2] > 4 * lead:
          got = got[:, :, 4 * lead:]
        else:
          print(f"  [decode] chunk at {begin}: {got.shape[2]} frames is "
                f"not more than the {4 * lead}-frame lead-in; keeping all",
                flush=True)
      pieces.append(got.to("cpu"))
      del part, got
    # Back to the chip for the step's own arithmetic. The peak is now this
    # one buffer rather than the decode's much larger intermediates.
    if not pieces:
      return inner(latents, *a, **kw)
    whole = torch.cat(pieces, dim=2).to(latents.device)
    return (whole,) if tupled else whole

  vae.decode = decode


def _fix_reference_encode():
  """Pins the reference encoder's two constants to the CPU.

  `encode_vae_condition` builds `latents_mean` and `latents_std` with
  a bare `torch.tensor(...)`, then moves the sampled latents to the
  host with an explicit `.cpu()` and subtracts. Under this backend's
  `with torch.device("tpu")` the two constants land on the chip while
  the latents are on the host, and the subtraction raises

      Expected all tensors to be on the same device, but found at
      least two devices, tpu:0 and cpu!

  This is the upstream body with only the two constructions pinned.
  """
  import torch
  from diffusers.modular_pipelines.minimax_h3 import encoders as enc

  original = enc.encode_vae_condition

  def encode_vae_condition(vae, pixels, pixel_mean, pixel_std,
                           encode_seed=42):
    cpu = torch.device("cpu")
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=cpu).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std, device=cpu).view(1, -1, 1, 1, 1)
    p_mean = torch.tensor(
        pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    p_std = torch.tensor(
        pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)

    scaled = (pixels.to(torch.float32).div(255.0) - p_mean) / p_std
    posterior = vae.encode(scaled, return_dict=False)[0]
    latents = posterior.sample(
        generator=torch.Generator().manual_seed(encode_seed))
    latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std

  enc.encode_vae_condition = encode_vae_condition
  del original


_HBM_ROWS: list = []


def _vae_on_demand(pipe, verbose=print):
  """Keep the VAE on the host; move it to the chip only for encode/decode.

  The VAE holds 9.70 GiB of HBM (ref2va, 31.25 GiB chip) and is idle
  during the whole denoise, which can leave too little memory to load the
  decode program.
  """
  import torch

  vae = getattr(pipe, "vae", None)
  if vae is None:
    return

  def move(mod, dev):
    n = 0
    for sub in mod.modules():
      for attr, t in list(sub.named_parameters(recurse=False)):
        if t.device != dev:
          sub._parameters[attr] = torch.nn.Parameter(
              t.detach().to(dev), requires_grad=False)
          n += 1
      for attr, t in list(sub.named_buffers(recurse=False)):
        if t is not None and t.device != dev:
          sub._buffers[attr] = t.detach().to(dev)
          n += 1
    return n

  host = torch.device("cpu")
  move(vae, host)          # start on the host
  verbose("  [vae-on-demand] VAE parked on the host (9.70GiB)", flush=True)

  # Wrap both entry points: ref2va also calls `vae.encode` for the
  # reference image before the denoise.
  said = {"up": 0}

  def wrap(name):
    inner = getattr(vae, name)

    def staged(z, *a, **k):
      dev = getattr(z, "device", None) or torch.device("tpu")
      if dev.type == "cpu":
        dev = torch.device("tpu", int(os.environ.get("RANK", 0)))
      up = move(vae, dev)
      if up and said["up"] < 2:
        said["up"] += 1
        verbose(f"  [vae-on-demand] before {name}: moved {up} tensors to chip", flush=True)
      try:
        return inner(z, *a, **k)
      finally:
        down = move(vae, host)
        if down and said["up"] <= 2:
          verbose(f"  [vae-on-demand] after {name}: moved {down} back to host", flush=True)

    setattr(vae, name, staged)

  for name in ("encode", "decode"):
    if hasattr(vae, name):
      wrap(name)



def _dit_of(pipe):
  """The DiT: `transformer` for t2va, `transformer_ref` for ref2va."""
  for name in ("transformer", "transformer_ref"):
    try:
      return getattr(pipe, name)
    except AttributeError:
      pass
  raise AttributeError("pipeline has neither transformer nor transformer_ref")


def _install_step_cache(pipe, threshold, verbose=print):
  """Lossy step skipping, ported from torchax's `--cache-threshold`.

  Before each DiT call, compute the cheap timestep embedding and accumulate
  its relative change since the last computed step; below the threshold,
  return the previous output. All block modulations are affine in the
  timestep embedding, so this is a cheap proxy for how much a step changes.

  As in torchax, the embedding is averaged over its first dim (ref2va
  alternates 2 and 3 rows), and state must be reset per request.
  Returns that reset function.
  """
  import torch

  tr = _dit_of(pipe)
  # Wrap the existing instance `forward` (from `_inputs_to_device`), not the
  # class method: the instance attribute takes precedence.
  inner = tr.forward
  st = {"last_mod": None, "last_out": None, "drift": 0.0,
        "skipped": 0, "calls": 0}

  import functools

  # Keep the original signature: the denoise loop filters arguments by
  # `inspect.signature(transformer.forward)`, which follows `__wrapped__`.
  @functools.wraps(inner)
  def forward(*a, **k):
    if not st.get("announced"):
      st["announced"] = True
      verbose("  [step-cache] wrapper reached for the first time", flush=True)
    timestep = k.get("timestep")
    if timestep is None and len(a) > 3:
      timestep = a[3]
    if timestep is not None and timestep.device.type != "tpu":
      timestep = timestep.to(next(tr.time_embedder.parameters()).device)
    st["calls"] += 1
    with torch.no_grad():
      proj = tr.time_proj(timestep)
      cur = tr.time_embedder(
          proj.to(next(tr.time_embedder.parameters()).dtype)).mean(0)
    if st["last_mod"] is not None and st["last_out"] is not None:
      moved = ((cur - st["last_mod"]).abs().mean()
               / (st["last_mod"].abs().mean() + 1e-9)).item()
      st["drift"] += moved
      if st["drift"] < threshold:
        st["skipped"] += 1
        return st["last_out"]
      st["drift"] = 0.0
    st["last_mod"] = cur
    out = inner(*a, **k)
    st["last_out"] = out
    return out

  tr.forward = forward

  def reset():
    if st["calls"]:
      verbose(f"  [step-cache] previous request: {st['calls']} calls, "
              f"{st['skipped']} skipped", flush=True)
    st.update(last_mod=None, last_out=None, drift=0.0, skipped=0, calls=0)

  verbose(f"  [step-cache] installed, threshold {threshold}", flush=True)
  return reset


def _dit_park_for_decode(pipe, verbose=print):
  """Move resident DiT blocks to the host during the VAE decode, then back.

  Mirror of `_vae_on_demand`; only meaningful with `--no-offload`. A
  resident DiT speeds up the t2va denoise (169 -> 34 s, byte-identical
  latents) but its 9.12 GiB leaves too little HBM for the decode program.
  Each move takes ~4 s at `.to()` bandwidth.
  """
  import torch

  blocks = list(getattr(_dit_of(pipe), "transformer_blocks", []))
  vae = getattr(pipe, "vae", None)
  if not blocks or vae is None:
    verbose("  [dit-park] transformer_blocks or vae not found; not installed", flush=True)
    return

  def move(dev):
    n = 0
    for blk in blocks:
      for sub in blk.modules():
        for attr, t in list(sub.named_parameters(recurse=False)):
          if t.device != dev:
            sub._parameters[attr] = torch.nn.Parameter(
                t.detach().to(dev), requires_grad=False)
            n += 1
        for attr, t in list(sub.named_buffers(recurse=False)):
          if t is not None and t.device != dev:
            sub._buffers[attr] = t.detach().to(dev)
            n += 1
    return n

  host = torch.device("cpu")
  chip = torch.device("tpu", int(os.environ.get("RANK", 0)))
  inner = vae.decode
  said = {"n": 0}

  def decode(*a, **k):
    t0 = time.perf_counter()
    down = move(host)
    t1 = time.perf_counter()
    try:
      return inner(*a, **k)
    finally:
      t2 = time.perf_counter()
      up = move(chip)
      t3 = time.perf_counter()
      if said["n"] < 2:
        said["n"] += 1
        verbose(f"  [dit-park] moved {down} off before decode ({t1 - t0:.1f}s), "
                f"{up} back after ({t3 - t2:.1f}s)", flush=True)

  vae.decode = decode
  verbose(f"  [dit-park] installed: {len(blocks)} blocks on the host during decode",
          flush=True)


def _hbm_snapshot(stage, pipe=None, verbose=print):
  """Log per-module HBM usage and the compilation cache's HBM at a stage.

  `_hbm_usage_summary()` only reports the compilation cache, so module
  weights are counted from parameter bytes.
  """
  import torch
  from torch_tpu._internal.device._device_module import TpuDeviceModule

  def on_chip(mod):
    if mod is None:
      return 0
    n = 0
    for t in list(mod.parameters()) + list(mod.buffers()):
      if t is not None and t.device.type != "cpu":
        n += t.numel() * t.element_size()
    return n

  parts = {}
  if pipe is not None:
    try:
      dit = _dit_of(pipe)
    except AttributeError:
      dit = getattr(pipe, getattr(pipe, "_dit_name", "transformer"), None)
    parts["DiT"] = on_chip(dit)
    parts["text_encoder"] = on_chip(getattr(pipe, "text_encoder", None))
    parts["VAE"] = on_chip(getattr(pipe, "vae", None))
  try:
    cache = str(TpuDeviceModule._hbm_usage_summary())
  except Exception as exc:  # noqa: BLE001
    cache = f"{type(exc).__name__}"
  row = (stage, {k: v / 2**30 for k, v in parts.items()}, cache)
  _HBM_ROWS.append(row)
  bits = "  ".join(f"{k} {v:.2f}GiB" for k, v in row[1].items())
  verbose(f"  [hbm] {stage:<14} {bits}  | {cache}", flush=True)



def _batch_decode_tiles(vae, batch, verbose=print):
  """Decode `_decode_clip`'s spatial tiles `batch` at a time along the batch dim.

  The stock path decodes each tile (28 per clip at 768x1344) separately;
  batching tiles took the native JAX pipeline's 345-frame decode from
  12.8 to 7.7 s. Slicing, `post_quant_conv` and `_stitch_tiles` are unchanged.
  """
  import torch

  if not vae.use_tiling:
    return

  def decode_clip(z):
    ratio = vae.spatial_compression_ratio
    height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
    ys, yl, yo = vae._split_tiles(height, vae.tile_sample_min_height,
                                  vae.tile_sample_min_overlap_height)
    xs, xl, xo = vae._split_tiles(width, vae.tile_sample_min_width,
                                  vae.tile_sample_min_overlap_width)
    tiles = [z[..., i // ratio:i // ratio + il // ratio,
               j // ratio:j // ratio + jl // ratio]
             for i, il in zip(ys, yl) for j, jl in zip(xs, xl)]
    outs = []
    for s0 in range(0, len(tiles), batch):
      group = tiles[s0:s0 + batch]
      n = len(group)
      group = group + [group[-1]] * (batch - n)     # pad the last group to avoid a recompile
      out = vae.decoder(vae.post_quant_conv(torch.cat(group, dim=0)))
      outs += list(out[:n].split(1, dim=0))
    ncol = len(xs)
    rows = [outs[r * ncol:(r + 1) * ncol] for r in range(len(ys))]
    return vae._stitch_tiles(rows, yo, xo)

  vae._decode_clip = decode_clip
  verbose(f"  [decode] decoding tiles in batches of {batch}", flush=True)


def _shard_decode(vae, verbose=print):
  """Split the VAE decode's chunks across ranks and reassemble on the host.

  Same loop as `_host_decode`, except each rank decodes chunks with
  `i % world == rank` and receives the rest; the blend/trim arithmetic is
  unchanged, so output is bit-identical. Each `_decode_clip` slice carries
  its own `token_overlap` and blending happens afterwards on the host, so
  no tensor sharding or device collective is needed (results go over a
  host gloo group, ~80 MB per chunk).
  """
  import os

  import torch
  import torch.distributed as dist

  rank = int(os.environ.get("RANK", 0))
  world = int(os.environ.get("WORLD_SIZE", 1))
  if world == 1:
    verbose("  [shard-decode] single rank; not splitting", flush=True)
    return
  # Results are gathered on the host, so use a separate gloo group (the
  # default group uses the tpu backend).
  group = dist.new_group(backend="gloo")
  original = vae._decode

  def _decode(z):
    # `DT_RANK_STAMPS=1`: log host timestamps at a few points per rank,
    # without synchronizing, to locate straggler ranks.
    _st = os.environ.get("DT_RANK_STAMPS") == "1"
    _t_in = time.perf_counter()
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = ((num_tokens + pad_tokens) // tokens_chunk_size
                  - int(token_drop > 0))
    if pad_tokens > 0:
      z = torch.cat(
          [z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    # Step 1: decode this rank's chunks into a list of (chunk, j, tensor).
    mine: list = []
    # `DT_DECODE_TILES=1`: distribute spatial tiles rather than clips.
    # 13 clips over 8 ranks puts 2 clips on the critical path; 13x28 tiles
    # give each rank 45-46 tiles (~19% less than 56). Each tile decodes as
    # in `_decode_clip`; stitching uses the VAE's own `_stitch_tiles` on the
    # host. Requires DT_DECODE_SHM=1 (tiles exchanged via /dev/shm).
    if (os.environ.get("DT_DECODE_TILES") == "1"
        and os.environ.get("DT_DECODE_SHM") == "1" and vae.use_tiling):
      import numpy as np
      ratio = vae.spatial_compression_ratio
      H = z.shape[-2] * ratio
      W = z.shape[-1] * ratio
      yi, yl, yo = vae._split_tiles(H, vae.tile_sample_min_height,
                                    vae.tile_sample_min_overlap_height)
      xi, xl, xo = vae._split_tiles(W, vae.tile_sample_min_width,
                                    vae.tile_sample_min_overlap_width)
      tasks = [(i, a, b) for i in range(num_chunks)
               for a in range(len(yi)) for b in range(len(xi))]
      run_id = getattr(_decode, "_tcalls", 0)
      _decode._tcalls = run_id + 1
      tbase = f"/dev/shm/h3_decode/t{run_id}"
      os.makedirs(tbase, exist_ok=True)
      mine_t = []
      for (i, a, b) in tasks[rank::world]:
        zc = z[:, :, i * tokens_chunk_size:
               i * tokens_chunk_size + tokens_chunk_size + vae.token_overlap]
        tile = zc[..., yi[a] // ratio: yi[a] // ratio + yl[a] // ratio,
                  xi[b] // ratio: xi[b] // ratio + xl[b] // ratio]
        out = vae.decoder(vae.post_quant_conv(tile)).to("cpu")
        name = f"{tbase}/{i}_{a}_{b}"
        np.save(name + ".tmp.npy", out.to(torch.float32).numpy())
        os.replace(name + ".tmp.npy", name + ".npy")
        mine_t.append(name + ".npy")
      dt = out.dtype if mine_t else torch.bfloat16
      dist.barrier(group=group)
      for i in range(num_chunks):
        rows = [[torch.from_numpy(np.load(f"{tbase}/{i}_{a}_{b}.npy")).to(dt)
                 for b in range(len(xi))] for a in range(len(yi))]
        clip = vae._stitch_tiles(rows, yo, xo)
        for j in range(int(token_drop > 0) + 1):
          frame_start = j * chunk_num_frames
          chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
          got_t = chunk[:, :, vae.frame_pre_padding:]
          mine.append((i, j, got_t))
      dist.barrier(group=group)
      for f in mine_t:
        try:
          os.remove(f)
        except FileNotFoundError:
          pass
      _tiles_done = True
    else:
      _tiles_done = False
    # `DT_DECODE_BATCH=1`: decode this rank's chunks as one batch.
    # 18.5 -> 16.2 s but not bit-identical (max diff 3.5e-2, reduction
    # order changes), so off by default.
    if os.environ.get("DT_DECODE_BATCH") == "1":
      todo = [i for i in range(num_chunks) if i % world == rank and not _tiles_done]
      slices = [z[:, :, i * tokens_chunk_size:
                  i * tokens_chunk_size + tokens_chunk_size + vae.token_overlap]
                for i in todo]
      batched = (len(slices) > 1
                 and all(x.shape == slices[0].shape for x in slices))
      if batched:
        big = vae._decode_clip(torch.cat(slices, dim=0))
        clips = [big[k:k + 1] for k in range(len(slices))]
      else:
        clips = [vae._decode_clip(x) for x in slices]
    else:
      todo = [i for i in range(num_chunks) if i % world == rank and not _tiles_done]
      # Per-chunk timing (DT_DECODE_TIMING=1). Synchronize after each chunk,
      # otherwise async dispatch attributes compute time to the later .to("cpu").
      _tm = os.environ.get("DT_DECODE_TIMING") == "1"
      clips = []
      for i in todo:
        _t0 = time.perf_counter()
        clips.append(vae._decode_clip(
            z[:, :, i * tokens_chunk_size:
              i * tokens_chunk_size + tokens_chunk_size
              + vae.token_overlap]))
        if _tm:
          _t1 = time.perf_counter()
          torch.accelerator.synchronize()
          verbose(f"  [dec-time] rank {rank} chunk {i}: dispatch {_t1 - _t0:.1f}s, "
                  f"done {time.perf_counter() - _t0:.1f}s", flush=True)
    _t2 = time.perf_counter()
    for i, clip in zip(todo, clips):
      start = i * tokens_chunk_size
      del start
      for j in range(int(token_drop > 0) + 1):
        frame_start = j * chunk_num_frames
        chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
        mine.append((i, j, chunk[:, :, vae.frame_pre_padding:].to("cpu")))
    if os.environ.get("DT_DECODE_TIMING") == "1":
      verbose(f"  [dec-time] rank {rank} to host {time.perf_counter() - _t2:.1f}s",
              flush=True)
    _t_cpu = time.perf_counter()

    # Step 2: every rank receives all chunks and assembles the clip itself.
    # Gathering only to rank 0 is faster but makes the ranks' workloads
    # diverge, and the next request then hangs; identical work on every
    # rank rules that out. Assembly is host CPU work.
    _t3 = time.perf_counter()
    got = {}
    if _tiles_done:
      # in tile mode every rank already assembled all clips
      for i, j, t in mine:
        got[(i, j)] = t
    elif os.environ.get("DT_DECODE_SHM") == "1":
      # Exchange via /dev/shm instead of gloo's all_gather_object, which
      # pickles ~1 GB per rank over TCP loopback (7.5-14 s per rank, slower
      # than the decode itself). Every rank still reads everything.
      import numpy as np
      run_id = getattr(_decode, "_calls", 0)
      _decode._calls = run_id + 1
      base = f"/dev/shm/h3_decode/r{run_id}"
      os.makedirs(base, exist_ok=True)
      for i, j, t in mine:
        a = t.to(torch.float32).numpy() if t.dtype == torch.bfloat16 else t.numpy()
        tmp = f"{base}/{i}_{j}.{t.dtype}.tmp.npy"
        np.save(tmp, a)
        os.replace(tmp, f"{base}/{i}_{j}.{t.dtype}.npy")   # atomic write
      dist.barrier(group=group)
      for name in os.listdir(base):
        if not name.endswith(".npy") or ".tmp." in name:
          continue
        ij, dt = name[:-4].split(".", 1)
        i, j = (int(x) for x in ij.split("_"))
        t = torch.from_numpy(np.load(f"{base}/{name}"))
        if dt == "torch.bfloat16":
          t = t.to(torch.bfloat16)
        got[(i, j)] = t
      dist.barrier(group=group)          # delete only after everyone has read
      # Each rank deletes its own files, so no single rank lags into the
      # next request. The directory itself is kept for reuse.
      for i, j, t in mine:
        try:
          os.remove(f"{base}/{i}_{j}.{t.dtype}.npy")
        except FileNotFoundError:
          pass
    else:
      boxes = [None] * world
      dist.all_gather_object(boxes, mine, group=group)
      for box in boxes:
        for i, j, t in box:
          got[(i, j)] = t
    if os.environ.get("DT_DECODE_TIMING") == "1":
      verbose(f"  [dec-time] rank {rank} gather {time.perf_counter() - _t3:.1f}s",
              flush=True)
    if _st:
      _t_out = time.perf_counter()
      verbose(f"  [stamp] rank {rank} decode start {_t_in % 10000:.1f} "
              f"decoded+to host {_t_cpu % 10000:.1f} gathered {_t_out % 10000:.1f}",
              flush=True)

    # Step 3: blend and trim exactly as `_host_decode` does, in order.
    decoded_chunks = []
    overlap = None
    for i in range(num_chunks):
      for j in range(int(token_drop > 0) + 1):
        chunk = got[(i, j)]
        if j == 0:
          if overlap is not None:
            chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
          decoded_chunks.append(chunk)
        else:
          overlap = chunk
    if overlap is not None:
      decoded_chunks.append(overlap)

    dec = torch.cat(decoded_chunks, dim=2)
    if pad_tokens > 0:
      intra_tail = vae.config.clip_length % temporal_ratio
      before = z.shape[2] - pad_tokens
      pad_frames = sum(
          intra_tail if intra_tail and (before + k) % tokens_chunk_size == 0
          else temporal_ratio
          for k in range(pad_tokens))
      dec = dec[:, :, :-pad_frames]
    return dec.to(z.device)

  vae._decode = _decode
  verbose(f"  [shard-decode] rank {rank}/{world}, chunks split across ranks", flush=True)


def _host_decode(vae):
  """Keeps `_decode`'s per-chunk results on the host instead of the chip.

  `AutoencoderKLMiniMaxH3._decode` accumulates every decoded chunk in
  `decoded_chunks` and `torch.cat`s at the end, so the whole clip is
  resident before anything leaves -- 1.7 GB at 480p/362, several of
  them live at once, and one chip runs out.

  Slicing the latents from outside (`--vae-clip`) is lossy because it
  duplicates this method's own chunking (overlapping, cross-faded chunks
  with leading pads). Instead this is `_decode`'s own loop with one
  change: each chunk moves to the host as it is produced and the concat
  happens there. The arithmetic is untouched.
  """
  import torch

  original = vae._decode

  def _decode(z):
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = ((num_tokens + pad_tokens) // tokens_chunk_size
                  - int(token_drop > 0))
    if pad_tokens > 0:
      z = torch.cat(
          [z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks = []
    overlap = None
    for i in range(num_chunks):
      start = i * tokens_chunk_size
      clip = vae._decode_clip(
          z[:, :, start:start + tokens_chunk_size + vae.token_overlap])
      for j in range(int(token_drop > 0) + 1):
        frame_start = j * chunk_num_frames
        chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
        chunk = chunk[:, :, vae.frame_pre_padding:].to("cpu")
        if j == 0:
          if overlap is not None:
            chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
          decoded_chunks.append(chunk)
        else:
          overlap = chunk
      del clip
    if overlap is not None:
      decoded_chunks.append(overlap)

    dec = torch.cat(decoded_chunks, dim=2)
    if pad_tokens > 0:
      intra_tail = vae.config.clip_length % temporal_ratio
      before = z.shape[2] - pad_tokens
      pad_frames = sum(
          intra_tail if intra_tail and (before + k) % tokens_chunk_size == 0
          else temporal_ratio
          for k in range(pad_tokens))
      dec = dec[:, :, :-pad_frames]
    return dec.to(z.device)

  vae._decode = _decode
  del original


def _decode_to_host(vae):
  """Hands the decoder's output to the host as it comes out."""
  import functools

  import torch

  inner = vae.decode

  @functools.wraps(inner)
  def decode(*a, **kw):
    out = inner(*a, **kw)
    sample = getattr(out, "sample", None)
    if torch.is_tensor(sample):
      out.sample = sample.to("cpu")
      return out
    return out.to("cpu") if torch.is_tensor(out) else out

  vae.decode = decode


def _tensor_attrs_to_device(obj, dev):
  """Moves an object's tensor attributes to the chip, in place.

  Schedulers are not modules, so `.to()` does not exist and the tensors
  are ordinary attributes. They are also rebuilt by `set_timesteps`, so
  this is wrapped around that call rather than done once.
  """
  import torch

  if obj is None:
    return
  for attr, value in list(vars(obj).items()):
    if torch.is_tensor(value) and value.device.type != "tpu":
      setattr(obj, attr, value.to(dev))
  inner = getattr(obj, "set_timesteps", None)
  if inner is not None and not getattr(obj, "_tpu_wrapped", False):
    import functools

    @functools.wraps(inner)
    def set_timesteps(*a, **kw):
      out = inner(*a, **kw)
      for attr, value in list(vars(obj).items()):
        if torch.is_tensor(value) and value.device.type != "tpu":
          setattr(obj, attr, value.to(dev))
      return out

    obj.set_timesteps = set_timesteps
    obj._tpu_wrapped = True


def _generate(pipe, args, reference):
  import torch

  kwargs = dict(
      prompt=args.prompt,
      height=args.height,
      width=args.width,
      num_frames=args.num_frames,
      num_inference_steps=args.steps,
      generator=torch.Generator(device="cpu").manual_seed(args.seed),
      output=["videos", "audio", "sampling_rate"],
  )
  # Request numpy output rather than PIL: converting hundreds of large
  # frames to PIL on every rank contends for host memory bandwidth and
  # slows ranks down. `_write` converts to uint8 with the same formula as
  # `numpy_to_pil`, so written frames are byte-identical.
  # `DT_OUTPUT_PIL=1` restores the old behaviour.
  if os.environ.get("DT_OUTPUT_PIL") != "1":
    kwargs["output_type"] = "np"
  if reference is not None:
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    # The pipeline input is `references` (plural, a list).
    kwargs["references"] = [MiniMaxH3ImageReference.from_file(args.reference)]
  return pipe(**kwargs)


def _write(result, args):
  import numpy as np

  frames = result.get("videos") if hasattr(result, "get") else None
  if frames is None:
    print("no frames on the result; nothing written", flush=True)
    return
  if os.environ.get("RANK", "0") != "0":
    return    # every rank holds the frames; eight writers would race on one file
  path = os.path.join(args.out, f"h3_{args.tag}_frames.npy")
  arr = np.asarray(frames[0])
  if arr.dtype != np.uint8:
    # output_type="np" yields floats in [0, 1]; same formula as `numpy_to_pil`.
    arr = (arr * 255).round().astype("uint8")
  tmp = path[:-len(".npy")] + ".tmp.npy"
  np.save(tmp, arr)
  os.replace(tmp, path)
  print(f"wrote {path}", flush=True)


if __name__ == "__main__":
  main()
