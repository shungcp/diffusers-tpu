"""MiniMax-H3 native JAX, resident server: load models once, serve requests.

`jax_pipeline.py` handles one request per process (the text tower is freed
after encoding and the DiT before decoding), so a second request would reload
and recompile. This splits it into a load phase and a per-request phase:

* **Resident on chip**: DiT (tensor parallel, ~7.5 GiB per chip) and the
  decoder (tensor parallel -- a replicated decoder does not fit next to a
  resident DiT).
* **Resident on host, moved to chip per request**: text tower (62 GiB,
  7.75 per chip), vision tower, VAE encoder. `device_put` per request and
  deleted right after use.
* **Per-request arrays passed as arguments**: the packed layout
  (`position_ids` / `token_tags` / row indices / per-row timestep index).
  `jax_pipeline` closes over them as constants, so every prompt is a new
  program; here prompts of the same length share one compilation. Sparse
  block tables are cached per layout.

The per-request math is the same as `jax_pipeline` (same modules, same wiring).

  python -m diffusers_tpu.models.minimax_h3.jax_server --requests reqs.jsonl --out out

Each line of `reqs.jsonl` is a JSON object: `{"tag": .., "prompt": ..,
"reference": optional, "seed": optional, "lossy": optional,
"num_frames": optional, "height": optional, "width": optional}`.
"""

import argparse
import functools
import json
import os
import time

import numpy as np

from .jax_pipeline import (TEXT_ENCODER_LAYER, _step_sigmas, decode_chunk_batch,
                           scheduler_step, sigma_schedule)


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--requests", required=True, help="JSON lines file, one request per line")
  ap.add_argument("--out", default="out")
  ap.add_argument("--height", type=int, default=768)
  ap.add_argument("--width", type=int, default=1344)
  ap.add_argument("--num-frames", type=int, default=345)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--shift", type=float, default=12.0)
  ap.add_argument("--audio-shift", type=float, default=3.0)
  ap.add_argument("--block-q", type=int, default=2048)
  ap.add_argument("--block-kv", type=int, default=2048,
                  help="2048 is fastest for ref2va and t2va up to 241 frames; "
                       "4096 is 3 s faster for lossless 345-frame t2va")
  ap.add_argument("--pad-sequence-to", type=int, default=128)
  ap.add_argument("--full-lane-rope", action=argparse.BooleanOptionalAction,
                  default=True)
  ap.add_argument("--decode-chunk-batch", type=int, default=0)  # 0 = auto
  # With a resident DiT and a tensor-parallel decoder, per-chunk scratch memory
  # is much larger than in the pipeline: K=10 needs 42.58 GiB and OOMs.
  # Auto only balances chunks under this cap. 0 = 8 at 480p and below
  # (2.7 -> 2.4 s), 4 above (8 at 768p needs 27.6 GiB and OOMs).
  ap.add_argument("--decode-chunk-cap", type=int, default=0)
  ap.add_argument("--no-audio", action="store_true",
                  help="Skip the audio VAE decode (video-only output).")
  ap.add_argument("--vae-tile-overlap-px", type=int, default=32)
  ap.add_argument("--cache-threshold", type=float, default=None,
                  help="step-skip threshold for requests with lossy=true; defaults "
                       "per workflow: t2va 0.20, ref2va 0.15 (chosen by visual review)")
  ap.add_argument("--sparse-window", type=int, default=16,
                  help="sparse attention window for requests with lossy=true (t2va and ref2va)")
  ap.add_argument("--sparse-dense-steps", type=int, default=6)
  ap.add_argument("--text-resident", action="store_true",
                  help="Keep the text and vision towers on the chips after the "
                       "first request instead of moving them from host each "
                       "time. Fits at 480p beside the DiT and decoder; at 768p "
                       "the decoder no longer fits.")
  ap.add_argument("--pika-fl2va", default="",
                  help="Pika-H3-Flex fl2va adapter dir: text-to-video requests use "
                       "its six-step ladder and re-noising sampler")
  ap.add_argument("--pika-ref2va", default="",
                  help="Pika-H3-Flex ref2va adapter dir, for reference requests")
  ap.add_argument("--compile-cache", default="")
  ap.add_argument("--warmup", default="",
                  help="request file used to precompile every shape at startup (3 steps, "
                       "sparse from step 1, results discarded) so the first real request "
                       "runs at steady state. Same format as --requests")
  ap.add_argument("--fp8", action="store_true",
                  help="float8_e4m3fn weights and activations for DiT linears (native fp8 MXU "
                       "on v7x). Do not use on v6e (no fp8 MXU): lossless t2va 112.4 -> 120.7 s, "
                       "DINO 0.9709 -> 0.9377")
  ap.add_argument("--qk-fp8", action="store_true",
                  help="Cast Q and K to float8_e4m3fn inside Pallas splash attention")
  ap.add_argument("--chips", type=int, default=0,
                  help="Use this many JAX devices; 0 uses all available devices")
  ap.add_argument("--text-bucket", type=int, default=64,
                  help="pad text length to a multiple of this so prompts in the same bucket "
                       "share one compilation (attention masks to the real length); 0 disables")
  return ap.parse_args()


class Server:
  """Load once; call `generate(req)` per request."""

  def __init__(self, args):
    os.environ.setdefault(
        "LIBTPU_INIT_ARGS",
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_scoped_vmem_limit_kib=65536",
    )
    os.environ.setdefault("DT_PAD_SEQUENCE_TO", str(args.pad_sequence_to))
    os.environ.setdefault("DT_FULL_LANE_ROPE", "1" if args.full_lane_rope else "0")
    import jax
    import jax.numpy as jnp
    from flax import nnx

    if args.compile_cache:
      jax.config.update("jax_compilation_cache_dir", args.compile_cache)
      jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
      jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from . import jax_dit as dit
    from . import jax_text as jtext
    from . import jax_vae as jvae
    from . import jax_vision as jvision
    from ..qwen3_vl import jax_vision as _qv
    _qv.ATTN_CHUNK = int(os.environ.get("DT_VISION_ATTN_CHUNK", "2048"))
    # Shard the vision tower's 16 heads across chips (2 per chip on 8 chips),
    # which keeps the 2048 attention chunk affordable. `DT_VISION_HEAD_SHARD=0`
    # computes all heads on every chip.
    if os.environ.get("DT_VISION_HEAD_SHARD", "1") == "1":
      _qv.HEAD_SHARD_AXIS = "tp"
    from diffusers_tpu.kernels import mesh_attention
    from diffusers_tpu.kernels import splash as csa
    from jax.sharding import NamedSharding, PartitionSpec as P

    self.a, self.jax, self.jnp, self.nnx = args, jax, jnp, nnx
    self.dit, self.jvae = dit, jvae
    self.mesh_attention = mesh_attention
    self.NS, self.P = NamedSharding, P
    t0 = time.perf_counter()
    devices = jax.devices()[:args.chips] if args.chips else jax.devices()
    self.mesh = mesh = jax.make_mesh((len(devices),), ("tp",),
                                     devices=devices,
                                     axis_types=(jax.sharding.AxisType.Auto,))
    self.snapshot = snapshot = dit.find_snapshot()
    self.dtype = dtype = jnp.bfloat16
    text_dir = os.path.join(snapshot, "text_encoder")

    # -- Host-resident: text tower, vision tower, VAE encoder -------------
    from transformers import AutoTokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snapshot, "tokenizer"))
    self.encoder, self.cfg_text = jtext.build(text_dir, dtype=dtype, abstract=True)
    jtext.load(self.encoder, text_dir, mesh=mesh, spec_fn=jtext.param_spec,
               memory_kind="pinned_host")
    self.vision, _ = jvision.build(text_dir, dtype=dtype, abstract=True)
    jvision.load(self.vision, text_dir, mesh=mesh, spec_fn=lambda k, n: P(),
                 memory_kind="pinned_host")
    self._load_vae_encoder()
    print(f"[server] text tower / vision tower / VAE encoder on host {time.perf_counter() - t0:.0f}s",
          flush=True)

    # -- Resident on chip: DiT -------------------------------------------
    _, Transformer = dit._build_modules()
    transformer = nnx.eval_shape(
        lambda: Transformer(**dit.CONFIG_FULL, dtype=dtype, rngs=nnx.Rngs(0)))
    # Two weight sets: t2va uses transformer/, ref2va uses transformer_ref/ (same
    # config). Both do not fit at once, so the active one is reloaded on workflow switch.
    self.Transformer = Transformer
    self.dit_kind = None
    self.pika = None
    self.transformer = transformer
    self._sdpa_build()
    self._load_decoder()
    self.calls = {}           # (kind, sparse layout) -> compiled DiT call
    print(f"[server] ready {time.perf_counter() - t0:.0f}s", flush=True)

  # ---------------------------------------------------------------------
  def _to_device(self, tree):
    jax = self.jax
    return jax.tree.map(
        lambda v: jax.device_put(v, self.NS(self.mesh, v.sharding.spec,
                                            memory_kind="device")), tree)

  def _load_vae_encoder(self):
    import glob
    import torch
    from ...safetensors_io import safe_open
    jax, nnx, jvae = self.jax, self.nnx, self.jvae
    vae_dir = os.path.join(self.snapshot, "vae")
    params = {}
    for shard in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))):
      with safe_open(shard, framework="pt") as reader:
        for k in reader.keys():
          if k.startswith(("encoder.", "quant_conv.")):
            params[k] = reader.get_tensor(k).to(torch.float32).numpy()
    ecfg = jvae.encoder_config(vae_dir)
    enc = jvae.Encoder3d(dtype=self.jnp.float32, rngs=nnx.Rngs(0), **ecfg)
    jvae.load_encoder(enc, params)
    qc = jvae.PostQuantConv(ecfg["latent_channels"] * 2, dtype=self.jnp.float32,
                            rngs=nnx.Rngs(0))
    jvae.load_quant_conv(qc, params, prefix="quant_conv.")
    self._eg, es = nnx.split(enc)
    self._qg, qs = nnx.split(qc)
    host = self.NS(self.mesh, self.P(), memory_kind="pinned_host")
    self._es_host = jax.device_put(es, host)
    self._qs_host = jax.device_put(qs, host)
    with open(os.path.join(vae_dir, "config.json")) as f:
      self.vae_cfg = json.load(f)

    eg, qg = self._eg, self._qg

    @jax.jit
    def encode_one(es, qs, tile):
      with jax.default_matmul_precision(os.environ.get("DT_VAE_ENC_PRECISION", "high")):
        return nnx.merge(qg, qs)(nnx.merge(eg, es)(tile))
    self._encode_one = encode_one

  def _sdpa_build(self):
    jax, jnp, a, mesh = self.jax, self.jnp, self.a, self.mesh
    from diffusers_tpu.kernels import splash as csa
    local = functools.partial(csa.attention, scale=None, block_q=a.block_q,
                              block_kv=a.block_kv, block_kv_compute=a.block_kv,
                              use_fixed_m=True, qk_fp8=a.qk_fp8)
    ma = self.mesh_attention
    from jax.experimental.shard_map import shard_map

    def sdpa(q, k, v, kv_len=None):
      if kv_len is None:
        k = k - jnp.mean(k, axis=2, keepdims=True)
        with jax.default_matmul_precision("default"):
          return ma._on_mesh(local, mesh, q, k, v, True)
      # Bucketed shape: mean of k over real rows only, padded rows zeroed (so
      # the kernel's fixed-m bound is not raised); kv_len is a replicated scalar.
      keep = (jnp.arange(k.shape[2]) < kv_len)[None, None, :, None]
      mean = jnp.sum(jnp.where(keep, k, 0).astype(jnp.float32), axis=2,
                     keepdims=True) / kv_len
      k = jnp.where(keep, k - mean.astype(k.dtype), 0)
      qs, kvs, out_c = ma._partition_specs(mesh, q.shape, True)
      n = jnp.reshape(kv_len, (1,)).astype(jnp.int32)
      with jax.default_matmul_precision("default"):
        out = shard_map(lambda a, b, c, m: local(a, b, c, kv_len=m[0]),
                        mesh=mesh, in_specs=(qs, kvs, kvs, self.P()),
                        out_specs=qs, check_rep=False)(q, k, v, n)
      return jax.lax.with_sharding_constraint(out, out_c)
    self.sdpa = sdpa

  def _sparse_sdpa(self, seq, qlen, vs, rpf, window):
    """Same block table as `jax_pipeline`: padded q length x real kv length."""
    jax, jnp, a, mesh = self.jax, self.jnp, self.a, self.mesh
    from ...kernels import sparse_splash
    bq, bkv = min(a.block_q, qlen), min(a.block_kv, seq)
    reach = window * rpf
    table = []
    for i in range(-(-qlen // bq)):
      q0, q1 = i * bq, min((i + 1) * bq, qlen)
      table.append([j for j in range(-(-seq // bkv))
                    if q0 < vs or j * bkv < vs
                    or (j * bkv < q1 + reach and min((j + 1) * bkv, seq) > q0 - reach)])
    kept = sum(len(r) for r in table)
    full = len(table) * (-(-seq // bkv))
    if kept == full:
      return None, kept, full
    local = functools.partial(sparse_splash.attention, scale=None,
                              block_q=a.block_q, block_kv=a.block_kv,
                              block_kv_compute=a.block_kv, use_fixed_m=True,
                              qk_fp8=a.qk_fp8, kv_table=table)
    ma, dense = self.mesh_attention, self.sdpa

    def sdpa(q, k, v):
      if q.shape[2] != qlen or k.shape[2] != seq:
        return dense(q, k, v)          # short token-refiner sequence
      k = k - jnp.mean(k, axis=2, keepdims=True)
      with jax.default_matmul_precision("default"):
        return ma._on_mesh(local, mesh, q, k, v, True)
    return sdpa, kept, full

  def _sparse_table(self, qlen, seq_real, vs, rpf, window, seq_shape):
    """Block table (numpy): q blocks span the padded q length, kv blocks only the
    real length; fixed width."""
    a = self.a
    bq, bkv = min(a.block_q, qlen), min(a.block_kv, seq_shape)
    reach = window * rpf
    nq, W = -(-qlen // bq), -(-qlen // bkv)
    rows = []
    for i in range(nq):
      q0, q1 = i * bq, min((i + 1) * bq, qlen)
      rows.append([j for j in range(-(-seq_real // bkv))
                   if q0 < vs or j * bkv < vs
                   or (j * bkv < q1 + reach and min((j + 1) * bkv, seq_real) > q0 - reach)])
    tab = np.array([[r + [r[-1]] * (W - len(r)) for r in rows]], np.int32)
    cnt = np.array([[len(r) for r in rows]], np.int32)
    return tab, cnt, W

  def _call_sparse_dyn(self, qlen, W):
    """Sparse DiT call: block table, counts, and kv length are all arguments."""
    key = ("t2va", "sparse_dyn", qlen, W)
    if key in self.calls:
      return self.calls[key]
    jax, jnp, nnx, a, mesh = self.jax, self.jnp, self.nnx, self.a, self.mesh
    from ...kernels import sparse_splash
    from jax.experimental.shard_map import shard_map
    graphdef, dense, ma, P = self.graphdef, self.sdpa, self.mesh_attention, self.P

    @jax.jit
    def call(state, video, audio, text, t, row_step, token_tags, position_ids,
             video_idx, audio_idx, text_idx, seq_len, text_len, tab, cnt):
      def attn(q, k, v, kv_len=None):
        if q.shape[2] != qlen or kv_len is None:
          return dense(q, k, v, kv_len=kv_len)        # token refiner
        keep = (jnp.arange(k.shape[2]) < kv_len)[None, None, :, None]
        mean = jnp.sum(jnp.where(keep, k, 0).astype(jnp.float32), axis=2,
                       keepdims=True) / kv_len
        k = jnp.where(keep, k - mean.astype(k.dtype), 0)
        qs, kvs, out_c = ma._partition_specs(mesh, q.shape, True)
        n = jnp.reshape(kv_len, (1,)).astype(jnp.int32)
        local = lambda a_, b_, c_, m_, t_, c2_: sparse_splash.attention(
            a_, b_, c_, scale=None, block_q=a.block_q, block_kv=a.block_kv,
            block_kv_compute=a.block_kv, use_fixed_m=True, qk_fp8=a.qk_fp8,
            kv_table_dyn=(t_, c2_), kv_len=m_[0])
        with jax.default_matmul_precision("default"):
          out = shard_map(local, mesh=mesh,
                          in_specs=(qs, kvs, kvs, P(), P(), P()),
                          out_specs=qs, check_rep=False)(q, k, v, n, tab, cnt)
        return jax.lax.with_sharding_constraint(out, out_c)
      model = nnx.merge(graphdef, state)
      return model(video, audio, text, t, row_step, token_tags, position_ids,
                   video_idx, audio_idx, text_idx, sdpa=attn, segments=None,
                   seq_len=seq_len, text_len=text_len)
    self.calls[key] = call
    return call

  def _ensure_dit(self, kind):
    """Put the t2va / ref2va DiT on chip (reloaded only on workflow switch)."""
    if self.dit_kind == kind:
      return
    jax, nnx, dit = self.jax, self.nnx, self.dit
    t0 = time.perf_counter()
    if self.dit_kind is not None:
      # Drop only the DiT weights and its compiled calls. Do not call
      # `jax.clear_caches()`: that also evicts the decoder / text / encoder
      # programs (~18 s extra decode on the next request). The DiT calls must be
      # dropped, since keeping both workflows' programs holds their HBM and the
      # first ref2va request hits RESOURCE_EXHAUSTED; cost is 3-8 s extra on the
      # first request after a switch (reloaded from the compile cache).
      del self.state
      self.calls.clear()
    self.transformer = nnx.eval_shape(
        lambda: self.Transformer(**dit.CONFIG_FULL, dtype=self.dtype,
                                 rngs=nnx.Rngs(0)))
    adapter = self.a.pika_ref2va if kind == "ref2va" else self.a.pika_fl2va
    lora, self.pika = None, None
    if adapter:
      lora, self.pika = dit.load_lora(adapter)
    dit.load_transformer(self.transformer, os.path.join(
        self.snapshot, "transformer_ref" if kind == "ref2va" else "transformer"),
        mesh=self.mesh, lora=lora)
    del lora
    if self.a.fp8:
      dit.quantize_transformer_fp8(self.transformer, mesh=self.mesh)
    self.graphdef, self.state = nnx.split(self.transformer)
    # Copy the time_embedder weights needed by the skip criterion to host, then
    # drop the module itself: it still references the on-chip arrays, which
    # would otherwise keep HBM pinned even if `self.state` is moved to host.
    te = self.transformer.time_embedder
    self.te_w = [np.asarray(x, np.float64) for x in (
        te.linear_1.weight[...], te.linear_1.bias[...],
        te.linear_2.weight[...], te.linear_2.bias[...])]
    self.transformer = None
    self.dit_kind = kind
    print(f"[server] DiT {kind} on chip {time.perf_counter() - t0:.0f}s", flush=True)

  def _call(self, attn, key):
    """Compiled DiT call; layout arrays are arguments, not constants."""
    if key in self.calls:
      return self.calls[key]
    jax, nnx = self.jax, self.nnx
    graphdef = self.graphdef

    @jax.jit
    def call(state, video, audio, text, t, row_step, token_tags, position_ids,
             video_idx, audio_idx, text_idx, seq_len, text_len):
      model = nnx.merge(graphdef, state)
      return model(video, audio, text, t, row_step, token_tags, position_ids,
                   video_idx, audio_idx, text_idx, sdpa=attn, segments=None,
                   seq_len=seq_len, text_len=text_len)
    self.calls[key] = call
    return call

  def _load_decoder(self):
    import glob
    from ...safetensors_io import safe_open
    jax, jnp, nnx, jvae, mesh = self.jax, self.jnp, self.nnx, self.jvae, self.mesh
    dec_dtype = self.dtype
    decoder = nnx.eval_shape(
        lambda: jvae.ViTDecoder(**jvae.CONFIG, dtype=dec_dtype, rngs=nnx.Rngs(0)))
    post_q = jvae.PostQuantConv(jvae.CONFIG["in_channels"], dtype=dec_dtype,
                                rngs=nnx.Rngs(0))
    params = {}
    for shard in sorted(glob.glob(os.path.join(self.snapshot, "vae", "*.safetensors"))):
      with safe_open(shard, framework="flax") as reader:
        for k in reader.keys():
          if k.startswith("decoder."):
            params[k] = reader.get_tensor(k)
          elif k.startswith("post_quant_conv."):
            leaf = getattr(post_q, k.split(".")[-1])
            leaf.value = jnp.asarray(reader.get_tensor(k)).astype(
                leaf.value.dtype).reshape(leaf.value.shape)
    jvae.load_decoder(decoder, params, mesh=mesh)    # tensor parallel so it fits beside the DiT
    self.dec_graph, self.dec_state = nnx.split(decoder)
    pq_graph, pq_state = nnx.split(post_q)
    dec_graph, sdpa = self.dec_graph, self.sdpa
    NS, P = self.NS, self.P

    @functools.partial(jax.jit, out_shardings=NS(mesh, P()))
    def decode(state, z, pq=pq_state):
      with jax.default_matmul_precision("highest"):
        z = nnx.merge(pq_graph, pq)(z)
        return nnx.merge(dec_graph, state)(z, sdpa=sdpa)
    self._decode = decode
    self.lat_mean = jnp.asarray(self.vae_cfg["latents_mean"], jnp.float32).reshape(1, -1, 1, 1, 1)
    self.lat_std = jnp.asarray(self.vae_cfg["latents_std"], jnp.float32).reshape(1, -1, 1, 1, 1)
    self.audio_dec = None
    if not self.a.no_audio:
      from diffusers_tpu.models.minimax_h3.audio_decode import AudioDecoder
      _hbm = (mesh.devices.flat[0].memory_stats() or {}).get("bytes_limit", 0) / 2**30
      _audio_cpu = mesh.devices.size == 1 and (0 < _hbm < 50)
      self.audio_dec = AudioDecoder(
          self.snapshot, on_cpu=_audio_cpu, devices=list(mesh.devices.flat))
      # Compile once for the default length; otherwise the first request
      # pays about 64 s for it.
      from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
          align_num_frames, audio_latent_num_frames)
      n = audio_latent_num_frames(align_num_frames(int(self.a.num_frames), 17, 5))
      if not _audio_cpu:
        self.audio_dec.decode(np.zeros((2 * n, self.audio_dec.channels), np.float32))

  # ---------------------------------------------------------------------
  def _encode_text(self, prompt, reference_image):
    """Move the text (+ vision) tower from host, run, delete. Returns
    (embeds, text tags, host-side prep)."""
    jax, jnp, nnx = self.jax, self.jnp, self.nnx
    from . import jax_ref2va as r2v
    # The ref2va text program (7183 tokens + vision tower) can exceed the HBM
    # left beside a resident DiT; most of it is the vision tower's chunked score
    # matrix. Shrinking/sharding the chunk avoids moving the DiT (which costs
    # ~12 s per request). `DT_SERVER_PARK=1` parks the DiT on host instead.
    parked = (os.environ.get("DT_SERVER_PARK") == "1" and reference_image is not None
              and getattr(self, "state", None) is not None)
    if parked:
      t_park = time.perf_counter()
      self.state = jax.tree.map(
          lambda v: jax.device_put(v, self.NS(self.mesh, v.sharding.spec,
                                              memory_kind="pinned_host")),
          self.state)
      jax.block_until_ready(self.state)
      self._park_s = time.perf_counter() - t_park
    t_h2d = time.perf_counter()
    resident = self.a.text_resident
    if resident and getattr(self, "_enc_dev", None) is not None:
      enc = self._enc_dev
    else:
      enc_g, enc_host = nnx.split(self.encoder)
      enc = nnx.merge(enc_g, self._to_device(enc_host))
      jax.block_until_ready(nnx.state(enc))
      if resident:
        # --text-resident: the towers stay on the chips after the first
        # request (480p leaves room for them beside the DiT and decoder).
        self._enc_dev = enc
    t_h2d = time.perf_counter() - t_h2d
    with jax.set_mesh(self.mesh):
      if reference_image is not None:
        if resident and getattr(self, "_vis_dev", None) is not None:
          vis = self._vis_dev
        else:
          vis_g, vis_host = nnx.split(self.vision)
          vis = nnx.merge(vis_g, self._to_device(vis_host))
          if resident:
            self._vis_dev = vis
        t_prep = time.perf_counter()
        pres = r2v.presentation(self.snapshot, prompt, reference_image)
        pos3 = r2v.text_positions(self.snapshot, pres["ids"], pres["grid_thw"],
                                  pres["mm_token_type_ids"])
        t_prep = time.perf_counter() - t_prep
        t_fwd = time.perf_counter()
        # As in t2va: pad the causal text tower's tail (positions continue),
        # then slice the output back.
        n_real = int(pres["ids"].shape[0])
        b = self.a.text_bucket
        pres_in, pos_in = pres, pos3
        if b:
          pad_to = -(-n_real // b) * b
          pad_id = self.tokenizer.pad_token_id or 0
          pres_in = dict(pres, ids=np.concatenate(
              [pres["ids"], np.full(pad_to - n_real, pad_id, pres["ids"].dtype)]))
          tail = pos3[:, -1:] + np.arange(1, pad_to - n_real + 1)[None, :]
          pos_in = np.concatenate([pos3, tail], axis=1)
        emb = r2v.encode_text_with_vision(enc, vis, pres_in, pos_in, self.cfg_text,
                                          self.dtype, TEXT_ENCODER_LAYER, jit=True)
        emb = jax.block_until_ready(emb)[:, :n_real]
        if not resident:
          r2v.release_text_jit()   # the towers go back to host after this
        del vis
        print(f"  [server] ref2va text: text tower to chip {t_h2d:.1f}s, host prep {t_prep:.1f}s, "
              f"vision+text tower {time.perf_counter() - t_fwd:.1f}s", flush=True)
        tags, extra = pres["tags"], pres
      else:
        from ..qwen3_vl import jax_text as qwen_text
        ids = np.asarray(self.tokenizer(prompt, return_tensors="np")["input_ids"], np.int32)
        n_real = ids.shape[1]
        # The text tower is causal, so tail padding cannot affect real tokens:
        # pad the input to the bucket and slice the output back, so prompts in
        # the same bucket share one compilation (avoids a ~9 s recompile).
        b = self.a.text_bucket
        if b:
          pad_to = -(-n_real // b) * b
          pad_id = self.tokenizer.pad_token_id or 0
          ids = np.concatenate(
              [ids, np.full((1, pad_to - n_real), pad_id, np.int32)], axis=1)
        pos = np.broadcast_to(np.arange(ids.shape[1])[None, :], (3, ids.shape[1]))
        cos, sin = qwen_text.mrope_cos_sin(
            pos, head_dim=self.cfg_text["head_dim"],
            rope_theta=self.cfg_text["rope_theta"],
            mrope_section=self.cfg_text["mrope_section"])
        if not hasattr(self, "_t2v_text_jit"):
          self._t2v_text_jit = nnx.jit(
              lambda m, i, c, s: m(i, c, s, stop_at_layer=TEXT_ENCODER_LAYER))
        emb = jax.block_until_ready(self._t2v_text_jit(
            enc, jnp.asarray(ids), jnp.asarray(cos, self.dtype),
            jnp.asarray(sin, self.dtype)))[:, :n_real]
        tags, extra = np.ones(n_real, np.int64), None
    del enc
    if parked:
      t_back = time.perf_counter()
      self.state = self._to_device(self.state)
      jax.block_until_ready(self.state)
      print(f"  [server] DiT parked on host {self._park_s:.1f}s, moved back "
            f"{time.perf_counter() - t_back:.1f}s (to make room for the ref2va text program)", flush=True)
    return emb, tags, extra

  def _encode_reference(self, image):
    jax = self.jax
    from . import jax_ref2va as r2v
    dev = self.NS(self.mesh, self.P())
    es = jax.device_put(self._es_host, dev)
    qs = jax.device_put(self._qs_host, dev)
    if os.environ.get("DT_ENC_JIT", "1") == "1":
      # The whole tiled encode as one program, tiles in groups (as in
      # `jax_pipeline`): 2.8 -> 0.5 s for a 3648x2048 reference.
      from . import jax_pipeline as jp
      os.environ.setdefault("DT_ENC_GROUP", "5")
      jp._ENC_ONE[0] = self._encode_one
      cond = r2v.encode_condition(jp._ENC_W_FN, image,
                                  self.vae_cfg["latents_mean"],
                                  self.vae_cfg["latents_std"], mesh=self.mesh,
                                  weights=(es, qs))
    else:
      enc1 = self._encode_one
      cond = r2v.encode_condition(lambda t: enc1(es, qs, t), image,
                                  self.vae_cfg["latents_mean"],
                                  self.vae_cfg["latents_std"], mesh=self.mesh)
    del es, qs
    return cond

  def generate(self, req):
    """One request. Returns (uint8 frames (F,H,W,3), timing dict)."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS, align_num_frames, audio_latent_num_frames,
        video_latent_num_frames)
    from . import jax_ref2va as r2v
    jax, jnp, a, dit = self.jax, self.jnp, self.a, self.dit
    T = {}
    t_all = time.perf_counter()
    ref_path = req.get("reference")
    kind = "ref2va" if ref_path else "t2va"
    lossy = bool(req.get("lossy"))
    seed = int(req.get("seed", 42))
    num_frames = align_num_frames(int(req.get("num_frames", a.num_frames)), 17, 5)
    self._ensure_dit(kind)     # no-op unless the workflow changes
    T["dit_swap"] = time.perf_counter() - t_all

    # Text (+ vision)
    t = time.perf_counter()
    image = r2v.load_reference(ref_path) if ref_path else None
    # DT_SERVER_TEXT_PROFILE=<dir>: profile the text stage of the 2nd request
    # (the 1st includes compilation).
    _tp = os.environ.get("DT_SERVER_TEXT_PROFILE")
    _n = getattr(self, "_req_n", 0); self._req_n = _n + 1
    if _tp and _n == 1:
      jax.profiler.start_trace(_tp)
    emb, text_tags, pres = self._encode_text(req["prompt"], image)
    if _tp and _n == 1:
      jax.profiler.stop_trace()
    T["text"] = time.perf_counter() - t

    # Reference condition + layout + noise
    t = time.perf_counter()
    tgen = torch.Generator().manual_seed(seed)
    # Per-request resolution (defaults to CLI args); each new resolution compiles once.
    height = int(req.get("height", a.height))
    width = int(req.get("width", a.width))
    if height % 32 or width % 32:
      raise ValueError(f"resolution must be a multiple of 32, got {width}x{height}")
    latent_h, latent_w = height // 16, width // 16
    latent_frames = video_latent_num_frames(num_frames, 17, 5)
    n_ref, cond_rows = 0, None
    if image is not None:
      cond = self._encode_reference(image)
      if self.pika is not None:
        # Pika's anchor noise: seed + 7919, frame-major [B, F, C, H, W].
        c_ = cond.shape
        noise = torch.randn((c_[0], c_[2], c_[1], c_[3], c_[4]),
                            generator=torch.Generator().manual_seed(seed + 7919),
                            dtype=torch.float32).permute(0, 2, 1, 3, 4)
      else:
        noise = randn_tensor(tuple(cond.shape), generator=tgen, dtype=torch.float32)
      noised = r2v.NOISE_AUG * torch.from_numpy(cond) + (1 - r2v.NOISE_AUG) * noise
      cond_rows = patchify_video_latents(noised, (1, 2, 2)).numpy().astype(np.float32)
      layout = r2v.layout(pres["tags"], pres["references"], r2v.condition_shape(image),
                          num_frames, height, width)
      n_ref = int(layout[5])
    else:
      layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
          text_token_tags=torch.as_tensor(text_tags), num_latent_frames=latent_frames,
          latent_height=latent_h, latent_width=latent_w,
          num_audio_latents=audio_latent_num_frames(num_frames), patch_size=(1, 2, 2),
          audio_channels=MINIMAX_H3_AUDIO_CHANNELS, audio_tag=2, video_tag=0)
    position_ids = jnp.asarray(np.asarray(layout[0])).astype(jnp.float32)
    token_tags = jnp.asarray(np.asarray(layout[1])).astype(jnp.int32)
    video_idx, audio_idx, text_idx = (jnp.asarray(np.asarray(x)).astype(jnp.int32)
                                      for x in layout[2:5])
    seq = int(position_ids.shape[0])
    # Text bucketing: padded text rows go at the end of the packed sequence and
    # attention masks to the real length, so output is unchanged while prompts
    # in the same bucket share a shape and a compilation.
    n_txt = int(emb.shape[1])   # not `T`: that is the timing dict
    bucket = a.text_bucket
    Tb = -(-n_txt // bucket) * bucket if bucket else n_txt
    extra = Tb - n_txt
    seq_len_arg, text_len_arg = None, None
    if bucket:
      if extra:
        emb = jnp.concatenate(
            [emb, jnp.zeros((1, extra, emb.shape[2]), emb.dtype)], axis=1)
        position_ids = jnp.concatenate(
            [position_ids, jnp.repeat(position_ids[-1:], extra, axis=0)])
        token_tags = jnp.concatenate([token_tags, jnp.ones((extra,), jnp.int32)])
        text_idx = jnp.concatenate(
            [text_idx, jnp.arange(seq, seq + extra, dtype=jnp.int32)])
      seq_len_arg, text_len_arg = jnp.int32(seq), jnp.int32(n_txt)
    seq_real, seq = seq, seq + extra
    T_info = (n_txt, Tb)
    v = randn_tensor((1, dit.CONFIG_FULL["in_channels"], latent_frames, latent_h, latent_w),
                     generator=tgen, dtype=torch.float32)
    au = randn_tensor((int(audio_idx.shape[0]), dit.CONFIG_FULL["audio_in_channels"]),
                      generator=tgen, dtype=torch.float32)
    video = jnp.asarray(patchify_video_latents(v, (1, 2, 2)).numpy())[None]
    audio = jnp.asarray(au.numpy())[None]
    pika_eps = None
    if self.pika is not None:
      # Pika's draws, as in `jax_pipeline --pika-adapter`: video then audio
      # from seed in its external layouts, re-noising eps from seed + 1.
      C_ = dit.CONFIG_FULL["in_channels"]
      vs_ = (1, latent_frames, C_, latent_h, latent_w)
      T_ = int(audio_idx.shape[0]) // 2
      ach = dit.CONFIG_FULL["audio_in_channels"]
      vrows = lambda x: jnp.asarray(patchify_video_latents(
          x.permute(0, 2, 1, 3, 4), (1, 2, 2)).reshape(1, -1, 4 * C_).numpy())
      arows = lambda x: jnp.asarray(x.view(1, T_, 2, ach).permute(0, 2, 1, 3)
                                    .reshape(1, 2 * T_, ach).numpy())
      g = torch.Generator().manual_seed(seed)
      video = vrows(torch.randn(vs_, generator=g, dtype=torch.float32))
      audio = arows(torch.randn((1, T_, 2 * ach), generator=g, dtype=torch.float32))
      g = torch.Generator().manual_seed(seed + 1)
      pika_eps = []
      for _ in range(len(self.pika["sampling"]["knots"]) - 2):
        ev = vrows(torch.randn(vs_, generator=g, dtype=torch.float32))
        ea = arows(torch.randn((1, T_, 2 * ach), generator=g, dtype=torch.float32))
        pika_eps.append((ev, ea))
    if n_ref:
      video = jnp.concatenate([jnp.asarray(cond_rows)[None], video], axis=1)
    row_step = jnp.zeros((seq,), jnp.int32).at[audio_idx].set(1)
    T_layout = (seq_len_arg, text_len_arg)
    if n_ref:
      row_step = row_step.at[video_idx[:n_ref]].set(2)
    T["prep"] = time.perf_counter() - t

    # Compiled calls: one dense; for lossy requests, one sparse per layout
    steps = int(req.get("steps", a.steps))
    dense_steps = int(req.get("sparse_dense_steps", a.sparse_dense_steps))
    sigmas = sigma_schedule(steps, a.shift)
    a_sigmas = sigma_schedule(steps, a.audio_shift)
    if self.pika is not None:
      sigmas, a_sigmas = _step_sigmas(a, self.pika)
    timesteps = 1.0 - sigmas[:-1]
    a_timesteps = 1.0 - a_sigmas[:-1]
    call = self._call(self.sdpa, (kind, "dense"))
    call_sparse, sparse_args = None, ()
    if lossy and a.sparse_window:
      # Only generated video rows (ref2va reference frames / audio come first and
      # act as global rows), as in jax_pipeline.
      vi = np.asarray(video_idx)[n_ref:]
      times = np.asarray(position_ids)[vi, 0]
      vs, rpf = int(vi.min()), int((times == times[0]).sum())
      qlen = (-(-seq // dit.PAD_SEQUENCE_TO) * dit.PAD_SEQUENCE_TO
              if dit.PAD_SEQUENCE_TO else seq)
      if bucket:
        # The block table depends on the video start (= text length), so pass it
        # as a runtime array: q over the padded length, kv up to the real length,
        # fixed width, so one program serves the bucket (bit-identical to a
        # static table).
        tab, cnt, W = self._sparse_table(qlen, seq_real, vs, rpf, a.sparse_window, seq)
        call_sparse = self._call_sparse_dyn(qlen, W)
        sparse_args = (jnp.asarray(tab), jnp.asarray(cnt))
      else:
        lay = (seq, qlen, vs, rpf, a.sparse_window)
        if (kind, lay) not in self.calls:
          sp, kept, full = self._sparse_sdpa(*lay)
          if sp is not None:
            self._call(sp, (kind, lay))
          else:
            self.calls[(kind, lay)] = None
        call_sparse = self.calls[(kind, lay)]

    # Step-skip cache (same criterion as jax_pipeline)
    W = self.te_w
    half = dit.CONFIG_FULL["freq_dim"] // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)

    def indicator(lv):
      u = np.unique(np.asarray(lv, np.float64))
      e = u[:, None] * freqs[None, :]
      e = np.concatenate([np.cos(e), np.sin(e)], -1)
      h = e @ W[0].T + W[1]
      h = h / (1 + np.exp(-h))
      return (h @ W[2].T + W[3]).mean(0)

    cache = ({"last_mod": None, "last_out": None, "drift": 0.0, "skipped": 0}
             if lossy and self.pika is None else None)

    def pika_step(x, v_, i, j):
      """Re-noising step: x0 = x + sigma v, then fresh noise at the next level."""
      sg, sn = (sigmas, a_sigmas)[j][i], (sigmas, a_sigmas)[j][i + 1]
      x0 = x.astype(jnp.float32) + float(sg) * v_.astype(jnp.float32)
      if i >= len(pika_eps):
        return x0
      return (1.0 - float(sn)) * x0 + float(sn) * pika_eps[i][j]
    args_layout = (row_step, token_tags, position_ids, video_idx, audio_idx,
                   text_idx) + T_layout

    def levels(i):
      lv = [timesteps[i], a_timesteps[i]]
      if n_ref:
        lv.append(max(float(timesteps[i]), r2v.NOISE_AUG))
      return jnp.asarray(lv, jnp.float32)

    t = time.perf_counter()
    with jax.set_mesh(self.mesh):
      for i in range(len(timesteps)):
        tt = levels(i)
        skip = False
        if cache is not None:
          cur = indicator(np.asarray(tt))
          if cache["last_mod"] is not None and cache["last_out"] is not None:
            cache["drift"] += float(np.abs(cur - cache["last_mod"]).mean()
                                    / (np.abs(cache["last_mod"]).mean() + 1e-9))
            if cache["drift"] < (a.cache_threshold if a.cache_threshold is not None
                                 else 0.20 if kind == "t2va" else 0.15):
              skip = True
            else:
              cache["drift"] = 0.0
          if not skip:
            cache["last_mod"] = cur
        if skip:
          dv, da = cache["last_out"]
          cache["skipped"] += 1
        else:
          if call_sparse is not None and i >= dense_steps:
            dv, da = call_sparse(self.state, video, audio, emb, tt, *args_layout,
                                 *sparse_args)
          else:
            dv, da = call(self.state, video, audio, emb, tt, *args_layout)
          if cache is not None:
            cache["last_out"] = (dv, da)
        if pika_eps is not None:
          gen = pika_step(video[:, n_ref:], dv[:, n_ref:], i, 0)
          video = jnp.concatenate([video[:, :n_ref], gen], axis=1) if n_ref else gen
          audio = pika_step(audio, da, i, 1)
        elif n_ref:
          gen = scheduler_step(video[:, n_ref:], dv[:, n_ref:], float(timesteps[i]),
                               float(sigmas[i]), float(sigmas[i + 1]))
          video = jnp.concatenate([video[:, :n_ref], gen], axis=1)
        else:
          video = scheduler_step(video, dv, float(timesteps[i]), float(sigmas[i]),
                                 float(sigmas[i + 1]))
        if pika_eps is None:
          audio = scheduler_step(audio, da, float(a_timesteps[i]), float(a_sigmas[i]),
                                 float(a_sigmas[i + 1]))
        if os.environ.get("DT_SERVER_STEP_TIMES") == "1":
          jax.block_until_ready(video)
          print(f"    step {i}: {time.perf_counter() - t:.1f}s", flush=True)
      jax.block_until_ready(video)
      T["denoise"] = time.perf_counter() - t
      T["skipped"] = cache["skipped"] if cache else 0

      # Decode
      t = time.perf_counter()
      # Audio first, so it queues with the video decode (saves about 0.3 s).
      pending_audio = (None if self.audio_dec is None else
                       self.audio_dec.start(np.asarray(jax.device_get(audio))[0]))
      vid = video[:, n_ref:]
      lat = vid.reshape(1, latent_frames, latent_h // 2, latent_w // 2,
                        dit.CONFIG_FULL["in_channels"], 2, 2)
      lat = jnp.transpose(lat, (0, 4, 1, 2, 5, 3, 6)).reshape(
          1, dit.CONFIG_FULL["in_channels"], latent_frames, latent_h, latent_w)
      lat = lat.astype(jnp.float32) * self.lat_std + self.lat_mean
      frames = self._decode_latents(lat)
    T["decode"] = time.perf_counter() - t
    T["wave"] = None
    if self.audio_dec is not None:
      t = time.perf_counter()
      T["wave"] = self.audio_dec.finish(pending_audio)
      T["audio"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t_all
    T["text_tokens"] = T_info
    return frames, T

  def _decode_latents(self, latents):
    """`jax_pipeline`'s temporal chunking + tiled decode + crossfade, K chunks at a time."""
    jax, jnp, jvae, a = self.jax, self.jnp, self.jvae, self.a
    clip_length, token_drop, ratio = 17, 3, 4
    tokens_chunk = -(-clip_length // ratio)
    pre_pad = (-clip_length) % ratio
    tok_overlap = (-token_drop) % tokens_chunk
    frame_overlap = max(tok_overlap * ratio - pre_pad, 0)
    chunk_frames = tokens_chunk * ratio
    dec_state, decode = self.dec_state, self._decode

    def decode_tiled(z):
      return jvae.tiled_decode(
          lambda t_: decode(dec_state, t_), z, tile_px=256,
          overlap_px=a.vae_tile_overlap_px,
          batch=28 if self.mesh.devices.size > 1 else 4)

    def blend(x, y, extent):
      extent = min(x.shape[2], y.shape[2], extent)
      if extent <= 0:
        return y
      w = jnp.arange(extent, dtype=y.dtype).reshape(1, 1, extent, 1, 1) / extent
      head = x[:, :, -extent:] * (1.0 - w) + y[:, :, :extent] * w
      return head if extent == y.shape[2] else jnp.concatenate([head, y[:, :, extent:]], axis=2)

    frames_lat = latents.shape[2]
    num_tokens = frames_lat + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk - int(token_drop > 0)
    z = latents.astype(self.dtype)
    if pad_tokens:
      z = jnp.concatenate([z, jnp.repeat(z[:, :, -1:], pad_tokens, axis=2)], axis=2)
    _base_cap = 8 if latents.shape[3] * latents.shape[4] <= 30 * 52 else 4
    K = decode_chunk_batch(frames_lat, a.decode_chunk_batch,
                           1 if self.mesh.devices.size == 1 else
                           a.decode_chunk_cap or
                           max(1, (_base_cap * self.mesh.devices.size) // 8))
    decoded, pieces, carry = {}, [], None
    for i in range(num_chunks):
      if i not in decoded:
        group = list(range(i, min(i + K, num_chunks)))
        zs = [z[:, :, g * tokens_chunk:g * tokens_chunk + tokens_chunk + tok_overlap]
              for g in group]
        zs += [zs[-1]] * (K - len(zs))
        out_g = decode_tiled(jnp.concatenate(zs, axis=0))
        for k, g in enumerate(group):
          decoded[g] = out_g[k:k + 1]
      out = decoded.pop(i)
      for j in range(int(token_drop > 0) + 1):
        f = j * chunk_frames
        piece = out[:, :, f:f + chunk_frames][:, :, pre_pad:]
        if j == 0:
          pieces.append(piece if carry is None else blend(carry, piece, frame_overlap))
        else:
          carry = piece
    if carry is not None:
      pieces.append(carry)
    frames = jnp.concatenate(pieces, axis=2)
    if pad_tokens:
      intra_tail = clip_length % ratio
      before = z.shape[2] - pad_tokens
      pad_frames = sum(intra_tail if intra_tail and (before + k) % tokens_chunk == 0
                       else ratio for k in range(pad_tokens))
      frames = frames[:, :, :-pad_frames]
    mean = jnp.asarray((0.485, 0.456, 0.406), jnp.float32).reshape(1, -1, 1, 1, 1)
    std = jnp.asarray((0.229, 0.224, 0.225), jnp.float32).reshape(1, -1, 1, 1, 1)
    frames = jnp.clip(frames.astype(jnp.float32) * std + mean, 0.0, 1.0)
    return np.asarray(jnp.round(jnp.transpose(frames[0], (1, 2, 3, 0)) * 255.0), np.uint8)


def main():
  args = parse_args()
  server = Server(args)
  os.makedirs(args.out, exist_ok=True)
  if args.warmup:
    t0 = time.perf_counter()
    with open(args.warmup) as f:
      warm = [json.loads(l) for l in f if l.strip()]
    for req in warm:
      # 3 steps, sparse from step 1: step 0's inputs come from host and later
      # steps' from the scheduler, with different shardings, so these are
      # separate programs that all need compiling here. With only 2 steps the
      # deduplicated sigma grid yields a single call and the real request still
      # compiles (~40 s each).
      req = dict(req, steps=3, sparse_dense_steps=1)
      _, T = server.generate(req)
      print(f"[warmup] {req.get('tag', '')}: {T['total']:.1f}s", flush=True)
    print(f"[warmup] {len(warm)} shapes, {time.perf_counter() - t0:.0f}s", flush=True)
  with open(args.requests) as f:
    reqs = [json.loads(l) for l in f if l.strip()]
  for n, req in enumerate(reqs):
    frames, T = server.generate(req)
    tag = req.get("tag", f"req{n}")
    np.save(os.path.join(args.out, f"{tag}_frames.npy"), frames)
    if T["wave"] is not None:
      np.save(os.path.join(args.out, f"{tag}_audio.npy"), T["wave"])
    print(f"[req {n}] {tag}: total {T['total']:.1f}s = dit_swap {T['dit_swap']:.1f} + "
          f"text {T['text']:.1f} + prep {T['prep']:.1f} + denoise {T['denoise']:.1f} "
          f"(skipped {T['skipped']}) + decode {T['decode']:.1f}"
          + (f" + audio {T['audio']:.2f}" if T["wave"] is not None else "") +
          f"  {tuple(frames.shape)}  "
          f"text_tokens {T['text_tokens'][0]}->{T['text_tokens'][1]}",
          flush=True)


if __name__ == "__main__":
  main()
