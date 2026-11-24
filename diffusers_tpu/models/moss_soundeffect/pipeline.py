"""MOSS-SoundEffect-v2 text-to-audio on TPU.

    prompt -> Qwen3-1.7B (torchax, 512 tokens) -> 100 steps of the audio DiT
    (native JAX, one program a step: [positive, negative] + CFG + Euler)
    -> DAC decode (native JAX) -> 48 kHz mono, cropped to the requested length

The official pipeline always denoises a 30 s latent (1500 frames at 50 Hz)
and crops the waveform, so every request has the same shapes and compiles
once. Defaults match ``MossSoundEffectPipeline.__call__``: 100 steps, CFG 4,
shift 5, seed 0, empty negative prompt, the ``" duration: <s>s"`` suffix.

Chips: a ``('cfg', second)`` mesh. The CFG branches go on ``cfg``; the second
axis splits the 1500-row sequence by default (each chip holds the whole
1.3B model and gathers keys and values), which beats splitting the heads at
this length. ``--replicas`` runs independent copies for throughput.

    PYTHONPATH=. python -m diffusers_tpu.models.moss_soundeffect.pipeline \\
        --prompt "Rain on a tin roof" --seconds 10 --runs 2
"""
import argparse
import os
import time
import wave

import jax
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import torchax

from diffusers_tpu.hub import local_snapshot
from diffusers_tpu.models.dac import jax_dac
from diffusers_tpu.models.moss_image.torchax_pipeline import _move_module, _shard, torchax_paused
from diffusers_tpu.models.moss_soundeffect import jax_dit
from diffusers_tpu.models.moss_soundeffect import text as text_mod

REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
SAMPLE_RATE = 48000
FULL_SECONDS = 30
HOP = 960

_ALL = ("cfg", "tensor")
TEXT_SHARDINGS = {
    r".*layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight": (_ALL, None),
    r".*layers\.\d+\.mlp\.down_proj\.weight": (None, _ALL),
    r".*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight": (_ALL, None),
    r".*layers\.\d+\.self_attn\.o_proj\.weight": (None, _ALL),
}


class TextCore(torch.nn.Module):
  """Qwen3's decoder layers and final norm (= ``hidden_states[-1]``), causal."""

  def __init__(self, model):
    super().__init__()
    self.layers = model.layers
    self.norm = model.norm

  def forward(self, embeds, cos, sin):
    h = embeds
    for layer in self.layers:
      h = layer(h, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=(cos, sin))
    return self.norm(h)


def sigmas(steps, shift):
  """The official ``FlowMatchScheduler(extra_one_step=True, sigma_min=0)``."""
  s = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]
  s = shift * s / (1 + (shift - 1) * s)
  return s.astype(np.float32)


class Generator:

  def __init__(self, repo, mesh, env, text_dtype=torch.bfloat16, dit_dtype=None,
               seq_parallel=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    self.mesh = mesh
    self.env = env
    snap = local_snapshot(repo)
    self.tokenizer = AutoTokenizer.from_pretrained(f"{snap}/tokenizer")
    import contextlib

    with contextlib.suppress(RuntimeError):  # loading is plain torch; a replica built before left torchax on
      torchax.disable_globally()
    lm = AutoModelForCausalLM.from_pretrained(
        f"{snap}/text_encoder", dtype=text_dtype, attn_implementation="sdpa").eval()
    self._lm = lm.model
    torchax.enable_globally()
    env._mesh = mesh
    env.default_device_or_sharding = NamedSharding(mesh, P())
    _move_module(lm.model.layers, env)
    _move_module(lm.model.norm, env)
    core = torchax.compile(TextCore(lm.model))
    core.params = _shard(core.params, TEXT_SHARDINGS, mesh)
    core.buffers = _shard(core.buffers, TEXT_SHARDINGS, mesh)
    self._text_core = core
    self.dit_params, self.config, self._step = jax_dit.make_step(
        f"{snap}/transformer", mesh, dit_dtype or jax.numpy.bfloat16, seq_parallel)
    self.dac_params, self._decode, _ = jax_dac.make_decoder(f"{snap}/vae/vae_128d_48k.pth", mesh=mesh)
    head_dim = self.config["dim"] // self.config["num_heads"]
    frames = SAMPLE_RATE * FULL_SECONDS // HOP
    self.frames = frames
    cos, sin = jax_dit.rope_table(frames, head_dim)
    self._cos, self._sin = self._put(cos, P()), self._put(sin, P())
    self._prompts = {}
    self.times = {}

  def _put(self, array, spec):
    return jax.device_put(array, NamedSharding(self.mesh, spec))

  def _put_torch(self, t):
    """A host torch tensor onto this generator's mesh, replicated (bf16 kept)."""
    t = t.detach()
    if t.dtype == torch.bfloat16:
      array = self._put(t.view(torch.int16).numpy(), P())
      return self.env.j2t_iso(jax.lax.bitcast_convert_type(array, jax.numpy.bfloat16))
    return self.env.j2t_iso(self._put(t.numpy(), P()))

  def encode(self, prompts):
    """[B, 512, 2048] bf16 on the mesh, padded rows zeroed; cached per prompt."""
    # One prompt at a time: a fixed [1, 512] shape, so a cached negative prompt
    # never leaves a batch of a new size to compile (8 s a replica).
    for prompt in prompts:
      if prompt in self._prompts:
        continue
      ids, lengths = text_mod.tokenize(self.tokenizer, [prompt])
      with torchax_paused():
        embeds = self._lm.embed_tokens(torch.from_numpy(ids).long())
        pos = torch.arange(ids.shape[1]).view(1, -1).expand(ids.shape[0], -1)
        cos, sin = self._lm.rotary_emb(embeds, pos)
      hidden = self._text_core(*(self._put_torch(t) for t in (embeds, cos, sin)))
      hidden = self.env.t2j_iso(hidden)
      keep = (np.arange(ids.shape[1])[None, :] < lengths[:, None])[..., None]
      self._prompts[prompt] = jax.numpy.where(self._put(keep, P()), hidden, 0)
    return jax.numpy.concatenate([self._prompts[p] for p in prompts])

  def denoise(self, *args, **kwargs):
    """The final latents, dispatched asynchronously (not waited on)."""
    steps = self.denoise_steps(*args, **kwargs)
    while True:
      try:
        next(steps)
      except StopIteration as done:
        return done.value

  def denoise_steps(self, prompt, negative_prompt="", steps=100, cfg_scale=4.0, shift=5.0, seed=0):
    """Yields after dispatching each step; returns the final latents."""
    cfg = cfg_scale != 1.0
    context = self.encode([prompt, negative_prompt] if cfg else [prompt])
    if cfg:
      context = jax.device_put(context, NamedSharding(self.mesh, P("cfg")))
    sig = sigmas(steps, shift)
    sig_next = np.append(sig[1:], 0.0).astype(np.float32)
    sinusoids = jax_dit.time_sinusoid(sig * 1000.0)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn((1, 128, self.frames), generator=gen, dtype=torch.float32)
    # The official noise is cast to the pipeline dtype (bf16) before the first step.
    latents = self._put(noise.to(torch.bfloat16).float().numpy(), P())
    guidance = self._put(np.float32(cfg_scale), P())
    for i in range(steps):
      latents = self._step(self.dit_params, latents, self._put(sinusoids[i:i + 1], P()),
                           context, self._cos, self._sin,
                           self._put(np.float32(sig_next[i] - sig[i]), P()), guidance)
      yield
    return latents

  def warmup_jax(self, cfg_scale=4.0):
    """Compile the step and the decoder from their shapes alone (a zero text
    context): plain JAX, so it can run in a thread beside other replicas'
    (torchax's dispatch mode is per thread, so the text encoder cannot)."""
    batch = 2 if cfg_scale != 1.0 else 1
    context = self._put(np.zeros((batch, text_mod.TEXT_LEN, 2048), jax.numpy.bfloat16),
                        P("cfg") if batch > 1 else P())
    latents = self._put(np.zeros((1, 128, self.frames), np.float32), P())
    latents = self._step(self.dit_params, latents, self._put(jax_dit.time_sinusoid([999.0]), P()),
                         context, self._cos, self._sin, self._put(np.float32(-0.01), P()),
                         self._put(np.float32(cfg_scale), P()))
    jax.block_until_ready(self._decode(self.dac_params, latents))

  def decode(self, latents, seconds):
    """float32 waveform [samples], cropped to ``seconds``."""
    return np.asarray(self._decode(self.dac_params, latents))[0, 0][:int(SAMPLE_RATE * seconds)]

  def __call__(self, prompt, seconds=10.0, negative_prompt="", steps=100, cfg_scale=4.0,
               shift=5.0, seed=0, append_duration=True):
    """float32 waveform [samples] at 48 kHz."""
    seconds = round(float(seconds), 1)
    if not 0 < seconds <= FULL_SECONDS:
      raise ValueError(f"seconds must be in (0, {FULL_SECONDS}]")
    formatted = text_mod.format_prompt(prompt, seconds, append_duration)
    t0 = time.perf_counter()
    jax.block_until_ready(self.encode([formatted, negative_prompt]))
    t1 = time.perf_counter()
    latents = jax.block_until_ready(
        self.denoise(formatted, negative_prompt, steps, cfg_scale, shift, seed))
    t2 = time.perf_counter()
    audio = self.decode(latents, seconds)
    t3 = time.perf_counter()
    self.times = {"text": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return audio


def run_concurrently(gens, requests, **common):
  """One request per generator (disjoint chips), steps dispatched round-robin so
  the replicas run side by side; nothing waits until the decodes."""
  iters, seconds = [], []
  for gen, req in zip(gens, requests):
    sec = round(float(req.get("seconds", common.get("seconds", 10.0))), 1)
    formatted = text_mod.format_prompt(req["prompt"], sec)
    neg = common.get("negative_prompt", "")
    gen.encode([formatted, neg])
    seconds.append(sec)
    iters.append(gen.denoise_steps(formatted, neg, common.get("steps", 100),
                                   common.get("cfg_scale", 4.0), common.get("shift", 5.0),
                                   req.get("seed", 0)))
  results = [None] * len(iters)
  live = list(range(len(iters)))
  while live:
    for i in list(live):
      try:
        next(iters[i])
      except StopIteration as done:
        results[i] = done.value
        live.remove(i)
  return [gen.decode(lat, sec) for gen, lat, sec in zip(gens, results, seconds)]


def serve(gens, args):
  """``--requests``: one JSON object per line, ``{"prompt", "seconds"?, "seed"?, "tag"?}``,
  served ``len(gens)`` at a time (one per replica). Each request encodes its prompt."""
  import json

  os.makedirs(args.out, exist_ok=True)
  with open(args.requests) as fh:
    requests = [json.loads(line) for line in fh if line.strip()]
  common = dict(seconds=args.seconds, negative_prompt=args.negative_prompt, steps=args.steps,
                cfg_scale=args.cfg_scale, shift=args.shift)
  n = len(gens)
  # Compile every replica at once: each has its own programs (its own chips),
  # and compiling them one after another took 270 s for four.
  import threading

  t0 = time.perf_counter()
  warm = [threading.Thread(target=g.warmup_jax, args=(args.cfg_scale,)) for g in gens]
  for th in warm:
    th.start()
  for g in gens:  # the text encoders, in this thread (torchax), meanwhile
    jax.block_until_ready(g.encode(["warm-up", args.negative_prompt]))
  for th in warm:
    th.join()
  print(f"warm-up (compile, {n} replicas in parallel): {time.perf_counter() - t0:.1f}s", flush=True)
  batches = []
  t_all = time.perf_counter()
  for start in range(0, len(requests), n):
    group = requests[start:start + n]
    for gen in gens:
      gen._prompts = {k: v for k, v in gen._prompts.items() if k == args.negative_prompt}
    t0 = time.perf_counter()
    audios = run_concurrently(gens[:len(group)], group, **common)
    dt = time.perf_counter() - t0
    batches.append((dt, len(group)))
    for index, (req, audio) in enumerate(zip(group, audios)):
      save_wav(os.path.join(args.out, f"{req.get('tag', f'{start + index:04d}')}.wav"), audio)
    print(f"requests {start}-{start + len(group) - 1}: {dt:.2f}s for {len(group)}", flush=True)
  warm = batches[1:] or batches
  per = sum(d for d, _ in warm) / sum(k for _, k in warm)
  print(f"{len(requests)} requests in {time.perf_counter() - t_all:.1f}s; warm {per:.2f}s per clip "
        f"({sum(d for d, _ in warm) / len(warm):.2f}s per batch of {n})", flush=True)


def save_wav(path, audio, sample_rate=SAMPLE_RATE):
  pcm = (np.clip(audio, -1, 1) * 32767).round().astype("<i2")
  with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    w.writeframes(pcm.tobytes())


def bench(gen, reps=10, profile_dir=None):
  """Device time of one step and of the decode, synchronised."""
  context = jax.device_put(gen.encode(["a dog barking duration: 10.0s", ""]),
                           NamedSharding(gen.mesh, P("cfg")))
  latents = gen._put(np.random.randn(1, 128, gen.frames).astype(np.float32), P())
  args = (gen._put(jax_dit.time_sinusoid([500.0]), P()), context, gen._cos, gen._sin,
          gen._put(np.float32(-0.01), P()), gen._put(np.float32(4.0), P()))

  def step():
    return gen._step(gen.dit_params, latents, *args)

  def timed(label, fn, n):
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n):
      out = fn()
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n
    print(f"{label:34s} {dt * 1e3:8.2f} ms", flush=True)
    return dt

  s = timed("one CFG step (DiT x2 + CFG + Euler)", step, reps)
  timed("DAC decode (30 s)", lambda: gen._decode(gen.dac_params, latents), 3)
  def encode_new():
    gen._prompts.clear()
    return gen.encode(["a dog barking duration: 10.0s", ""])

  timed("text encode (new prompt + negative)", encode_new, 3)
  print(f"=> 100 steps = {100 * s:.2f} s")
  if profile_dir:
    jax.profiler.start_trace(profile_dir, create_perfetto_trace=True)
    for _ in range(3):
      out = step()
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    print(f"trace: {profile_dir}")


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", default=REPO)
  p.add_argument("--prompt", default="The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.")
  p.add_argument("--negative-prompt", default="")
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--steps", type=int, default=100)
  p.add_argument("--cfg-scale", type=float, default=4.0)
  p.add_argument("--shift", type=float, default=5.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--chips", type=int, default=0,
                 help="chips per replica; 0 = all chips / --replicas")
  p.add_argument("--replicas", type=int, default=1,
                 help="model copies on disjoint chips, serving --requests side by side")
  p.add_argument("--cfg", type=int, default=0, help="CFG batch axis; 0 = 2 if the chip count is even")
  p.add_argument("--split", choices=("seq", "heads"), default="seq",
                 help="what the second mesh axis splits: the 1500-row sequence (keys and "
                      "values gathered; faster here) or the heads (tensor parallel)")
  p.add_argument("--requests", default=None, help="JSONL of requests to serve, models resident")
  p.add_argument("--out", default="out", help="directory for --requests outputs")
  p.add_argument("--bench", action="store_true")
  p.add_argument("--profile", default=None)
  p.add_argument("--output", default="moss_soundeffect.wav")
  p.add_argument("--compile-cache-dir", default="~/.cache/moss_soundeffect_jax")
  args = p.parse_args()
  print(args)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  devices = jax.devices()
  per = args.chips or len(devices) // args.replicas
  if per * args.replicas > len(devices):
    raise SystemExit(f"{args.replicas} replicas x {per} chips > {len(devices)} chips")
  gens = []
  t0 = time.perf_counter()
  for r in range(args.replicas):
    group = devices[r * per:(r + 1) * per]
    cfg = args.cfg or (2 if per % 2 == 0 and args.cfg_scale != 1.0 else 1)
    mesh = Mesh(mesh_utils.create_device_mesh((cfg, per // cfg), group), ("cfg", "tensor"))
    print(f"replica {r}: chips {[d.id for d in group]}, mesh cfg={cfg} x {args.split}={per // cfg}, "
          f"{group[0].device_kind}", flush=True)
    gens.append(Generator(args.repo, mesh, torchax.default_env(),
                          seq_parallel=args.split == "seq" and per // cfg > 1))
  gen = gens[0]
  print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
  if args.requests:
    serve(gens, args)
    return
  if args.bench:
    bench(gen, profile_dir=args.profile)
    return
  audio = None
  for i in range(args.runs):
    t0 = time.perf_counter()
    audio = gen(args.prompt, args.seconds, args.negative_prompt, args.steps, args.cfg_scale,
                args.shift, args.seed)
    split = ", ".join(f"{k} {v:.2f}" for k, v in gen.times.items())
    print(f"run {i}: {time.perf_counter() - t0:.2f}s ({split})"
          + (" (includes compilation)" if i == 0 else ""), flush=True)
  save_wav(args.output, audio)
  print(f"wrote {args.output} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
  main()
