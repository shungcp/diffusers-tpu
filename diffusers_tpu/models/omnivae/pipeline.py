"""OmniVAE text -> video + audio (``t2av_recon_distill_avclip``) with native JAX models.

    prompt -> T2AV template + chat template -> Qwen3.5-0.8B (layer 23 of 24)
    50 steps (FlowMatchEuler, shift 3), both towers + bridges each step:
      dual CFG (default, the release evaluation): [empty / bridge off,
        empty / bridge on, prompt / bridge on],
        v = v00 + s_mod (v0B - v00) + s_text (vTB - v0B), all scales 4
      simple CFG: [prompt, empty], v = pos + g (pos - neg)
    the model output is negated, then x <- x + (sigma_next - sigma) out
    -> Wan2.2 VAE decode (video, 24 fps) and DAC decode (audio, 48 kHz)

The model timestep is ``1 - sigma`` (``(1000 - timestep) / 1000``). The
scheduler's shift is 3 for both modalities: the official code means to set 5
(video) / 3 (audio) but the assignment never reaches the scheduler, so the
released numbers come from 3.

    PYTHONPATH=. python -m diffusers_tpu.models.omnivae.pipeline --prompt "..."
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.dac import jax_dac
from diffusers_tpu.models.omnivae import jax_dit as jd
from diffusers_tpu.models.omnivae import text as text_mod
from diffusers_tpu.models.wan_vae import jax_wan22_vae

REPO = "OpenMOSS-Team/OmniVAE"
EXPERIMENT = "t2av_recon_distill_avclip"
SAMPLE_RATE = 48000
HOP = 960


def release_root(root=None):
  if root:
    return root
  if os.environ.get("OMNIVAE_RELEASE_ROOT"):
    return os.environ["OMNIVAE_RELEASE_ROOT"]
  from diffusers_tpu.hub import local_snapshot

  return local_snapshot(REPO)


def paths(root, experiment=EXPERIMENT):
  m = os.path.join(root, "models")
  return {
      "dit": os.path.join(m, "dit", "t2av", experiment),
      "video_vae": os.path.join(m, "vae", "audio_video", "recon_distill_avclip", "state_dict.pt"),
      "audio_vae": os.path.join(m, "vae", "audio_only", "recon_distill_avclip_ft_decoder", "state_dict.pt"),
      "text": os.path.join(m, "text_encoder", "Qwen3.5-0.8B-Base"),
  }


def sigmas(dit_dir, steps):
  """The release scheduler's sigmas (``FlowMatchEulerDiscreteScheduler``, shift 3), last 0."""
  from diffusers import FlowMatchEulerDiscreteScheduler

  s = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(dit_dir, "scheduler"))
  s.set_timesteps(steps)
  return s.sigmas.numpy().astype(np.float32)


def latent_shape(frames, height, width, fps, audio_seconds=None):
  t = 1 + (frames - 1) // 4
  seconds = frames / fps if audio_seconds is None else audio_seconds
  ta = max(1, int(round(seconds * SAMPLE_RATE / HOP)))
  return (t, height // 16, width // 16), ta


def make_attention(mesh=None):
  """[B, S, H, D] attention with key mask [B, Sk] (valid keys a prefix): the
  splash kernel per sample (``kv_len``) for long sequences on TPU, dense otherwise."""
  if jax.default_backend() == "cpu":
    return jd.dense_attention
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  from diffusers_tpu.kernels import block_sizes, splash

  parts = 1 if mesh is None else mesh.shape["tensor"]

  def attention(q, k, v, mask=None, layout=None):
    sq, sk = q.shape[1], k.shape[1]
    if max(sq, sk) < 1024:
      return jd.dense_attention(q, k, v, mask)
    # Short sides (the bridges' 252 audio rows) padded to a multiple of 128;
    # padded keys masked with ``kv_len``, padded queries dropped.
    pq, pk = -sq % 128, -sk % 128
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))  # [B, H, S, D]
    if pq:
      q = jnp.pad(q, ((0, 0), (0, 0), (0, pq), (0, 0)))
    if layout is not None:
      # [x_0, cap, x_1, cap, ...] (see ``jax_dit.unify``): keys become every x
      # row and one caption copy, valid keys a prefix again.
      m, lc = layout
      sk = parts * m + lc
      pk = -sk % 128
      n = parts * m + mask[:, m:m + lc].sum(-1).astype(jnp.int32)
    elif mask is None:
      n = jnp.full((q.shape[0],), sk, jnp.int32)
    else:
      n = mask.sum(-1).astype(jnp.int32)
      k = jnp.where(mask[:, None, :, None], k, 0)
    masked = mask is not None or pk > 0
    split = parts > 1 and (sq + pq) % (parts * 8) == 0 and sq + pq >= 1024
    if split:   # this chip's query rows against every key
      blocks = block_sizes.resolve_block_sizes((sq + pq) // parts, sk + pk, q.shape[-1], {}, "residual",
                                               block_sizes.detect_spec())
    else:
      blocks = block_sizes.resolve_block_sizes(sq + pq, sk + pk, q.shape[-1], {}, "residual",
                                               block_sizes.detect_spec())

    def local(q, k, v, n):
      if layout is not None:
        m, lc = layout
        def order(t):
          b, h = t.shape[:2]
          t = t.reshape(b, h, parts, m + lc, t.shape[-1])
          return jnp.concatenate([t[:, :, :, :m].reshape(b, h, parts * m, -1), t[:, :, 0, m:]], 2)
        k, v = order(k), order(v)
        valid = (jnp.arange(k.shape[2])[None] < n[:, None])[:, None, :, None]
        k = jnp.where(valid, k, 0)
      if pk:  # after the key gather: pad locally
        k, v = (jnp.pad(t, ((0, 0), (0, 0), (0, pk), (0, 0))) for t in (k, v))
      if not masked:
        return splash.attention(q, k, v, scale=None, **blocks)
      outs = [splash.attention(q[i:i + 1], k[i:i + 1], v[i:i + 1], scale=None, kv_len=n[i], **blocks)
              for i in range(q.shape[0])]
      return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)

    if split:
      rows = P(None, None, "tensor", None)
      out = shard_map(local, mesh=mesh, in_specs=(rows, P(), P(), P()), out_specs=rows,
                      check_rep=False)(q, k, v, n)
    elif parts > 1:  # short query side: every chip computes all rows
      out = shard_map(local, mesh=mesh, in_specs=(P(),) * 4, out_specs=P(), check_rep=False)(q, k, v, n)
    else:
      out = local(q, k, v, n)
    return out[:, :, :sq].transpose(0, 2, 1, 3)

  def kv_rows(q, k, v, mask=None):
    """Few queries (replicated) over row-split keys: per-chip partial
    softmax statistics merged with pmax / psum (exact; float32)."""
    del mask  # the bridge's keys are all valid

    def local(q, k, v):
      s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
      m = jax.lax.pmax(jnp.max(s, -1, keepdims=True), "tensor")
      e = jnp.exp(s - m)
      l = jax.lax.psum(jnp.sum(e, -1, keepdims=True), "tensor")
      o = jax.lax.psum(jnp.einsum("bhqk,bkhd->bqhd", e.astype(v.dtype), v,
                                  preferred_element_type=jnp.float32), "tensor")
      return (o / jnp.transpose(l, (0, 2, 1, 3))).astype(v.dtype)

    rows = P(None, "tensor", None, None)
    return shard_map(local, mesh=mesh, in_specs=(P(), rows, rows), out_specs=P(), check_rep=False)(q, k, v)

  if parts > 1:
    from jax.sharding import NamedSharding

    attention.kv_rows = lambda q, k, v, mask=None: (kv_rows(q, k, v) if k.shape[1] % parts == 0
                                                    else attention(q, k, v, mask))

    attention.parts = parts
    spec = NamedSharding(mesh, P(None, "tensor", None))
    attention.rows = lambda x: (jax.lax.with_sharding_constraint(x, spec)
                                if x.shape[1] % parts == 0 else x)
  return attention


class Generator:

  def __init__(self, root=None, mesh=None, env=None, frames=121, height=256, width=256, fps=24.0,
               steps=50, cfg_mode="dual", guidance=4.0, audio_seconds=None):
    import torch
    import torchax

    root = release_root(root)
    self.p = paths(root)
    self.frames, self.height, self.width, self.fps = frames, height, width, fps
    self.grid, self.ta = latent_shape(frames, height, width, fps, audio_seconds)
    self.cfg_mode, self.guidance, self.steps = cfg_mode, float(guidance), steps
    self.mesh = mesh
    self.text = text_mod.TextEncoder(self.p["text"], dtype=torch.bfloat16,
                                     env=env or torchax.default_env(), mesh=mesh)
    vp, ap, bp, self.vc, self.ac = jd.load(self.p["dit"], jnp.bfloat16)
    put = self._put
    # Rotate-half RoPE: interleaved pairs make (..., 64, 2) layouts on TPU.
    self.vp, self.ap, self.bp = (jax.tree.map(put, jd.permute_rope(d)) for d in (vp, ap, bp))
    self.sig = sigmas(self.p["dit"], steps)
    self.vae, _, self._decode_video = jax_wan22_vae.make(self.p["video_vae"], mesh, dtype=jnp.bfloat16)
    self.dac, self._decode_audio, _ = jax_dac.make_decoder(self.p["audio_vae"], mesh=mesh)
    f, h, w = self.grid
    self.vrope = jd.bridge_rope(self.ac, jd.video_time_indices(f, h, w, self.ta), 128)
    self.arope = jd.bridge_rope(self.ac, np.arange(self.ta), 128)
    self._attention = make_attention(mesh)
    self._steps = {}
    self.times = {}

  def _put(self, a):
    if self.mesh is None:
      return jnp.asarray(a)
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return jax.device_put(a, NamedSharding(self.mesh, P()))

  # -- conditioning ---------------------------------------------------------

  def conditioning(self, prompt, seed=0, template=None):
    """(cap [n, Lc, 1024], mask [n, Lc], cap lengths) in CFG slot order."""
    pos = self.text.encode(text_mod.chat(self.text.tokenizer, text_mod.wrap_prompt(prompt, template, seed)))
    neg = self.text.encode(text_mod.chat(self.text.tokenizer, ""))
    embs = [pos, neg] if self.cfg_mode == "simple" else [neg, neg, pos]
    lc = max(e.shape[0] for e in embs)
    lc = -(-lc // 64) * 64   # one program per 64-token bucket
    cap = np.zeros((len(embs), lc, embs[0].shape[1]), np.float32)
    mask = np.zeros((len(embs), lc), bool)
    for i, e in enumerate(embs):
      cap[i, :len(e)], mask[i, :len(e)] = e, True
    return cap, mask, [len(e) for e in embs]

  def _step_fn(self, lc):
    if lc in self._steps:
      return self._steps[lc]
    vc, ac, attention = self.vc, self.ac, self._attention
    vrope, arope = self.vrope, self.arope
    dual = self.cfg_mode == "dual"
    g = self.guidance
    n = 3 if dual else 2
    gate = jnp.asarray([0.0, 1.0, 1.0]) if dual else None

    def step(vp, ap, bp, lat_v, lat_a, cap, mask, v_pos, v_cap_pos, a_pos, a_cap_pos, s, s_next):
      t = jnp.full((n,), 1.0, jnp.float32) - s
      pv, pa = jd.joint_forward(vp, ap, bp, vc, ac, jnp.repeat(lat_v, n, 0), jnp.repeat(lat_a, n, 0),
                                t, cap, mask, v_pos, v_cap_pos, a_pos, a_cap_pos, vrope, arope,
                                gate, attention, half=True)
      if dual:
        cfg = lambda p: p[0:1] + g * (p[1:2] - p[0:1]) + g * (p[2:3] - p[1:2])
      else:
        cfg = lambda p: p[0:1] + g * (p[0:1] - p[1:2])
      dt = s_next - s
      return lat_v - dt * cfg(pv), lat_a - dt * cfg(pa)

    self._steps[lc] = jax.jit(step)
    return self._steps[lc]

  def noise(self, seed):
    """Video then audio latents from one CPU ``torch.Generator`` (the official
    order; on GPU it is a CUDA generator, so values differ across devices)."""
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    f, h, w = self.grid
    v = torch.randn((1, 48, f, h, w), generator=gen, dtype=torch.float32)
    a = torch.randn((1, 128, self.ta), generator=gen, dtype=torch.float32)
    return self._put(v.numpy()), self._put(a.numpy())

  def denoise(self, cond, lat_v, lat_a):
    cap, mask, lens = cond
    step = self._step_fn(cap.shape[1])
    v_pos, v_cap_pos = jd.positions(lens, cap.shape[1], self.grid)
    a_pos, a_cap_pos = jd.positions(lens, cap.shape[1], (self.ta, 1, 1))
    args = [self._put(a) for a in (cap, mask, v_pos, v_cap_pos, a_pos, a_cap_pos)]
    for i in range(len(self.sig) - 1):
      lat_v, lat_a = step(self.vp, self.ap, self.bp, lat_v, lat_a, *args,
                          self._put(self.sig[i]), self._put(self.sig[i + 1]))
    return lat_v, lat_a

  def decode(self, lat_v, lat_a):
    """(frames [F, H, W, 3] uint8, audio [samples] float32 at 48 kHz)."""
    video = np.asarray(self._decode_video(self.vae, self._put(lat_v)))[0]
    frames = np.round((np.clip(video, -1, 1) + 1.0) * 127.5).astype(np.uint8).transpose(1, 2, 3, 0)
    wave = np.clip(np.asarray(self._decode_audio(self.dac, self._put(lat_a)))[0, 0], -1, 1)
    want = int(round(self.frames * SAMPLE_RATE / self.fps))
    wave = np.pad(wave, (0, max(0, want - len(wave))))[:want]
    return frames, wave

  def __call__(self, prompt, seed=42, template=None):
    t0 = time.perf_counter()
    cond = self.conditioning(prompt, seed, template)
    lat_v, lat_a = self.noise(seed)
    t1 = time.perf_counter()
    lat_v, lat_a = jax.block_until_ready(self.denoise(cond, lat_v, lat_a))
    t2 = time.perf_counter()
    out = self.decode(lat_v, lat_a)
    t3 = time.perf_counter()
    self.times = {"text": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return out


def save_mp4(path, frames, audio, fps=24.0, rate=SAMPLE_RATE):
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
  from frames_to_mp4 import write

  write(frames, path, fps=int(round(fps)), wave=np.stack([audio, audio]), rate=rate)


def main():
  import argparse

  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--root", default=None, help="release bundle (default $OMNIVAE_RELEASE_ROOT or the HF cache)")
  p.add_argument("--prompt", required=True)
  p.add_argument("--frames", type=int, default=121)
  p.add_argument("--height", type=int, default=256)
  p.add_argument("--width", type=int, default=256)
  p.add_argument("--fps", type=float, default=24.0)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--cfg-mode", choices=("dual", "simple"), default="dual")
  p.add_argument("--guidance", type=float, default=4.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--template", type=int, default=None, help="T2AV template index (default: drawn with the seed)")
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--chips", type=int, default=0, help="0 = all; the video rows split over them")
  p.add_argument("--output", default="omnivae.mp4")
  p.add_argument("--compile-cache-dir", default="~/.cache/omnivae_jax")
  args = p.parse_args()
  print(args, flush=True)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh

  chips = args.chips or len(jax.devices())
  mesh = Mesh(mesh_utils.create_device_mesh((1, chips), jax.devices()[:chips]), ("cfg", "tensor"))
  print(f"mesh: {chips} chip(s), video rows split over them", flush=True)
  t0 = time.perf_counter()
  gen = Generator(args.root, mesh, frames=args.frames, height=args.height, width=args.width, fps=args.fps,
                  steps=args.steps, cfg_mode=args.cfg_mode, guidance=args.guidance)
  print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
  for i in range(args.runs):
    t0 = time.perf_counter()
    frames, audio = gen(args.prompt, args.seed, args.template)
    split = ", ".join(f"{k} {v:.2f}" for k, v in gen.times.items())
    print(f"run {i}: {time.perf_counter() - t0:.2f}s ({split})" + (" (includes compilation)" if i == 0 else ""),
          flush=True)
  save_mp4(args.output, frames, audio, args.fps)
  print(f"wrote {args.output} ({frames.shape}, {len(audio) / SAMPLE_RATE:.2f} s audio)")


if __name__ == "__main__":
  main()
