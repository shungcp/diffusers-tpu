"""VideoGym-LTX-2.3 (LTX-2.3 image + text to video with audio) on TPU.

    prompt -> Gemma-3-12B (torchax; 49 hidden states, left-padded to 1024)
           -> text connectors (JAX)                      [negative, positive]
    image  -> H.264 CRF 33 re-compression (as diffusers / ltx-pipelines)
           -> LTX-2 video VAE encode (torchax) = latent frame 0, kept fixed
    30 Euler steps, each one DiT forward (JAX) over 4 rows:
        [negative, positive, positive with STG at block 28, positive without audio<->video attention]
        x0 = x - sigma v;  g = x0_pos + (cfg - 1)(x0_pos - x0_neg) + stg (x0_pos - x0_stg)
                                       + (modality - 1)(x0_pos - x0_iso)
        video: cfg 3, stg 1, modality 1; audio (diffusers' defaults): cfg 7, stg 1, modality 3
        g <- rescale(g, x0_pos, 0.7);  v = (x - g) / sigma
    -> LTX-2 video VAE decode (JAX, rows over the chips), audio VAE decode + vocoder (torchax)

Everything follows diffusers' ``LTX2ImageToVideoPipeline`` (0.42) with the
VideoGym model card's settings; the noise comes from a CPU
``torch.Generator(seed)`` (video, then audio), so on GPU (CUDA generator)
the same seed gives different noise.

    PYTHONPATH=. python -m diffusers_tpu.models.ltx2.pipeline --image puzzle.png --prompt "..."
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.ltx2 import jax_connectors as jc
from diffusers_tpu.models.ltx2 import jax_dit as jd
from diffusers_tpu.models.ltx2 import jax_sp
from diffusers_tpu.models.ltx2 import jax_vae
from diffusers_tpu.models.ltx2 import text as text_mod

REPO = "OpenMOSS-Team/VideoGym-LTX-2.3"


def _snapshot(repo):
  if os.path.isdir(repo):
    return repo
  from diffusers_tpu.hub import local_snapshot

  return local_snapshot(repo)


def _extract(module_fn):
  """A torch module's forward as (weights, pure JAX function) via torchax."""
  import torchax

  return torchax.extract_jax(module_fn)


def _vae_modules(vae, audio_vae, vocoder):
  import torch

  class Enc(torch.nn.Module):

    def __init__(self, v):
      super().__init__()
      self.v = v

    def forward(self, x):
      return self.v.encode(x).latent_dist.mode()

  class Audio(torch.nn.Module):

    def __init__(self, a, voc):
      super().__init__()
      self.a, self.voc = a, voc

    def forward(self, z):
      return self.voc(self.a.decode(z, return_dict=False)[0])

  return Enc(vae), Audio(audio_vae, vocoder)


def _is_high_hbm():
  """True on high-HBM chips (v7x: 95 GiB/device), False on v6e (32 GiB)."""
  try:
    d0 = jax.devices()[0]
    limit = d0.memory_stats().get("bytes_limit", 0)
    return limit >= 64 * (1 << 30) or "7x" in getattr(d0, "device_kind", "").lower()
  except Exception:  # noqa: BLE001
    return False


def _default_resident(num_layers, cfg_groups=1):
  """Keep all blocks whole on high-HBM chips (v7x: 95 GiB -> 48); on v6e (32 GiB) 20 on one group,
  16 with two CFG groups (each chip then holds 1/4 instead of 1/8 of the gathered blocks)."""
  if _is_high_hbm():
    return num_layers
  return min(16 if cfg_groups > 1 else 20, num_layers)


class Generator:

  def __init__(self, repo, mesh, height=512, width=768, frames=129, fps=24.0, steps=30,
               guidance=3.0, stg=1.0, rescale=0.7, stg_blocks=(28,), crf=None, resident=None, modality=1.0,
               audio_guidance=7.0, audio_stg=1.0, audio_modality=3.0, fp8=False, qk_fp8=False,
               load_dit=True, load_text=True):
    import json
    import torch
    import torchax
    from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    snap = _snapshot(repo)
    self.snap, self.mesh = snap, mesh
    self.height, self.width, self.frames, self.fps = height, width, frames, fps
    self.steps, self.guidance, self.stg, self.rescale = steps, float(guidance), float(stg), float(rescale)
    self.stg_blocks = tuple(stg_blocks)
    # diffusers' defaults for the audio terms (the model card sets only the video ones): audio CFG 7 and
    # modality guidance 3, which needs a 4th row with the audio<->video attentions off.
    self.modality, self.audio_guidance = float(modality), float(audio_guidance)
    self.audio_stg, self.audio_modality = float(audio_stg), float(audio_modality)
    self.rows = 4 if max(self.modality, self.audio_modality) > 1.0 else 3
    self.crf = crf
    self.times = {}
    self.keep_steps = False  # tests: keep every step's packed video latents in .record
    if len(mesh.axis_names) == 2:
      self.cfg_axis, self.axis = mesh.axis_names[0], mesh.axis_names[1]
      self.every = tuple(mesh.axis_names)
      assert self.rows % mesh.shape[self.cfg_axis] == 0, (
          f"rows={self.rows} must be divisible by cfg mesh axis size {mesh.shape[self.cfg_axis]}"
      )
    else:
      self.cfg_axis, self.axis = None, mesh.axis_names[0]
      self.every = self.axis
    self.env = torchax.default_env()

    if load_text:
      self.text = text_mod.TextEncoder(os.path.join(snap, "text_encoder"), os.path.join(snap, "tokenizer"),
                                       dtype=torch.bfloat16, env=self.env, mesh=mesh)
      cp, self.ccfg = jc.load(os.path.join(snap, "connectors"), jnp.bfloat16)
      self.cparams = jc.place(cp, mesh, self.every)
      self._connect = jax.jit(lambda p, h, m: jc.forward(p, self.ccfg, h, m))
    else:
      self.text = None
      self.cparams = None
      self._connect = None

    if load_dit:
      dp, self.cfg = jd.load(os.path.join(snap, "transformer"), jnp.bfloat16)
      if resident is None or resident < 0:
        resident = _default_resident(self.cfg["num_layers"], mesh.shape[self.cfg_axis] if self.cfg_axis else 1)
      self.resident = min(int(resident), self.cfg["num_layers"])
      # Blocks kept whole on every chip (evenly spaced): no all-gather for them.
      res = (set(np.linspace(0, self.cfg["num_layers"] - 1, self.resident).round().astype(int).tolist())
             if self.resident else set())
      self.params = jax_sp.place(dp, mesh, self.axis, res, fp8=fp8)
      del dp
      self.blocks_fn = jax_sp.make_blocks_fn(mesh, self.cfg, self.params, self.axis, res, stg_blocks,
                                             cfg_axis=self.cfg_axis, qk_fp8=qk_fp8)
    else:
      with open(os.path.join(snap, "transformer", "config.json")) as f:
        self.cfg = json.load(f)
      self.resident = 0
      self.params = None
      self.blocks_fn = None

    vae = AutoencoderKLLTX2Video.from_pretrained(os.path.join(snap, "vae"), torch_dtype=torch.bfloat16).eval()
    avae = AutoencoderKLLTX2Audio.from_pretrained(os.path.join(snap, "audio_vae"), torch_dtype=torch.bfloat16).eval()
    voc = LTX2VocoderWithBWE.from_pretrained(os.path.join(snap, "vocoder"), torch_dtype=torch.bfloat16).eval()
    self.vae_mean = vae.latents_mean.float().numpy()
    self.vae_std = vae.latents_std.float().numpy()
    self.vae_scaling = float(vae.config.scaling_factor)
    self.audio_mean = avae.latents_mean.float().numpy()
    self.audio_std = avae.latents_std.float().numpy()
    self.sample_rate = int(voc.config.output_sampling_rate)
    mel_ratio = avae.mel_compression_ratio if hasattr(avae, "mel_compression_ratio") else 4
    self.latent_mel = avae.config.mel_bins // mel_ratio
    self.audio_channels = avae.config.latent_channels
    temporal = getattr(avae, "temporal_compression_ratio", 4)
    per_second = avae.config.sample_rate / avae.config.mel_hop_length / float(temporal)
    self.audio_frames = round(frames / fps * per_second)
    enc, aud = _vae_modules(vae, avae, voc)
    (self.w_enc, f_enc), (self.w_aud, f_aud) = (_extract(m) for m in (enc, aud))
    self.w_enc, self.w_aud = (jax.tree.map(self._put, w) for w in (self.w_enc, self.w_aud))
    self._encode = jax.jit(lambda w, x: f_enc(w, (x,)))
    # Native JAX decoder, rows split over all chips with on-TPU uint8 conversion (4x smaller PCIe D2H).
    self.w_dec, self._decode = jax_vae.make_decoder(os.path.join(snap, "vae"), mesh, self.every, to_uint8=True)
    self._audio = jax.jit(lambda w, z: f_aud(w, (z,)))

    self.lf, self.lh, self.lw = (frames - 1) // 8 + 1, height // 32, width // 32
    self.n_tokens = self.lf * self.lh * self.lw
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
    from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift

    c = sched.config
    mu = calculate_shift(self.n_tokens, c.get("base_image_seq_len", 1024), c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.95), c.get("max_shift", 2.05))
    sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    self.sigmas = sched.sigmas.float().numpy()
    self.timesteps = sched.timesteps.float().numpy()
    self.rope = jd.ropes(self.cfg, self.lf, self.lh, self.lw, fps, self.audio_frames) if load_dit else None
    idx = np.ones((self.n_tokens,), np.int32)
    idx[:self.lh * self.lw] = 0  # latent frame 0 is the image (t = 0)
    self.token_t_np = idx
    self.first_np = (idx == 0).astype(np.float32)[None, :, None]
    self.token_t = self._put(idx)
    self.first = self._put(self.first_np)
    self._step = jax.jit(self._step_fn) if load_dit else None

  def _put(self, a, spec=None):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return jax.device_put(a, NamedSharding(self.mesh, spec or P()))

  # -- one Euler step: DiT on [neg, pos, pos+STG], guidance in x0, update ------

  def _step_fn(self, params, x, a, ctx, actx, s, s_next):
    t = s * 1000.0
    rows = self.rows
    tv = jnp.stack([jnp.zeros((rows,)), jnp.full((rows,), t)], 1)
    tt = jnp.full((rows,), t)
    stg = jnp.asarray([False, False, True, False][:rows])
    iso = jnp.asarray([False, False, False, True][:rows])
    xin = jnp.repeat(x, rows, 0).astype(jnp.bfloat16)
    ain = jnp.repeat(a, rows, 0).astype(jnp.bfloat16)
    pv, pa = jd.forward(params, self.cfg, xin, ain, ctx, actx, tv, tt, tt, self.rope, stg=stg,
                        stg_blocks=self.stg_blocks, token_t=self.token_t, blocks_fn=self.blocks_fn,
                        isolate=iso if rows == 4 else None)

    def guide(sample, out, cfg, stg_scale, mod):
      x0 = sample - out.astype(jnp.float32) * s  # [rows, ...]
      neg, pos, pst = x0[0:1], x0[1:2], x0[2:3]
      g = pos + (cfg - 1.0) * (pos - neg) + stg_scale * (pos - pst)
      if rows == 4 and mod != 1.0:
        g = g + (mod - 1.0) * (pos - x0[3:4])
      if self.rescale > 0:
        std_p = jnp.std(pos, ddof=1, keepdims=True, axis=tuple(range(1, pos.ndim)))
        std_g = jnp.std(g, ddof=1, keepdims=True, axis=tuple(range(1, g.ndim)))
        g = self.rescale * (g * (std_p / std_g)) + (1 - self.rescale) * g
      return (sample - g) / s

    v = guide(x, pv, self.guidance, self.stg, self.modality)
    va = guide(a, pa, self.audio_guidance, self.audio_stg, self.audio_modality)
    x_new = x + (s_next - s) * v
    x_new = self.first * x + (1 - self.first) * x_new  # the image frame stays
    return x_new, a + (s_next - s) * va

  # -- conditioning ---------------------------------------------------------

  def conditioning(self, prompt, negative=""):
    hs, ms = [], []
    for p in (negative, prompt):
      h, m = self.text.encode(p)
      hs.append(h)
      ms.append(m)
    h = jnp.concatenate(hs, 0)
    m = self._put(np.concatenate(ms, 0))
    ctx, actx = self._connect(self.cparams, h, m)
    extra = self.rows - 2  # positive with STG (and positive with the modalities isolated)
    ctx = jnp.concatenate([ctx] + [ctx[1:2]] * extra, 0)
    actx = jnp.concatenate([actx] + [actx[1:2]] * extra, 0)
    return ctx, actx

  def image_latent(self, image):
    """PIL image -> normalized latent frame [1, 128, 1, h, w] (float32 numpy)."""
    import PIL.Image
    import torch
    from diffusers.pipelines.ltx2.utils import DEFAULT_IMAGE_CRF, apply_image_conditioning_crf
    from diffusers.video_processor import VideoProcessor

    crf = DEFAULT_IMAGE_CRF if self.crf is None else self.crf
    if crf:
      image = PIL.Image.fromarray(apply_image_conditioning_crf(np.array(image.convert("RGB")), crf))
    proc = VideoProcessor(vae_scale_factor=32, resample="bilinear")
    x = proc.preprocess(image, height=self.height, width=self.width).to(torch.bfloat16)  # [1, 3, H, W]
    x = self._put(jnp.asarray(x.float().numpy()[:, :, None], jnp.bfloat16))  # [1, 3, 1, H, W]
    z = np.asarray(self._encode(self.w_enc, x).astype(jnp.float32))
    return (z - self.vae_mean.reshape(1, -1, 1, 1, 1)) * self.vae_scaling / self.vae_std.reshape(1, -1, 1, 1, 1)

  def noise(self, seed, z0):
    """Initial packed latents, video then audio from one CPU generator (diffusers' order)."""
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    shape = (1, 128, self.lf, self.lh, self.lw)
    nv = torch.randn(shape, generator=gen, dtype=torch.float32).numpy()
    na = torch.randn((1, self.audio_channels, self.audio_frames, self.latent_mel), generator=gen,
                     dtype=torch.float32).numpy()
    lat = np.concatenate([z0, nv[:, :, 1:]], 2)  # frame 0 = image latent
    x = lat.reshape(1, 128, -1).transpose(0, 2, 1)  # pack [B, F*H*W, C]
    a = na.transpose(0, 2, 1, 3).reshape(1, self.audio_frames, -1)  # [B, L, C*M]
    return x.astype(np.float32), a.astype(np.float32)

  # -- generation -----------------------------------------------------------

  def denoise(self, ctx, actx, x, a):
    x, a = self._put(x), self._put(a)
    self.record = []
    for i in range(self.steps):
      x, a = self._step(self.params, x, a, ctx, actx, self._put(self.sigmas[i]), self._put(self.sigmas[i + 1]))
      if self.keep_steps:
        self.record.append((np.asarray(x), np.asarray(a)))
    return x, a

  def decode(self, x, a):
    lat = np.asarray(x).transpose(0, 2, 1).reshape(1, 128, self.lf, self.lh, self.lw)
    lat = lat * self.vae_std.reshape(1, -1, 1, 1, 1) / self.vae_scaling + self.vae_mean.reshape(1, -1, 1, 1, 1)
    frames = np.asarray(self._decode(self.w_dec, self._put(jnp.asarray(lat, jnp.bfloat16)))[0])  # [F, H, W, 3] uint8
    al = np.asarray(a) * self.audio_std + self.audio_mean
    al = al.reshape(1, self.audio_frames, self.audio_channels, self.latent_mel).transpose(0, 2, 1, 3)
    wave = np.asarray(self._audio(self.w_aud, self._put(jnp.asarray(al, jnp.bfloat16))).astype(jnp.float32))[0]
    return frames, wave

  def __call__(self, image, prompt, negative="", seed=10):
    t0 = time.perf_counter()
    ctx, actx = self.conditioning(prompt, negative)
    z0 = self.image_latent(image)
    x, a = self.noise(seed, z0)
    t1 = time.perf_counter()
    x, a = jax.block_until_ready(self.denoise(ctx, actx, x, a))
    t2 = time.perf_counter()
    out = self.decode(x, a)
    t3 = time.perf_counter()
    self.times = {"encode": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return out


def save_mp4(path, frames, wave, fps, rate):
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
  from frames_to_mp4 import write

  write(frames, path, fps=int(round(fps)), wave=wave, rate=rate)


def main():
  import argparse

  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from PIL import Image

  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", default=REPO)
  p.add_argument("--image", required=True)
  p.add_argument("--prompt", required=True)
  p.add_argument("--negative-prompt", default="")
  p.add_argument("--height", type=int, default=512)
  p.add_argument("--width", type=int, default=768)
  p.add_argument("--frames", type=int, default=129, help="8k + 1")
  p.add_argument("--fps", type=float, default=24.0)
  p.add_argument("--steps", type=int, default=30)
  p.add_argument("--guidance", type=float, default=3.0)
  p.add_argument("--stg", type=float, default=1.0)
  p.add_argument("--rescale", type=float, default=0.7)
  p.add_argument("--crf", type=int, default=None, help="image re-compression (default 33; 0 = off)")
  p.add_argument("--seed", type=int, default=10)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--resident", type=int, default=None,
                 help="DiT blocks kept whole on every chip (default: auto = 48 on v7x, 16 on a v6e-8)")
  p.add_argument("--cfg-parallel", type=int, default=None, choices=(1, 2, 4),
                 help="CFG-parallel groups over the 4 DiT rows (default: auto = 4 on v7x, 2 on a v6e-8)")
  p.add_argument("--fp8", action="store_true", help="Quantize large DiT block linears to v7x float8_e4m3fn (W8A8)")
  p.add_argument("--qk-fp8", action="store_true", help="Cast Q and K to float8_e4m3fn inside Pallas splash attention")
  p.add_argument("--encode-only", default="", help="Stage 0 only: save conditioning + initial noise to .npz and exit (skips DiT)")
  p.add_argument("--decode-latents", default="", help="Stage 3 only: decode denoised latents .npz to --output mp4 (skips text & DiT)")
  p.add_argument("--output", default="ltx2.mp4")
  p.add_argument("--compile-cache-dir", default="~/.cache/ltx2_jax")
  args = p.parse_args()
  print(args, flush=True)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  n = len(jax.devices())
  cp = args.cfg_parallel
  if cp is None:
    # v7x: 4 groups x 2; v6e-8: 2 groups x 4 (13.9 vs 14.8 s with 16 / 20 resident blocks).
    cp = 4 if (_is_high_hbm() and n >= 4 and n % 4 == 0) else (2 if n >= 8 and n % 2 == 0 else 1)
  if cp > 1:
    assert n % cp == 0, f"cfg_parallel={cp} must divide device count {n}"
    sp = n // cp
    mesh = Mesh(mesh_utils.create_device_mesh((cp, sp)), ("cfg", "sp"))
    print(f"mesh: {n} devices ({cp} cfg x {sp} sp)", flush=True)
  else:
    mesh = Mesh(mesh_utils.create_device_mesh((n,)), ("sp",))
    print(f"mesh: {n} devices (1-D sp)", flush=True)

  load_dit = not (args.encode_only or args.decode_latents)
  load_text = not bool(args.decode_latents)
  t0 = time.perf_counter()
  gen = Generator(args.repo, mesh, height=args.height, width=args.width, frames=args.frames, fps=args.fps,
                  steps=args.steps, guidance=args.guidance, stg=args.stg, rescale=args.rescale, crf=args.crf,
                  resident=args.resident, fp8=args.fp8, qk_fp8=args.qk_fp8,
                  load_dit=load_dit, load_text=load_text)
  print(f"load: {time.perf_counter() - t0:.1f}s (resident={gen.resident}, fp8={args.fp8}, qk_fp8={args.qk_fp8})",
        flush=True)

  if args.encode_only:
    t0 = time.perf_counter()
    image = Image.open(args.image).convert("RGB")
    ctx, actx = gen.conditioning(args.prompt, args.negative_prompt)
    z0 = gen.image_latent(image)
    x, a = gen.noise(args.seed, z0)
    os.makedirs(os.path.dirname(os.path.abspath(args.encode_only)), exist_ok=True)
    np.savez(
        args.encode_only,
        ctx=np.asarray(ctx, dtype=np.float32),
        actx=np.asarray(actx, dtype=np.float32),
        x=np.asarray(x, dtype=np.float32),
        a=np.asarray(a, dtype=np.float32),
        sigmas=gen.sigmas.astype(np.float32),
        token_t=gen.token_t_np.astype(np.int32),
        first=gen.first_np.astype(np.float32),
        lf=np.int32(gen.lf),
        lh=np.int32(gen.lh),
        lw=np.int32(gen.lw),
        fps=np.float32(gen.fps),
    )
    print(f"wrote Stage 0 conditioning to {args.encode_only} in {time.perf_counter() - t0:.2f}s", flush=True)
    return

  if args.decode_latents:
    t0 = time.perf_counter()
    lat = np.load(args.decode_latents, allow_pickle=False)
    x_j = gen._put(np.asarray(lat["x"], dtype=np.float32))
    a_j = gen._put(np.asarray(lat["a"], dtype=np.float32))
    frames, wave = gen.decode(x_j, a_j)
    save_mp4(args.output, frames, wave, args.fps, gen.sample_rate)
    print(f"decoded {args.decode_latents} -> {args.output} in {time.perf_counter() - t0:.2f}s "
          f"({frames.shape}, {wave.shape[-1] / gen.sample_rate:.2f} s audio)", flush=True)
    return

  image = Image.open(args.image).convert("RGB")
  for i in range(args.runs):
    t0 = time.perf_counter()
    frames, wave = gen(image, args.prompt, args.negative_prompt, args.seed)
    split = ", ".join(f"{k} {v:.2f}" for k, v in gen.times.items())
    print(f"run {i}: {time.perf_counter() - t0:.2f}s ({split})" + (" (includes compilation)" if i == 0 else ""),
          flush=True)
  for d in jax.devices()[:1]:
    print(f"  peak {d.memory_stats().get('peak_bytes_in_use', 0) / 2**30:.1f} GiB", flush=True)
  save_mp4(args.output, frames, wave, args.fps, gen.sample_rate)
  print(f"wrote {args.output} ({frames.shape}, {wave.shape[-1] / gen.sample_rate:.2f} s audio)")


if __name__ == "__main__":
  main()
