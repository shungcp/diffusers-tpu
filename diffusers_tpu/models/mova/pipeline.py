"""MOVA image + text -> video + audio (TI2VA) with native JAX models.

    first frame -> Wan VAE encode (frame 0, then zeros) -> condition (4 mask + 16 latent)
    prompt / negative -> UMT5-XXL (512 tokens, padded rows zeroed)
    50 steps: [positive, negative] through video expert (high noise, then low
              noise below t = 900) + audio tower + bridge, CFG, Euler
    -> Wan VAE decode (video) and DAC decode (audio, 48 kHz)

The schedule is the official ``FlowMatchPairScheduler``'s: timesteps from a
shifted linear grid, and each Euler step's sigma is the entry of the
1000-step *training* table nearest to the timestep (``timestep_to_sigma``).

    PYTHONPATH=. python -m diffusers_tpu.models.mova.pipeline --image ref.png --prompt "..."
"""
import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.dac import jax_dac
from diffusers_tpu.models.mova import jax_dit
from diffusers_tpu.models.wan_vae import jax_wan_vae

REPO = "OpenMOSS-Team/MOVA-360p"
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指")


def schedule(steps, shift=5.0, num_train=1000, boundary_ratio=0.9):
  """(timesteps [steps], sigma_from [steps], sigma_to [steps], expert [steps]).

  ``expert`` is 0 for ``video_dit`` and 1 for ``video_dit_2`` (from the first
  timestep below ``boundary_ratio * num_train``)."""
  def table(n):
    s = np.linspace(1.0, 0.0, n + 1, dtype=np.float32)[:-1]  # extra_one_step, sigma_min 0
    s = (shift * s / (1 + (shift - 1) * s)).astype(np.float32)
    return s * np.float32(num_train), s

  train_t, train_s = table(num_train)
  t, _ = table(steps)
  snap = lambda x: train_s[int(np.argmin(np.abs(train_t - np.float32(x))))]
  s_from = np.array([snap(x) for x in t], np.float32)
  s_to = np.array([snap(x) for x in t[1:]] + [0.0], np.float32)
  expert = (t < boundary_ratio * num_train).astype(np.int32)
  expert = np.maximum.accumulate(expert)
  return t, s_from, s_to, expert


def first_frame_condition(encode, vae_params, vae_config, image, frames):
  """image [H, W, 3] uint8 -> condition [1, 20, T, H/8, W/8] (mask + normalized latent)."""
  x = image.astype(np.float32) / 255.0 * 2 - 1
  hh, ww = x.shape[:2]
  video = np.zeros((1, 3, frames, hh, ww), np.float32)
  video[0, :, 0] = x.transpose(2, 0, 1)
  latent = jax_wan_vae.normalize(vae_config, encode(vae_params, jnp.asarray(video)))
  t = latent.shape[2]
  mask = np.zeros((1, 4, t, hh // 8, ww // 8), np.float32)
  mask[:, :, 0] = 1.0  # latent frame 0 holds the given image (4 pixel frames' worth of ones)
  return jnp.concatenate([jnp.asarray(mask), latent], 1)


def clean_prompt(text):
  import html
  import re

  from diffusers_tpu.models.moss_soundeffect.text import fix_text

  text = html.unescape(html.unescape(fix_text(text))).strip()
  return re.sub(r"\s+", " ", text).strip()


class TextEncoder:
  """UMT5-XXL (transformers' ``UMT5EncoderModel``): [1, 512, 4096], padded rows zero."""

  def __init__(self, snapshot, dtype=None):
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snapshot, "tokenizer"))
    self.model = UMT5EncoderModel.from_pretrained(os.path.join(snapshot, "text_encoder"),
                                                  dtype=dtype or torch.bfloat16).eval()
    self.cache = {}

  def __call__(self, prompt):
    import torch

    if prompt not in self.cache:
      enc = self.tokenizer([clean_prompt(prompt)], padding="max_length", max_length=512,
                           truncation=True, add_special_tokens=True, return_tensors="pt")
      with torch.no_grad():
        h = self.model(enc.input_ids, enc.attention_mask).last_hidden_state.float()
      n = int(enc.attention_mask.sum())
      h[:, n:] = 0
      self.cache[prompt] = h.numpy()
    return self.cache[prompt]


def make_step(vconf, aconf, bconf, video_fps, attention=jax_dit.dense_attention, video_layer=None):
  """jitted ``(video, audio, bridge, latents_v, latents_a, condition, context, t,
  s_from, s_to, guidance) -> (latents_v, latents_a)``: one CFG + Euler step for
  both modalities; ``context`` [2, 512, 4096] = [positive, negative]."""

  def step(video, audio, bridge, lat_v, lat_a, condition, context, t, s_from, s_to, guidance):
    batch = context.shape[0]
    x = jnp.concatenate([lat_v, condition], 1)
    x = jnp.broadcast_to(x, (batch,) + x.shape[1:])
    a = jnp.broadcast_to(lat_a, (batch,) + lat_a.shape[1:])
    v_pred, a_pred = jax_dit.dual_step(video, vconf, audio, aconf, bridge, bconf, x, a, context,
                                       t, t, video_fps, attention=attention,
                                       video_layer=video_layer)
    v_pred, a_pred = v_pred.astype(jnp.float32), a_pred.astype(jnp.float32)
    if batch > 1:
      v_pred = v_pred[1:2] + guidance * (v_pred[0:1] - v_pred[1:2])
      a_pred = a_pred[1:2] + guidance * (a_pred[0:1] - a_pred[1:2])
    dt = s_to - s_from
    return lat_v + dt * v_pred, lat_a + dt * a_pred

  return jax.jit(step)


def audio_frames(num_frames, video_fps, sample_rate=48000, hop=960):
  samples = int(sample_rate * num_frames / video_fps)
  return (samples - 1) // hop + 1, samples


def crop_and_resize(image, height, width):
  """The official reference-image preparation: center crop to the aspect, resize."""
  from PIL import Image

  image = np.array(image.convert("RGB"))
  ih, iw, _ = image.shape
  if ih / iw < height / width:
    cw = int(ih / height * width)
    left = (iw - cw) // 2
    image = image[:, left:left + cw]
  else:
    ch = int(iw / width * height)
    top = (ih - ch) // 2
    image = image[top:top + ch, :]
  return np.array(Image.fromarray(image).resize((width, height)))


# ---------------------------------------------------------------------------
# TPU generator
# ---------------------------------------------------------------------------

_ALL = ("cfg", "tensor")
TEXT_SHARDINGS = {
    r"block\.\d+\.layer\.0\.SelfAttention\.(q|k|v)\.weight": (_ALL, None),
    r"block\.\d+\.layer\.0\.SelfAttention\.o\.weight": (None, _ALL),
    r"block\.\d+\.layer\.1\.DenseReluDense\.(wi_0|wi_1)\.weight": (_ALL, None),
    r"block\.\d+\.layer\.1\.DenseReluDense\.wo\.weight": (None, _ALL),
}


def _text_core_class():
  import torch

  class TextCore(torch.nn.Module):
    """UMT5's encoder blocks and final norm, an additive key mask [1, 1, 1, L]."""

    def __init__(self, encoder):
      super().__init__()
      self.block = encoder.block
      self.final_layer_norm = encoder.final_layer_norm

    def forward(self, embeds, mask):
      h = embeds
      for layer in self.block:
        h = layer(h, mask)
      return self.final_layer_norm(h)

  return TextCore


class Generator:
  """Every model resident on ``mesh`` (``('cfg', 'tensor')``, cfg = 1): both
  video experts, the audio tower and bridge tensor parallel; UMT5 through
  torchax, tensor parallel; the Wan VAE and DAC with rows / time split."""

  def __init__(self, repo=REPO, mesh=None, env=None, height=352, width=640, frames=193,
               video_fps=24.0, vae_chunk=4, seq_parallel=True):
    import contextlib

    import torch
    import torchax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from transformers import AutoTokenizer, UMT5EncoderModel

    from diffusers_tpu.hub import local_snapshot
    from diffusers_tpu.models.moss_image.torchax_pipeline import _move_module, _shard

    if height % 16 or width % 16 or (frames - 1) % 4:
      raise ValueError("height and width must be multiples of 16, frames 1 + 4k")
    self.mesh, self.env = mesh, env or torchax.default_env()
    self.height, self.width, self.frames, self.video_fps = height, width, frames, video_fps
    snap = local_snapshot(repo)
    self.times = {}

    # Text: embedding lookup on the host, the blocks through torchax.
    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    with contextlib.suppress(RuntimeError):
      torchax.disable_globally()
    t5 = UMT5EncoderModel.from_pretrained(os.path.join(snap, "text_encoder"), dtype=torch.bfloat16,
                                          attn_implementation="eager").eval()
    self._embed = t5.encoder.embed_tokens
    # ``wo`` in float32 (as transformers keeps it for T5 in half precision): its
    # row-parallel partial sums are then reduced in float32. In bfloat16 that
    # reduction doubles the error against float32 (rel 8% vs 4%; 3% with this).
    for blk in t5.encoder.block:
      blk.layer[1].DenseReluDense.wo.float()
    torchax.enable_globally()
    self.env._mesh = mesh
    self.env.default_device_or_sharding = NamedSharding(mesh, P())
    core = _text_core_class()(t5.encoder)
    _move_module(core, self.env)
    core = torchax.compile(core)
    core.params = _shard(core.params, TEXT_SHARDINGS, mesh)
    core.buffers = _shard(core.buffers, TEXT_SHARDINGS, mesh)
    self._text = core
    self._prompts = {}
    torchax.disable_globally()

    # DiT: both experts resident (cfg = 1: 57 GB of experts leave no room for two copies).
    self.audio, self.aconf = jax_dit.load_tower(os.path.join(snap, "audio_dit"))
    self.bridge, self.bconf = jax_dit.load_bridge(os.path.join(snap, "dual_tower_bridge"))
    self.audio, self.bridge = jax_dit.place(self.audio, mesh), jax_dit.place(self.bridge, mesh)
    self.experts = []
    for d in ("video_dit", "video_dit_2"):
      v, self.vconf = jax_dit.load_tower(os.path.join(snap, d))
      self.experts.append(jax_dit.place(v, mesh))
      del v
    tokens = ((frames - 1) // 4 + 1) * (height // 16) * (width // 16)
    parts = mesh.shape["tensor"]
    layer = (jax_dit.make_video_layer(mesh, self.vconf, self.bconf)
             if seq_parallel and tokens % parts == 0 and self.vconf["num_heads"] % parts == 0
             else None)
    self._step = make_step(self.vconf, self.aconf, self.bconf, video_fps,
                           attention=jax_dit.make_attention(mesh), video_layer=layer)

    # VAEs.
    # VAE rows over every chip when the latent rows divide evenly (44 over 4),
    # else over ``tensor`` alone (44 over 8 does not).
    rows = ("cfg", "tensor") if (height // 8) % mesh.size == 0 else ("tensor",)
    self.vae, self.vae_config, self._encode, decode = jax_wan_vae.make(
        os.path.join(snap, "video_vae"), mesh, dtype=jnp.bfloat16, chunk=vae_chunk, rows=rows)
    self._decode_video = lambda p, z: decode(p, jax_wan_vae.denormalize(self.vae_config, z))
    self.dac, self._decode_audio, _ = jax_dac.make_decoder(os.path.join(snap, "audio_vae"), mesh=mesh)
    self.audio_len, self.audio_samples = audio_frames(frames, video_fps)

  def _put(self, array, spec=None):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return jax.device_put(array, NamedSharding(self.mesh, spec or P()))

  def encode_text(self, prompt):
    """[1, 512, 4096] bf16 on the mesh, padded rows zeroed; cached per prompt."""
    import torch
    import torchax

    if prompt not in self._prompts:
      enc = self.tokenizer([clean_prompt(prompt)], padding="max_length", max_length=512,
                           truncation=True, add_special_tokens=True, return_tensors="np")
      ids, keep = enc["input_ids"], enc["attention_mask"]
      with torch.no_grad():
        embeds = self._embed(torch.from_numpy(ids).long())
      bias = np.where(keep[:, None, None, :] > 0, 0.0, np.finfo(np.float32).min)
      torchax.enable_globally()
      try:
        e = self.env.j2t_iso(jax.lax.bitcast_convert_type(
            self._put(embeds.view(torch.int16).numpy()), jnp.bfloat16))
        m = self.env.j2t_iso(self._put(bias.astype(jnp.bfloat16)))
        h = self.env.t2j_iso(self._text(e, m))
      finally:
        torchax.disable_globally()
      self._prompts[prompt] = jnp.where(self._put(keep[..., None] > 0), h, 0).astype(jnp.bfloat16)
    return self._prompts[prompt]

  def condition(self, image):
    """image [H, W, 3] uint8 -> [1, 20, T, h, w] on the mesh."""
    return first_frame_condition(lambda p, v: self._encode(p, self._put(v)), self.vae,
                                 self.vae_config, image, self.frames)

  def noise(self, seed):
    """The official order on the global torch CPU generator: video, then audio."""
    import torch

    torch.manual_seed(int(seed))
    t = (self.frames - 1) // 4 + 1
    lat_v = torch.randn((1, 16, t, self.height // 8, self.width // 8), dtype=torch.float32)
    lat_a = torch.randn((1, 128, self.audio_len), dtype=torch.float32)
    return self._put(lat_v.numpy()), self._put(lat_a.numpy())

  def denoise(self, condition, context, lat_v, lat_a, steps=50, cfg_scale=5.0, shift=5.0):
    t, s_from, s_to, expert = schedule(steps, shift)
    sinusoids = jax_dit.time_sinusoid(t)
    guidance = self._put(np.float32(cfg_scale))
    for i in range(steps):
      lat_v, lat_a = self._step(self.experts[expert[i]], self.audio, self.bridge, lat_v, lat_a,
                                condition, context, self._put(sinusoids[i:i + 1]),
                                self._put(s_from[i]), self._put(s_to[i]), guidance)
    return lat_v, lat_a

  def decode(self, lat_v, lat_a):
    """(frames [F, H, W, 3] uint8, audio [samples] float32 at 48 kHz)."""
    # Replicated first: the step leaves the latents in whatever layout it chose,
    # and a split one compiles the VAE chunks into far bigger programs.
    video = self._decode_video(self.vae, self._put(lat_v))
    audio = self._decode_audio(self.dac, self._put(lat_a))
    video = np.asarray(video)[0].transpose(1, 2, 3, 0)
    frames = np.round((video + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return frames, np.asarray(audio)[0, 0][:self.audio_samples]

  def __call__(self, image, prompt, negative_prompt=NEGATIVE_PROMPT, steps=50, cfg_scale=5.0,
               shift=5.0, seed=42):
    import time

    from jax.sharding import PartitionSpec as P

    t0 = time.perf_counter()
    cond = self.condition(image)
    context = jnp.concatenate([self.encode_text(prompt), self.encode_text(negative_prompt)])
    context = self._put(context, P("cfg"))  # [positive, negative]: split over cfg when 2
    lat_v, lat_a = self.noise(seed)
    jax.block_until_ready((cond, context))
    t1 = time.perf_counter()
    lat_v, lat_a = jax.block_until_ready(
        self.denoise(cond, context, lat_v, lat_a, steps, cfg_scale, shift))
    t2 = time.perf_counter()
    out = self.decode(lat_v, lat_a)
    t3 = time.perf_counter()
    self.times = {"encode": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return out


def save_mp4(path, frames, audio, fps=24.0, rate=48000):
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
  from frames_to_mp4 import write

  write(frames, path, fps=int(round(fps)), wave=np.stack([audio, audio]), rate=rate)


def main():
  import argparse
  import time

  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from PIL import Image

  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", default=REPO)
  p.add_argument("--image", required=True)
  p.add_argument("--prompt", required=True)
  p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
  p.add_argument("--height", type=int, default=352)
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--frames", type=int, default=193)
  p.add_argument("--fps", type=float, default=24.0)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--cfg-scale", type=float, default=5.0)
  p.add_argument("--shift", type=float, default=5.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--tensor-parallel", action="store_true",
                 help="video layers tensor parallel (Megatron) instead of sequence parallel")
  p.add_argument("--cfg", type=int, default=0, choices=(0, 1, 2),
                 help="CFG-parallel mesh axis: 0 = 2 with 8+ chips (each half holds both experts "
                      "split over the rest), else 1; 1 above 480x832 on chips under 64 GB")
  p.add_argument("--output", default="mova.mp4")
  p.add_argument("--compile-cache-dir", default="~/.cache/mova_jax")
  args = p.parse_args()
  print(args, flush=True)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  n = len(jax.devices())
  if not args.cfg:
    # Two CFG groups need both experts on half the chips: at 720p on a v6e (32 GB a chip) that
    # plus the activations does not fit (the VAE encode already fails), so one group of all chips.
    hbm = jax.devices()[0].memory_stats().get("bytes_limit", 0)
    big = args.height * args.width > 480 * 832 and hbm < 64 * 2**30
    args.cfg = 2 if n >= 8 and not big else 1
  cfg = args.cfg
  if cfg not in (1, 2) or n % cfg:
    raise SystemExit(f"--cfg must be 1 or 2 and divide the {n} chips (got {cfg})")
  mesh = Mesh(mesh_utils.create_device_mesh((cfg, n // cfg)), ("cfg", "tensor"))
  print(f"mesh cfg={cfg} x tensor={n // cfg}, {jax.devices()[0].device_kind}", flush=True)
  t0 = time.perf_counter()
  gen = Generator(args.repo, mesh, height=args.height, width=args.width, frames=args.frames,
                  video_fps=args.fps, seq_parallel=not args.tensor_parallel)
  print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
  image = crop_and_resize(Image.open(args.image), args.height, args.width)
  for i in range(args.runs):
    t0 = time.perf_counter()
    frames, audio = gen(image, args.prompt, args.negative_prompt, args.steps, args.cfg_scale,
                        args.shift, args.seed)
    split = ", ".join(f"{k} {v:.1f}" for k, v in gen.times.items())
    print(f"run {i}: {time.perf_counter() - t0:.1f}s ({split})"
          + (" (includes compilation)" if i == 0 else ""), flush=True)
  for d in jax.devices():
    st = d.memory_stats()
    print(f"  {d.id}: peak {st.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB", flush=True)
  save_mp4(args.output, frames, audio, args.fps)
  print(f"wrote {args.output} ({frames.shape}, {len(audio) / 48000:.2f} s audio)")


if __name__ == "__main__":
  main()
