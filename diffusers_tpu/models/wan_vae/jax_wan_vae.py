"""The Wan 2.1 video VAE (diffusers ``AutoencoderKLWan``, ``is_residual=False``) in JAX.

Upstream runs both directions chunk by chunk -- the encoder on frame 0 and
then four frames at a time, the decoder one latent frame at a time -- and
threads a feature cache through every causal convolution. Here every frame
goes through at once, channels-last (``[B, T, H, W, C]``), and the cache
logic becomes three closed forms, each exactly what the chunked loop
computes:

* a causal 3x3x3 convolution with the previous chunk's last two frames
  cached is the same convolution over the whole clip with two zero frames
  in front;
* the encoder's temporal downsample (``downsample3d``) passes frame 0
  through (its first chunk has one frame and skips ``time_conv``) and runs
  ``time_conv`` (kernel 3, stride 2, no padding) over the whole clip:
  ``[x0] + conv(x)``;
* the decoder's temporal upsample (``upsample3d``) leaves latent frame 0 to
  the spatial upsample alone (its first chunk caches a ``"Rep"`` marker) and
  runs ``time_conv`` as a causal convolution over frames 1.. only -- frame 0
  is not in their temporal context, the cache starts from zeros -- then
  splits its doubled channels into two frames each.

The mid-block attention is per frame (single head over the pixels), and the
norms are ``F.normalize`` over channels times ``sqrt(C) * gamma``.

With ``mesh`` the rows are split over every chip: convolution halos are
neighbour exchanges, the attention gathers its frame's keys.

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.wan_vae.jax_wan_vae VAE_DIR
compares encode and decode against diffusers on CPU.
"""
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def load(vae_dir, dtype=jnp.float32):
  """(params, config) from a diffusers ``AutoencoderKLWan`` folder; convs as
  [kT, kH, kW, C_in, C_out] (2-D ones as [kH, kW, C_in, C_out])."""
  from safetensors.numpy import load_file

  with open(os.path.join(vae_dir, "config.json")) as fh:
    config = json.load(fh)
  state = {}
  for path in glob.glob(os.path.join(vae_dir, "*.safetensors")):
    state.update(load_file(path))
  params = {}
  for name, value in state.items():
    value = np.asarray(value, np.float32)
    if value.ndim == 5:    # [O, I, kT, kH, kW]
      value = value.transpose(2, 3, 4, 1, 0)
    elif value.ndim == 4 and name.endswith(".weight"):  # 2-D conv [O, I, kH, kW]
      value = value.transpose(2, 3, 1, 0)
    elif name.endswith(".gamma"):
      value = value.reshape(-1)
    params[name] = jnp.asarray(value, dtype)
  return params, config


# ---------------------------------------------------------------------------
# layers
#
# ``st`` carries the row constraint and the time cache: ``st.prev`` (the
# previous chunk's entries; empty for the first chunk or a whole clip) and
# ``st.next`` (this chunk's, for the next one).
# ---------------------------------------------------------------------------


class _State:

  def __init__(self, constrain, prev):
    self.constrain, self.prev, self.next = constrain, prev, {}


def _conv3d(x, params, name, st, stride_t=1, pad_t=None):
  """Causal 3-D conv: the previous chunk's last ``kt - 1`` input frames (zeros
  at the start) in front of time; ``pad_t`` (no cache) overrides."""
  w = params[name + ".weight"].astype(x.dtype)
  kt, kh, kw = w.shape[:3]
  if pad_t is None and kt > 1:
    prev = st.prev.get(name)
    if prev is None:
      prev = jnp.zeros(x.shape[:1] + (kt - 1,) + x.shape[2:], x.dtype)
    x = st.constrain(jnp.concatenate([st.constrain(prev.astype(x.dtype)), x], 1))
    st.next[name] = st.constrain(x[:, -(kt - 1):])
    pad_t = 0
  y = jax.lax.conv_general_dilated(
      x, w, (stride_t, 1, 1), ((pad_t or 0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2)),
      dimension_numbers=("NDHWC", "DHWIO", "NDHWC"))
  return st.constrain(y + params[name + ".bias"].astype(x.dtype))


def _conv2d_frames(x, params, name, st, stride=1, pad=None):
  """A 2-D conv applied to every frame: [B, T, H, W, C]."""
  b, t = x.shape[:2]
  w = params[name + ".weight"].astype(x.dtype)
  kh, kw = w.shape[:2]
  pad = ((kh // 2, kh // 2), (kw // 2, kw // 2)) if pad is None else pad
  y = jax.lax.conv_general_dilated(
      x.reshape(b * t, *x.shape[2:]), w, (stride, stride), pad,
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  y = y + params[name + ".bias"].astype(x.dtype)
  return st.constrain(y.reshape(b, t, *y.shape[1:]))


def _rms(x, params, name):
  """``WanRMS_norm``: ``F.normalize`` over channels * sqrt(C) * gamma (float32)."""
  xf = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(jnp.square(xf), -1, keepdims=True))
  y = xf / jnp.maximum(norm, 1e-12) * np.sqrt(x.shape[-1]) * params[name + ".gamma"]
  return y.astype(x.dtype)


def _resnet(x, params, name, st):
  h = _conv3d(x, params, name + ".conv_shortcut", st) if name + ".conv_shortcut.weight" in params else x
  y = jax.nn.silu(_rms(x, params, name + ".norm1"))
  y = _conv3d(y, params, name + ".conv1", st)
  y = jax.nn.silu(_rms(y, params, name + ".norm2"))
  y = _conv3d(y, params, name + ".conv2", st)
  return y + h


def _attention(x, params, name, st):
  """Single-head attention over each frame's pixels."""
  b, t, hh, ww, c = x.shape
  y = _rms(x, params, name + ".norm")
  qkv = _conv2d_frames(y, params, name + ".to_qkv", st)
  q, k, v = jnp.split(qkv.reshape(b * t, hh * ww, 3 * c), 3, -1)
  out = jax.nn.dot_product_attention(q[:, :, None], k[:, :, None], v[:, :, None])[:, :, 0]
  out = _conv2d_frames(out.reshape(b, t, hh, ww, c), params, name + ".proj", st)
  return x + out


def _mid(x, params, name, st):
  x = _resnet(x, params, name + ".resnets.0", st)
  x = _attention(x, params, name + ".attentions.0", st)
  return _resnet(x, params, name + ".resnets.1", st)


def _downsample(x, params, name, temporal, st):
  """``WanResample`` down: zero-pad right/bottom, 3x3 stride-2 conv; then, for
  ``downsample3d``, ``time_conv`` (k3, s2, no padding): over the first chunk
  frame 0 passes and the conv runs over all of it; over a later chunk the
  conv runs over [the previous chunk's last frame] + the chunk."""
  x = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 0)))
  x = _conv2d_frames(x, params, name + ".resample.1", st, stride=2, pad=((0, 0), (0, 0)))
  if temporal:
    key = name + ".time_conv"
    prev = st.prev.get(key)
    st.next[key] = st.constrain(x[:, -1:])
    if prev is None:
      if x.shape[1] >= 3:
        y = _conv3d(x, params, key, st, stride_t=2, pad_t=0)
        x = jnp.concatenate([x[:, :1], y], 1)
      else:
        x = x[:, :1]
    else:
      x = _conv3d(st.constrain(jnp.concatenate([st.constrain(prev.astype(x.dtype)), x], 1)), params, key, st,
                  stride_t=2, pad_t=0)
  return x


def _upsample(x, params, name, temporal, st):
  """``WanResample`` up: for ``upsample3d`` frames after latent frame 0 go
  through a causal ``time_conv`` (frame 0 not in their context) whose 2C
  outputs become two frames each; then nearest x2 and a 3x3 conv per frame."""
  if temporal:
    first = name + ".started" not in st.prev
    st.next[name + ".started"] = jnp.zeros((), x.dtype)
    head, rest = (x[:, :1], x[:, 1:]) if first else (None, x)
    if rest.shape[1]:
      y = _conv3d(rest, params, name + ".time_conv", st)  # [B, T, H, W, 2C]
      b, t, hh, ww, c2 = y.shape
      y = y.reshape(b, t, hh, ww, 2, c2 // 2).transpose(0, 1, 4, 2, 3, 5).reshape(b, 2 * t, hh, ww, c2 // 2)
      x = y if head is None else jnp.concatenate([head, y], 1)
    else:  # latent frame 0 alone: the conv's cache starts from zeros
      st.next[name + ".time_conv"] = jnp.zeros(x.shape[:1] + (2,) + x.shape[2:], x.dtype)
  x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
  return _conv2d_frames(st.constrain(x), params, name + ".resample.1", st)


# ---------------------------------------------------------------------------
# encoder / decoder
# ---------------------------------------------------------------------------


def _constrainer(mesh, rows):
  if mesh is None:
    return lambda t: t
  spec = NamedSharding(mesh, P(None, None, rows, None, None))
  return lambda t: jax.lax.with_sharding_constraint(t, spec)


def encode_chunk(params, config, video, cache, *, mesh=None, rows=("cfg", "tensor"),
                 dtype=jnp.float32):
  """(latent mean, cache) for the next frames of a clip: the first chunk is
  1 + 4k frames (``cache`` empty), later ones 4k."""
  st = _State(_constrainer(mesh, rows), cache)
  x = st.constrain(video.transpose(0, 2, 3, 4, 1).astype(dtype))
  x = _conv3d(x, params, "encoder.conv_in", st)
  num_res = config["num_res_blocks"]
  temporal = config["temperal_downsample"]
  levels = len(config["dim_mult"])
  idx = 0
  for i in range(levels):
    for _ in range(num_res):
      x = _resnet(x, params, f"encoder.down_blocks.{idx}", st)
      idx += 1
    if i != levels - 1:
      x = _downsample(x, params, f"encoder.down_blocks.{idx}", temporal[i], st)
      idx += 1
  x = _mid(x, params, "encoder.mid_block", st)
  x = jax.nn.silu(_rms(x, params, "encoder.norm_out"))
  x = _conv3d(x, params, "encoder.conv_out", st)
  x = _conv3d(x, params, "quant_conv", st)
  z = config["z_dim"]
  return x[..., :z].transpose(0, 4, 1, 2, 3).astype(jnp.float32), st.next  # the mean (argmax)


def decode_chunk(params, config, latents, cache, *, mesh=None, rows=("cfg", "tensor"),
                 dtype=jnp.float32):
  """(video, cache) for the next latent frames (denormalized)."""
  st = _State(_constrainer(mesh, rows), cache)
  x = st.constrain(latents.transpose(0, 2, 3, 4, 1).astype(dtype))
  x = _conv3d(x, params, "post_quant_conv", st)
  x = _conv3d(x, params, "decoder.conv_in", st)
  x = _mid(x, params, "decoder.mid_block", st)
  temporal = list(config["temperal_downsample"])[::-1]
  levels = len(config["dim_mult"])
  for i in range(levels):
    for j in range(config["num_res_blocks"] + 1):
      x = _resnet(x, params, f"decoder.up_blocks.{i}.resnets.{j}", st)
    if i != levels - 1:
      x = _upsample(x, params, f"decoder.up_blocks.{i}.upsamplers.0", temporal[i], st)
  x = jax.nn.silu(_rms(x, params, "decoder.norm_out"))
  x = _conv3d(x, params, "decoder.conv_out", st)
  return jnp.clip(x.astype(jnp.float32), -1, 1).transpose(0, 4, 1, 2, 3), st.next


def encode(params, config, video, **kw):
  """video [B, 3, T, H, W] in [-1, 1] -> latent mean [B, z, 1 + (T-1)/4, H/8, W/8]."""
  return encode_chunk(params, config, video, {}, **kw)[0]


def decode(params, config, latents, **kw):
  """latents [B, z, T, h, w] (denormalized) -> video [B, 3, 1 + 4(T-1), 8h, 8w] in [-1, 1]."""
  return decode_chunk(params, config, latents, {}, **kw)[0]


def normalize(config, latents):
  mean = jnp.asarray(config["latents_mean"], jnp.float32).reshape(1, -1, 1, 1, 1)
  std = jnp.asarray(config["latents_std"], jnp.float32).reshape(1, -1, 1, 1, 1)
  return (latents - mean) / std


def denormalize(config, latents):
  mean = jnp.asarray(config["latents_mean"], jnp.float32).reshape(1, -1, 1, 1, 1)
  std = jnp.asarray(config["latents_std"], jnp.float32).reshape(1, -1, 1, 1, 1)
  return latents * std + mean


def make(vae_dir, mesh=None, dtype=jnp.float32, chunk=None, rows=("cfg", "tensor")):
  """(params on ``mesh``, config, encode, decode). With ``chunk`` (latent
  frames) both run that many latent frames at a time after frame 0, cache
  threaded through -- the same result in a fraction of the memory."""
  params, config = load(vae_dir, dtype)
  if mesh is not None:
    params = jax.device_put(params, NamedSharding(mesh, P()))
  kw = dict(mesh=mesh, dtype=dtype, rows=rows)
  enc_c = jax.jit(lambda p, v, c: encode_chunk(p, config, v, c, **kw))
  dec_c = jax.jit(lambda p, z, c: decode_chunk(p, config, z, c, **kw))

  def run(fn, p, x, first, step):
    if chunk is None:
      return fn(p, x, {})[0]
    outs, cache = [], {}
    bounds = [0, first] + list(range(first + step, x.shape[2], step)) + [x.shape[2]]
    bounds = sorted(set(bounds))
    for a, b in zip(bounds[:-1], bounds[1:]):
      out, cache = fn(p, x[:, :, a:b], cache)
      outs.append(out)
    return jnp.concatenate(outs, 2)

  enc = lambda p, video: run(enc_c, p, video, 1, 4 * (chunk or 1))
  dec = lambda p, latents: run(dec_c, p, latents, 1, chunk or 1)
  return params, config, enc, dec


def _check_chunks(vae_dir, frames=17, height=32, width=48, chunk=2):
  """Chunked encode / decode against the whole clip, float32, CPU."""
  jax.config.update("jax_default_matmul_precision", "highest")
  rng = np.random.default_rng(0)
  video = jnp.asarray(rng.uniform(-1, 1, (1, 3, frames, height, width)), jnp.float32)
  latents = jnp.asarray(rng.standard_normal((1, 16, (frames - 1) // 4 + 1, height // 8, width // 8)), jnp.float32)
  _, _, enc0, dec0 = make(vae_dir)
  p, _, enc1, dec1 = make(vae_dir, chunk=chunk)
  rel = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))
  print(f"chunked encode rel {rel(enc1(p, video), enc0(p, video)):.2e}, "
        f"decode rel {rel(dec1(p, latents), dec0(p, latents)):.2e}")


def _check(vae_dir, frames=9, height=64, width=96):
  """Encode and decode against diffusers' AutoencoderKLWan, float32, CPU."""
  import torch
  from diffusers import AutoencoderKLWan

  jax.config.update("jax_default_matmul_precision", "highest")
  ref = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
  g = torch.Generator().manual_seed(0)
  video = torch.rand(1, 3, frames, height, width, generator=g) * 2 - 1
  lat_t = (frames - 1) // 4 + 1
  latents = torch.randn(1, 16, lat_t, height // 8, width // 8, generator=g)
  with torch.no_grad():
    want_z = ref.encode(video).latent_dist.mode().numpy()
    want_v = ref.decode(latents).sample.numpy()
  params, config = load(vae_dir)
  z = np.asarray(encode(params, config, jnp.asarray(video.numpy())))
  v = np.asarray(decode(params, config, jnp.asarray(latents.numpy())))
  rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
  print(f"encode: {z.shape} vs {want_z.shape}, rel L2 {rel(z, want_z):.2e}")
  print(f"decode: {v.shape} vs {want_v.shape}, rel L2 {rel(v, want_v):.2e}, max {np.abs(v - want_v).max():.2e}")


if __name__ == "__main__":
  import sys

  _check(sys.argv[1])
  _check_chunks(sys.argv[1])
