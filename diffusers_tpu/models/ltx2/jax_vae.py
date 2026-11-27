"""The LTX-2 video VAE decoder (diffusers ``AutoencoderKLLTX2Video.decoder``) in JAX.

conv_in -> mid block -> up blocks (an upsampler: 3-D conv then depth-to-space
over time / height / width, the first ``s_t - 1`` frames dropped; then
ResNets) -> per-pixel RMS norm, SiLU, conv_out -> 4 x 4 unpatchify. Every
ResNet is RMS norm (over channels, no weights) -> SiLU -> conv, twice, plus a
LayerNorm + 1x1 conv shortcut when the width changes. Convolutions are 3x3x3,
zero-padded in space and edge-replicated in time (``decoder_causal`` false).

Layout NDHWC. With ``mesh`` the rows are split over every chip (XLA turns the
3x3 convolutions into neighbour halo exchanges).

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.ltx2.jax_vae VAE_DIR
compares against diffusers on CPU.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np


def load(vae_dir, dtype=jnp.bfloat16):
  from diffusers_tpu.models.ltx2 import jax_dit as jd

  with open(os.path.join(vae_dir, "config.json")) as fh:
    config = json.load(fh)
  params = {}
  for name, v in jd._iter_state(vae_dir):
    if not name.startswith("decoder."):
      continue
    v = np.asarray(v, np.float32)
    if v.ndim == 5:  # Conv3d [O, I, kt, kh, kw] -> DHWIO
      v = v.transpose(2, 3, 4, 1, 0)
    key = name[len("decoder."):].replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")  # causal-conv wrapper
    params[key] = v.astype(jnp.dtype(dtype))
  return params, config


def _conv(x, p, name, cst):
  """3x3x3 (or 1x1x1) conv: time edge-replicated, space zero-padded."""
  w = p[name + ".weight"]
  kt, kh, kw = w.shape[:3]
  if kt > 1:
    r = (kt - 1) // 2
    x = jnp.concatenate([jnp.repeat(x[:, :1], r, 1), x, jnp.repeat(x[:, -1:], r, 1)], 1)
  y = jax.lax.conv_general_dilated(x, w.astype(x.dtype), (1, 1, 1), ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2)),
                                   dimension_numbers=("NDHWC", "DHWIO", "NDHWC"))
  return cst(y + p[name + ".bias"].astype(x.dtype))


def _pixel_rms(x, eps=1e-8):
  xf = x.astype(jnp.float32)
  return (xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + eps)).astype(x.dtype)


def _layer_norm(x, w, b, eps=1e-6):
  xf = x.astype(jnp.float32)
  m = xf.mean(-1, keepdims=True)
  y = (xf - m) * jax.lax.rsqrt(jnp.square(xf - m).mean(-1, keepdims=True) + eps)
  return (y * w.astype(jnp.float32) + b.astype(jnp.float32)).astype(x.dtype)


def _resnet(x, p, name, cst):
  h = _conv(jax.nn.silu(_pixel_rms(x)), p, name + ".conv1", cst)
  h = _conv(jax.nn.silu(_pixel_rms(h)), p, name + ".conv2", cst)
  if name + ".conv_shortcut.weight" in p:
    x = _layer_norm(x, p[name + ".norm3.weight"], p[name + ".norm3.bias"])
    x = _conv(x, p, name + ".conv_shortcut", cst)
  return h + x


def _upsample(x, p, name, stride, cst):
  s0, s1, s2 = stride
  y = _conv(x, p, name + ".conv", cst)
  b, t, h, w, c = y.shape
  c2 = c // (s0 * s1 * s2)
  y = y.reshape(b, t, h, w, c2, s0, s1, s2).transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, t * s0, h * s1, w * s2, c2)
  return cst(y[:, s0 - 1:])


_STRIDES = {"spatial": (1, 2, 2), "temporal": (2, 1, 1), "spatiotemporal": (2, 2, 2)}


def decode(params, config, z, cst=lambda t: t):
  """z [B, 128, F, h, w] (denormalized latents, as diffusers takes them) ->
  video [B, F', H, W, 3] in [-1, 1]-ish (not clipped)."""
  x = cst(jnp.transpose(z, (0, 2, 3, 4, 1)))
  x = _conv(x, params, "conv_in", cst)
  layers = list(reversed(config["decoder_layers_per_block"]))
  for i in range(layers[0]):
    x = _resnet(x, params, f"mid_block.resnets.{i}", cst)
  for i, kind in enumerate(config["upsample_type"]):
    pre = f"up_blocks.{i}"
    if pre + ".conv_in.conv1.weight" in params:
      x = _resnet(x, params, pre + ".conv_in", cst)
    if pre + ".upsamplers.0.conv.weight" in params:
      x = _upsample(x, params, pre + ".upsamplers.0", _STRIDES[kind], cst)
    for j in range(layers[i + 1]):
      x = _resnet(x, params, f"{pre}.resnets.{j}", cst)
  x = _conv(jax.nn.silu(_pixel_rms(x)), params, "conv_out", cst)
  p, pt = config["patch_size"], config["patch_size_t"]
  b, t, h, w, c = x.shape
  x = x.reshape(b, t, h, w, c // (pt * p * p), pt, p, p)  # channel = (c, pt, p_w, p_h)
  x = x.transpose(0, 1, 5, 2, 7, 3, 6, 4).reshape(b, t * pt, h * p, w * p, -1)
  return x


def make_decoder(vae_dir, mesh=None, axis=None, dtype=jnp.bfloat16):
  """(params on the mesh, jitted decode(params, z) -> [B, F, H, W, 3])."""
  from jax.sharding import NamedSharding
  from jax.sharding import PartitionSpec as P

  params, config = load(vae_dir, dtype)
  if mesh is None:
    return params, jax.jit(lambda prm, z: decode(prm, config, z))
  rep = NamedSharding(mesh, P())
  params = jax.tree.map(lambda v: jax.device_put(v, rep), params)
  rows = NamedSharding(mesh, P(None, None, axis, None, None))
  cst = lambda t: jax.lax.with_sharding_constraint(t, rows) if t.shape[2] % mesh.shape[axis] == 0 else t
  return params, jax.jit(lambda prm, z: decode(prm, config, z, cst))


def _check(vae_dir, frames=3, height=4, width=6):
  import torch
  from diffusers import AutoencoderKLLTX2Video

  jax.config.update("jax_default_matmul_precision", "highest")
  ref = AutoencoderKLLTX2Video.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
  z = torch.randn(1, 128, frames, height, width)
  with torch.no_grad():
    r = ref.decode(z, return_dict=False)[0].numpy().transpose(0, 2, 3, 4, 1)
  params, dec = make_decoder(vae_dir, dtype=jnp.float32)
  o = np.asarray(dec(params, jnp.asarray(z.numpy())))
  print("decode", o.shape, r.shape, "rel %.2e" % (np.linalg.norm(o - r) / np.linalg.norm(r)))


if __name__ == "__main__":
  import sys

  _check(sys.argv[1])
