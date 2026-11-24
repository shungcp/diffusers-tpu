"""MOSS-Image's VAE decoder (diffusers ``AutoencoderKLFlux2``) in JAX, NHWC.

Traced through torchax, the diffusers decoder spends 71% of its time on
relayouts (0.22 s of 0.23 at 1024x1024 on a v6e-4, 1.0 s at 2048x2048): it is
NCHW in float32, and every GroupNorm reshapes ``[B, C, H, W]`` into
``[B, groups, C/groups * H * W]``. Written channels-last the convolutions
read their natural layout and the norms reduce over trailing axes.

What is here is the whole decode the pipeline runs after the last step:

    latents [B, 128, h, w] -> BatchNorm de-normalisation (running stats)
      -> 2x2 unpatchify to [B, 32, 2h, 2w] -> post_quant_conv -> decoder
      -> clamp, [0, 255] uint8 [B, H, W, 3]

The decoder is the standard SD one: conv_in, a mid block (ResNet, one
single-head self-attention over every pixel, ResNet), four up blocks of three
ResNets (nearest x2 + conv between them), GroupNorm, SiLU, conv_out. The mid
attention is over (H/8)*(W/8) positions -- 65536 at 2048x2048 -- so it runs
the splash kernel rather than materializing the score matrix.

With ``mesh``, rows are split over every chip; the partitioner adds the conv
halos and the norms' all-reduces, and the mid attention gathers keys/values.

    PYTHONPATH=. python -m diffusers_tpu.models.moss_image.jax_vae --size 512
checks it against diffusers on CPU.
"""
import argparse
import functools
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from diffusers_tpu.kernels import splash  # at import: see jax_dit.py

GROUPS = 32
EPS = 1e-6


def load(vae_dir, dtype=jnp.float32):
  """Decoder weights from the snapshot's ``vae/`` as a flat dict; convs HWIO."""
  from safetensors.numpy import load_file

  (path,) = glob.glob(os.path.join(vae_dir, "*.safetensors"))
  raw = load_file(path)
  params = {}
  for name, value in raw.items():
    if not name.startswith(("decoder.", "post_quant_conv.", "bn.running")):
      continue
    if value.ndim == 4:  # OIHW -> HWIO
      value = value.transpose(2, 3, 1, 0)
    keep = jnp.float32 if name.startswith("bn.") else dtype
    params[name] = jnp.asarray(value, keep)
  return params


def _conv(x, params, name, constrain):
  w = params[name + ".weight"]
  pad = (w.shape[0] - 1) // 2
  y = jax.lax.conv_general_dilated(
      x, w.astype(x.dtype), (1, 1), ((pad, pad), (pad, pad)),
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  return constrain(y + params[name + ".bias"].astype(x.dtype))


def _group_norm(x, params, name):
  b, h, w, c = x.shape
  xf = x.astype(jnp.float32).reshape(b, h, w, GROUPS, c // GROUPS)
  mean = xf.mean(axis=(1, 2, 4), keepdims=True)
  var = jnp.square(xf - mean).mean(axis=(1, 2, 4), keepdims=True)
  y = ((xf - mean) * jax.lax.rsqrt(var + EPS)).reshape(b, h, w, c)
  y = y * params[name + ".weight"] + params[name + ".bias"]
  return y.astype(x.dtype)


def _resnet(x, params, name, constrain):
  h = jax.nn.silu(_group_norm(x, params, name + ".norm1"))
  h = _conv(h, params, name + ".conv1", constrain)
  h = jax.nn.silu(_group_norm(h, params, name + ".norm2"))
  h = _conv(h, params, name + ".conv2", constrain)
  if name + ".conv_shortcut.weight" in params:
    x = _conv(x, params, name + ".conv_shortcut", constrain)
  return x + h


def _attention(x, params, name, attend):
  b, h, w, c = x.shape
  y = _group_norm(x, params, name + ".group_norm").reshape(b, h * w, c)

  def linear(t, which):
    return t @ params[f"{name}.{which}.weight"].T.astype(t.dtype) + params[f"{name}.{which}.bias"].astype(t.dtype)

  q, k, v = (linear(y, n)[:, None] for n in ("to_q", "to_k", "to_v"))  # [B, 1, N, C]
  out = attend(q, k, v)[:, 0]
  return x + linear(out, "to_out.0").reshape(b, h, w, c)


def _upsample(x, params, name, constrain):
  x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
  return _conv(constrain(x), params, name + ".conv", constrain)


def _make_attend(mesh, rows):
  """Single-head attention over every pixel, the splash kernel per chip."""
  def local(q, k, v):
    return splash.attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16),
                            v.astype(jnp.bfloat16), scale=None, block_q=2048,
                            block_kv=2048, block_kv_compute=2048)

  if jax.default_backend() == "cpu":  # the Pallas kernel is TPU-only
    t = lambda a: a.transpose(0, 2, 1, 3)
    return lambda q, k, v: t(jax.nn.dot_product_attention(t(q), t(k), t(v)))
  if mesh is None:
    return lambda q, k, v: local(q, k, v).astype(q.dtype)

  qspec, kvspec = P(None, None, rows, None), P()
  run = shard_map(local, mesh=mesh, in_specs=(qspec, kvspec, kvspec),
                  out_specs=qspec, check_rep=False)
  return lambda q, k, v: run(q, k, v).astype(q.dtype)


def decode(params, latents, *, mesh=None, rows=("cfg", "tensor"), dtype=jnp.float32,
           batch_norm_eps=1e-4):
  """latents [B, 128, h, w] (as the DiT emits them) -> uint8 [B, 16h, 16w, 3]."""
  if mesh is None:
    constrain = lambda t: t
  else:
    spec = NamedSharding(mesh, P(None, rows, None, None))
    constrain = lambda t: jax.lax.with_sharding_constraint(t, spec)
  attend = _make_attend(mesh, rows)

  z = latents.astype(jnp.float32)
  scale = jnp.sqrt(params["bn.running_var"] + batch_norm_eps)
  z = z * scale[None, :, None, None] + params["bn.running_mean"][None, :, None, None]
  b, c, h, w = z.shape
  # [B, 32, 2, 2, h, w] -> [B, 2h, 2w, 32]  (the pipeline's permute, in NHWC)
  z = z.reshape(b, c // 4, 2, 2, h, w).transpose(0, 4, 2, 5, 3, 1).reshape(b, 2 * h, 2 * w, c // 4)
  x = constrain(z.astype(dtype))
  x = _conv(x, params, "post_quant_conv", constrain)
  x = _conv(x, params, "decoder.conv_in", constrain)
  x = _resnet(x, params, "decoder.mid_block.resnets.0", constrain)
  x = _attention(x, params, "decoder.mid_block.attentions.0", attend)
  x = _resnet(x, params, "decoder.mid_block.resnets.1", constrain)
  for i in range(4):
    for j in range(3):
      x = _resnet(x, params, f"decoder.up_blocks.{i}.resnets.{j}", constrain)
    if f"decoder.up_blocks.{i}.upsamplers.0.conv.weight" in params:
      x = _upsample(x, params, f"decoder.up_blocks.{i}.upsamplers.0", constrain)
  x = jax.nn.silu(_group_norm(x, params, "decoder.conv_norm_out"))
  x = _conv(x, params, "decoder.conv_out", constrain)
  x = jnp.clip(x.astype(jnp.float32), -1, 1)
  return jnp.round((x / 2 + 0.5) * 255).astype(jnp.uint8)


def make_decoder(vae_dir, mesh=None, dtype=jnp.float32, batch_norm_eps=1e-4):
  """(params, jitted ``latents -> uint8 image``), weights replicated on ``mesh``."""
  params = load(vae_dir, dtype)
  if mesh is not None:
    params = jax.device_put(params, NamedSharding(mesh, P()))
  # 16 MiB scoped VMEM: the decoder compiles in ~11 s instead of ~25 s at
  # the 32 MiB default (v6e-8, 1024x1024), and its run time is ~10 ms either
  # way. `DT_MOSS_VAE_VMEM_KIB=0` keeps the global value.
  import os
  kib = int(os.environ.get("DT_MOSS_VAE_VMEM_KIB", 16384))
  # A TPU compile option: other backends reject it (the CPU check below).
  options = ({"xla_tpu_scoped_vmem_limit_kib": kib}
             if kib and jax.default_backend() == "tpu" else None)
  fn = jax.jit(functools.partial(decode, mesh=mesh, dtype=dtype,
                                 batch_norm_eps=batch_norm_eps),
               compiler_options=options)
  return params, fn


def _check(args):
  """Against diffusers' AutoencoderKLFlux2 + the pipeline's decode, on CPU."""
  import torch
  from diffusers import AutoencoderKLFlux2

  from diffusers_tpu.models.moss_image import remote

  vae_dir = os.path.join(remote.snapshot(), "vae")
  vae = AutoencoderKLFlux2.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
  h = w = args.size // 16
  latents = torch.randn(1, 128, h, w, generator=torch.Generator().manual_seed(0))
  with torch.no_grad():
    bn = vae.bn
    z = latents * (bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
    z = z.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
    ref = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
  ref = np.round((ref[0].permute(1, 2, 0).numpy() / 2 + 0.5) * 255)
  jax.config.update("jax_default_matmul_precision", "highest")
  params, fn = make_decoder(vae_dir, batch_norm_eps=vae.config.batch_norm_eps)
  out = np.asarray(fn(params, jnp.asarray(latents.numpy())))[0].astype(np.float64)
  diff = np.abs(out - ref)
  print(f"max |diff| {diff.max():.0f}/255, mean {diff.mean():.4f}, "
        f"PSNR {10 * np.log10(255 ** 2 / max(np.square(diff).mean(), 1e-12)):.1f} dB")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--size", type=int, default=512)
  _check(p.parse_args())
