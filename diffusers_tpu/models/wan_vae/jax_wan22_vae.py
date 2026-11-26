"""The Wan 2.2 video VAE decoder (z48, 16x spatial with a 2x2 output patch, 4x
time) in JAX, as OmniVAE runs it: the whole clip at once (``decode`` without
the streaming cache), channels-last ``[B, T, H, W, C]``.

Differences from the Wan 2.1 decoder (``jax_wan_vae``):

* each up stage is an ``Up_ResidualBlock``: ``num_res_blocks + 1`` residual
  blocks and the resample, plus a parameter-free ``DupUp3D`` shortcut (channels
  repeated, then unfolded into 2x2 pixels / 2 frames; the first frame of a
  temporal stage is dropped, as with ``first_chunk``);
* the mid attention can carry q / k RMSNorms (``nn.RMSNorm``, whose eps is
  the dtype's machine epsilon -- 7.8e-3 in bfloat16);
* the head emits 12 channels, unpatchified to 3 x (2 x 2) pixels;
* a 1x1x1 ``conv2`` runs before the decoder, and latents are denormalized
  with the checkpoint's ``mean_tensor`` / ``std_tensor``.

The causal 3-D convolutions see two zero frames in front; ``upsample3d``
leaves frame 0 to the spatial path and runs ``time_conv`` over frames 1..
(exactly the non-streaming branch of the official ``Resample``).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def load_omnivae(path, dtype=jnp.float32, prefix="video_vae."):
  """(params, config) from an OmniVAE ``state_dict.pt`` (``model_state_dict``,
  ``video_vae.*``: decoder, ``conv2`` and the latent statistics)."""
  import torch

  ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
  sd = ck.get("model_state_dict", ck)
  params = {}
  for name, value in sd.items():
    if not name.startswith(prefix):
      continue
    name = name[len(prefix):]
    if not (name.startswith("decoder.") or name.startswith("conv2.") or name.endswith("_tensor")):
      continue
    v = value.float().numpy()
    if v.ndim == 5:          # [O, I, kT, kH, kW] -> [kT, kH, kW, I, O]
      v = v.transpose(2, 3, 4, 1, 0)
    elif v.ndim == 4 and name.endswith(".weight"):  # 2-D conv [O, I, kH, kW]
      v = v.transpose(2, 3, 1, 0)
    elif name.endswith(".gamma"):
      v = v.reshape(-1)
    params[name] = jnp.asarray(v, jnp.float32 if name.endswith("_tensor") else dtype)
  stages = sorted({int(k.split(".")[2]) for k in params if k.startswith("decoder.upsamples.")})
  config = {"stages": []}
  for i in stages:
    pre = f"decoder.upsamples.{i}.upsamples."
    blocks = sorted({int(k[len(pre):].split(".")[0]) for k in params if k.startswith(pre)})
    res = [b for b in blocks if f"{pre}{b}.residual.0.gamma" in params]
    up = [b for b in blocks if f"{pre}{b}.resample.1.weight" in params]
    config["stages"].append({
        "res": len(res),
        "up": bool(up),
        "temporal": bool(up) and f"{pre}{up[0]}.time_conv.weight" in params,
    })
  return params, config


def _conv3d(x, params, name, constrain):
  """Causal 3-D conv: ``2 * (kt // 2)`` zero frames in front, spatial 'same'."""
  w = params[name + ".weight"].astype(x.dtype)
  kt, kh, kw = w.shape[:3]
  y = jax.lax.conv_general_dilated(
      x, w, (1, 1, 1), ((2 * (kt // 2), 0), (kh // 2, kh // 2), (kw // 2, kw // 2)),
      dimension_numbers=("NDHWC", "DHWIO", "NDHWC"))
  return constrain(y + params[name + ".bias"].astype(x.dtype))


def _conv2d_frames(x, params, name, constrain):
  b, t = x.shape[:2]
  w = params[name + ".weight"].astype(x.dtype)
  kh, kw = w.shape[:2]
  y = jax.lax.conv_general_dilated(
      x.reshape(b * t, *x.shape[2:]), w, (1, 1), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  y = y + params[name + ".bias"].astype(x.dtype)
  return constrain(y.reshape(b, t, *y.shape[1:]))


def _rms(x, gamma):
  """``RMS_norm``: ``F.normalize`` over channels * sqrt(C) * gamma (float32 math)."""
  xf = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(jnp.square(xf), -1, keepdims=True))
  y = xf / jnp.maximum(norm, 1e-12) * np.sqrt(x.shape[-1]) * gamma.astype(jnp.float32)
  return y.astype(x.dtype)


def _rms_norm(x, w):
  """``nn.RMSNorm(dim)`` (eps = the dtype's machine epsilon), in x's dtype as torch does."""
  eps = float(jnp.finfo(x.dtype).eps)
  xf = x.astype(jnp.float32)
  y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + eps)
  return (y.astype(x.dtype) * w.astype(x.dtype))


def _resnet(x, params, name, constrain):
  h = _conv3d(x, params, name + ".shortcut", constrain) if name + ".shortcut.weight" in params else x
  y = jax.nn.silu(_rms(x, params[name + ".residual.0.gamma"]))
  y = _conv3d(y, params, name + ".residual.2", constrain)
  y = jax.nn.silu(_rms(y, params[name + ".residual.3.gamma"]))
  y = _conv3d(y, params, name + ".residual.6", constrain)
  return y + h


def _attention(x, params, name, constrain):
  b, t, hh, ww, c = x.shape
  y = _rms(x, params[name + ".norm.gamma"])
  qkv = _conv2d_frames(y, params, name + ".to_qkv", constrain)
  q, k, v = jnp.split(qkv.reshape(b * t, hh * ww, 3 * c), 3, -1)
  if name + ".q_norm.weight" in params:
    q = _rms_norm(q, params[name + ".q_norm.weight"])
    k = _rms_norm(k, params[name + ".k_norm.weight"])
  out = jax.nn.dot_product_attention(q[:, :, None], k[:, :, None], v[:, :, None])[:, :, 0]
  out = _conv2d_frames(out.reshape(b, t, hh, ww, c), params, name + ".proj", constrain)
  return x + out


def _resample_up(x, params, name, temporal, constrain):
  if temporal and x.shape[1] > 1:
    y = _conv3d(x[:, 1:], params, name + ".time_conv", constrain)   # [B, T-1, H, W, 2C]
    b, t, hh, ww, c2 = y.shape
    y = y.reshape(b, t, hh, ww, 2, c2 // 2).transpose(0, 1, 4, 2, 3, 5).reshape(b, 2 * t, hh, ww, c2 // 2)
    x = jnp.concatenate([x[:, :1], y], 1)
  x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
  return _conv2d_frames(constrain(x), params, name + ".resample.1", constrain)


def _dup_up(x, out_c, ft, fs):
  """``DupUp3D`` with ``first_chunk``: channels repeated, unfolded into
  ``ft`` frames x ``fs`` x ``fs`` pixels, the first ``ft - 1`` frames dropped."""
  b, t, hh, ww, c = x.shape
  x = jnp.repeat(x, out_c * ft * fs * fs // c, axis=-1)
  x = x.reshape(b, t, hh, ww, out_c, ft, fs, fs).transpose(0, 1, 5, 2, 6, 3, 7, 4)
  x = x.reshape(b, t * ft, hh * fs, ww * fs, out_c)
  return x[:, ft - 1:]


def decode(params, config, latents, *, mesh=None, rows=("cfg", "tensor"), dtype=jnp.bfloat16):
  """normalized latents [B, 48, T, h, w] -> video [B, 3, 1 + 4(T-1), 16h, 16w] in [-1, 1]
  (not clamped, as the official decode)."""
  if mesh is None:
    constrain = lambda t: t
  else:
    spec = NamedSharding(mesh, P(None, None, rows, None, None))
    constrain = lambda t: jax.lax.with_sharding_constraint(t, spec)
  z = latents.astype(dtype)
  mean = params["mean_tensor"].astype(dtype).reshape(1, -1, 1, 1, 1)
  std = params["std_tensor"].astype(dtype).reshape(1, -1, 1, 1, 1)
  x = constrain((z * std + mean).transpose(0, 2, 3, 4, 1))
  x = _conv3d(x, params, "conv2", constrain)
  x = _conv3d(x, params, "decoder.conv1", constrain)
  x = _resnet(x, params, "decoder.middle.0", constrain)
  x = _attention(x, params, "decoder.middle.1", constrain)
  x = _resnet(x, params, "decoder.middle.2", constrain)
  for i, st in enumerate(config["stages"]):
    pre = f"decoder.upsamples.{i}.upsamples."
    main = x
    for j in range(st["res"]):
      main = _resnet(main, params, f"{pre}{j}", constrain)
    if st["up"]:
      main = _resample_up(main, params, f"{pre}{st['res']}", st["temporal"], constrain)
      out_c = main.shape[-1]
      main = main + _dup_up(x, out_c, 2 if st["temporal"] else 1, 2)
    x = main
  x = jax.nn.silu(_rms(x, params["decoder.head.0.gamma"]))
  x = _conv3d(x, params, "decoder.head.2", constrain)                 # [B, T, H, W, 12]
  b, t, hh, ww, _ = x.shape
  x = x.reshape(b, t, hh, ww, 3, 2, 2).transpose(0, 1, 2, 6, 3, 5, 4)  # (c r q): r = w, q = h
  x = x.reshape(b, t, 2 * hh, 2 * ww, 3)
  return x.astype(jnp.float32).transpose(0, 4, 1, 2, 3)


def make(path, mesh=None, dtype=jnp.bfloat16):
  """(params on ``mesh``, config, jitted ``latents -> video``)."""
  params, config = load_omnivae(path, dtype)
  if mesh is not None:
    params = jax.device_put(params, NamedSharding(mesh, P()))
  fn = jax.jit(lambda p, z: decode(p, config, z, mesh=mesh, dtype=dtype))
  return params, config, fn


def _check(path, ref):
  """Against the official decode of random latents (``vae.npz`` from omni_reference.py)."""
  jax.config.update("jax_default_matmul_precision", "highest")
  r = np.load(ref)
  params, config, fn = make(path, dtype=jnp.float32)
  out = np.asarray(fn(params, jnp.asarray(r["video_latents"])), np.float64)
  want = r["video"].astype(np.float64)
  print(f"config {config}")
  print(f"video decode: {out.shape} vs {want.shape}, rel {np.linalg.norm(out - want) / np.linalg.norm(want):.2e}, "
        f"max {np.abs(out - want).max():.2e}")


if __name__ == "__main__":
  import sys

  _check(sys.argv[1], sys.argv[2])
