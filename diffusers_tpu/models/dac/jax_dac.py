"""The DAC decoder (continuous-latent variant) in JAX, channels-last.

DAC (Descript Audio Codec) as the OpenMOSS audio models use it: a continuous
128-dim latent at sample_rate / hop (48 kHz / 960 = 50 Hz), decoded by

  post_quant_conv (1x1)
  -> Conv1d(latent, decoder_dim, 7)
  -> per rate r: Snake, ConvTranspose1d(stride r, kernel 2r), 3 residual
   units (Snake, Conv1d k7 dilated 1/3/9, Snake, Conv1d k1, + skip);
   channels halve each time
  -> Snake, Conv1d(-> 1, 7), tanh

MOSS-SoundEffect-v2 (decoder_dim 2048, rates 8-5-4-3-2), MOVA and OmniVAE
share this decoder (``vae_128d_48k``). Every convolution is weight-normed in
the checkpoint; the norm is folded into a plain weight at load.

Activations are ``[B, T, C]``. With ``mesh`` the time axis is split over
every chip: each convolution's halo is a neighbour exchange the partitioner
inserts, and nothing else crosses chips (no normalization layers).

  PYTHONPATH=. python -m diffusers_tpu.models.dac.jax_dac --check REF.npz
compares against reference arrays written from the official implementation.
"""
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def _fold_weight_norm(state, prefix):
  """``weight_g * weight_v / ||weight_v||`` (norm over all but dim 0)."""
  v = state[prefix + ".weight_v"].astype(np.float32)
  g = state[prefix + ".weight_g"].astype(np.float32)
  norm = np.sqrt(np.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
  return g * v / norm


def load_state(path):
  """(state_dict as numpy, kwargs) from a DAC ``.pth`` (``{"state_dict",
  "metadata"}``) or an OmniVAE ``state_dict.pt`` (``{"model_state_dict",
  "config", ...}``), read without unpickling code (``weights_only``)."""
  import torch

  blob = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
  if "model_state_dict" in blob:  # an OmniVAE checkpoint: ``audio_vae.*`` + its training config
    pre = "audio_vae."
    state = {k[len(pre):]: v.float().numpy() for k, v in blob["model_state_dict"].items()
             if k.startswith(pre) and (".decoder." in k or "post_quant_conv" in k or
                                       k.startswith(pre + "decoder.") or k.startswith(pre + "post_quant_conv"))}
    audio = blob["config"]["model"]["audio"]
    return state, {"decoder_rates": list(audio["decoder_rates"]), "latent_dim": audio["latent_dim"],
                   "decoder_dim": audio["decoder_dim"], "sample_rate": audio["sample_rate"]}
  state = {k: v.float().numpy() for k, v in blob["state_dict"].items()}
  return state, dict(blob["metadata"]["kwargs"])


def load_state_folder(folder):
  """(state_dict as numpy float32, kwargs) from a diffusers-style folder
  (``config.json`` + ``*.safetensors``), e.g. MOVA's ``audio_vae/``."""
  import glob as _glob
  import json as _json
  import os as _os

  from safetensors.numpy import load_file

  with open(_os.path.join(folder, "config.json")) as fh:
    kwargs = {k: v for k, v in _json.load(fh).items() if not k.startswith("_")}
  state = {}
  for path in sorted(_glob.glob(_os.path.join(folder, "*.safetensors"))):
    state.update({k: np.asarray(v, np.float32) for k, v in load_file(path).items()})
  return state, kwargs


def decoder_params(state, kwargs, dtype=jnp.float32):
  """The decoder's weights, weight norm folded; convs as [K, C_in, C_out]."""
  params = {}

  def conv(name, prefix, transpose=False):
    w = (_fold_weight_norm(state, prefix) if prefix + ".weight_v" in state
      else state[prefix + ".weight"])
    # torch Conv1d: [out, in, k]; ConvTranspose1d: [in, out, k].
    w = w.transpose(2, 0, 1) if transpose else w.transpose(2, 1, 0)
    params[name + ".w"] = jnp.asarray(w, dtype)
    params[name + ".b"] = jnp.asarray(state[prefix + ".bias"], dtype)

  def snake(name, prefix):
    params[name] = jnp.asarray(state[prefix + ".alpha"].reshape(-1), jnp.float32)

  conv("post_quant", "post_quant_conv")
  conv("in", "decoder.model.0")
  rates = kwargs["decoder_rates"]
  for i in range(len(rates)):
    pre = f"decoder.model.{i + 1}.block"
    snake(f"up{i}.snake", f"{pre}.0")
    conv(f"up{i}.conv", f"{pre}.1", transpose=True)
    for j in range(3):
      res = f"{pre}.{j + 2}.block"
      snake(f"up{i}.res{j}.snake1", f"{res}.0")
      conv(f"up{i}.res{j}.conv1", f"{res}.1")
      snake(f"up{i}.res{j}.snake2", f"{res}.2")
      conv(f"up{i}.res{j}.conv2", f"{res}.3")
  last = len(rates) + 1
  snake("out.snake", f"decoder.model.{last}")
  conv("out", f"decoder.model.{last + 1}")
  return params


def _snake(x, alpha):
  a = alpha.astype(jnp.float32)
  xf = x.astype(jnp.float32)
  return (xf + jnp.reciprocal(a + 1e-9) * jnp.square(jnp.sin(a * xf))).astype(x.dtype)


def _conv(x, params, name, constrain, dilation=1):
  w = params[name + ".w"].astype(x.dtype)
  pad = (w.shape[0] - 1) * dilation // 2
  y = jax.lax.conv_general_dilated(
    x, w, (1,), ((pad, pad),), rhs_dilation=(dilation,),
    dimension_numbers=("NWC", "WIO", "NWC"))
  return constrain(y + params[name + ".b"].astype(x.dtype))


def _conv_transpose(x, params, name, stride, constrain):
  """torch ConvTranspose1d(k=2s, stride=s, padding=ceil(s/2), output_padding=s%2):
  exactly s times longer."""
  w = params[name + ".w"].astype(x.dtype)  # [K, C_in, C_out]
  k = w.shape[0]
  pad = math.ceil(stride / 2)
  lo = k - 1 - pad
  hi = k - 1 - pad + stride % 2
  y = jax.lax.conv_general_dilated(
    x, w[::-1], (1,), ((lo, hi),), lhs_dilation=(stride,),
    dimension_numbers=("NWC", "WIO", "NWC"))
  return constrain(y + params[name + ".b"].astype(x.dtype))


def decode(params, latents, *, rates, mesh=None, time_axes=("cfg", "tensor"),
     dtype=jnp.float32):
  """latents [B, latent_dim, T] (as the DiT emits them) -> audio [B, 1, T * hop]."""
  if mesh is None:
    constrain = lambda t: t
  else:
    spec = NamedSharding(mesh, P(None, time_axes, None))
    constrain = lambda t: jax.lax.with_sharding_constraint(t, spec)
  x = constrain(latents.transpose(0, 2, 1).astype(dtype))
  x = _conv(x, params, "post_quant", constrain)
  x = _conv(x, params, "in", constrain)
  for i, rate in enumerate(rates):
    x = _snake(x, params[f"up{i}.snake"])
    x = _conv_transpose(x, params, f"up{i}.conv", rate, constrain)
    for j, dilation in enumerate((1, 3, 9)):
      y = _snake(x, params[f"up{i}.res{j}.snake1"])
      y = _conv(y, params, f"up{i}.res{j}.conv1", constrain, dilation)
      y = _snake(y, params[f"up{i}.res{j}.snake2"])
      y = _conv(y, params, f"up{i}.res{j}.conv2", constrain)
      x = x + y
  x = _snake(x, params["out.snake"])
  x = jnp.tanh(_conv(x, params, "out", constrain).astype(jnp.float32))
  return x.transpose(0, 2, 1)


def make_decoder(path, mesh=None, dtype=jnp.float32):
  """(params on ``mesh``, jitted ``latents -> audio``, kwargs) for a DAC ``.pth``
  file or a diffusers-style folder of safetensors."""
  import os as _os

  state, kwargs = load_state_folder(path) if _os.path.isdir(path) else load_state(path)
  params = decoder_params(state, kwargs, dtype)
  if mesh is not None:
    params = jax.device_put(params, NamedSharding(mesh, P()))
  fn = jax.jit(functools.partial(decode, rates=tuple(kwargs["decoder_rates"]),
                 mesh=mesh, dtype=dtype))
  return params, fn, kwargs


def _check(args):
  """Against arrays from the official decoder: {"latents", "audio", "path"}."""
  ref = np.load(args.check)
  jax.config.update("jax_default_matmul_precision", "highest")
  params, fn, _ = make_decoder(str(ref["path"]))
  out = np.asarray(fn(params, jnp.asarray(ref["latents"])), np.float64)
  want = ref["audio"].astype(np.float64)
  err = np.abs(out - want)
  snr = 10 * np.log10(np.sum(want ** 2) / max(np.sum((out - want) ** 2), 1e-30))
  print(f"shape {out.shape} vs {want.shape}; max |diff| {err.max():.2e}, SNR {snr:.1f} dB")


if __name__ == "__main__":
  import argparse

  p = argparse.ArgumentParser()
  p.add_argument("--check", required=True)
  _check(p.parse_args())
