"""MiniMax-H3's BigVGAN audio decoder in plain JAX.

Same math as diffusers' `AutoencoderKLMiniMaxH3Audio.decode`, laid out for the
TPU. Run through torchax, 52% of the 0.53 s went to layout copies around the
depthwise transposed convolutions and replicate pads of the alias-free
activations. Here each activation (2x upsample -> SnakeBeta -> 2x downsample)
is written in polyphase form: the two phases of the upsampled signal are
filtered, activated and filtered back without ever being interleaved, so the
whole activation is shifted slices and elementwise math on a `(B, C, T)`
array, which XLA fuses into one pass.
"""

import numpy as np


def _np(t):
  return t.detach().float().numpy()


def _w(m):
  """A weight-normalized conv's effective weight (`weight` itself is only set by a forward hook)."""
  import torch
  if hasattr(m, "weight_g"):
    return _np(torch._weight_norm(m.weight_v, m.weight_g, 0))
  return _np(m.weight)


def load_params(vae):
  """`(arrays, spec)` from a loaded `AutoencoderKLMiniMaxH3Audio`, weight norm
  folded in. `arrays` is a pytree of the learned weights; `spec` holds the
  static layout (strides, dilations, paddings) and the fixed resampling taps."""
  dec = vae.decoder
  a = {"in_w": _np(vae.dec_in_proj.weight), "in_b": _np(vae.dec_in_proj.bias),
       "pre_w": _w(dec.conv_pre), "pre_b": _np(dec.conv_pre.bias),
       "post_w": _w(dec.conv_post), "ups": [], "blocks": []}
  spec = {"ups": [], "blocks": [], "num_kernels": dec.num_kernels}
  first = dec.activation_post
  up, down = _np(first.upsample.filter).reshape(-1), _np(first.downsample.lowpass.filter).reshape(-1)
  spec["up"], spec["down"] = tuple(float(v) for v in up), tuple(float(v) for v in down)

  def act(m):
    # Every activation uses the same fixed Kaiser-sinc taps; only alpha/beta are learned.
    assert np.array_equal(_np(m.upsample.filter).reshape(-1), up)
    assert np.array_equal(_np(m.downsample.lowpass.filter).reshape(-1), down)
    return {"alpha": np.exp(_np(m.act.alpha)), "inv_beta": 1.0 / (np.exp(_np(m.act.beta)) + 1e-9)}

  for u in dec.ups:
    c = u[0]
    a["ups"].append({"w": _w(c), "b": _np(c.bias)})
    spec["ups"].append((c.stride[0], c.padding[0]))
  for blk in dec.resblocks:
    acts = [act(m) for m in blk.activations]
    a["blocks"].append({
        "convs1": [(_w(c), _np(c.bias)) for c in blk.convs1],
        "convs2": [(_w(c), _np(c.bias)) for c in blk.convs2],
        "acts1": acts[::2], "acts2": acts[1::2]})
    spec["blocks"].append(([(c.dilation[0], c.padding[0]) for c in blk.convs1],
                           [(c.dilation[0], c.padding[0]) for c in blk.convs2]))
  a["act_post"] = act(dec.activation_post)
  return a, spec


def _shift(xp, pad, off, n):
  """`x[t + off]` for t in [0, n), from `xp` = `x` padded by `pad` on the left."""
  return xp[..., pad + off:pad + off + n]


def _activation(x, a, f, g):
  """Alias-free SnakeBeta (`MiniMaxH3AudioActivation1d`, ratio 2, 12 taps), polyphase."""
  import jax.numpy as jnp
  t = x.shape[-1]
  # Upsample: phase 0 / 1 of the 2T-long signal, from x with edge replication.
  xp = jnp.pad(x, ((0, 0), (0, 0), (3, 3)), mode="edge")
  y0 = 2.0 * sum(float(f[2 * j + 1]) * _shift(xp, 3, 2 - j, t) for j in range(6))
  y1 = 2.0 * sum(float(f[2 * j]) * _shift(xp, 3, 3 - j, t) for j in range(6))
  alpha = a["alpha"][None, :, None]
  inv_beta = a["inv_beta"][None, :, None]
  z0 = y0 + inv_beta * jnp.sin(alpha * y0) ** 2
  z1 = y1 + inv_beta * jnp.sin(alpha * y1) ** 2
  # Downsample (stride 2, replicate pad 5 left / 6 right in the 2T domain):
  # samples before the start are z[0] = z0[0], after the end z[2T-1] = z1[-1].
  first, last = z0[..., :1], z1[..., -1:]
  z0p = jnp.concatenate([first] * 2 + [z0] + [last] * 3, axis=-1)
  z1p = jnp.concatenate([first] * 3 + [z1] + [last] * 2, axis=-1)
  out = 0.0
  for k in range(12):
    if k % 2:
      out = out + float(g[k]) * _shift(z0p, 2, (k - 5) // 2, t)
    else:
      out = out + float(g[k]) * _shift(z1p, 3, (k - 6) // 2, t)
  return out


def _conv(x, w, b, dilation=1, padding=0, precision=None):
  import jax
  y = jax.lax.conv_general_dilated(
      x, w, (1,), [(padding, padding)], rhs_dilation=(dilation,),
      dimension_numbers=("NCH", "OIH", "NCH"), precision=precision)
  return y if b is None else y + b[None, :, None]


def _conv_transpose(x, w, b, stride, pad, precision=None):
  import jax
  import jax.numpy as jnp
  k = w.shape[-1]
  rhs = jnp.flip(jnp.swapaxes(w, 0, 1), -1)        # (C_in, C_out, K) -> OIH, flipped
  y = jax.lax.conv_general_dilated(
      x, rhs, (1,), [(k - 1 - pad, k - 1 - pad)], lhs_dilation=(stride,),
      dimension_numbers=("NCH", "OIH", "NCH"), precision=precision)
  return y + b[None, :, None]


def decode(arrays, spec, z, precision="highest"):
  """`z`: denormalized latents `(B, 32, T)` -> waveform `(B, 1, 800 * T)` in [-1, 1]."""
  import jax.numpy as jnp
  A, f, g = arrays, spec["up"], spec["down"]
  x = _conv(z, A["in_w"], A["in_b"], precision=precision)
  x = _conv(x, A["pre_w"], A["pre_b"], padding=3, precision=precision)
  nk = spec["num_kernels"]
  for i, (up, (stride, pad)) in enumerate(zip(A["ups"], spec["ups"])):
    x = _conv_transpose(x, up["w"], up["b"], stride, pad, precision)
    acc = None
    for j in range(nk):
      blk, (geo1, geo2) = A["blocks"][i * nk + j], spec["blocks"][i * nk + j]
      h = x
      for (w1, b1), (w2, b2), (d1, p1), (d2, p2), a1, a2 in zip(
          blk["convs1"], blk["convs2"], geo1, geo2, blk["acts1"], blk["acts2"]):
        r = _conv(_activation(h, a1, f, g), w1, b1, d1, p1, precision)
        r = _conv(_activation(r, a2, f, g), w2, b2, d2, p2, precision)
        h = h + r
      acc = h if acc is None else acc + h
    x = acc / nk
  x = _activation(x, A["act_post"], f, g)
  x = _conv(x, A["post_w"], None, padding=3, precision=precision)
  return jnp.clip(x, -1.0, 1.0)
