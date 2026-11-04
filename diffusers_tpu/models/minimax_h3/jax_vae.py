"""MiniMax-H3's video VAE (decoder and encoder) in native JAX.

The DiT lives in `jax_dit.py`. H3's decoder is not a convolutional stack
but a **36-layer vision transformer** -- 32 heads of 64, 2048 wide -- over
`frames x height x width` patch tokens, plus four register tokens and a
CLS token that are appended, carried through every layer, and dropped at
the end. The encoder is convolutional and is only needed for `ref2va`'s
image conditioning.

Two details in the reference both change the output:

* every RMSNorm and the Q/K norms run in **float32** whatever the
  compute dtype is, then cast back;
* the residual branches are scaled by learned per-channel vectors
  (`scale1`, `scale2`) initialised at zero, so a block that ignores
  them is not a small error -- it is the whole residual.

The rotary embedding is also its own thing: coordinates are normalised
to `[-1, 1)` per axis, scaled by `2*pi`, the three axes' angles are
concatenated and then *tiled twice*, so the first `rope_dim_ratio`
fraction of each head is rotated as one block rather than in pairs.
"""

import functools
import math

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
  """RMS norm that always reduces in float32.

  The reference's q and k norms are `elementwise_affine=False`; use
  `affine=False` for them, or the model gains 72 parameters that no
  checkpoint tensor fills.
  """

  def __init__(self, dim, eps=1e-5, *, affine=True, dtype=jnp.float32,
               rngs=None):
    self.weight = nnx.Param(jnp.ones((dim,), dtype)) if affine else None
    self.eps = eps

  def __call__(self, x):
    out = x.astype(jnp.float32)
    out = out * jax.lax.rsqrt(
        jnp.mean(jnp.square(out), axis=-1, keepdims=True) + self.eps)
    if self.weight is not None:
      out = out * self.weight[...].astype(jnp.float32)
    return out.astype(x.dtype)


class LayerNorm(nnx.Module):
  def __init__(self, dim, eps=1e-6, *, dtype=jnp.float32, rngs=None):
    self.weight = nnx.Param(jnp.ones((dim,), dtype))
    self.bias = nnx.Param(jnp.zeros((dim,), dtype))
    self.eps = eps

  def __call__(self, x):
    out = x.astype(jnp.float32)
    out = (out - jnp.mean(out, -1, keepdims=True)) * jax.lax.rsqrt(
        jnp.var(out, -1, keepdims=True) + self.eps)
    out = out * self.weight[...].astype(jnp.float32) + self.bias[...].astype(
        jnp.float32)
    return out.astype(x.dtype)


class Linear(nnx.Module):
  def __init__(self, inp, out, *, bias=True, dtype=jnp.bfloat16, rngs=None):
    self.weight = nnx.Param(jnp.zeros((out, inp), dtype))
    self.bias = nnx.Param(jnp.zeros((out,), dtype)) if bias else None

  def __call__(self, x):
    y = jnp.einsum("...i,oi->...o", x, self.weight[...].astype(x.dtype))
    return y if self.bias is None else y + self.bias[...].astype(x.dtype)


def rope_angles(position_ids, dim, theta=100.0, num_axes=3):
  """`(cos, sin)` for the decoder's three-axis rotary.

  `position_ids` is `(batch, tokens, 3)` already normalised to
  `[-1, 1)`. The reference concatenates the three axes' angles and
  then tiles the result twice, which is what makes the rotation a
  single split-halves rotation over `dim` channels rather than three
  independent ones.
  """
  inv_freq = 1.0 / (theta ** jnp.arange(0, 1, 2 * num_axes / dim,
                                        dtype=jnp.float32))
  angles = 2.0 * math.pi * position_ids[..., None] * inv_freq
  angles = angles.reshape(*angles.shape[:2], -1)
  angles = jnp.tile(angles, 2)[:, :, None, :]
  return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
  """Rotate the leading `cos.shape[-1]` channels of each head."""
  width = cos.shape[-1]
  turn, keep = x[..., :width], x[..., width:]
  first, second = jnp.split(turn, 2, axis=-1)
  rotated = jnp.concatenate([-second, first], axis=-1)
  out = turn * cos.astype(x.dtype) + rotated * sin.astype(x.dtype)
  return jnp.concatenate([out, keep], axis=-1) if keep.shape[-1] else out


class Attention(nnx.Module):
  def __init__(self, dim, heads, head_dim, eps=1e-5, *, bias=True,
               dtype=jnp.bfloat16, rngs=None):
    inner = heads * head_dim
    self.heads = heads
    self.head_dim = head_dim
    self.to_q = Linear(dim, inner, bias=bias, dtype=dtype, rngs=rngs)
    self.to_k = Linear(dim, inner, bias=bias, dtype=dtype, rngs=rngs)
    self.to_v = Linear(dim, inner, bias=bias, dtype=dtype, rngs=rngs)
    self.norm_q = RMSNorm(head_dim, eps, affine=False, rngs=rngs)
    self.norm_k = RMSNorm(head_dim, eps, affine=False, rngs=rngs)
    self.to_out = Linear(inner, dim, bias=bias, dtype=dtype, rngs=rngs)

  def __call__(self, x, rope, sdpa=None):
    shape = x.shape[:-1] + (self.heads, self.head_dim)
    q = self.norm_q(self.to_q(x).reshape(shape))
    k = self.norm_k(self.to_k(x).reshape(shape))
    v = self.to_v(x).reshape(shape)
    if rope is not None:
      q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    if sdpa is not None:
      out = sdpa(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                 v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    else:
      scale = 1.0 / math.sqrt(self.head_dim)
      scores = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * scale
      out = jnp.einsum("bhqk,bkhd->bqhd",
                       jax.nn.softmax(scores, -1).astype(v.dtype), v)
    return self.to_out(out.reshape(x.shape[:-1] + (-1,)))


class FeedForward(nnx.Module):
  """diffusers' `FeedForward(mult, activation_fn="swiglu")`.

  The projection emits gate and value in one tensor; splitting it the
  other way round silently produces a different function.
  """

  def __init__(self, dim, mult=4, *, bias=True, dtype=jnp.bfloat16,
               rngs=None):
    inner = dim * mult
    self.proj = Linear(dim, inner * 2, bias=bias, dtype=dtype, rngs=rngs)
    self.out = Linear(inner, dim, bias=bias, dtype=dtype, rngs=rngs)

  def __call__(self, x):
    # diffusers' SwiGLU is `value, gate = chunk(2); value * silu(gate)`:
    # the first half is the value, the second the gate.
    value, gate = jnp.split(self.proj(x), 2, axis=-1)
    return self.out(value * jax.nn.silu(gate))


class Block(nnx.Module):
  def __init__(self, dim, heads, head_dim, ffn_mult=4, eps=1e-5, *, bias=True,
               dtype=jnp.bfloat16, rngs=None):
    self.norm1 = RMSNorm(dim, eps, rngs=rngs)
    self.attn = Attention(dim, heads, head_dim, eps, bias=bias, dtype=dtype,
                          rngs=rngs)
    self.scale1 = nnx.Param(jnp.zeros((dim,), jnp.float32))
    self.norm2 = RMSNorm(dim, eps, rngs=rngs)
    self.ff = FeedForward(dim, ffn_mult, bias=bias, dtype=dtype, rngs=rngs)
    self.scale2 = nnx.Param(jnp.zeros((dim,), jnp.float32))

  def __call__(self, x, rope, sdpa=None):
    h = self.norm1(x)
    x = x + self.attn(h, rope, sdpa) * self.scale1[...].astype(x.dtype)
    h = self.norm2(x)
    return x + self.ff(h) * self.scale2[...].astype(x.dtype)


class PostQuantConv(nnx.Module):
  """The 1x1x1 conv `decode` applies before the decoder sees anything.

  A single pointwise projection, but required: skipping it changes the
  output substantially.
  """

  def __init__(self, channels, *, dtype=jnp.float32, rngs=None):
    self.weight = nnx.Param(jnp.zeros((channels, channels, 1, 1, 1), dtype))
    self.bias = nnx.Param(jnp.zeros((channels,), dtype))

  def __call__(self, z):
    w = self.weight[...].reshape(self.weight[...].shape[:2]).astype(z.dtype)
    out = jnp.einsum("bcfhw,oc->bofhw", z, w)
    return out + self.bias[...].astype(z.dtype).reshape(1, -1, 1, 1, 1)


class ViTDecoder(nnx.Module):
  """The decoder: patch tokens in, pixels out."""

  def __init__(self, *, in_channels=24, out_channels=3,
               num_layers=36, num_attention_heads=32, attention_head_dim=64,
               num_register_tokens=4, patch_size=16, patch_size_t=4,
               ffn_mult=4, norm_eps=1e-5, rope_dim_ratio=0.75,
               rope_theta=100.0, dtype=jnp.bfloat16, rngs=None):
    # The reference derives the width rather than taking it.
    dim = num_attention_heads * attention_head_dim
    self.patch_size = patch_size
    self.patch_size_t = patch_size_t
    self.out_channels = out_channels
    self.num_register_tokens = num_register_tokens
    self.rope_dim = int(attention_head_dim * rope_dim_ratio)
    self.rope_theta = rope_theta
    self.proj_in = Linear(in_channels, dim, dtype=dtype, rngs=rngs)
    self.register_tokens = nnx.Param(
        jnp.zeros((1, num_register_tokens, dim), dtype))
    self.transformer_blocks = nnx.List([
        Block(dim, num_attention_heads, attention_head_dim, ffn_mult,
              dtype=dtype, rngs=rngs)
        for _ in range(num_layers)])
    self.norm_out = LayerNorm(dim, norm_eps, rngs=rngs)
    self.proj_out = Linear(
        dim, out_channels * patch_size_t * patch_size * patch_size,
        dtype=dtype, rngs=rngs)

  def __call__(self, latents, sdpa=None):
    batch, channels, frames, height, width = latents.shape
    x = jnp.transpose(latents, (0, 2, 3, 4, 1)).reshape(
        batch, frames * height * width, channels)
    x = self.proj_in(x)
    patches = x.shape[1]

    # Register tokens and a zero CLS token ride along and are dropped
    # after `proj_out`; they are not positioned, so their coordinates
    # are zeros.
    tail = jnp.concatenate([
        jnp.broadcast_to(self.register_tokens[...].astype(x.dtype),
                         (batch,) + self.register_tokens[...].shape[1:]),
        jnp.zeros((batch, 1, x.shape[-1]), x.dtype),
    ], axis=1)
    x = jnp.concatenate([x, tail], axis=1)

    grids = [2.0 * (jnp.arange(0.5, size, dtype=jnp.float32) / size) - 1.0
             for size in (frames, height, width)]
    coords = jnp.stack(jnp.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    coords = jnp.broadcast_to(coords, (batch,) + coords.shape)
    coords = jnp.concatenate([
        coords,
        jnp.zeros((batch, self.num_register_tokens + 1, 3), coords.dtype),
    ], axis=1)
    rope = rope_angles(coords, self.rope_dim, self.rope_theta)

    for block in self.transformer_blocks:
      x = block(x, rope, sdpa)

    x = self.proj_out(self.norm_out(x))[:, :patches]
    x = x.reshape(batch, frames, height, width, self.out_channels,
                  self.patch_size_t, self.patch_size, self.patch_size)
    x = jnp.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7))
    return x.reshape(batch, self.out_channels,
                     frames * self.patch_size_t,
                     height * self.patch_size,
                     width * self.patch_size)


CONFIG = dict(in_channels=24, out_channels=3, num_layers=36,
              num_attention_heads=32, attention_head_dim=64,
              num_register_tokens=4, patch_size=16, patch_size_t=4,
              rope_dim_ratio=0.75, norm_eps=1e-5)

_RENAMES = (
    (".to_out.0.", ".to_out."),
    (".ff.net.0.proj.", ".ff.proj."),
    (".ff.net.2.", ".ff.out."),
)


_COLUMN = ("to_q", "to_k", "to_v", "proj")      # ff.proj emits gate|value
_ROW = ("to_out", "out")


def decoder_spec(name, ndim):
  """Tensor-parallel layout for the decoder, matching the DiT's.

  Attention heads are split across the `tp` axis, as in the transformer.
  Without it every chip runs the whole decoder (119 s vs 10.7 s on 8
  chips).
  """
  from jax.sharding import PartitionSpec as P

  parent = name.split(".")[-2] if "." in name else ""
  if ndim == 2 and parent in _COLUMN:
    return P("tp", None)
  if ndim == 2 and parent in _ROW:
    return P(None, "tp")
  if ndim == 1 and parent in _COLUMN:
    return P("tp")
  return P()


def load_decoder(model, params, mesh=None, spec_for=None):
  """Fill the nnx decoder from a dict of arrays keyed by module path."""
  filled, skipped = 0, []
  prefix = "decoder."

  def resolve(name):
    node = model
    for part in name.split(".")[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    return getattr(node, name.split(".")[-1])

  for key, value in params.items():
    if not key.startswith(prefix):
      continue
    name = key[len(prefix):]
    for old, new in _RENAMES:
      name = name.replace(old, new)
    try:
      param = resolve(name)
    except AttributeError:
      skipped.append(key)
      continue
    want = param.value if hasattr(param, "value") else param[...]
    value = jnp.asarray(value).astype(want.dtype).reshape(want.shape)
    if mesh is not None:
      spec = (spec_for or decoder_spec)(name, value.ndim)
      value = jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
    if hasattr(param, "value"):
      param.value = value
    else:
      param[...] = value
    filled += 1

  # Fail fast on an incomplete load; unfilled parameters otherwise only
  # surface deep inside the forward pass.
  import jax as _jax
  from flax import nnx as _nnx

  expected = len(_jax.tree.leaves(_nnx.state(model, _nnx.Param)))
  if filled != expected:
    raise ValueError(
        f"filled {filled} of {expected} decoder parameters"
        + (f"; unresolved {skipped[:4]}" if skipped else ""))
  return filled, skipped


# --------------------------------------------------------------------------
# the encoder, which `ref2va` needs and `t2va` does not
#
# Where the decoder is a transformer, the encoder is an ordinary 3D
# convolutional stack -- six levels, two residual blocks each, 116
# tensors. Three details in it are not ordinary:
#
#   * the convolutions are **causal in time**: `kernel_t - 1` zero
#     frames are prepended and none appended, so a frame never sees the
#     future. Padding symmetrically instead would be a silent change.
#   * spatial padding is **reflect**, not zeros.
#   * the group norm is applied **per latent frame**: the temporal axis
#     is folded into the batch so statistics never mix across frames.
# --------------------------------------------------------------------------


class GroupNorm3d(nnx.Module):
  """Group norm whose statistics never cross a frame boundary."""

  def __init__(self, groups, channels, eps=1e-6, *, dtype=jnp.float32,
               rngs=None):
    self.groups = groups
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((channels,), dtype))
    self.bias = nnx.Param(jnp.zeros((channels,), dtype))

  def __call__(self, x):
    b, c, f, h, w = x.shape
    out = x.astype(jnp.float32).transpose(0, 2, 1, 3, 4).reshape(
        b * f, self.groups, c // self.groups * h * w)
    mean = jnp.mean(out, -1, keepdims=True)
    var = jnp.var(out, -1, keepdims=True)
    out = (out - mean) * jax.lax.rsqrt(var + self.eps)
    out = out.reshape(b, f, c, h, w).transpose(0, 2, 1, 3, 4)
    scale = self.weight[...].astype(jnp.float32).reshape(1, -1, 1, 1, 1)
    shift = self.bias[...].astype(jnp.float32).reshape(1, -1, 1, 1, 1)
    return (out * scale + shift).astype(x.dtype)


class CausalConv3d(nnx.Module):
  """Reflect-padded in space, causal in time."""

  def __init__(self, in_c, out_c, kernel, stride=1, spatial_padding=0,
               temporal_padding=0, *, dtype=jnp.float32, rngs=None):
    k = (kernel, kernel, kernel) if isinstance(kernel, int) else kernel
    s = (stride, stride, stride) if isinstance(stride, int) else stride
    self.weight = nnx.Param(jnp.zeros((out_c, in_c) + tuple(k), dtype))
    self.bias = nnx.Param(jnp.zeros((out_c,), dtype))
    self.stride = tuple(s)
    self.spatial_padding = spatial_padding
    self.temporal_padding = temporal_padding

  def __call__(self, x):
    if self.spatial_padding:
      p = self.spatial_padding
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (p, p), (p, p)), mode="reflect")
    if self.temporal_padding:
      x = jnp.pad(x, ((0, 0), (0, 0), (self.temporal_padding, 0), (0, 0),
                      (0, 0)))
    y = jax.lax.conv_general_dilated(
        x.astype(jnp.float32), self.weight[...].astype(jnp.float32),
        window_strides=self.stride, padding="VALID",
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))
    return (y + self.bias[...].astype(jnp.float32).reshape(
        1, -1, 1, 1, 1)).astype(x.dtype)


class ResnetBlock3d(nnx.Module):
  def __init__(self, in_c, out_c, groups=32, eps=1e-6, *, dtype=jnp.float32,
               rngs=None):
    self.norm1 = GroupNorm3d(groups, in_c, eps, dtype=dtype, rngs=rngs)
    self.conv1 = CausalConv3d(in_c, out_c, 3, spatial_padding=1,
                              temporal_padding=2, dtype=dtype, rngs=rngs)
    self.norm2 = GroupNorm3d(groups, out_c, eps, dtype=dtype, rngs=rngs)
    self.conv2 = CausalConv3d(out_c, out_c, 3, spatial_padding=1,
                              temporal_padding=2, dtype=dtype, rngs=rngs)
    self.conv_shortcut = (
        CausalConv3d(in_c, out_c, 1, dtype=dtype, rngs=rngs)
        if in_c != out_c else None)

  def __call__(self, x):
    h = self.conv1(jax.nn.silu(self.norm1(x)))
    h = self.conv2(jax.nn.silu(self.norm2(h)))
    return h + (x if self.conv_shortcut is None else self.conv_shortcut(x))


class Downsample3d(nnx.Module):
  """Strided causal conv; the odd spatial pad goes on the far side only."""

  def __init__(self, channels, spatial_stride, temporal_stride, *,
               dtype=jnp.float32, rngs=None):
    self.spatial_stride = spatial_stride
    self.conv = CausalConv3d(channels, channels, 3,
                             stride=(temporal_stride, spatial_stride,
                                     spatial_stride),
                             temporal_padding=2, dtype=dtype, rngs=rngs)

  def __call__(self, x):
    if self.spatial_stride == 2:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1), (0, 1)), mode="reflect")
    return self.conv(x)


class DownBlock3d(nnx.Module):
  def __init__(self, in_c, out_c, layers, spatial_stride, temporal_stride,
               groups=32, eps=1e-6, *, dtype=jnp.float32, rngs=None):
    self.resnets = nnx.List([
        ResnetBlock3d(in_c if i == 0 else out_c, out_c, groups, eps,
                      dtype=dtype, rngs=rngs) for i in range(layers)])
    self.downsamplers = nnx.List(
        [Downsample3d(out_c, spatial_stride, temporal_stride, dtype=dtype,
                      rngs=rngs)]
        if spatial_stride > 1 or temporal_stride > 1 else [])

  def __call__(self, x):
    for r in self.resnets:
      x = r(x)
    for d in self.downsamplers:
      x = d(x)
    return x


class Encoder3d(nnx.Module):
  """`ref2va`'s image conditioning path. 116 tensors."""

  def __init__(self, *, in_channels=3, latent_channels=24,
               block_out_channels=(128, 256, 256, 512, 512, 1024),
               layers_per_block=2,
               spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
               temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
               norm_num_groups=32, norm_eps=1e-6, dtype=jnp.float32,
               rngs=None):
    self.conv_in = CausalConv3d(in_channels, block_out_channels[0], 3,
                                spatial_padding=1, temporal_padding=2,
                                dtype=dtype, rngs=rngs)
    blocks, prev = [], block_out_channels[0]
    for i, out_c in enumerate(block_out_channels):
      blocks.append(DownBlock3d(prev, out_c, layers_per_block,
                                spatial_downsample_factors[i],
                                temporal_downsample_factors[i],
                                norm_num_groups, norm_eps, dtype=dtype,
                                rngs=rngs))
      prev = out_c
    self.down_blocks = nnx.List(blocks)
    self.norm_out = GroupNorm3d(norm_num_groups, prev, norm_eps, dtype=dtype,
                                rngs=rngs)
    self.conv_out = CausalConv3d(prev, latent_channels * 2, 3,
                                 spatial_padding=1, temporal_padding=2,
                                 dtype=dtype, rngs=rngs)

  def __call__(self, x):
    x = self.conv_in(x)
    for block in self.down_blocks:
      x = block(x)
    return self.conv_out(jax.nn.silu(self.norm_out(x)))


def encoder_config(vae_dir):
  """The encoder's shape, read off the checkpoint rather than assumed."""
  import json
  import os

  with open(os.path.join(vae_dir, "config.json")) as f:
    cfg = json.load(f)
  return dict(
      in_channels=cfg["in_channels"],
      latent_channels=cfg["latent_channels"],
      block_out_channels=tuple(cfg["block_out_channels"]),
      layers_per_block=cfg["layers_per_block"],
      spatial_downsample_factors=tuple(cfg["spatial_downsample_factors"]),
      temporal_downsample_factors=tuple(cfg["temporal_downsample_factors"]),
      norm_num_groups=cfg["norm_num_groups"],
      norm_eps=cfg["norm_eps"],
  )


def load_encoder(model, params, mesh=None, spec_for=None):
  """Fill the conv encoder from a dict of arrays keyed by module path.

  Same contract as `load_decoder`: every parameter must be written, and an
  incomplete load raises.
  """
  filled, skipped = 0, []
  prefix = "encoder."

  def resolve(name):
    node = model
    for part in name.split(".")[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    return getattr(node, name.split(".")[-1])

  for key, value in params.items():
    if not key.startswith(prefix):
      continue
    name = key[len(prefix):]
    try:
      param = resolve(name)
    except (AttributeError, IndexError, KeyError):
      skipped.append(key)
      continue
    want = param.value if hasattr(param, "value") else param[...]
    value = jnp.asarray(value).astype(want.dtype).reshape(want.shape)
    if mesh is not None and spec_for is not None:
      value = jax.device_put(
          value, jax.sharding.NamedSharding(mesh, spec_for(name, value.ndim)))
    if hasattr(param, "value"):
      param.value = value
    else:
      param[...] = value
    filled += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if filled != expected:
    raise ValueError(
        f"filled {filled} of {expected} encoder parameters"
        + (f"; unresolved {skipped[:4]}" if skipped else ""))
  return filled, skipped


# --------------------------------------------------------------------------
# the chunking around the encoder, which `ref2va` needs
#
# `Encoder3d` is only the inner module. The reference's `encode` is
# `quant_conv(encoder(x))` run over 17-frame chunks, with the last
# `token_drop` latent frames dropped.
# --------------------------------------------------------------------------

CLIP_LENGTH = 17        # vae.config.clip_length
TOKEN_DROP = 3          # vae.config.token_drop


def load_quant_conv(model, params, prefix="quant_conv."):
  """Fill a `PostQuantConv` from `quant_conv.*` (48 channels) or
  `post_quant_conv.*` (24). Both are the same 1x1x1 pointwise conv."""
  seen = 0
  for key, value in params.items():
    if not key.startswith(prefix):
      continue
    # Defensive: never let `post_quant_conv.*` keys fill `quant_conv`.
    if prefix == "quant_conv." and key.startswith("post_quant_conv."):
      continue
    leaf = getattr(model, key[len(prefix):])
    want = leaf.value if hasattr(leaf, "value") else leaf[...]
    leaf.value = jnp.asarray(value).astype(want.dtype).reshape(want.shape)
    seen += 1
  if seen != 2:
    raise ValueError(f"{prefix} expected 2 tensors, found {seen}")
  return seen


def encode_video(encoder, quant_conv, x, clip_length=CLIP_LENGTH,
                 token_drop=TOKEN_DROP):
  """`AutoencoderKLMiniMaxH3._encode`, without the spatial tiling.

  A single frame has no temporal extent and goes through on its own.
  The reference is explicit that padding it up to `clip_length` by
  repetition instead is *not* the conditioning the model was trained
  with -- it would return `clip_length // 4 - token_drop` latent
  frames rather than one.
  """
  def one(clip):
    return quant_conv(encoder(clip))

  frames = x.shape[2]
  if frames == 1:
    return one(x)
  pad = (-frames) % clip_length
  if pad:
    x = jnp.concatenate([x, jnp.repeat(x[:, :, -1:], pad, axis=2)], axis=2)
  moments = jnp.concatenate(
      [one(x[:, :, i * clip_length:(i + 1) * clip_length])
       for i in range(x.shape[2] // clip_length)], axis=2)
  return moments[:, :, :-token_drop] if token_drop > 0 else moments


# --------------------------------------------------------------------------
# spatial tiling
# --------------------------------------------------------------------------
# `AutoencoderKLMiniMaxH3` tiles the decode by default -- 256x256 pixel
# tiles, 64-pixel minimum overlap, linearly blended -- and its docstring
# says the released frames are the blended ones, so tiling is part of the
# model's output, not just a memory optimisation. Decoding whole frames
# matches the reference only when the frame fits in a single tile.

SPATIAL_RATIO = 16


def split_tiles(length, tile_size, min_overlap, ratio=SPATIAL_RATIO):
  """`_split_tiles`: tile starts, lengths and overlaps, in pixels."""
  if tile_size >= length:
    return [0], [length], []
  n = -(-length // tile_size)
  while tile_size * n - min_overlap * (n - 1) - length < 0:
    n += 1
  overlaps = [min_overlap] * (n - 1)
  remaining = tile_size * n - sum(overlaps) - length
  for i in range(remaining // ratio):
    overlaps[i % (n - 1)] += ratio
  starts = [0]
  for i in range(n - 1):
    starts.append(starts[-1] + tile_size - overlaps[i])
  return starts, [tile_size] * n, overlaps


def blend_axis(a, b, extent, axis):
  """`_blend`: cross-fade `a`'s tail into `b`'s head along `axis`."""
  import jax.numpy as jnp

  extent = min(a.shape[axis], b.shape[axis], extent)
  if extent <= 0:
    return b
  shape = [1] * a.ndim
  shape[axis] = extent
  pos = jnp.arange(extent, dtype=b.dtype).reshape(shape) / extent
  sa = [slice(None)] * a.ndim
  sa[axis] = slice(-extent, None)
  sb = [slice(None)] * b.ndim
  sb[axis] = slice(0, extent)
  head = a[tuple(sa)] * (1.0 - pos) + b[tuple(sb)] * pos
  if extent == b.shape[axis]:
    return head
  rest = [slice(None)] * b.ndim
  rest[axis] = slice(extent, None)
  return jnp.concatenate([head, b[tuple(rest)]], axis=axis)


def tiled_decode(decode_one, z, tile_px=256, overlap_px=64,
                 ratio=SPATIAL_RATIO, batch=0, mesh=None):
  """`_decode_clip` with tiling on: decode each tile, blend, stitch.

  `decode_one` takes a latent tile and returns its pixels.

  `batch` stacks tiles and runs `decode_one` under `vmap` in groups of
  that many. Every tile is the same size, so this is the same arithmetic
  in one program instead of N (28 tiles at 768x1344). 0 decodes tiles one
  at a time; the output is the same either way.
  """
  import jax
  import jax.numpy as jnp

  ys, yl, yo = split_tiles(z.shape[-2] * ratio, tile_px, overlap_px, ratio)
  xs, xl, xo = split_tiles(z.shape[-1] * ratio, tile_px, overlap_px, ratio)
  if not yo and not xo:
    return decode_one(z)
  cuts = [(i_pos, i_len, j_pos, j_len)
          for i_pos, i_len in zip(ys, yl)
          for j_pos, j_len in zip(xs, xl)]
  tiles = [z[..., i_pos // ratio:i_pos // ratio + i_len // ratio,
             j_pos // ratio:j_pos // ratio + j_len // ratio]
           for i_pos, i_len, j_pos, j_len in cuts]
  if mesh is not None and mesh.devices.size == 1 and not batch:
    batch = 4
  if mesh is not None and mesh.devices.size > 1:
    # Tiles split over the chips (weights replicated), the way
    # `tiled_encode(mesh=)` does it: 28 tiles of a 768p chunk -> 4 a chip.
    from jax.sharding import NamedSharding, PartitionSpec as P
    n = len(tiles)
    ndev = mesh.devices.size
    stacked = jnp.stack(tiles + [tiles[-1]] * ((-n) % ndev))
    stacked = jax.device_put(stacked, NamedSharding(mesh, P(mesh.axis_names[0])))
    flat = jax.vmap(decode_one)(stacked)[:n]
    ncol = len(xs)
    rows = [[flat[i * ncol + j] for j in range(ncol)]
            for i in range(len(ys))]
  elif batch:
    stacked = jnp.stack(tiles)
    outs = [jax.vmap(decode_one)(stacked[s:s + batch])
            for s in range(0, stacked.shape[0], batch)]
    flat = jnp.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
    ncol = len(xs)
    rows = [[flat[i * ncol + j] for j in range(ncol)]
            for i in range(len(ys))]
  else:
    rows = [[decode_one(t) for t in tiles[i * len(xs):(i + 1) * len(xs)]]
            for i in range(len(ys))]
  out_rows = []
  for i, row in enumerate(rows):
    out = []
    for j, tile in enumerate(row):
      if i > 0:
        tile = blend_axis(rows[i - 1][j], tile, yo[i - 1], -2)
      if j > 0:
        tile = blend_axis(row[j - 1], tile, xo[j - 1], -1)
      if i < len(rows) - 1:
        tile = tile[..., :-yo[i], :]
      if j < len(row) - 1:
        tile = tile[..., :, :-xo[j]]
      out.append(tile)
    out_rows.append(jnp.concatenate(out, axis=-1))
  return jnp.concatenate(out_rows, axis=-2)


def tiled_encode(encode_one, x, tile_px=256, overlap_px=64,
                 ratio=SPATIAL_RATIO, batch=0, mesh=None):
  """`_encode_clip` with tiling on: encode each pixel tile, blend, stitch.

  `encode_one` takes a pixel tile and returns its moments
  (`quant_conv(encoder(tile))`). The tiles are laid out in pixel space
  and the overlaps mapped onto the latent grid (`overlap // ratio`),
  exactly as the reference does; encoding the whole frame at once is a
  different function, because every GroupNorm takes its statistics per
  tile. A 3584x2048 image is 11x14 tiles of 256.

  `batch` stacks tiles and runs `encode_one` under `vmap` in groups of
  that many -- every tile has the same size.
  """
  import os

  import jax
  import jax.numpy as jnp

  ys, yl, yo = split_tiles(x.shape[-2], tile_px, overlap_px, ratio)
  xs, xl, xo = split_tiles(x.shape[-1], tile_px, overlap_px, ratio)
  if not yo and not xo:
    return encode_one(x)
  tiles = [x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
           for i_pos, i_len in zip(ys, yl) for j_pos, j_len in zip(xs, xl)]
  ncol = len(xs)
  if mesh is not None and mesh.devices.size == 1 and not batch:
    batch = 4
  if mesh is not None and mesh.devices.size > 1:
    # All tiles at once, split over the chips along the tile axis (one
    # chip encoding 154 tiles of a 3584x2048 image takes ~90 s). Padded
    # with copies of the last tile to a multiple of the chip count.
    from jax.sharding import NamedSharding, PartitionSpec as P
    n = len(tiles)
    ndev = mesh.devices.size
    group = int(os.environ.get("DT_ENC_GROUP", "0")) * ndev
    if group and group < n:
      # Groups of DT_ENC_GROUP tiles a chip, one after another under
      # `lax.map`: as one program the 154-tile encode needs 34.7 GB of
      # temporaries at `high` precision.
      sh = NamedSharding(mesh, P(mesh.axis_names[0]))
      stacked = jnp.stack(tiles + [tiles[-1]] * ((-n) % group))
      grouped = stacked.reshape((-1, group) + stacked.shape[1:])
      out = jax.lax.map(
          lambda g: jax.vmap(encode_one)(jax.lax.with_sharding_constraint(g, sh)),
          grouped)
      flat = out.reshape((-1,) + out.shape[2:])[:n]
    else:
      stacked = jnp.stack(tiles + [tiles[-1]] * ((-n) % ndev))
      stacked = jax.device_put(stacked, NamedSharding(mesh, P(mesh.axis_names[0])))
      flat = jax.vmap(encode_one)(stacked)[:n]
    rows = [[flat[i * ncol + j] for j in range(ncol)] for i in range(len(ys))]
  elif batch:
    stacked = jnp.stack(tiles)
    outs = [jax.vmap(encode_one)(stacked[s:s + batch])
            for s in range(0, stacked.shape[0], batch)]
    flat = jnp.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
    rows = [[flat[i * ncol + j] for j in range(ncol)] for i in range(len(ys))]
  else:
    rows = [[encode_one(t) for t in tiles[i * ncol:(i + 1) * ncol]]
            for i in range(len(ys))]
  lyo = [o // ratio for o in yo]
  lxo = [o // ratio for o in xo]
  out_rows = []
  for i, row in enumerate(rows):
    out = []
    for j, tile in enumerate(row):
      if i > 0:
        tile = blend_axis(rows[i - 1][j], tile, lyo[i - 1], -2)
      if j > 0:
        tile = blend_axis(row[j - 1], tile, lxo[j - 1], -1)
      if i < len(rows) - 1:
        tile = tile[..., :-lyo[i], :]
      if j < len(row) - 1:
        tile = tile[..., :, :-lxo[j]]
      out.append(tile)
    out_rows.append(jnp.concatenate(out, axis=-1))
  return jnp.concatenate(out_rows, axis=-2)
