"""Video VAEs for Sand.ai MAGI-2-preview in native JAX / Flax NNX.

Implements both video VAEs used by MAGI-2-preview:

1. :class:`TurboVAED` (decoder-only, ``bfloat16`` compute on TPU, ``float32``
   parameter-free RMSNorms):
   Distilled shallow 3D video decoder matching ``inference/model/turbo_vaed.py``
   and ``TurboV3-Wan22-TinyShallow_7_7.json``. Key architectural details:

   * Non-causal symmetric temporal edge-replication padding ``(k_t - 1) // 2``
     on both temporal ends, and symmetric spatial zero-padding.
   * Parameter-free ``float32`` RMSNorm: ``eps=1e-8`` on ``norm1``, ``norm2``,
     ``norm_up_1``, and the final inline norm; ``eps=resnet_norm_eps`` (default
     ``1e-6``) on shortcut ``norm3`` when ``in_channels != out_channels``.
   * **ReLU quirk**: when ``is_upsampler_modified=True`` (every ResNet inside an
     up-block with ``spatio_temporal_scale=True``), the *first* activation
     before ``conv1`` is ``ReLU`` (NOT ``SiLU``), while the *second* activation
     before ``conv2`` remains ``SiLU``.
   * Supports both standard 3D convolutions and depthwise-separable 3D
     convolutions (``decoder_is_dw_conv`` / ``decoder_dw_kernel_size``),
     including ``1x1x1`` depthwise+pointwise shortcuts when
     ``is_dw_conv=True`` and ``in_channels != out_channels``.
   * Exact inverse ``(c r q)`` unpatchify (``r`` = width subpixel before ``q`` =
     height subpixel) matching ``_turbo_unpatchify``.
   * Exact sliding-window decoding (``first_chunk_size=7, step_size=7``, 1
     latent frame context overlap -> 4 pixel frames trimmed at internal
     boundaries, last-frame tail padding & trimming -> 125 pixel frames for
     ``T=32``, 249 pixel frames for ``T=63``).
   * Exact 48-channel Wan2.2 latent denormalization (``z * std + mean`` in
     ``float32``).

2. :class:`Wan2_2_VAEEncoder` (encoder-only, ``float32``, for I2V reference
   images & multi-frame clips):
   Exact encoder half of ``inference/model/vae2_2.py`` (``Wan2.2_VAE.pth``):

   * Exact ``(c r q)`` ``patchify(2)`` (``r`` before ``q``).
   * Causal 3D convolutions with ``CACHE_T = 2`` feature-cache chunking
     (frame 0 alone, then 4 frames at a time: ``iter_ = 1 + (T - 1) // 4``).
   * ``dim=160, dim_mult=(1, 2, 4, 4), num_res_blocks=2,
     temporal_downsample=(False, True, True), z_dim=48``, ``AvgDown3D``
     shortcuts, single-head per-frame spatial ``AttentionBlock`` in the middle
     block, post-encoder ``conv1`` (``96 -> 96`` 1x1x1 conv), split ``mu``,
     and 48-channel normalization ``(mu - mean) / std``.

All internal activations run in channels-last ``NDHWC`` ``(B, T, H, W, C)``
layout for optimal TPU MXU utilization, with Conv3d weights transposed
``OIDHW -> DHWIO`` and Conv2d weights transposed ``OIHW -> HWIO`` at load time.

Self-test::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.magi2.jax_video_vae
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.magi2.config import (
    Magi2TurboVAEDConfig,
    WAN22_LATENT_MEAN,
    WAN22_LATENT_STD,
)


CACHE_T = 2


# ---------------------------------------------------------------------------
# Shared Helpers (Patchify / Unpatchify / Norms / Weight Transposes)
# ---------------------------------------------------------------------------


def _to_numpy_f32(tensor: Any) -> np.ndarray:
  """Convert a torch.Tensor or array-like to a contiguous float32 ndarray."""
  if isinstance(tensor, np.ndarray):
    return np.ascontiguousarray(tensor, dtype=np.float32)
  # Avoid top-level torch import.
  if hasattr(tensor, "detach"):
    return np.ascontiguousarray(tensor.detach().cpu().float().numpy())
  return np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))


def _transpose_conv3d_oidhw_to_dhwio(w: np.ndarray) -> np.ndarray:
  """PyTorch Conv3d (O, I/G, kT, kH, kW) -> JAX DHWIO (kT, kH, kW, I/G, O)."""
  return np.ascontiguousarray(w.transpose(2, 3, 4, 1, 0))


def _transpose_conv2d_oihw_to_hwio(w: np.ndarray) -> np.ndarray:
  """PyTorch Conv2d (O, I, kH, kW) -> JAX HWIO (kH, kW, I, O)."""
  return np.ascontiguousarray(w.transpose(2, 3, 1, 0))


def _apply_activation(x: jax.Array, act_fn: str) -> jax.Array:
  name = act_fn.lower()
  if name in ("swish", "silu"):
    return jax.nn.silu(x)
  if name == "relu":
    return jax.nn.relu(x)
  if name == "gelu":
    return jax.nn.gelu(x, approximate=False)
  if name == "mish":
    return x * jnp.tanh(jax.nn.softplus(x))
  raise ValueError(f"Unsupported activation function: {act_fn!r}")


def wan_patchify_ndhwc(x: jax.Array, patch_size: int = 2) -> jax.Array:
  """Exact ``(c r q)`` patchify in NDHWC layout matching ``vae2_2.py:patchify``.

  Upstream PyTorch (NCTHW)::

      rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=p, r=p)

  Note that ``r`` (width subpixel) precedes ``q`` (height subpixel) in the
  channel dimension: ``out_c = (c * p + r) * p + q``.
  """
  if patch_size == 1:
    return x
  if x.ndim != 5:
    raise ValueError(f"Expected 5D NDHWC tensor, got shape {x.shape}")
  b, t, h, w, c = x.shape
  if h % patch_size != 0 or w % patch_size != 0:
    raise ValueError(
        f"Spatial dims ({h}, {w}) must be divisible by patch_size={patch_size}"
    )
  # (B, T, H//p, q, W//p, r, C) -> (B, T, H//p, W//p, C, r, q)
  x = x.reshape(b, t, h // patch_size, patch_size, w // patch_size, patch_size, c)
  x = x.transpose(0, 1, 2, 4, 6, 5, 3)
  return x.reshape(b, t, h // patch_size, w // patch_size, c * patch_size * patch_size)


def turbo_unpatchify_ndhwc(x: jax.Array, patch_size: int = 2) -> jax.Array:
  """Exact inverse ``(c r q)`` unpatchify in NDHWC matching ``_turbo_unpatchify``.

  Upstream PyTorch (NCTHW)::

      x = x.view(b, c, p, p, f, h, w)
      x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
      x = x.view(b, c, f, h * p, w * p)

  Channel layout is ``(c, r, q)`` where ``r`` is the width subpixel and ``q`` is
  the height subpixel.
  """
  if patch_size == 1:
    return x
  if x.ndim != 5:
    raise ValueError(f"Expected 5D NDHWC tensor, got shape {x.shape}")
  b, t, h, w, cp2 = x.shape
  p2 = patch_size * patch_size
  if cp2 % p2 != 0:
    raise ValueError(f"Channel dim {cp2} not divisible by patch_size^2={p2}")
  c = cp2 // p2
  # (B, T, H, W, C, r, q) -> (B, T, H, q, W, r, C) -> (B, T, H*p, W*p, C)
  x = x.reshape(b, t, h, w, c, patch_size, patch_size)
  x = x.transpose(0, 1, 2, 6, 3, 5, 4)
  return x.reshape(b, t, h * patch_size, w * patch_size, c)


def wan_patchify(x_ncthw: jax.Array, patch_size: int = 2) -> jax.Array:
  """NCTHW wrapper around :func:`wan_patchify_ndhwc`."""
  if patch_size == 1:
    return x_ncthw
  x = jnp.transpose(x_ncthw, (0, 2, 3, 4, 1))
  x = wan_patchify_ndhwc(x, patch_size=patch_size)
  return jnp.transpose(x, (0, 4, 1, 2, 3))


def turbo_unpatchify(x_ncthw: jax.Array, patch_size: int = 2) -> jax.Array:
  """NCTHW wrapper around :func:`turbo_unpatchify_ndhwc`."""
  if patch_size == 1:
    return x_ncthw
  x = jnp.transpose(x_ncthw, (0, 2, 3, 4, 1))
  x = turbo_unpatchify_ndhwc(x, patch_size=patch_size)
  return jnp.transpose(x, (0, 4, 1, 2, 3))


def _rms_norm_param_free(x: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Parameter-free float32 RMSNorm along the channel axis (last dim)."""
  x_f32 = x.astype(jnp.float32)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  out = x_f32 * jax.lax.rsqrt(var + eps)
  return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# Part 1: TurboVAED (Decoder-Only, bf16 compute, f32 norms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboVAEDConfig:
  """Full configuration for :class:`TurboVAED`.

  Compatible with both ``TurboV3-Wan22-TinyShallow_7_7.json`` and
  :class:`~diffusers_tpu.models.magi2.config.Magi2TurboVAEDConfig`.
  Per-stage tuples are stored in upstream JSON (encoder) order and reversed
  inside :class:`TurboVAEDDecoder3d`, matching ``turbo_vaed.py`` verbatim.
  """

  latent_channels: int = 48
  out_channels: int = 3
  block_out_channels: tuple[int, ...] = (64, 128, 256, 512)
  layers_per_block: tuple[int, ...] = (2, 2, 2, 3, 3)
  spatio_temporal_scaling: tuple[bool, ...] = (False, True, True, True)
  spatio_only: tuple[bool, ...] = (False, True, False, False)
  decoder_is_dw_conv: tuple[bool, ...] = (False, False, False, False, False)
  decoder_dw_kernel_size: int = 3
  patch_size: int = 2
  use_unpatchify: bool = True
  first_chunk_size: int = 7
  step_size: int = 7
  temporal_compression_ratio: int = 4
  spatial_compression_ratio: int = 16
  resnet_norm_eps: float = 1e-6
  rms_norm_eps: float = 1e-8
  decoder_causal: bool = False

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "TurboVAEDConfig":
    """Build from a ``TurboV3-*.json`` dictionary (ignores extra training keys)."""
    layers = tuple(
        int(v)
        for v in data.get(
            "decoder_layers_per_block",
            data.get("layers_per_block", (2, 2, 2, 3, 3)),
        )
    )
    block_out = tuple(
        int(v)
        for v in data.get(
            "decoder_block_out_channels",
            data.get("block_out_channels", (64, 128, 256, 512)),
        )
    )
    st_scaling = tuple(
        bool(v)
        for v in data.get(
            "decoder_spatio_temporal_scaling",
            data.get("spatio_temporal_scaling", (False, True, True, True)),
        )
    )
    spatio_only = tuple(
        bool(v)
        for v in data.get(
            "decoder_spatio_only",
            data.get("spatio_only", (False,) * len(block_out)),
        )
    )
    is_dw = tuple(
        bool(v)
        for v in data.get("decoder_is_dw_conv", (False,) * len(layers))
    )
    return cls(
        latent_channels=int(data.get("latent_channels", 48)),
        out_channels=int(data.get("out_channels", 3)),
        block_out_channels=block_out,
        layers_per_block=layers,
        spatio_temporal_scaling=st_scaling,
        spatio_only=spatio_only,
        decoder_is_dw_conv=is_dw,
        decoder_dw_kernel_size=int(data.get("decoder_dw_kernel_size", 3)),
        patch_size=int(data.get("patch_size", 2)),
        use_unpatchify=bool(data.get("use_unpatchify", True)),
        first_chunk_size=int(data.get("first_chunk_size", 7)),
        step_size=int(data.get("step_size", 7)),
        temporal_compression_ratio=int(data.get("temporal_compression_ratio", 4)),
        spatial_compression_ratio=int(data.get("spatial_compression_ratio", 16)),
        resnet_norm_eps=float(data.get("resnet_norm_eps", 1e-6)),
        rms_norm_eps=float(data.get("rms_norm_eps", 1e-8)),
        decoder_causal=bool(data.get("decoder_causal", False)),
    )

  @classmethod
  def from_json(cls, path: str | Path) -> "TurboVAEDConfig":
    with open(path, "r", encoding="utf-8") as fh:
      return cls.from_dict(json.load(fh))

  @classmethod
  def from_config(
      cls, cfg: "TurboVAEDConfig | Magi2TurboVAEDConfig | dict[str, Any] | str | Path"
  ) -> "TurboVAEDConfig":
    if isinstance(cfg, cls):
      return cfg
    if isinstance(cfg, Magi2TurboVAEDConfig):
      return cls(
          latent_channels=cfg.latent_channels,
          out_channels=cfg.out_channels,
          block_out_channels=tuple(cfg.block_out_channels),
          layers_per_block=tuple(cfg.layers_per_block),
          spatio_temporal_scaling=tuple(cfg.spatio_temporal_scaling),
          spatio_only=tuple(cfg.spatio_only),
          decoder_is_dw_conv=(False,) * len(cfg.layers_per_block),
          patch_size=cfg.patch_size,
          use_unpatchify=cfg.use_unpatchify,
          first_chunk_size=cfg.first_chunk_size,
          step_size=cfg.step_size,
          rms_norm_eps=cfg.rms_norm_eps,
      )
    if isinstance(cfg, dict):
      return cls.from_dict(cfg)
    if isinstance(cfg, (str, Path)):
      return cls.from_json(cfg)
    raise TypeError(f"Unsupported TurboVAED config type: {type(cfg)!r}")


class TurboVAEDConv3d(nnx.Module):
  """Non-causal 3D conv with symmetric temporal edge-replication padding.

  Matches ``TurboVAEDCausalConv3d`` in ``turbo_vaed.py`` (which asserts
  ``is_causal == False``):
  - Temporal axis: replicates first frame ``(k_t - 1) // 2`` times on the left
    and last frame ``(k_t - 1) // 2`` times on the right (when ``k_t > 1``).
  - Spatial axes: zero-padding ``(k_h // 2, k_w // 2)`` on both sides.
  - Weight layout: ``DHWIO`` ``(kT, kH, kW, C_in // groups, C_out)``.
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int | tuple[int, int, int] = 3,
      stride: int | tuple[int, int, int] = 1,
      dilation: int | tuple[int, int, int] = 1,
      groups: int = 1,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride, stride)
    if isinstance(dilation, int):
      # Matches turbo_vaed.py: int dilation applies to temporal axis only.
      dilation = (dilation, 1, 1)

    self.kernel_size = tuple(kernel_size)
    self.stride = tuple(stride)
    self.dilation = tuple(dilation)
    self.groups = int(groups)
    kt, kh, kw = self.kernel_size
    self.spatial_pad = ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2))
    self.kernel = nnx.Param(
        jnp.zeros((kt, kh, kw, in_channels // self.groups, out_channels), dtype)
    )
    self.bias = nnx.Param(jnp.zeros((out_channels,), dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    kt = self.kernel_size[0]
    if kt > 1:
      pad_t = (kt - 1) // 2
      pad_left = jnp.repeat(x[:, :1, :, :, :], pad_t, axis=1)
      pad_right = jnp.repeat(x[:, -1:, :, :, :], pad_t, axis=1)
      x = jnp.concatenate([pad_left, x, pad_right], axis=1)
    y = jax.lax.conv_general_dilated(
        x,
        self.kernel[...].astype(x.dtype),
        window_strides=self.stride,
        padding=self.spatial_pad,
        rhs_dilation=self.dilation,
        feature_group_count=self.groups,
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    return y + self.bias[...].astype(x.dtype)


class TurboVAEDDepthwiseSeparableConv3d(nnx.Module):
  """Depthwise + pointwise 3D conv matching ``TurboVAEDCausalDepthwiseSeparableConv3d``.

  Uses symmetric temporal edge-replication padding before the depthwise conv,
  followed by a ``1x1x1`` pointwise conv.
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int | tuple[int, int, int] = 3,
      stride: int | tuple[int, int, int] = 1,
      dilation: int | tuple[int, int, int] = 1,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.depthwise_conv = TurboVAEDConv3d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=in_channels,
        dtype=dtype,
    )
    self.pointwise_conv = TurboVAEDConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        dtype=dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.pointwise_conv(self.depthwise_conv(x))


class TurboVAEDConv2d(nnx.Module):
  """NHWC 2D convolution with ``HWIO`` weights ``(kH, kW, C_in, C_out)``."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int | tuple[int, int] = 3,
      stride: int | tuple[int, int] = 1,
      padding: int | tuple[tuple[int, int], tuple[int, int]] = 1,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride)
    if isinstance(padding, int):
      pad_spec = ((padding, padding), (padding, padding))
    else:
      pad_spec = tuple((int(a), int(b)) for a, b in padding)
    self.stride = tuple(stride)
    self.padding = pad_spec
    kh, kw = kernel_size
    self.kernel = nnx.Param(jnp.zeros((kh, kw, in_channels, out_channels), dtype))
    self.bias = nnx.Param(jnp.zeros((out_channels,), dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        self.kernel[...].astype(x.dtype),
        window_strides=self.stride,
        padding=self.padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + self.bias[...].astype(x.dtype)


class TurboVAEDConv2dSplitUpsampler(nnx.Module):
  """Per-frame 2D conv + pixel_shuffle upsampler (used only if ``use_unpatchify=False``)."""

  def __init__(
      self,
      in_channels: int,
      kernel_size: int | tuple[int, int] = 3,
      stride: int | tuple[int, int] = 2,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride)
    self.scale = int(stride[0])
    kh, kw = kernel_size
    self.conv = TurboVAEDConv2d(
        in_channels,
        in_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=((kh // 2, kh // 2), (kw // 2, kw // 2)),
        dtype=dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    b, t, h, w, c = x.shape
    r = self.scale
    y = self.conv(x.reshape(b * t, h, w, c))
    c_out = c // (r * r)
    # PyTorch pixel_shuffle uses (C_out, r_h, r_w) channel order.
    y = y.reshape(b * t, h, w, c_out, r, r).transpose(0, 1, 4, 2, 5, 3)
    return y.reshape(b, t, h * r, w * r, c_out)


class TurboVAEDWanResample(nnx.Module):
  """Spatial (``upsample2d``) or spatio-temporal (``upsample3d``) upsampler.

  Matches ``WanResample`` in ``turbo_vaed.py``:
  - ``upsample3d`` runs ``time_conv(dim -> 2*dim, k=(3,1,1))``, splits channel
    dim as ``(n_split=2, C)`` (NOT ``(C, 2)``) interleaving even/odd frames, and
    drops frame 0 when ``is_first_chunk=True``.
  - Both modes perform 2x ``nearest-exact`` spatial upsampling followed by a
    per-frame 3x3 ``Conv2d``.
  """

  def __init__(
      self,
      dim: int,
      mode: str,
      upsample_out_dim: int | None = None,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    if mode not in ("upsample2d", "upsample3d", "none"):
      raise ValueError(f"Unsupported TurboVAEDWanResample mode: {mode!r}")
    self.dim = dim
    self.mode = mode
    out_dim = dim // 2 if upsample_out_dim is None else upsample_out_dim
    if mode in ("upsample2d", "upsample3d"):
      self.conv = TurboVAEDConv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, dtype=dtype)
    else:
      self.conv = None
    if mode == "upsample3d":
      self.time_conv = TurboVAEDConv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, dtype=dtype)
    else:
      self.time_conv = None

  def __call__(self, x: jax.Array, is_first_chunk: bool = True) -> jax.Array:
    if self.mode == "none":
      return x
    if self.mode == "upsample3d":
      assert self.time_conv is not None
      x = self.time_conv(x)
      b, t, h, w, c2 = x.shape
      c = c2 // 2
      # Exact match: rearrange(x, "b (n_split c) t h w -> b c (t n_split) h w", n_split=2)
      x = x.reshape(b, t, h, w, 2, c).transpose(0, 1, 4, 2, 3, 5).reshape(b, t * 2, h, w, c)
      if is_first_chunk:
        x = x[:, 1:, :, :, :]
    # 2x nearest-exact upsampling along H (axis 2) and W (axis 3).
    x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
    b, t, h, w, c = x.shape
    assert self.conv is not None
    y = self.conv(x.reshape(b * t, h, w, c))
    return y.reshape(b, t, h, w, y.shape[-1])


class TurboVAEDResNetBlock3d(nnx.Module):
  """3D ResNet block matching ``TurboVAEDResNetBlock3d`` in ``turbo_vaed.py``.

  CRITICAL QUIRKS preserved verbatim:
  - ``norm1`` and ``norm2`` use parameter-free float32 RMSNorm with ``eps=1e-8``.
  - When ``is_upsampler_modified=True``, the first activation is ``ReLU`` (NOT
    ``SiLU``); the second activation is ALWAYS ``SiLU``.
  - When ``in_channels != out_channels``, ``norm3`` uses ``eps=resnet_norm_eps``
    (default ``1e-6``), and ``conv_shortcut`` uses the same conv op
    (depthwise-separable ``1x1x1`` if ``is_dw_conv=True``, else standard
    ``1x1x1`` conv).
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int | None = None,
      *,
      eps: float = 1e-6,
      rms_norm_eps: float = 1e-8,
      non_linearity: str = "swish",
      is_upsampler_modified: bool = False,
      is_dw_conv: bool = False,
      dw_kernel_size: int = 3,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    out_channels = in_channels if out_channels is None else out_channels
    self.non_linearity = non_linearity
    self.is_upsampler_modified = bool(is_upsampler_modified)
    self.is_dw_conv = bool(is_dw_conv)
    self.rms_norm_eps = float(rms_norm_eps)
    self.shortcut_norm_eps = float(eps)

    conv_cls = TurboVAEDDepthwiseSeparableConv3d if is_dw_conv else TurboVAEDConv3d
    k = dw_kernel_size if is_dw_conv else 3

    self.conv1 = conv_cls(in_channels, out_channels, kernel_size=k, dtype=dtype)
    self.conv2 = conv_cls(out_channels, out_channels, kernel_size=k, dtype=dtype)
    if in_channels != out_channels:
      self.conv_shortcut = conv_cls(
          in_channels, out_channels, kernel_size=1, stride=1, dtype=dtype
      )
    else:
      self.conv_shortcut = None

  def __call__(self, inputs: jax.Array) -> jax.Array:
    h = _rms_norm_param_free(inputs, self.rms_norm_eps)
    if self.is_upsampler_modified:
      h = jax.nn.relu(h)
    else:
      h = _apply_activation(h, self.non_linearity)
    h = self.conv1(h)

    h = _rms_norm_param_free(h, self.rms_norm_eps)
    h = _apply_activation(h, self.non_linearity)
    h = self.conv2(h)

    if self.conv_shortcut is not None:
      inputs = _rms_norm_param_free(inputs, self.shortcut_norm_eps)
      inputs = self.conv_shortcut(inputs)

    return h + inputs


class TurboVAEDMidBlock3d(nnx.Module):
  """Mid block: sequence of ``num_layers`` ResNet blocks (no attention)."""

  def __init__(
      self,
      in_channels: int,
      num_layers: int = 1,
      *,
      resnet_eps: float = 1e-6,
      rms_norm_eps: float = 1e-8,
      resnet_act_fn: str = "swish",
      is_dw_conv: bool = False,
      dw_kernel_size: int = 3,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.resnets = nnx.List([
        TurboVAEDResNetBlock3d(
            in_channels=in_channels,
            out_channels=in_channels,
            eps=resnet_eps,
            rms_norm_eps=rms_norm_eps,
            non_linearity=resnet_act_fn,
            is_upsampler_modified=False,
            is_dw_conv=is_dw_conv,
            dw_kernel_size=dw_kernel_size,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ])

  def __call__(self, x: jax.Array) -> jax.Array:
    for resnet in self.resnets:
      x = resnet(x)
    return x


class TurboVAEDUpBlock3d(nnx.Module):
  """Up block: optional ``conv_in`` ResNet -> optional ``WanResample`` -> ``resnets``."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int | None = None,
      num_layers: int = 1,
      *,
      resnet_eps: float = 1e-6,
      rms_norm_eps: float = 1e-8,
      resnet_act_fn: str = "swish",
      spatio_temporal_scale: bool = True,
      is_dw_conv: bool = False,
      dw_kernel_size: int = 3,
      spatio_only: bool = False,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    out_channels = in_channels if out_channels is None else out_channels
    if in_channels != out_channels:
      self.conv_in = TurboVAEDResNetBlock3d(
          in_channels=in_channels,
          out_channels=out_channels,
          eps=resnet_eps,
          rms_norm_eps=rms_norm_eps,
          non_linearity=resnet_act_fn,
          is_upsampler_modified=False,
          is_dw_conv=is_dw_conv,
          dw_kernel_size=dw_kernel_size,
          dtype=dtype,
      )
    else:
      self.conv_in = None

    if spatio_temporal_scale:
      mode = "upsample2d" if spatio_only else "upsample3d"
      self.upsamplers = nnx.List([
          TurboVAEDWanResample(
              dim=out_channels, mode=mode, upsample_out_dim=out_channels, dtype=dtype
          )
      ])
    else:
      self.upsamplers = None

    # Every ResNet in an up-block with spatio_temporal_scale=True uses ReLU for act1!
    self.resnets = nnx.List([
        TurboVAEDResNetBlock3d(
            in_channels=out_channels,
            out_channels=out_channels,
            eps=resnet_eps,
            rms_norm_eps=rms_norm_eps,
            non_linearity=resnet_act_fn,
            is_upsampler_modified=spatio_temporal_scale,
            is_dw_conv=is_dw_conv,
            dw_kernel_size=dw_kernel_size,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ])

  def __call__(self, x: jax.Array, is_first_chunk: bool = True) -> jax.Array:
    if self.conv_in is not None:
      x = self.conv_in(x)
    if self.upsamplers is not None:
      for upsampler in self.upsamplers:
        x = upsampler(x, is_first_chunk=is_first_chunk)
    for resnet in self.resnets:
      x = resnet(x)
    return x


class TurboVAEDDecoder3d(nnx.Module):
  """3D Decoder network matching ``TurboVAEDDecoder3d`` in ``turbo_vaed.py``."""

  def __init__(self, config: TurboVAEDConfig, *, dtype: jnp.dtype = jnp.bfloat16):
    if config.patch_size != 2:
      raise ValueError(f"TurboVAED requires patch_size == 2, got {config.patch_size}")
    if config.decoder_causal and not all(config.decoder_is_dw_conv):
      raise ValueError("TurboVAED non-DW convs require decoder_causal=False")

    self.patch_size = config.patch_size
    self.use_unpatchify = config.use_unpatchify
    self.rms_norm_eps = config.rms_norm_eps

    # Reverse all per-stage tuples exactly as in upstream TurboVAEDDecoder3d.
    block_out_channels = tuple(reversed(config.block_out_channels))
    spatio_temporal_scaling = tuple(reversed(config.spatio_temporal_scaling))
    layers_per_block = tuple(reversed(config.layers_per_block))
    decoder_is_dw_conv = tuple(reversed(config.decoder_is_dw_conv))
    spatio_only = tuple(reversed(config.spatio_only))

    if len(layers_per_block) != len(block_out_channels) + 1:
      raise ValueError(
          f"layers_per_block ({len(layers_per_block)}) must equal "
          f"len(block_out_channels) + 1 ({len(block_out_channels) + 1})"
      )

    output_channel = block_out_channels[0]
    self.conv_in = TurboVAEDConv3d(
        in_channels=config.latent_channels,
        out_channels=output_channel,
        kernel_size=3,
        stride=1,
        dtype=dtype,
    )
    self.mid_block = TurboVAEDMidBlock3d(
        in_channels=output_channel,
        num_layers=layers_per_block[0],
        resnet_eps=config.resnet_norm_eps,
        rms_norm_eps=config.rms_norm_eps,
        is_dw_conv=decoder_is_dw_conv[0],
        dw_kernel_size=config.decoder_dw_kernel_size,
        dtype=dtype,
    )

    up_blocks = []
    for i in range(len(block_out_channels)):
      input_channel = output_channel
      output_channel = block_out_channels[i]
      up_blocks.append(
          TurboVAEDUpBlock3d(
              in_channels=input_channel,
              out_channels=output_channel,
              num_layers=layers_per_block[i + 1],
              resnet_eps=config.resnet_norm_eps,
              rms_norm_eps=config.rms_norm_eps,
              spatio_temporal_scale=spatio_temporal_scaling[i],
              is_dw_conv=decoder_is_dw_conv[i + 1],
              dw_kernel_size=config.decoder_dw_kernel_size,
              spatio_only=spatio_only[i],
              dtype=dtype,
          )
      )
    self.up_blocks = nnx.List(up_blocks)

    if not self.use_unpatchify:
      self.upsampler2d_1 = TurboVAEDConv2dSplitUpsampler(
          in_channels=output_channel, kernel_size=3, stride=(2, 2), dtype=dtype
      )
      output_channel = output_channel // 4
    else:
      self.upsampler2d_1 = None

    conv_out_channels = (
        config.out_channels * self.patch_size * self.patch_size
        if (self.use_unpatchify and self.patch_size >= 2)
        else config.out_channels
    )
    self.conv_out = TurboVAEDConv3d(
        in_channels=output_channel,
        out_channels=conv_out_channels,
        kernel_size=3,
        stride=1,
        dtype=dtype,
    )

  def __call__(self, x: jax.Array, is_first_chunk: bool = True) -> jax.Array:
    h = self.conv_in(x)
    h = self.mid_block(h)
    for up_block in self.up_blocks:
      h = up_block(h, is_first_chunk=is_first_chunk)

    if not self.use_unpatchify:
      assert self.upsampler2d_1 is not None
      h = _rms_norm_param_free(h, self.rms_norm_eps)
      h = jax.nn.silu(h)
      h = self.upsampler2d_1(h)

    # Final inline parameter-free float32 RMSNorm (eps=1e-8) -> SiLU -> conv_out.
    h = _rms_norm_param_free(h, self.rms_norm_eps)
    h = jax.nn.silu(h)
    h = self.conv_out(h)

    if self.use_unpatchify:
      h = turbo_unpatchify_ndhwc(h, self.patch_size)
    return h


class TurboVAED(nnx.Module):
  """MAGI-2-preview distilled TurboVAED video decoder in Flax NNX.

  Computes in ``dtype`` (default ``jnp.bfloat16`` for TPU) with ``float32``
  RMSNorms and latent denormalization.
  """

  def __init__(
      self,
      config: TurboVAEDConfig | Magi2TurboVAEDConfig | dict[str, Any] | str | Path | None = None,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.config = (
        TurboVAEDConfig() if config is None else TurboVAEDConfig.from_config(config)
    )
    self.dtype = dtype
    self.decoder = TurboVAEDDecoder3d(self.config, dtype=dtype)

  def decode_chunk(self, z_ndhwc: jax.Array, is_first_chunk: bool = True) -> jax.Array:
    """Decode a single denormalized NDHWC latent chunk."""
    return self.decoder(z_ndhwc.astype(self.dtype), is_first_chunk=is_first_chunk)

  def _denormalize_ndhwc(self, z_ndhwc: jax.Array) -> jax.Array:
    mean = jnp.asarray(WAN22_LATENT_MEAN, dtype=jnp.float32).reshape(1, 1, 1, 1, -1)
    std = jnp.asarray(WAN22_LATENT_STD, dtype=jnp.float32).reshape(1, 1, 1, 1, -1)
    return (z_ndhwc.astype(jnp.float32) * std + mean).astype(self.dtype)

  def decode(
      self,
      z: jax.Array,
      *,
      input_layout: str = "ncthw",
      output_layout: str = "ncthw",
      denormalize: bool = True,
      output_offload: bool = False,
      chunk_fn: Any = None,
  ) -> jax.Array | np.ndarray:
    """Sliding-window decode matching ``TurboVAED._sliding_window_decode``.

    Args:
      z: Video latents, shape ``(B, 48, T, H, W)`` if ``input_layout="ncthw"``
        or ``(B, T, H, W, 48)`` if ``input_layout="ndhwc"``.
      input_layout: ``"ncthw"`` (PyTorch/DiT convention) or ``"ndhwc"``.
      output_layout: ``"ncthw"`` (returns ``(B, 3, T_out, H_out, W_out)``) or
        ``"ndhwc"`` (returns ``(B, T_out, H_out, W_out, 3)``).
      denormalize: If True, applies Wan2.2 48-ch ``z * std + mean`` in float32.
      output_offload: If True, copies each decoded chunk to host NumPy to limit
        TPU HBM usage at high resolution (e.g., 1080p); returns a NumPy array.
      chunk_fn: Optional callable ``(z_chunk_ndhwc, is_first_chunk) -> out_ndhwc``
        (e.g., a JIT-compiled wrapper around :meth:`decode_chunk`).

    Returns:
      Decoded RGB video in ``[-1, 1]`` with ``4 * T - 3`` pixel frames
      (e.g., ``125`` for ``T=32``, ``249`` for ``T=63``).
    """
    if input_layout == "ncthw":
      z_ndhwc = jnp.transpose(z, (0, 2, 3, 4, 1))
    elif input_layout == "ndhwc":
      z_ndhwc = z
    else:
      raise ValueError(f"Unsupported input_layout: {input_layout!r}")

    if denormalize:
      z_ndhwc = self._denormalize_ndhwc(z_ndhwc)
    else:
      z_ndhwc = z_ndhwc.astype(self.dtype)

    run_chunk = self.decoder if chunk_fn is None else chunk_fn

    first_chunk_size = self.config.first_chunk_size
    step = self.config.step_size
    r_t = self.config.temporal_compression_ratio
    overlap = 1 * r_t

    num_frames = int(z_ndhwc.shape[1])

    # 1. Pad tail frames with last frame so num_frames == first_chunk_size + n * step.
    num_padding_frames = 0
    if num_frames < first_chunk_size:
      num_padding_frames = first_chunk_size - num_frames
    elif (num_frames - first_chunk_size) % step != 0:
      num_padding_frames = step - (num_frames - first_chunk_size) % step

    if num_padding_frames > 0:
      tail = jnp.repeat(z_ndhwc[:, -1:, :, :, :], num_padding_frames, axis=1)
      z_ndhwc = jnp.concatenate([z_ndhwc, tail], axis=1)
      num_frames = num_frames + num_padding_frames

    # 2. Decode overlapping windows.
    out_chunks = []
    if num_frames == first_chunk_size:
      out = run_chunk(z_ndhwc, is_first_chunk=True)
      out_chunks.append(np.asarray(out) if output_offload else out)
    else:
      # First chunk: attach 1 right-context latent frame, trim right overlap.
      out = run_chunk(z_ndhwc[:, 0 : first_chunk_size + 1], is_first_chunk=True)
      out = out[:, :-overlap]
      out_chunks.append(np.asarray(out) if output_offload else out)

      # Middle chunks: 1 left + 1 right context frame; last chunk: 1 left frame.
      for i in range(first_chunk_size, num_frames, step):
        is_last_chunk = i + step == num_frames
        left = i - 1
        right = i + step if is_last_chunk else i + step + 1
        out_ = run_chunk(z_ndhwc[:, left:right], is_first_chunk=False)
        if is_last_chunk:
          out_ = out_[:, overlap:]
        else:
          out_ = out_[:, overlap:-overlap]
        out_chunks.append(np.asarray(out_) if output_offload else out_)

    if output_offload:
      out_all: jax.Array | np.ndarray = np.concatenate(out_chunks, axis=1)
    else:
      out_all = jnp.concatenate(out_chunks, axis=1)

    # 3. Remove padded tail frames.
    if num_padding_frames > 0:
      out_all = out_all[:, : -num_padding_frames * r_t]

    if output_layout == "ncthw":
      if output_offload:
        return np.transpose(out_all, (0, 4, 1, 2, 3))
      return jnp.transpose(out_all, (0, 4, 1, 2, 3))
    if output_layout == "ndhwc":
      return out_all
    raise ValueError(f"Unsupported output_layout: {output_layout!r}")

  def __call__(self, z: jax.Array, **kwargs: Any) -> jax.Array | np.ndarray:
    return self.decode(z, **kwargs)


def _iter_turbo_conv_params(
    prefix: str, mod: TurboVAEDConv3d | TurboVAEDDepthwiseSeparableConv3d
) -> Iterator[tuple[str, nnx.Param, str]]:
  if isinstance(mod, TurboVAEDDepthwiseSeparableConv3d):
    yield f"{prefix}.depthwise_conv.weight", mod.depthwise_conv.kernel, "conv3d"
    yield f"{prefix}.depthwise_conv.bias", mod.depthwise_conv.bias, "bias"
    yield f"{prefix}.pointwise_conv.weight", mod.pointwise_conv.kernel, "conv3d"
    yield f"{prefix}.pointwise_conv.bias", mod.pointwise_conv.bias, "bias"
  else:
    yield f"{prefix}.conv.weight", mod.kernel, "conv3d"
    yield f"{prefix}.conv.bias", mod.bias, "bias"


def _iter_turbo_resnet_params(
    prefix: str, block: TurboVAEDResNetBlock3d
) -> Iterator[tuple[str, nnx.Param, str]]:
  yield from _iter_turbo_conv_params(f"{prefix}.conv1", block.conv1)
  yield from _iter_turbo_conv_params(f"{prefix}.conv2", block.conv2)
  if block.conv_shortcut is not None:
    yield from _iter_turbo_conv_params(f"{prefix}.conv_shortcut", block.conv_shortcut)


def _iter_turbo_vaed_params(
    decoder: TurboVAEDDecoder3d,
) -> Iterator[tuple[str, nnx.Param, str]]:
  """Yield ``(upstream_decoder_rel_key, nnx.Param, kind)`` for all TurboVAED weights."""
  yield from _iter_turbo_conv_params("conv_in", decoder.conv_in)

  for j, res in enumerate(decoder.mid_block.resnets):
    yield from _iter_turbo_resnet_params(f"mid_block.resnets.{j}", res)

  for i, up in enumerate(decoder.up_blocks):
    up_prefix = f"up_blocks.{i}"
    if up.conv_in is not None:
      yield from _iter_turbo_resnet_params(f"{up_prefix}.conv_in", up.conv_in)
    if up.upsamplers is not None:
      for u_idx, up_mod in enumerate(up.upsamplers):
        u_prefix = f"{up_prefix}.upsamplers.{u_idx}"
        if up_mod.conv is not None:
          yield f"{u_prefix}.resample.1.weight", up_mod.conv.kernel, "conv2d"
          yield f"{u_prefix}.resample.1.bias", up_mod.conv.bias, "bias"
        if up_mod.time_conv is not None:
          yield from _iter_turbo_conv_params(f"{u_prefix}.time_conv", up_mod.time_conv)
    for j, res in enumerate(up.resnets):
      yield from _iter_turbo_resnet_params(f"{up_prefix}.resnets.{j}", res)

  if decoder.upsampler2d_1 is not None:
    yield "upsampler2d_1.conv.weight", decoder.upsampler2d_1.conv.kernel, "conv2d"
    yield "upsampler2d_1.conv.bias", decoder.upsampler2d_1.conv.bias, "bias"

  yield from _iter_turbo_conv_params("conv_out", decoder.conv_out)


def _extract_turbo_state_dict(ckpt: dict[str, Any]) -> tuple[dict[str, Any], int]:
  """Extract decoder state_dict from ``turbo_vae/checkpoint.ckpt`` matching ``get_turbo_vaed``.

  Returns ``(decoder_rel_state_dict, num_skipped_non_decoder_keys)``.
  """
  if "ema_state_dict" in ckpt:
    raw = ckpt["ema_state_dict"]
  elif "state_dict" in ckpt:
    raw = ckpt["state_dict"]
  elif "ema_model" in ckpt:
    raw = ckpt["ema_model"]
  elif "model" in ckpt:
    raw = ckpt["model"]
  else:
    raw = ckpt

  stripped = {
      (k[7:] if k.startswith("module.") else k): v for k, v in raw.items()
  }
  has_decoder_prefix = any(k.startswith("decoder.") for k in stripped)

  cleaned: dict[str, Any] = {}
  skipped = 0
  for k, val in stripped.items():
    if has_decoder_prefix:
      if k.startswith("decoder."):
        cleaned[k[len("decoder."):]] = val
      else:
        skipped += 1
    else:
      if k.startswith("encoder."):
        skipped += 1
      else:
        cleaned[k] = val
  return cleaned, skipped


def load_turbo_vaed_weights(
    model: TurboVAED,
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
    verbose: bool = False,
) -> TurboVAED:
  """Populate :class:`TurboVAED` parameters from a PyTorch state dict (transposing convs)."""
  cleaned, skipped = _extract_turbo_state_dict(state_dict)
  expected_map = list(_iter_turbo_vaed_params(model.decoder))
  expected_keys = {k for k, _, _ in expected_map}

  missing: list[str] = []
  loaded = 0
  for key, param, kind in expected_map:
    if key not in cleaned:
      missing.append(key)
      continue
    arr = _to_numpy_f32(cleaned[key])
    if kind == "conv3d":
      arr = _transpose_conv3d_oidhw_to_dhwio(arr)
    elif kind == "conv2d":
      arr = _transpose_conv2d_oihw_to_hwio(arr)
    else:
      arr = arr.reshape(param[...].shape)
    if arr.shape != param[...].shape:
      raise ValueError(
          f"TurboVAED shape mismatch for {key!r}: got {arr.shape}, "
          f"expected {param[...].shape}"
      )
    param[...] = jnp.asarray(arr, dtype=param[...].dtype)
    loaded += 1

  unexpected = sorted(set(cleaned.keys()) - expected_keys)
  total_nnx_params = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if strict and (missing or unexpected or loaded != total_nnx_params):
    raise ValueError(
        f"TurboVAED weight load mismatch: loaded={loaded}/{total_nnx_params}, "
        f"missing={missing[:10]} (total {len(missing)}), "
        f"unexpected={unexpected[:10]} (total {len(unexpected)})"
    )
  if verbose:
    print(
        f"[magi2] TurboVAED loaded {loaded}/{total_nnx_params} tensors "
        f"(skipped={skipped} non-decoder, missing={len(missing)}, "
        f"unexpected={len(unexpected)}, dtype={model.dtype})"
    )
  return model


def load_turbo_vaed(
    config: TurboVAEDConfig | Magi2TurboVAEDConfig | dict[str, Any] | str | Path,
    ckpt_path: str | Path,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    strict: bool = True,
    verbose: bool = False,
) -> TurboVAED:
  """Load :class:`TurboVAED` from ``TurboV3-*.json`` + ``checkpoint.ckpt`` (or ``.safetensors``)."""
  model = TurboVAED(config, dtype=dtype)
  ckpt_path = Path(ckpt_path)
  if ckpt_path.suffix == ".safetensors":
    from diffusers_tpu.safetensors_io import safe_open

    state_dict: dict[str, Any] = {}
    with safe_open(str(ckpt_path), framework="np") as reader:
      for k in reader.keys():
        if k.startswith("encoder.") or k.startswith("module.encoder."):
          continue
        state_dict[k] = reader.get_tensor(k)
  else:
    import torch

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

  return load_turbo_vaed_weights(model, state_dict, strict=strict, verbose=verbose)


# ---------------------------------------------------------------------------
# Part 2: Wan2_2_VAEEncoder (Encoder-Only, float32, I2V Reference Images)
# ---------------------------------------------------------------------------


class WanRMSNorm(nnx.Module):
  """``RMSNorm`` from ``vae2_2.py``: ``F.normalize(x, dim=-1, eps=1e-12) * sqrt(C) * gamma``."""

  def __init__(self, dim: int, *, dtype: jnp.dtype = jnp.float32):
    self.dim = int(dim)
    self.scale = float(dim) ** 0.5
    self.gamma = nnx.Param(jnp.ones((dim,), dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(jnp.square(x_f32), axis=-1, keepdims=True))
    y = (
        (x_f32 / jnp.maximum(norm, 1e-12))
        * self.scale
        * self.gamma[...].astype(jnp.float32)
    )
    return y.astype(x.dtype)


class WanCausalConv3d(nnx.Module):
  """Causal 3D conv matching ``CausalConv3d`` in ``vae2_2.py`` (NDHWC, DHWIO weights)."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int | tuple[int, int, int],
      stride: int | tuple[int, int, int] = 1,
      padding: int | tuple[int, int, int] = 0,
      *,
      dtype: jnp.dtype = jnp.float32,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride, stride)
    if isinstance(padding, int):
      padding = (padding, padding, padding)

    self.kernel_size = tuple(kernel_size)
    self.stride = tuple(stride)
    self.pad_t = 2 * int(padding[0])  # causal left-only temporal padding
    self.pad_h = int(padding[1])
    self.pad_w = int(padding[2])

    kt, kh, kw = self.kernel_size
    self.kernel = nnx.Param(jnp.zeros((kt, kh, kw, in_channels, out_channels), dtype))
    self.bias = nnx.Param(jnp.zeros((out_channels,), dtype))

  def __call__(self, x: jax.Array, cache_x: jax.Array | None = None) -> jax.Array:
    pad_t_left = self.pad_t
    if cache_x is not None and pad_t_left > 0:
      x = jnp.concatenate([cache_x.astype(x.dtype), x], axis=1)
      pad_t_left -= int(cache_x.shape[1])
    y = jax.lax.conv_general_dilated(
        x,
        self.kernel[...].astype(x.dtype),
        window_strides=self.stride,
        padding=((pad_t_left, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)),
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    return y + self.bias[...].astype(x.dtype)


def _run_cached_causal_conv3d(
    conv: WanCausalConv3d,
    x: jax.Array,
    feat_cache: list[jax.Array | None] | None,
    feat_idx: list[int],
) -> jax.Array:
  """Apply a 3x3x3 ``WanCausalConv3d`` and update the ``CACHE_T=2`` rolling cache."""
  if feat_cache is None:
    return conv(x, cache_x=None)
  idx = feat_idx[0]
  cache_x = x[:, -CACHE_T:, :, :, :]
  if cache_x.shape[1] < CACHE_T and feat_cache[idx] is not None:
    cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
  y = conv(x, cache_x=feat_cache[idx])
  feat_cache[idx] = cache_x
  feat_idx[0] += 1
  return y


class WanDownResample(nnx.Module):
  """Encoder ``Resample`` (``downsample2d`` or ``downsample3d``) from ``vae2_2.py``."""

  def __init__(self, dim: int, mode: str, *, dtype: jnp.dtype = jnp.float32):
    if mode not in ("downsample2d", "downsample3d"):
      raise ValueError(f"Unsupported WanDownResample mode: {mode!r}")
    self.dim = dim
    self.mode = mode
    # Exact match to ZeroPad2d((0, 1, 0, 1)) -> Conv2d(dim, dim, 3, stride=2, padding=0).
    self.conv = TurboVAEDConv2d(
        dim, dim, kernel_size=3, stride=2, padding=((0, 1), (0, 1)), dtype=dtype
    )
    if mode == "downsample3d":
      self.time_conv = WanCausalConv3d(
          dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), dtype=dtype
      )
    else:
      self.time_conv = None

  def __call__(
      self,
      x: jax.Array,
      feat_cache: list[jax.Array | None] | None = None,
      feat_idx: list[int] | None = None,
  ) -> jax.Array:
    b, t, h, w, c = x.shape
    y = self.conv(x.reshape(b * t, h, w, c))
    x = y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])

    if self.mode == "downsample3d" and feat_cache is not None:
      assert feat_idx is not None and self.time_conv is not None
      idx = feat_idx[0]
      if feat_cache[idx] is None:
        # First chunk (including all T=1 single-image encodes): skip time_conv!
        feat_cache[idx] = x
        feat_idx[0] += 1
      else:
        cache_x = x[:, -1:, :, :, :]
        x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    return x


class WanResidualBlock(nnx.Module):
  """Encoder ``ResidualBlock`` from ``vae2_2.py``."""

  def __init__(self, in_dim: int, out_dim: int, *, dtype: jnp.dtype = jnp.float32):
    self.norm1 = WanRMSNorm(in_dim, dtype=dtype)
    self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1, dtype=dtype)
    self.norm2 = WanRMSNorm(out_dim, dtype=dtype)
    self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1, dtype=dtype)
    self.shortcut = (
        WanCausalConv3d(in_dim, out_dim, 1, padding=0, dtype=dtype)
        if in_dim != out_dim
        else None
    )

  def __call__(
      self,
      x: jax.Array,
      feat_cache: list[jax.Array | None] | None = None,
      feat_idx: list[int] | None = None,
  ) -> jax.Array:
    h = x if self.shortcut is None else self.shortcut(x)
    idx_ref = [0] if feat_idx is None else feat_idx
    x = jax.nn.silu(self.norm1(x))
    x = _run_cached_causal_conv3d(self.conv1, x, feat_cache, idx_ref)
    x = jax.nn.silu(self.norm2(x))
    x = _run_cached_causal_conv3d(self.conv2, x, feat_cache, idx_ref)
    return x + h


class WanAttentionBlock(nnx.Module):
  """Middle single-head per-frame spatial ``AttentionBlock`` from ``vae2_2.py``."""

  def __init__(self, dim: int, *, dtype: jnp.dtype = jnp.float32):
    self.dim = dim
    self.norm = WanRMSNorm(dim, dtype=dtype)
    self.to_qkv = TurboVAEDConv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0, dtype=dtype)
    self.proj = TurboVAEDConv2d(dim, dim, kernel_size=1, stride=1, padding=0, dtype=dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    identity = x
    b, t, h, w, c = x.shape
    x2d = self.norm(x.reshape(b * t, h, w, c))
    qkv = self.to_qkv(x2d).reshape(b * t, h * w, 3 * c)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    # Single-head non-causal spatial attention, scale = 1 / sqrt(c).
    out = jax.nn.dot_product_attention(
        q[:, :, None, :], k[:, :, None, :], v[:, :, None, :]
    )[:, :, 0, :]
    out = self.proj(out.reshape(b * t, h, w, c)).reshape(b, t, h, w, c)
    return out + identity


def _avg_down_3d_ndhwc(
    x: jax.Array,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int,
) -> jax.Array:
  """Exact parameter-free ``AvgDown3D`` residual shortcut from ``vae2_2.py``."""
  factor = factor_t * factor_s * factor_s
  if (in_channels * factor) % out_channels != 0:
    raise ValueError(
        f"in_channels * factor ({in_channels * factor}) must be divisible by "
        f"out_channels ({out_channels})"
    )
  group_size = (in_channels * factor) // out_channels
  b, t, h, w, c = x.shape
  pad_t = (factor_t - t % factor_t) % factor_t
  if pad_t > 0:
    x = jnp.pad(x, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)))
    t = t + pad_t
  x = x.reshape(
      b,
      t // factor_t,
      factor_t,
      h // factor_s,
      factor_s,
      w // factor_s,
      factor_s,
      c,
  )
  # Match PyTorch permute(0, 1, 3, 5, 7, 2, 4, 6) in NCTHW -> channel order (C, ft, fs_h, fs_w).
  x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
  x = x.reshape(b, t // factor_t, h // factor_s, w // factor_s, out_channels, group_size)
  return jnp.mean(x, axis=-1)


class WanDownResidualBlock(nnx.Module):
  """``DownResidualBlock`` from ``vae2_2.py``: ``mult`` ResBlocks + optional ``Resample`` + ``AvgDown3D``."""

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      mult: int = 2,
      *,
      temporal_downsample: bool = False,
      down_flag: bool = False,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.factor_t = 2 if temporal_downsample else 1
    self.factor_s = 2 if down_flag else 1

    self.resnets = nnx.List([
        WanResidualBlock(in_dim if i == 0 else out_dim, out_dim, dtype=dtype)
        for i in range(mult)
    ])
    if down_flag:
      mode = "downsample3d" if temporal_downsample else "downsample2d"
      self.resample = WanDownResample(out_dim, mode=mode, dtype=dtype)
    else:
      self.resample = None

  def __call__(
      self,
      x: jax.Array,
      feat_cache: list[jax.Array | None] | None = None,
      feat_idx: list[int] | None = None,
  ) -> jax.Array:
    x_copy = x
    for res in self.resnets:
      x = res(x, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.resample is not None:
      x = self.resample(x, feat_cache=feat_cache, feat_idx=feat_idx)
    return x + _avg_down_3d_ndhwc(
        x_copy, self.in_dim, self.out_dim, self.factor_t, self.factor_s
    )


class Wan2_2_VAEEncoder(nnx.Module):
  """Encoder-only Wan2.2 VAE (``float32``) matching ``vae2_2.py`` / ``Wan2.2_VAE.pth``.

  Default hyperparameters match MAGI-2-preview:
  ``dim=160, z_dim=48, dim_mult=(1, 2, 4, 4), num_res_blocks=2,
  temporal_downsample=(False, True, True), patch_size=2``.
  """

  def __init__(
      self,
      dim: int = 160,
      z_dim: int = 48,
      dim_mult: tuple[int, ...] = (1, 2, 4, 4),
      num_res_blocks: int = 2,
      temporal_downsample: tuple[bool, ...] = (False, True, True),
      patch_size: int = 2,
      *,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.dim = int(dim)
    self.z_dim = int(z_dim)
    self.dim_mult = tuple(dim_mult)
    self.num_res_blocks = int(num_res_blocks)
    self.temporal_downsample = tuple(temporal_downsample)
    self.patch_size = int(patch_size)
    self.dtype = dtype

    dims = [self.dim * u for u in [1] + list(self.dim_mult)]
    # Patchified RGB input has 3 * patch_size^2 = 12 channels.
    self.encoder_conv1 = WanCausalConv3d(
        3 * self.patch_size * self.patch_size, dims[0], 3, padding=1, dtype=dtype
    )

    downsamples = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      t_down = (
          self.temporal_downsample[i]
          if i < len(self.temporal_downsample)
          else False
      )
      down_flag = i != len(self.dim_mult) - 1
      downsamples.append(
          WanDownResidualBlock(
              in_dim,
              out_dim,
              mult=self.num_res_blocks,
              temporal_downsample=t_down,
              down_flag=down_flag,
              dtype=dtype,
          )
      )
    self.downsamples = nnx.List(downsamples)

    mid_dim = dims[-1]
    self.middle_res0 = WanResidualBlock(mid_dim, mid_dim, dtype=dtype)
    self.middle_attn = WanAttentionBlock(mid_dim, dtype=dtype)
    self.middle_res2 = WanResidualBlock(mid_dim, mid_dim, dtype=dtype)

    self.head_norm = WanRMSNorm(mid_dim, dtype=dtype)
    self.head_conv = WanCausalConv3d(mid_dim, self.z_dim * 2, 3, padding=1, dtype=dtype)
    # Post-encoder 1x1x1 quant conv (top-level `conv1` in WanVAE_).
    self.quant_conv = WanCausalConv3d(
        self.z_dim * 2, self.z_dim * 2, 1, padding=0, dtype=dtype
    )

    # Number of active cache slots (conv1 + 2 per ResBlock + 1 per downsample3d + head_conv).
    n_resblocks = len(self.dim_mult) * self.num_res_blocks + 2
    n_tdown = sum(
        1
        for i in range(len(self.dim_mult) - 1)
        if i < len(self.temporal_downsample) and self.temporal_downsample[i]
    )
    self._num_cache_slots = 1 + 2 * n_resblocks + n_tdown + 1

  def _encode_chunk(
      self,
      x_ndhwc: jax.Array,
      feat_cache: list[jax.Array | None],
  ) -> jax.Array:
    feat_idx = [0]
    x = _run_cached_causal_conv3d(self.encoder_conv1, x_ndhwc, feat_cache, feat_idx)
    for stage in self.downsamples:
      x = stage(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = self.middle_res0(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = self.middle_attn(x)
    x = self.middle_res2(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = jax.nn.silu(self.head_norm(x))
    x = _run_cached_causal_conv3d(self.head_conv, x, feat_cache, feat_idx)
    return x

  def encode(
      self,
      video: jax.Array,
      *,
      input_layout: str = "ncthw",
      output_layout: str = "ncthw",
      normalize: bool = True,
  ) -> jax.Array:
    """Encode a reference image ``(T=1)`` or video ``(T=1+4k)`` into normalized latents.

    Args:
      video: RGB tensor in ``[-1, 1]``, shape ``(B, 3, T, H, W)`` if
        ``input_layout="ncthw"`` or ``(B, T, H, W, 3)`` if ``"ndhwc"``.
      input_layout: ``"ncthw"`` or ``"ndhwc"``.
      output_layout: ``"ncthw"`` (returns ``(B, 48, T_lat, H//16, W//16)``) or
        ``"ndhwc"`` (returns ``(B, T_lat, H//16, W//16, 48)``).
      normalize: If True, applies 48-ch Wan2.2 normalization ``(mu - mean) / std``.

    Returns:
      Normalized ``float32`` latent mean ``mu``.
    """
    if input_layout == "ncthw":
      x = jnp.transpose(video, (0, 2, 3, 4, 1)).astype(self.dtype)
    elif input_layout == "ndhwc":
      x = video.astype(self.dtype)
    else:
      raise ValueError(f"Unsupported input_layout: {input_layout!r}")

    x = wan_patchify_ndhwc(x, patch_size=self.patch_size)
    t = int(x.shape[1])
    iter_ = 1 + (t - 1) // 4
    feat_cache: list[jax.Array | None] = [None] * self._num_cache_slots

    chunks = []
    for i in range(iter_):
      if i == 0:
        x_chunk = x[:, :1, :, :, :]
      else:
        x_chunk = x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :]
      chunks.append(self._encode_chunk(x_chunk, feat_cache))

    out = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=1)
    y = self.quant_conv(out)
    mu, _log_var = jnp.split(y, 2, axis=-1)

    mu = mu.astype(jnp.float32)
    if normalize:
      mean = jnp.asarray(WAN22_LATENT_MEAN, dtype=jnp.float32).reshape(1, 1, 1, 1, -1)
      inv_std = 1.0 / jnp.asarray(WAN22_LATENT_STD, dtype=jnp.float32).reshape(
          1, 1, 1, 1, -1
      )
      mu = (mu - mean) * inv_std

    if output_layout == "ncthw":
      return jnp.transpose(mu, (0, 4, 1, 2, 3))
    if output_layout == "ndhwc":
      return mu
    raise ValueError(f"Unsupported output_layout: {output_layout!r}")

  def __call__(self, video: jax.Array, **kwargs: Any) -> jax.Array:
    return self.encode(video, **kwargs)


def _iter_wan_resblock_params(
    prefix: str, block: WanResidualBlock
) -> Iterator[tuple[str, nnx.Param, str]]:
  yield f"{prefix}.residual.0.gamma", block.norm1.gamma, "gamma"
  yield f"{prefix}.residual.2.weight", block.conv1.kernel, "conv3d"
  yield f"{prefix}.residual.2.bias", block.conv1.bias, "bias"
  yield f"{prefix}.residual.3.gamma", block.norm2.gamma, "gamma"
  yield f"{prefix}.residual.6.weight", block.conv2.kernel, "conv3d"
  yield f"{prefix}.residual.6.bias", block.conv2.bias, "bias"
  if block.shortcut is not None:
    yield f"{prefix}.shortcut.weight", block.shortcut.kernel, "conv3d"
    yield f"{prefix}.shortcut.bias", block.shortcut.bias, "bias"


def _iter_wan22_encoder_params(
    model: Wan2_2_VAEEncoder,
) -> Iterator[tuple[str, nnx.Param, str]]:
  """Yield all 86 ``(wan22_pth_key, nnx.Param, kind)`` pairs for the encoder."""
  yield "encoder.conv1.weight", model.encoder_conv1.kernel, "conv3d"
  yield "encoder.conv1.bias", model.encoder_conv1.bias, "bias"

  for s_idx, stage in enumerate(model.downsamples):
    for r_idx, res in enumerate(stage.resnets):
      yield from _iter_wan_resblock_params(
          f"encoder.downsamples.{s_idx}.downsamples.{r_idx}", res
      )
    if stage.resample is not None:
      rs_prefix = f"encoder.downsamples.{s_idx}.downsamples.{len(stage.resnets)}"
      yield f"{rs_prefix}.resample.1.weight", stage.resample.conv.kernel, "conv2d"
      yield f"{rs_prefix}.resample.1.bias", stage.resample.conv.bias, "bias"
      if stage.resample.time_conv is not None:
        yield f"{rs_prefix}.time_conv.weight", stage.resample.time_conv.kernel, "conv3d"
        yield f"{rs_prefix}.time_conv.bias", stage.resample.time_conv.bias, "bias"

  yield from _iter_wan_resblock_params("encoder.middle.0", model.middle_res0)
  yield "encoder.middle.1.norm.gamma", model.middle_attn.norm.gamma, "gamma"
  yield "encoder.middle.1.to_qkv.weight", model.middle_attn.to_qkv.kernel, "conv2d"
  yield "encoder.middle.1.to_qkv.bias", model.middle_attn.to_qkv.bias, "bias"
  yield "encoder.middle.1.proj.weight", model.middle_attn.proj.kernel, "conv2d"
  yield "encoder.middle.1.proj.bias", model.middle_attn.proj.bias, "bias"
  yield from _iter_wan_resblock_params("encoder.middle.2", model.middle_res2)

  yield "encoder.head.0.gamma", model.head_norm.gamma, "gamma"
  yield "encoder.head.2.weight", model.head_conv.kernel, "conv3d"
  yield "encoder.head.2.bias", model.head_conv.bias, "bias"

  yield "conv1.weight", model.quant_conv.kernel, "conv3d"
  yield "conv1.bias", model.quant_conv.bias, "bias"


def load_wan22_vae_encoder_weights(
    model: Wan2_2_VAEEncoder,
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
    verbose: bool = False,
) -> Wan2_2_VAEEncoder:
  """Populate :class:`Wan2_2_VAEEncoder` from ``Wan2.2_VAE.pth`` state dict."""
  if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
    state_dict = state_dict["state_dict"]
  elif "model" in state_dict and isinstance(state_dict["model"], dict):
    state_dict = state_dict["model"]

  cleaned: dict[str, Any] = {}
  skipped_decoder = 0
  for key, val in state_dict.items():
    k = key[7:] if key.startswith("module.") else key
    if k.startswith("decoder.") or k.startswith("conv2."):
      skipped_decoder += 1
      continue
    cleaned[k] = val

  expected_map = list(_iter_wan22_encoder_params(model))
  expected_keys = {k for k, _, _ in expected_map}

  missing: list[str] = []
  loaded = 0
  for key, param, kind in expected_map:
    if key not in cleaned:
      missing.append(key)
      continue
    arr = _to_numpy_f32(cleaned[key])
    if kind == "conv3d":
      arr = _transpose_conv3d_oidhw_to_dhwio(arr)
    elif kind == "conv2d":
      arr = _transpose_conv2d_oihw_to_hwio(arr)
    else:
      # RMSNorm gamma stored as (C, 1, 1, 1) or (C, 1, 1) -> reshape to (C,).
      arr = arr.reshape(param[...].shape)
    if arr.shape != param[...].shape:
      raise ValueError(
          f"Wan2_2_VAEEncoder shape mismatch for {key!r}: got {arr.shape}, "
          f"expected {param[...].shape}"
      )
    param[...] = jnp.asarray(arr, dtype=param[...].dtype)
    loaded += 1

  unexpected = sorted(set(cleaned.keys()) - expected_keys)
  total_nnx_params = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if strict and (missing or unexpected or loaded != total_nnx_params):
    raise ValueError(
        f"Wan2_2_VAEEncoder weight load mismatch: loaded={loaded}/{total_nnx_params}, "
        f"missing={missing[:10]} (total {len(missing)}), "
        f"unexpected={unexpected[:10]} (total {len(unexpected)})"
    )
  if verbose:
    print(
        f"[magi2] Wan2_2_VAEEncoder loaded {loaded}/{total_nnx_params} tensors "
        f"(skipped {skipped_decoder} decoder/conv2 tensors, dtype={model.dtype})"
    )
  return model


def load_wan22_vae_encoder(
    ckpt_path: str | Path,
    *,
    dtype: jnp.dtype = jnp.float32,
    strict: bool = True,
    verbose: bool = False,
) -> Wan2_2_VAEEncoder:
  """Load :class:`Wan2_2_VAEEncoder` from ``vae/Wan2.2_VAE.pth`` (or ``.safetensors``)."""
  model = Wan2_2_VAEEncoder(dtype=dtype)
  ckpt_path = Path(ckpt_path)
  if ckpt_path.is_dir():
    ckpt_path = ckpt_path / "Wan2.2_VAE.pth"

  if ckpt_path.suffix == ".safetensors":
    from diffusers_tpu.safetensors_io import safe_open

    state_dict: dict[str, Any] = {}
    with safe_open(str(ckpt_path), framework="np") as reader:
      for k in reader.keys():
        if k.startswith("decoder.") or k.startswith("conv2."):
          continue
        state_dict[k] = reader.get_tensor(k)
  else:
    import torch

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

  return load_wan22_vae_encoder_weights(model, state_dict, strict=strict, verbose=verbose)


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
  print("=== Running MAGI-2 Video VAE self-tests ===")

  # 1. Exact (c r q) patchify / unpatchify round-trip and axis-order check.
  rng = np.random.default_rng(0)
  x_img = jnp.asarray(rng.standard_normal((1, 2, 8, 12, 3), dtype=np.float32))
  patched = wan_patchify_ndhwc(x_img, patch_size=2)
  assert patched.shape == (1, 2, 4, 6, 12), f"unexpected patch shape {patched.shape}"
  # Verify r (width) precedes q (height): out_c = (c*2 + r)*2 + q.
  for c in range(3):
    for r in range(2):
      for q in range(2):
        expected = x_img[:, :, q::2, r::2, c]
        got = patched[:, :, :, :, (c * 2 + r) * 2 + q]
        assert np.allclose(np.asarray(got), np.asarray(expected), atol=0.0)
  unpatched = turbo_unpatchify_ndhwc(patched, patch_size=2)
  assert np.array_equal(np.asarray(unpatched), np.asarray(x_img)), "patchify/unpatchify mismatch"
  print("[PASS] (c r q) patchify <-> turbo_unpatchify round-trip exact")

  # 2. TurboVAED small-channel shape, sliding-window frame count & weight-loader tests.
  small_turbo_cfg = TurboVAEDConfig(
      latent_channels=48,
      out_channels=3,
      block_out_channels=(8, 16, 16, 32),
      layers_per_block=(1, 1, 1, 1, 1),
      spatio_temporal_scaling=(False, True, True, True),
      spatio_only=(False, True, False, False),
      decoder_is_dw_conv=(False, False, True, True, False),
      decoder_dw_kernel_size=5,
      patch_size=2,
      use_unpatchify=True,
      first_chunk_size=7,
      step_size=7,
  )
  turbo = TurboVAED(small_turbo_cfg, dtype=jnp.float32)

  # Build synthetic PyTorch-shaped checkpoint (OIDHW / OIHW) and test strict loader.
  syn_sd: dict[str, np.ndarray] = {}
  for key, param, kind in _iter_turbo_vaed_params(turbo.decoder):
    shape = param[...].shape
    if kind == "conv3d":
      # DHWIO -> OIDHW
      pt_shape = (shape[4], shape[3], shape[0], shape[1], shape[2])
    elif kind == "conv2d":
      # HWIO -> OIHW
      pt_shape = (shape[3], shape[2], shape[0], shape[1])
    else:
      pt_shape = shape
    syn_sd["module.decoder." + key] = rng.standard_normal(pt_shape).astype(np.float32) * 0.02
  syn_sd["module.encoder.ignored.weight"] = np.zeros((1,), dtype=np.float32)
  syn_sd["module.discriminator.ignored.weight"] = np.zeros((1,), dtype=np.float32)
  load_turbo_vaed_weights(turbo, {"ema_state_dict": syn_sd}, strict=True, verbose=True)

  # Verify sliding-window output frame counts: T=1 -> 1, T=7 -> 25, T=32 -> 125, T=63 -> 249.
  for t_lat, want_t in ((1, 1), (7, 25), (14, 53), (32, 125), (63, 249)):
    z = jnp.asarray(rng.standard_normal((1, 48, t_lat, 2, 2), dtype=np.float32))
    out = turbo.decode(z)
    assert out.shape == (1, 3, want_t, 32, 32), (
        f"TurboVAED T={t_lat}: got {out.shape}, want (1, 3, {want_t}, 32, 32)"
    )
    assert np.all(np.isfinite(np.asarray(out))), f"Non-finite output for T={t_lat}"
  print("[PASS] TurboVAED sliding-window shapes (T=1->1, 7->25, 14->53, 32->125, 63->249) & strict loader")

  # Verify ReLU vs SiLU quirk: up_blocks[0..2] have spatio_temporal_scale=True -> ReLU.
  assert turbo.decoder.up_blocks[0].resnets[0].is_upsampler_modified is True
  assert turbo.decoder.up_blocks[1].resnets[0].is_upsampler_modified is True
  assert turbo.decoder.up_blocks[2].resnets[0].is_upsampler_modified is True
  assert turbo.decoder.up_blocks[3].resnets[0].is_upsampler_modified is False
  assert turbo.decoder.up_blocks[1].conv_in is not None
  assert turbo.decoder.up_blocks[1].conv_in.is_upsampler_modified is False
  print("[PASS] TurboVAED is_upsampler_modified (ReLU vs SiLU) flags verified")

  # 3. Wan2_2_VAEEncoder small-channel shape, T=1 & T=5 causal chunking, and strict loader.
  wan_enc = Wan2_2_VAEEncoder(
      dim=16,
      z_dim=48,
      dim_mult=(1, 2, 4, 4),
      num_res_blocks=2,
      temporal_downsample=(False, True, True),
      dtype=jnp.float32,
  )
  wan_params = list(_iter_wan22_encoder_params(wan_enc))
  assert len(wan_params) == 86, f"Expected 86 encoder params, got {len(wan_params)}"
  wan_sd: dict[str, np.ndarray] = {}
  for key, param, kind in wan_params:
    shape = param[...].shape
    if kind == "conv3d":
      pt_shape = (shape[4], shape[3], shape[0], shape[1], shape[2])
    elif kind == "conv2d":
      pt_shape = (shape[3], shape[2], shape[0], shape[1])
    elif kind == "gamma":
      pt_shape = (shape[0], 1, 1) if "middle.1.norm" in key else (shape[0], 1, 1, 1)
    else:
      pt_shape = shape
    wan_sd[key] = (
        np.ones(pt_shape, dtype=np.float32)
        if kind == "gamma"
        else rng.standard_normal(pt_shape).astype(np.float32) * 0.02
    )
  wan_sd["decoder.ignored.weight"] = np.zeros((1,), dtype=np.float32)
  wan_sd["conv2.weight"] = np.zeros((48, 48, 1, 1, 1), dtype=np.float32)
  load_wan22_vae_encoder_weights(wan_enc, wan_sd, strict=True, verbose=True)

  for t_in, want_t_lat in ((1, 1), (5, 2)):
    vid = jnp.asarray(rng.standard_normal((1, 3, t_in, 32, 48), dtype=np.float32))
    lat = wan_enc.encode(vid)
    assert lat.shape == (1, 48, want_t_lat, 2, 3), (
        f"Wan2_2_VAEEncoder T={t_in}: got {lat.shape}, want (1, 48, {want_t_lat}, 2, 3)"
    )
    assert lat.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(lat)))
  print("[PASS] Wan2_2_VAEEncoder (86 params, T=1->1, T=5->2) & strict loader")
  print("=== All MAGI-2 Video VAE self-tests passed! ===")


if __name__ == "__main__":
  _run_self_tests()
