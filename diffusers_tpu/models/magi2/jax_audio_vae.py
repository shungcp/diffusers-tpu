"""Stereo Audio VAE (`AutoencoderOobleck`) for MAGI-2-preview in native JAX / Flax NNX.

Self-contained JAX/Flax NNX decoder port of Stable Audio Open 1.0's Oobleck
autoencoder matching ``inference/model/sa_audio_vae.py`` and
``inference/pipeline/audio_decoder.py``:

* Config defaults: ``channels=128``, ``c_mults=(1, 2, 4, 8, 16)`` (prepended
  with ``1`` -> ``[1, 1, 2, 4, 8, 16]``), ``strides=(2, 4, 4, 8, 8)`` (total
  upsampling ``2048x``), ``latent_dim=64``, ``io_channels=2`` (stereo),
  ``use_snake=True`` (log-scale :class:`SnakeBeta`), ``final_tanh=False``,
  3 :class:`ResidualUnit` blocks per decoder stage with dilations ``(1, 3, 9)``.
* Selective safetensors loading via :func:`diffusers_tpu.safetensors_io.safe_open`:
  reads ONLY ``pretransform.model.decoder.*`` tensors from
  ``stable-audio-open-1.0/model.safetensors``, completely skipping the unused
  3.94 GiB diffusion weights and encoder weights on disk.
* Load-time fusion of legacy PyTorch ``torch.nn.utils.weight_norm``
  (``weight_g``, ``weight_v`` -> ``weight = g * v / ||v||`` with default
  ``dim=0`` for BOTH ``Conv1d`` ``(O, I, K)`` and ``ConvTranspose1d``
  ``(I, O, K)``), transposed into channels-last ``WIO`` ``(K, C_in, C_out)``
  layout for optimal TPU execution in ``NWC`` ``(B, T, C)``.
* Exact :func:`resample_audio_25hz_to_44100` using ``scipy.signal.resample``
  with ratio ``441 / 512`` (``512,000 -> 441,000`` stereo @ 44.1 kHz,
  compensating for DiT's 25.0 Hz audio latent grid vs Oobleck's native
  ``44100 / 2048 = 21.533203125 Hz`` latent rate).

Self-test::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.magi2.jax_audio_vae
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.magi2.config import Magi2AudioVAEConfig


SA_RESCALE_MEAN: tuple[float, ...] = (0.0,)
SA_RESCALE_STD: tuple[float, ...] = (1.0,)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioVAEConfig:
  """Configuration for Stable Audio Open 1.0 Oobleck stereo audio decoder."""

  sample_rate: int = 44100
  io_channels: int = 2
  latent_dim: int = 64
  channels: int = 128
  c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
  strides: tuple[int, ...] = (2, 4, 4, 8, 8)
  use_snake: bool = True
  use_nearest_upsample: bool = False
  final_tanh: bool = False
  soft_clip: bool = False
  dit_latent_fps: float = 25.0
  rescale_mean: tuple[float, ...] = SA_RESCALE_MEAN
  rescale_std: tuple[float, ...] = SA_RESCALE_STD

  @property
  def downsampling_ratio(self) -> int:
    r = 1
    for s in self.strides:
      r *= int(s)
    return r

  @property
  def native_latent_fps(self) -> float:
    return self.sample_rate / self.downsampling_ratio

  @classmethod
  def from_model_config_dict(cls, full_config: dict[str, Any]) -> "AudioVAEConfig":
    """Parse ``stable-audio-open-1.0/model_config.json`` or an autoencoder dict."""
    sample_rate = int(full_config.get("sample_rate", 44100))
    if "model" in full_config and "pretransform" in full_config["model"]:
      vae_cfg = full_config["model"]["pretransform"]["config"]
    elif "model" in full_config and "decoder" in full_config["model"]:
      vae_cfg = full_config["model"]
    else:
      vae_cfg = full_config

    dec_wrapper = vae_cfg.get("decoder", {})
    dec_cfg = dec_wrapper.get("config", dec_wrapper)
    return cls(
        sample_rate=sample_rate,
        io_channels=int(dec_cfg.get("out_channels", vae_cfg.get("io_channels", 2))),
        latent_dim=int(dec_cfg.get("latent_dim", vae_cfg.get("latent_dim", 64))),
        channels=int(dec_cfg.get("channels", 128)),
        c_mults=tuple(int(v) for v in dec_cfg.get("c_mults", (1, 2, 4, 8, 16))),
        strides=tuple(int(v) for v in dec_cfg.get("strides", (2, 4, 4, 8, 8))),
        use_snake=bool(dec_cfg.get("use_snake", True)),
        use_nearest_upsample=bool(dec_cfg.get("use_nearest_upsample", False)),
        final_tanh=bool(dec_cfg.get("final_tanh", False)),
        soft_clip=bool(dec_wrapper.get("soft_clip", False)),
    )

  @classmethod
  def from_json(cls, path: str | Path) -> "AudioVAEConfig":
    with open(path, "r", encoding="utf-8") as fh:
      return cls.from_model_config_dict(json.load(fh))

  @classmethod
  def from_config(
      cls, cfg: "AudioVAEConfig | Magi2AudioVAEConfig | dict[str, Any] | str | Path | None"
  ) -> "AudioVAEConfig":
    if cfg is None:
      return cls()
    if isinstance(cfg, cls):
      return cfg
    if isinstance(cfg, Magi2AudioVAEConfig):
      return cls(
          sample_rate=cfg.sample_rate,
          io_channels=cfg.io_channels,
          latent_dim=cfg.latent_dim,
          channels=cfg.channels,
          c_mults=tuple(cfg.c_mults),
          strides=tuple(cfg.strides),
          final_tanh=cfg.final_tanh,
          dit_latent_fps=cfg.dit_latent_fps,
      )
    if isinstance(cfg, dict):
      return cls.from_model_config_dict(cfg)
    if isinstance(cfg, (str, Path)):
      p = Path(cfg)
      if p.is_dir():
        p = p / "model_config.json"
      return cls.from_json(p)
    raise TypeError(f"Unsupported AudioVAEConfig source: {type(cfg)!r}")


# ---------------------------------------------------------------------------
# Weight-Norm Fusion & Audio Post-Processing Helpers
# ---------------------------------------------------------------------------


def fuse_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
  """Fuse legacy PyTorch ``torch.nn.utils.weight_norm`` (``dim=0``) into ``weight``.

  In ``sa_audio_vae.py``, both ``wn_conv1d`` (weight shape ``(C_out, C_in, K)``)
  and ``wn_conv_transpose1d`` (weight shape ``(C_in, C_out, K)``) call
  ``weight_norm(module)`` with default ``dim=0``. The L2 norm is therefore
  reduced over ``axis=(1, 2)`` (all axes except ``dim=0``) in both cases:

      ``w = weight_g * weight_v / ||weight_v||_{axes=(1, 2)}``
  """
  g = np.asarray(weight_g, dtype=np.float32)
  v = np.asarray(weight_v, dtype=np.float32)
  norm = np.sqrt(np.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
  return g * (v / np.maximum(norm, 1e-12))


def resample_audio_25hz_to_44100(
    audio_2xN: np.ndarray,
    *,
    axis: int = -1,
    ratio: float = 441.0 / 512.0,
) -> np.ndarray:
  """Exact sinc/FFT resampling from DiT's 25 Hz audio latent rate to 44.1 kHz.

  Oobleck decodes 250 latent frames at ``2048x`` -> ``512,000`` samples
  (corresponding to 10 s at ``25 * 2048 = 51,200 Hz``). Multiplying length by
  ``441 / 512`` via ``scipy.signal.resample`` converts ``(2, 512_000)`` ->
  ``(2, 441_000)`` stereo @ 44.1 kHz (exact match to ``inference_engine.py:L681``).

  Args:
    audio_2xN: Stereo waveform array of shape ``(2, N)`` or ``(B, 2, N)``
      (time along ``axis=-1`` by default).
    axis: Time axis to resample along (default ``-1`` for ``(2, N)``).
    ratio: Length scaling ratio (default ``441 / 512``).

  Returns:
    Resampled ``float32`` stereo waveform array with length
    ``int(audio_2xN.shape[axis] * ratio)`` along ``axis``.
  """
  from scipy.signal import resample

  arr = np.asarray(audio_2xN, dtype=np.float32)
  new_length = int(arr.shape[axis] * ratio)
  out = resample(arr, new_length, axis=axis)
  return np.asarray(out, dtype=np.float32)


def resample_audio_sinc(
    audio: np.ndarray,
    time_stretching: float = 441.0 / 512.0,
    *,
    axis: int = 0,
) -> np.ndarray:
  """Upstream ``audio_decoder.py:resample_audio_sinc`` for ``(N, C)`` waveforms."""
  return resample_audio_25hz_to_44100(audio, axis=axis, ratio=time_stretching)


def adaptive_volume_scaling(
    audio_samples: np.ndarray,
    target_rms_db: float = -31.0,
    max_gain_db: float = 0.0,
) -> np.ndarray:
  """Exact port of ``audio_decoder.py:adaptive_volume_scaling``."""
  arr = np.asarray(audio_samples, dtype=np.float32)
  rms = float(np.sqrt(np.mean(arr**2)))
  if rms < 1e-10:
    return arr
  current_rms_db = 20.0 * math.log10(rms + 1e-10)
  desired_gain_db = target_rms_db - current_rms_db
  actual_gain_db = min(desired_gain_db, max_gain_db)
  scaled = arr * (10.0 ** (actual_gain_db / 20.0))
  peak = float(np.max(np.abs(scaled)))
  if peak > 1.0:
    scaled = scaled / peak
  return scaled


# ---------------------------------------------------------------------------
# Flax NNX Layers (Channels-Last NWC: (B, T, C), Kernels WIO: (K, C_in, C_out))
# ---------------------------------------------------------------------------


class SnakeBeta(nnx.Module):
  """Log-scale ``SnakeBeta`` activation matching ``sa_audio_vae.py:SnakeBeta``.

  Computes in ``float32``::

      alpha = exp(self.alpha)  # if alpha_logscale else self.alpha
      beta  = exp(self.beta)   # if alpha_logscale else self.beta
      y = x + (1 / (beta + 1e-9)) * sin(x * alpha)^2
  """

  def __init__(
      self,
      channels: int,
      *,
      alpha_logscale: bool = True,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.alpha_logscale = bool(alpha_logscale)
    self.alpha = nnx.Param(jnp.zeros((channels,), dtype))
    self.beta = nnx.Param(jnp.zeros((channels,), dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    xf = x.astype(jnp.float32)
    a = self.alpha[...].astype(jnp.float32)
    b = self.beta[...].astype(jnp.float32)
    if self.alpha_logscale:
      a = jnp.exp(a)
      b = jnp.exp(b)
    out = xf + jnp.reciprocal(b + 1e-9) * jnp.square(jnp.sin(xf * a))
    return out.astype(x.dtype)


class ELUActivation(nnx.Module):
  """Stateless ELU fallback when ``use_snake=False``."""

  def __call__(self, x: jax.Array) -> jax.Array:
    return jax.nn.elu(x)


class OobleckConv1d(nnx.Module):
  """1D convolution in ``NWC`` layout with ``WIO`` weights ``(K, C_in, C_out)``."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      *,
      stride: int = 1,
      padding: int | str = 0,
      dilation: int = 1,
      use_bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.kernel_size = int(kernel_size)
    self.stride = int(stride)
    self.padding = padding
    self.dilation = int(dilation)
    self.kernel = nnx.Param(jnp.zeros((kernel_size, in_channels, out_channels), dtype))
    self.bias = nnx.Param(jnp.zeros((out_channels,), dtype)) if use_bias else None

  def __call__(self, x: jax.Array) -> jax.Array:
    if isinstance(self.padding, str):
      pad_spec: Any = self.padding.upper()
    else:
      pad_spec = ((int(self.padding), int(self.padding)),)
    y = jax.lax.conv_general_dilated(
        x,
        self.kernel[...].astype(x.dtype),
        window_strides=(self.stride,),
        padding=pad_spec,
        rhs_dilation=(self.dilation,),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    if self.bias is not None:
      y = y + self.bias[...].astype(x.dtype)
    return y


class OobleckConvTranspose1d(nnx.Module):
  """Exact PyTorch ``ConvTranspose1d`` in ``NWC`` layout with ``WIO`` weights.

  PyTorch ``ConvTranspose1d(in_ch, out_ch, k, stride=s, padding=p, output_padding=op)``
  corresponds to ``lax.conv_general_dilated`` with reversed kernel ``w[::-1]``,
  ``lhs_dilation=(s,)``, ``window_strides=(1,)``, and padding
  ``((k - 1 - p, k - 1 - p + op),)``.

  Note that ``sa_audio_vae.py`` uses ``k = 2 * stride``, ``p = ceil(stride / 2)``,
  and default ``output_padding = 0``. For all even strides ``(2, 4, 4, 8, 8)``,
  this upsamples length by an exact factor of ``stride``.
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      *,
      stride: int = 1,
      padding: int = 0,
      output_padding: int = 0,
      use_bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.kernel_size = int(kernel_size)
    self.stride = int(stride)
    self.padding = int(padding)
    self.output_padding = int(output_padding)
    # Stored as [K, C_in, C_out] (transposed from PyTorch [C_in, C_out, K]).
    self.kernel = nnx.Param(jnp.zeros((kernel_size, in_channels, out_channels), dtype))
    self.bias = nnx.Param(jnp.zeros((out_channels,), dtype)) if use_bias else None

  def __call__(self, x: jax.Array) -> jax.Array:
    lo = self.kernel_size - 1 - self.padding
    hi = self.kernel_size - 1 - self.padding + self.output_padding
    y = jax.lax.conv_general_dilated(
        x,
        self.kernel[...][::-1].astype(x.dtype),
        window_strides=(1,),
        padding=((lo, hi),),
        lhs_dilation=(self.stride,),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    if self.bias is not None:
      y = y + self.bias[...].astype(x.dtype)
    return y


class NearestUpsampleConv1d(nnx.Module):
  """Optional nearest-neighbor upsample + bias-free Conv1d (``use_nearest_upsample=True``)."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      stride: int,
      *,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.stride = int(stride)
    self.conv = OobleckConv1d(
        in_channels,
        out_channels,
        kernel_size=2 * stride,
        stride=1,
        padding="same",
        use_bias=False,
        dtype=dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.conv(jnp.repeat(x, self.stride, axis=1))


class ResidualUnit(nnx.Module):
  """Oobleck residual unit: ``Act -> Conv1d(k=7, dilation=d) -> Act -> Conv1d(k=1) + x``."""

  def __init__(
      self,
      channels: int,
      dilation: int,
      *,
      use_snake: bool = True,
      dtype: jnp.dtype = jnp.float32,
  ):
    padding = (dilation * (7 - 1)) // 2
    self.act1: nnx.Module = SnakeBeta(channels, dtype=dtype) if use_snake else ELUActivation()
    self.conv1 = OobleckConv1d(
        channels, channels, kernel_size=7, dilation=dilation, padding=padding, dtype=dtype
    )
    self.act2: nnx.Module = SnakeBeta(channels, dtype=dtype) if use_snake else ELUActivation()
    self.conv2 = OobleckConv1d(channels, channels, kernel_size=1, padding=0, dtype=dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    h = self.act1(x)
    h = self.conv1(h)
    h = self.act2(h)
    h = self.conv2(h)
    return h + x


class DecoderBlock(nnx.Module):
  """Oobleck decoder block: ``Act -> Upsample(stride) -> 3x ResidualUnit(1, 3, 9)``."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      stride: int,
      *,
      use_snake: bool = True,
      use_nearest_upsample: bool = False,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.act: nnx.Module = (
        SnakeBeta(in_channels, dtype=dtype) if use_snake else ELUActivation()
    )
    self.use_nearest_upsample = bool(use_nearest_upsample)
    if use_nearest_upsample:
      self.upsample: nnx.Module = NearestUpsampleConv1d(
          in_channels, out_channels, stride=stride, dtype=dtype
      )
    else:
      self.upsample = OobleckConvTranspose1d(
          in_channels,
          out_channels,
          kernel_size=2 * stride,
          stride=stride,
          padding=math.ceil(stride / 2),
          output_padding=0,
          use_bias=True,
          dtype=dtype,
      )
    self.res_units = nnx.List([
        ResidualUnit(out_channels, dilation=d, use_snake=use_snake, dtype=dtype)
        for d in (1, 3, 9)
    ])

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.upsample(self.act(x))
    for unit in self.res_units:
      x = unit(x)
    return x


class OobleckDecoder(nnx.Module):
  """Oobleck 1D convolutional audio decoder matching ``sa_audio_vae.py:OobleckDecoder``.

  With Stable Audio Open 1.0 config (``channels=128, c_mults=(1, 2, 4, 8, 16),
  strides=(2, 4, 4, 8, 8), latent_dim=64, io_channels=2``):

  - Prepends ``1`` to ``c_mults`` -> ``[1, 1, 2, 4, 8, 16]``.
  - ``conv_in``: ``64 -> 2048`` (``k=7, pad=3, bias=True``)
  - 5 ``DecoderBlock`` stages: ``2048->1024 (s=8)``, ``1024->512 (s=8)``,
    ``512->256 (s=4)``, ``256->128 (s=4)``, ``128->128 (s=2)``
  - ``act_out``: ``SnakeBeta(128)``
  - ``conv_out``: ``128 -> 2`` (``k=7, pad=3, bias=False``!)
  """

  def __init__(self, config: AudioVAEConfig, *, dtype: jnp.dtype = jnp.float32):
    self.config = config
    c_mults = [1] + list(config.c_mults)
    if len(c_mults) - 1 != len(config.strides):
      raise ValueError(
          f"len(c_mults) ({len(config.c_mults)}) must equal len(strides) ({len(config.strides)})"
      )

    self.conv_in = OobleckConv1d(
        config.latent_dim,
        c_mults[-1] * config.channels,
        kernel_size=7,
        padding=3,
        use_bias=True,
        dtype=dtype,
    )

    blocks = []
    for i in range(len(c_mults) - 1, 0, -1):
      blocks.append(
          DecoderBlock(
              in_channels=c_mults[i] * config.channels,
              out_channels=c_mults[i - 1] * config.channels,
              stride=config.strides[i - 1],
              use_snake=config.use_snake,
              use_nearest_upsample=config.use_nearest_upsample,
              dtype=dtype,
          )
      )
    self.blocks = nnx.List(blocks)

    out_c0 = c_mults[0] * config.channels
    self.act_out: nnx.Module = (
        SnakeBeta(out_c0, dtype=dtype) if config.use_snake else ELUActivation()
    )
    # Upstream OobleckDecoder uses bias=False on the final output convolution!
    self.conv_out = OobleckConv1d(
        out_c0,
        config.io_channels,
        kernel_size=7,
        padding=3,
        use_bias=False,
        dtype=dtype,
    )
    self.final_tanh = bool(config.final_tanh)

  def __call__(self, x_nwc: jax.Array) -> jax.Array:
    x = self.conv_in(x_nwc)
    for block in self.blocks:
      x = block(x)
    x = self.conv_out(self.act_out(x))
    if self.final_tanh:
      x = jnp.tanh(x)
    return x


class AutoencoderOobleck(nnx.Module):
  """MAGI-2-preview Stable Audio Open 1.0 stereo audio decoder (Flax NNX)."""

  def __init__(
      self,
      config: AudioVAEConfig | Magi2AudioVAEConfig | dict[str, Any] | str | Path | None = None,
      *,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.config = AudioVAEConfig.from_config(config)
    self.dtype = dtype
    self.decoder = OobleckDecoder(self.config, dtype=dtype)

  @property
  def sample_rate(self) -> int:
    return self.config.sample_rate

  @property
  def downsampling_ratio(self) -> int:
    return self.config.downsampling_ratio

  def decode(
      self,
      latents: jax.Array,
      *,
      input_layout: str = "bct",
      output_layout: str = "bct",
  ) -> jax.Array:
    """Decode continuous 64-dim audio latents into stereo PCM waveforms.

    Args:
      latents: Shape ``(B, 64, T)`` or ``(64, T)`` if ``input_layout="bct"``
        (matching PyTorch ``SAAudioFeatureExtractor.decode``), or ``(B, T, 64)``
        if ``input_layout="btc"``.
      input_layout: ``"bct"`` or ``"btc"``.
      output_layout: ``"bct"`` (returns ``(B, 2, T * 2048)``) or ``"btc"``
        (returns ``(B, T * 2048, 2)``).

    Returns:
      Decoded stereo waveform tensor (``float32`` by default).
    """
    if input_layout == "bct":
      if latents.ndim == 2:
        latents = latents[None, :, :]
      elif latents.ndim != 3:
        raise ValueError(f"Expected 2D or 3D latents, got shape {latents.shape}")
      x = jnp.transpose(latents, (0, 2, 1))
    elif input_layout == "btc":
      if latents.ndim == 2:
        latents = latents[None, :, :]
      elif latents.ndim != 3:
        raise ValueError(f"Expected 2D or 3D latents, got shape {latents.shape}")
      x = latents
    else:
      raise ValueError(f"Unsupported input_layout: {input_layout!r}")

    # Apply rescale_std / rescale_mean (identity [1.0]/[0.0] for Stable Audio Open).
    mean = jnp.asarray(self.config.rescale_mean, dtype=jnp.float32).reshape(1, 1, -1)
    std = jnp.asarray(self.config.rescale_std, dtype=jnp.float32).reshape(1, 1, -1)
    x = (x.astype(jnp.float32) * std + mean).astype(self.dtype)

    out = self.decoder(x)
    if self.config.soft_clip:
      out = jnp.tanh(out)

    if output_layout == "bct":
      return jnp.transpose(out, (0, 2, 1))
    if output_layout == "btc":
      return out
    raise ValueError(f"Unsupported output_layout: {output_layout!r}")

  def decode_to_44100(
      self,
      latents: jax.Array,
      *,
      input_layout: str = "bct",
  ) -> np.ndarray:
    """Decode latents and resample ``441 / 512`` to stereo 44.1 kHz NumPy waveform.

    For ``T = 250`` DiT audio latent frames:
    ``decode`` produces ``(B, 2, 512_000)`` -> resampled to ``(B, 2, 441_000)``
    (or ``(2, 441_000)`` when ``B == 1`` and input was 2D ``(64, 250)``).
    """
    squeeze_batch = latents.ndim == 2
    wav = np.asarray(
        self.decode(latents, input_layout=input_layout, output_layout="bct"),
        dtype=np.float32,
    )
    if squeeze_batch:
      wav = wav[0]
    return resample_audio_25hz_to_44100(wav, axis=-1)

  def __call__(self, latents: jax.Array, **kwargs: Any) -> jax.Array:
    return self.decode(latents, **kwargs)


# ---------------------------------------------------------------------------
# Selective Safetensors Loader (Only `pretransform.model.decoder.*`)
# ---------------------------------------------------------------------------


def _iter_oobleck_conv_specs(
    prefix: str,
    conv: OobleckConv1d | OobleckConvTranspose1d,
    *,
    is_transpose: bool,
) -> Iterator[tuple[str, nnx.Param, str]]:
  yield prefix, conv.kernel, ("conv_t1d" if is_transpose else "conv1d")
  if conv.bias is not None:
    yield f"{prefix}.bias", conv.bias, "bias"


def _iter_oobleck_act_specs(
    prefix: str, act: nnx.Module
) -> Iterator[tuple[str, nnx.Param, str]]:
  if isinstance(act, SnakeBeta):
    yield f"{prefix}.alpha", act.alpha, "vec"
    yield f"{prefix}.beta", act.beta, "vec"


def _iter_oobleck_decoder_specs(
    decoder: OobleckDecoder,
) -> Iterator[tuple[str, nnx.Param, str]]:
  """Yield ``(decoder_rel_prefix_or_key, nnx.Param, kind)`` for all OobleckDecoder params.

  Matches the exact ``nn.Sequential`` layer indices of ``sa_audio_vae.py:OobleckDecoder``:
  - ``decoder.layers.0``: ``conv_in``
  - ``decoder.layers.{1..N}``: ``DecoderBlock`` (``layers.0`` act, ``layers.1``
    upsample, ``layers.{2,3,4}`` ``ResidualUnit`` with ``layers.{0,1,2,3}``)
  - ``decoder.layers.{N+1}``: ``act_out``
  - ``decoder.layers.{N+2}``: ``conv_out`` (no bias)
  """
  yield from _iter_oobleck_conv_specs("decoder.layers.0", decoder.conv_in, is_transpose=False)

  for idx, block in enumerate(decoder.blocks, start=1):
    b_pre = f"decoder.layers.{idx}.layers"
    yield from _iter_oobleck_act_specs(f"{b_pre}.0", block.act)
    if block.use_nearest_upsample:
      assert isinstance(block.upsample, NearestUpsampleConv1d)
      yield from _iter_oobleck_conv_specs(
          f"{b_pre}.1.1", block.upsample.conv, is_transpose=False
      )
    else:
      assert isinstance(block.upsample, OobleckConvTranspose1d)
      yield from _iter_oobleck_conv_specs(
          f"{b_pre}.1", block.upsample, is_transpose=True
      )
    for r_idx, unit in enumerate(block.res_units, start=2):
      r_pre = f"{b_pre}.{r_idx}.layers"
      yield from _iter_oobleck_act_specs(f"{r_pre}.0", unit.act1)
      yield from _iter_oobleck_conv_specs(f"{r_pre}.1", unit.conv1, is_transpose=False)
      yield from _iter_oobleck_act_specs(f"{r_pre}.2", unit.act2)
      yield from _iter_oobleck_conv_specs(f"{r_pre}.3", unit.conv2, is_transpose=False)

  last = len(decoder.blocks) + 1
  yield from _iter_oobleck_act_specs(f"decoder.layers.{last}", decoder.act_out)
  yield from _iter_oobleck_conv_specs(
      f"decoder.layers.{last + 1}", decoder.conv_out, is_transpose=False
  )


def _extract_fused_conv_weight(
    state: dict[str, np.ndarray], prefix: str
) -> tuple[np.ndarray, list[str]]:
  """Fuse weight_norm (legacy or parametrized) or read plain ``.weight``."""
  g_key = f"{prefix}.weight_g"
  v_key = f"{prefix}.weight_v"
  if g_key in state and v_key in state:
    return fuse_weight_norm(state[g_key], state[v_key]), [g_key, v_key]

  p0_key = f"{prefix}.parametrizations.weight.original0"
  p1_key = f"{prefix}.parametrizations.weight.original1"
  if p0_key in state and p1_key in state:
    return fuse_weight_norm(state[p0_key], state[p1_key]), [p0_key, p1_key]

  w_key = f"{prefix}.weight"
  if w_key in state:
    return np.asarray(state[w_key], dtype=np.float32), [w_key]

  raise KeyError(f"Missing weight_g/weight_v (or weight) for {prefix!r}")


def load_oobleck_decoder_weights(
    model: AutoencoderOobleck,
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
    verbose: bool = False,
) -> AutoencoderOobleck:
  """Load ``pretransform.model.decoder.*`` (or ``decoder.*``) tensors into :class:`AutoencoderOobleck`."""
  decoder_state: dict[str, np.ndarray] = {}
  skipped_other = 0
  for key, val in state_dict.items():
    k = key
    if k.startswith("pretransform.model."):
      k = k[len("pretransform.model."):]
    if not k.startswith("decoder."):
      skipped_other += 1
      continue
    if hasattr(val, "detach"):
      arr = val.detach().cpu().float().numpy()
    else:
      arr = np.asarray(val, dtype=np.float32)
    decoder_state[k] = arr

  consumed: set[str] = set()
  missing: list[str] = []
  loaded = 0

  for key_or_prefix, param, kind in _iter_oobleck_decoder_specs(model.decoder):
    if kind in ("conv1d", "conv_t1d"):
      try:
        w, used_keys = _extract_fused_conv_weight(decoder_state, key_or_prefix)
      except KeyError:
        missing.append(f"{key_or_prefix}.weight_g/weight_v")
        continue
      consumed.update(used_keys)
      # PyTorch Conv1d: (C_out, C_in, K) -> WIO (K, C_in, C_out).
      # PyTorch ConvTranspose1d: (C_in, C_out, K) -> WIO (K, C_in, C_out).
      w = w.transpose(2, 0, 1) if kind == "conv_t1d" else w.transpose(2, 1, 0)
      arr = np.ascontiguousarray(w, dtype=np.float32)
    else:
      if key_or_prefix not in decoder_state:
        missing.append(key_or_prefix)
        continue
      consumed.add(key_or_prefix)
      arr = np.asarray(decoder_state[key_or_prefix], dtype=np.float32).reshape(
          param[...].shape
      )

    if arr.shape != param[...].shape:
      raise ValueError(
          f"AutoencoderOobleck shape mismatch for {key_or_prefix!r}: "
          f"got {arr.shape}, expected {param[...].shape}"
      )
    param[...] = jnp.asarray(arr, dtype=param[...].dtype)
    loaded += 1

  unexpected = sorted(set(decoder_state.keys()) - consumed)
  total_nnx_params = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if strict and (missing or unexpected or loaded != total_nnx_params):
    raise ValueError(
        f"AutoencoderOobleck weight load mismatch: loaded={loaded}/{total_nnx_params}, "
        f"missing={missing[:10]} (total {len(missing)}), "
        f"unexpected={unexpected[:10]} (total {len(unexpected)})"
    )
  if verbose:
    print(
        f"[magi2] AutoencoderOobleck loaded {loaded}/{total_nnx_params} params "
        f"({len(consumed)} ckpt tensors fused, skipped {skipped_other} non-decoder keys, "
        f"sr={model.sample_rate}, ds={model.downsampling_ratio})"
    )
  return model


def load_audio_vae(
    model_path: str | Path,
    *,
    config: AudioVAEConfig | Magi2AudioVAEConfig | dict[str, Any] | None = None,
    dtype: jnp.dtype = jnp.float32,
    strict: bool = True,
    verbose: bool = False,
) -> AutoencoderOobleck:
  """Selectively load ONLY ``pretransform.model.decoder.*`` from ``stable-audio-open-1.0``.

  Uses :func:`diffusers_tpu.safetensors_io.safe_open` so the 3.94 GiB diffusion
  backbone weights in ``model.safetensors`` are never read from disk.

  Args:
    model_path: Directory containing ``model.safetensors`` (and optional
      ``model_config.json``), or direct path to ``model.safetensors``.
    config: Optional explicit config override; if omitted, reads
      ``model_config.json`` when present in ``model_path`` or falls back to
      :class:`AudioVAEConfig` defaults for Stable Audio Open 1.0.
    dtype: Compute/parameter dtype (default ``jnp.float32``).
    strict: Verify every decoder parameter is populated with zero leftovers.
    verbose: Print load summary.
  """
  from diffusers_tpu.safetensors_io import safe_open

  path = Path(model_path)
  if path.is_dir():
    weights_path = path / "model.safetensors"
    cfg_path = path / "model_config.json"
  else:
    weights_path = path
    cfg_path = path.parent / "model_config.json"

  if not weights_path.exists():
    raise FileNotFoundError(f"Audio VAE safetensors file not found: {weights_path}")

  if config is not None:
    resolved_cfg = AudioVAEConfig.from_config(config)
  elif cfg_path.exists():
    resolved_cfg = AudioVAEConfig.from_json(cfg_path)
  else:
    resolved_cfg = AudioVAEConfig()

  model = AutoencoderOobleck(resolved_cfg, dtype=dtype)

  # Read ONLY `pretransform.model.decoder.*` (or standalone `decoder.*`) from disk!
  # Calling reader.keys() parses only the safetensors JSON header; get_tensor()
  # seeks directly to each decoder tensor, skipping ~3.94 GiB of diffusion weights.
  state_dict: dict[str, np.ndarray] = {}
  skipped_diffusion = 0
  skipped_encoder = 0
  with safe_open(str(weights_path), framework="np") as reader:
    for key in reader.keys():
      if key.startswith("pretransform.model.decoder.") or key.startswith("decoder."):
        state_dict[key] = reader.get_tensor(key)
      elif key.startswith("pretransform.model.encoder.") or key.startswith("encoder."):
        skipped_encoder += 1
      else:
        skipped_diffusion += 1

  if verbose:
    print(
        f"[magi2] Selective safetensors read from {weights_path.name}: "
        f"{len(state_dict)} decoder tensors read, skipped {skipped_encoder} encoder "
        f"& {skipped_diffusion} diffusion/other tensors"
    )
  return load_oobleck_decoder_weights(model, state_dict, strict=strict, verbose=verbose)


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
  print("=== Running MAGI-2 Stereo Audio VAE (Oobleck) self-tests ===")
  rng = np.random.default_rng(0)

  # 1. Verify dim=0 weight_norm fusion for both Conv1d (O, I, K) & ConvTranspose1d (I, O, K).
  for shape in ((16, 8, 7), (16, 8, 16)):
    v = rng.standard_normal(shape).astype(np.float32)
    g = rng.uniform(0.5, 2.0, size=(shape[0], 1, 1)).astype(np.float32)
    fused = fuse_weight_norm(g, v)
    assert fused.shape == shape
    # Every slice along dim=0 must have L2 norm equal to g[i, 0, 0].
    slice_norms = np.sqrt(np.sum(fused * fused, axis=(1, 2)))
    assert np.allclose(slice_norms, g[:, 0, 0], atol=1e-5), "weight_norm dim=0 mismatch"
  print("[PASS] weight_norm fusion (dim=0) verified for Conv1d & ConvTranspose1d")

  # 2. Verify ConvTranspose1d exact upsampling & convolution arithmetic vs manual reference.
  for stride in (2, 4, 8):
    k = 2 * stride
    pad = math.ceil(stride / 2)
    layer = OobleckConvTranspose1d(3, 5, kernel_size=k, stride=stride, padding=pad)
    w_pt = rng.standard_normal((3, 5, k)).astype(np.float32)  # (C_in, C_out, K)
    b_pt = rng.standard_normal((5,)).astype(np.float32)
    layer.kernel[...] = jnp.asarray(w_pt.transpose(2, 0, 1))
    layer.bias[...] = jnp.asarray(b_pt)

    x_np = rng.standard_normal((1, 7, 3)).astype(np.float32)  # NWC
    y = np.asarray(layer(jnp.asarray(x_np)))
    assert y.shape == (1, 7 * stride, 5), f"stride={stride}: got {y.shape}"

    # Direct PyTorch ConvTranspose1d definition: out[j] = sum_{i, u: i*s - pad + u == j} x[i] @ w_pt[:, :, u] + b
    ref = np.broadcast_to(b_pt[None, None, :], (1, 7 * stride, 5)).copy()
    for i in range(7):
      for u in range(k):
        j = i * stride - pad + u
        if 0 <= j < 7 * stride:
          ref[0, j] += x_np[0, i] @ w_pt[:, :, u]
    assert np.allclose(y, ref, atol=1e-5), f"ConvTranspose1d math mismatch for stride={stride}"
  print("[PASS] ConvTranspose1d exact PyTorch math verified for strides (2, 4, 8)")

  # 3. Full 5-stage AutoencoderOobleck (scaled-down channels=4 for fast CPU test)
  #    with exact Stable Audio Open strides (2, 4, 4, 8, 8) -> 2048x upsampling!
  small_cfg = AudioVAEConfig(
      sample_rate=44100,
      io_channels=2,
      latent_dim=64,
      channels=4,
      c_mults=(1, 2, 4, 8, 16),
      strides=(2, 4, 4, 8, 8),
      use_snake=True,
      use_nearest_upsample=False,
      final_tanh=False,
      soft_clip=False,
  )
  assert small_cfg.downsampling_ratio == 2048
  ae = AutoencoderOobleck(small_cfg, dtype=jnp.float32)
  total_params = len(jax.tree.leaves(nnx.state(ae, nnx.Param)))
  assert total_params == 145, f"Expected 145 NNX params for 5-stage Oobleck, got {total_params}"

  # Build synthetic `stable-audio-open-1.0/model.safetensors`-shaped dict with legacy weight_norm.
  syn_sd: dict[str, np.ndarray] = {
      "model.model.transformer.ignored.weight": np.zeros((2, 2), dtype=np.float32),
      "pretransform.model.encoder.layers.0.weight_g": np.ones((4, 1, 1), dtype=np.float32),
  }
  for key_or_prefix, param, kind in _iter_oobleck_decoder_specs(ae.decoder):
    full_pre = f"pretransform.model.{key_or_prefix}"
    shape = param[...].shape
    if kind == "conv1d":
      # WIO (K, I, O) -> PyTorch (O, I, K)
      pt_shape = (shape[2], shape[1], shape[0])
      syn_sd[f"{full_pre}.weight_g"] = rng.uniform(0.5, 1.5, size=(pt_shape[0], 1, 1)).astype(np.float32)
      syn_sd[f"{full_pre}.weight_v"] = rng.standard_normal(pt_shape).astype(np.float32) * 0.02
    elif kind == "conv_t1d":
      # WIO (K, I, O) -> PyTorch (I, O, K)
      pt_shape = (shape[1], shape[2], shape[0])
      syn_sd[f"{full_pre}.weight_g"] = rng.uniform(0.5, 1.5, size=(pt_shape[0], 1, 1)).astype(np.float32)
      syn_sd[f"{full_pre}.weight_v"] = rng.standard_normal(pt_shape).astype(np.float32) * 0.02
    else:
      syn_sd[full_pre] = rng.standard_normal(shape).astype(np.float32) * 0.02

  # Confirm exact count of decoder checkpoint tensors (182) before fusion.
  dec_ckpt_count = sum(1 for k in syn_sd if k.startswith("pretransform.model.decoder."))
  assert dec_ckpt_count == 182, f"Expected 182 decoder ckpt tensors, got {dec_ckpt_count}"
  load_oobleck_decoder_weights(ae, syn_sd, strict=True, verbose=True)

  # Decode 3D (B, 64, T) and 2D (64, T) latents -> 2048x stereo waveform.
  z_3d = jnp.asarray(rng.standard_normal((1, 64, 4), dtype=np.float32))
  wav_3d = ae.decode(z_3d)
  assert wav_3d.shape == (1, 2, 4 * 2048), f"unexpected wav shape {wav_3d.shape}"
  assert np.all(np.isfinite(np.asarray(wav_3d)))

  wav_2d = ae.decode(z_3d[0])
  assert wav_2d.shape == (1, 2, 4 * 2048)
  assert np.allclose(np.asarray(wav_3d), np.asarray(wav_2d), atol=1e-6)
  print("[PASS] AutoencoderOobleck 5-stage decode (182 ckpt -> 145 fused params, 2048x stereo)")

  # 4. Verify 25 Hz -> 44.1 kHz resampling (512,000 -> 441,000).
  try:
    import scipy.signal  # noqa: F401
    has_scipy = True
  except ImportError:
    has_scipy = False

  assert int(512_000 * (441.0 / 512.0)) == 441_000
  if has_scipy:
    t_axis = np.linspace(0.0, 10.0, 512_000, endpoint=False, dtype=np.float32)
    sine = np.stack([np.sin(2 * np.pi * 440.0 * t_axis), np.cos(2 * np.pi * 880.0 * t_axis)])
    resamp = resample_audio_25hz_to_44100(sine)
    assert resamp.shape == (2, 441_000), f"expected (2, 441000), got {resamp.shape}"
    assert resamp.dtype == np.float32
    print("[PASS] resample_audio_25hz_to_44100 (2, 512000) -> (2, 441000) @ 44.1 kHz")
  else:
    print("[SKIP] scipy not installed; verified exact length ratio 512000 * 441 / 512 == 441000")

  print("=== All MAGI-2 Stereo Audio VAE self-tests passed! ===")


if __name__ == "__main__":
  _run_self_tests()
