"""Configuration schemas & constants for Sand.ai MAGI-2 (Preview 114B & Flagship)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
    ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, "
    "room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, "
    "thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, "
    "looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, "
    "disorganized, messy, cheap sound"
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, "
    "undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, "
    "exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, "
    "stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky "
    "voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, "
    "coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural "
    "pauses, awkward silence, choppy, disconnected, multiple speakers, two voices, background talking, out of tune, "
    "off-key, autotune artifacts"
)

# Exact 48-channel Wan2.2-TI2V-5B latent normalization shared by Wan2_2_VAE and TurboVAED.
WAN22_LATENT_MEAN: tuple[float, ...] = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)

WAN22_LATENT_STD: tuple[float, ...] = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)

MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2
MODALITY_TIME = 3

# Resolution presets: (stage1_H, stage1_W, stage1_T_lat, audio_T, stage2_H, stage2_W, stage2_T_lat)
RESOLUTION_PRESETS: dict[str, dict[str, int | float | bool]] = {
    "272p": {
        "height": 256,
        "width": 448,
        "video_latent_t": 32,
        "audio_latent_t": 250,
        "fps": 12.5,
        "use_refiner": False,
    },
    "540p": {
        "height": 512,
        "width": 896,
        "video_latent_t": 32,
        "audio_latent_t": 250,
        "fps": 12.5,
        "use_refiner": False,
    },
    "1080p": {
        "height": 512,
        "width": 896,
        "video_latent_t": 32,
        "audio_latent_t": 250,
        "refiner_height": 1088,
        "refiner_width": 1920,
        "refiner_latent_t": 63,
        "fps": 25.0,
        "use_refiner": True,
    },
}


@dataclass(frozen=True)
class Magi2PreviewConfig:
  """Architecture configuration for Stage 1 MAGI-2 Preview (113.93B / 5.95B active)."""

  num_layers: int = 40
  hidden_size: int = 3072
  head_dim: int = 128
  num_heads_q: int = 24
  num_heads_kv: int = 24
  video_in_channels: int = 48
  audio_in_channels: int = 64
  text_in_channels: int = 5120
  dense_intermediate_size: int = 8192  # int(3072 * 4 * 2/3) = 8192
  mm_layers: tuple[int, ...] = (0, 1, 38, 39)
  moe_layers: tuple[int, ...] = tuple(range(2, 38))
  # Attention extras
  attn_gating: bool = True
  attn_sinks: bool = True
  sink_token_num: int = 1
  # 4-stream Manifold-Constrained Hyper-Connections (mHC)
  mhc_enable: bool = True
  mhc_num_stream: int = 4
  mhc_alpha_init: float = 0.01
  mhc_sinkhorn_iters: int = 20
  mhc_sinkhorn_eps: float = 1e-12
  # Multi-Head Latent MoE
  moe_num_heads: int = 12
  moe_num_experts: int = 256  # per head -> 12 * 256 = 3072 total experts/layer
  moe_top_k: int = 6
  moe_route_norm: bool = True
  moe_route_scale: float = 4.9
  moe_expert_intermediate_size: int = 1280
  moe_shared_intermediate_size: int = 1280
  moe_modality_shared_intermediate_size: int = 1280
  # Timestep & RoPE
  time_channel_dim: int = 64
  rope_temperature: float = 10000.0
  rms_norm_eps: float = 1e-6
  # Sampling defaults
  num_inference_steps: int = 100
  shift: float = 7.0
  video_cfg: float = 5.0
  audio_cfg: float = 7.0

  @property
  def stream_hidden_size(self) -> int:
    return self.hidden_size * (self.mhc_num_stream if self.mhc_enable else 1)

  @property
  def moe_head_dim(self) -> int:
    assert self.hidden_size % self.moe_num_heads == 0
    return self.hidden_size // self.moe_num_heads

  @property
  def moe_total_experts(self) -> int:
    return self.moe_num_heads * self.moe_num_experts

  def is_moe_layer(self, layer_idx: int) -> bool:
    return int(layer_idx) in self.moe_layers

  def num_modality_for_layer(self, layer_idx: int) -> int:
    return 3 if int(layer_idx) in self.mm_layers else 1

  @classmethod
  def from_checkpoint(cls, checkpoint_dir: str | Path) -> "Magi2PreviewConfig":
    ckpt = Path(checkpoint_dir)
    for cand in (ckpt / "config.json", ckpt.parent / "config.json"):
      if cand.exists():
        return cls.from_json(cand)
    return cls()

  @classmethod
  def from_json(cls, path: str | Path) -> "Magi2PreviewConfig":
    data = json.loads(Path(path).read_text())
    arch = data.get("arch_config", data)
    ev = data.get("evaluation_config", {})
    moe = arch.get("moe_config", {})
    mhc = arch.get("mhc_config", {})
    d = int(arch.get("hidden_size", 3072))
    hd = int(arch.get("head_dim", 128))
    nq = d // hd
    nkv = int(arch.get("num_query_groups", nq))
    factor = int(arch.get("intermediate_factor", 4))
    dense_inter = int(d * factor * 2 / 3) // 4 * 4
    return cls(
        num_layers=int(arch.get("num_layers", 40)),
        hidden_size=d,
        head_dim=hd,
        num_heads_q=nq,
        num_heads_kv=nkv,
        video_in_channels=int(arch.get("video_in_channels", 48)),
        audio_in_channels=int(arch.get("audio_in_channels", 64)),
        text_in_channels=int(arch.get("text_in_channels", 5120)),
        dense_intermediate_size=dense_inter,
        mm_layers=tuple(arch.get("mm_layers", (0, 1, 38, 39))),
        moe_layers=tuple(moe.get("moe_layers", tuple(range(2, 38)))),
        attn_gating=bool(arch.get("attn_gating", {}).get("enable", True)),
        attn_sinks=bool(arch.get("attn_sinks", {}).get("enable", True)),
        sink_token_num=int(arch.get("attn_sinks", {}).get("sink_token_num", 1)),
        mhc_enable=bool(mhc.get("enable", True)),
        mhc_num_stream=int(mhc.get("num_stream", 4)),
        mhc_alpha_init=float(mhc.get("alpha_init", 0.01)),
        moe_num_heads=int(moe.get("num_heads", 12)),
        moe_num_experts=int(moe.get("num_experts", 256)),
        moe_top_k=int(moe.get("top_k", 6)),
        moe_route_norm=bool(moe.get("route_norm", True)),
        moe_route_scale=float(moe.get("route_scale", 4.9)),
        moe_expert_intermediate_size=int(moe.get("expert_intermediate_size", 1280)),
        moe_shared_intermediate_size=int(moe.get("shared_expert_intermediate_size", 1280)),
        moe_modality_shared_intermediate_size=int(
            moe.get("modality_specific_expert_intermediate_size", 1280)
        ),
        time_channel_dim=int(ev.get("data_proxy_config", {}).get("time_channel_dim", 64)),
        num_inference_steps=int(ev.get("num_inference_steps", 100)),
        shift=float(ev.get("shift", 7.0)),
        video_cfg=float(ev.get("video_txt_guidance_scale", 5.0)),
        audio_cfg=float(ev.get("audio_txt_guidance_scale", 7.0)),
    )


@dataclass(frozen=True)
class Magi2RefinerConfig:
  """Architecture configuration for Stage 2 MAGI-2 Refiner (6.72B dense GQA)."""

  num_layers: int = 30
  hidden_size: int = 4096
  head_dim: int = 128
  num_heads_q: int = 32
  num_heads_kv: int = 8
  video_in_channels: int = 48
  audio_in_channels: int = 64
  text_in_channels: int = 5120
  intermediate_size: int = 10920  # int(4096 * 4 * 2/3) // 4 * 4 = 10920
  mm_layers: tuple[int, ...] = (0, 1, 28, 29)
  local_attn_layers: tuple[int, ...] = tuple(range(30))
  enable_attn_gating: bool = True
  rope_temperature: float = 10000.0
  rms_norm_eps: float = 1e-6
  # 3D Block-Grid Window Attention
  block_t_size: int = 8
  block_h_size: int = 4
  block_w_size: int = 4
  block_t_radius: int = 2
  block_h_radius: int = 2
  block_w_radius: int = 2
  # Sampling & noise augmentation
  num_inference_steps: int = 5
  shift: float = 5.0
  video_cfg: float = 2.0
  noise_value: int = 220

  @classmethod
  def from_json(cls, path: str | Path) -> "Magi2RefinerConfig":
    data = json.loads(Path(path).read_text())
    arch = data.get("magi2_refiner_arch_config", data)
    ev = data.get("evaluation_config", {})
    dp = ev.get("magi2_refiner_data_proxy_config", {})
    ac = dp.get("attn_config", {})
    win = ac.get("window", {})
    d = int(arch.get("hidden_size", 4096))
    hd = int(arch.get("head_dim", 128))
    inter = int(d * 4 * 2 / 3) // 4 * 4
    return cls(
        num_layers=int(arch.get("num_layers", 30)),
        hidden_size=d,
        head_dim=hd,
        num_heads_q=d // hd,
        num_heads_kv=int(arch.get("num_query_groups", 8)),
        video_in_channels=int(arch.get("video_in_channels", 48)),
        audio_in_channels=int(arch.get("audio_in_channels", 64)),
        text_in_channels=int(arch.get("text_in_channels", 5120)),
        intermediate_size=inter,
        mm_layers=tuple(arch.get("mm_layers", (0, 1, 28, 29))),
        local_attn_layers=tuple(arch.get("local_attn_layers", tuple(range(30)))),
        enable_attn_gating=bool(arch.get("enable_attn_gating", True)),
        block_t_size=int(ac.get("block_t_size", 8)),
        block_h_size=int(ac.get("block_size", 4)),
        block_w_size=int(ac.get("block_size", 4)),
        block_t_radius=int(win.get("block_t_radius", 2)),
        block_h_radius=int(win.get("block_h_radius", 2)),
        block_w_radius=int(win.get("block_w_radius", 2)),
        num_inference_steps=int(ev.get("magi2_refiner_num_inference_steps", 5)),
        shift=float(ev.get("magi2_refiner_shift", 5.0)),
        video_cfg=float(ev.get("magi2_refiner_video_txt_guidance_scale", 2.0)),
        noise_value=int(ev.get("magi2_refiner_noise_value", 220)),
    )


@dataclass(frozen=True)
class Magi2TurboVAEDConfig:
  """Distilled TurboVAED decoder config (`TurboV3-Wan22-TinyShallow_7_7.json`)."""

  latent_channels: int = 48
  out_channels: int = 3
  block_out_channels: tuple[int, ...] = (64, 128, 256, 512)
  layers_per_block: tuple[int, ...] = (2, 2, 2, 3, 3)
  spatio_temporal_scaling: tuple[bool, ...] = (False, True, True, True)
  spatio_only: tuple[bool, ...] = (False, True, False, False)
  patch_size: int = 2
  use_unpatchify: bool = True
  first_chunk_size: int = 7
  step_size: int = 7
  rms_norm_eps: float = 1e-8


@dataclass(frozen=True)
class Magi2AudioVAEConfig:
  """Stable Audio Open 1.0 Oobleck stereo audio decoder config."""

  sample_rate: int = 44100
  io_channels: int = 2
  latent_dim: int = 64
  channels: int = 128
  c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
  strides: tuple[int, ...] = (2, 4, 4, 8, 8)
  final_tanh: bool = False
  dit_latent_fps: float = 25.0

  @property
  def downsampling_ratio(self) -> int:
    r = 1
    for s in self.strides:
      r *= s
    return r


@dataclass(frozen=True)
class Magi2Qwen35Config:
  """Qwen3.5-27B text-only backbone config (skip_layer=2 -> layer 61 output)."""

  num_hidden_layers: int = 64
  skip_layer: int = 2  # extract hidden_states[-3], i.e. after 0-indexed layer 61
  hidden_size: int = 5120
  intermediate_size: int = 17408
  vocab_size: int = 248320
  rms_norm_eps: float = 1e-6
  # Full attention (every 4th layer: 3, 7, ..., 63)
  full_attn_interval: int = 4
  num_attention_heads: int = 24
  num_key_value_heads: int = 4
  head_dim: int = 256
  attn_output_gate: bool = True
  partial_rotary_factor: float = 0.25
  rope_theta: float = 10_000_000.0
  # Linear attention (Gated DeltaNet)
  linear_conv_kernel_dim: int = 4
  linear_num_key_heads: int = 16
  linear_key_head_dim: int = 128
  linear_num_value_heads: int = 48
  linear_value_head_dim: int = 128
  max_length: int = 7000
  layer_types: tuple[str, ...] = field(default_factory=tuple)

  def is_full_attn(self, layer_idx: int) -> bool:
    if self.layer_types:
      return self.layer_types[layer_idx] == "full_attention"
    return (layer_idx + 1) % self.full_attn_interval == 0
