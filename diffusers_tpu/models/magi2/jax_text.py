"""Qwen3.5-27B hybrid text encoder in native JAX (Flax NNX) for MAGI-2-preview.

Matches ``transformers.Qwen3_5TextModel`` and Sand.ai's upstream
``MAGI-2-preview/inference/model/qwen35.py`` bit-for-bit, with TPU v7x-8
``TP=8`` sharding (~6 GiB/chip for layers ``0..61``).

Architecture (``Qwen/Qwen3.5-27B``, 64 decoder layers, 3:1 hybrid schedule):

* **Early exit (``skip_layer=2``)**: MAGI-2 conditions on
  ``outputs.hidden_states[-3]`` (= output after 0-indexed layer 61, before the
  final RMSNorm). Layers 62, 63, final ``norm``, ``model.visual.*``,
  ``lm_head.*``, and ``mtp.*`` are neither loaded nor executed when
  ``skip_layer=2``.
* **Zero-centered RMSNorm** (``Qwen3_5RMSNorm``): ``input_layernorm``,
  ``post_attention_layernorm``, ``self_attn.q_norm``, ``self_attn.k_norm``, and
  final ``norm`` all compute ``rsqrt(mean(x^2) + eps) * (1.0 + weight)`` in
  float32 (weights initialized to zeros).
* **Gated DeltaNet** (``linear_attention``, 48 layers total, 47 in ``0..61``):
  16 K-heads x 128, 48 V-heads x 128 (1:3 K->V repeat_interleave), causal
  depthwise Conv1d (kernel=4, SiLU), L2-normalized Q/K (eps=1e-6),
  ``g = -exp(A_log) * softplus(a + dt_bias)``, ``beta = sigmoid(b)``,
  chunkwise-parallel gated delta rule in float32 (mathematically identical to
  the token-by-token recurrence), standard (1-centered, NOT ``1+w``)
  ``Qwen3_5RMSNormGated`` with ``silu(z)`` gating, and ``out_proj``.
  At load time ``in_proj_qkv`` and ``conv1d`` are split into Q/K/V slices so
  all 16 K-heads (2/chip) and 48 V-heads (6/chip) shard cleanly across TP=8
  with a single all-reduce at ``out_proj``.
* **Full Attention** (``full_attention``, layers 3, 7, ..., 59, 63):
  GQA 6:1 (24 Q-heads / 4 KV-heads, ``head_dim=256``), per-head interleaved
  ``attn_output_gate`` in ``q_proj`` (``24 * 256 * 2 = 12288`` ->
  ``[q, gate]`` per head, output multiplied by ``sigmoid(gate)``), per-head
  zero-centered QK-norm before RoPE, partial split-half RoPE on the leading
  ``64/256`` dims (``partial_rotary_factor=0.25``, ``rope_theta=1e7``), and
  causal attention. Under TP=8 where ``4 % 8 != 0``, ``k_proj`` / ``v_proj``
  are replicated (only ~21 MB/layer) and KV heads are locally duplicated 2x
  (4 -> 8 -> 1 KV head per chip for its 3 Q heads) with zero extra collectives.
* **Prompt preprocessing & tokenizer**: exact CJK ``Split`` pre-tokenizer
  patch, ``json_to_compact_markdown`` for JSON prompts containing
  ``"global_layer"``, NO chat template / NO system prompt, ``max_length=7000``,
  ``padding="longest"``, ``padding_side="right"``, ``truncation=True``, and
  I2V ``<Figure N>`` ``offset_mapping`` mean-pooling helper.

Self-test:

    JAX_PLATFORMS=cpu python -m diffusers_tpu.models.magi2.jax_text
"""

from __future__ import annotations

from collections.abc import Iterable
import concurrent.futures as cf
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
  from .config import Magi2Qwen35Config
else:
  import sys

  _THIS_DIR = str(Path(__file__).resolve().parent)
  if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
  from config import Magi2Qwen35Config


DEFAULT_SKIP_LAYER = 2
DEFAULT_MAX_LENGTH = 7000
DEFAULT_GDN_CHUNK_SIZE = int(os.environ.get("DT_GDN_CHUNK_SIZE", "64"))
DEFAULT_GDN_MODE = os.environ.get("DT_GDN_MODE", "chunk")

# Exact CJK character isolation regex from MAGI-2-preview/inference/model/qwen35.py
CJK_SPLIT_PATTERN = (
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
    r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)

# Attention kernel controls (matching qwen3_vl/jax_text.py conventions).
TEXT_ATTN_CHUNK = int(os.environ.get("DT_TEXT_ATTN_CHUNK", "0"))
TEXT_SPLASH_MIN = int(os.environ.get("DT_TEXT_SPLASH_MIN", "2048"))
TEXT_SPLASH_BLK = int(os.environ.get("DT_TEXT_SPLASH_BLK", "512"))

# Preload Pallas splash attention at import time when JAX TPU is available so
# importing inside the first jit trace never alters the trace context.
try:
  from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
      splash_attention_kernel as _sk_preload,
  )
except Exception:
  _sk_preload = None


# ---------------------------------------------------------------------------
# Prompt preprocessing & tokenizer (exact match to MAGI-2 qwen35.py)
# ---------------------------------------------------------------------------


def strip_empty(obj: Any) -> Any:
  if isinstance(obj, dict):
    return {
        key: value
        for key, value in ((key, strip_empty(value)) for key, value in obj.items())
        if value is not None and value != [] and value != {}
    }
  if isinstance(obj, list):
    return [strip_empty(value) for value in obj if value is not None]
  return obj


def _one_line(value: Any) -> str:
  if value is None:
    return ""
  if not isinstance(value, str):
    value = str(value)
  return " ".join(value.split())


def _as_dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
  if value is None:
    return []
  return value if isinstance(value, list) else [value]


def _pick(values: dict[str, Any], keys: Iterable[str]) -> list[tuple[str, Any]]:
  return [
      (key, values[key])
      for key in keys
      if key in values and values[key] not in (None, [], {})
  ]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
  """Convert a structured MAGI-2 JSON prompt into compact markdown.

  Verbatim port of ``MAGI-2-preview/inference/model/qwen35.py``. Returns the
  original input unchanged if the parsed object is not a dict containing
  ``"global_layer"``.
  """
  if isinstance(raw_json, str):
    obj = json.loads(raw_json.strip())
  else:
    obj = raw_json

  obj = strip_empty(obj)
  if not isinstance(obj, dict) or "global_layer" not in obj:
    return (
        raw_json
        if isinstance(raw_json, str)
        else json.dumps(raw_json, ensure_ascii=False)
    )

  global_layer = _as_dict(obj.get("global_layer"))
  dynamic_layer = _as_dict(obj.get("dynamic_layer"))
  reference_layer = obj.get("reference_layer", [])

  lines: list[str] = []
  context = _one_line(global_layer.get("context"))
  description = _one_line(global_layer.get("description"))
  if context or description:
    lines.append(f"context: {context}" if context else "context")
    if description:
      lines.append(description)

  aesthetics = _as_dict(global_layer.get("aesthetics"))
  aesthetic_parts = []
  for label, key in (
      ("style", "style"),
      ("mood", "mood_atmosphere"),
      ("color", "color_scheme"),
  ):
    value = _one_line(aesthetics.get(key))
    if value:
      aesthetic_parts.append(f"{label}={value}")
  if aesthetic_parts:
    lines.append("aesthetics: " + "; ".join(aesthetic_parts))

  audio = _as_dict(global_layer.get("audio_baseline"))
  dialogue = _as_dict(audio.get("dialogue"))
  audio_parts = []
  language = _one_line(dialogue.get("language"))
  speakers = _as_list(dialogue.get("speaker_tags"))
  ambience = _one_line(audio.get("ambience"))
  if language:
    audio_parts.append(f"language={language}")
  if speakers:
    audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
  if ambience:
    audio_parts.append(f"ambience={ambience}")
  if audio_parts:
    lines.append("audio: " + "; ".join(audio_parts))

  subjects = global_layer.get("alive_subjects_static")
  if isinstance(subjects, list) and subjects:
    lines.append("subjects:")
    for subject in subjects:
      if not isinstance(subject, dict):
        continue
      sid = _one_line(subject.get("subject_id"))
      desc = _one_line(subject.get("description"))
      position = _one_line(subject.get("position"))
      orientation = _one_line(subject.get("orientation"))
      attrs = []
      for key, value in _pick(
          _as_dict(subject.get("visual_attributes")),
          ["gender", "age_group", "ethnicity", "clothing", "appearance_details"],
      ):
        value = _one_line(value)
        if value:
          attrs.append(f"{key}={value}")
      row = " - " + " | ".join([part for part in [sid, desc] if part])
      extra = "; ".join([part for part in [position, orientation] if part])
      if extra:
        row += f" ({extra})"
      if attrs:
        row += " :: " + "; ".join(attrs)
      lines.append(row)

  objects = global_layer.get("objects_static")
  if isinstance(objects, list) and objects:
    lines.append("objects:")
    for obj_item in objects:
      if not isinstance(obj_item, dict):
        continue
      oid = _one_line(obj_item.get("object_id"))
      desc = _one_line(obj_item.get("description"))
      shape = _one_line(obj_item.get("shape_and_color"))
      position = _one_line(obj_item.get("position"))
      row = " - " + " | ".join([part for part in [oid, desc] if part])
      details = "; ".join([part for part in [shape, position] if part])
      if details:
        row += " :: " + details
      lines.append(row)

  segments = dynamic_layer.get("timeline_segments")
  if isinstance(segments, list) and segments:
    lines.append("timeline:")
    for segment in segments:
      if not isinstance(segment, dict):
        continue
      basic = _as_dict(segment.get("segment_basic_info"))
      timestamp = _one_line(basic.get("timestamp_range"))
      desc = _one_line(basic.get("segment_description"))
      head = f" - {timestamp}" if timestamp else " -"
      if desc:
        head += f" {desc}"
      lines.append(head.rstrip())

      segment_audio = _as_dict(segment.get("audio"))
      for dialogue_line in segment_audio.get("dialogue_lines", []) or []:
        if not isinstance(dialogue_line, dict):
          continue
        speaker = _one_line(dialogue_line.get("speaker"))
        text = _one_line(dialogue_line.get("text"))
        timestamp = _one_line(dialogue_line.get("timestamp"))
        if text:
          prefix = (
              f"   - dialogue {speaker}: " if speaker else "   - dialogue: "
          )
          line = prefix + text
          if timestamp:
            line += f" ({timestamp})"
          lines.append(line)

      for alive in segment.get("alive_subjects", []) or []:
        if not isinstance(alive, dict):
          continue
        sid = _one_line(alive.get("subject_id"))
        action = _as_dict(alive.get("action"))
        parts = [
            _one_line(action.get("primary_action")),
            _one_line(action.get("interaction")),
            _one_line(action.get("facial_expression")),
        ]
        parts = [part for part in parts if part]
        if parts:
          lines.append(f"   - action {sid}: " + "; ".join(parts))

      for obj_item in segment.get("objects", []) or []:
        if not isinstance(obj_item, dict):
          continue
        oid = _one_line(obj_item.get("object_id"))
        state = _as_dict(obj_item.get("dynamic_state"))
        parts = [
            _one_line(state.get("state_change")),
            _one_line(state.get("motion_detail")),
        ]
        parts = [part for part in parts if part]
        if parts:
          lines.append(f"   - objects {oid}: " + "; ".join(parts))

  if isinstance(reference_layer, list):
    lines.extend(str(line) for line in reference_layer if str(line).strip())

  return "\n".join(line for line in lines if line.strip())


def normalize_prompt(prompt: str) -> str:
  """Convert JSON storyboard prompts to compact markdown; pass plain text through."""
  try:
    parsed_json = json.loads(prompt)
    if isinstance(parsed_json, (dict, list)):
      return json_to_compact_markdown(prompt)
  except (json.JSONDecodeError, TypeError):
    pass
  return prompt


def load_tokenizer(model_path: str | Path):
  """Load Qwen3.5 tokenizer with MAGI-2's exact CJK Split pre-tokenizer patch.

  Uses ``padding_side="right"``, NO chat template, and NO system prompt.
  """
  from tokenizers import Regex, pre_tokenizers
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="right")
  cjk_split = pre_tokenizers.Split(
      pattern=Regex(CJK_SPLIT_PATTERN),
      behavior="isolated",
  )
  original_pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
  tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
      [cjk_split, original_pre_tokenizer]
  )
  return tokenizer


def tokenize_prompt(
    tokenizer,
    prompt: str | list[str],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_offsets_mapping: bool = False,
) -> dict[str, np.ndarray]:
  """Tokenize prompt(s) exactly as ``Qwen35TextEncoder`` does (no chat template)."""
  prompts = [prompt] if isinstance(prompt, str) else list(prompt)
  normalized = [normalize_prompt(p) for p in prompts]
  encoded = tokenizer(
      normalized,
      return_tensors="np",
      padding="longest",
      max_length=max_length,
      truncation=True,
      return_offsets_mapping=return_offsets_mapping,
  )
  return {k: np.asarray(v) for k, v in encoded.items()}


def get_target_token_indices(
    tokenizer,
    prompt: str,
    target_str: str | None,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[int] | None:
  """Return token indices whose character spans overlap ``target_str`` (e.g. ``<Figure 1>``)."""
  if not target_str:
    return None
  prompt = normalize_prompt(prompt)
  inputs = tokenizer(
      [prompt],
      return_tensors="np",
      padding="longest",
      return_offsets_mapping=True,
      max_length=max_length,
      truncation=True,
  )
  offsets = inputs["offset_mapping"][0]
  start_char_idx = prompt.find(target_str)
  if start_char_idx == -1:
    return None
  end_char_idx = start_char_idx + len(target_str)

  indices: list[int] = []
  for index, (start, end) in enumerate(offsets):
    start_i, end_i = int(start), int(end)
    if start_i == 0 and end_i == 0 and index != 0:
      continue
    if max(start_i, start_char_idx) < min(end_i, end_char_idx):
      indices.append(index)
  return indices


def get_special_token(
    tokenizer,
    prompt: str,
    target_strs: list[str],
    text_feature,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
):
  """Mean-pool ``text_feature`` ``(1, seq, 5120)`` over each target span (e.g. ``["<Figure 1>"]``).

  Returns an array of shape ``(len(target_strs), hidden_size)`` (i.e.
  ``(1, 5120)`` for a single I2V reference image) in the same backend/dtype as
  ``text_feature``.
  """
  is_numpy = isinstance(text_feature, np.ndarray)
  if not is_numpy:
    import jax.numpy as jnp

  hidden_size = int(text_feature.shape[-1])
  embeddings = []
  for target_str in target_strs:
    indices = get_target_token_indices(
        tokenizer, prompt, target_str, max_length=max_length
    )
    if indices:
      if is_numpy:
        pooled = np.mean(
            text_feature[0, indices, :].astype(np.float32), axis=0
        ).astype(text_feature.dtype)
      else:
        idx = jnp.asarray(indices, dtype=jnp.int32)
        pooled = jnp.mean(
            text_feature[0, idx, :].astype(jnp.float32), axis=0
        ).astype(text_feature.dtype)
      embeddings.append(pooled)
    else:
      if is_numpy:
        embeddings.append(np.zeros((hidden_size,), dtype=text_feature.dtype))
      else:
        embeddings.append(jnp.zeros((hidden_size,), dtype=text_feature.dtype))

  if is_numpy:
    return np.stack(embeddings, axis=0)
  return jnp.stack(embeddings, axis=0)


def pool_figure_tokens(
    tokenizer,
    prompt: str,
    figure_ids: int | Iterable[int] | np.ndarray,
    text_feature,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
):
  """Pool ``<Figure N>`` token spans for I2V conditioning -> ``(num_figures, 5120)``."""
  if isinstance(figure_ids, (int, np.integer)):
    ids = [int(figure_ids)]
  else:
    ids = [int(i) for i in np.asarray(figure_ids).reshape(-1).tolist()]
  target_strs = [f"<Figure {i}>" for i in ids]
  return get_special_token(
      tokenizer, prompt, target_strs, text_feature, max_length=max_length
  )


# ---------------------------------------------------------------------------
# Config & host partial RoPE
# ---------------------------------------------------------------------------


def config_from(
    text_encoder_dir: str | Path,
    *,
    skip_layer: int = DEFAULT_SKIP_LAYER,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Magi2Qwen35Config:
  """Parse a HuggingFace Qwen3.5 ``config.json`` (multimodal or text-only)."""
  cfg_path = os.path.join(str(text_encoder_dir), "config.json")
  with open(cfg_path, "r", encoding="utf-8") as f:
    raw = json.load(f)
  cfg = raw.get("text_config", raw)
  rope_params = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}

  num_layers = int(cfg.get("num_hidden_layers", 64))
  full_attn_interval = int(cfg.get("full_attention_interval", 4))
  raw_layer_types = cfg.get("layer_types")
  if raw_layer_types:
    layer_types = tuple(str(t) for t in raw_layer_types)
  else:
    layer_types = tuple(
        "full_attention" if (i + 1) % full_attn_interval == 0 else "linear_attention"
        for i in range(num_layers)
    )

  return Magi2Qwen35Config(
      num_hidden_layers=num_layers,
      skip_layer=skip_layer,
      hidden_size=int(cfg.get("hidden_size", 5120)),
      intermediate_size=int(cfg.get("intermediate_size", 17408)),
      vocab_size=int(cfg.get("vocab_size", 248320)),
      rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
      full_attn_interval=full_attn_interval,
      num_attention_heads=int(cfg.get("num_attention_heads", 24)),
      num_key_value_heads=int(cfg.get("num_key_value_heads", 4)),
      head_dim=int(cfg.get("head_dim", 256)),
      attn_output_gate=bool(cfg.get("attn_output_gate", True)),
      partial_rotary_factor=float(
          rope_params.get(
              "partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25)
          )
      ),
      rope_theta=float(
          rope_params.get("rope_theta", cfg.get("rope_theta", 10_000_000.0))
      ),
      linear_conv_kernel_dim=int(cfg.get("linear_conv_kernel_dim", 4)),
      linear_num_key_heads=int(cfg.get("linear_num_key_heads", 16)),
      linear_key_head_dim=int(cfg.get("linear_key_head_dim", 128)),
      linear_num_value_heads=int(cfg.get("linear_num_value_heads", 48)),
      linear_value_head_dim=int(cfg.get("linear_value_head_dim", 128)),
      max_length=max_length,
      layer_types=layer_types,
  )


def partial_rope_cos_sin(
    position_ids: np.ndarray,
    head_dim: int = 256,
    partial_rotary_factor: float = 0.25,
    rope_theta: float = 10_000_000.0,
    mrope_section: tuple[int, int, int] = (11, 11, 10),
) -> tuple[np.ndarray, np.ndarray]:
  """Build Qwen3.5 partial split-half (m)RoPE ``(cos, sin)`` tables on the host.

  Matches ``transformers.Qwen3_5TextRotaryEmbedding``:
  - ``rotary_dim = int(head_dim * partial_rotary_factor)`` (64 of 256 for 27B).
  - ``inv_freq = 1 / (rope_theta ** (arange(0, rotary_dim, 2) / rotary_dim))``
    computed in float32.
  - Duplicated across halves (split-half ``rotate_half``), NOT pairwise
    interleaved: ``cos = [cos_half, cos_half]``, shape ``(..., rotary_dim)``.
  - Accepts 1D ``(seq,)``, 2D ``(batch, seq)``, or 3D ``(3, seq)`` /
    ``(3, batch, seq)`` position IDs. For text-only inputs all 3 mRoPE axes
    agree, collapsing to standard 1D partial RoPE.
  """
  pos = np.asarray(position_ids, dtype=np.float32)
  rotary_dim = int(head_dim * partial_rotary_factor)
  if rotary_dim % 2 != 0:
    raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
  inv_freq = 1.0 / (
      np.float32(rope_theta)
      ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim))
  )

  if pos.ndim in (1, 2):
    angles = pos[..., None] * inv_freq
    cos_h = np.cos(angles).astype(np.float32)
    sin_h = np.sin(angles).astype(np.float32)
  elif pos.ndim in (3,) and pos.shape[0] == 3:
    # Interleaved mRoPE axis recomposition across (T, H, W).
    angles = pos[..., None] * inv_freq  # (3, ..., rotary_dim // 2)
    cos_all = np.cos(angles).astype(np.float32)
    sin_all = np.sin(angles).astype(np.float32)
    cos_h = cos_all[0].copy()
    sin_h = sin_all[0].copy()
    for axis, offset in ((1, 1), (2, 2)):
      idx = slice(offset, mrope_section[axis] * 3, 3)
      cos_h[..., idx] = cos_all[axis, ..., idx]
      sin_h[..., idx] = sin_all[axis, ..., idx]
  else:
    raise ValueError(
        f"Expected position_ids shape (seq,), (batch, seq), or (3, ...); got {pos.shape}"
    )

  return np.concatenate([cos_h, cos_h], axis=-1), np.concatenate([sin_h, sin_h], axis=-1)


# ---------------------------------------------------------------------------
# Splash attention & Gated Delta Rule kernels
# ---------------------------------------------------------------------------

_SPLASH_KERNELS: dict[tuple[int, int, int], Any] = {}


def _causal_kernel(n: int, g: int, blk: int):
  """Cache one Pallas splash causal MQA kernel per ``(padded_seq, gqa_ratio, blk)``."""
  key = (n, g, blk)
  if key not in _SPLASH_KERNELS:
    import jax
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk,
        splash_attention_mask as sm,
    )

    with jax.ensure_compile_time_eval():
      _SPLASH_KERNELS[key] = sk.make_splash_mqa_single_device(
          sm.MultiHeadMask([sm.CausalMask((n, n))] * g),
          block_sizes=sk.BlockSizes(
              block_q=blk, block_kv=blk, block_kv_compute=blk
          ),
      )
  return _SPLASH_KERNELS[key]


def _splash_causal(q, k, v, scale: float):
  """Device-local causal GQA attention via Pallas splash.

  Args:
    q: ``(s, hq, d)``
    k: ``(s, hkv, d)``
    v: ``(s, hkv, d)``
  """
  import jax
  import jax.numpy as jnp

  s, hq, d = q.shape
  hkv = k.shape[1]
  g = hq // hkv
  blk = TEXT_SPLASH_BLK
  sp = (-s) % blk
  n = s + sp
  if sp:
    q, k, v = (jnp.pad(t, ((0, sp), (0, 0), (0, 0))) for t in (q, k, v))
  kernel = _causal_kernel(n, g, blk)
  qh = (q * scale).reshape(n, hkv, g, d).transpose(1, 2, 0, 3)  # (hkv, g, n, d)
  kh, vh = k.transpose(1, 0, 2), v.transpose(1, 0, 2)  # (hkv, n, d)
  out = jax.vmap(kernel)(qh, kh, vh)  # (hkv, g, n, d)
  return out.transpose(2, 0, 1, 3).reshape(n, hq, d)[:s]


def l2norm(x, eps: float = 1e-6):
  """FLA / HF ``modeling_qwen3_5.l2norm`` along the last axis."""
  import jax
  import jax.numpy as jnp

  return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    *,
    use_qk_l2norm: bool = True,
):
  """Exact token-by-token Gated Delta Rule via ``jax.lax.scan`` in float32.

  Matches ``transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule``.

  Args:
    query: ``(B, S, H, K)``
    key: ``(B, S, H, K)``
    value: ``(B, S, H, V)``
    g: ``(B, S, H)`` log-space decay (``<= 0``)
    beta: ``(B, S, H)`` update gate in ``(0, 1)``
    initial_state: optional ``(B, H, K, V)`` float32 recurrent state
    use_qk_l2norm: apply FLA L2-norm (``eps=1e-6``) to Q and K in float32

  Returns:
    ``(core_attn_out, final_state)`` where ``core_attn_out`` has shape
    ``(B, S, H, V)`` in ``query.dtype`` and ``final_state`` is ``(B, H, K, V)``
    in float32.
  """
  import jax
  import jax.numpy as jnp

  initial_dtype = query.dtype
  b, _, h, k_dim = query.shape
  v_dim = value.shape[-1]

  q = query.astype(jnp.float32)
  k = key.astype(jnp.float32)
  v = value.astype(jnp.float32)
  g32 = g.astype(jnp.float32)
  beta32 = beta.astype(jnp.float32)

  if use_qk_l2norm:
    q = l2norm(q, eps=1e-6)
    k = l2norm(k, eps=1e-6)
  q = q * (k_dim**-0.5)

  state0 = (
      jnp.zeros((b, h, k_dim, v_dim), dtype=jnp.float32)
      if initial_state is None
      else initial_state.astype(jnp.float32)
  )

  # Move sequence dim S to leading scan axis: (S, B, H, ...).
  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(g32, 0, 1),
      jnp.swapaxes(beta32, 0, 1),
  )

  def step(state, x_t):
    q_t, k_t, v_t, g_t, beta_t = x_t
    state = state * jnp.exp(g_t)[..., None, None]
    kv_mem = jnp.sum(state * k_t[..., :, None], axis=-2)  # (B, H, V)
    delta = (v_t - kv_mem) * beta_t[..., None]  # (B, H, V)
    state = state + k_t[..., :, None] * delta[..., None, :]  # (B, H, K, V)
    out_t = jnp.sum(state * q_t[..., :, None], axis=-2)  # (B, H, V)
    return state, out_t

  with jax.default_matmul_precision("highest"):
    final_state, outs = jax.lax.scan(step, state0, xs)
  # outs: (S, B, H, V) -> (B, S, H, V)
  return jnp.swapaxes(outs, 0, 1).astype(initial_dtype), final_state


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = DEFAULT_GDN_CHUNK_SIZE,
    initial_state=None,
    *,
    use_qk_l2norm: bool = True,
):
  """Chunkwise-parallel Gated Delta Rule in float32 (TPU MXU friendly).

  Exact JAX port of ``transformers`` ``torch_chunk_gated_delta_rule``.
  Mathematically equivalent to :func:`recurrent_gated_delta_rule` while using
  batched MXU matmuls + triangular solve across all chunks in parallel followed
  by a short ``S // chunk_size`` ``lax.scan``.
  """
  import jax
  import jax.numpy as jnp

  initial_dtype = query.dtype
  b, seq_len, h, k_dim = query.shape
  v_dim = value.shape[-1]
  c = int(chunk_size)

  # (B, S, H, ...) -> (B, H, S, ...) in float32
  q = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  k = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  v = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  g32 = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)
  beta32 = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)

  if use_qk_l2norm:
    q = l2norm(q, eps=1e-6)
    k = l2norm(k, eps=1e-6)
  q = q * (k_dim**-0.5)

  pad = (-seq_len) % c
  if pad:
    q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
    g32 = jnp.pad(g32, ((0, 0), (0, 0), (0, pad)))
    beta32 = jnp.pad(beta32, ((0, 0), (0, 0), (0, pad)))

  total_len = seq_len + pad
  num_chunks = total_len // c

  v_beta = v * beta32[..., None]
  k_beta = k * beta32[..., None]

  # Reshape into chunks: (B, H, NC, C, ...)
  q = q.reshape(b, h, num_chunks, c, k_dim)
  k = k.reshape(b, h, num_chunks, c, k_dim)
  k_beta = k_beta.reshape(b, h, num_chunks, c, k_dim)
  v_beta = v_beta.reshape(b, h, num_chunks, c, v_dim)
  g_c = g32.reshape(b, h, num_chunks, c)

  causal_mask = jnp.tril(jnp.ones((c, c), dtype=jnp.bool_))
  strict_lower = jnp.tril(jnp.ones((c, c), dtype=jnp.float32), k=-1)

  with jax.default_matmul_precision("highest"):
    # Phase 1: intra-chunk UT transform (all chunks computed in parallel).
    cum_decay = jnp.cumsum(g_c, axis=-1)  # (B, H, NC, C)
    diff = cum_decay[..., :, None] - cum_decay[..., None, :]  # (B, H, NC, C, C)
    # g <= 0 so diff <= 0 on lower triangle; mask strictly upper before exp.
    pairwise_decay = jnp.where(
        causal_mask, jnp.exp(jnp.where(causal_mask, diff, -1e30)), 0.0
    )

    k_t = jnp.swapaxes(k, -1, -2)  # (B, H, NC, K, C)
    ut_system = jnp.matmul(k_beta, k_t) * pairwise_decay * strict_lower
    intra_chunk_attn = jnp.matmul(q, k_t) * pairwise_decay

    exp_cum = jnp.exp(cum_decay)[..., None]  # (B, H, NC, C, 1)
    decayed_k_beta = k_beta * exp_cum

    # Single batched lower-triangular unit-diagonal solve for [v_beta | decayed_k_beta].
    rhs = jnp.concatenate([v_beta, decayed_k_beta], axis=-1)  # (B, H, NC, C, V + K)
    sol = jax.scipy.linalg.solve_triangular(
        ut_system, rhs, lower=True, unit_diagonal=True
    )
    new_values = sol[..., :v_dim]  # (B, H, NC, C, V)
    k_cumdecay = sol[..., v_dim:]  # (B, H, NC, C, K)

    q_decayed = q * exp_cum
    k_decayed = (
        k * jnp.exp(jnp.minimum(cum_decay[..., -1:] - cum_decay, 0.0))[..., None]
    )
    chunk_decay = jnp.exp(cum_decay[..., -1])[..., None, None]  # (B, H, NC, 1, 1)

    # Phase 2: inter-chunk state recurrence scan over NC chunks.
    state0 = (
        jnp.zeros((b, h, k_dim, v_dim), dtype=jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )
    xs = tuple(
        jnp.moveaxis(t, 2, 0)
        for t in (
            new_values,
            k_cumdecay,
            intra_chunk_attn,
            q_decayed,
            k_decayed,
            chunk_decay,
        )
    )

    def chunk_step(state, x_i):
      u_i, w_i, aqk_i, q_i, k_i, d_i = x_i
      v_new = u_i - jnp.matmul(w_i, state)  # (B, H, C, V)
      out_i = jnp.matmul(q_i, state) + jnp.matmul(aqk_i, v_new)  # (B, H, C, V)
      state = state * d_i + jnp.matmul(jnp.swapaxes(k_i, -1, -2), v_new)
      return state, out_i

    final_state, out_chunks = jax.lax.scan(chunk_step, state0, xs)

  # (NC, B, H, C, V) -> (B, NC, C, H, V) -> (B, seq_len, H, V)
  out = jnp.transpose(out_chunks, (1, 0, 3, 2, 4)).reshape(b, total_len, h, v_dim)
  return out[:, :seq_len, :, :].astype(initial_dtype), final_state


# ---------------------------------------------------------------------------
# Sharding layout helpers (TP=8 TPU v7x-8)
# ---------------------------------------------------------------------------

_COLUMN_PARENTS = frozenset({
    "q_proj",
    "gate_proj",
    "up_proj",
    "in_proj_q",
    "in_proj_k",
    "in_proj_v",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "conv1d_q",
    "conv1d_k",
    "conv1d_v",
})
_KV_PARENTS = frozenset({"k_proj", "v_proj"})
_ROW_PARENTS = frozenset({"o_proj", "down_proj", "out_proj"})
_HEAD_1D_PARENTS = frozenset({"A_log", "dt_bias"})


def _resolve_axis_name(mesh, axis_name: str | None) -> str:
  if axis_name is not None:
    if mesh is not None and axis_name not in mesh.axis_names:
      raise ValueError(
          f"axis_name={axis_name!r} not in mesh.axis_names={mesh.axis_names}"
      )
    return axis_name
  if mesh is not None:
    for candidate in ("tp", "tensor"):
      if candidate in mesh.axis_names:
        return candidate
    return mesh.axis_names[0]
  return "tp"


def param_spec(
    keys: Iterable[str],
    ndim: int,
    *,
    axis_name: str = "tp",
    shard_kv: bool = False,
    shard_embed: bool = True,
):
  """Return the TP ``PartitionSpec`` for a Qwen3.5 parameter path.

  By default ``shard_kv=False`` so ``k_proj`` / ``v_proj`` (4 KV heads in
  Qwen3.5-27B) are safely replicated under ``TP=8`` instead of failing
  ``4 % 8 != 0``. Use :func:`make_param_spec` to shard KV heads automatically
  whenever ``num_key_value_heads % tp_size == 0``.
  """
  from jax.sharding import PartitionSpec as P

  key_list = [str(k) for k in keys]
  parent = next(
      (k for k in reversed(key_list) if k not in ("weight", "value", ".value")),
      "",
  )
  if ndim == 2 and parent == "embed_tokens":
    return P(None, axis_name) if shard_embed else P()
  if ndim == 2 and parent in _COLUMN_PARENTS:
    return P(axis_name, None)
  if ndim == 2 and parent in _KV_PARENTS:
    return P(axis_name, None) if shard_kv else P()
  if ndim == 2 and parent in _ROW_PARENTS:
    return P(None, axis_name)
  if ndim == 1 and parent in _HEAD_1D_PARENTS:
    return P(axis_name)
  return P()


def make_param_spec(
    cfg: Magi2Qwen35Config,
    mesh,
    *,
    axis_name: str | None = None,
):
  """Build a config- and mesh-aware ``spec_fn(keys, ndim)`` that never violates divisibility."""
  from jax.sharding import PartitionSpec as P

  axis = _resolve_axis_name(mesh, axis_name)
  tp = int(mesh.shape[axis]) if (mesh is not None and axis in mesh.axis_names) else 1
  if tp <= 1:
    return lambda keys, ndim: P()

  shard_embed = cfg.hidden_size % tp == 0
  shard_q = cfg.num_attention_heads % tp == 0
  shard_kv = cfg.num_key_value_heads % tp == 0
  shard_mlp = cfg.intermediate_size % tp == 0
  shard_gdn_k = cfg.linear_num_key_heads % tp == 0
  shard_gdn_v = cfg.linear_num_value_heads % tp == 0

  def _spec(keys: Iterable[str], ndim: int):
    key_list = [str(k) for k in keys]
    parent = next(
        (k for k in reversed(key_list) if k not in ("weight", "value", ".value")),
        "",
    )
    if ndim == 2 and parent == "embed_tokens":
      return P(None, axis) if shard_embed else P()
    if ndim == 2 and parent == "q_proj":
      return P(axis, None) if shard_q else P()
    if ndim == 2 and parent == "o_proj":
      return P(None, axis) if shard_q else P()
    if ndim == 2 and parent in _KV_PARENTS:
      return P(axis, None) if shard_kv else P()
    if ndim == 2 and parent in ("gate_proj", "up_proj"):
      return P(axis, None) if shard_mlp else P()
    if ndim == 2 and parent == "down_proj":
      return P(None, axis) if shard_mlp else P()
    if ndim == 2 and parent in ("in_proj_q", "in_proj_k", "conv1d_q", "conv1d_k"):
      return P(axis, None) if shard_gdn_k else P()
    if ndim == 2 and parent in (
        "in_proj_v",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d_v",
    ):
      return P(axis, None) if shard_gdn_v else P()
    if ndim == 2 and parent == "out_proj":
      return P(None, axis) if shard_gdn_v else P()
    if ndim == 1 and parent in _HEAD_1D_PARENTS:
      return P(axis) if shard_gdn_v else P()
    return P()

  return _spec


# ---------------------------------------------------------------------------
# Flax NNX modules (lazy import so the module can be inspected without flax)
# ---------------------------------------------------------------------------

_MODULES_CACHE = None


def _build_modules():
  """Construct Flax NNX module classes lazily."""
  global _MODULES_CACHE
  if _MODULES_CACHE is not None:
    return _MODULES_CACHE

  import jax
  import jax.numpy as jnp
  from flax import nnx

  class ZeroCenteredRMSNorm(nnx.Module):
    """``Qwen3_5RMSNorm``: ``rsqrt(mean(x^2) + eps) * (1.0 + weight)`` in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
      self.weight = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))
      self.eps = float(eps)

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      norm = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
      out = norm * (1.0 + self.weight[...].astype(jnp.float32))
      return out.astype(x.dtype)

  class RMSNormGated(nnx.Module):
    """``Qwen3_5RMSNormGated``: standard (1-centered) RMSNorm * ``silu(gate)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
      self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
      self.eps = float(eps)

    def __call__(self, x, gate):
      in_dtype = x.dtype
      x32 = x.astype(jnp.float32)
      norm = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
      # Match HF modeling_qwen3_5.py:L216-219:
      #   hidden_states = self.weight * hidden_states.to(input_dtype)
      #   hidden_states = hidden_states * silu(gate.to(float32))
      norm_cast = norm.astype(in_dtype).astype(jnp.float32)
      out = (self.weight[...].astype(jnp.float32) * norm_cast) * jax.nn.silu(
          gate.astype(jnp.float32)
      )
      return out.astype(in_dtype)

  class Linear(nnx.Module):
    """Bias-free linear layer with PyTorch ``(out_features, in_features)`` weight."""

    def __init__(self, in_f: int, out_f: int, *, dtype, rngs=None):
      del rngs
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype=dtype))

    def __call__(self, x):
      return x @ self.weight[...].T

  def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

  def apply_partial_rope(x, cos, sin):
    """Apply split-half partial RoPE to leading ``cos.shape[-1]`` dims of ``x``.

    ``x`` has shape ``(B, S, H, head_dim)``; ``cos``/``sin`` have shape
    ``(S, rotary_dim)`` or ``(B, S, rotary_dim)``.
    """
    rotary_dim = cos.shape[-1]
    if cos.ndim == 2:
      c = cos[None, :, None, :].astype(x.dtype)
      s = sin[None, :, None, :].astype(x.dtype)
    else:
      c = cos[:, :, None, :].astype(x.dtype)
      s = sin[:, :, None, :].astype(x.dtype)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_rot_emb = x_rot * c + rotate_half(x_rot) * s
    if rotary_dim == x.shape[-1]:
      return x_rot_emb
    return jnp.concatenate([x_rot_emb, x_pass], axis=-1)

  def causal_depthwise_conv1d_silu(x, weight):
    """Exact causal depthwise Conv1d (left-pad ``K-1``, right-pad ``0``) + SiLU.

    Implemented as ``K`` shifted elementwise multiply-adds so channel-sharded
    tensors ``P(None, None, axis)`` execute 100% locally on TPU VPU with zero
    cross-chip communication.

    Args:
      x: ``(B, S, C)``
      weight: ``(C, K)`` (HF ``(C, 1, K)`` squeezed on axis 1)
    """
    seq_len = x.shape[1]
    k_size = weight.shape[1]
    padded = jnp.pad(x.astype(jnp.float32), ((0, 0), (k_size - 1, 0), (0, 0)))
    w32 = weight.astype(jnp.float32)
    acc = padded[:, 0:seq_len, :] * w32[:, 0]
    for j in range(1, k_size):
      acc = acc + padded[:, j : j + seq_len, :] * w32[:, j]
    return jax.nn.silu(acc).astype(x.dtype)

  class Qwen3_5Attention(nnx.Module):
    """Full causal GQA attention with per-head output gate, QK-norm, and partial RoPE."""

    def __init__(self, cfg: Magi2Qwen35Config, *, dtype, rngs=None):
      self.heads = cfg.num_attention_heads
      self.kv_heads = cfg.num_key_value_heads
      self.head_dim = cfg.head_dim
      self.attn_output_gate = cfg.attn_output_gate
      self.scale = cfg.head_dim**-0.5
      self._mesh = None
      self._axis_name = "tp"

      q_mult = 2 if cfg.attn_output_gate else 1
      self.q_proj = Linear(
          cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim * q_mult, dtype=dtype, rngs=rngs
      )
      self.k_proj = Linear(
          cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, dtype=dtype, rngs=rngs
      )
      self.v_proj = Linear(
          cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, dtype=dtype, rngs=rngs
      )
      self.o_proj = Linear(
          cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, dtype=dtype, rngs=rngs
      )
      self.q_norm = ZeroCenteredRMSNorm(cfg.head_dim, cfg.rms_norm_eps)
      self.k_norm = ZeroCenteredRMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, mask=None):
      from jax.sharding import NamedSharding, PartitionSpec as P

      b, s, _ = x.shape
      if self.attn_output_gate:
        # Interleaved per head: (B, S, num_heads, 2 * head_dim) -> [q | gate].
        qg = self.q_proj(x).reshape(b, s, self.heads, self.head_dim * 2)
        q = qg[..., : self.head_dim]
        gate = qg[..., self.head_dim :]
      else:
        q = self.q_proj(x).reshape(b, s, self.heads, self.head_dim)
        gate = None

      q = self.q_norm(q)
      k = self.k_norm(self.k_proj(x).reshape(b, s, self.kv_heads, self.head_dim))
      v = self.v_proj(x).reshape(b, s, self.kv_heads, self.head_dim)

      q = apply_partial_rope(q, cos, sin).astype(v.dtype)
      k = apply_partial_rope(k, cos, sin).astype(v.dtype)

      mesh = self._mesh
      axis = self._axis_name
      tp = int(mesh.shape[axis]) if (mesh is not None and axis in mesh.axis_names) else 1
      use_tp_shard = mesh is not None and tp > 1 and (self.heads % tp == 0)

      if use_tp_shard:
        head_sharding = NamedSharding(mesh, P(None, None, axis, None))
        q = jax.lax.with_sharding_constraint(q, head_sharding)
        if self.kv_heads % tp != 0:
          # e.g. 4 KV heads across TP=8 -> locally duplicate KV 2x to 8 heads
          # so each chip owns 1 KV head aligned with its 3 Q heads (zero ICI comm).
          kv_dup = (
              tp // self.kv_heads
              if (tp % self.kv_heads == 0)
              else (self.heads // self.kv_heads)
          )
          k = jnp.repeat(k, kv_dup, axis=2)
          v = jnp.repeat(v, kv_dup, axis=2)
        k = jax.lax.with_sharding_constraint(k, head_sharding)
        v = jax.lax.with_sharding_constraint(v, head_sharding)

      if (
          mask is None
          and b == 1
          and s >= TEXT_SPLASH_MIN
          and jax.default_backend() == "tpu"
      ):
        if use_tp_shard:
          spec = P(None, axis, None)
          local_splash = lambda a, c, e: _splash_causal(a, c, e, self.scale)
          out = jax.shard_map(
              local_splash,
              mesh=mesh,
              in_specs=(spec, spec, spec),
              out_specs=spec,
              check_vma=False,
          )(q[0], k[0], v[0])[None]
        else:
          out = _splash_causal(q[0], k[0], v[0], self.scale)[None]
      elif TEXT_ATTN_CHUNK and mask is None and s > TEXT_ATTN_CHUNK:
        c = TEXT_ATTN_CHUNK
        n_blk = -(-s // c)
        qp = jnp.pad(q, ((0, 0), (0, n_blk * c - s), (0, 0), (0, 0)))
        qc = qp.reshape(b, n_blk, c, self.heads, self.head_dim).swapaxes(0, 1)
        cols = jnp.arange(s)[None, :]
        kf, vf = k.astype(jnp.float32), v.astype(jnp.float32)

        def one_chunk(args):
          qi, start = args
          rows = start + jnp.arange(c)[:, None]
          m = (cols <= rows)[None, None]
          with jax.default_matmul_precision("highest"):
            return jax.nn.dot_product_attention(
                qi.astype(jnp.float32), kf, vf, mask=m, scale=self.scale
            )

        out = jax.lax.map(one_chunk, (qc, jnp.arange(n_blk) * c))
        out = out.swapaxes(0, 1).reshape(b, n_blk * c, self.heads, self.head_dim)[:, :s]
        out = out.astype(v.dtype)
      else:
        def _local_sdpa(ql, kl, vl, ml=None):
          with jax.default_matmul_precision("highest"):
            return jax.nn.dot_product_attention(
                ql.astype(jnp.float32),
                kl.astype(jnp.float32),
                vl.astype(jnp.float32),
                mask=ml,
                is_causal=ml is None,
                scale=self.scale,
            ).astype(vl.dtype)

        if use_tp_shard:
          spec4 = P(None, None, axis, None)
          if mask is None:
            out = jax.shard_map(
                lambda ql, kl, vl: _local_sdpa(ql, kl, vl, None),
                mesh=mesh,
                in_specs=(spec4, spec4, spec4),
                out_specs=spec4,
                check_vma=False,
            )(q, k, v)
          else:
            out = jax.shard_map(
                _local_sdpa,
                mesh=mesh,
                in_specs=(spec4, spec4, spec4, P()),
                out_specs=spec4,
                check_vma=False,
            )(q, k, v, mask)
        else:
          out = _local_sdpa(q, k, v, mask)

      if gate is not None:
        out = out * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(out.dtype)
      return self.o_proj(out.reshape(b, s, self.heads * self.head_dim))

  class Qwen3_5GatedDeltaNet(nnx.Module):
    """Exact Qwen3.5 Gated DeltaNet (``linear_attention``) with TP=8 head sharding.

    HF's ``in_proj_qkv`` and ``conv1d`` are stored split into Q, K, and V
    components so ``num_k_heads=16`` (2/chip) and ``num_v_heads=48`` (6/chip)
    shard identically across ``TP=8`` with zero cross-chip communication until
    ``out_proj``.
    """

    def __init__(
        self,
        cfg: Magi2Qwen35Config,
        *,
        dtype,
        gdn_mode: str = DEFAULT_GDN_MODE,
        chunk_size: int = DEFAULT_GDN_CHUNK_SIZE,
        rngs=None,
    ):
      self.hidden_size = cfg.hidden_size
      self.num_k_heads = cfg.linear_num_key_heads
      self.num_v_heads = cfg.linear_num_value_heads
      self.head_k_dim = cfg.linear_key_head_dim
      self.head_v_dim = cfg.linear_value_head_dim
      self.key_dim = self.num_k_heads * self.head_k_dim
      self.value_dim = self.num_v_heads * self.head_v_dim
      self.conv_kernel_size = cfg.linear_conv_kernel_dim
      self.gdn_mode = gdn_mode
      self.chunk_size = int(chunk_size)
      self._mesh = None
      self._axis_name = "tp"

      # Split Q/K/V projections (loaded from HF's concatenated in_proj_qkv.weight).
      self.in_proj_q = Linear(cfg.hidden_size, self.key_dim, dtype=dtype, rngs=rngs)
      self.in_proj_k = Linear(cfg.hidden_size, self.key_dim, dtype=dtype, rngs=rngs)
      self.in_proj_v = Linear(cfg.hidden_size, self.value_dim, dtype=dtype, rngs=rngs)
      self.in_proj_z = Linear(cfg.hidden_size, self.value_dim, dtype=dtype, rngs=rngs)
      self.in_proj_b = Linear(cfg.hidden_size, self.num_v_heads, dtype=dtype, rngs=rngs)
      self.in_proj_a = Linear(cfg.hidden_size, self.num_v_heads, dtype=dtype, rngs=rngs)

      # Split depthwise causal conv1d kernels (loaded from HF's conv1d.weight [C, 1, K]).
      self.conv1d_q = nnx.Param(jnp.zeros((self.key_dim, self.conv_kernel_size), dtype=dtype))
      self.conv1d_k = nnx.Param(jnp.zeros((self.key_dim, self.conv_kernel_size), dtype=dtype))
      self.conv1d_v = nnx.Param(jnp.zeros((self.value_dim, self.conv_kernel_size), dtype=dtype))

      # Discretization parameters kept in float32 to prevent exp(A_log) overflow.
      self.A_log = nnx.Param(jnp.zeros((self.num_v_heads,), dtype=jnp.float32))
      self.dt_bias = nnx.Param(jnp.ones((self.num_v_heads,), dtype=jnp.float32))

      self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
      self.out_proj = Linear(self.value_dim, cfg.hidden_size, dtype=dtype, rngs=rngs)

    def __call__(self, x, linear_mask=None):
      from jax.sharding import PartitionSpec as P

      if linear_mask is not None:
        # Match apply_mask_to_padding_states in modeling_qwen3_5.py:L223.
        x = (x * linear_mask[:, :, None].astype(x.dtype)).astype(x.dtype)

      b, s, _ = x.shape

      q = causal_depthwise_conv1d_silu(self.in_proj_q(x), self.conv1d_q[...])
      k = causal_depthwise_conv1d_silu(self.in_proj_k(x), self.conv1d_k[...])
      v = causal_depthwise_conv1d_silu(self.in_proj_v(x), self.conv1d_v[...])
      z = self.in_proj_z(x).reshape(b, s, self.num_v_heads, self.head_v_dim)
      b_gate = self.in_proj_b(x)
      a_gate = self.in_proj_a(x)

      q = q.reshape(b, s, self.num_k_heads, self.head_k_dim)
      k = k.reshape(b, s, self.num_k_heads, self.head_k_dim)
      v = v.reshape(b, s, self.num_v_heads, self.head_v_dim)

      beta = jax.nn.sigmoid(b_gate.astype(jnp.float32))
      g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
          a_gate.astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
      )

      v_per_k = self.num_v_heads // self.num_k_heads
      if v_per_k > 1:
        # jnp.repeat matches torch.repeat_interleave along head axis 2.
        q = jnp.repeat(q, v_per_k, axis=2)
        k = jnp.repeat(k, v_per_k, axis=2)

      def _run_rule(ql, kl, vl, gl, betal):
        if self.gdn_mode == "recurrent":
          out_l, _ = recurrent_gated_delta_rule(
              ql, kl, vl, gl, betal, use_qk_l2norm=True
          )
          return out_l

        # Optional Tokamax Pallas Mosaic TPU Gated DeltaNet (2.6x faster on TPU7x).
        gdn_backend = os.environ.get("DT_MAGI2_GDN_BACKEND", "auto").strip().lower()
        if gdn_backend not in ("jax", "ref", "off", "0", "false") and ql.shape[-1] == 128 and vl.shape[-1] == 128:
          try:
            if any(d.platform == "tpu" for d in jax.devices()):
              from tokamax._src.ops.kda import api as _tm_kda  # type: ignore[import-not-found]
              s_orig = ql.shape[1]
              pad_s = (-s_orig) % 64
              if pad_s > 0:
                ql_p = jnp.pad(ql, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
                kl_p = jnp.pad(kl, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
                vl_p = jnp.pad(vl, ((0, 0), (0, pad_s), (0, 0), (0, 0)))
                gl_p = jnp.pad(gl, ((0, 0), (0, pad_s), (0, 0)))
                bl_p = jnp.pad(betal, ((0, 0), (0, pad_s), (0, 0)))
              else:
                ql_p, kl_p, vl_p, gl_p, bl_p = ql, kl, vl, gl, betal
              out_p, _ = _tm_kda.gated_delta_net(
                  ql_p.astype(jnp.bfloat16),
                  kl_p.astype(jnp.bfloat16),
                  vl_p.astype(jnp.bfloat16),
                  gl_p.astype(jnp.float32),
                  bl_p.astype(jnp.float32),
                  use_qk_l2norm=True,
                  implementation="mosaic",
              )
              return out_p[:, :s_orig, :, :].astype(vl.dtype)
          except Exception:
            if gdn_backend in ("tokamax", "pallas", "1", "true"):
              raise

        out_l, _ = chunk_gated_delta_rule(
            ql, kl, vl, gl, betal, chunk_size=self.chunk_size, use_qk_l2norm=True
        )
        return out_l

      mesh = self._mesh
      axis = self._axis_name
      tp = int(mesh.shape[axis]) if (mesh is not None and axis in mesh.axis_names) else 1
      if mesh is not None and tp > 1 and (self.num_v_heads % tp == 0):
        spec4 = P(None, None, axis, None)
        spec3 = P(None, None, axis)
        core_out = jax.shard_map(
            _run_rule,
            mesh=mesh,
            in_specs=(spec4, spec4, spec4, spec3, spec3),
            out_specs=spec4,
            check_vma=False,
        )(q, k, v, g, beta)
      else:
        core_out = _run_rule(q, k, v, g, beta)

      core_out = self.norm(core_out, z)
      core_out = core_out.reshape(b, s, self.value_dim)
      return self.out_proj(core_out)

  class Qwen3_5MLP(nnx.Module):
    def __init__(self, hidden: int, intermediate: int, *, dtype, rngs=None):
      self.gate_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.up_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.down_proj = Linear(intermediate, hidden, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

  class Qwen3_5DecoderLayer(nnx.Module):
    def __init__(
        self,
        cfg: Magi2Qwen35Config,
        layer_idx: int,
        *,
        dtype,
        gdn_mode: str = DEFAULT_GDN_MODE,
        chunk_size: int = DEFAULT_GDN_CHUNK_SIZE,
        rngs=None,
    ):
      self.layer_idx = layer_idx
      self.is_full_attn = cfg.is_full_attn(layer_idx)
      self.input_layernorm = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
      if self.is_full_attn:
        self.self_attn = Qwen3_5Attention(cfg, dtype=dtype, rngs=rngs)
      else:
        self.linear_attn = Qwen3_5GatedDeltaNet(
            cfg, dtype=dtype, gdn_mode=gdn_mode, chunk_size=chunk_size, rngs=rngs
        )
      self.post_attention_layernorm = ZeroCenteredRMSNorm(
          cfg.hidden_size, cfg.rms_norm_eps
      )
      self.mlp = Qwen3_5MLP(cfg.hidden_size, cfg.intermediate_size, dtype=dtype, rngs=rngs)

    def __call__(self, x, cos, sin, full_mask=None, linear_mask=None):
      residual = x
      h = self.input_layernorm(x)
      if self.is_full_attn:
        h = self.self_attn(h, cos, sin, mask=full_mask)
      else:
        h = self.linear_attn(h, linear_mask=linear_mask)
      x = residual + h
      return x + self.mlp(self.post_attention_layernorm(x))

  class Qwen3_5TextModel(nnx.Module):
    """Qwen3.5-27B text encoder with early exit at ``num_hidden_layers - skip_layer``."""

    def __init__(
        self,
        cfg: Magi2Qwen35Config,
        *,
        dtype=jnp.bfloat16,
        keep_layers: int | None = None,
        gdn_mode: str = DEFAULT_GDN_MODE,
        chunk_size: int = DEFAULT_GDN_CHUNK_SIZE,
        rngs=None,
    ):
      self.cfg = cfg
      self.skip_layer = int(cfg.skip_layer)
      if keep_layers is None:
        keep_layers = cfg.num_hidden_layers - self.skip_layer
      if keep_layers < 0 or keep_layers > cfg.num_hidden_layers:
        raise ValueError(
            f"Invalid keep_layers={keep_layers} for num_hidden_layers={cfg.num_hidden_layers}"
        )
      self.keep_layers = int(keep_layers)
      self._mesh = None
      self._axis_name = "tp"

      self.embed_tokens = nnx.Param(
          jnp.zeros((cfg.vocab_size, cfg.hidden_size), dtype=dtype)
      )
      self.layers = nnx.List([
          Qwen3_5DecoderLayer(
              cfg, i, dtype=dtype, gdn_mode=gdn_mode, chunk_size=chunk_size, rngs=rngs
          )
          for i in range(self.keep_layers)
      ])
      # Only allocate final RMSNorm when skip_layer == 0 (all layers run).
      self.norm = (
          ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
          if self.skip_layer == 0 and self.keep_layers == cfg.num_hidden_layers
          else None
      )

    def set_mesh(self, mesh, axis_name: str | None = None):
      axis = _resolve_axis_name(mesh, axis_name)
      self._mesh = mesh
      self._axis_name = axis
      for layer in self.layers:
        mixer = layer.self_attn if layer.is_full_attn else layer.linear_attn
        mixer._mesh = mesh
        mixer._axis_name = axis
      return self

    def _compute_default_cos_sin(self, seq_len: int):
      rotary_dim = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
      inv_freq = 1.0 / (
          jnp.asarray(self.cfg.rope_theta, dtype=jnp.float32)
          ** (
              jnp.arange(0, rotary_dim, 2, dtype=jnp.float32)
              / jnp.asarray(rotary_dim, dtype=jnp.float32)
          )
      )
      pos = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
      angles = pos * inv_freq[None, :]
      cos_h = jnp.cos(angles)
      sin_h = jnp.sin(angles)
      return jnp.concatenate([cos_h, cos_h], axis=-1), jnp.concatenate(
          [sin_h, sin_h], axis=-1
      )

    def __call__(
        self,
        input_ids,
        cos=None,
        sin=None,
        attention_mask=None,
        inputs_embeds=None,
        stop_at_layer: int | None = None,
    ):
      """Forward pass returning hidden states after ``stop_at_layer`` layers.

      When ``skip_layer == 2`` (MAGI-2 default), ``keep_layers == 62``, so this
      runs layers ``0..61`` and returns ``hidden_states[-3]`` directly without
      final RMSNorm. When ``skip_layer == 0`` and all layers run, applies final
      ``self.norm`` to match ``outputs.last_hidden_state``.
      """
      from jax.sharding import NamedSharding, PartitionSpec as P

      x = (
          self.embed_tokens[...][input_ids]
          if inputs_embeds is None
          else inputs_embeds
      )
      # Gather column-sharded embedding output once into replicated residual stream.
      if self._mesh is not None:
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(self._mesh, P(None, None, None))
        )

      _, seq_len, _ = x.shape
      if cos is None or sin is None:
        cos, sin = self._compute_default_cos_sin(seq_len)

      full_mask = None
      linear_mask = None
      if attention_mask is not None:
        if attention_mask.ndim == 2:
          linear_mask = attention_mask
          causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
          key_valid = attention_mask.astype(jnp.bool_)[:, None, None, :]
          full_mask = causal[None, None, :, :] & key_valid
          # Guard against all-False rows (e.g. left-padded dummy rows) to avoid NaN softmax.
          any_valid = jnp.any(full_mask, axis=-1, keepdims=True)
          eye = jnp.eye(seq_len, dtype=jnp.bool_)[None, None, :, :]
          full_mask = jnp.where(any_valid, full_mask, eye)
        elif attention_mask.ndim == 4:
          full_mask = attention_mask
        else:
          raise ValueError(
              f"Expected 2D or 4D attention_mask, got shape {attention_mask.shape}"
          )

      limit = self.keep_layers if stop_at_layer is None else min(stop_at_layer, self.keep_layers)
      for idx in range(limit):
        x = self.layers[idx](x, cos, sin, full_mask=full_mask, linear_mask=linear_mask)

      if (
          self.norm is not None
          and self.skip_layer == 0
          and limit == self.cfg.num_hidden_layers
      ):
        x = self.norm(x)
      return x

  _MODULES_CACHE = (
      ZeroCenteredRMSNorm,
      RMSNormGated,
      Linear,
      Qwen3_5Attention,
      Qwen3_5GatedDeltaNet,
      Qwen3_5MLP,
      Qwen3_5DecoderLayer,
      Qwen3_5TextModel,
  )
  return _MODULES_CACHE


def build(
    cfg_or_dir: Magi2Qwen35Config | str | Path,
    *,
    dtype=None,
    skip_layer: int | None = None,
    keep_layers: int | None = None,
    gdn_mode: str = DEFAULT_GDN_MODE,
    chunk_size: int = DEFAULT_GDN_CHUNK_SIZE,
    mesh=None,
    axis_name: str | None = None,
    abstract: bool = False,
    rngs=None,
):
  """Build an uninitialized ``Qwen3_5TextModel`` shaped from a config or checkpoint dir."""
  from dataclasses import replace

  import jax.numpy as jnp
  from flax import nnx

  if dtype is None:
    dtype = jnp.bfloat16

  if isinstance(cfg_or_dir, Magi2Qwen35Config):
    cfg = cfg_or_dir
    if skip_layer is not None and skip_layer != cfg.skip_layer:
      cfg = replace(cfg, skip_layer=skip_layer)
  else:
    cfg = config_from(
        cfg_or_dir,
        skip_layer=DEFAULT_SKIP_LAYER if skip_layer is None else skip_layer,
    )

  (_, _, _, _, _, _, _, Qwen3_5TextModel) = _build_modules()

  def _make():
    return Qwen3_5TextModel(
        cfg,
        dtype=dtype,
        keep_layers=keep_layers,
        gdn_mode=gdn_mode,
        chunk_size=chunk_size,
        rngs=rngs or nnx.Rngs(0),
    )

  model = nnx.eval_shape(_make) if abstract else _make()
  if mesh is not None:
    model.set_mesh(mesh, axis_name=axis_name)
  return model, cfg


# ---------------------------------------------------------------------------
# Streaming safetensors & state_dict weight loading
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = ("model.visual.", "visual.", "lm_head.", "mtp.")
_STRIP_PREFIXES = ("model.language_model.", "language_model.", "model.")


def _canonical_text_key(key: str) -> str | None:
  """Normalize HF safetensors key to ``Qwen3_5TextModel``-relative path, or None to skip."""
  if key.startswith(_SKIP_PREFIXES):
    return None
  name = key
  for pref in _STRIP_PREFIXES:
    if name.startswith(pref):
      name = name[len(pref) :]
      break
  if name.startswith(("embed_tokens.", "layers.", "norm.")):
    return name
  return None


def _resolve_param(model, parts: list[str]):
  if parts == ["embed_tokens", "weight"]:
    return model.embed_tokens
  obj = model
  for part in parts:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def _param_aval(param):
  return param.value if hasattr(param, "value") else param[...]


def _set_param(param, value):
  if hasattr(param, "value"):
    param.value = value
  else:
    param[...] = value


def _split_gdn_tensor(
    name: str, arr: np.ndarray, cfg: Magi2Qwen35Config
) -> list[tuple[list[str], np.ndarray]]:
  """Expand HF's fused ``in_proj_qkv`` and ``conv1d`` into TP-friendly Q/K/V params."""
  parts = name.split(".")
  key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
  val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
  conv_dim = 2 * key_dim + val_dim

  if len(parts) == 5 and parts[2] == "linear_attn" and parts[3] == "in_proj_qkv" and parts[4] == "weight":
    if arr.shape[0] != conv_dim:
      raise ValueError(f"{name}: expected {conv_dim} rows, got {arr.shape}")
    prefix = parts[:3]
    return [
        (prefix + ["in_proj_q", "weight"], arr[:key_dim]),
        (prefix + ["in_proj_k", "weight"], arr[key_dim : 2 * key_dim]),
        (prefix + ["in_proj_v", "weight"], arr[2 * key_dim :]),
    ]

  if len(parts) == 5 and parts[2] == "linear_attn" and parts[3] == "conv1d" and parts[4] == "weight":
    w = arr[:, 0, :] if arr.ndim == 3 else arr
    if w.shape != (conv_dim, cfg.linear_conv_kernel_dim):
      raise ValueError(
          f"{name}: expected {(conv_dim, cfg.linear_conv_kernel_dim)}, got {w.shape}"
      )
    prefix = parts[:3]
    return [
        (prefix + ["conv1d_q"], w[:key_dim]),
        (prefix + ["conv1d_k"], w[key_dim : 2 * key_dim]),
        (prefix + ["conv1d_v"], w[2 * key_dim :]),
    ]

  return [(parts, arr)]


def _should_load_key(name: str, keep_layers: int, has_final_norm: bool) -> bool:
  if name == "norm.weight":
    return has_final_norm
  parts = name.split(".")
  if parts[0] == "layers":
    return int(parts[1]) < keep_layers
  return name == "embed_tokens.weight"


def _select_needed_shards(
    text_encoder_dir: str | Path, keep_layers: int, has_final_norm: bool
) -> list[str]:
  """Return only safetensors shards that contain at least one required weight."""
  text_dir = str(text_encoder_dir)
  index_path = os.path.join(text_dir, "model.safetensors.index.json")
  if os.path.isfile(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
      weight_map = json.load(f).get("weight_map", {})
    needed_files: set[str] = set()
    for raw_key, shard_name in weight_map.items():
      canon = _canonical_text_key(raw_key)
      if canon is not None and _should_load_key(canon, keep_layers, has_final_norm):
        needed_files.add(os.path.join(text_dir, shard_name))
    if needed_files:
      return sorted(needed_files)

  shards = sorted(glob.glob(os.path.join(text_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"No .safetensors files found in {text_dir}")
  return shards


def load(
    model,
    text_encoder_dir: str | Path,
    *,
    mesh=None,
    spec_fn=None,
    axis_name: str | None = None,
    memory_kind: str | None = None,
    max_workers: int | None = None,
) -> int:
  """Stream Qwen3.5 text-encoder safetensors directly into ``model`` on ``mesh``.

  Uses ``diffusers_tpu.safetensors_io.safe_open`` with ``framework="np"``
  (sequential reads, zero PyTorch dependency). Skips ``model.visual.*``,
  ``lm_head.*``, ``mtp.*``, layers ``>= model.keep_layers`` (layers 62, 63 when
  ``skip_layer=2``), and ``norm.weight`` when ``model.norm is None``.
  """
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import NamedSharding

  from ...safetensors_io import safe_open

  cfg = model.cfg
  keep_layers = model.keep_layers
  has_final_norm = model.norm is not None

  if mesh is not None:
    resolved_axis = _resolve_axis_name(mesh, axis_name)
    model.set_mesh(mesh, axis_name=resolved_axis)
    if spec_fn is None:
      spec_fn = make_param_spec(cfg, mesh, axis_name=resolved_axis)

  shards = _select_needed_shards(text_encoder_dir, keep_layers, has_final_norm)

  def _place(parts: list[str], arr: np.ndarray, want_dtype):
    casted = np.asarray(arr, dtype=np.dtype(want_dtype))
    if mesh is not None and spec_fn is not None:
      sharding = NamedSharding(
          mesh, spec_fn(parts, casted.ndim), memory_kind=memory_kind
      )
      return jax.device_put(casted, sharding)
    return jnp.asarray(casted, dtype=want_dtype)

  def _read_shard(shard_path: str):
    placements: list[tuple[list[str], Any]] = []
    with safe_open(shard_path, framework="np") as reader:
      for raw_key in reader.keys():
        canon = _canonical_text_key(raw_key)
        if canon is None or not _should_load_key(canon, keep_layers, has_final_norm):
          continue
        arr = reader.get_tensor(raw_key)
        for parts, sub_arr in _split_gdn_tensor(canon, arr, cfg):
          param = _resolve_param(model, parts)
          want = _param_aval(param)
          if tuple(sub_arr.shape) != tuple(want.shape):
            raise ValueError(
                f"{raw_key} -> {'.'.join(parts)}: checkpoint {sub_arr.shape} vs model {want.shape}"
            )
          placements.append((parts, _place(parts, sub_arr, want.dtype)))
    return placements

  workers = (
      min(len(shards), 8)
      if max_workers is None
      else max(1, min(len(shards), max_workers))
  )
  if workers > 1 and mesh is not None:
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
      all_results = list(ex.map(_read_shard, shards))
  else:
    all_results = [_read_shard(s) for s in shards]

  loaded_paths: set[tuple[str, ...]] = set()
  for shard_items in all_results:
    for parts, value in shard_items:
      key_tuple = tuple(parts)
      if key_tuple in loaded_paths:
        raise ValueError(f"Duplicate weight loaded for {'.'.join(parts)}")
      _set_param(_resolve_param(model, parts), value)
      loaded_paths.add(key_tuple)

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if len(loaded_paths) != expected:
    raise ValueError(
        f"Loaded {len(loaded_paths)} tensors into {expected} parameters "
        f"(keep_layers={keep_layers}, has_final_norm={has_final_norm})"
    )
  return len(loaded_paths)


def load_from_state_dict(
    model,
    state_dict: dict[str, Any],
    *,
    mesh=None,
    spec_fn=None,
    axis_name: str | None = None,
) -> int:
  """Load weights from an in-memory dict of numpy or torch tensors (for tests/parity)."""
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import NamedSharding

  cfg = model.cfg
  keep_layers = model.keep_layers
  has_final_norm = model.norm is not None

  if mesh is not None:
    resolved_axis = _resolve_axis_name(mesh, axis_name)
    model.set_mesh(mesh, axis_name=resolved_axis)
    if spec_fn is None:
      spec_fn = make_param_spec(cfg, mesh, axis_name=resolved_axis)

  loaded = 0
  for raw_key, raw_val in state_dict.items():
    canon = _canonical_text_key(raw_key)
    if canon is None or not _should_load_key(canon, keep_layers, has_final_norm):
      continue
    if hasattr(raw_val, "detach"):
      t = raw_val.detach().cpu()
      if str(t.dtype) == "torch.bfloat16":
        import ml_dtypes
        import torch

        arr = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
      else:
        arr = t.numpy()
    else:
      arr = np.asarray(raw_val)

    for parts, sub_arr in _split_gdn_tensor(canon, arr, cfg):
      param = _resolve_param(model, parts)
      want = _param_aval(param)
      if tuple(sub_arr.shape) != tuple(want.shape):
        raise ValueError(
            f"{raw_key} -> {'.'.join(parts)}: shape {sub_arr.shape} vs model {want.shape}"
        )
      casted = np.asarray(sub_arr, dtype=np.dtype(want.dtype))
      if mesh is not None and spec_fn is not None:
        val = jax.device_put(casted, NamedSharding(mesh, spec_fn(parts, casted.ndim)))
      else:
        val = jnp.asarray(casted, dtype=want.dtype)
      _set_param(param, val)
      loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"Loaded {loaded} tensors into {expected} parameters")
  return loaded


# ---------------------------------------------------------------------------
# High-level encoder wrapper (drop-in mirror of MAGI-2 Qwen35TextEncoder)
# ---------------------------------------------------------------------------

_DEFAULT_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 7040)


def _select_bucket(seq_len: int, buckets: tuple[int, ...] = _DEFAULT_BUCKETS) -> int:
  for b in buckets:
    if seq_len <= b:
      return b
  # Round up to next multiple of 64.
  return ((seq_len + 63) // 64) * 64


class Qwen35TextEncoder:
  """End-to-end MAGI-2 Qwen3.5-27B text encoder (tokenizer + sharded NNX model).

  Because every layer in Qwen3.5 (causal Conv1d, Gated DeltaNet, causal GQA)
  is strictly causal, single-prompt ``B=1`` sequences can be right-padded to a
  small set of static length buckets and sliced back to ``[:, :orig_len, :]``
  with bit-identical outputs and minimal TPU JIT recompilation.
  """

  def __init__(
      self,
      model_path: str | Path,
      *,
      mesh=None,
      axis_name: str | None = None,
      dtype=None,
      skip_layer: int = DEFAULT_SKIP_LAYER,
      max_length: int = DEFAULT_MAX_LENGTH,
      memory_kind: str | None = None,
      use_buckets: bool = True,
      buckets: tuple[int, ...] = _DEFAULT_BUCKETS,
  ):
    self.model_path = str(model_path)
    self.skip_layer = int(skip_layer)
    self.max_length = int(max_length)
    self.mesh = mesh
    self.axis_name = _resolve_axis_name(mesh, axis_name)
    self.use_buckets = use_buckets
    self.buckets = buckets

    self.tokenizer = load_tokenizer(self.model_path)
    self.model, self.cfg = build(
        self.model_path,
        dtype=dtype,
        skip_layer=self.skip_layer,
        mesh=mesh,
        axis_name=self.axis_name,
        abstract=True,
    )
    load(
        self.model,
        self.model_path,
        mesh=mesh,
        axis_name=self.axis_name,
        memory_kind=memory_kind,
    )
    self._compiled: dict[tuple[int, bool], Any] = {}

  def _get_forward_fn(self, seq_len: int, with_mask: bool):
    key = (seq_len, with_mask)
    if key not in self._compiled:
      import jax
      from flax import nnx

      graphdef = nnx.graphdef(self.model)

      @jax.jit
      def _run(st, input_ids, cos, sin, mask):
        m = nnx.merge(graphdef, st)
        return m(input_ids, cos=cos, sin=sin, attention_mask=mask)

      self._compiled[key] = _run
    return self._compiled[key]

  def get_target_token_indices(
      self, prompt: str, target_str: str | None
  ) -> list[int] | None:
    return get_target_token_indices(
        self.tokenizer, prompt, target_str, max_length=self.max_length
    )

  def get_special_token(
      self, prompt: str, target_strs: list[str], text_feature
  ):
    return get_special_token(
        self.tokenizer,
        prompt,
        target_strs,
        text_feature,
        max_length=self.max_length,
    )

  def pool_figure_tokens(self, prompt: str, figure_ids, text_feature):
    return pool_figure_tokens(
        self.tokenizer,
        prompt,
        figure_ids,
        text_feature,
        max_length=self.max_length,
    )

  def encode(self, prompt: str):
    """Encode a single prompt into ``(1, seq_len, 5120)`` hidden states."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P

    encoded = tokenize_prompt(
        self.tokenizer, prompt, max_length=self.max_length
    )
    ids_np = encoded["input_ids"].astype(np.int32)  # (1, orig_len)
    orig_len = int(ids_np.shape[1])

    run_len = _select_bucket(orig_len, self.buckets) if self.use_buckets else orig_len
    if run_len > orig_len:
      pad_id = int(self.tokenizer.pad_token_id or 0)
      ids_np = np.pad(
          ids_np, ((0, 0), (0, run_len - orig_len)), constant_values=pad_id
      )

    cos_np, sin_np = partial_rope_cos_sin(
        np.arange(run_len, dtype=np.int32),
        head_dim=self.cfg.head_dim,
        partial_rotary_factor=self.cfg.partial_rotary_factor,
        rope_theta=self.cfg.rope_theta,
    )

    if self.mesh is not None:
      rep = NamedSharding(self.mesh, P())
      ids = jax.device_put(ids_np, rep)
      cos = jax.device_put(cos_np, rep)
      sin = jax.device_put(sin_np, rep)
    else:
      ids = jnp.asarray(ids_np)
      cos = jnp.asarray(cos_np)
      sin = jnp.asarray(sin_np)

    fn = self._get_forward_fn(run_len, with_mask=False)
    out = fn(nnx.state(self.model), ids, cos, sin, None)
    return out[:, :orig_len, :]


# ---------------------------------------------------------------------------
# Self-test: tiny 27B-regime parity against transformers.Qwen3_5TextModel
# ---------------------------------------------------------------------------


def _test_prompt_preprocessing():
  sample_json = json.dumps({
      "global_layer": {
          "context": "A sunlit coastal café",
          "description": "Wide establishing shot of waves outside.",
          "aesthetics": {
              "style": "cinematic",
              "mood_atmosphere": "warm",
              "color_scheme": "golden",
          },
          "audio_baseline": {
              "dialogue": {"language": "en", "speaker_tags": ["Alice", "Bob"]},
              "ambience": "ocean waves",
          },
          "alive_subjects_static": [{
              "subject_id": "S1",
              "description": "woman drinking coffee",
              "position": "left",
              "orientation": "facing right",
              "visual_attributes": {"gender": "female", "clothing": "blue coat"},
          }],
      },
      "dynamic_layer": {
          "timeline_segments": [{
              "segment_basic_info": {
                  "timestamp_range": "0-2s",
                  "segment_description": "She smiles.",
              },
              "audio": {
                  "dialogue_lines": [
                      {"speaker": "Alice", "text": "Good morning!", "timestamp": "0.5s"}
                  ]
              },
          }]
      },
      "reference_layer": ["<Figure 1>"],
  })
  md = normalize_prompt(sample_json)
  assert "context: A sunlit coastal café" in md
  assert "aesthetics: style=cinematic; mood=warm; color=golden" in md
  assert "audio: language=en; speakers=Alice,Bob; ambience=ocean waves" in md
  assert " - S1 | woman drinking coffee (left; facing right) :: gender=female; clothing=blue coat" in md
  assert "   - dialogue Alice: Good morning! (0.5s)" in md
  assert md.endswith("<Figure 1>")
  assert normalize_prompt("plain text prompt") == "plain text prompt"
  assert normalize_prompt('{"foo": "bar"}') == '{"foo": "bar"}'


def _test_tp8_sharding_specs():
  from types import SimpleNamespace

  cfg = Magi2Qwen35Config()
  fake_mesh = SimpleNamespace(axis_names=("tp",), shape={"tp": 8})
  spec_fn = make_param_spec(cfg, fake_mesh)

  assert tuple(spec_fn(["embed_tokens", "weight"], 2)) == (None, "tp")
  assert tuple(spec_fn(["layers", "3", "self_attn", "q_proj", "weight"], 2)) == ("tp", None)
  # 4 KV heads across 8 chips MUST replicate instead of failing 4 % 8 != 0!
  assert tuple(spec_fn(["layers", "3", "self_attn", "k_proj", "weight"], 2)) == ()
  assert tuple(spec_fn(["layers", "3", "self_attn", "v_proj", "weight"], 2)) == ()
  assert tuple(spec_fn(["layers", "3", "self_attn", "o_proj", "weight"], 2)) == (None, "tp")
  assert tuple(spec_fn(["layers", "0", "linear_attn", "in_proj_q", "weight"], 2)) == ("tp", None)
  assert tuple(spec_fn(["layers", "0", "linear_attn", "in_proj_v", "weight"], 2)) == ("tp", None)
  assert tuple(spec_fn(["layers", "0", "linear_attn", "conv1d_q"], 2)) == ("tp", None)
  assert tuple(spec_fn(["layers", "0", "linear_attn", "A_log"], 1)) == ("tp",)
  assert tuple(spec_fn(["layers", "0", "linear_attn", "dt_bias"], 1)) == ("tp",)
  assert tuple(spec_fn(["layers", "0", "linear_attn", "out_proj", "weight"], 2)) == (None, "tp")
  assert tuple(spec_fn(["layers", "0", "mlp", "gate_proj", "weight"], 2)) == ("tp", None)
  assert tuple(spec_fn(["layers", "0", "mlp", "down_proj", "weight"], 2)) == (None, "tp")

  # Verify layer counts for layers 0..61 (skip_layer=2)
  assert sum(cfg.is_full_attn(i) for i in range(64)) == 16
  assert sum(cfg.is_full_attn(i) for i in range(62)) == 15
  assert sum(not cfg.is_full_attn(i) for i in range(62)) == 47


def test_tiny_parity() -> int:
  """Verify JAX Qwen3.5 against recurrent GDN, padding invariance, and HF ``Qwen3_5TextModel``."""
  import jax
  import jax.numpy as jnp

  _test_prompt_preprocessing()
  _test_tp8_sharding_specs()
  print("[1/4] Prompt preprocessing & TP=8 PartitionSpec checks PASSED")

  # Check chunkwise vs recurrent Gated Delta Rule on non-chunk-multiple & multi-chunk seqs.
  rng = np.random.default_rng(0)
  for seq in (19, 140):
    q = jnp.asarray(rng.standard_normal((2, seq, 6, 16)).astype(np.float32))
    k = jnp.asarray(rng.standard_normal((2, seq, 6, 16)).astype(np.float32))
    v = jnp.asarray(rng.standard_normal((2, seq, 6, 16)).astype(np.float32))
    g = jnp.asarray(-np.abs(rng.standard_normal((2, seq, 6))).astype(np.float32) * 0.2)
    beta = jnp.asarray(1.0 / (1.0 + np.exp(-rng.standard_normal((2, seq, 6)))).astype(np.float32))
    out_c, st_c = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32)
    out_r, st_r = recurrent_gated_delta_rule(q, k, v, g, beta)
    err_o = float(np.max(np.abs(np.asarray(out_c) - np.asarray(out_r))))
    err_s = float(np.max(np.abs(np.asarray(st_c) - np.asarray(st_r))))
    assert err_o < 1e-5 and err_s < 1e-5, f"GDN chunk vs recurrent mismatch: out={err_o}, st={err_s}"
  print("[2/4] Gated Delta Rule chunkwise vs recurrent parity PASSED")

  # Tiny 27B-regime config: exact 27B head ratios & TP=8 divisibility:
  # 6:1 GQA (24Q/4KV -> 4 % 8 != 0), 1:3 GDN (8K/24V -> 1K/3V per chip on TP=8),
  # 0.25 partial RoPE (8/32 dims), 6 layers (skip_layer=2 -> layers 0..3).
  layer_types = (
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
  )
  tiny_cfg_skip2 = Magi2Qwen35Config(
      num_hidden_layers=6,
      skip_layer=2,
      hidden_size=64,
      intermediate_size=128,
      vocab_size=256,
      rms_norm_eps=1e-6,
      full_attn_interval=4,
      num_attention_heads=24,
      num_key_value_heads=4,
      head_dim=32,
      attn_output_gate=True,
      partial_rotary_factor=0.25,
      rope_theta=10_000_000.0,
      linear_conv_kernel_dim=4,
      linear_num_key_heads=8,
      linear_key_head_dim=8,
      linear_num_value_heads=24,
      linear_value_head_dim=8,
      layer_types=layer_types,
  )
  tiny_cfg_skip0 = Magi2Qwen35Config(
      **{**tiny_cfg_skip2.__dict__, "skip_layer": 0}
  )

  try:
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
  except Exception as exc:
    print(f"[3/4] Skipping HuggingFace Qwen3_5TextModel parity (unavailable: {exc})")
    return 0

  torch.manual_seed(42)
  hf_cfg = Qwen3_5TextConfig(
      vocab_size=tiny_cfg_skip2.vocab_size,
      hidden_size=tiny_cfg_skip2.hidden_size,
      intermediate_size=tiny_cfg_skip2.intermediate_size,
      num_hidden_layers=tiny_cfg_skip2.num_hidden_layers,
      num_attention_heads=tiny_cfg_skip2.num_attention_heads,
      num_key_value_heads=tiny_cfg_skip2.num_key_value_heads,
      head_dim=tiny_cfg_skip2.head_dim,
      rms_norm_eps=tiny_cfg_skip2.rms_norm_eps,
      linear_conv_kernel_dim=tiny_cfg_skip2.linear_conv_kernel_dim,
      linear_num_key_heads=tiny_cfg_skip2.linear_num_key_heads,
      linear_key_head_dim=tiny_cfg_skip2.linear_key_head_dim,
      linear_num_value_heads=tiny_cfg_skip2.linear_num_value_heads,
      linear_value_head_dim=tiny_cfg_skip2.linear_value_head_dim,
      layer_types=list(layer_types),
      rope_parameters={
          "rope_type": "default",
          "rope_theta": tiny_cfg_skip2.rope_theta,
          "partial_rotary_factor": tiny_cfg_skip2.partial_rotary_factor,
          "mrope_section": [2, 1, 1],
      },
  )
  hf_model = Qwen3_5TextModel(hf_cfg).to(torch.float32).eval()
  # Randomize norm/gate parameters so 1+w vs w and dt_bias bugs cannot hide behind 0/1 inits.
  with torch.no_grad():
    for name, p in hf_model.named_parameters():
      if "norm.weight" in name or "layernorm.weight" in name:
        if "linear_attn.norm.weight" in name:
          p.copy_(1.0 + 0.2 * torch.randn_like(p))
        else:
          p.copy_(0.2 * torch.randn_like(p))
      elif name.endswith("dt_bias"):
        p.copy_(0.5 + 0.3 * torch.randn_like(p))

  sd = hf_model.state_dict()
  # Also test VLM-style `model.language_model.` prefix stripping + skipping visual/lm_head/mtp.
  prefixed_sd = {f"model.language_model.{k}": v for k, v in sd.items()}
  prefixed_sd["model.visual.dummy.weight"] = torch.zeros(4, 4)
  prefixed_sd["lm_head.weight"] = torch.zeros(256, 64)
  prefixed_sd["mtp.fc.weight"] = torch.zeros(64, 64)

  jax_skip2, _ = build(tiny_cfg_skip2, dtype=jnp.float32, chunk_size=16)
  load_from_state_dict(jax_skip2, prefixed_sd)
  jax_skip0, _ = build(tiny_cfg_skip0, dtype=jnp.float32, chunk_size=16)
  load_from_state_dict(jax_skip0, prefixed_sd)

  seq = 37  # non-multiple of chunk_size to test padding too
  input_ids_np = rng.integers(0, tiny_cfg_skip2.vocab_size, size=(1, seq), dtype=np.int32)
  cos_np, sin_np = partial_rope_cos_sin(
      np.arange(seq, dtype=np.int32),
      head_dim=tiny_cfg_skip2.head_dim,
      partial_rotary_factor=tiny_cfg_skip2.partial_rotary_factor,
      rope_theta=tiny_cfg_skip2.rope_theta,
  )

  with torch.no_grad():
    hf_out = hf_model(
        input_ids=torch.tensor(input_ids_np, dtype=torch.long),
        attention_mask=torch.ones((1, seq), dtype=torch.long),
        output_hidden_states=True,
        return_dict=True,
    )
    ref_skip2 = hf_out.hidden_states[-3].numpy()
    ref_skip0 = hf_out.last_hidden_state.numpy()

  got_skip2 = np.asarray(jax_skip2(jnp.asarray(input_ids_np), jnp.asarray(cos_np), jnp.asarray(sin_np)))
  got_skip0 = np.asarray(jax_skip0(jnp.asarray(input_ids_np), jnp.asarray(cos_np), jnp.asarray(sin_np)))

  diff2 = np.abs(ref_skip2 - got_skip2)
  rel2 = float(diff2.max() / max(float(np.abs(ref_skip2).max()), 1e-9))
  diff0 = np.abs(ref_skip0 - got_skip0)
  rel0 = float(diff0.max() / max(float(np.abs(ref_skip0).max()), 1e-9))
  print(f"[3/4] HF parity skip_layer=2: max|diff|={diff2.max():.3e} rel={rel2:.3e}")
  print(f"      HF parity skip_layer=0: max|diff|={diff0.max():.3e} rel={rel0:.3e}")
  assert rel2 < 1e-4 and rel0 < 1e-4

  # Verify strict causality / right-padding bucket invariance for B=1.
  ids_pad = np.pad(input_ids_np, ((0, 0), (0, 64 - seq)), constant_values=0)
  got_pad = np.asarray(jax_skip2(jnp.asarray(ids_pad)))[:, :seq, :]
  pad_err = float(np.max(np.abs(got_pad - got_skip2)))
  assert pad_err < 1e-5, f"Right-padding changed causal outputs: {pad_err}"

  # Verify right-padded B=2 batch with attention_mask against HF.
  ids_b2 = rng.integers(0, tiny_cfg_skip2.vocab_size, size=(2, 24), dtype=np.int32)
  mask_b2 = np.ones((2, 24), dtype=np.int32)
  mask_b2[1, 13:] = 0
  with torch.no_grad():
    ref_b2 = hf_model(
        input_ids=torch.tensor(ids_b2, dtype=torch.long),
        attention_mask=torch.tensor(mask_b2, dtype=torch.long),
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-3].numpy()
  got_b2 = np.asarray(
      jax_skip2(jnp.asarray(ids_b2), attention_mask=jnp.asarray(mask_b2))
  )
  valid = mask_b2.astype(bool)[..., None]
  diff_b2 = np.max(np.abs(np.where(valid, ref_b2 - got_b2, 0.0)))
  rel_b2 = float(diff_b2 / max(float(np.abs(ref_b2).max()), 1e-9))
  print(f"[4/4] Right-padding invariance ({pad_err:.3e}) & B=2 masked parity (rel={rel_b2:.3e}) PASSED")
  assert rel_b2 < 1e-4

  # Multi-device TP parity check (exercises TP=8 with 24Q/4KV & 8K/24V on 8-chip hosts).
  n_dev = jax.device_count()
  tp_deg = 8 if n_dev >= 8 else (4 if n_dev >= 4 else (2 if n_dev >= 2 else 1))
  if tp_deg >= 2:
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:tp_deg]), ("tp",))
    jax_tp, _ = build(tiny_cfg_skip2, dtype=jnp.float32, chunk_size=16, mesh=mesh)
    load_from_state_dict(jax_tp, prefixed_sd, mesh=mesh)
    got_tp = np.asarray(jax_tp(jnp.asarray(input_ids_np), jnp.asarray(cos_np), jnp.asarray(sin_np)))
    tp_err = float(np.max(np.abs(got_tp - got_skip2)))
    print(f"      TP={tp_deg} sharded parity: max|diff|={tp_err:.3e} PASSED")
    assert tp_err < 1e-4

  print("MATCH")
  return 0


if __name__ == "__main__":
  raise SystemExit(test_tiny_parity())
