"""Optional PE (Prompt Expansion) helper for Sand.ai MAGI-2-preview.

Short visual-only prompts are heavily attenuated in audio by MAGI-2's official
429-token ``DEFAULT_NEGATIVE_PROMPT`` (which suppresses silence/music noise) under
``audio_cfg=7.0``. This module expands a plain-text prompt into the exact upstream
JSON schema read by ``json_to_compact_markdown`` (specifically
``dynamic_layer.timeline_segments`` with ``dialogue_lines``, ``ambient_sound``,
and ``sound_effects``).

Modes:
  - ``builtin`` (default, 100% offline, zero latency): Deterministic structured PE
    builder that extracts any quoted dialogue (``"..."``, ``“...”``, ``「...」``)
    into timestamped ``dialogue_lines`` and synthesizes grounded Foley & ambient
    sound cues matching the scene.
  - ``api``: Optional Gemini/OpenAI-compatible REST expander (uses
    ``GEMINI_API_KEY`` or ``OPENAI_API_KEY``), falling back seamlessly to
    ``builtin`` if unavailable or if the returned JSON fails schema validation.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

_QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]{2,200})"'),
    re.compile(r'“([^”\n]{2,200})”'),
    re.compile(r'「([^」\n]{2,200})」'),
    re.compile(r"'([^'\n]{3,200})'"),
)


def _extract_dialogues(text: str) -> list[str]:
  found: list[str] = []
  for pat in _QUOTE_PATTERNS:
    for m in pat.finditer(text):
      s = m.group(1).strip()
      if s and s not in found:
        found.append(s)
  return found


def _infer_scene_audio(text: str) -> tuple[str, str]:
  """Infer realistic ambient sound and grounded Foley from keywords."""
  low = text.lower()
  if any(k in low for k in ("cafe", "coffee", "barista", "latte", "espresso", "咖啡")):
    return (
        "Soft cozy cafe ambience with distant espresso machine hiss, gentle acoustic room tone, and faint rain outside.",
        "Crisp ceramic cup gently clinking onto wooden saucer, faint steam hiss, soft rustle of fabric.",
    )
  if any(k in low for k in ("snow", "fox", "forest", "woods", "mountain", "winter", "雪", "森林")):
    return (
        "Quiet crisp outdoor winter air with soft breeze rustling through distant pine trees.",
        "Soft rhythmic crunching footsteps on fresh powdery snow, gentle breath in cold air.",
    )
  if any(k in low for k in ("ocean", "beach", "sea", "wave", "rain", "storm", "海", "雨")):
    return (
        "Natural open-air coastal ambience with rolling ocean waves and steady sea breeze.",
        "Rhythmic gentle water splashes, soft wind gusts against clothing.",
    )
  if any(k in low for k in ("city", "street", "car", "neon", "night", "城市", "街")):
    return (
        "Natural urban street ambience with distant low traffic hum and gentle night breeze.",
        "Crisp footsteps on pavement, subtle clothing movement and ambient city echoes.",
    )
  return (
      "Clean natural environmental room tone matching the setting with subtle spatial depth.",
      "Subtle realistic movement Foley, natural fabric rustle, and grounded contact sounds matching on-screen motion.",
  )


def build_builtin_pe_json(
    prompt: str,
    *,
    seconds: float = 10.0,
    is_i2v: bool = False,
    num_refs: int = 0,
) -> dict[str, Any]:
  """Deterministically wrap a short prompt into valid upstream MAGI-2 PE JSON."""
  clean_prompt = prompt.strip()
  dialogues = _extract_dialogues(clean_prompt)
  ambient, foley = _infer_scene_audio(clean_prompt)

  total_s = max(2.0, float(seconds))
  t1 = round(total_s * 0.35, 1)
  t2 = round(total_s * 0.72, 1)
  t_end = round(total_s, 1)

  subj_ref = "<Figure 1>" if is_i2v and num_refs <= 1 else "the main subject"
  has_dialogue = len(dialogues) > 0
  speaker_desc = (
      f"Primary subject ({subj_ref}) portrayed with natural photorealistic lighting, "
      "expressive micro-expressions, and consistent identity across all frames."
  )

  # Place dialogue cleanly in segment 2 (mid-clip) for natural lip-sync and breathing room.
  seg2_dialogue: list[dict[str, Any]] = []
  if has_dialogue:
    seg2_dialogue.append({
        "speaker": "subject 1",
        "text": " ".join(dialogues),
        "emotion": "warm, natural, conversational",
        "voice_characteristics": "clear natural close-mic human voice with natural breath and cadence",
    })

  timeline_segments = [
      {
          "time_range": f"0.0s - {t1:.1f}s",
          "visual_action": (
              f"Establishing shot: {clean_prompt}. Natural physical motion begins smoothly "
              "with steady cinematic framing, crisp textures, and authentic depth of field."
          ),
          "camera_movement": "Gentle natural handheld stabilization with subtle slow push-in.",
          "sound_effects": foley,
          "ambient_sound": ambient,
          "dialogue_lines": [],
      },
      {
          "time_range": f"{t1:.1f}s - {t2:.1f}s",
          "visual_action": (
              f"Continuing naturally from the opening moment, {subj_ref} engages expressively with the scene"
              + (
                  f", speaking naturally with clear lip synchronization: \"{dialogues[0]}\"."
                  if has_dialogue
                  else ", displaying fluid lifelike motion, natural eye gaze, and realistic lighting interaction."
              )
          ),
          "camera_movement": "Smooth subtle medium close-up focus keeping the subject sharp.",
          "sound_effects": foley,
          "ambient_sound": ambient,
          "dialogue_lines": seg2_dialogue,
      },
      {
          "time_range": f"{t2:.1f}s - {t_end:.1f}s",
          "visual_action": (
              f"Gentle resolution of the moment: {clean_prompt}. Natural settle in expression "
              "and environment with persistent fine details and natural lighting."
          ),
          "camera_movement": "Steady cinematic hold with subtle atmospheric motion.",
          "sound_effects": foley,
          "ambient_sound": ambient,
          "dialogue_lines": [],
      },
  ]

  pe: dict[str, Any] = {
      "static_layer": {
          "scene_setting": clean_prompt,
          "lighting_and_atmosphere": "Natural cinematic photorealistic lighting with authentic shadows, filmic grain, and rich color fidelity.",
          "visual_style": "Live-action 35mm cinema photography, natural skin and material textures, physically grounded motion.",
          "subjects": [
              {
                  "id": "subject 1",
                  "description": speaker_desc,
              }
          ],
      },
      "dynamic_layer": {
          "timeline_segments": timeline_segments,
      },
  }
  if is_i2v and num_refs > 0:
    pe["reference_layer"] = {
        "references": [
            {
                "id": f"Figure {i + 1}",
                "role": "Primary visual identity and scene anchor" if i == 0 else f"Reference subject {i + 1}",
            }
            for i in range(num_refs)
        ]
    }
  return pe


def _try_api_enhance(
    prompt: str,
    *,
    seconds: float = 10.0,
    is_i2v: bool = False,
    num_refs: int = 0,
    timeout_s: float = 15.0,
) -> dict[str, Any] | None:
  """Optional REST LLM PE expansion via GEMINI_API_KEY or OPENAI_API_KEY."""
  sys_inst = (
      "You are a cinematic Prompt Expander for MAGI-2 audio-video generation. "
      "Return ONLY valid JSON with keys: 'static_layer' and 'dynamic_layer'. "
      "'dynamic_layer' MUST contain 'timeline_segments' (a list of 3 dicts covering "
      f"0.0s to {seconds:.1f}s, each with 'time_range', 'visual_action', 'camera_movement', "
      "'sound_effects', 'ambient_sound', and 'dialogue_lines' list of "
      "{'speaker': 'subject 1', 'text': ..., 'emotion': ...}). "
      "Preserve any quoted dialogue verbatim in dialogue_lines."
  )
  user_msg = f"Expand into MAGI-2 PE JSON (is_i2v={is_i2v}, num_refs={num_refs}):\n{prompt}"

  gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
  if gemini_key:
    model = os.environ.get("MAGI2_ENHANCE_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{sys_inst}\n\n{user_msg}"}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }
    try:
      req = urllib.request.Request(
          url,
          data=json.dumps(payload).encode("utf-8"),
          headers={"Content-Type": "application/json"},
          method="POST",
      )
      with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
      txt = data["candidates"][0]["content"]["parts"][0]["text"]
      parsed = json.loads(txt)
      if isinstance(parsed, dict) and isinstance(
          parsed.get("dynamic_layer", {}).get("timeline_segments"), list
      ):
        return parsed
    except Exception:
      pass

  openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
  if openai_key:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("MAGI2_ENHANCE_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.4,
    }
    try:
      req = urllib.request.Request(
          f"{base}/chat/completions",
          data=json.dumps(payload).encode("utf-8"),
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Bearer {openai_key}",
          },
          method="POST",
      )
      with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
      txt = data["choices"][0]["message"]["content"]
      parsed = json.loads(txt)
      if isinstance(parsed, dict) and isinstance(
          parsed.get("dynamic_layer", {}).get("timeline_segments"), list
      ):
        return parsed
    except Exception:
      pass
  return None


def enhance_prompt_to_pe_json_str(
    prompt: str,
    *,
    mode: str = "builtin",
    seconds: float = 10.0,
    is_i2v: bool = False,
    num_refs: int = 0,
) -> str:
  """Expand a plain-text prompt into a validated MAGI-2 PE JSON string.

  If ``prompt`` is already valid PE JSON containing ``dynamic_layer.timeline_segments``,
  it is returned unchanged.
  """
  stripped = prompt.strip()
  if stripped.startswith("{"):
    try:
      obj = json.loads(stripped)
      if isinstance(obj, dict) and isinstance(
          obj.get("dynamic_layer", {}).get("timeline_segments"), list
      ):
        return stripped
    except Exception:
      pass

  pe_dict: dict[str, Any] | None = None
  if mode.lower() == "api":
    pe_dict = _try_api_enhance(
        stripped, seconds=seconds, is_i2v=is_i2v, num_refs=num_refs
    )
  if pe_dict is None:
    pe_dict = build_builtin_pe_json(
        stripped, seconds=seconds, is_i2v=is_i2v, num_refs=num_refs
    )
  return json.dumps(pe_dict, ensure_ascii=False)
