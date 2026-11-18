"""One DiT program for every denoising step: timestep slots by row role.

The pipelines reduce each row's timestep with `torch.unique(sorted=True)`.
On the first step the video and audio timesteps are equal, so there is one
distinct timestep (a `(1, 2688)` time embedding) and every row indexes slot
0; later steps have two. The adaLN indices therefore differ between steps,
segmented modulation bakes them into the graph as Python constants, and the
torch_tpu backend compiles -- and in a warm process traces -- two 1350-op
programs where one would do.

Here every role keeps a fixed slot (generated video and text 0, generated
audio 1, then the video and audio conditioning rows), duplicates included,
so the indices are the same on every step. Each row still gets its own
timestep, so the model computes the same thing; the time embedding on the
first step gains a duplicate row, which moves bfloat16 rounding about as much
as padding it alone does (DINO 0.977 vs 0.969 against stock, c673). Pika
480p/121 t2va: cold run 0 69 -> 43 s, warm 20 -> 14 s, cache 2.45 -> 1.41 GB
(c672); ref2va cold 114 -> 86 s, programs 15 -> 7 (c674). On by default;
DT_FIXED_TSLOTS=0 restores the stock plan.
"""

import torch


def build_row_timesteps(video_indices, audio_indices, num_condition_video_rows,
                        num_condition_audio_rows, num_text_tokens,
                        video_timestep, audio_timestep,
                        condition_video_timestep, condition_audio_timestep):
  video_indices = video_indices.cpu()
  audio_indices = audio_indices.cpu()
  length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
  slots = [video_timestep, audio_timestep]
  index = torch.zeros((length,), dtype=torch.long)
  index[audio_indices[num_condition_audio_rows:]] = 1
  if num_condition_video_rows:
    slots.append(condition_video_timestep)
    index[video_indices[:num_condition_video_rows]] = len(slots) - 1
  if num_condition_audio_rows:
    slots.append(condition_audio_timestep)
    index[audio_indices[:num_condition_audio_rows]] = len(slots) - 1
  return torch.tensor(slots, dtype=torch.float32), index


def build_row_timesteps_padded(*args):
  """Diagnostic (DT_FIXED_TSLOTS=pad): the stock sorted plan, padded to two slots by repeating the last
  timestep. Rows index exactly as before; only the time embedding's row count changes."""
  unique, index = _stock(*args)
  if unique.numel() == 1:
    unique = torch.cat([unique, unique])
  return unique, index


_stock = None


def install(verbose=print):
  patched = []
  for mod, cls in (("diffusers.modular_pipelines.pika_h3_flex.before_denoise",
                    "PikaH3FlexSetTimestepsStep"),
                   ("diffusers.modular_pipelines.minimax_h3.before_denoise",
                    "MiniMaxH3SetTimestepsStep")):
    try:
      klass = getattr(__import__(mod, fromlist=[cls]), cls)
    except (ImportError, AttributeError):
      continue
    global _stock
    import os
    if os.environ.get("DT_FIXED_TSLOTS") == "pad":
      _stock = klass.build_row_timesteps
      klass.build_row_timesteps = staticmethod(build_row_timesteps_padded)
    else:
      klass.build_row_timesteps = staticmethod(build_row_timesteps)
    patched.append(cls)
  verbose(f"  [tslots] fixed timestep slots by row role: {', '.join(patched)}",
          flush=True)
