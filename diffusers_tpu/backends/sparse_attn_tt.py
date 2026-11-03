"""Temporal-window block-sparse attention for torch_tpu (port of torchax `--sparse-window`).

Lossy. The first `dense_steps` DiT calls are dense; afterwards each video row
only attends to video rows within `window` latent frames (text and audio rows
still attend to everything). On torchax 768p/345 frames sparsity alone took
120.5 -> 90.3 s at DINO 0.9733 (see README "Which half of the recipe costs
the quality?").

Three pieces:

1. Kernel: `sparse_splash.attention` has the same interface as the dense
   `kernels/splash.attention` plus a `kv_table` (kv blocks each q block
   visits). It is registered via `jax_op` as in `ours_splash_tt`.
2. When to go sparse: DiT calls are counted (including skipped-step calls,
   matching torchax's `calls_this_request`); from call `dense_steps` on we
   switch to sparse. Video start and rows per frame are derived once per
   request from `video_indices` / `position_ids`.
3. Compile cache: sparse and dense programs have identical input shapes, so
   the `AotBlocks` cache key includes the sparse flag. Ops must be
   registered before tracing (see `ours_splash_tt`), hence
   `AotBlocks.PRE_TRACE`.
"""

from __future__ import annotations

import functools
import os

# Must be a module-level import: with `from __future__ import annotations`,
# `jax_op` resolves the "jax.Array" annotation string in module globals.
import jax

STATE: dict = {"active": False, "layout": None, "op": None, "op_key": None,
               "seq": 0, "calls": 0}
_REGISTERED: dict = {}


def _table(seq, layout, bq, bkv):
  """Kv blocks each q block attends to, in ascending order.

  Copy of `Sparsity.table` in `models/minimax_h3/h3_tpu.py`; not imported
  because that module imports torchax, which the torch_tpu env lacks.
  """
  video_start, rows_per_frame, window = layout
  block_q, block_kv = min(bq, seq), min(bkv, seq)
  reach = window * rows_per_frame
  num_q, num_kv = -(-seq // block_q), -(-seq // block_kv)
  rows = []
  for i in range(num_q):
    q0, q1 = i * block_q, min((i + 1) * block_q, seq)
    rows.append([
        j for j in range(num_kv)
        if q0 < video_start or j * block_kv < video_start
        or (j * block_kv < q1 + reach
            and min((j + 1) * block_kv, seq) > q0 - reach)
    ])
  return rows


def register(seq: int, bq: int, bkv: int, fixed_m: bool):
  """Register a sparse op for the current layout (call before compiling)."""
  import torch
  from diffusers_tpu.kernels import sparse_splash

  qk_fp8 = os.environ.get("DT_OURS_QK_FP8") == "1"
  layout = STATE["layout"]
  key = (seq, layout, bq, bkv, fixed_m, qk_fp8)
  if key in _REGISTERED:
    STATE["op"], STATE["op_key"], STATE["seq"] = _REGISTERED[key], key, seq
    return
  table = _table(seq, layout, bq, bkv)
  kept = sum(len(r) for r in table)
  full = len(table) * (-(-seq // min(bkv, seq)))
  if kept == full:
    # No block skipped means dense attention; use the dense kernel. This
    # happens for ref2va, where the window covers the whole sequence (same
    # as torchax). Forcing the sparse kernel here fails an assert at trace.
    _REGISTERED[key] = None
    STATE["op"], STATE["op_key"], STATE["seq"] = None, key, seq
    print(f"  [sparse] layout {layout} keeps {kept}/{full} kv blocks, "
          f"same as dense; not switching to the sparse kernel", flush=True)
    return

  # Not functools.partial: `jax_op` infers the schema from the signature and
  # requires every parameter (even bound ones) to be annotated; this kernel's
  # `kv_table` is not.
  def fn(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
    return sparse_splash.attention(
        query, key, value, scale=None, block_q=bq, block_kv=bkv,
        block_kv_compute=bkv, use_fixed_m=fixed_m, qk_fp8=qk_fp8,
        kv_table=table)
  vs, rpf, win = layout
  name = (f"pallas::h3_sparse_q{bq}_kv{bkv}_s{seq}_v{vs}_r{rpf}_w{win}"
          + ("_fm" if fixed_m else "")
          + ("_fp8" if qk_fp8 else ""))
  op = torch.tpu.pallas.jax_op(name, fn)
  _REGISTERED[key] = op
  STATE["op"], STATE["op_key"], STATE["seq"] = op, key, seq
  print(f"  [sparse] registered {name}: keeps {kept}/{full} kv blocks "
        f"({100 * kept / full:.0f}%)", flush=True)


def install(pipe, window: int, dense_steps: int, bq: int, bkv: int,
            fixed_m: bool, verbose=print):
  """Install the call-counting wrapper and pre-trace hook; returns a per-request reset."""
  import torch
  from diffusers_tpu.backends.aot_dit import AotBlocks

  # ref2va's DiT is `transformer_ref`. `_dit_of` is not imported from
  # torchtpu_h3 because that module runs as __main__ and would re-execute.
  try:
    tr = pipe.transformer
  except AttributeError:
    tr = pipe.transformer_ref
  inner = tr.forward   # keep existing _inputs_to_device / step-skip wrappers

  @functools.wraps(inner)
  def forward(*a, **k):
    if STATE["layout"] is None:
      video = k["video_indices"]
      pos = k["position_ids"]
      vi = video.to("cpu")
      audio = k.get("audio_indices")
      if audio is not None and audio.numel() > 0:
        target_vi = vi[vi > int(audio.to("cpu").max())]
        if target_vi.numel() > 0:
          vi = target_vi
      times = pos.to("cpu")[vi, 0]
      STATE["layout"] = (int(vi.min()), int((times == times[0]).sum()),
                         window)
      verbose(f"  [sparse] layout {STATE['layout']} (video start, rows per frame, window)",
              flush=True)
    STATE["active"] = STATE["calls"] >= dense_steps
    STATE["calls"] += 1
    # Flip the global the dispatch function reads. Registration happens in
    # PRE_TRACE using the block's real input length, which can differ from
    # the `token_tags` length (it does for ref2va).
    from diffusers_tpu.backends import ours_splash_tt as _ot
    _ot.SPARSE_OP = STATE["op"] if STATE["active"] else None
    _ot.SPARSE_SEQ = STATE["seq"] if STATE["active"] else 0
    return inner(*a, **k)

  tr.forward = forward

  def pre_trace(args, kwargs):
    if not STATE["active"]:
      return
    hs = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(hs, torch.Tensor) or hs.dim() < 2:
      return       # not a DiT block (e.g. VAE); ignore
    register(hs.shape[1], bq, bkv, fixed_m)
    from diffusers_tpu.backends import ours_splash_tt as _ot
    _ot.SPARSE_OP = STATE["op"]
    _ot.SPARSE_SEQ = STATE["seq"]

  AotBlocks.PRE_TRACE.append(pre_trace)
  # Key on layout, not op_key: op_key is only set in PRE_TRACE inside
  # `_build`, so using it would change the key after the first compile and
  # trigger a needless recompile.
  AotBlocks.KEY_EXTRA.append(lambda: ("sparse", STATE["layout"])
                             if STATE["active"] else ("dense",))

  def reset():
    STATE.update(active=False, calls=0)

  verbose(f"  [sparse] installed: first {dense_steps} calls dense, then window {window}",
          flush=True)
  return reset
