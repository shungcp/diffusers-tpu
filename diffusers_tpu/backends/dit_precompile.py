"""Compile a request's DiT program while its text encoder runs.

A new prompt length changes the DiT sequence length, so step 0 compiles a new
program (~15-20 s). The length is known as soon as the prompt is tokenized,
seconds before the text encoder finishes. When the text encoder starts, this
predicts the new sequence length from the last DiT call (same layout, text
rows changed by the token-count difference), and builds that program in a
background thread from zero tensors of the predicted shapes. Step 0 waits for
it instead of compiling from scratch. A wrong prediction only wastes the
background compile: step 0 then misses and compiles as before.

Only the shapes matter to the program, never the values, so zero inputs are
fine -- except with segment-wise modulation, which bakes runs computed from
`adaln_indices` values into the program; with segmod installed this stays off.
c734: the build must enter the caller's torch-function modes. c733: a background compile overlapping a request slowed it by ~3 s and was not
slower itself; outputs bit-identical.  `DT_DIT_PRECOMPILE=1` enables.
"""
import contextlib
import copy
import os
import threading
import time

import torch

_S = {"meta": None, "pending": {}, "text_len": None, "meta_text_len": None}


def _say(msg):
  if os.environ.get("RANK", "0") == "0":
    print(msg, flush=True)


def _meta(x):
  if isinstance(x, torch.Tensor):
    return ("T", tuple(x.shape), x.dtype, x.device)
  if isinstance(x, (tuple, list)):
    return (type(x), [_meta(v) for v in x])
  if isinstance(x, dict):
    return (dict, {k: _meta(v) for k, v in x.items()})
  return ("V", x)


def _make(m, n_old, n_new, device_ok):
  tag = m[0]
  if tag == "T":
    _, shape, dtype, dev = m
    shape = tuple(n_new if s == n_old else s for s in shape)
    return torch.zeros(shape, dtype=dtype, device=dev if device_ok else "meta")
  if tag == "V":
    return m[1]
  if tag is dict:
    return {k: _make(v, n_old, n_new, device_ok) for k, v in m[1].items()}
  return tag(_make(v, n_old, n_new, device_ok) for v in m[1])


def install(aot, text_model, transformer=None) -> bool:
  if os.environ.get("DT_DIT_PRECOMPILE") != "1":
    return False
  if type(aot).PRE_TRACE:
    _say("  [precompile] off: segment-wise modulation bakes value-derived runs "
         "into the program")
    return False
  orig_run = aot._run

  def run(block, args, kwargs, group=False):
    if group:
      key = (aot._shape_key(args, kwargs) + ("g",)
             + tuple(f() for f in aot.KEY_EXTRA))
      ev = _S["pending"].get(key)
      if ev is not None:
        t0 = time.perf_counter()
        ev.wait()
        _S["pending"].pop(key, None)
        _say(f"  [precompile] step waited {time.perf_counter() - t0:.1f}s "
             f"for the background program")
      if _S["text_len"] is not None:
        _S["meta"] = (block, _meta(args), _meta(kwargs), args[0].shape[1])
        _S["meta_text_len"] = _S["text_len"]
    return orig_run(block, args, kwargs, group=group)
  aot._run = run

  def build(key, members, margs, mkwargs, n_old, n_new, modes):
    go = _S.get("go")
    if go is not None:
      # `DT_DIT_PRECOMPILE_START=prefetch`: start once the text encoder begins
      # streaming its layers. The build's first ~7 s are Dynamo/AOTAutograd
      # (pure Python) and slow the encoder's Python dispatch through the GIL
      # (c741/c742); the layer copies wait in C++, a better time to overlap.
      go.wait(timeout=60)
    t0 = time.perf_counter()
    try:
      # Enter the caller's torch-function modes (e.g. `with torch.device`):
      # the program's guards check the mode stack, which is per thread, and a
      # program built without them fails its guards on the main thread (c734).
      es = contextlib.ExitStack()
      from torch.utils._device import DeviceContext
      for m in modes:
        es.enter_context(torch.device(m.device) if isinstance(m, DeviceContext) else m)
      a = _make(margs, n_old, n_new, True)
      k = _make(mkwargs, n_old, n_new, True)
      hit = aot._build(tuple(copy.copy(b) for b in members), a, k,
                       persist_path=aot._persist_path(key))
      # Not done: warming the token refiner's Pallas kernel here too. Step 0
      # fell 10 -> 5.3 s but the text stage rose 16.8 -> 25.5 s: that compile
      # stalls the main thread (c739), a net loss.
      es.close()
      aot._cache[key] = hit
      _say(f"  [precompile] built rows {n_old}->{n_new} in background in "
           f"{time.perf_counter() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
      _say(f"  [precompile] background build failed: {type(exc).__name__}: "
           f"{' '.join(str(exc).split())[:200]}")
    finally:
      _S["pending"][key].set()

  def pre_hook(module, args, kwargs):
    ids = kwargs.get("input_ids", args[0] if args else None)
    if not isinstance(ids, torch.Tensor):
      return
    n_text = int(ids.shape[-1])
    _S["text_len"] = n_text
    meta = _S["meta"]
    if meta is None or n_text == _S["meta_text_len"]:
      return
    members, margs, mkwargs, n_old = meta
    n_new = n_old + n_text - _S["meta_text_len"]
    key = (aot._shape_key(_make(margs, n_old, n_new, False),
                          _make(mkwargs, n_old, n_new, False))
           + ("g",) + tuple(f() for f in aot.KEY_EXTRA))
    if key in aot._cache or key in _S["pending"]:
      return
    _S["pending"][key] = threading.Event()
    _say(f"  [precompile] text {_S['meta_text_len']}->{n_text} tokens: "
         f"building DiT rows {n_old}->{n_new} in background")
    if os.environ.get("DT_DIT_PRECOMPILE_START") == "prefetch":
      _S["go"] = threading.Event()
    modes = torch.overrides._get_current_function_mode_stack()
    threading.Thread(target=build, daemon=True,
                     args=(key, members, margs, mkwargs, n_old, n_new, modes)).start()

  text_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
  if os.environ.get("DT_DIT_PRECOMPILE_START") == "prefetch":
    from diffusers_tpu.backends.layerwise_offload import LayerwiseOffloadManager
    orig_pf = LayerwiseOffloadManager.prefetch_layer

    def prefetch_layer(self, *a, **k):
      go = _S.pop("go", None)
      if go is not None:
        go.set()
      return orig_pf(self, *a, **k)
    LayerwiseOffloadManager.prefetch_layer = prefetch_layer
  _say("  [precompile] DiT program for a new text length is built while the "
       "text encoder runs")
  return True
