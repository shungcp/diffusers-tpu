"""AOT-compile the DiT's blocks on torch_tpu, one artifact for all fifty.

The key point: pass the module as an argument, do not close over it.

    def flat_forward(module, *args):
        return module(*args)

    with torch.no_grad():
        compiled = torch.compile(flat_forward, backend="tpu",
                                 fullgraph=True, dynamic=False)
        art = compiled.aot_compile(((block_0, *args), {}))
    # and then, for every one of the fifty blocks:
    compiled(block_i, *args)

Dynamo lifts each module's parameters into the graph's arguments, so:

* One artifact serves all fifty blocks (bit-exact across blocks).
* It composes with host-side weight streaming: weights are arguments,
  not constants baked into the program, so a block can be moved to the
  chip right before its call. Compiling the whole transformer as one
  graph would instead need all weights resident (32.6 GiB per rank
  against a 31.25 GiB chip).

Parameters must have `requires_grad=False` (`.eval()` does not clear it):
otherwise aot_autograd takes the autograd path, torch_tpu never attaches
`.serialize`, and PyTorch rejects the result with "does not implement
SerializableCallable".
"""

from __future__ import annotations

import os
import time

import torch


def _flat_forward(module, *args, **kwargs):
  """The module is the first argument so its parameters get lifted.

  Calls `type(module).forward(module, ...)` rather than `module(...)`:

  * `module(...)` runs forward hooks via `_call_impl`; the offload
    manager's hooks are plain Python (e.g. `time.perf_counter`) that
    Dynamo cannot trace.
  * `install()` replaces each block's `forward`, so `module(...)` would
    recurse into this function.

  Hooks thus stay in Python outside the graph; only the block's
  arithmetic is compiled.
  """
  # Read the class's forward off the instance (stored by `AotBlocks`): Dynamo
  # refuses `type(module).forward` for classes diffusers marks allow-in-graph.
  fwd = module.__dict__.get("_aot_fwd") or type(module).forward
  return fwd(module, *args, **kwargs)


def _flat_group(modules, hidden_states, *rest, **kwargs):
  """Chain a group of blocks into a single graph.

  Uses `type(m).forward` like `_flat_forward`, so hooks stay in Python.
  Saves a program launch per layer and keeps intermediates on chip.
  """
  for m in modules:
    fwd = m.__dict__.get("_aot_fwd") or type(m).forward
    hidden_states = fwd(m, hidden_states, *rest, **kwargs)
  return hidden_states


class AotBlocks:
  """Compiles one transformer block, then runs all of them through it.

  Installed by replacing each block's `forward`. The offload manager's
  hooks stay where they are: they stage a block's weights onto the
  chip before its forward runs, and this calls the compiled artifact
  with that block as an argument, so the weights arrive the same way
  they always did.
  """

  def __init__(self, blocks, *, dynamic: bool = False, group: int = 1,
               verbose=print):
    self._blocks = list(blocks)
    self._dynamic = dynamic
    #: Number of blocks compiled into one graph.
    #:
    #: With one block per program, ~16% of the denoise window is idle
    #: program-launch overhead (~25 ms per launch, 50 layers x 19 steps),
    #: and each block's trailing all-reduce is fully exposed. Grouping
    #: reduces both launches and sync points.
    self._group = max(1, int(group))
    self._manager = None
    self._say = verbose
    self._art = None
    self._compiled = None
    self._key = None
    self._cache: dict = {}
    self._originals = [b.forward for b in self._blocks]
    for b in self._blocks:
      b.__dict__["_aot_fwd"] = type(b).forward

  @staticmethod
  def _shape_key(args, kwargs):
    # Recurse into tuples and lists: H3's block takes `rotary_emb` as a
    # tuple of two tensors, which must not end up in the key as tensors.
    def one(a):
      if isinstance(a, torch.Tensor):
        return ("t", tuple(a.shape), str(a.dtype))
      if isinstance(a, (tuple, list)):
        return ("seq", tuple(one(x) for x in a))
      if isinstance(a, dict):
        return ("map", tuple((k, one(v)) for k, v in sorted(a.items())))
      return ("v", repr(a))
    return (tuple(one(a) for a in args),
            tuple((k, one(v)) for k, v in sorted(kwargs.items())))

  #: Callbacks `(args, kwargs)` run before each compile.
  #
  #: Some rewrites need runtime values as Python constants (e.g. segmented
  #: modulation derives its runs from `adaln_indices`), which cannot be
  #: computed during tracing.
  PRE_TRACE: list = []

  #: Extra cache-key parts, each a zero-arg callable. Needed when inputs of
  #: the same shape map to different programs (e.g. dense vs sparse
  #: attention steps).
  KEY_EXTRA: list = []

  def _build(self, block, args, kwargs):
    for fn in self.PRE_TRACE:
      fn(args, kwargs)
    begin = time.perf_counter()
    # Put the block's original `forward` back for the duration of the
    # compile. `install()` leaves a closure in the instance `__dict__`,
    # and `aot_compile` serialises its guards (`save=True`, hard-coded
    # in `aot_compile_fullgraph`) -- which walks the module, reaches
    # that closure, reaches the `AotBlocks` it captures, reaches the
    # compiled artifact already stored on it, and dies with "cannot
    # pickle 'torch._C._dynamo.guards.RootGuardManager' object".
    # Compiling against a pristine module avoids the cycle entirely.
    # When grouping, `block` is a tuple of modules; restore each one.
    members = block if isinstance(block, tuple) else (block,)
    installed = [(m, m.__dict__.pop("forward", None)) for m in members]
    try:
      with torch.no_grad():
        fn = _flat_group if isinstance(block, tuple) else _flat_forward
        # `DT_SPMD_SAFE=1`: compile the block's all-reduces into the same
        # graph as its compute. By default torch_tpu's `SplitCompiler`
        # splits the graph at every collective (to avoid deadlocks if ranks
        # fuse differently), leaving each all-reduce exposed (~11 ms each).
        # `spmd_safe` declares that every rank runs the same op sequence,
        # which holds for DiT blocks (no rank-dependent branches).
        if os.environ.get("DT_SPMD_SAFE") == "1":
          spmd_safe_fn = getattr(torch.tpu.distributed, "spmd_safe", None)
          if spmd_safe_fn is None:
            from torch_tpu._internal.distributed.spmd_util import (
                spmd_safe as spmd_safe_fn,
            )
          fn = spmd_safe_fn(fn)
          self._say("  [aot] spmd_safe: graph no longer split at all-reduce", flush=True)
        compiled = torch.compile(fn, backend="tpu",
                                 fullgraph=True, dynamic=self._dynamic)
        art = compiled.aot_compile(((block, *args), kwargs))
    finally:
      for m, fn in installed:
        if fn is not None:
          m.__dict__["forward"] = fn
    # Log the shape key so distinct compiles can be told apart.
    self._say(f"  [aot] compiled one block in {time.perf_counter() - begin:.1f}s"
              f", artifact {type(art).__name__}, serializable="
              f"{hasattr(art, 'serialize')}, reused across {len(self._blocks)} layers",
              flush=True)
    self._say(f"  [aot] key={self._shape_key(args, kwargs)}", flush=True)
    return compiled, art

  def _run(self, block, args, kwargs, group: bool = False):
    key = (self._shape_key(args, kwargs) + (("g",) if group else ())
           + tuple(f() for f in self.KEY_EXTRA))
    # Cache one program per shape: the time embedding is (1, 2688) on step 1
    # and (2, 2688) afterwards, so a single-slot cache would recompile twice
    # per request (~26 s each when grouping 10 blocks).
    hit = self._cache.get(key)
    if hit is None:
      if self._cache:
        self._say(f"  [aot] new shape, compiling program #{len(self._cache) + 1}",
                  flush=True)
      hit = self._build(block, args, kwargs)
      self._cache[key] = hit
    self._compiled, self._art = hit
    self._key = key
    # `DT_AOT_CALL_ART=1`: call this artifact directly, bypassing the
    # `torch.compile` wrapper. The wrapper picks a program from Dynamo's
    # frame cache, which is shared by all instances of the same function and
    # ignores KEY_EXTRA, so it can pick the dense program for a sparse step.
    # The artifact checks its own guards and raises on mismatch.
    if os.environ.get("DT_AOT_CALL_ART") == "1" and self._art is not None:
      return self._art(block, *args, **kwargs)
    return self._compiled(block, *args, **kwargs)

  def set_manager(self, manager):
    """Set the offload manager. When grouping, we stage weights ourselves:
    non-leading blocks in a group never go through `_call_impl`, so their
    offload hooks do not fire."""
    self._manager = manager

  def _stage(self, idx_list, up: bool):
    m = self._manager
    if m is None:
      return
    for i in idx_list:
      (m.prefetch_layer if up else m.release_layer)(i)

  def _run_group(self, members, start, args, kwargs):
    idx = list(range(start, start + len(members)))
    self._stage(idx, True)
    try:
      return self._run(tuple(members), args, kwargs, group=True)
    finally:
      self._stage(idx, False)

  def install(self):
    if self._group > 1:
      n = self._group
      for start in range(0, len(self._blocks), n):
        members = self._blocks[start:start + n]
        head = members[0]

        def forward(*args, _m=members, _s=start, **kwargs):
          return self._run_group(_m, _s, args, kwargs)
        head.forward = forward
        for rest in members[1:]:
          # Other blocks in the group become identity; the leader computed them.
          def passthrough(*args, **kwargs):
            return args[0]
          rest.forward = passthrough
      for block in self._blocks:
        block.requires_grad_(False)
      self._say(f"  [aot] {n} blocks per graph, "
                f"{len(self._blocks)} layers -> "
                f"{-(-len(self._blocks) // n)} launches/step", flush=True)
      return self

    for block in self._blocks:
      def forward(*args, _b=block, **kwargs):
        return self._run(_b, args, kwargs)
      block.forward = forward
    for block in self._blocks:
      block.requires_grad_(False)
    return self

  def release(self):
    """Drop compiled artifacts so the next request recompiles.

    Works around program-allocation OOMs (`error_0101`) on a second 768p
    request. Dropping our references is not enough: `TpuBackend` keeps its
    own `_compiled_executables`, cleared by
    `torch_tpu._internal.compile.clear_cache()`. Recompiling one block takes
    ~7 s, negligible against a ~350 s request.
    """
    had = self._compiled is not None
    self._compiled = None
    self._art = None
    self._key = None
    self._cache.clear()
    try:
      from torch_tpu._internal import compile as tt_compile

      tt_compile.clear_cache()
      self._say("  [aot] also cleared TpuBackend's cached executables", flush=True)
    except Exception as exc:  # noqa: BLE001
      self._say(f"  [aot] clear_cache unavailable: {type(exc).__name__}",
                flush=True)
    return had

  def uninstall(self):
    for block, original in zip(self._blocks, self._originals):
      block.forward = original
