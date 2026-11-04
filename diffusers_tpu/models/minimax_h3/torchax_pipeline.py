#!/usr/bin/env python3
"""End-to-end MiniMax-H3 with the transformer compiled by torchax on the mesh.

Runs the real diffusers pipeline -- prompt in, video and audio out -- with
`pipe.transformer` replaced by a torchax-compiled, tensor-parallel stand-in
that keeps the model's own signature. The text encoder and VAE can
optionally be moved to the TPU as well; otherwise they run on the host.

  python -m diffusers_tpu.models.minimax_h3.torchax_pipeline \
      --num-frames 49 --steps 4 --out ./out
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import time

os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_enable_async_collective_permute=true "
    "--xla_tpu_scoped_vmem_limit_kib=65536",
)

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import torch

REPO = "MiniMaxAI/MiniMax-H3"
KV_BLOCK = 2048
MODALITY_NUM = 3
QK_BARRIER = os.environ.get("DT_QK_BARRIER", "1") == "1"
# Optimization barrier after the q/k norm and rope (see `install_attention`);
# DT_QK_BARRIER=0 disables it for ablation.
# Distinct sigma schedules in the packed sequence: video and audio.
MODALITY_COUNT = 2
SHARDINGS = (
    (".to_q.weight", P("tp", None)),
    (".to_k.weight", P("tp", None)),
    (".to_v.weight", P("tp", None)),
    (".to_q.weight_scale", P("tp")),
    (".to_k.weight_scale", P("tp")),
    (".to_v.weight_scale", P("tp")),
    (".to_out.0.weight", P(None, "tp")),
    (".ff.net.0.proj.weight", P("tp", None)),
    (".ff.net.0.proj.weight_scale", P("tp")),
    (".ff.net.2.weight", P(None, "tp")),
    (".adaln_proj.linear.weight", P("tp", None)),
    (".adaln_proj.linear.bias", P("tp")),
)


def spec_for(name: str) -> P:
  for suffix, spec in SHARDINGS:
    if name.endswith(suffix):
      return spec
  return P()


class TorchaxTransformer(torch.nn.Module):
  """`pipe.transformer`, compiled and sharded, behind its own signature.

  The pipeline keeps calling `transformer(hidden_states=..., ...)` and
  getting a tuple back. Inside, the arguments cross into jax, the compiled
  program runs on the mesh, and the results cross back. The program is
  cached on the shapes it was traced for, so a loop of identical steps
  compiles once.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      mesh: jax.sharding.Mesh,
      offload: bool = False,
  ):
    super().__init__()
    import torchax

    self._model = model
    self._mesh = mesh
    self._env = torchax.default_env()
    self._cache: dict = {}
    self.calls: list[float] = []
    self.pad_timestep = False
    self.segmented = True
    self._runs = None
    # TeaCache-style step skipping. The indicator is the block modulation,
    # which depends only on `temb` -- two rows -- so the decision costs
    # nothing and stays on the host, out of the compiled program.
    self.cache_threshold = 0.0
    self.report_memory = False
    self.scoped_vmem_kib = 0
    self.profile_dir = None
    self.sequence_parallel = False
    self.reuse_steps: frozenset[int] = frozenset()
    self.pad_sequence_to: int = 0
    self.profile_call = 8
    self.sparse_window = 0
    self.sparse_dense_steps = 0
    self.calls_this_request = 0
    self._last_modulation = None
    self._last_output = None
    self._drift = 0.0
    self.skipped = 0
    # On one chip the 66 GB of weights do not fit in 32 GB of HBM, so they
    # sit in `pinned_host` and the jitted step copies each one in. The
    # scheduler overlaps a layer's copy with the previous layer's compute;
    # how much it hides depends on the compute-to-transfer ratio, which at
    # these sequence lengths is comfortably above one.
    self.offload = offload
    self._native = None
    # The host copy uses the same mesh partitioning as the device copy, so
    # each chip only uploads its own shard. (Pinning everything to
    # `devices()[0]` would leave too little HBM for activations on small
    # meshes.)
    self._device = {
        name: NamedSharding(mesh, spec_for(name))
        for name, _ in list(model.named_parameters())
        + list(model.named_buffers())
    }
    self._params = {
        name: jax.device_put(
            _to_jax(tensor),
            NamedSharding(mesh, spec_for(name), memory_kind="pinned_host")
            if offload
            else self._device[name],
        )
        for name, tensor in list(model.named_parameters())
        + list(model.named_buffers())
    }

  def use_native_dit(self) -> None:
    """Run the fifty blocks as the native JAX module, same weights.

    The native path is about 30% faster per call on eight chips (same
    attention, far fewer collective-permutes). The VAE and Qwen3-VL stay
    torch; only the transformer changes hands.

    The weights are not read again: the nnx module is filled from the jax
    arrays already in `self._params`, keyed by the diffusers module path.
    """
    import jax.numpy as jnp
    from flax import nnx

    from . import jax_dit as dit

    _, Transformer = dit._build_modules()
    model = nnx.eval_shape(
        lambda: Transformer(**dit.CONFIG_FULL, dtype=jnp.bfloat16,
                            rngs=nnx.Rngs(0)))
    # With `--offload` the native weights stay in pinned host memory too,
    # wherever the traced ones went; on device they would not fit.
    host = (jax.sharding.SingleDeviceSharding(
        jax.devices()[0], memory_kind="pinned_host") if self.offload else None)
    dit.load_transformer_from(model, self._params, mesh=self._mesh,
                              place=host)
    # `nnx.split` makes the weights a jit argument; closing over the
    # module would bake ~60 GB of constants into the HLO.
    # An explicit splash `sdpa` is required: the module's dense fallback
    # materializes a score matrix of tens of GB at these sequence lengths.
    import functools

    from ...kernels import mesh_attention
    from ...kernels import splash as csa

    local = functools.partial(csa.attention, scale=None, block_q=2048,
                              block_kv=2048, block_kv_compute=2048,
                              use_fixed_m=True)

    def sdpa(q, k, v):
      k = k - jnp.mean(k, axis=2, keepdims=True)
      return mesh_attention._on_mesh(local, self._mesh, q, k, v, True)

    self._native_sdpa = sdpa
    self._native = nnx.split(model)
    print(f"  [native] native JAX DiT loaded, {len(self._params)} tensors,"
          " weights passed as jit arguments", flush=True)

  def __getattr__(self, name: str):
    # Only the wrapped module itself comes from `nn.Module`; everything
    # else -- `config`, `dtype`, `transformer_blocks` -- is the model's,
    # and the pipeline reads several of them through properties that
    # would otherwise fail silently into "no such attribute".
    if name == "_model":
      return super().__getattr__(name)
    return getattr(self._model, name)

  # The signature is spelled out rather than `**kwargs`: the denoise
  # block builds its layout arguments by intersecting what it has with
  # `inspect.signature(transformer.forward).parameters`, so a wrapper
  # that takes only `**kwargs` is handed none of them.
  def forward(
      self,
      hidden_states: torch.Tensor,
      audio_hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      timestep: torch.Tensor,
      timestep_indices: torch.Tensor,
      token_tags: torch.Tensor,
      position_ids: torch.Tensor,
      video_indices: torch.Tensor,
      audio_indices: torch.Tensor,
      text_indices: torch.Tensor,
      attention_kwargs: dict | None = None,
      return_dict: bool = True,
  ):
    _raw_timestep = timestep
    if self.pad_timestep:
      has_ref_video = (
          video_indices.numel() > 0
          and audio_indices.numel() > 0
          and int(video_indices.min()) < int(audio_indices.min())
      )
      target_mods = 3 if has_ref_video else MODALITY_COUNT
      if timestep.shape[0] < target_mods:
        if has_ref_video:
          timestep = torch.stack([timestep[0], timestep[0], timestep[-1]])
          timestep_indices = torch.zeros_like(timestep_indices)
          timestep_indices[audio_indices] = 1
          timestep_indices[video_indices[video_indices < audio_indices.min()]] = 2
        else:
          timestep = torch.cat(
              [timestep, timestep[-1:].expand(MODALITY_COUNT - timestep.shape[0])]
          )
          timestep_indices = torch.zeros_like(timestep_indices)
          timestep_indices[audio_indices] = 1

    # Round the packed row axis up before upstream builds the buffer
    # from `position_ids.shape[0]`. The padded rows copy the last row's
    # coordinates, tag and timestep, so they extend the final run rather
    # than making a new one, and `segment_runs` is unchanged apart from
    # that run's end. Nothing reads them: the outputs are gathered by
    # `video_indices` and `audio_indices`, which stop at the true length.
    global TRUE_SEQ_LEN
    TRUE_SEQ_LEN = 0
    if self.pad_sequence_to:
      true_len = position_ids.shape[0]
      rounded = -(-true_len // self.pad_sequence_to) * self.pad_sequence_to
      if rounded != true_len:
        extra = rounded - true_len
        tail = lambda t: torch.cat([t, t[-1:].expand(
            extra, *t.shape[1:])])
        position_ids = tail(position_ids)
        token_tags = tail(token_tags)
        timestep_indices = tail(timestep_indices)
        TRUE_SEQ_LEN = true_len

    kwargs = {
        "hidden_states": hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "attention_kwargs": attention_kwargs,
    }
    if return_dict:
      raise ValueError("this stand-in only returns tuples")
    # Only the tensors cross into jax; the rest (`attention_kwargs`, the
    # optional arguments the t2va path leaves unset) are closed over, so
    # they take part in the cache key rather than the trace.
    if self.segmented:
      runs = segment_runs(token_tags, timestep_indices)
      if runs != self._runs:
        if self.sequence_parallel:
          install_sequence_parallel(
              self._model,
              self._mesh,
              runs,
              reassoc=getattr(self, "reassoc", False),
          )
        else:
          install_segmented_modulation(
              self._model, runs, reassoc=getattr(self, "reassoc", False)
          )
        self._runs = runs
        print(f"  [segments] {runs}"
              + ("  [sequence parallel]" if self.sequence_parallel else ""))

    names = sorted(n for n, v in kwargs.items() if torch.is_tensor(v))
    rest = {n: v for n, v in kwargs.items() if not torch.is_tensor(v)}
    # The runs are baked into the traced program, so two layouts with the
    # same shapes but different runs are different programs.
    # Dense for the first `sparse_dense_steps` calls of a request: the
    # windowed approximation costs the most quality early in the trajectory.
    layout = None
    sparse_threshold = (
        1
        if getattr(self, "_in_warmup", False) and self.pad_timestep
        else self.sparse_dense_steps
    )
    if self.sparse_window and self.calls_this_request >= sparse_threshold:
      video = kwargs["video_indices"]
      audio = kwargs.get("audio_indices")
      if audio is not None and audio.numel() > 0:
        target_video = video[video > audio.max()]
        if target_video.numel() > 0:
          video = target_video
      positions = kwargs["position_ids"]
      times = positions[video, 0]
      layout = (
          int(video.min()),
          int((times == times[0]).sum()),
          self.sparse_window,
      )
    if layout is not None and _SPARSE.get("announced") != layout:
      _SPARSE["announced"] = layout
      print(f"  [sparse] active layout={layout}", flush=True)
    _SPARSE["layout"] = layout
    # The table is baked into the traced program, so sparse and dense are
    # different programs and the layout belongs in the key.
    key = (
        tuple((n, tuple(kwargs[n].shape), kwargs[n].dtype) for n in names),
        tuple(sorted((k, repr(v)) for k, v in rest.items())),
        self._runs,
        layout,
    )
    fn = self._cache.get(key)
    if fn is None:
      native_graph = self._native[0] if self._native is not None else None
      native_sdpa = getattr(self, "_native_sdpa", None)
      native_device = jax.sharding.SingleDeviceSharding(
          jax.devices()[0], memory_kind="device")

      def call(params, inputs):
        if self.offload and self._native is None:
          # The native path feeds its own nnx state under one key and
          # has already been placed; looking that key up in the traced
          # layout is a `KeyError: '__native__'`.
          params = {
              k: jax.device_put(v, self._device[k]) for k, v in params.items()
          }
        if self._native is not None:
          state = params["__native__"]
          if self.offload:
            # jax 0.11 carries the memory space in the type system, so
            # a `pinned_host` operand makes `dot_general` raise rather
            # than transfer. The copy has to be asked for.
            state = jax.tree.map(
                lambda v: jax.device_put(v, native_device), state)
          return _native_call(native_graph, state, inputs, native_sdpa)
        with self._env:
          out = torch.func.functional_call(
              self._model,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              kwargs={
                  **{k: self._env.j2t_iso(v) for k, v in inputs.items()},
                  **rest,
                  "return_dict": False,
              },
          )
        return tuple(self._env.t2j_iso(t) for t in out)

      # Per program, not per process. `xla_tpu_scoped_vmem_limit_kib`
      # (default 16 MiB of v6e's 128) governs the non-Pallas ops around
      # the kernel. Raising it globally via LIBTPU_INIT_ARGS sped up the
      # transformer ~2% but slowed the VAE, so it is applied here only.
      options = (
          {"xla_tpu_scoped_vmem_limit_kib": str(self.scoped_vmem_kib)}
          if self.scoped_vmem_kib
          else None
      )
      fn = jax.jit(call, compiler_options=options)
      self._cache[key] = fn
      print(
          "  [compile] "
          + " ".join(f"{n}{tuple(kwargs[n].shape)}" for n in names)
      )
      if self.report_memory:
        # Whether a block configuration costs more scratch is a property
        # of the compiled program, not of a timed run, so ask the
        # compiler rather than inferring it from wall clock.
        probe = {n: _to_jax(kwargs[n]) for n in names}
        with jax.set_mesh(self._mesh):
          analysis = fn.lower(self._params, probe).compile().memory_analysis()
        mb = lambda v: (v or 0) / 2**20
        print(
            f"  [memory] temp {mb(analysis.temp_size_in_bytes):9.0f} MiB"
            f"  args {mb(analysis.argument_size_in_bytes):9.0f} MiB"
            f"  out {mb(analysis.output_size_in_bytes):8.0f} MiB"
            f"  const {mb(analysis.generated_code_size_in_bytes):7.0f} MiB",
            flush=True,
        )
    if self.calls_this_request == 0:
      # Free HBM at the first denoise call, to spot earlier stages (e.g.
      # the text encoder) leaving memory held.
      stats = jax.local_devices()[0].memory_stats() or {}
      used = stats.get("bytes_in_use", 0) / 2**30
      limit = stats.get("bytes_limit", 0) / 2**30
      print(f"  [hbm] before first denoise: used {used:.2f} / {limit:.2f} GiB,"
            f" free {limit - used:.2f} GiB", flush=True)
    # Track the peak, which (not the starting usage) decides whether a
    # configuration fits on a chip.
    _st = jax.local_devices()[0].memory_stats() or {}
    _peak = _st.get("peak_bytes_in_use", 0) / 2**30
    if _peak > getattr(self, "_hbm_peak", 0.0):
      self._hbm_peak = _peak
      print(f"  [hbm] peak {_peak:.2f} / "
            f"{_st.get('bytes_limit', 0) / 2**30:.2f} GiB", flush=True)
    inputs = {n: _to_jax(kwargs[n]) for n in names}
    # H3_DUMP_INPUTS=<dir>: save the transformer's input tensors for a
    # tensor-by-tensor comparison with the native pipeline. Done here, not
    # in `_native_call`, which runs under the jit trace (tracers only).
    # Use a call after step 0 when the other backend reads `--init-latents`
    # from this dump, since step 0 then agrees by construction.
    _dump = os.environ.get("H3_DUMP_INPUTS")
    # H3_DUMP_CALL indexes transformer calls across the process, warmup
    # calls included (keyed on an index to avoid a host sync per step).
    # H3_DUMP_EVERY=<dir>: save `hidden_states` on every call, to read the
    # backends' divergence as a curve over steps.
    _every = os.environ.get("H3_DUMP_EVERY")
    if _every:
      import numpy as _np
      self._ev_call = getattr(self, "_ev_call", -1) + 1
      os.makedirs(_every, exist_ok=True)
      _np.save(os.path.join(_every, f"hs_{self._ev_call:03d}.npy"),
               _np.asarray(jax.device_get(inputs["hidden_states"]),
                           _np.float32))
    _wc = int(os.environ.get("H3_DUMP_CALL", "0"))
    if _dump and not os.path.exists(os.path.join(_dump, ".done")):
      self._dump_call = getattr(self, "_dump_call", -1) + 1
      if self._dump_call == _wc:
        import numpy as _np
        _t = float(_np.asarray(jax.device_get(inputs["timestep"])).ravel()[0])
        os.makedirs(_dump, exist_ok=True)
        for _k, _v in inputs.items():
          _np.save(os.path.join(_dump, f"{_k}.npy"),
                   _np.asarray(jax.device_get(_v), _np.float32))
        open(os.path.join(_dump, ".done"), "w").close()
        print(f"  [dump] call {_wc} t={_t:.6f}: saved {len(inputs)} "
              f"tensors to {_dump}", flush=True)
    if self.cache_threshold > 0.0:
      with torch.no_grad():
        # The indicator is the timestep embedding, which every block's
        # modulation is an affine function of. Two rows wide, so it costs
        # nothing and never enters the compiled program.
        projected = self._model.time_proj(_raw_timestep)
        current = self._model.time_embedder(
            projected.to(next(self._model.time_embedder.parameters()).dtype)
        )
        # ref2va conditions the reference frames on their own timestep,
        # so the leading axis is 2 on one step and 3 on the next and
        # subtracting the raw tensors raised. Averaging it away leaves a
        # per-channel vector whose width is fixed.
        current = current.mean(0)
      if self._last_modulation is not None and self._last_output is not None:
        moved = (
            (current - self._last_modulation).abs().mean()
            / (self._last_modulation.abs().mean() + 1e-9)
        ).item()
        self._drift += moved
        if self._drift < self.cache_threshold:
          self.skipped += 1
          self.calls.append(0.0)
          self.calls_this_request += 1
          return self._last_output
        self._drift = 0.0
      self._last_modulation = current

    # Late steps move the output very little (adjacent steps differ by
    # 30-70% early and 5-7% from the ninth on), so a named step can return
    # the previous one instead of running. Changes the output.
    if (
        self.calls_this_request in self.reuse_steps
        and self._last_output is not None
    ):
      self.calls.append(0.0)
      self.calls_this_request += 1
      self.skipped += 1
      return self._last_output

    # Tracing the whole request overflows the device trace buffer: a
    # 134 s window came back with 808,602 host events and zero device
    # events. One step is enough and fits.
    tracing = (
        self.profile_dir is not None
        and self.calls_this_request == self.profile_call
    )
    if tracing:
      jax.profiler.start_trace(self.profile_dir)
    start = time.perf_counter()
    with jax.set_mesh(self._mesh):
      feed = self._params
      if self._native is not None:
        feed = {"__native__": self._native[1]}
      out = jax.block_until_ready(fn(feed, inputs))
    self.calls.append(time.perf_counter() - start)
    if tracing:
      jax.profiler.stop_trace()
      print(f"  [profile] traced call {self.profile_call}"
            f" -> {self.profile_dir}", flush=True)
    self.calls_this_request += 1
    result = tuple(torch.from_numpy(jax.device_get(o).astype("float32")) for o in out
    )
    if self.cache_threshold > 0.0 or self.reuse_steps:
      self._last_output = result
    return result


def _to_jax(tensor: torch.Tensor) -> jax.Array:
  t = tensor.detach().contiguous()
  if t.dtype == torch.bfloat16:
    return jnp.asarray(t.view(torch.uint16).numpy()).view(jnp.bfloat16)
  if t.dtype == getattr(torch, "float8_e4m3fn", None):
    return jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
  if t.is_floating_point():
    return jnp.asarray(t.float().numpy())
  return jnp.asarray(t.numpy())


def merge_pika_lora(model: torch.nn.Module, adapter_dir: str) -> dict:
  """Merge a Pika-H3-Flex adapter into the transformer's weights, in float32.

  Returns the adapter manifest (ladder, shifts, trunk).
  """
  import json
  import os

  from safetensors.torch import load_file

  with open(os.path.join(adapter_dir, "manifest.json")) as f:
    manifest = json.load(f)
  scale = float(manifest["alpha"]) / float(manifest["rank"])
  state = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
  modules = dict(model.named_modules())
  merged = 0
  with torch.no_grad():
    for key, a in state.items():
      if ".lora_A." not in key:
        continue
      b = state[key.replace(".lora_A.", ".lora_B.")]
      lin = modules[key.split(".lora_A.")[0]]
      lin.weight.copy_((lin.weight.float() + scale * (b.float() @ a.float()))
                       .to(lin.weight.dtype))
      merged += 1
  print(f"  [pika] merged {merged} LoRA modules", flush=True)
  return manifest


def pika_scheduler(base, manifest: dict, shift_key: str, seed: int):
  """A copy of `base` (the pipeline's H3 scheduler) running Pika's sampler.

  The fixed knots / 1000 through this modality's shift, whatever
  `num_inference_steps` the pipeline asks for; each step takes
  x0 = x + sigma * v and re-noises it to the next level with fresh noise.
  """
  import numpy as np

  cls = type(base)

  class PikaRenoise(cls):
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
      q = np.asarray(manifest["sampling"]["knots"], np.float64) / 1000.0
      s = float(manifest["sampling"][shift_key])
      ladder = (s * q / (1 + (s - 1) * q)).astype(np.float32)
      super().set_timesteps(device=device, sigmas=torch.from_numpy(ladder))
      self._gen = torch.Generator().manual_seed(seed + 1)

    def step(self, model_output, timestep, sample, return_dict=True):
      if self._step_index is None:
        self._step_index = (self.index_for_timestep(timestep)
                            if self._begin_index is None else self._begin_index)
      i = self._step_index
      sigma, nxt = float(self.sigmas[i]), float(self.sigmas[i + 1])
      x0 = sample.float() + sigma * model_output.float()
      if nxt > 0.0:
        eps = torch.randn(x0.shape, generator=self._gen, dtype=torch.float32)
        x0 = (1.0 - nxt) * x0 + nxt * eps.to(x0.device)
      self._step_index += 1
      out = x0.to(sample.dtype)
      if not return_dict:
        return (out,)
      from diffusers.schedulers.scheduling_minimax_h3 import (
          MiniMaxH3SchedulerOutput)
      return MiniMaxH3SchedulerOutput(prev_sample=out)

  sched = PikaRenoise.from_config(base.config)
  return sched


def install_int8(model: torch.nn.Module, dynamic: Sequence[str]) -> None:
  """Swaps every block's big Linears for their int8 equivalents.

  The scales in `int8_scales.json` were calibrated on one real step.
  `dynamic` names the layers whose activation scale is computed per row
  at runtime rather than read from the calibration.

  The quantised weight is registered as a parameter called `weight`, so
  `TorchaxTransformer` picks it up with the rest and `spec_for` shards
  it exactly as the bf16 weight it replaces.
  """
  import json

  from .int8_linear import convert

  with open(Path(__file__).with_name("int8_scales.json")) as fh:
    calibrated = json.load(fh)
  # `None` means every calibrated layer; the calibration is then only
  # used to name the candidates, not for its values.
  dynamic = set(calibrated) if dynamic is None else set(dynamic)
  swapped = 0
  for block in model.transformer_blocks:
    # Per-row first: `convert` only replaces what is still an nn.Linear,
    # so the calibrated pass then skips these.
    rows = {k: v for k, v in calibrated.items() if k in dynamic}
    swapped += convert(block, rows, dynamic=True)
    swapped += convert(
        block, {k: v for k, v in calibrated.items() if k not in dynamic}
    )
  print(f"  int8: {swapped} linears over {len(model.transformer_blocks)}"
        f" blocks, per-row on {sorted(dynamic)}")


def install_fp8(model: torch.nn.Module, dynamic: Sequence[str] | None) -> None:
  """Swaps every block's big Linears for native v7x float8_e4m3fn equivalents."""
  import json

  from .fp8_linear import convert

  with open(Path(__file__).with_name("int8_scales.json")) as fh:
    calibrated = json.load(fh)
  dynamic_set = set(calibrated) if dynamic is None else set(dynamic)
  swapped = 0
  for block in model.transformer_blocks:
    rows = {k: v for k, v in calibrated.items() if k in dynamic_set}
    swapped += convert(block, rows, dynamic=True)
    swapped += convert(
        block, {k: v for k, v in calibrated.items() if k not in dynamic_set}
    )
  print(f"  fp8 (e4m3fn): {swapped} linears over {len(model.transformer_blocks)}"
        f" blocks, per-row on {sorted(dynamic_set)}")


# The sparse layout of the call being traced, read by `_sparse_attention`.
# A dict rather than an argument because the kernel is reached through
# `csa.attention`, five frames below anything this file controls.
_SPARSE: dict = {"layout": None}
_DENSE_ATTENTION = None


def _sparse_attention(query, key, value, *, scale, block_q, block_kv, **kw):
  """`custom_splash_attention.attention`, block-sparse while `_SPARSE` says so.

  The block table depends only on where the video rows start and how many
  rows a latent frame has, so it is the same on every chip: tensor
  parallel splits the heads, not the sequence. That is what makes the
  one-chip kernel usable here unchanged.
  """
  layout = _SPARSE["layout"]
  if layout is None:
    return _DENSE_ATTENTION(query, key, value, scale=scale, block_q=block_q,
                            block_kv=block_kv, **kw)
  from ...kernels import sparse_splash
  from .h3_tpu import Sparsity

  seq = query.shape[2]
  video_start, rows_per_frame, window = layout
  table = Sparsity.table(seq, video_start, rows_per_frame, window,
                         min(block_q, seq), min(block_kv, seq))
  return sparse_splash.attention(query, key, value, scale=scale,
                                 block_q=block_q, block_kv=block_kv,
                                 kv_table=table, **kw)


def install_sparse_attention() -> None:
  """Routes `csa.attention` through the block-sparse kernel."""
  global _DENSE_ATTENTION
  if _DENSE_ATTENTION is not None:
    return
  from ...kernels import splash as csa

  _DENSE_ATTENTION, csa.attention = csa.attention, _sparse_attention


# Set by the wrapper when `--pad-sequence-to` rounds the packed
# sequence up. Everything outside attention is row-independent, so the
# fifty blocks may run on a padded row axis and the extra rows fall away
# at `proj_out`, which gathers by `video_indices`/`audio_indices` and
# never names them. Attention is the one place a padded row could reach
# a real one, and it is handed k and v sliced back. The native JAX path
# measured this at -25% a step on 57,290, eight chips.
TRUE_SEQ_LEN = 0


def _native_call(graphdef, state, inputs, sdpa):
  """One transformer call through the native JAX module.

  The state arrives as a jit argument and the module is rebuilt from it
  inside the trace, so the weights are operands rather than constants.
  """
  from flax import nnx


  model = nnx.merge(graphdef, state)
  out = model(
      inputs["hidden_states"],
      inputs["audio_hidden_states"],
      inputs["encoder_hidden_states"],
      inputs["timestep"],
      inputs["timestep_indices"],
      inputs["token_tags"],
      inputs["position_ids"],
      inputs["video_indices"],
      inputs["audio_indices"],
      inputs["text_indices"],
      sdpa=sdpa,
  )
  return tuple(out)


def _rotate_half_matrix(rotary_dim: int, head_dim: int, dtype) -> torch.Tensor:
  """R with ``x @ R == cat(-x2, x1, 0)``, x1/x2 the halves of x[:rotary_dim].

  One non-zero (+-1) per column, so the bfloat16 matmul is exact.
  """
  half = rotary_dim // 2
  r = torch.zeros(head_dim, head_dim, dtype=dtype)
  index = torch.arange(half)
  r[index + half, index] = -1  # out[:half] = -x2
  r[index, index + half] = 1  # out[half:rotary_dim] = x1
  return r


def _full_lane_rope(x, cos, sin, rotate):
  """The rotary embedding over all 128 lanes instead of the leading 96.

  H3 rotates 96 of the 128 head channels, and the reference does it by
  slicing those 96, chunking them in half and concatenating -- none of
  it aligned to the 128-lane vector registers, so XLA spends the block's
  time on relayouts (~15 ms a block). Padding the tables
  instead (cos with 1, sin with 0) and rotating with a signed
  permutation matmul keeps every operation on whole vregs, and the
  arithmetic is the same bf16 multiply-add. Measured 4.4% of a one-chip
  request.
  """
  pad = x.shape[-1] - cos.shape[-1]
  cos = torch.nn.functional.pad(cos.to(x.dtype), (0, pad), value=1.0)
  sin = torch.nn.functional.pad(sin.to(x.dtype), (0, pad), value=0.0)
  cos, sin = cos[None, :, None, :], sin[None, :, None, :]
  return x * cos + (x @ rotate) * sin


def install_attention(
    model: torch.nn.Module,
    mesh: jax.sharding.Mesh,
    fixed_m: bool = False,
    block_q: int = KV_BLOCK,
    block_kv: int = KV_BLOCK,
    full_lane_rope: bool = False,
    keep_heads: bool = False,
    qk_fp8: bool = False,
) -> None:
  """Routes every attention through the runtime's splash kernel."""
  import torchax

  from ...kernels import mesh_attention as attention
  from diffusers.models.transformers.transformer_minimax_h3 import (
      _apply_rotary_emb,
  )

  if keep_heads:
    # `mesh_attention` constrains self-attention's output to the *sequence*
    # layout (the one-axis branch of `_partition_specs`), so
    # every block flips from head-sharded to sequence-sharded and back.
    # H3's `to_out.0.weight` is split on its contraction -- `P(None,
    # "tp")` -- which wants the head layout, so the flip is bought and
    # then undone. Keeping the heads leaves one all-reduce after
    # `to_out`, which is what 1-D tensor parallelism is supposed to be.
    #
    # Patched here rather than upstream: the sequence layout is right
    # for the models that constraint was written for.
    original = attention._partition_specs

    def partition_specs(mesh_, query_shape, self_attention):
      query_spec, kv_spec, output = original(
          mesh_, query_shape, self_attention
      )
      if self_attention and len(mesh_.axis_names) == 1:
        return query_spec, kv_spec, query_spec
      return query_spec, kv_spec, output

    attention._partition_specs = partition_specs

  sdpa = attention.configure(
      mesh,
      torchax.default_env(),
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv,
      use_fixed_m=fixed_m,
      qk_fp8=qk_fp8,
  )

  def bind(attn):
    def forward(hidden_states, rotary_emb=None, attention_mask=None):
      del attention_mask  # MiniMax-H3 never passes one.
      query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
      key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
      value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
      query, key = attn.norm_q(query), attn.norm_k(key)
      if rotary_emb is not None:
        if full_lane_rope:
          cos, sin = rotary_emb
          rotate = _rotate_half_matrix(
              cos.shape[-1], query.shape[-1], query.dtype
          ).to(query.device)
          query = _full_lane_rope(query, cos, sin, rotate)
          key = _full_lane_rope(key, cos, sin, rotate)
        else:
          query = _apply_rotary_emb(query, *rotary_emb)
          key = _apply_rotary_emb(key, *rotary_emb)
      if QK_BARRIER:
        # Without this XLA fuses the RMSNorm and the rotation back into
        # the q and k projections, turning two clean matmuls into
        # something much slower (30 and 23 ms for the projections against
        # an ideal 9).
        env = torchax.default_env()
        barriered = jax.lax.optimization_barrier(
            (env.t2j_iso(query), env.t2j_iso(key))
        )
        query, key = (env.j2t_iso(x) for x in barriered)
      if TRUE_SEQ_LEN and TRUE_SEQ_LEN != key.shape[1]:
        # The sequence was rounded up outside attention; those rows are
        # keys to nobody. The padded queries stay -- the kernel pads the
        # query axis itself and their output rows are never read.
        key, value = key[:, :TRUE_SEQ_LEN], value[:, :TRUE_SEQ_LEN]
      out = sdpa(
          query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
      )
      out = out.transpose(1, 2).flatten(2, 3).type_as(query)
      return attn.to_out[1](attn.to_out[0](out))

    attn.forward = forward

  for block in model.transformer_blocks:
    bind(block.attn)


def segment_runs(
    token_tags: torch.Tensor, timestep_indices: torch.Tensor
) -> tuple[tuple[int, int, int], ...]:
  """The contiguous runs of a constant AdaLN index.

  `adaln_indices = timestep_indices * 3 + token_tags`, and the packed
  sequence is laid out `[text | audio | video]`, so for t2va this is three
  runs. Returned as `(start, end, index)`, which the block's modulation
  broadcasts over instead of gathering a row per token.
  """
  index = timestep_indices * MODALITY_NUM + token_tags
  edges = (index[1:] != index[:-1]).nonzero().flatten() + 1
  bounds = [0, *edges.tolist(), index.shape[0]]
  return tuple(
      (start, end, int(index[start]))
      for start, end in zip(bounds, bounds[1:])
  )


def install_segmented_modulation(
    model: torch.nn.Module,
    runs: tuple[tuple[int, int, int], ...],
    reassoc: bool = False,
) -> None:
  """Modulates by run rather than by row.

  Gathering the six modulation tensors materialises six copies of the
  hidden states. The last run is the video rows, 98% of the sequence, so
  broadcasting its row over the whole tensor and writing the other runs
  back over the head does the same arithmetic without them. Measured
  1.21x on a 15 s clip, agreeing to one bfloat16 ULP.
  """
  *head_runs, (bulk_start, _, bulk_index) = runs
  head_width = sum(end - start for start, end, _ in head_runs)
  mod_fn = (
      (lambda h, scale, shift: h + h * scale + shift)
      if reassoc
      else (lambda h, scale, shift: h * (1.0 + scale) + shift)
  )

  def apply(hidden, combine):
    out = combine(hidden, bulk_index)
    if not head_runs:
      return out
    head = torch.cat(
        [combine(hidden[:, start:end], index) for start, end, index in head_runs],
        dim=1,
    )
    return out.index_copy(
        1, torch.arange(head_width, device=hidden.device), head
    )

  def bind(block):
    def forward(
        hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None
    ):
      del adaln_indices  # The runs carry what it addressed.
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
          block.adaln_proj(temb)
      )
      residual = hidden_states
      normed = apply(
          block.norm1(hidden_states),
          lambda h, i: mod_fn(h, scale_msa[i], shift_msa[i]),
      )
      attended = block.attn(normed, rotary_emb, attention_mask)
      hidden_states = residual + apply(
          attended, lambda h, i: h * gate_msa[i]
      )
      residual = hidden_states
      normed = apply(
          block.norm2(hidden_states),
          lambda h, i: mod_fn(h, scale_mlp[i], shift_mlp[i]),
      )
      return residual + apply(block.ff(normed), lambda h, i: h * gate_mlp[i])

    block.forward = forward

  for block in model.transformer_blocks:
    bind(block)


def install_sequence_parallel(
    model: torch.nn.Module,
    mesh: jax.sharding.Mesh,
    runs: tuple[tuple[int, int, int], ...],
    reassoc: bool = False,
) -> None:
  """Shards the residual stream on sequence between the Linears.

  The weights stay tensor-parallel; only the stream's layout changes.
  Otherwise the stream is replicated, so every chip runs `norm1`, the two
  modulations, both gated residual adds and `norm2` over all 103,976
  rows -- eight copies of the same arithmetic, and the profile puts the
  fusions they live in at 18.3% of a step.

  Sharding the stream on the sequence makes that work `S/8` per chip.
  It costs nothing in bytes: a column-parallel Linear needs whole rows,
  so an all-gather goes in front of `to_q` and `ff.net.0.proj`, and the
  row-parallel `to_out.0` and `ff.net.2` end in a reduce-scatter instead
  of an all-reduce. Ring all-reduce *is* reduce-scatter plus all-gather,
  so the pair moves the same bytes the single collective did.

  Attention itself is untouched -- it still needs the whole sequence,
  and its Q/K/V are already head-sharded rather than replicated, so
  `norm_q` and the rope were never redundant.
  """
  from torchax import default_env

  env = default_env()
  by_sequence = NamedSharding(mesh, P(None, "tp", None))
  whole = NamedSharding(mesh, P(None, None, None))
  mod_fn = (
      (lambda h, scale, shift: h + h * scale + shift)
      if reassoc
      else (lambda h, scale, shift: h * (1.0 + scale) + shift)
  )

  def constrain(tensor: torch.Tensor, sharding) -> torch.Tensor:
    return env.j2t_iso(
        jax.lax.with_sharding_constraint(env.t2j_iso(tensor), sharding)
    )

  *head_runs, (_, _, bulk_index) = runs
  head_width = sum(end - start for start, end, _ in head_runs)

  def apply(hidden, combine):
    """The segmented modulation, unchanged; see `install_segmented_modulation`."""
    out = combine(hidden, bulk_index)
    if not head_runs:
      return out
    head = torch.cat(
        [combine(hidden[:, start:end], index) for start, end, index in head_runs],
        dim=1,
    )
    return out.index_copy(
        1, torch.arange(head_width, device=hidden.device), head
    )

  def bind(block):
    def forward(
        hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None
    ):
      del adaln_indices
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
          block.adaln_proj(temb)
      )
      hidden_states = constrain(hidden_states, by_sequence)
      residual = hidden_states
      normed = apply(
          block.norm1(hidden_states),
          lambda h, i: mod_fn(h, scale_msa[i], shift_msa[i]),
      )
      attended = block.attn(
          constrain(normed, whole), rotary_emb, attention_mask
      )
      hidden_states = residual + apply(
          constrain(attended, by_sequence), lambda h, i: h * gate_msa[i]
      )
      residual = hidden_states
      normed = apply(
          block.norm2(hidden_states),
          lambda h, i: mod_fn(h, scale_mlp[i], shift_mlp[i]),
      )
      ff_output = constrain(block.ff(constrain(normed, whole)), by_sequence)
      return residual + apply(ff_output, lambda h, i: h * gate_mlp[i])

    block.forward = forward

  for block in model.transformer_blocks:
    bind(block)


def install_reassociated_modulation(model: torch.nn.Module) -> None:
  """`h + h * scale + shift` instead of `h * (1 + scale) + shift`.

  The same value in exact arithmetic and a different one in bfloat16. Run
  against the stock path it measures the noise floor: how far apart two
  equivalent implementations of this model land after the whole schedule.
  Any optimisation that lands at that distance has not been shown to
  change anything.
  """

  fired = [0]

  def bind(block):
    def forward(
        hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None
    ):
      # Confirm once that the rebound forward actually runs.
      if not fired[0]:
        fired[0] = 1
        print(f"  [reassoc] rebound forward active on {len(list(model.transformer_blocks))} blocks", flush=True)
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
          block.adaln_proj(temb)
      )

      def modulate(hidden, scale, shift):
        return (
            hidden
            + hidden * scale.index_select(0, adaln_indices)
            + shift.index_select(0, adaln_indices)
        )

      residual = hidden_states
      normed = modulate(block.norm1(hidden_states), scale_msa, shift_msa)
      attended = block.attn(normed, rotary_emb, attention_mask)
      hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attended
      residual = hidden_states
      normed = modulate(block.norm2(hidden_states), scale_mlp, shift_mlp)
      return residual + gate_mlp.index_select(0, adaln_indices) * block.ff(normed)

    block.forward = forward

  for block in model.transformer_blocks:
    bind(block)


def install_block_dump(model: torch.nn.Module) -> None:
  """`DT_DUMP_BLOCKS=<dir>`: per-block probes in the native `jax_dit` format.

  Forward hooks see tracer-backed tensors under torchax, so they hand the
  jax arrays to `jax.debug.callback`, which writes them at run time
  (`jax_dit._blk_save`: norms and strided slices only). Used to compare
  the native and torch DiTs layer by layer.
  """
  import functools
  import torchax
  from .jax_dit import _blk_save

  env = torchax.default_env()

  def save(tag, t):
    jax.debug.callback(functools.partial(_blk_save, tag), env.t2j_iso(t))

  blocks = list(model.transformer_blocks)
  blocks[0].register_forward_pre_hook(
      lambda m, args: save("x0", args[0]) or None)
  for i, b in enumerate(blocks):
    if i in (0, 4, 9, 19, 29, 39, 49):
      b.register_forward_hook(
          lambda m, args, out, i=i: save(
              f"b{i:02d}", out[0] if isinstance(out, tuple) else out) or None)
  model.norm_out.register_forward_hook(
      lambda m, args, out: save("normout", out) or None)
  print(f"  [blk] dump hooks installed on {len(blocks)} blocks", flush=True)


def pin_modulation(model: torch.nn.Module, mesh: jax.sharding.Mesh) -> None:
  """Pins the modulation tensors so the blocks' gathers have a sharding."""
  import torchax

  spec = NamedSharding(mesh, P(None, "tp"))

  for block in model.transformer_blocks:
    module = block.adaln_proj
    inner = module.forward

    def forward(temb, inner=inner):
      env = torchax.default_env()
      return tuple(
          env.j2t_iso(jax.lax.with_sharding_constraint(env.t2j_iso(t), spec))
          for t in inner(temb)
      )

    module.forward = forward


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--workflow",
      default="t2va",
      choices=("t2va", "ref2va", "fl2va"),
      help="ref2va is the reference-to-video path; it uses a different "
      "transformer partition (transformer_ref).",
  )
  ap.add_argument(
      "--text-encoder-on-tpu",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Run Qwen3-VL's first 50 decoder layers on the mesh. "
      "Default: on for ref2va (thousands of image tokens; ~275 -> 191 s "
      "per request), off for t2va (a handful of text tokens is faster "
      "on the host). --no-text-encoder-on-tpu forces the host path.",
  )
  ap.add_argument(
      "--text-encoder-resident",
      type=int,
      default=0,
      help="How many of Qwen3-VL's 50 layers keep their weights in HBM "
      "instead of streaming them from pinned host every call. The "
      "stage is bound by the host stream (48.8 GB at ~28 GB/s). Layers "
      "are dealt round-robin over the chips; all 50 does not fit next "
      "to H3 on eight chips, 32 does.",
  )
  ap.add_argument(
      "--text-encoder-tp",
      action="store_true",
      help="Shard Qwen3-VL's decoder layers over all eight chips "
      "instead of running them on one. The stage is host-bandwidth "
      "bound (48.8 GB streamed at ~28 GB/s), so this splits the "
      "stream eight ways. Each layer runs inside one "
      "`shard_map` so the head reshape stays device-local.",
  )
  ap.add_argument(
      "--attention-keep-heads",
      action="store_true",
      help="Leave self-attention's output in the head layout instead of "
      "the sequence layout. The layout flip costs one all-to-all per "
      "block; on a ref2va step about 21.6%% of the step was exposed "
      "collective-permute.",
  )
  ap.add_argument(
      "--vae-on-demand",
      action="store_true",
      help="Park the VAE in pinned host memory and bring it back "
      "only for encode/decode. It is idle through text encoding and "
      "denoising; freeing its ~9.7 GiB per chip leaves headroom, e.g. "
      "for --text-encoder-resident. Bit-identical output.",
  )
  ap.add_argument(
      "--vae-fused-clip",
      action="store_true",
      help="Cut, decode and stitch a whole clip group in one jitted "
      "program instead of ~540 eager dispatches a decode. Bit-identical "
      "output; median stage time 3.32 -> 2.89 s.",
  )
  ap.add_argument(
      "--hbm-ballast-gib", type=float, default=0.0,
      help="Hold this much HBM per chip, unused. A diagnostic for how "
      "free-memory pressure affects the transformer's step time.",
  )
  ap.add_argument("--vae-tile", type=int,
                  help="The VAE's tile edge; its config says 256.")
  ap.add_argument(
      "--vae-overlap", type=int,
      help="The tile overlap; its config says 64. The tiles overlap, so "
      "the decoder recomputes the overlap: at 480p the 3x4 tiles cover "
      "1.97x the frame and pad to 16 of 12, so 2.63x the necessary "
      "work. 256/32 gives 2x4 = 8 tiles that fill the mesh exactly and "
      "1.31x. Narrower blending -- check the seams.",
  )
  ap.add_argument(
      "--max-duration",
      type=float,
      default=0.0,
      help="Lift the pipeline's 15 s cap. 362 frames at 24 fps "
      "is 15.08 s, just over.",
  )
  ap.add_argument(
      "--reference",
      help="Image reference for ref2va.",
  )
  ap.add_argument(
      "--reference-short-edge",
      type=int,
      help="What the reference image's short edge is resized to before "
      "Qwen3-VL sees it. The released checkpoint says 2048, which turns "
      "a 1344x768 reference into 3584x2048 = 28,672 patches, and those "
      "land in the packed sequence twice: 7,168 vision tokens and 7,168 "
      "rows of reference latents. Halving this takes the sequence from "
      "55,284 to 44,532. Changes what the model is shown.",
  )
  ap.add_argument("--prompt", default="A red fox trotting through a snowy pine forest")
  ap.add_argument("--num-frames", type=int, default=49)
  # The canvas. Multiples of 32; the model is trained at 768x1344
  # and 5-15 s at 24 fps, so 345 frames is the longest legal clip.
  ap.add_argument("--height", type=int, default=768)
  ap.add_argument("--width", type=int, default=1344)
  ap.add_argument("--steps", type=int, default=4)
  ap.add_argument("--pika-adapter", default=None,
                  help="Pika-H3-Flex adapter dir (fl2va/ or ref2va/): merge the "
                       "LoRA and sample its six-step re-noising ladder")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", default="out")
  ap.add_argument("--fixed-m", action="store_true")
  ap.add_argument("--tag", default="t2va")
  # Off by default: padding the timestep to a fixed width saves the second
  # compile, but it changes the matmul's shape and so its accumulation
  # order, and at 20 steps the trajectories end 17 dB apart. Precompiling
  # both shapes in a warmup is the way to the same latency.
  ap.add_argument("--timestep-pad", action="store_true")
  ap.add_argument("--no-segmented", action="store_true")
  ap.add_argument(
      "--cache-threshold",
      type=float,
      default=0.0,
      help="Skip a step when the modulation has moved less than this since "
      "the last computed one. 0 disables.",
  )
  ap.add_argument("--block-q", type=int, default=KV_BLOCK)
  # Swept at 128 granularity on the real 103,976 sequence; the
  # autotuner's analytic pick is 4096/2048 and measures 4.3% slower.
  ap.add_argument("--block-kv", type=int, default=KV_BLOCK)
  ap.add_argument(
      "--fast-vae",
      action="store_true",
      help="Multi-clip batched video VAE with stitch, blend and "
      "postprocess on the TPU instead of the host.",
  )
  ap.add_argument(
      "--vae-bf16",
      action="store_true",
      help="Run the VAE's ViT decoder in bfloat16, rope and norm in "
      "float32. Changes the output; measure it separately.",
  )
  ap.add_argument(
      "--vae-clips-per-batch",
      type=int,
      default=1,
      help="Temporal clips decoded in one batch. One clip is 28 tiles "
      "at 768p and 12 at 480p, neither a multiple of eight, so a single "
      "clip wastes 14%% and 33%% on padding; two give 56 = 8 x 7 and "
      "24 = 8 x 3. 1 keeps the old per-clip path.",
  )
  ap.add_argument(
      "--sequence-parallel",
      action="store_true",
      help="Shard the residual stream on sequence between the "
      "Linears. Weights stay tensor-parallel; this only stops "
      "every chip redoing the same layernorm, modulation and "
      "residual add over all 103,976 rows.",
  )
  ap.add_argument(
      "--full-lane-rope",
      action="store_true",
      help="Rotate over all 128 head lanes with a padded table "
      "and a permutation matmul instead of slicing 96 of them. "
      "Same arithmetic; about 4.4%% faster on one chip.",
  )
  ap.add_argument(
      "--reuse-steps",
      default="",
      help="Comma-separated transformer calls that return the "
      "previous call's output instead of running. Changes the "
      "output.",
  )
  ap.add_argument(
      "--pad-sequence-to",
      type=int,
      default=0,
      help="Round the packed sequence up to a multiple of this "
      "outside attention; 0 is off. 57,290 is not even a multiple "
      "of eight, and the native path measured 128 at -25%% a step "
      "on eight chips.",
  )
  ap.add_argument("--profile", help="Trace one step into this directory.")
  ap.add_argument(
      "--vae-stitch-on-device",
      action="store_true",
      help="Stitch and blend the spatial tiles on the TPU and "
      "bring back only the stitched clip. Standalone it took "
      "the decode from 8.73 s to 6.98 s.",
  )
  ap.add_argument(
      "--dump-latents",
      help="Save the VAE decode's input here, so the stage can "
      "be benchmarked on its own.",
  )
  ap.add_argument(
      "--profile-vae",
      help="Trace the whole video VAE decode stage here.",
  )
  ap.add_argument("--profile-call", type=int, default=8,
                  help="Which transformer call of the timed "
                  "request to trace. Steady state, not the first.")
  ap.add_argument(
      "--sparse-window",
      type=int,
      default=0,
      help="Latent frames a video row may attend to either "
      "side of itself, at block granularity. Changes the "
      "output. 0 is dense; 16 is a tested setting.",
  )
  ap.add_argument("--sparse-dense-steps", type=int, default=6)
  ap.add_argument(
      "--scoped-vmem-kib",
      type=int,
      default=0,
      help="Per-op scratchpad VMEM for the transformer program "
      "only. 0 keeps XLA's 16 MiB default; 98304 measured best.",
  )
  ap.add_argument(
      "--memory-analysis",
      action="store_true",
      help="Print the compiled program's scratch and argument "
      "sizes. A block size that needs more scratch leaves less "
      "room for everything else, which a one-block benchmark "
      "cannot show.",
  )
  ap.add_argument(
      "--int8",
      action="store_true",
      help="Quantise the big Linears to int8 with the calibrated scales.",
  )
  ap.add_argument(
      "--int8-dynamic",
      nargs="*",
      default=None,
      help="Which Linears take a per-row activation scale; the default "
      "is all of them. A calibrated scale cannot be used here: the "
      "activations grow with depth -- attn.to_out.0 runs 19 at block 0 "
      "and 512 at block 49 against a calibrated ceiling of 32, so 42%% "
      "of its elements saturate and the output is noise. "
      "A row scale cannot saturate and lands within 1.5-3x of the floor "
      "set by quantising the weight at all.",
  )
  ap.add_argument(
      "--fp8",
      action="store_true",
      help="Quantise the big Linears to native v7x float8_e4m3fn.",
  )
  ap.add_argument(
      "--fp8-dynamic",
      nargs="*",
      default=None,
      help="Which Linears take a per-row FP8 activation scale; default is all.",
  )
  ap.add_argument(
      "--qk-fp8",
      action="store_true",
      help="Cast Q and K to float8_e4m3fn inside the Pallas splash kernel.",
  )
  ap.add_argument(
      "--chips",
      type=int,
      default=0,
      help="Use this many JAX devices; 0 uses all available devices.",
  )
  ap.add_argument(
      "--warmup",
      action="store_true",
      help="Build both timestep programs before the timed request, so the "
      "two compiles land in startup instead of in the first two steps.",
  )
  ap.add_argument(
      "--native-dit",
      action="store_true",
      help="Run the transformer as the native JAX module instead of the "
           "traced torch one. Same weights, same VAE and text encoder; "
           "about 2.18 s a call against 3.1 on eight chips.",
  )
  ap.add_argument(
      "--offload",
      action="store_true",
      help="Keep the weights in pinned host memory and copy each "
           "shard in as the step needs it. Follows the mesh "
           "partitioning, so a few chips can run a model whose weights "
           "alone would fill them.",
  )
  ap.add_argument(
      "--reassoc",
      action="store_true",
      help="Algebraically identical modulation, associated differently. "
      "Measures how far two equivalent implementations drift apart.",
  )
  args = ap.parse_args()
  if args.text_encoder_on_tpu is None:
    args.text_encoder_on_tpu = args.workflow == "ref2va" and args.chips != 1

  import os

  from diffusers import ModularPipeline

  os.makedirs(args.out, exist_ok=True)
  devices = jax.devices()[: args.chips] if args.chips else jax.devices()
  mesh = jax.make_mesh(
      (len(devices),),
      ("tp",),
      devices=devices,
      axis_types=(jax.sharding.AxisType.Auto,),
  )
  print(f"{devices[0].device_kind} x {len(devices)}")
  # Log every non-default flag up front so a run's settings are recorded.
  print("  [args] " + " ".join(
      f"{k}={v}" for k, v in sorted(vars(args).items())
      if v not in (None, False, 0, "", "out")), flush=True)

  start = time.perf_counter()
  from .jax_dit import find_snapshot
  model_path = find_snapshot(REPO)
  _local = True
  pipe = ModularPipeline.from_pretrained(model_path, workflow=args.workflow,
                                         local_files_only=_local)
  if args.max_duration:
    # e.g. 362 frames (17 * 21 + 5) decode fine but are 15.08 s at
    # 24 fps, over the pipeline's 15 s cap. Patched here because
    # `--warmup` calls the pipeline well before the timed request.
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MiniMaxH3ModularPipeline,
    )

    cap = args.max_duration
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: cap)
    type(pipe).max_duration = property(lambda self: cap)
    print(f"  [duration] cap raised to {cap}s")
  # Naming them is not optional: a bare `load_components()` leaves the
  # processor unset, and the text encoder step needs it for the
  # multimodal token type ids.
  # `ref2va` denoises against `transformer_ref`, a different 61.7 GB
  # partition; naming the right one is also what keeps `from_pretrained`
  # from pulling both.
  which = "transformer_ref" if args.workflow == "ref2va" else "transformer"
  if os.path.isdir(model_path):
    for _v in vars(pipe).values():
      _items = _v.values() if isinstance(_v, dict) else (_v if isinstance(_v, (list, tuple)) else ())
      for _spec in _items:
        if hasattr(_spec, "pretrained_model_name_or_path"):
          _spec.pretrained_model_name_or_path = model_path
  pipe.load_components(
      names=[
          "text_encoder",
          "tokenizer",
          "processor",
          which,
          "vae",
          "audio_vae",
          "scheduler",
          "audio_scheduler",
      ],
      torch_dtype=torch.bfloat16,
      local_files_only=_local,
  )
  missing = [n for n in pipe.component_names if getattr(pipe, n, None) is None]
  if missing:
    print(f"warning: still unset: {missing}")
  print(f"load: {time.perf_counter() - start:.0f}s")

  # `ref2va` denoises against a different checkpoint partition, exposed
  # as `transformer_ref`; `t2va` and `fl2va` use `transformer`.
  transformer = getattr(pipe, which)
  print(f"  [workflow] {args.workflow}, denoising against {which}")
  pika = merge_pika_lora(transformer, args.pika_adapter) if args.pika_adapter else None
  if args.fp8:
    install_fp8(transformer, args.fp8_dynamic)
  elif args.int8:
    install_int8(transformer, args.int8_dynamic)
  if args.hbm_ballast_gib:
    count = int(args.hbm_ballast_gib * 2**30 / 4)
    _ballast = jax.device_put(
        jnp.zeros((count,), jnp.float32),
        jax.sharding.NamedSharding(mesh, P()),
    )
    jax.block_until_ready(_ballast)
    print(f"  [hbm] ballast {args.hbm_ballast_gib:.1f} GiB/chip", flush=True)

  install_attention(
      transformer, mesh, args.fixed_m, args.block_q, args.block_kv,
      full_lane_rope=args.full_lane_rope,
      keep_heads=args.attention_keep_heads,
      qk_fp8=args.qk_fp8,
  )
  pin_modulation(transformer, mesh)
  if os.environ.get("DT_DUMP_BLOCKS"):
    install_block_dump(transformer)
  wrapped = TorchaxTransformer(transformer, mesh, offload=args.offload)
  wrapped.pad_timestep = args.timestep_pad
  wrapped.segmented = not args.no_segmented
  wrapped.cache_threshold = args.cache_threshold
  wrapped.report_memory = args.memory_analysis
  wrapped.scoped_vmem_kib = args.scoped_vmem_kib
  wrapped.sequence_parallel = args.sequence_parallel
  wrapped.pad_sequence_to = args.pad_sequence_to
  wrapped.reuse_steps = frozenset(
      int(x) for x in args.reuse_steps.split(',') if x
  )
  wrapped.profile_dir = args.profile
  wrapped.profile_call = args.profile_call
  if args.native_dit:
    wrapped.use_native_dit()
  wrapped.sparse_window = args.sparse_window
  wrapped.sparse_dense_steps = args.sparse_dense_steps
  if args.sparse_window:
    install_sparse_attention()
  wrapped.reassoc = args.reassoc
  if args.reassoc:
    install_reassociated_modulation(transformer)
  pipe.update_components(**{which: wrapped})
  if pika is not None:
    pipe.update_components(
        scheduler=pika_scheduler(pipe.scheduler, pika, "video_shift", args.seed),
        audio_scheduler=pika_scheduler(pipe.audio_scheduler, pika, "audio_shift",
                                       args.seed))

  # On the host the video VAE decode dominates (458 s of 529 s on a
  # 124-frame clip). Every interior tile has the same shape, so one
  # compiled program serves them all.
  from .vae_batched import BatchedTileDecoder, install_fast_video_vae

  if args.fast_vae:
    # Compared with `BatchedTileDecoder`:
    #   - batches several clips so the tile count divides the mesh (a
    #     28-tile clip padded to 32 wastes 14%; four clips are 112 = 8 x 14);
    #   - stitching, temporal blending, denormalization and the uint8
    #     conversion run in one compiled program instead of on the host;
    #   - bf16 crosses to jax via a uint16 bitcast, not float32.
    # Without `--vae-bf16` the decoder stays float32 and the path is lossless.
    install_fast_video_vae(
        pipe,
        mesh,
        clips_per_batch=args.vae_clips_per_batch,
        use_bf16=args.vae_bf16,
    )
  else:
    # Batched over the mesh rather than one tile at a time on one chip;
    # the VAE's own tiling geometry and blending are untouched.
    if args.workflow == "ref2va":
      # On the host the reference encode dominates a ref2va request
      # (40 s at 768p, 118 s at 480p, against a 22 s denoise).
      from .vae_batched import install_batched_encoder

      install_batched_encoder(pipe.vae, mesh)
      print("  [vae] encoder also placed on the mesh")

    if args.vae_tile:
      pipe.vae.tile_sample_min_height = args.vae_tile
      pipe.vae.tile_sample_min_width = args.vae_tile
    if args.vae_overlap is not None:
      pipe.vae.tile_sample_min_overlap_height = args.vae_overlap
      pipe.vae.tile_sample_min_overlap_width = args.vae_overlap
    decoder = BatchedTileDecoder(pipe.vae, mesh)
    if args.vae_clips_per_batch > 1:
      decoder.install_lookahead(args.vae_clips_per_batch)
    decoder.stitch_on_device = args.vae_stitch_on_device
    decoder.fused_clip = args.vae_fused_clip
    pipe.vae._decode_clip = decoder
    if args.vae_on_demand:
      # The VAE (~9.7 GiB) is idle during text encoding and denoising;
      # parking it frees room for `--text-encoder-resident`.
      from . import vae_on_demand
      vae_on_demand.install(pipe.vae, mesh)

  # The audio VAE decode is a single call, so it needs no tiling.
  from .vae_batched import CompiledAudioDecoder

  audio = CompiledAudioDecoder(
      pipe.audio_vae.dec_in_proj, pipe.audio_vae.decoder, mesh
  )
  pipe.audio_vae.dec_in_proj = torch.nn.Identity()
  pipe.audio_vae.decoder = audio

  # Time the components directly: the executor does not call the objects
  # in `blocks.sub_blocks`, so wrapping those would not work.
  stage: dict[str, float] = {}

  def clock(label, owner, attr):
    inner = getattr(owner, attr)

    def timed(*a, **k):
      begin = time.perf_counter()
      out = inner(*a, **k)
      stage[label] = stage.get(label, 0.0) + time.perf_counter() - begin
      return out

    setattr(owner, attr, timed)

  if args.text_encoder_on_tpu:
    from . import text_encoder_tpu

    text_encoder_tpu.install(
        pipe.text_encoder, mesh,
        resident=args.text_encoder_resident,
        tensor_parallel=args.text_encoder_tp,
    )
    # `DT_TX_VISION_ON_HOST=1`: language model on the TPU, vision tower
    # left on the host, to isolate the two when debugging.
    if os.environ.get("DT_TX_VISION_ON_HOST") != "1":
      text_encoder_tpu.install_vision(pipe.text_encoder)
    else:
      print("  [text_encoder] vision tower left on host (DT_TX_VISION_ON_HOST=1)",
            flush=True)
  clock("text encode", pipe.text_encoder, "forward")

  # `DT_TEXT_DUMP=<path>`: save the text encoder's output, to compare the
  # host and TPU text-encoder paths directly.
  _dump = os.environ.get("DT_TEXT_DUMP")
  if _dump:
    # Hook `get_qwen3vl_prompt_embeds`, not `text_encoder.forward`: the
    # pipeline calls `text_encoder.model(...)` and bypasses the outer
    # forward. This captures the conditioning fed to the DiT on both paths.
    from diffusers.modular_pipelines.minimax_h3 import encoders as _enc

    _inner_ge = _enc.get_qwen3vl_prompt_embeds

    def _dumping(*a, **k):
      out = _inner_ge(*a, **k)
      arr = out[0] if isinstance(out, (tuple, list)) else out
      try:
        import numpy as _np
        v = _np.asarray(arr.detach().to(torch.float32).cpu().numpy())
        _np.save(_dump, v)
        print(f"  [text-dump] {v.shape} {v.dtype} mean {v.mean():.5f} "
              f"std {v.std():.5f} -> {_dump}", flush=True)
      except Exception as exc:  # noqa: BLE001
        print(f"  [text-dump] save failed: {type(exc).__name__}: {exc}",
              flush=True)
      return out

    _enc.get_qwen3vl_prompt_embeds = _dumping
  # Clock Qwen3-VL's vision tower and language model separately: ref2va
  # sends ~7000 image tokens through them, against t2va's ~10.
  vl = getattr(pipe.text_encoder, "model", None)
  for attr in ("visual", "vision_tower", "vision_model"):
    tower = getattr(vl, attr, None)
    if tower is not None:
      clock(f"VL:{attr}", tower, "forward")
  if getattr(vl, "language_model", None) is not None:
    clock("VL:language_model", vl.language_model, "forward")
  if args.profile_vae:
    # Trace the VAE decode stage on its own, to separate device work from
    # host-side tiling/stitching. (Splitting with `block_until_ready`
    # would serialize the async work and distort the timing.)
    inner_decode = pipe.vae.decode

    def traced_decode(*a, **kw):
      jax.profiler.start_trace(args.profile_vae)
      try:
        return inner_decode(*a, **kw)
      finally:
        jax.profiler.stop_trace()
        print(f"  [profile] VAE decode -> {args.profile_vae}", flush=True)

    pipe.vae.decode = traced_decode
  if args.dump_latents:
    # The decode only needs the latents; dumping them lets the VAE be
    # benchmarked without rerunning warmup and denoising.
    inner_dump = pipe.vae.decode

    def dumping_decode(z, *a, **kw):
      import numpy as np

      np.save(args.dump_latents, z.detach().to(torch.float32).numpy())
      print(f"  [latents] {tuple(z.shape)} -> {args.dump_latents}",
            flush=True)
      return inner_dump(z, *a, **kw)

    pipe.vae.decode = dumping_decode
  clock("video VAE decode", pipe.vae, "decode")
  clock("audio VAE decode", pipe.audio_vae, "decode")
  clock("postprocess video", pipe.video_processor, "postprocess_video")
  clock("text encode", pipe.text_encoder, "__call__")
  clock("prepare layout", pipe.blocks.sub_blocks["denoise.prepare_layout"], "__call__")
  clock("prepare latents", pipe.blocks.sub_blocks["denoise.prepare_latents"], "__call__")
  clock("after denoise", pipe.blocks.sub_blocks["denoise.after_denoise"], "__call__")
  # Clock every non-denoise sub-block so nothing hides in "rest".
  # Patch the block's *class*, not the instance or the dict: `obj(...)`
  # resolves `__call__` on the type, and `sub_blocks` is rebuilt on access.
  _patched = set()
  for name, block in list(pipe.blocks.sub_blocks.items()):
    if name.startswith("denoise."):
      continue
    cls = type(block)
    if cls in _patched:
      continue
    _patched.add(cls)
    inner_call = cls.__call__
    label = f"block:{name}"

    def timed_call(self, *a, _inner=inner_call, _label=label, **kw):
      begin = time.perf_counter()
      out = _inner(self, *a, **kw)
      stage[_label] = stage.get(_label, 0.0) + time.perf_counter() - begin
      return out

    cls.__call__ = timed_call
  if hasattr(pipe.vae, "encode"):
    clock("video VAE encode", pipe.vae, "encode")

  # Built before the warmup, not after: `ref2va` requires `references`
  # on every call, and the warmup is a call.
  extra = {}
  if args.workflow == "ref2va":
    if not args.reference:
      raise SystemExit("ref2va needs --reference")
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    extra["references"] = [MiniMaxH3ImageReference.from_file(args.reference)]
    if args.reference_short_edge:
      # Set before the warmup: the edge decides the sequence length, and
      # the warmup's whole job is to compile the shapes the timed call
      # will use.
      pipe.register_to_config(
          reference_image_short_edge=args.reference_short_edge
      )
      print(f"  [reference] short edge {args.reference_short_edge}"
            f" (default 2048)")

  if args.warmup:
    # A throwaway request at the same geometry and the fewest steps that
    # still reaches both timestep shapes. Its programs are what the timed
    # request reuses -- the cache keys are the shapes, not the values.
    #
    # With sparsity on, three steps is not enough: the sparse layout is
    # part of the cache key, so the first sparse call of the timed
    # request would compile a second program inside the measurement. Go
    # one step past the dense prefix.
    if args.timestep_pad:
      warm_steps = max(2, args.sparse_dense_steps + 2) if args.sparse_window else 2
    else:
      warm_steps = 3
      if args.sparse_window:
        warm_steps = max(warm_steps, args.sparse_dense_steps + 2)
    warm = time.perf_counter()
    wrapped._in_warmup = True
    try:
      pipe(
          prompt=args.prompt,
          num_frames=args.num_frames,
          **extra,
          height=args.height,
          width=args.width,
          num_inference_steps=warm_steps,
          generator=torch.Generator().manual_seed(args.seed),
          output=["videos"],
      )
    finally:
      wrapped._in_warmup = False
    print(f"warmup: {time.perf_counter() - warm:.1f}s")
    wrapped.calls.clear()
    wrapped.calls_this_request = 0
    wrapped.skipped = 0
    wrapped._last_output = None
    wrapped._last_modulation = None
    wrapped._drift = 0.0
    stage.clear()

  start = time.perf_counter()

  results = pipe(
      prompt=args.prompt,
      num_frames=args.num_frames,
      **extra,
      height=args.height,
      width=args.width,
      num_inference_steps=args.steps,
      generator=torch.Generator().manual_seed(args.seed),
      output=["videos", "audio", "sampling_rate"],
  )
  wall = time.perf_counter() - start
  # Read `wrapped` directly: the pipeline exposes it as `transformer` or
  # `transformer_ref` depending on the workflow.
  calls = wrapped.calls
  print(f"generate: {wall:.1f}s")
  print(
      f"  transformer: {len(calls)} calls, "
      + " ".join(f"{c:.1f}" for c in calls)
      + f"  = {sum(calls):.1f}s ({sum(calls)/wall*100:.0f}%)"
      + (f"  skipped {wrapped.skipped}" if wrapped.skipped else "")
  )
  print(f"  host (text encode, VAE, packing): {wall - sum(calls):.1f}s")
  decoder = getattr(pipe.vae, "_decode_clip", None)
  if hasattr(decoder, "compute"):
    print(
        f"    VAE breakdown: device compute {decoder.compute:.1f}s,"
        f" transfer back {decoder.transfer:.1f}s"
    )
  for label, seconds in sorted(stage.items(), key=lambda kv: -kv[1]):
    print(f"    {label:22s} {seconds:7.1f}s  {seconds/wall*100:4.1f}%")
  # Only the `block:` entries are pipeline blocks; the rest are nested
  # inside them and would be double-counted.
  outer = sum(v for k, v in stage.items() if k.startswith("block:"))
  print(f"    {'rest (packing etc)':22s} "
        f"{wall - sum(calls) - outer:7.1f}s")

  # `export_to_video` writes frames only; `encode_video` muxes the audio.
  from diffusers.utils.export_utils import encode_video

  path = f"{args.out}/h3_{args.tag}.mp4"
  encode_video(
      results["videos"][0],
      fps=24,
      output_path=path,
      audio=results["audio"][0],
      audio_sample_rate=results["sampling_rate"],
  )
  import numpy as np

  np.save(f"{args.out}/h3_{args.tag}_frames.npy", np.asarray(results["videos"][0]))

  # Also write `<out>/<tag>/video.npy` + `audio.wav`, the layout the
  # quality-scoring script reads.
  import os
  import wave

  folder = f"{args.out}/{args.tag}"
  os.makedirs(folder, exist_ok=True)
  # A symlink, not a copy: the frame arrays are about a gigabyte each.
  link = f"{folder}/video.npy"
  if os.path.lexists(link):
    os.remove(link)
  os.symlink(f"{args.out}/h3_{args.tag}_frames.npy", link)
  audio = np.asarray(results["audio"][0])
  if audio.ndim == 1:
    audio = audio[:, None]
  elif audio.shape[0] < audio.shape[1]:
    # The pipeline returns channel-first; `wave` wants frames x channels.
    audio = audio.T
  with wave.open(f"{folder}/audio.wav", "wb") as handle:
    handle.setnchannels(audio.shape[1])
    handle.setsampwidth(2)
    handle.setframerate(int(results["sampling_rate"]))
    handle.writeframes(
        np.clip(audio * 32767.0, -32768, 32767).astype("<i2").tobytes()
    )
  print(f"wrote {path} and {folder}/ for quality.py")
  decoder = getattr(pipe.vae, "_decode_clip", None)
  if getattr(decoder, "stitch", 0.0):
    # Host-side compute within the VAE stage (stitch and spatial blend).
    print(f"  VAE host-side stitch+blend: {decoder.stitch:.2f}s")


if __name__ == "__main__":
  main()
