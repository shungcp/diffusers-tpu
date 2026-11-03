"""MiniMax-H3 on one v6e chip: the heavy modules, run through torchax.

The transformer is 66 GB and a chip has 32, so the block stack lives in
pinned host memory and each block's weights are copied in by a
`device_put` inside the jitted step, right before the block that uses
them; XLA overlaps that copy with the previous block's compute. Everything
else of the transformer (projections, refiner, time embedding, output
heads -- about 2 GB) stays in HBM.

Each runner replaces one module's `forward` in place, so the diffusers
modular pipeline drives the request unchanged and only the arithmetic
moves to the TPU.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax

# Swept on v6e at 38222 rows with fixed-m: 83.7 ms per call against 88.4
# for 2048 throughout. bq = bkv = 4096 falls off a cliff.
BLOCKS = {"block_q": 2048, "block_kv": 4096, "block_kv_compute": 512}



def to_jax(tensor: torch.Tensor) -> jax.Array:
  """CPU torch -> host jax, keeping bfloat16 (numpy has none)."""
  tensor = tensor.detach()
  if tensor.dtype == torch.bfloat16:
    return jnp.asarray(tensor.float().numpy()).astype(jnp.bfloat16)
  return jnp.asarray(tensor.numpy())


def to_torch(array: jax.Array) -> torch.Tensor:
  if array.dtype == jnp.bfloat16:
    # Move the raw 16-bit words: half the bytes of a float32 round trip.
    bits = np.asarray(jax.lax.bitcast_convert_type(array, jnp.uint16))
    return torch.from_numpy(bits).view(torch.bfloat16)
  return torch.from_numpy(np.asarray(array))


class Placement:
  """The one chip, its HBM and its pinned host memory."""

  def __init__(self):
    devices = jax.devices()
    if len(devices) != 1:
      print(f"warning: {len(devices)} chips visible; set TPU_VISIBLE_CHIPS")
    self.device = devices[0]
    self.mesh = jax.make_mesh(
        (1,), ("tp",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    self.hbm = jax.sharding.SingleDeviceSharding(
        self.device, memory_kind="device"
    )
    self.host = jax.sharding.SingleDeviceSharding(
        self.device, memory_kind="pinned_host"
    )
    self.env = torchax.default_env()


def rotate_half_matrix(rotary_dim: int, head_dim: int, dtype) -> torch.Tensor:
  """R with `x @ R == cat(-x2, x1, 0)`, x1/x2 the halves of x[:rotary_dim].

  One nonzero (+-1) per column, so the bf16 matmul is exact.
  """
  half = rotary_dim // 2
  r = torch.zeros(head_dim, head_dim, dtype=dtype)
  idx = torch.arange(half)
  r[idx + half, idx] = -1  # out[:half] = -x2
  r[idx, idx + half] = 1  # out[half:rotary_dim] = x1
  return r


def full_lane_rope(x, cos, sin, rotate):
  """`_apply_rotary_emb` over all 128 lanes instead of the leading 96.

  The reference slices the first `rotary_dim` channels, chunks and
  concatenates them, none of it lane-aligned; XLA spent ~15 ms per block on
  the relayouts. Padding the tables (cos with 1, sin with 0) to the full
  head and rotating with a signed permutation matmul keeps every op on
  whole 128-lane vregs, with the same bf16 arithmetic.
  """
  pad = x.shape[-1] - cos.shape[-1]
  cos = torch.nn.functional.pad(cos.to(x.dtype), (0, pad), value=1.0)
  sin = torch.nn.functional.pad(sin.to(x.dtype), (0, pad), value=0.0)
  cos, sin = cos[None, :, None, :], sin[None, :, None, :]
  return x * cos + (x @ rotate) * sin


def bind_attention(attn, sdpa, rope: str = "full", fused_qkv: bool = False,
                   qk_barrier: bool = True) -> None:
  """Routes one `MiniMaxH3Attention` through the runtime's splash kernel.

  `rope="full"` uses `full_lane_rope`, `"reference"` diffusers' own.
  `fused_qkv` runs q/k/v as one matmul over the concatenated weights.
  `qk_barrier` materializes the q/k projections before their norm and rope.
  """
  from diffusers.models.transformers.transformer_minimax_h3 import (
      _apply_rotary_emb,
  )

  def forward(hidden_states, rotary_emb=None, attention_mask=None):
    del attention_mask  # MiniMax-H3 never passes one.
    if fused_qkv:
      weight = torch.cat(
          [attn.to_q.weight, attn.to_k.weight, attn.to_v.weight]
      )
      query, key, value = (hidden_states @ weight.T).unflatten(
          -1, (3 * attn.heads, -1)
      ).chunk(3, dim=-2)
    else:
      query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
      key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
      value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
    if qk_barrier:
      query, key = barrier(query), barrier(key)
    query, key = attn.norm_q(query), attn.norm_k(key)
    if rotary_emb is not None and rope == "full":
      cos, sin = rotary_emb
      rotate = rotate_half_matrix(
          cos.shape[-1], query.shape[-1], query.dtype
      ).to(query.device)
      query = full_lane_rope(query, cos, sin, rotate)
      key = full_lane_rope(key, cos, sin, rotate)
    elif rotary_emb is not None:
      query = _apply_rotary_emb(query, *rotary_emb)
      key = _apply_rotary_emb(key, *rotary_emb)
    # The kernel wants (batch, heads, seq, dim); the model keeps the heads
    # after the sequence.
    out = sdpa(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    )
    out = out.transpose(1, 2).flatten(2, 3).type_as(query)
    return attn.to_out[1](attn.to_out[0](out))

  attn.forward = forward


def barrier(tensor: torch.Tensor) -> torch.Tensor:
  """Materializes `tensor`: XLA may not fuse across it.

  Without it XLA fuses the norm, the modulation gather and the swiglu into
  the operand prologue / epilogue of the adjacent matmul, which then
  recomputes them per tile: the FFN's two matmuls ran at under 50% MFU.
  """
  env = torchax.default_env()
  return env.j2t_iso(jax.lax.optimization_barrier(env.t2j_iso(tensor)))


class Segments:
  """The runs of equal `adaln_indices` along the packed sequence.

  `bounds` is a static tuple of `(start, end)` set by the runner while it
  traces (the packing is `[text | audio | video]`, the same every step);
  None falls back to diffusers' per-row gather.
  """

  bounds: tuple[tuple[int, int], ...] | None = None

  @staticmethod
  def of(adaln_indices: torch.Tensor) -> tuple[tuple[int, int], ...]:
    _, counts = torch.unique_consecutive(adaln_indices, return_counts=True)
    ends = counts.cumsum(0).tolist()
    return tuple(zip([0] + ends[:-1], ends))


class Sparsity:
  """Temporal-window block sparsity for the transformer's self-attention.

  Output-changing: each video row attends to the text and audio rows and
  to the video rows within `window` latent frames of it (at the kernel's
  block granularity, so a little more); the text and audio rows still
  attend to everything. The first `dense_steps` steps of a request and the
  layers in `dense_layers` stay dense. `head_windows`, a (layers, heads)
  array, replaces `window` with one window per head.
  With `early_windows`, only the first `full_steps` steps are fully dense
  and the rest of the first `dense_steps` use those (wider) per-head windows.
  """

  def __init__(self, window: int, dense_steps: int = 0,
               dense_layers: tuple[int, ...] = (), head_windows=None,
               early_windows=None, full_steps: int = 2):
    self.window, self.dense_steps = window, dense_steps
    self.dense_layers = frozenset(dense_layers)
    self.head_windows = head_windows
    self.early_windows, self.full_steps = early_windows, full_steps

  def phase(self, step):
    """None (dense), "early" or "late" for a request's `step`-th call."""
    if step >= self.dense_steps:
      return "late"
    if self.early_windows is not None and step >= self.full_steps:
      return "early"
    return None

  def layer_window(self, layer, phase="late"):
    """`window`, or this layer's per-head windows as a tuple."""
    windows = self.early_windows if phase == "early" else self.head_windows
    if windows is None:
      return self.window
    return tuple(int(w) for w in windows[layer])

  @staticmethod
  def table(seq, video_start, rows_per_frame, window, block_q, block_kv):
    """Each q block's KV blocks, ascending."""
    reach = window * rows_per_frame
    num_q, num_kv = -(-seq // block_q), -(-seq // block_kv)
    rows = []
    for i in range(num_q):
      q0, q1 = i * block_q, min((i + 1) * block_q, seq)
      rows.append([
          j for j in range(num_kv)
          if q0 < video_start or j * block_kv < video_start
          or (j * block_kv < q1 + reach and min((j + 1) * block_kv, seq) > q0 - reach)
      ])
    return rows


# The sparse layout of the call being traced, read by `_attention`.
_SPARSE: dict = {"layout": None}


def _attention(query, key, value, *, scale, block_q, block_kv, **kw):
  """`custom_splash_attention.attention`, block-sparse while `_SPARSE` says so."""
  from ...kernels import splash as csa

  layout = _SPARSE["layout"]
  if layout is None:
    return _dense_attention(query, key, value, scale=scale, block_q=block_q,
                            block_kv=block_kv, **kw)
  from ...kernels import sparse_splash

  seq = query.shape[2]
  video_start, rows_per_frame, window = layout
  blocks = min(block_q, seq), min(block_kv, seq)
  if isinstance(window, tuple):
    classes = sorted(set(window))
    table = ([Sparsity.table(seq, video_start, rows_per_frame, w, *blocks)
              for w in classes], [classes.index(w) for w in window])
  else:
    table = Sparsity.table(seq, video_start, rows_per_frame, window, *blocks)
  return sparse_splash.attention(query, key, value, scale=scale,
                                 block_q=block_q, block_kv=block_kv,
                                 kv_table=table, **kw)


_dense_attention = None


def bind_block(block, points: frozenset[str] = frozenset({"attn", "ff", "ff2"}),
               segments: Segments | None = None):
  """`MiniMaxH3TransformerBlock.forward` with barriers at `points`.

  Same arithmetic as diffusers' forward; `points` picks which matmul inputs
  are materialized: `attn` (the q/k/v input), `ff` (the FFN input), `ff2`
  (the swiglu output).

  With `segments` (and its `bounds` set), the AdaLN modulation is applied
  per run instead of through six `index_select`s, each of which
  materializes a full (seq, hidden) float32 tensor: the longest run's
  coefficients broadcast over the whole sequence, then the other runs'
  rows are rewritten. Same per-row arithmetic.
  """
  cut = lambda name, t: barrier(t) if name in points else t
  proj, out = block.ff.net[0].proj, block.ff.net[2]

  def per_run(adaln_indices, fn, *tables):
    """`fn(rows, *coefficients)` with each row's own coefficients."""
    bounds = segments.bounds if segments is not None else None
    if bounds is None:
      return fn(slice(None), *(t.index_select(0, adaln_indices) for t in tables))
    rows = torch.stack([adaln_indices[start] for start, _ in bounds])
    coeffs = [t.index_select(0, rows) for t in tables]
    base = max(range(len(bounds)), key=lambda i: bounds[i][1] - bounds[i][0])
    result = fn(slice(None), *(c[base] for c in coeffs))
    for i, (start, end) in enumerate(bounds):
      if i != base:
        result[:, start:end] = fn(slice(start, end), *(c[i] for c in coeffs))
    return result

  def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    del attention_mask
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.adaln_proj(temb)
    )
    norm = block.norm1(hidden_states)
    x = per_run(
        adaln_indices,
        lambda r, scale, shift: norm[:, r] * (1.0 + scale) + shift,
        scale_msa, shift_msa,
    )
    attn = block.attn(cut("attn", x), rotary_emb)
    residual = hidden_states
    hidden_states = per_run(
        adaln_indices,
        lambda r, gate: residual[:, r] + gate * attn[:, r],
        gate_msa,
    )
    norm = block.norm2(hidden_states)
    x = per_run(
        adaln_indices,
        lambda r, scale, shift: norm[:, r] * (1.0 + scale) + shift,
        scale_mlp, shift_mlp,
    )
    x, gate = proj(cut("ff", x)).chunk(2, dim=-1)
    x = out(cut("ff2", x * torch.nn.functional.silu(gate)))
    residual = hidden_states
    return per_run(
        adaln_indices,
        lambda r, gate: residual[:, r] + gate * x[:, r],
        gate_mlp,
    )

  block.forward = forward


def run_streamed(modules, streamed, hidden, call, hbm, lockstep=False):
  """`hidden = call(module, params, hidden)` over a stack of host-resident layers.

  Each layer's weights are `device_put` to HBM right before the layer; XLA
  schedules the copies to overlap the preceding layers' compute, as far
  ahead as its memory budget allows (see `scheduler_options`).

  `lockstep` instead routes layer i+1's host weights through an
  optimization barrier with layer i's input. XLA cannot keep a barrier
  operand in host memory, so it lands the copy before the barrier: layer
  i+1's weights are in HBM by the time layer i starts, and copies cannot
  run further ahead than that. For the transformer this is 3.5% faster
  (7.76 vs 8.04 s/step) than letting the scheduler place the copies.
  """
  env = torchax.default_env()
  if not lockstep:
    for module, host_params in zip(modules, streamed):
      params = {k: env.j2t_iso(jax.device_put(v, hbm)) for k, v in host_params.items()}
      hidden = call(module, params, hidden)
    return hidden
  ready = jax.device_put(streamed[0], hbm)
  for i, module in enumerate(modules):
    params = ready
    if i + 1 < len(modules):
      h, upcoming = jax.lax.optimization_barrier(
          (env.t2j_iso(hidden), streamed[i + 1])
      )
      hidden = env.j2t_iso(h)
      ready = jax.device_put(upcoming, hbm)
    hidden = call(module, {k: env.j2t_iso(v) for k, v in params.items()}, hidden)
  return hidden


def scheduler_options(percent: int | None) -> dict[str, str]:
  """Caps the share of HBM XLA's scheduler may plan into, `None` for all.

  The scheduler hoists the streamed weight copies as early as its budget
  allows, and the budget is the whole chip: uncapped, the text encoder
  plans 32 GB of prefetched layers and cannot load once the transformer
  and VAE are resident.
  """
  if percent is None:
    return {}
  return {"xla_tpu_scheduler_percent_shared_memory_limit": str(percent)}


class TransformerRunner:
  """`MiniMaxH3Transformer3DModel.forward` on one chip, blocks streamed."""

  def __init__(
      self,
      model,
      placement: Placement,
      use_fixed_m: bool = True,
      barriers: frozenset[str] | None = frozenset({"ff"}),
      memory_percent: int | None = None,
      baseline: bool = False,
      segmented: bool = True,
  ):
    """`baseline` turns off lockstep streaming, the full-lane rope and the
    q/k barrier, for before/after measurements; `segmented` picks the
    per-run AdaLN modulation (see `bind_block`)."""
    from ...kernels import mesh_attention as attention

    self.lockstep = not baseline
    self.sparsity: Sparsity | None = None
    self.request_steps = 0
    self._last_output = None
    self.reuse_steps: frozenset[int] = frozenset()  # calls that reuse the last output
    global _dense_attention
    if _dense_attention is None:
      from ...kernels import splash as csa

      _dense_attention, csa.attention = csa.attention, _attention
    self.segments = Segments() if segmented else None
    self._bounds = None
    self.model = model
    self.p = placement
    self.blocks = list(model.transformer_blocks)
    self.calls = 0
    self.step_seconds: list[float] = []
    # Kept on the runner so a caller can re-bind attention afterwards
    # and vary `rope` / `qk_barrier` on their own.
    sdpa = self.sdpa = attention.configure(
        placement.mesh,
        placement.env,
        **BLOCKS,
        use_fixed_m=use_fixed_m,
    )
    # Only the block stack goes through the splash kernel; the refiner's
    # few hundred text rows are left to torchax's own SDPA.
    for block in self.blocks:
      if baseline:
        bind_attention(block.attn, sdpa, rope="reference", qk_barrier=False)
      else:
        bind_attention(block.attn, sdpa)
      if barriers is not None:
        bind_block(block, barriers, self.segments)

    start = time.perf_counter()
    resident, streamed = {}, []
    for name, tensor in model.named_parameters():
      if not name.startswith("transformer_blocks."):
        resident[name] = jax.device_put(to_jax(tensor), placement.hbm)
    for buffer_name, tensor in model.named_buffers():
      resident[buffer_name] = jax.device_put(to_jax(tensor), placement.hbm)
    for block in self.blocks:
      streamed.append({
          name: jax.device_put(to_jax(tensor), placement.host)
          for name, tensor in block.named_parameters()
      })
    self.resident = resident
    self.streamed = streamed
    gb = sum(v.size * v.dtype.itemsize for b in streamed for v in b.values())
    print(
        f"[transformer] staged in {time.perf_counter() - start:.0f}s:"
        f" {len(resident)} resident, {len(streamed)} blocks"
        f" ({gb/1e9:.1f} GB) in pinned host"
    )
    self._step = self.jit(memory_percent)
    model.forward = self.forward

  def jit(self, memory_percent: int | None = None):
    return jax.jit(
        self._forward,
        static_argnums=(3, 4),
        compiler_options=scheduler_options(memory_percent),
    )

  # The traced body. `resident` and `streamed` are jax pytrees; everything
  # torch-side is the model's own modules called functionally.
  def _forward(self, resident, streamed, inputs, bounds, sparse=None):
    if self.segments is not None:
      self.segments.bounds = bounds
    env, model, hbm = self.p.env, self.model, self.p.hbm
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
    )

    def sub(prefix):
      return {
          k[len(prefix):]: env.j2t_iso(v)
          for k, v in resident.items()
          if k.startswith(prefix)
      }

    def call(module, prefix, *args):
      return torch.func.functional_call(module, sub(prefix), args=args)

    with env:
      t = {k: env.j2t_iso(v) for k, v in inputs.items()}
      rotary = call(model.rope, "rope.", t["position_ids"])
      video = call(model.proj_in, "proj_in.", t["hidden_states"].float())
      audio = call(
          model.audio_proj_in, "audio_proj_in.", t["audio_hidden_states"].float()
      )
      text = call(
          model.context_embedder,
          "context_embedder.",
          t["encoder_hidden_states"].to(torch.bfloat16),
      )
      text = call(model.token_refiner, "token_refiner.", text)
      seq = t["position_ids"].shape[0]
      hidden = text.new_zeros((text.shape[0], seq, text.shape[-1]))
      hidden = hidden.index_copy(1, t["text_indices"], text)
      hidden = hidden.index_copy(1, t["video_indices"], video.to(text.dtype))
      hidden = hidden.index_copy(1, t["audio_indices"], audio.to(text.dtype))

      temb = call(model.time_proj, "time_proj.", t["timestep"])
      temb = call(model.time_embedder, "time_embedder.", temb.float())
      adaln_indices = (
          t["timestep_indices"] * MINIMAX_H3_MODALITY_NUM + t["token_tags"]
      )

      # DT_DUMP_BLOCKS=<dir>: dump the same per-block probes, in the same
      # format, as the native `jax_dit` (`jax_dit._blk_save`), for a
      # layer-by-layer comparison of the two implementations.
      _dump = bool(os.environ.get("DT_DUMP_BLOCKS"))
      if _dump:
        import functools as _ft
        from .jax_dit import _blk_save
        jax.debug.callback(_ft.partial(_blk_save, "x0"), env.t2j_iso(hidden))
        jax.debug.callback(_ft.partial(_blk_save, "temb"), env.t2j_iso(temb))

      def _block(block, params, h):
        out = self._call_block(
            block, params, (h, temb, adaln_indices, rotary), sparse)
        i = self.blocks.index(block)
        if _dump and i in (0, 4, 9, 19, 29, 39, 49):
          jax.debug.callback(_ft.partial(_blk_save, f"b{i:02d}"),
                             env.t2j_iso(out))
        return out

      hidden = run_streamed(
          self.blocks,
          streamed,
          hidden,
          _block,
          hbm,
          lockstep=self.lockstep,
      )

      hidden = call(
          model.norm_out, "norm_out.", hidden, temb, t["timestep_indices"]
      ).float()
      if _dump:
        jax.debug.callback(_ft.partial(_blk_save, "normout"),
                           env.t2j_iso(hidden))
      video_out = call(model.proj_out, "proj_out.", hidden).index_select(
          1, t["video_indices"]
      )
      audio_out = call(
          model.audio_proj_out, "audio_proj_out.", hidden
      ).index_select(1, t["audio_indices"])
    return env.t2j_iso(video_out), env.t2j_iso(audio_out)

  def _call_block(self, block, params, args, sparse):
    """One block, with `sparse` (a layout or None) set while it traces."""
    layer = self.blocks.index(block)
    dense = sparse is None or layer in self.sparsity.dense_layers
    _SPARSE["layout"] = None if dense else (
        *sparse[:2], self.sparsity.layer_window(layer, sparse[2]))
    try:
      return torch.func.functional_call(block, params, args=args)
    finally:
      _SPARSE["layout"] = None

  def _velocity_stats(self, video):
    """Prints how far the video velocity moved since the previous call
    (relative L2, cosine): the headroom for reusing it across steps."""
    v = video.astype(jnp.float32)
    prev, self._prev_velocity = getattr(self, "_prev_velocity", None), v
    if prev is not None and prev.shape == v.shape:
      rel = float(jnp.linalg.norm(v - prev) / jnp.linalg.norm(v))
      cos = float((v * prev).sum() / jnp.linalg.norm(v) / jnp.linalg.norm(prev))
      print(f"[vstats] call {self.calls}: rel {rel:.4f} cos {cos:.4f}", flush=True)

  def begin_request(self):
    """Resets the step count `Sparsity.dense_steps` is measured in."""
    self.request_steps = 0
    self._last_output = None

  def forward(
      self,
      hidden_states,
      audio_hidden_states,
      encoder_hidden_states,
      timestep,
      timestep_indices,
      token_tags,
      position_ids,
      video_indices,
      audio_indices,
      text_indices,
      attention_kwargs=None,
      return_dict=True,
  ):
    del attention_kwargs
    if self.request_steps in self.reuse_steps and self._last_output is not None:
      # Output-changing: hand back the previous call's velocities (late
      # steps move them ~5% a step, see H3_VSTATS).
      self.request_steps += 1
      video, audio = self._last_output
      if not return_dict:
        return video, audio
      from diffusers.models.transformers.transformer_minimax_h3 import (
          MiniMaxH3TransformerOutput,
      )

      return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)
    dtypes = (hidden_states.dtype, audio_hidden_states.dtype)
    inputs, bounds, sparse = self.prepare(
        hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
        timestep_indices, token_tags, position_ids, video_indices,
        audio_indices, text_indices,
    )
    self.request_steps += 1
    start = time.perf_counter()
    with jax.set_mesh(self.p.mesh):
      video, audio = jax.block_until_ready(
          self._step(self.resident, self.streamed, inputs, bounds, sparse)
      )
    dt = time.perf_counter() - start
    self.step_seconds.append(dt)
    self.calls += 1
    if self.calls <= 3 or self.calls % 10 == 0:
      print(f"[transformer] call {self.calls}: {dt:.2f}s", flush=True)
    if os.environ.get("H3_VSTATS"):
      self._velocity_stats(video)
    video = to_torch(video).to(dtypes[0])
    audio = to_torch(audio).to(dtypes[1])
    self._last_output = video, audio
    if not return_dict:
      return video, audio
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerOutput,
    )

    return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)

  def prepare(
      self,
      hidden_states,
      audio_hidden_states,
      encoder_hidden_states,
      timestep,
      timestep_indices,
      token_tags,
      position_ids,
      video_indices,
      audio_indices,
      text_indices,
  ):
    """The step's inputs on device, and the static modulation runs.

    The pipeline passes each step's *unique* timesteps, 1 on the first step
    (video and audio coincide) and 2 after: two shapes, each compiled once
    (warm up with >= 3 steps). Padding the count instead would save the
    compile but changes the output.
    """
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
    )

    bounds = None
    if self.segments is not None:
      bounds = Segments.of(
          timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
      )
      if bounds != self._bounds:
        print(f"[transformer] modulation runs: {[e - s for s, e in bounds]}")
        self._bounds = bounds
    inputs = {
        "hidden_states": hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep.float(),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }
    inputs = {
        k: jax.device_put(to_jax(v.cpu()), self.p.hbm) for k, v in inputs.items()
    }
    sparse = None
    phase = self.sparsity and self.sparsity.phase(self.request_steps)
    if phase:
      # Video rows are the trailing block of the packing, frame-major.
      target_vi = video_indices
      if audio_indices is not None and audio_indices.numel() > 0:
        after_audio = video_indices[video_indices > audio_indices.max()]
        if after_audio.numel() > 0:
          target_vi = after_audio
      video_start = int(target_vi.min())
      assert int(target_vi.max()) == position_ids.shape[0] - 1
      times = position_ids[target_vi, 0]
      rows_per_frame = int((times == times[0]).sum())
      sparse = (video_start, rows_per_frame, phase)
    return inputs, bounds, sparse


class TextEncoderRunner:
  """`Qwen3VLModel.forward` on one chip, for text-only presentations.

  MiniMax-H3 reads `hidden_states[layer]` and nothing after it, so only the
  first `layer` decoder layers run (about 49 GB, streamed from pinned host
  like the transformer's blocks); the embedding lookup and the rotary table
  are cheap and stay on the CPU. The prompt is padded at the end to a
  multiple of `pad_to`: under the causal mask the real rows never see the
  padding, so the result is exact and one compile serves every prompt of
  the same bucket.
  """

  def __init__(self, text_encoder, placement: Placement, layer: int = 50,
               pad_to: int = 128, memory_percent: int | None = 20):
    self.model = text_encoder.model
    self.lm = self.model.language_model
    self.p = placement
    self.layer = layer
    self.pad_to = pad_to
    self.seconds: list[float] = []
    # Eager attention with an explicit causal mask: SDPA's `is_causal` and
    # GQA flags are not something to rely on through torchax.
    self.lm.config._attn_implementation = "eager"
    self.layers = list(self.lm.layers[:layer])
    start = time.perf_counter()
    self.streamed = [
        {
            name: jax.device_put(to_jax(tensor), placement.host)
            for name, tensor in layer_module.named_parameters()
        }
        for layer_module in self.layers
    ]
    gb = sum(v.size * v.dtype.itemsize for b in self.streamed for v in b.values())
    print(
        f"[text_encoder] staged in {time.perf_counter() - start:.0f}s:"
        f" {len(self.layers)} layers ({gb/1e9:.1f} GB) in pinned host"
    )
    self._step = jax.jit(
        self._forward, compiler_options=scheduler_options(memory_percent)
    )
    self.model.forward = self.forward

  def _forward(self, streamed, hidden, cos, sin):
    env = self.p.env
    length = hidden.shape[1]
    neg = jnp.finfo(hidden.dtype).min
    mask = jnp.where(
        jnp.tril(jnp.ones((length, length), bool)), 0, neg
    ).astype(hidden.dtype)[None, None]
    with env:
      h = env.j2t_iso(hidden)
      rotary = (env.j2t_iso(cos), env.j2t_iso(sin))
      mask = env.j2t_iso(mask)
      h = run_streamed(
          self.layers,
          streamed,
          h,
          lambda layer, params, x: torch.func.functional_call(
              layer,
              params,
              args=(x,),
              kwargs={"position_embeddings": rotary, "attention_mask": mask},
          ),
          self.p.hbm,
      )
    return env.t2j_iso(h)

  def forward(self, input_ids, attention_mask=None, mm_token_type_ids=None,
              use_cache=False, output_hidden_states=True, **vision):
    del attention_mask, use_cache, output_hidden_states
    if vision or (mm_token_type_ids is not None and mm_token_type_ids.any()):
      raise NotImplementedError("TextEncoderRunner handles text-only prompts")
    length = input_ids.shape[1]
    padded = -(-length // self.pad_to) * self.pad_to
    ids = torch.nn.functional.pad(input_ids, (0, padded - length))
    with torch.no_grad():
      hidden = self.lm.embed_tokens(ids)
      # Text-only mrope: all three position axes are the token index.
      positions = torch.arange(padded).view(1, 1, -1).expand(3, 1, -1)
      cos, sin = self.lm.rotary_emb(hidden, positions)
    args = [jax.device_put(to_jax(t), self.p.hbm) for t in (hidden, cos, sin)]
    start = time.perf_counter()
    with jax.set_mesh(self.p.mesh):
      out = jax.block_until_ready(self._step(self.streamed, *args))
    dt = time.perf_counter() - start
    self.seconds.append(dt)
    print(f"[text_encoder] {length} tokens (padded {padded}): {dt:.2f}s",
          flush=True)
    out = to_torch(out)[:, :length]
    from transformers.modeling_outputs import BaseModelOutputWithPast

    return BaseModelOutputWithPast(
        last_hidden_state=None,
        hidden_states=(None,) * self.layer + (out,),
    )


class _ClipDecoder(torch.nn.Module):
  """`AutoencoderKLMiniMaxH3._decode_clip`, split into device-side pieces.

  Same tiles, same blending (diffusers' own `_split_tiles` /
  `_stitch_tiles`): `split` cuts the clip's latent into its spatial tiles,
  `decode` runs one tile through the decoder, `stitch` blends the decoded
  tiles. Batching tiles through `decode` measured slower (per clip, 1 / 4 /
  14 tiles per call: 0.97 / 1.12 / 1.18 s), and one program over all of a
  clip's tiles took 188 s to compile, so the runner calls `decode` per tile.
  """

  def __init__(self, vae):
    super().__init__()
    self.post_quant_conv = vae.post_quant_conv
    self.decoder = vae.decoder
    self.vae = vae

  def _layout(self, z):
    vae, ratio = self.vae, self.vae.spatial_compression_ratio
    rows = vae._split_tiles(
        z.shape[-2] * ratio, vae.tile_sample_min_height,
        vae.tile_sample_min_overlap_height,
    )
    cols = vae._split_tiles(
        z.shape[-1] * ratio, vae.tile_sample_min_width,
        vae.tile_sample_min_overlap_width,
    )
    return rows, cols

  def split(self, z):
    ratio = self.vae.spatial_compression_ratio
    (ys, y_len, _), (xs, x_len, _) = self._layout(z)
    return torch.cat([
        z[..., y // ratio:(y + h) // ratio, x // ratio:(x + w) // ratio]
        for y, h in zip(ys, y_len)
        for x, w in zip(xs, x_len)
    ])

  def forward(self, tile):
    return type(self.decoder).forward(self.decoder, self.post_quant_conv(tile))

  def stitch(self, z, decoded):
    (ys, _, y_overlaps), (xs, _, x_overlaps) = self._layout(z)
    rows = [
        [decoded[r * len(xs) + c] for c in range(len(xs))]
        for r in range(len(ys))
    ]
    return self.vae._stitch_tiles(rows, y_overlaps, x_overlaps)


class VAEDecoderRunner:
  """The video VAE's decode on one chip, one temporal clip per call.

  Replaces `vae._decode_clip`: the clip's spatial tiles go through the ViT
  decoder (36 layers, ~3.6 GB, resident in HBM) one at a time and are
  blended on device; only the finished clip comes back. The temporal
  chunking and cross-fade stay in diffusers on the CPU.

  The checkpoint is float32 (the reference recipe is float16 autocast on
  CUDA); here it runs in `dtype`, the norms inside staying float32 as the
  module itself does.
  """

  def __init__(self, vae, placement: Placement, dtype=torch.bfloat16):
    if not vae.use_tiling:
      raise NotImplementedError("VAEDecoderRunner expects tiled decoding")
    self.p = placement
    self.dtype = dtype
    self.calls = 0
    self.seconds = 0.0
    self.module = _ClipDecoder(vae)
    self.params = {
        name: jax.device_put(to_jax(tensor.to(dtype)), placement.hbm)
        for name, tensor in self.module.named_parameters()
    }
    # The rope's inverse frequencies, kept in their own (float32) dtype.
    self.params.update({
        name: jax.device_put(to_jax(tensor), placement.hbm)
        for name, tensor in self.module.named_buffers()
    })
    gb = sum(v.size * v.dtype.itemsize for v in self.params.values())
    print(f"[vae] decoder resident: {gb/1e9:.1f} GB")
    env = placement.env

    def split(z):
      with env:
        return env.t2j_iso(self.module.split(env.j2t_iso(z)))

    def decode(params, tiles, i):
      with env:
        return env.t2j_iso(torch.func.functional_call(
            self.module,
            {k: env.j2t_iso(v) for k, v in params.items()},
            args=(env.j2t_iso(jax.lax.dynamic_slice_in_dim(tiles, i, 1)),),
        ))

    def stitch(z, decoded):
      with env:
        return env.t2j_iso(self.module.stitch(
            env.j2t_iso(z), [env.j2t_iso(d) for d in decoded]
        ))

    self._split, self._decode, self._stitch = map(
        jax.jit, (split, decode, stitch)
    )
    vae._decode_clip = self.forward

  def forward(self, z):
    dtype = z.dtype
    z = jax.device_put(to_jax(z.to(self.dtype)), self.p.hbm)
    start = time.perf_counter()
    with jax.set_mesh(self.p.mesh):
      tiles = self._split(z)
      decoded = [
          self._decode(self.params, tiles, i) for i in range(len(tiles))
      ]
      out = jax.block_until_ready(self._stitch(z, decoded))
    self.seconds += time.perf_counter() - start
    self.calls += 1
    return to_torch(out).to(dtype)
