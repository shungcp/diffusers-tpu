#!/usr/bin/env python3
"""Qwen3-VL's decoder stack on the mesh, for `ref2va`.

`t2va` presents ten text tokens and the encoder is a rounding error. A
`ref2va` request turns its reference image into about seven thousand
tokens through the same model, and on the host that measured **67.2 s of
a 113.5 s request** -- the single largest item, four times the denoise.

MiniMax-H3 reads `hidden_states[50]` and nothing after it, so only the
first fifty of the sixty-four layers run.

The weights (48.8 GB for those fifty layers) are sharded over the mesh
and kept in pinned host memory rather than resident: the H3 transformer
already holds most of HBM, which leaves about 2.6 GB a chip, well under
the 6.1 GB a resident shard would need. Each layer is copied in just
before it runs and the scheduler overlaps the copy with the previous
layer's compute.

The split divides cleanly on eight chips: 64 query heads, 8 key/value
heads, 25600 of intermediate.

Qwen3-VL injects visual features into the first few layers
("deepstack"), which is why this cannot reuse the one-chip text-only
runner (`h3_tpu.TextEncoderRunner`), which refuses image prompts.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import os
import torch

# Qwen3-VL decoder layer, by parameter-name suffix. Attention splits on
# heads, the MLP on its intermediate width; `o_proj` and `down_proj`
# contract over the split axis and end in one all-reduce each.
SHARDINGS = (
    (".self_attn.q_proj.weight", P("tp", None)),
    (".self_attn.k_proj.weight", P("tp", None)),
    (".self_attn.v_proj.weight", P("tp", None)),
    (".self_attn.o_proj.weight", P(None, "tp")),
    (".mlp.gate_proj.weight", P("tp", None)),
    (".mlp.up_proj.weight", P("tp", None)),
    (".mlp.down_proj.weight", P(None, "tp")),
)


def spec_for(name: str) -> P:
  """Replicated, deliberately.

  Tensor parallel looked right -- 64 query heads and 8 key/value heads
  both divide eight chips -- but Qwen3-VL reshapes with
  `view(batch, seq, -1, head_dim)`, and under SPMD that reshape sees the
  *global* head count and all-gathers the shards straight back. The
  split bought nothing and the scores were still formed over all 64
  heads.

  Replicated instead: the whole fifty-layer forward is about 350 TFLOP,
  roughly a second on one chip, and every chip doing it redundantly
  costs no wall time. The weights stream per layer either way.

  `SHARDINGS` records the tensor-parallel layout this replaced.
  """
  del name
  return P()


def _to_jax(tensor: torch.Tensor) -> jax.Array:
  # Already a torchax tensor? Then `view(torch.uint16)` below is read as
  # a *reshape* with a dtype for a shape -- "Shapes must be 1D sequences
  # of concrete values of integer type, got [torch.uint16]". Hand it
  # straight to the environment instead.
  import torchax

  if isinstance(tensor, torchax.tensor.Tensor):
    return torchax.default_env().t2j_iso(tensor)
  tensor = tensor.detach().contiguous()
  if tensor.dtype == torch.bfloat16:
    return jnp.asarray(tensor.view(torch.uint16).numpy().view(jnp.bfloat16))
  if tensor.is_floating_point():
    return jnp.asarray(tensor.float().numpy())
  return jnp.asarray(tensor.numpy())


def _splash_causal(module, query, key, value, attention_mask, **kwargs):
  """A `transformers` attention implementation backed by splash attention.

  Registered as `"splash"` and selected through
  `config._attn_implementation`, so the model's own layer code is
  untouched.

  Qwen3-VL's own path materialises the scores, and at 7186 tokens that
  is `64 x 7186^2 x 4 B = 13.2 GB` -- more than a chip has free with the
  H3 transformer resident. Splash never forms them. Our own kernel has
  no causal mask (H3 packs one document and needs none), so this uses
  the stock JAX one with `CausalMask`.
  """
  import torchax
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_kernel as kernel,
  )
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_mask as masks,
  )

  del attention_mask, kwargs  # causal is in the mask object
  env = torchax.default_env()
  heads = query.shape[1]
  groups = heads // key.shape[1]
  if groups > 1:  # grouped-query: 64 query heads over 8 key/value heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)

  q, k, v = (env.t2j_iso(t) for t in (query, key, value))
  length = q.shape[2]
  # The kernel wants whole blocks; the causal mask keeps the padding
  # from reaching any real row.
  block = 1024
  padded = -(-length // block) * block
  pad = padded - length
  if pad:
    width = ((0, 0), (0, 0), (0, pad), (0, 0))
    q, k, v = (jnp.pad(x, width) for x in (q, k, v))

  mask = masks.MultiHeadMask(
      [masks.CausalMask((padded, padded)) for _ in range(heads)]
  )
  splash = kernel.make_splash_mha(
      mask,
      block_sizes=kernel.BlockSizes(
          block_q=block, block_kv=block, block_kv_compute=block
      ),
      head_shards=1,
      q_seq_shards=1,
  )
  scale = module.scaling
  out = jax.vmap(splash)(q * scale, k, v)[:, :, :length]
  return env.j2t_iso(out).transpose(1, 2), None


def _splash_full(module, query, key, value, attention_mask, **kwargs):
  """Splash attention with no mask, for the vision tower.

  28,744 patches over 16 heads is a 26.4 GB score matrix in bfloat16 --
  exactly what the chip refused. Splash never forms it.

  `head_dim` is 72 here, not a multiple of the 128-lane tile, so q, k
  and v are zero-padded to 128: the extra channels contribute nothing to
  `q . k` because both sides are zero, and the output is sliced back.
  """
  import torchax
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_kernel as kernel,
  )
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_mask as masks,
  )

  del attention_mask, kwargs  # one image, one segment, nothing to mask
  env = torchax.default_env()
  q, k, v = (env.t2j_iso(t) for t in (query, key, value))
  heads, length, dim = q.shape[1], q.shape[2], q.shape[3]

  lanes = -(-dim // 128) * 128
  if lanes != dim:
    pad = ((0, 0), (0, 0), (0, 0), (0, lanes - dim))
    q, k, v = (jnp.pad(x, pad) for x in (q, k, v))
  block = 1024
  padded = -(-length // block) * block
  if padded != length:
    pad = ((0, 0), (0, 0), (0, padded - length), (0, 0))
    q, k, v = (jnp.pad(x, pad) for x in (q, k, v))

  # Padded keys are masked out by segment: zero keys still score 0 and
  # would otherwise take a share of every softmax.
  ids = jnp.concatenate([jnp.zeros((length,), jnp.int32),
                         jnp.ones((padded - length,), jnp.int32)])
  segments = kernel.SegmentIds(q=ids, kv=ids)

  def local(q_, k_, v_):
    mask = masks.MultiHeadMask(
        [masks.FullMask((padded, padded)) for _ in range(q_.shape[1])]
    )
    splash = kernel.make_splash_mha(
        mask,
        block_sizes=kernel.BlockSizes(
            block_q=block, block_kv=block, block_kv_compute=block
        ),
        head_shards=1,
        q_seq_shards=1,
    )
    return jax.vmap(lambda a, b, c: splash(a, b, c, segment_ids=segments))(q_, k_, v_)

  q = q * module.scaling
  if VISION_MESH is not None and heads % VISION_MESH.devices.size == 0:
    # Heads split over the mesh (16 heads, 2 a chip): each chip runs its
    # own heads' attention on the replicated sequence.
    spec = P(None, "tp", None, None)
    out = jax.shard_map(local, mesh=VISION_MESH, in_specs=(spec, spec, spec),
                        out_specs=spec, check_vma=False)(q, k, v)
  else:
    out = local(q, k, v)
  out = out[:, :, :length, :dim]
  return env.j2t_iso(out).transpose(1, 2), None


VISION_MESH = None


def register_splash_attention() -> None:
  """Makes `"splash"` selectable as `config._attn_implementation`."""
  from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

  ALL_ATTENTION_FUNCTIONS["splash"] = _splash_causal
  ALL_ATTENTION_FUNCTIONS["splash_full"] = _splash_full


def _TEXT_TP_SPECS(name: str) -> P:
  """Where each of a Qwen3-VL decoder layer's weights is cut.

  The heads split on `q/k/v` and the contraction splits on `o`/`down`,
  the ordinary one-dimensional tensor-parallel layout. 64 query heads
  and 8 key/value heads both divide by eight. The norms are tiny and
  stay whole.
  """
  heads = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
  contracted = ("o_proj", "down_proj")
  if any(f"{p}.weight" in name for p in heads):
    return P("tp", None)
  if any(f"{p}.weight" in name for p in contracted):
    return P(None, "tp")
  return P()  # layernorms, q_norm, k_norm


def _psum_outputs(layers) -> None:
  """Sums `o_proj` and `down_proj` over the mesh, inside the shard_map.

  Their contraction is split, so each chip holds a partial sum. The
  residual add that follows is on replicated activations, so the sum has
  to land before it -- summing the layer's output instead would add the
  residual eight times.
  """
  import torchax

  def wrap(linear):
    inner = linear.forward

    def forward(hidden):
      env = torchax.default_env()
      out = env.t2j_iso(inner(hidden))
      # In float32. The one-chip path contracts 8192 terms inside the
      # matmul's own accumulator; sharded, the last eight of those are a
      # separate bf16 add, and that extra rounding came out at 1.6 ulp a
      # layer -- harmless once, 3.3% after fifty.
      summed = jax.lax.psum(out.astype(jnp.float32), "tp")
      return env.j2t_iso(summed.astype(out.dtype))

    linear.forward = forward

  for layer in layers:
    # Once. The patch is permanent, so a process that has installed the
    # sharded stack cannot then run the one-chip one -- the psum would
    # be outside any shard_map.
    for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
      if not getattr(linear, "_psummed", False):
        wrap(linear)
        linear._psummed = True


class TextModelOnMesh:
  """Replaces `Qwen3VLTextModel.forward` with a sharded, jitted one."""

  def __init__(self, text_model, mesh: jax.sharding.Mesh, layers: int = 50,
               resident: int = 0, tensor_parallel: bool = False):
    import torchax

    self._model = text_model
    self._mesh = mesh
    self._resident = resident
    self._tp = tensor_parallel
    self._env = torchax.default_env()
    self._layers = list(text_model.layers[:layers])
    self._depth = layers
    # Splash attention, so the score matrix is never materialized.
    register_splash_attention()
    text_model.config._attn_implementation = "splash"
    for layer in self._layers:
      layer.self_attn.config._attn_implementation = "splash"

    # A sharded array in host memory needs the memory kind on the
    # *named* sharding; routing it through a single-device host sharding
    # raises "Memory kind mismatch with xla::PjRtBuffers".
    # One chip, not the mesh. The replicated weights make the split
    # pointless, and a Pallas kernel under an eight-device mesh raises
    # "Mosaic kernels cannot be automatically partitioned".
    chips = list(jax.devices())
    chip = chips[0]
    host_kind = jax.sharding.SingleDeviceSharding(
        chip, memory_kind="pinned_host"
    )
    device_kind = jax.sharding.SingleDeviceSharding(chip, memory_kind="device")

    def host(name):
      del name
      return host_kind

    def device(name):
      del name
      return device_kind

    self._device_for = device
    start = time.perf_counter()
    if tensor_parallel:
      self._chips = None
      spec_for = _TEXT_TP_SPECS
      host_named = lambda spec: jax.sharding.NamedSharding(
          mesh, spec, memory_kind="pinned_host"
      )
      # The 1.8 s this stage costs is the host link, not arithmetic:
      # 48.8 GB at 28.3 GB/s is 1.72 s. Cut eight ways it is 6.1 GB a
      # chip. The weights still live on the host -- resident in HBM they
      # starve H3 -- only the stream splits.
      self._streamed = [
          {
              name: jax.device_put(
                  _to_jax(tensor), host_named(spec_for(name))
              )
              for name, tensor in layer.named_parameters()
          }
          for layer in self._layers
      ]
      _psum_outputs(self._layers)
    elif resident:
      # Streaming is what the stage costs: 48.8 GB at the 28.3 GB/s this
      # host sustains is 1.72 s, against 1.81 s measured. Replicated the
      # weights do not fit -- 48.8 GB against a 32 GB chip -- but dealt
      # round-robin they are 6.1 GB each, and H3's own shard is 8.25.
      # Nothing is parallel here: one chip works while seven idle. What
      # goes away is the host link.
      self._chips = [
          chips[i % len(chips)] if i < resident else None
          for i in range(layers)
      ]
      self._streamed = [
          {
              name: jax.device_put(
                  _to_jax(tensor),
                  jax.sharding.SingleDeviceSharding(
                      self._chips[index], memory_kind="device"
                  )
                  if self._chips[index] is not None
                  else host_kind,
              )
              for name, tensor in layer.named_parameters()
          }
          for index, layer in enumerate(self._layers)
      ]
    else:
      self._chips = None
      self._streamed = [
          {
              name: jax.device_put(_to_jax(tensor), host(name))
              for name, tensor in layer.named_parameters()
          }
          for layer in self._layers
      ]
    # The stack's final norm is not part of any layer, so it has to be
    # handed in as well -- left on the host it is a plain Parameter and
    # torchax refuses to mix it with its own tensors.
    # Under tensor parallelism the activation comes out of the last
    # layer replicated over the mesh, so the final norm's weights have
    # to be on the mesh too, not pinned to chip 0.
    norm_kind = (
        jax.sharding.NamedSharding(mesh, P()) if tensor_parallel
        else device_kind
    )
    self._norm = {
        name: jax.device_put(_to_jax(tensor), norm_kind)
        for name, tensor in text_model.norm.named_parameters()
    }
    total = sum(
        v.size * v.dtype.itemsize
        for layer in self._streamed for v in layer.values()
    )
    if tensor_parallel:
      where = (f"sharded by head over {len(chips)} chips, streaming "
               f"{total / len(chips) / 1e9:.1f} GB per chip")
    elif resident:
      where = (f"first {resident} layers resident round-robin in HBM of "
               f"{len(chips)} chips ("
               f"{total * resident / layers / len(chips) / 1e9:.1f} GB/chip),"
               f" rest streamed")
    else:
      where = "in pinned host"
    print(f"  [text_encoder] {layers} layers {total / 1e9:.1f} GB {where},"
          f" took {time.perf_counter() - start:.0f}s", flush=True)
    # Compiled, not eager: run op by op the first matmul alone asks
    # for a 6.25 GB program and HBM has 4.89 GB free with the H3
    # transformer already resident.
    if tensor_parallel:
      from jax.experimental.shard_map import shard_map

      replicated = P(None, None, None)
      self._compiled = jax.jit(
          shard_map(
              self._one_layer,
              mesh=mesh,
              in_specs=(
                  {name: spec_for(name) for name in self._streamed[0]},
                  replicated,            # hidden
                  P(None, None, None),   # cos
                  P(None, None, None),   # sin
                  P(None, None, None, None),  # causal mask
              ),
              out_specs=replicated,
              check_rep=False,
          )
      )
    else:
      self._compiled = jax.jit(self._one_layer)
    self._norm_fn = jax.jit(self._apply_norm)
    self._deepstack_fn = jax.jit(self._deepstack)
    self.seconds: list[float] = []
    text_model.forward = self.forward

  def _one_layer(self, params, hidden, cos, sin, mask):
    """One decoder layer, compiled once and reused by all fifty.

    The whole stack in a single program asked for 14.17 GB against
    12.99 free: eager attention materialises a 7186 x 7186 score matrix
    per layer and XLA kept several live. The layers are identical, so
    one program serves them all and the peak is a layer's worth.
    """
    env = self._env
    with env:
      h = env.j2t_iso(hidden)
      out = torch.func.functional_call(
          self._layers[0],
          {k: env.j2t_iso(v) for k, v in params.items()},
          args=(h,),
          kwargs={
              "position_embeddings": (env.j2t_iso(cos), env.j2t_iso(sin)),
              "attention_mask": env.j2t_iso(mask),
          },
      )
    return env.t2j_iso(out)

  def _apply_norm(self, norm, hidden):
    env = self._env
    with env:
      out = torch.func.functional_call(
          self._model.norm,
          {k: env.j2t_iso(v) for k, v in norm.items()},
          args=(env.j2t_iso(hidden),),
      )
    return env.t2j_iso(out)

  @staticmethod
  def _deepstack(hidden, embeds, mask):
    """`_deepstack_process`: add the visual features back at the image rows."""
    return hidden + jnp.where(mask[..., None], embeds, 0).astype(hidden.dtype)

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      use_cache=False,
      visual_pos_masks=None,
      deepstack_visual_embeds=None,
      **kwargs,
  ):
    del past_key_values, use_cache, kwargs
    from transformers.modeling_outputs import BaseModelOutputWithPast

    model = self._model
    with torch.no_grad():
      if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)
      length = inputs_embeds.shape[1]
      # Diagnostics (DT_TEXT_DIAG=1). With an image, Qwen3-VL needs 3D
      # mrope positions (time/height/width differ); the fallback below
      # fills all three axes with the same arange, which is only correct
      # for text-only prompts. This shows which path a request took.
      if os.environ.get("DT_TEXT_DIAG") == "1":
        pid_desc = "None (fallback)" if position_ids is None else (
            f"{tuple(position_ids.shape)} axes_equal="
            f"{bool((position_ids[0] == position_ids[1]).all() and (position_ids[1] == position_ids[2]).all())}")
        dv = deepstack_visual_embeds
        print(f"  [text-diag] seq={length} position_ids={pid_desc}"
              f" visual_pos_masks="
              f"{None if visual_pos_masks is None else tuple(visual_pos_masks.shape)}"
              f" image_rows="
              f"{0 if visual_pos_masks is None else int(visual_pos_masks.sum())}"
              f" deepstack={None if dv is None else len(dv)}", flush=True)
      if position_ids is None:
        index = torch.arange(length).view(1, 1, -1).expand(3, 1, -1)
        position_ids = index
      cos, sin = model.rotary_emb(inputs_embeds, position_ids)
      if self._tp:
        # Splash carries the causal mask in the kernel's own mask object
        # and `_splash_causal` drops this argument, so materialising
        # 7186 x 7186 here only moves 103 MB across the link for nothing.
        causal = torch.zeros((1, 1, 1, 1), dtype=inputs_embeds.dtype)
      else:
        low = torch.finfo(inputs_embeds.dtype).min
        causal = torch.full((length, length), low, dtype=inputs_embeds.dtype)
        causal = torch.triu(causal, diagonal=1)[None, None]

    if self._tp:
      replicated = jax.sharding.NamedSharding(self._mesh, P())
      put = lambda t: jax.device_put(_to_jax(t), replicated)
    else:
      put = lambda t: jax.device_put(_to_jax(t), self._device_for(None))
    # Rows the image occupies, broadcast to the deepstack adds. A zero
    # mask when the prompt carries no image makes the add a no-op.
    if visual_pos_masks is None:
      visual_pos_masks = torch.zeros(
          (1, inputs_embeds.shape[1]), dtype=torch.bool
      )
    scattered = []
    for embeds in deepstack_visual_embeds or []:
      full = torch.zeros_like(inputs_embeds)
      full[visual_pos_masks] = embeds.to(inputs_embeds.dtype)
      scattered.append(put(full))

    begin = time.perf_counter()
    hidden = put(inputs_embeds)
    jcos, jsin, jmask = put(cos), put(sin), put(causal)
    jvisual = put(visual_pos_masks)
    # With the weights resident the activation is what travels: cos,
    # sin and the mask are the same every layer, so they ride along to
    # each chip once and stay there.
    carried: dict = {}
    for index in range(self._depth):
      if self._tp:
        params = {
            k: jax.device_put(
                v, jax.sharding.NamedSharding(self._mesh, _TEXT_TP_SPECS(k))
            )
            for k, v in self._streamed[index].items()
        }
        args = (hidden, jcos, jsin, jmask)
      elif self._chips is None or self._chips[index] is None:
        params = {
            k: jax.device_put(v, self._device_for(k))
            for k, v in self._streamed[index].items()
        }
        # The streamed layers run on chip 0; the activation may be
        # wherever the last resident layer left it.
        hidden = jax.device_put(hidden, self._device_for(None))
        args = (hidden, jcos, jsin, jmask)
      else:
        here = jax.sharding.SingleDeviceSharding(
            self._chips[index], memory_kind="device"
        )
        if self._chips[index] not in carried:
          carried[self._chips[index]] = tuple(
              jax.device_put(t, here) for t in (jcos, jsin, jmask, jvisual)
          )
        held = carried[self._chips[index]]
        params = self._streamed[index]
        args = (jax.device_put(hidden, here), *held[:3])
      hidden = self._compiled(params, *args)
      if index < len(scattered):
        if self._tp or self._chips is None or self._chips[index] is None:
          hidden = self._deepstack_fn(hidden, scattered[index], jvisual)
        else:
          here = jax.sharding.SingleDeviceSharding(
              self._chips[index], memory_kind="device"
          )
          hidden = self._deepstack_fn(
              hidden,
              jax.device_put(scattered[index], here),
              carried[self._chips[index]][3],
          )
    if self._chips is not None and not self._tp:
      hidden = jax.device_put(hidden, self._device_for(None))
    # `hidden_states[50]` must be the pre-norm output of layer 50 (the
    # full tower is deeper, so in the reference it is the un-normalized
    # residual stream). Returning `norm(hidden)` there squashes the
    # massive-activation channels and badly degrades the conditioning.
    pre = jax.block_until_ready(hidden)
    out = jax.block_until_ready(self._norm_fn(self._norm, hidden))
    elapsed = time.perf_counter() - begin
    self.seconds.append(elapsed)
    print(f"  [text_encoder] {length} tokens: {elapsed:.2f}s", flush=True)

    hidden = torch.from_numpy(jax.device_get(out).astype("float32")).to(
        inputs_embeds.dtype
    )
    hidden_pre = torch.from_numpy(
        jax.device_get(pre).astype("float32")).to(inputs_embeds.dtype)
    if self._tp:
      # Hand the chips back their memory before the denoise starts.
      # Measured: the sharded stack leaves 0.98 GiB resident, free HBM
      # at the first denoise call goes 12.56 -> 11.58 GiB, and the two
      # biggest transformer programs each cost 0.5 s more. Releasing
      # per layer did nothing -- it is the tail that is held: the
      # deepstack scatters are 3 x 73 MB, and `out`, the angles and the
      # mask stay live until this function returns.
      for array in (out, pre, jcos, jsin, jmask, jvisual, *scattered):
        array.delete()
      scattered.clear()
    return BaseModelOutputWithPast(
        last_hidden_state=hidden,
        hidden_states=(None,) * self._depth + (hidden_pre,),
    )


def install(text_encoder, mesh: jax.sharding.Mesh, layers: int = 50,
            resident: int = 0, tensor_parallel: bool = False):
  """Puts Qwen3-VL's first `layers` decoder layers on the mesh."""
  return TextModelOnMesh(
      text_encoder.model.language_model, mesh, layers, resident,
      tensor_parallel
  )


class _OnDevice:
  """A vision block's output left on the chip.

  The tower passes a block's output straight to the next block without
  reading it, so for 23 of the 27 it never has to come back.
  """

  __slots__ = ("array",)

  def __init__(self, array):
    self.array = array

  @property
  def shape(self):
    return self.array.shape


class VisionTowerOnChip:
  """Qwen3-VL's vision blocks, compiled onto one chip.

  27 blocks of width 1152, about 1.2 GB in bfloat16, so these stay
  resident rather than streaming. On the host the tower measured
  20.0 s -- 40% of a ref2va request once the language model moved.

  Only the blocks move. The preamble (patch embedding, the interpolated
  position table, the rotary angles) and the deepstack mergers are cheap
  and stay where they are, which keeps this out of the way of
  `grid_thw`: the tower derives shapes from it, and a traced `grid_thw`
  makes those shapes abstract while a plain one mixes CPU tensors into
  XLA arithmetic. Neither happens if the tower's own forward runs
  unchanged and only `blk.forward` is swapped.

  The blocks are identical, so one program serves all 27 and their
  weights ride in as an argument.
  """

  def __init__(self, tower, mesh=None):
    import torchax

    global VISION_MESH
    self._tower = tower
    self._env = torchax.default_env()
    if mesh is not None and mesh.devices.size > 1:
      # Replicated over the mesh; attention splits heads across it.
      self._device = jax.sharding.NamedSharding(mesh, P())
      VISION_MESH = mesh
    else:
      chip = jax.devices()[0]
      self._device = jax.sharding.SingleDeviceSharding(chip, memory_kind="device")
    self._blocks = list(tower.blocks)
    self._params = [
        {
            name: jax.device_put(_to_jax(tensor), self._device)
            for name, tensor in block.named_parameters()
        }
        for block in self._blocks
    ]
    total = sum(
        v.size * v.dtype.itemsize for b in self._params for v in b.values()
    )
    register_splash_attention()
    tower.config._attn_implementation = "splash_full"
    for block in self._blocks:
      block.attn.config._attn_implementation = "splash_full"
    print(f"  [vision] {len(self._blocks)} blocks {total / 1e9:.2f} GB"
          f" resident on chip 0", flush=True)
    self._cache: dict = {}
    self._dtype = next(tower.parameters()).dtype
    self._angles: dict = {}
    self.seconds: list[float] = []
    for index, block in enumerate(self._blocks):
      block.forward = self._bind(index, block)

  def _bind(self, index, block):
    inner = block.forward
    # The tower only looks inside a block's output at the deepstack
    # layers and at the end; everywhere else it just hands it to the
    # next block. Those outputs stay on the chip behind this wrapper,
    # which takes the round trips from 27 down to 4 -- 7 GB of host
    # traffic down to under one.
    last = len(self._blocks) - 1
    needed_on_host = (
        index == last or index in self._tower.deepstack_visual_indexes
    )

    def forward(hidden_states, cu_seqlens=None, max_seqlen=None,
                position_embeddings=None, **kwargs):
      del kwargs
      put = lambda t: jax.device_put(_to_jax(t), self._device)
      if index == 0:
        # A fresh pass through the tower. Dropping the old entries keeps
        # the cache from matching a recycled `id`.
        self._angles.clear()
      # The same angles for all 27 blocks; upload them once.
      cos, sin = self._angles.setdefault(
          id(position_embeddings),
          tuple(put(t) for t in position_embeddings),
      )
      if isinstance(hidden_states, _OnDevice):
        hidden = hidden_states.array
        shape = hidden.shape
      else:
        hidden = put(hidden_states)
        shape = tuple(hidden_states.shape)
      key = (tuple(shape), int(max_seqlen or 0))
      fn = self._cache.get(key)
      if fn is None:
        print(f"  [vision] block {index} sees {shape}, "
              f"max_seqlen={max_seqlen}, cu_seqlens={cu_seqlens}", flush=True)
        def call(params, hidden, jcos, jsin):
          env = self._env
          ours = block.forward
          block.forward = inner  # functional_call dispatches through the forward
          try:
            with env:
              out = torch.func.functional_call(
                  block,
                  {k: env.j2t_iso(v) for k, v in params.items()},
                  args=(env.j2t_iso(hidden),),
                  kwargs={
                      "cu_seqlens": cu_seqlens,
                      "max_seqlen": max_seqlen,
                      "position_embeddings": (
                          env.j2t_iso(jcos), env.j2t_iso(jsin)
                      ),
                  },
              )
          finally:
            block.forward = ours
          return env.t2j_iso(out)

        fn = jax.jit(call)
        self._cache[key] = fn
      begin = time.perf_counter()
      out = fn(self._params[index], hidden, cos, sin)
      if needed_on_host:
        out = jax.block_until_ready(out)
      self.seconds.append(time.perf_counter() - begin)
      if index == len(self._blocks) - 1:
        print(f"  [vision] 27 blocks total {sum(self.seconds):.2f}s", flush=True)
        self.seconds.clear()
      if not needed_on_host:
        return _OnDevice(out)
      return torch.from_numpy(
          jax.device_get(out).astype("float32")
      ).to(self._dtype)

    return forward


def install_vision(text_encoder, mesh=None) -> VisionTowerOnChip | None:
  """Puts the vision blocks on the mesh (chip 0 without one). None when there is none."""
  tower = getattr(text_encoder.model, "visual", None)
  return VisionTowerOnChip(tower, mesh) if tower is not None else None
