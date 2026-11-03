"""Tensor-parallel H3 for torch_tpu, over the stock diffusers module.

Uses torch.distributed's `tpu` backend for all-reduce / all-gather. Rather
than rewrite the stock diffusers module against parallel linears, each
block's linears are sliced in place on the host and the row-parallel ones
all-reduce through a forward hook:

    to_q, to_k, to_v      column-parallel, one head group per rank
    to_out[0]             row-parallel, all-reduce after
    ff.net[0].proj        column-parallel, and SwiGLU's gate and up
                          halves are each split so the pairing survives
    ff.net[2]             row-parallel, all-reduce after

Slicing on the host before anything reaches a chip is also what makes
this compose with `layerwise_offload`: each rank parks its own shard,
so the 60 GiB of blocks becomes 15 at four chips rather than 60 on each.

56 heads divides by 1, 2, 4, 7 and 8.
"""

import torch


def _divide(a, b):
  if a % b:
    raise ValueError(f"{a} does not divide into {b} ranks")
  return a // b


def _column(linear, rank, world, *, halves=1):
  """Keep this rank's slice of the output features.

  `halves` is for SwiGLU, whose projection emits `[gate | up]` in one
  tensor: splitting it as a single block would hand rank 0 all of the
  gate and rank 1 all of the up, which is not the same function. Each
  half is sliced separately and the pieces re-joined.
  """
  out, _ = linear.weight.shape
  per = _divide(out, halves)
  piece = _divide(per, world)
  rows = torch.cat([
      linear.weight.data[h * per + rank * piece: h * per + (rank + 1) * piece]
      for h in range(halves)
  ])
  linear.weight.data = rows
  if linear.bias is not None:
    linear.bias.data = torch.cat([
        linear.bias.data[h * per + rank * piece: h * per + (rank + 1) * piece]
        for h in range(halves)
    ])
  linear.out_features = rows.shape[0]


def _row(linear, rank, world, all_reduce):
  """Keep this rank's slice of the input features and sum the outputs."""
  _, inp = linear.weight.shape
  piece = _divide(inp, world)
  linear.weight.data = linear.weight.data[:, rank * piece:(rank + 1) * piece]
  linear.in_features = piece
  # The bias must not be added once per rank.
  if linear.bias is not None and rank != 0:
    linear.bias.data = torch.zeros_like(linear.bias.data)
  linear.register_forward_hook(lambda _m, _i, out: all_reduce(out))


def _adaln(module, rank, world, all_gather):
  """Column-parallel the AdaLN projection and gather its output back.

  `adaln_proj.linear` is `2688 x (6 * 5376 * 3)` -- 260.2 M parameters a
  block (13.0 B over fifty blocks, ~43% of each block). Left unsharded,
  tensor parallel only halves the model instead of dividing it by world
  size. The JAX path shards it the same way (`P("tp", None)`).

  Correctness: the module does
  `linear(temb).view(-1, 6 * hidden).chunk(6, -1)`, so the flat output
  axis is consumed positionally. Slicing that axis by rank and
  concatenating the pieces back **in rank order** reproduces it
  exactly, so an all-gather on the output restores the original
  tensor and everything downstream is untouched.

  Cost: the gathered tensor is `(rows, 6 * hidden * modalities)` with
  `rows` in the single digits -- about 580 KB, against the 26 GiB of
  adaln weights that no longer have to cross from host memory every
  step.
  """
  linear = module.linear
  _column(linear, rank, world)
  linear.register_forward_hook(
      lambda _m, _i, out: all_gather(out, dim=-1)
  )


def shard_transformer(transformer, rank, world, all_reduce, all_gather=None):
  """Slice every block of a stock H3 transformer across `world` ranks."""
  if world == 1:
    return 0
  done = 0
  for block in transformer.transformer_blocks:
    attn = block.attn
    for name in ("to_q", "to_k", "to_v"):
      _column(getattr(attn, name), rank, world)
    _row(attn.to_out[0], rank, world, all_reduce)
    attn.heads = _divide(attn.heads, world)
    if hasattr(attn, "inner_dim"):
      attn.inner_dim = _divide(attn.inner_dim, world)
    # `ff` is diffusers' FeedForward with activation_fn="swiglu":
    # net[0] is the activation holding the projection, net[2] the
    # output linear.
    act = block.ff.net[0]
    proj = getattr(act, "proj", None)
    if proj is not None:
      _column(proj, rank, world, halves=2)
    _row(block.ff.net[2], rank, world, all_reduce)
    if all_gather is not None and getattr(block, "adaln_proj", None) is not None:
      _adaln(block.adaln_proj, rank, world, all_gather)
    done += 1
  return done


def init_parallel():
  """Brings up the distributed group. Returns (rank, world, all_reduce).

  Falls back to one rank when the launcher did not start a group, so the
  single-chip path keeps working unchanged.
  """
  import os

  world = int(os.environ.get("WORLD_SIZE", 1))
  rank = int(os.environ.get("RANK", 0))
  if world == 1:
    init_parallel.all_gather = lambda t, dim=-1: t
    return 0, 1, lambda t: t

  import torch.distributed as dist

  if not dist.is_initialized():
    dist.init_process_group(
        backend="tpu_dist",
        init_method="env://",
        world_size=world,
        rank=rank,
    )
  device_group = dist.new_group(list(range(world)), backend="tpu_dist")
  group_name = device_group.group_name

  def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
      t = torch.ops._c10d_functional.all_reduce(input_, "sum", group_name)
      return torch.ops._c10d_functional.wait_tensor(t)
    dist.all_reduce(input_, group=device_group)
    return input_

  def _all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim < 0:
      dim += input_.dim()
    if torch.compiler.is_compiling():
      input_size = list(input_.size())
      output = torch.ops._c10d_functional.all_gather_into_tensor(
          input_.contiguous(), world, group_name
      )
      output = torch.ops._c10d_functional.wait_tensor(output)
      if dim != 0:
        output = output.reshape([world] + input_size)
        output = output.movedim(0, dim)
      input_size[dim] *= world
      return output.reshape(input_size)
    input_size = list(input_.size())
    output_tensor = torch.empty(
        [world * input_size[0]] + input_size[1:],
        dtype=input_.dtype,
        device=input_.device,
    )
    dist.all_gather_into_tensor(
        output_tensor, input_.contiguous(), group=device_group
    )
    if dim != 0:
      output_tensor = output_tensor.reshape([world] + input_size)
      output_tensor = output_tensor.movedim(0, dim)
      input_size[dim] *= world
      output_tensor = output_tensor.reshape(input_size)
    return output_tensor

  init_parallel.all_gather = _all_gather
  init_parallel.group_name = group_name
  return rank, world, _all_reduce


def shard_text_encoder(encoder, rank, world, all_reduce, verbose=print):
  """Shard the Qwen3-VL decoder layers across ranks.

  The bottleneck is host-to-chip transfer, not compute: each rank would
  otherwise move 58.1 GiB of weights at ~2 GB/s (38.8 s of ref2va's text
  encoder). With eight ranks each moves 7.3 GiB.

  Manual sharding works because `hidden_shape = (*input_shape, -1,
  head_dim)` infers heads from the actual (sliced) tensor size, and
  `num_key_value_groups` (64 q heads / 8 kv heads) is unchanged by the split.
  """
  if world <= 1:
    return 0
  for attr in ("model.language_model.layers", "language_model.layers",
               "model.layers", "layers"):
    node = encoder
    try:
      for part in attr.split("."):
        node = getattr(node, part)
    except AttributeError:
      continue
    n = 0
    for layer in node:
      attn = getattr(layer, "self_attn", None)
      mlp = getattr(layer, "mlp", None)
      if attn is None or mlp is None:
        continue
      for name in ("q_proj", "k_proj", "v_proj"):
        _column(getattr(attn, name), rank, world)
      _row(attn.o_proj, rank, world, all_reduce)
      for name in ("gate_proj", "up_proj"):
        _column(getattr(mlp, name), rank, world)
      _row(mlp.down_proj, rank, world, all_reduce)
      n += 1
    verbose(f"  [shard-text] sharded {n} layers, per-rank weights now 1/{world}",
            flush=True)
    return n
  verbose("  [shard-text] layer stack not found", flush=True)
  return 0
