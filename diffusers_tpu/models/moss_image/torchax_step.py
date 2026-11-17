"""One whole MOSS-Image sampling step as a single torchax program.

The upstream pipeline calls the transformer once per branch and then runs
the scheduler's Euler update eagerly, with a host sync per step (the
scheduler's ``.item()``). Here a step is one jitted call:

    pred = DiT([positive, negative])          # one batch of two
    latents += dt * (neg + g * (pos - neg))   # CFG and Euler, float32

Batching the branches needs the two things the upstream code gets from
running them separately:

* **Different prompt lengths.** Both are padded to one length. The joint
  sequence is ``[image | text | pad]``, so a sample's valid keys are a
  prefix and the splash kernel's runtime ``kv_len`` masks the rest; the
  two text-refinement blocks get the same mask. Padded rows are computed
  and never read.
* **Layer skipping in the negative branch** (``uncond_skip_layers``): a
  skipped layer is computed for the whole batch and each sample keeps its
  input or the layer's output (one extra layer in 42).

The image rows' RoPE depends on each prompt's *true* length (its ``f``
coordinate), so the table is per sample.

On a ``('cfg', 'tensor')`` mesh the batch is split over ``cfg`` and heads
over ``tensor``: with ``cfg=2`` the two branches run side by side on two
halves of the chips, inside one program.
"""
import functools

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from diffusers_tpu.models.moss_image.torchax_dit import _block


def make_masked_splash(mesh, env, use_fixed_m=False, block_q=None, block_kv=None):
  """``sdpa(q, k, v, kv_len)`` for [B, H, S, D] with per-sample valid lengths."""
  from jax.experimental.shard_map import shard_map

  from diffusers_tpu.kernels import block_sizes
  from diffusers_tpu.kernels import splash

  qkv = P("cfg", "tensor", None, None)
  pinned = {"block_q": block_q, "block_kv": block_kv, "block_kv_compute": block_kv}
  mode = "residual-fixedm" if use_fixed_m else "residual"
  hw = block_sizes.detect_spec()
  resolved = {}

  def local(q, k, v, kv_len, blocks):
    # Past 8192 keys one masked body for every block is much faster than
    # branching to it where the padding starts (2048x2048: 723 -> 495 ms a
    # step); at 1024x1024 the branch is ~2 ms a step faster.
    mask_all = k.shape[2] > 8192
    outs = [splash.attention(q[i:i + 1], k[i:i + 1], v[i:i + 1], scale=None,
                             use_fixed_m=use_fixed_m, kv_len=kv_len[i],
                             kv_mask_all=mask_all, **blocks)
            for i in range(q.shape[0])]
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)

  def sdpa(query, key, value, kv_len):
    jq, jk, jv, jl = env.t2j_iso((query, key, value, kv_len))
    shape = (jq.shape[2], jk.shape[2], jq.shape[-1])
    if shape not in resolved:
      resolved[shape] = block_sizes.resolve_block_sizes(*shape, pinned, mode, hw)
      print(f"  [splash] seq={shape[0]} -> {resolved[shape]}", flush=True)
    # Padded keys are zeroed (the fixed-m bound assumes it), then k-smoothed
    # over the valid keys only; softmax is invariant to both.
    valid = (jnp.arange(jk.shape[2])[None, :] < jl[:, None])[:, None, :, None]
    jk = jnp.where(valid, jk, 0)
    mean = jk.sum(2, keepdims=True) / jl[:, None, None, None].astype(jk.dtype)
    jk = jnp.where(valid, jk - mean, 0)
    out = shard_map(functools.partial(local, blocks=resolved[shape]), mesh=mesh,
                    in_specs=(qkv, qkv, qkv, P("cfg")), out_specs=qkv,
                    check_rep=False)(jq, jk, jv, jl.astype(jnp.float32))
    return env.j2t_iso(out)

  return sdpa


def _text_block(block, x, rope, rotate, situ, key_mask):
  """``_TextBlock.forward`` with a key mask over padded prompt rows."""
  q, k, v = block.qkv(block.norm1(x)).chunk(3, -1)
  shape = (*q.shape[:2], block.heads, q.shape[-1] // block.heads)
  q = rotate(block.q_norm(q.reshape(shape)), rope).transpose(1, 2)
  k = rotate(block.k_norm(k.reshape(shape)), rope).transpose(1, 2)
  v = v.reshape(shape).transpose(1, 2)
  out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                                       attn_mask=key_mask[:, None, None, :])
  x = x + block.proj(out.transpose(1, 2).flatten(2))
  y = block.norm2(x)
  return x + block.ffn_down(situ(block.ffn_up(y), block.ffn_gate(y)))


class StepCore(torch.nn.Module):
  """DiT on a padded batch, then CFG and the Euler update."""

  def __init__(self, model, sdpa, env, mesh):
    super().__init__()
    import inspect

    upstream = inspect.getmodule(type(model))
    self._rotate = upstream._rotate
    self._situ = upstream._situ
    self._sdpa = sdpa
    self._env = env
    self._mesh = mesh
    self.video_embedder = model.video_embedder
    self.text_embedder = model.text_embedder
    self.time_embedding = model.time_embedding
    self.shared_block_modulation = model.shared_block_modulation
    self.layers = model.layers
    self.final_norm_video = model.final_norm_video
    self.final_linear_video = model.final_linear_video
    self.video_head_modulation = model.video_head_modulation

  def _attention(self, kv_len):
    rotate, sdpa = self._rotate, self._sdpa

    def attention(q, k, v, rope, heads):
      shape = (*q.shape[:2], heads, q.shape[-1] // heads)
      q = rotate(q.reshape(shape), rope).transpose(1, 2)
      k = rotate(k.reshape(shape), rope).transpose(1, 2)
      v = v.reshape(shape).transpose(1, 2)
      return sdpa(q.to(v.dtype), k.to(v.dtype), v, kv_len).transpose(1, 2).flatten(2)

    return attention

  def dit(self, latents, sinusoid, context, rope, kv_len, skip, skip_layers):
    batch, _, height, width = latents.shape
    n_img = height * width
    time = self.time_embedding(sinusoid.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(
        time.to(self.shared_block_modulation.weight.dtype)).view(time.shape[0], 6, -1)
    head = self.video_head_modulation(
        time.to(self.video_head_modulation.weight.dtype)).view(time.shape[0], 2, -1)
    video = latents.flatten(2).transpose(1, 2)
    video = self.video_embedder(video.to(self.video_embedder.weight.dtype))
    rope = rope.to(video.dtype)
    text_rope = rope[:, n_img:]
    key_mask = torch.arange(context.shape[1], device=context.device)[None, :] < (kv_len - n_img)[:, None]
    text = self.text_embedder.proj(context.to(self.text_embedder.proj.weight.dtype))
    for block in self.text_embedder.blocks:
      text = _text_block(block, text, text_rope, self._rotate, self._situ, key_mask)
    x = torch.cat((video, text), 1)
    attention = self._attention(kv_len)
    for index, layer in enumerate(self.layers):
      out = _block(layer, x, rope, mod, attention, self._situ)
      x = torch.where(skip[:, None, None], x, out) if index in skip_layers else out
    x = self.final_norm_video(x[:, :n_img].to(self.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(batch, -1, height, width)

  def forward(self, latents, sinusoid, context, rope, kv_len, skip, dt, guidance,
              skip_layers=()):
    """latents [1, C, H, W] fp32 -> the next latents. ``context`` etc. hold
    ``B`` rows: [positive] or [positive, negative]."""
    batch = context.shape[0]
    x = latents.expand(batch, -1, -1, -1)
    if batch > 1:
      env = self._env
      x = env.j2t_iso(jax.lax.with_sharding_constraint(
          env.t2j_iso(x), NamedSharding(self._mesh, P("cfg"))))
    pred = self.dit(x, sinusoid, context, rope, kv_len, skip, skip_layers).float()
    if batch > 1:
      pred = pred[1:2] + guidance * (pred[0:1] - pred[1:2])
    return latents + dt * pred
