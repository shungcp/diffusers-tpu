"""LTX-2.3's 48 blocks sequence parallel over one mesh axis, in one shard_map.

Tensor parallelism all-reduces the [3, 6528, 4096] residual stream four times
a block (1.1 GB a block a chip on 8 chips) and is ICI-bound at 1.26 s a
forward on a v6e-8. Here every chip keeps 1/n of the video rows and runs
each token-wise op on them with the block's whole weights:

* weights: split by rows over the axis and all-gathered block by block
  (XLA overlaps the next block's gather with this block's compute), except
  the ``resident`` blocks, kept whole on every chip (no gather at all);
* video self-attention: Ulysses (all-to-all to all rows x 1/n of the heads,
  the splash kernel, all-to-all back);
* text and audio->video cross-attention: local queries, replicated keys;
* video->audio attention: the audio queries against this chip's video keys,
  partial softmaxes merged exactly (pmax / psum);
* the audio stream (134 rows) runs whole on every chip.
"""
import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.ltx2 import jax_dit as jd


def _gathered(name, v, n):
  return v.ndim == 2 and name.endswith(".weight") and v.shape[0] % n == 0 and v.shape[0] >= 512


def param_specs(params, axis, n, resident=()):
  from jax.sharding import PartitionSpec as P

  out = {k: P() for k in params if k != "blocks"}
  out["blocks"] = [{k: (P(axis, None) if i not in resident and _gathered(k, v, n) else P())
                    for k, v in blk.items()} for i, blk in enumerate(params["blocks"])]
  return out


def place(params, mesh, axis="sp", resident=()):
  from jax.sharding import NamedSharding

  specs = param_specs(params, axis, mesh.shape[axis], resident)
  return jax.tree.map(lambda v, s: jax.device_put(v, NamedSharding(mesh, s)), params, specs)


def _splash(q, k, v):
  if jax.default_backend() == "cpu" or q.shape[2] % 128 or k.shape[2] % 128:  # tests / small sizes
    t = lambda z: z.transpose(0, 2, 1, 3)
    return t(jd.dense_attention(t(q), t(k), t(v)))
  from diffusers_tpu.kernels import block_sizes, splash

  blocks = block_sizes.resolve_block_sizes(q.shape[2], k.shape[2], q.shape[-1], {}, "residual",
                                           block_sizes.detect_spec())
  return splash.attention(q, k, v, scale=None, **blocks)


def _attentions(axis):
  def video_self(q, k, v, bias=None):  # Ulysses: [B, S/n, H, D] -> [B, S, H/n, D] and back
    a2a = lambda t: jax.lax.all_to_all(t, axis, 2, 1, tiled=True)
    q, k, v = (a2a(t).transpose(0, 2, 1, 3) for t in (q, k, v))
    o = _splash(q, k, v).transpose(0, 2, 1, 3)
    return jax.lax.all_to_all(o, axis, 1, 2, tiled=True)

  def v2a(q, k, v, bias=None):  # replicated audio queries, this chip's video keys
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
    m = jax.lax.pmax(jnp.max(s, -1, keepdims=True), axis)
    p = jnp.exp(s - m)
    l = jax.lax.psum(jnp.sum(p, -1, keepdims=True), axis)
    o = jnp.einsum("bhqk,bkhd->bqhd", p.astype(v.dtype), v, preferred_element_type=jnp.float32)
    o = jax.lax.psum(o, axis) / jnp.swapaxes(l, 1, 2)
    return o.astype(v.dtype)

  dense = jd.dense_attention
  return {"video_self": video_self, "audio_self": dense, "video_text": dense, "audio_text": dense,
          "a2v": dense, "v2a": v2a}


def make_blocks_fn(mesh, cfg, params, axis="sp", resident=(), stg_blocks=()):
  """``blocks_fn`` for ``jax_dit.forward``: x rows split over ``axis``."""
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  # Imported here, outside any trace: importing the kernel modules changes
  # JAX config, which mid-trace makes the next call retrace and recompile.
  from diffusers_tpu.kernels import block_sizes, splash  # noqa: F401

  n = mesh.shape[axis]
  specs = param_specs(params, axis, n, resident)["blocks"]
  attention = _attentions(axis)
  rows = P(None, axis, None)
  rope_specs = {"video": (P(axis, None, None),) * 2, "ca_video": (P(axis, None, None),) * 2,
                "audio": (P(),) * 2, "ca_audio": (P(),) * 2}
  stg_set = set(stg_blocks)

  def body(blocks, x, a, ctx, actx, cond, rope, stg, iso):
    for i, p in enumerate(blocks):
      p = {k: jax.lax.all_gather(v, axis, axis=0, tiled=True) if specs[i][k] != P() else v
           for k, v in p.items()}
      vo = stg if i in stg_set else None
      x, a = jd.block(p, x, a, ctx, actx, cond, rope, None, cfg, attention, vo, iso)
    return x, a

  def blocks_fn(blocks, x, a, ctx, actx, cond, rope, stg, iso=None):
    cond_specs = {k: P() for k in cond}
    if cond.get("token_t") is not None:
      cond_specs["token_t"] = P(axis)
    else:
      cond = dict(cond)
      cond.pop("token_t", None)
      cond_specs.pop("token_t", None)
    stg_spec = None if stg is None else P()
    iso_spec = None if iso is None else P()
    fn = shard_map(body, mesh=mesh,
                   in_specs=(specs, rows, P(), P(), P(), cond_specs, rope_specs, stg_spec, iso_spec),
                   out_specs=(rows, P()), check_rep=False)
    return fn(blocks, x, a, ctx, actx, cond, rope, stg, iso)

  return blocks_fn
