"""``torchax`` backend for MOVA (40L Video + 30L Audio + 30L Bidirectional Bridge).

Traces pure-PyTorch :class:`MovaTowerTorch` and :class:`MovaBridgeTorch`
(``mova/torch_dit.py``) through ``torchax`` into the exact same single fused
40-layer ``shard_map`` + Pallas ``splash`` / ``sparse_splash`` program as
``mova/jax_dit.make_dual_blocks_fn``, sharing the fast ``ml_dtypes`` weight
loader and ``jax_dit.place`` (resident / sharded / W8A8 FP8) with zero extra HBM.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from . import jax_dit
from .torch_dit import (
    MovaBridgeTorch,
    MovaTowerTorch,
    MovaTpuHooks,
)


def _bind_block_params(blk: Any, p: dict[str, jax.Array], pre: str, j2t: Any) -> None:
  """Bind one tower block's sharded/replicated JAX params into ``MovaBlockTorch``."""
  blk._parameters["modulation"] = j2t(p[pre + "modulation"])
  blk._parameters["self_attn_norm_q"] = j2t(p[pre + "self_attn.norm_q.weight"])
  blk._parameters["self_attn_norm_k"] = j2t(p[pre + "self_attn.norm_k.weight"])
  blk._parameters["norm3_weight"] = j2t(p[pre + "norm3.weight"])
  blk._parameters["norm3_bias"] = j2t(p[pre + "norm3.bias"])
  blk._parameters["cross_attn_norm_q"] = j2t(p[pre + "cross_attn.norm_q.weight"])
  blk._parameters["cross_attn_norm_k"] = j2t(p[pre + "cross_attn.norm_k.weight"])

  def _lin(mod: Any, name: str) -> None:
    mod._parameters["weight"] = j2t(p[name + ".weight"])
    mod._parameters["bias"] = j2t(p[name + ".bias"])
    sc = p.get(name + ".scale")
    mod._parameters["scale"] = j2t(sc) if sc is not None else None

  for nm in ("q", "k", "v", "o"):
    _lin(getattr(blk, f"self_attn_{nm}"), pre + f"self_attn.{nm}")
    _lin(getattr(blk, f"cross_attn_{nm}"), pre + f"cross_attn.{nm}")
  _lin(blk.ffn_0, pre + "ffn.0")
  _lin(blk.ffn_2, pre + "ffn.2")


def _bind_bridge_cond(cond: Any, bp: dict[str, jax.Array], name: str, j2t: Any) -> None:
  """Bind one ``ConditionalCrossAttentionBlock``'s JAX params into ``MovaBridgeCondTorch``."""
  cond._parameters["y_norm_weight"] = j2t(bp[name + ".y_norm.weight"])
  cond._parameters["y_norm_bias"] = j2t(bp[name + ".y_norm.bias"])
  pre = name + ".inner."
  cond._parameters["norm_q"] = j2t(bp[pre + "norm_q.weight"])
  cond._parameters["norm_k"] = j2t(bp[pre + "norm_k.weight"])
  for nm in ("q", "k", "v", "o"):
    lin = getattr(cond, nm)
    lin._parameters["weight"] = j2t(bp[pre + f"{nm}.weight"])
    lin._parameters["bias"] = j2t(bp[pre + f"{nm}.bias"])
    sc = bp.get(pre + f"{nm}.scale")
    lin._parameters["scale"] = j2t(sc) if sc is not None else None


def make_torchax_dual_blocks_fn(
    mesh: Any,
    vconf: dict[str, Any],
    aconf: dict[str, Any],
    bconf: dict[str, Any],
    video_sample: dict[str, jax.Array],
    audio_sample: dict[str, jax.Array],
    bridge_sample: dict[str, jax.Array],
    *,
    resident: bool = False,
    qk_fp8: bool = False,
    block_kv: int | None = None,
    sparse_kv_table: Any = None,
):
  """Drop-in ``torchax`` replacement for :func:`jax_dit.make_dual_blocks_fn`.

  Builds ``meta``-device ``MovaTowerTorch`` / ``MovaBridgeTorch`` shells with
  ``pretranspose=True`` (matching ``jax_dit``'s ``(in, out)`` layout), and
  traces all 40 video + 30 bridge + 30 audio layers inside a single fused
  ``shard_map`` with Ulysses SP, Pallas Splash / Sparse-Splash, and W8A8 FP8.
  """
  import torchax
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  from diffusers_tpu.kernels import sparse_splash, splash  # noqa: F401

  T = "tensor"
  vheads = int(vconf["num_heads"])
  if vheads % int(mesh.shape[T]):
    raise ValueError(f"sequence parallel needs {vheads} video heads to divide tensor={mesh.shape[T]}")
  n_layers = int(vconf["num_layers"])
  shared = min(n_layers, int(aconf["num_layers"]))

  with torch.device("meta"):
    v_shell = MovaTowerTorch(vconf, dtype=torch.bfloat16, pretranspose=True)
    a_shell = MovaTowerTorch(aconf, dtype=torch.bfloat16, pretranspose=True)
    b_shell = MovaBridgeTorch(bconf, shared, dtype=torch.bfloat16, pretranspose=True)
  v_shell.eval()
  a_shell.eval()
  b_shell.eval()

  env = torchax.default_env()
  t2j = env.t2j_iso
  j2t = env.j2t_iso

  def _fp8_lin_ta(x_t, qw_t, sc_t, b_t):
    xj, qwj, scj, bj = t2j(x_t), t2j(qw_t), t2j(sc_t), t2j(b_t)
    f8_max = jnp.float32(448.0)
    x32 = xj.astype(jnp.float32)
    xs = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True), jnp.float32(1e-12)) / f8_max
    qx = jnp.clip(x32 / xs, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        qx, qwj, (((xj.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    y = (acc * xs * scj).astype(xj.dtype) + bj.astype(xj.dtype)
    return j2t(y)

  def _local_dense_j(q, k, v):
    out = jax_dit._splash_local(*(t.transpose(0, 2, 1, 3) for t in (q, k, v)), qk_fp8=False)
    return out.transpose(0, 2, 1, 3)

  def _local_dense_ta(q_t, k_t, v_t):
    return j2t(_local_dense_j(t2j(q_t), t2j(k_t), t2j(v_t)))

  def _ulysses_self_ta(q_t, k_t, v_t):
    q, k, v = t2j(q_t), t2j(k_t), t2j(v_t)
    if int(mesh.shape[T]) > 1:
      to_h = lambda z: jax.lax.all_to_all(z, T, 2, 1, tiled=True)
      q, k, v = to_h(q), to_h(k), to_h(v)
    out = jax_dit._splash_local(
        *(z.transpose(0, 2, 1, 3) for z in (q, k, v)),
        qk_fp8=qk_fp8,
        block_kv=block_kv,
        kv_table=sparse_kv_table,
    ).transpose(0, 2, 1, 3)
    if int(mesh.shape[T]) > 1:
      out = jax.lax.all_to_all(out, T, 1, 2, tiled=True)
    return j2t(out)

  def _v2a_ta(q_t, k_t, v_t):
    q, k, v = t2j(q_t), t2j(k_t), t2j(v_t)
    sc = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
    if int(mesh.shape[T]) > 1:
      m = jax.lax.pmax(jnp.max(sc, -1, keepdims=True), T)
      e = jnp.exp(sc - m)
      l = jax.lax.psum(jnp.sum(e, -1, keepdims=True), T)
      o = jax.lax.psum(
          jnp.einsum("bhqk,bkhd->bqhd", e.astype(v.dtype), v, preferred_element_type=jnp.float32), T
      )
    else:
      m = jnp.max(sc, -1, keepdims=True)
      e = jnp.exp(sc - m)
      l = jnp.sum(e, -1, keepdims=True)
      o = jnp.einsum("bhqk,bkhd->bqhd", e.astype(v.dtype), v, preferred_element_type=jnp.float32)
    return j2t((o / jnp.transpose(l, (0, 2, 1, 3))).astype(v.dtype))

  hooks = MovaTpuHooks(
      video_self_attn=_ulysses_self_ta,
      video_cross_attn=_local_dense_ta,
      a2v_attn=_local_dense_ta,
      v2a_attn=_v2a_ta,
      audio_attn=_local_dense_ta,
      fp8_linear=_fp8_lin_ta,
      sp_size=int(mesh.shape[T]),
      qk_fp8=qk_fp8,
  )

  vp_keys = {k for k in video_sample if k.startswith("blocks.")}
  ap_keys = {k for k in audio_sample if k.startswith("blocks.")}
  vp_specs = jax_dit.param_specs({k: video_sample[k] for k in vp_keys}, resident=resident)
  ap_specs = jax_dit.param_specs({k: audio_sample[k] for k in ap_keys}, resident=resident)
  bp_specs = jax_dit.param_specs(bridge_sample, resident=resident)

  def _gather_if_sharded(d: dict[str, jax.Array], specs: dict[str, Any]) -> dict[str, jax.Array]:
    if resident:
      return d
    out = {}
    for k, val in d.items():
      sp = specs[k]
      if sp == P():
        out[k] = val
      elif sp == P(None, T):
        out[k] = jax.lax.all_gather(val, T, axis=1, tiled=True)
      else:
        out[k] = jax.lax.all_gather(val, T, axis=0, tiled=True)
    return out

  def body(vp, ap, bp, vx, ax, vctx, actx, vtm, atm, vcos, vsin, acos, asin, vrope, arope):
    with env:
      vx_t, ax_t = j2t(vx), j2t(ax)
      vctx_t, actx_t = j2t(vctx), j2t(actx)
      vtm_t, atm_t = j2t(vtm), j2t(atm)
      vcos_t, vsin_t = j2t(vcos), j2t(vsin)
      acos_t, asin_t = j2t(acos), j2t(asin)
      vrope_t = (j2t(vrope[0]), j2t(vrope[1]))
      arope_t = (j2t(arope[0]), j2t(arope[1]))

      for i in range(n_layers):
        if i < shared:
          b_i_keys = [k for k in bp if f"_conditioners.{i}." in k]
          bp_i = _gather_if_sharded({k: bp[k] for k in b_i_keys}, bp_specs)
          _bind_bridge_cond(b_shell.v2a[i], bp_i, f"video_to_audio_conditioners.{i}", j2t)
          _bind_bridge_cond(b_shell.a2v[i], bp_i, f"audio_to_video_conditioners.{i}", j2t)
          v2a_t = b_shell.v2a[i](ax_t, vx_t, arope_t, vrope_t, attn_fn=hooks.v2a_attn, hooks=hooks)
          vx_t = vx_t + b_shell.a2v[i](vx_t, ax_t, vrope_t, arope_t, attn_fn=hooks.a2v_attn, hooks=hooks)

        pre_v = f"blocks.{i}."
        vp_i = _gather_if_sharded({k: vp[k] for k in vp if k.startswith(pre_v)}, vp_specs)
        _bind_block_params(v_shell.blocks[i], vp_i, pre_v, j2t)
        vx_t = v_shell.blocks[i](
            vx_t, vctx_t, vtm_t, vcos_t, vsin_t,
            self_attn_fn=hooks.video_self_attn, cross_attn_fn=hooks.video_cross_attn, hooks=hooks,
        )

        if i < shared:
          ap_i = _gather_if_sharded({k: ap[k] for k in ap if k.startswith(pre_v)}, ap_specs)
          _bind_block_params(a_shell.blocks[i], ap_i, pre_v, j2t)
          ax_t = a_shell.blocks[i](
              ax_t + v2a_t, actx_t, atm_t, acos_t, asin_t,
              self_attn_fn=hooks.audio_attn, cross_attn_fn=hooks.audio_attn, hooks=hooks,
          )
      return t2j(vx_t), t2j(ax_t)

  rows, batch, rep = P("cfg", T, None), P("cfg"), P()
  in_specs = (
      vp_specs, ap_specs, bp_specs,
      rows, batch, batch, batch, rep, rep,
      P(T, None), P(T, None), rep, rep,
      (P(T, None), P(T, None)), (rep, rep),
  )
  out_specs = (rows, batch)
  mapped = shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)

  def run(video, audio, bridge, vx, ax, vctx, actx, vtm, atm, vcos, vsin, acos, asin, vrope, arope):
    vp = {k: video[k] for k in vp_keys}
    ap = {k: audio[k] for k in ap_keys}
    return mapped(vp, ap, bridge, vx, ax, vctx, actx, vtm, atm, vcos, vsin, acos, asin, vrope, arope)

  return run
