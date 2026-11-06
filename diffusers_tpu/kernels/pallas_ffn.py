# Copyright 2026 Google LLC
#
# Sequence-Tiled Pallas Megakernel for SwiGLU FeedForward + Dynamic FP8 Quantization
# on TPU v7x (Ironwood).
#
# Eliminates HBM materialization of:
#   1. Dynamic FP8 quantized input activations `x_fp8` and `x_scale`
#   2. Gated projection outputs `value` and `gate` (`[S, 2 * ffn_local]`)
#   3. Intermediate SwiGLU activations `mid = value * silu(gate)` (`[S, ffn_local]`)
#   4. Dynamic FP8 quantized intermediate activations `mid_fp8` and `mid_scale`
# while keeping the local shard of `W_val`, `W_gate`, and `W_out` stationary in VMEM
# across all sequence tiles `B_M`.

from __future__ import annotations

import functools
import os
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from . import fp8_cast
from .splash import _splash_vmem_limit_bytes

NT_DIMS = (((1,), (1,)), ((), ()))


def _swiglu_ffn_bf16_kernel(
    x_ref,
    w_val_ref,
    w_gate_ref,
    w_out_ref,
    out_ref,
):
  x_tile = x_ref[...]  # (B_M, hidden) bf16
  w_val = w_val_ref[...]  # (ffn_local, hidden) bf16
  w_gate = w_gate_ref[...]  # (ffn_local, hidden) bf16
  w_out = w_out_ref[...]  # (hidden, ffn_local) bf16

  val = lax.dot_general(
      x_tile, w_val, NT_DIMS, preferred_element_type=jnp.float32
  ).astype(jnp.bfloat16)
  gate = lax.dot_general(
      x_tile, w_gate, NT_DIMS, preferred_element_type=jnp.float32
  ).astype(jnp.bfloat16)
  mid = (val * jax.nn.silu(gate)).astype(jnp.bfloat16)
  out = lax.dot_general(
      mid, w_out, NT_DIMS, preferred_element_type=jnp.float32
  )
  out_ref[...] = out.astype(out_ref.dtype)


def _swiglu_ffn_fp8_kernel(
    x_ref,
    w_val_ref,
    ws_val_ref,
    w_gate_ref,
    ws_gate_ref,
    w_out_ref,
    ws_out_ref,
    out_ref,
):
  f8_max = jnp.float32(448.0)
  x_tile = x_ref[...]  # (B_M, hidden) bf16
  x32 = x_tile.astype(jnp.float32)
  x_max = jnp.maximum(
      jnp.max(jnp.abs(x32), axis=1, keepdims=True), jnp.float32(1e-12)
  )
  x_scale = x_max / f8_max
  x_q = fp8_cast.quantize_saturating((x32 / x_scale).astype(jnp.bfloat16))

  val_acc = lax.dot_general(
      x_q, w_val_ref[...], NT_DIMS, preferred_element_type=jnp.float32
  )
  gate_acc = lax.dot_general(
      x_q, w_gate_ref[...], NT_DIMS, preferred_element_type=jnp.float32
  )
  val = (val_acc * x_scale * ws_val_ref[0:1, :]).astype(jnp.bfloat16)
  gate = (gate_acc * x_scale * ws_gate_ref[0:1, :]).astype(jnp.bfloat16)

  mid = (val * jax.nn.silu(gate)).astype(jnp.bfloat16)
  mid32 = mid.astype(jnp.float32)
  mid_max = jnp.maximum(
      jnp.max(jnp.abs(mid32), axis=1, keepdims=True), jnp.float32(1e-12)
  )
  mid_scale = mid_max / f8_max
  mid_q = fp8_cast.quantize_saturating((mid32 / mid_scale).astype(jnp.bfloat16))

  out_acc = lax.dot_general(
      mid_q, w_out_ref[...], NT_DIMS, preferred_element_type=jnp.float32
  )
  out = out_acc * mid_scale * ws_out_ref[0:1, :]
  out_ref[...] = out.astype(out_ref.dtype)


def pallas_swiglu_ffn_local(
    x: jax.Array,
    w_val: jax.Array,
    w_gate: jax.Array,
    w_out: jax.Array,
    ws_val: jax.Array | None = None,
    ws_gate: jax.Array | None = None,
    ws_out: jax.Array | None = None,
    *,
    block_m: int = 512,
) -> jax.Array:
  """Runs the sequence-tiled Pallas SwiGLU FFN megakernel on a single shard.

  Args:
    x: `(B, S, hidden)` or `(S, hidden)` bfloat16 activation.
    w_val: `(ffn_local, hidden)` bfloat16 or float8_e4m3fn value weight shard.
    w_gate: `(ffn_local, hidden)` bfloat16 or float8_e4m3fn gate weight shard.
    w_out: `(hidden, ffn_local)` bfloat16 or float8_e4m3fn output weight shard.
    ws_val: `(1, ffn_local)` float32 weight scale for `w_val` (when FP8).
    ws_gate: `(1, ffn_local)` float32 weight scale for `w_gate` (when FP8).
    ws_out: `(1, hidden)` float32 weight scale for `w_out` (when FP8).
    block_m: Sequence tile size (must divide flattened `B * S` or will pad).

  Returns:
    Partial output `(..., hidden)` before cross-TP AllReduce / ReduceScatter.
  """
  orig_shape = x.shape
  hidden = orig_shape[-1]
  x_2d = x.reshape(-1, hidden)
  seq_len = x_2d.shape[0]
  bm = min(block_m, seq_len)
  if bm >= 128:
    bm = (bm // 128) * 128
  pad_len = (-seq_len) % bm
  if pad_len > 0:
    x_2d = jnp.pad(x_2d, ((0, pad_len), (0, 0)))
  padded_seq = x_2d.shape[0]
  num_tiles = padded_seq // bm

  ffn_local = w_val.shape[0]
  is_fp8 = w_val.dtype == jnp.float8_e4m3fn and ws_val is not None
  max_ffn = 1792 if is_fp8 else 896
  max_bm = 512 if (not is_fp8 and ffn_local >= 896) else block_m
  if block_m > max_bm:
    block_m = max_bm
  if ffn_local > max_ffn and ffn_local % max_ffn == 0:
    n_chunks = ffn_local // max_ffn
    acc = None
    for c in range(n_chunks):
      s, e = c * max_ffn, (c + 1) * max_ffn
      part = pallas_swiglu_ffn_local(
          x,
          w_val[s:e],
          w_gate[s:e],
          w_out[:, s:e],
          ws_val.reshape(-1)[s:e] if is_fp8 else None,
          ws_gate.reshape(-1)[s:e] if is_fp8 else None,
          ws_out if is_fp8 else None,
          block_m=block_m,
      )
      acc = part if acc is None else (acc + part)
    return acc

  if is_fp8:
    ws_val_2d = ws_val.reshape(1, ffn_local).astype(jnp.float32)
    ws_gate_2d = ws_gate.reshape(1, ffn_local).astype(jnp.float32)
    ws_out_2d = ws_out.reshape(1, hidden).astype(jnp.float32)
    # Broadcast row 0 to (NUM_SUBLANES=8, cols) so Pallas tile rules are happy
    ws_val_tile = jnp.broadcast_to(ws_val_2d, (8, ffn_local))
    ws_gate_tile = jnp.broadcast_to(ws_gate_2d, (8, ffn_local))
    ws_out_tile = jnp.broadcast_to(ws_out_2d, (8, hidden))
    in_specs = [
        pl.BlockSpec((bm, hidden), lambda i: (i, 0)),
        pl.BlockSpec((ffn_local, hidden), lambda i: (0, 0)),
        pl.BlockSpec((8, ffn_local), lambda i: (0, 0)),
        pl.BlockSpec((ffn_local, hidden), lambda i: (0, 0)),
        pl.BlockSpec((8, ffn_local), lambda i: (0, 0)),
        pl.BlockSpec((hidden, ffn_local), lambda i: (0, 0)),
        pl.BlockSpec((8, hidden), lambda i: (0, 0)),
    ]
    operands = [
        x_2d,
        w_val,
        ws_val_tile,
        w_gate,
        ws_gate_tile,
        w_out,
        ws_out_tile,
    ]
    kernel_fn = _swiglu_ffn_fp8_kernel
  else:
    in_specs = [
        pl.BlockSpec((bm, hidden), lambda i: (i, 0)),
        pl.BlockSpec((ffn_local, hidden), lambda i: (0, 0)),
        pl.BlockSpec((ffn_local, hidden), lambda i: (0, 0)),
        pl.BlockSpec((hidden, ffn_local), lambda i: (0, 0)),
    ]
    operands = [x_2d, w_val, w_gate, w_out]
    kernel_fn = _swiglu_ffn_bf16_kernel

  out_shape = jax.ShapeDtypeStruct((padded_seq, hidden), x.dtype)
  out_spec = pl.BlockSpec((bm, hidden), lambda i: (i, 0))

  out_2d = pl.pallas_call(
      kernel_fn,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_spec,
          grid=(num_tiles,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_splash_vmem_limit_bytes(),
      ),
  )(*operands)

  if pad_len > 0:
    out_2d = out_2d[:seq_len]
  return out_2d.reshape(orig_shape)


def pallas_swiglu_ffn_sharded(
    x: jax.Array,
    w_proj: jax.Array,
    w_out: jax.Array,
    ws_proj: jax.Array | None = None,
    ws_out: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh | None = None,
    block_m: int = 512,
) -> jax.Array:
  """Mesh-aware Pallas SwiGLU FFN megakernel over TP-sharded weights.

  Assumes `w_proj` `(2 * ffn_dim, hidden)` has been permuted into
  `(tp, 2, ffn_dim // tp, hidden)` along axis 0 so each TP rank's local shard
  `w_proj_local` `(2 * ffn_local, hidden)` splits cleanly into
  `w_val = w_proj_local[:ffn_local]` and `w_gate = w_proj_local[ffn_local:]`.
  """
  from jax.sharding import PartitionSpec as P

  is_fp8 = w_proj.dtype == jnp.float8_e4m3fn and ws_proj is not None
  if mesh is not None:
    tp_size = mesh.size
    tp_axis = mesh.axis_names[0]
  else:
    abs_mesh = jax.sharding.get_abstract_mesh()
    tp_size = dict(abs_mesh.shape).get("tp", 1)
    tp_axis = "tp"

  if tp_size == 1:
    ffn_local = w_proj.shape[0] // 2
    w_val, w_gate = w_proj[:ffn_local], w_proj[ffn_local:]
    ws_val = ws_proj[:ffn_local] if is_fp8 else None
    ws_gate = ws_proj[ffn_local:] if is_fp8 else None
    with jax.default_matmul_precision("default"):
      return pallas_swiglu_ffn_local(
          x, w_val, w_gate, w_out, ws_val, ws_gate, ws_out, block_m=block_m
      )

  smap = (
      functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
      if mesh is not None
      else functools.partial(jax.shard_map, check_vma=False)
  )
  scatter_seq = (
      x.ndim == 3
      and x.shape[1] % tp_size == 0
      and os.environ.get("DT_PALLAS_FFN_ALLREDUCE") != "1"
  )
  out_spec = P(None, tp_axis, None) if scatter_seq else P(None, None, None)

  if is_fp8:
    def _local_fp8(x_l, wp_l, wo_l, wsp_l, wso_l):
      fl = wp_l.shape[0] // 2
      with jax.default_matmul_precision("default"):
        part = pallas_swiglu_ffn_local(
            x_l,
            wp_l[:fl],
            wp_l[fl:],
            wo_l,
            wsp_l[:fl],
            wsp_l[fl:],
            wso_l,
            block_m=block_m,
        )
      if scatter_seq:
        return lax.psum_scatter(part, tp_axis, scatter_dimension=1, tiled=True)
      return lax.psum(part, tp_axis)

    return smap(
        _local_fp8,
        in_specs=(
            P(None, None, None),
            P(tp_axis, None),
            P(None, tp_axis),
            P(tp_axis),
            P(),
        ),
        out_specs=out_spec,
    )(x, w_proj, w_out, ws_proj, ws_out)
  else:
    def _local_bf16(x_l, wp_l, wo_l):
      fl = wp_l.shape[0] // 2
      with jax.default_matmul_precision("default"):
        part = pallas_swiglu_ffn_local(
            x_l, wp_l[:fl], wp_l[fl:], wo_l, block_m=block_m
        )
      if scatter_seq:
        return lax.psum_scatter(part, tp_axis, scatter_dimension=1, tiled=True)
      return lax.psum(part, tp_axis)

    return smap(
        _local_bf16,
        in_specs=(P(None, None, None), P(tp_axis, None), P(None, tp_axis)),
        out_specs=out_spec,
    )(x, w_proj, w_out)
