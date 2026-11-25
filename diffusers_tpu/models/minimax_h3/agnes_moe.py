"""Agnes-Video-H3-MoE on the JAX MiniMax-H3 DiT (`--moe <snapshot>`).

Agnes-AI/Agnes-Video-H3-MoE replaces the feed-forward of DiT blocks 15-34 of
MiniMax-H3 FL2VA (the same weights as MiniMax-H3's `transformer`) with an
"E8T1 + shared" mixture of experts:

* one always-on shared SwiGLU (width 3072) and eight private SwiGLUs (6144
  each), built by copying neurons of the dense 14336-wide FFN listed in
  `init.json`, plus a rank-64 LoRA on each of their two projections
  (`step-60.safetensors`, alpha = rank, so the merged weight is W + B @ A);
* a router `argmax(normalize(x) @ R.T / T)` picking one private expert per
  token; the output is `shared(x) + expert[argmax](x)` (no probability weight).

The LoRA is merged on the host (fp32, then bfloat16) and the experts are
stacked. Checked against the released algorithm (DiffSynth layout, unmerged
LoRA) on layer 15, fp32 on the CPU: relative error 6.0e-7.

How the layer is split over the 8 chips/devices (`DT_MOE_DP` / `DT_MOE_EP`):

* TP (default on v6e-8): every expert's width split over `tp`, every chip sees
  every token (an all-gather in front); tokens sorted by expert, megablox
  gmm_v2 with silu fused, the sum over `tp` as a hand-written ring
  reduce-scatter that overlaps the next block's compute (`DT_MOE_RING`).
  480x832x124, 50 steps: 26.4 s per request vs dense 27.5 s; 768p 110.2 vs
  104.3 s (the input all-gather is what it still pays).
* DP (default on chips with >= 64 GiB HBM, v7x-8): the 20 layers' experts
  replicated on every device (31 GiB), rows stay split, no communication at
  all, immune to routing skew. v7x-8: 480p 17.5 s vs dense 19.8; 768p 79.2 vs
  80.0. Optional fp8 experts (`DT_MOE_FP8`).
* EP (opt-in): one expert per device, ragged all-to-all. Fast for even
  routing, but the real routing is very skewed -- in most layers one expert
  takes 55-80% of the rows (L19: 86%; mean max/avg 4.2x), so one device does
  most of the work and EP loses end to end on both chips.

DiffSynth's `fc1` is `[gate; up]` (`silu(gate) * up`); diffusers' `ff.net.0.proj`
is `[up; gate]` (`value * silu(gate)`): `fc1 == concat(proj[W:], proj[:W])`.
"""
import functools
import glob
import json
import os

import numpy as np

MOE_LAYERS = tuple(range(15, 35))
GMM = os.environ.get("DT_MOE_GMM", "1") == "1"
RING = os.environ.get("DT_MOE_RING", "1") == "1"  # 0: one psum after the whole layer   # 0: jax.lax.ragged_dot (expert in-projection then (up, gate))
DP_ENV = os.environ.get("DT_MOE_DP", "auto").strip().lower()  # auto (>=64 GiB HBM, v7x) | 1 | 0
EP = os.environ.get("DT_MOE_EP", "0") == "1"  # 1: expert-parallel (1 expert/chip, ragged_all_to_all, zero-drop)
EP_SLABS = int(os.environ.get("DT_MOE_EP_SLABS", "1"))
EP_TILE_M = int(os.environ.get("DT_MOE_EP_TILE_M", "256"))
EP_TILE_K = int(os.environ.get("DT_MOE_EP_TILE_K", "1024"))
EP_TILE_N_IN = int(os.environ.get("DT_MOE_EP_TILE_N_IN", "1024"))
EP_TILE_N_OUT = int(os.environ.get("DT_MOE_EP_TILE_N_OUT", "1024"))
DP_TILE_M = int(os.environ.get("DT_MOE_DP_TILE_M", "256"))
DP_TILE_K = int(os.environ.get("DT_MOE_DP_TILE_K", "2048"))
DP_TILE_N_IN = int(os.environ.get("DT_MOE_DP_TILE_N_IN", "1024"))
DP_TILE_N_OUT = int(os.environ.get("DT_MOE_DP_TILE_N_OUT", "1024"))
EP_BACKEND = os.environ.get("DT_MOE_EP_BACKEND", "gmm1")  # gmm1 | ragged1
EP_SHARED_POS = os.environ.get("DT_MOE_EP_SHARED", "between")  # between | after
MOE_FP8 = os.environ.get("DT_MOE_FP8", "0") == "1"  # 1: W8A8 e4m3fn experts in DP-8 (halves MoE HBM)
FAST_LOAD = os.environ.get("DT_MOE_FAST_LOAD", "1") == "1"
WIDTH = 14336
EXPERTS = 8


def _snapshot(path):
  path = os.path.expanduser(path)
  if os.path.isdir(os.path.join(path, "FL2VA")) or os.path.exists(os.path.join(path, "init.json")):
    return path
  hits = glob.glob(os.path.join(path, "snapshots", "*"))
  if hits:
    return hits[0]
  raise FileNotFoundError(f"no Agnes-Video-H3-MoE snapshot at {path}")


def _tensor_reader(transformer_dir):
  from safetensors import safe_open
  index = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
  wm = json.load(open(index))["weight_map"]

  def get(name):
    with safe_open(os.path.join(transformer_dir, wm[name]), "pt") as f:
      return f.get_tensor(name).float()
  return get


def build_layer(layer, get_base, spec, lora, tp, expert_gate_first=True):
  """Host weights of one MoE layer in the layout `_moe_local` expects.

  Returns (shared_in (D, 2*FS), shared_out (FS, D), expert_in (E, D, 2*FE),
  expert_out (E, FE, D), router (D, E) float32, temperature). The in-projections'
  columns are interleaved per tensor-parallel shard as (tp, [value, gate], w/tp)
  so each chip's slice holds matching value and gate columns.
  """
  import torch
  p1 = get_base(f"transformer_blocks.{layer}.ff.net.0.proj.weight")   # (2W, D) = [up; gate]
  p2 = get_base(f"transformer_blocks.{layer}.ff.net.2.weight")        # (D, W)

  def merged(name, ids, gate_first=False):
    ids = torch.as_tensor(ids, dtype=torch.long)
    w = ids.numel()
    gate, up = p1[WIDTH + ids], p1[ids]
    pre = f"blocks.{layer}.mlp.{name}"
    d1 = lora[f"{pre}.fc1.lora_B.default.weight"] @ lora[f"{pre}.fc1.lora_A.default.weight"]
    gate, up = gate + d1[:w], up + d1[w:]                    # DiffSynth rows: [gate; up]
    out = p2[:, ids] + lora[f"{pre}.fc2.lora_B.default.weight"] @ lora[f"{pre}.fc2.lora_A.default.weight"]
    if w % tp:
      raise ValueError(f"expert width {w} does not divide over tp={tp}")
    a, b = (gate, up) if gate_first else (up, gate)
    inter = torch.stack((a.reshape(tp, w // tp, -1), b.reshape(tp, w // tp, -1)), 1)
    return inter.reshape(2 * w, -1).T.contiguous(), out.T.contiguous()   # (D, 2w), (w, D)

  s_in, s_out = merged("shared", spec["shared_indices"])
  ex = [merged(f"experts.{e}", spec["private_indices"][e], expert_gate_first) for e in range(EXPERTS)]
  e_in = torch.stack([a for a, _ in ex])
  e_out = torch.stack([b for _, b in ex])
  router = lora[f"blocks.{layer}.mlp.router.weight"].T.contiguous()   # (D, E)
  return s_in, s_out, e_in, e_out, router, float(spec.get("temperature", 1.0))


def _moe_rows(x2, s_in, s_out, e_in, e_out, router, temperature):
  """This chip's partial MoE output for the rows `x2` (before the sum over tp)."""
  import jax
  import jax.numpy as jnp
  xf = x2.astype(jnp.float32)
  xf = xf / jnp.maximum(jnp.linalg.norm(xf, axis=-1, keepdims=True), 1e-12)
  chosen = jnp.argmax(xf @ router / temperature, axis=-1)
  order = jnp.argsort(chosen, stable=True)
  sizes = jnp.bincount(chosen, length=EXPERTS).astype(jnp.int32)
  # Rows back to their places by a gather through the inverse permutation:
  # scattering the (tokens, hidden) result cost 4.2 ms at 57,720 tokens on a
  # v6e chip, the integer scatter + gather 2.3 ms (c907d). ragged_dot beat
  # the megablox gmm kernel at every tiling tried (3.1 vs >= 6 ms).
  inv = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype))
  xs = x2.at[order].get(unique_indices=True, mode="promise_in_bounds")
  if GMM:
    # megablox gmm_v2 with silu(gate) * up fused into the first product (the
    # expert in-projection is laid out (tp, [gate, up], w/tp) for it):
    # 4.63 -> 4.31 ms a layer at 14,976 rows on a v6e-8 (c913).
    from ...kernels import gmm_v2
    bf16 = jnp.dtype(jnp.bfloat16)
    h = gmm_v2.gmm_v2(xs, e_in, sizes, fuse_act="silu", preferred_element_type=bf16)
    y = gmm_v2.gmm_v2(h, e_out, sizes, preferred_element_type=bf16)
  else:
    h = jax.lax.ragged_dot(xs, e_in, sizes)
    v, g = jnp.split(h, 2, axis=-1)
    y = jax.lax.ragged_dot(v * jax.nn.silu(g), e_out, sizes)
  out = y.at[inv].get(unique_indices=True, mode="promise_in_bounds")
  v, g = jnp.split(x2 @ s_in, 2, axis=-1)
  return out + (v * jax.nn.silu(g)) @ s_out


def _moe_local(x, s_in, s_out, e_in, e_out, router, *, temperature, axis):
  """Whole layer on one shard: every row, then one sum over `axis`."""
  import jax
  shape = x.shape
  out = _moe_rows(x.reshape(-1, shape[-1]), s_in, s_out, e_in, e_out, router, temperature)
  return jax.lax.psum(out, axis).reshape(shape)


def _moe_ring(x, s_in, s_out, e_in, e_out, router, *, temperature, axis):
  """Whole layer on one shard, reduce-scattered over `axis` as a ring.

  The rows are cut into one block per chip. At step t each chip computes its
  partial output for block (i - 1 - t) mod n, adds it to the running sum it
  just received from its left neighbour and passes the sum on; after n - 1
  hops chip i holds block i summed over every chip -- rows split over `axis`,
  the layout the residual stream has between layers. Each collective-permute
  overlaps the next block's compute (the dense FFN gets this overlap from
  XLA's collective matmul, which a Pallas gmm in the middle rules out).
  14,976 rows on a v6e-8: 4.35 -> 3.55 ms a layer (c917b; without the input
  all-gather, which the whole layer still pays: 4.33 ms with it); end to end
  480p 27.4 -> 26.4 s. On v7x-8 the ring gains nothing over one psum and
  compiles 33 s longer. Its output differs from the psum version by
  summation order only (DINO 0.952 at 480p on v6e vs MoE's own noise floor
  0.895); carrying the running sum in float32 did not bring it closer
  (0.947) and cost all the speed, so it stays bfloat16.
  """
  import jax
  b, rows, d = x.shape
  n = jax.lax.axis_size(axis)
  m = rows // n
  x2 = x.reshape(-1, d)
  i = jax.lax.axis_index(axis)

  def part(blk):
    xc = jax.lax.dynamic_slice_in_dim(x2, blk * m, m, 0)
    return _moe_rows(xc, s_in, s_out, e_in, e_out, router, temperature)
  perm = [(j, (j + 1) % n) for j in range(n)]
  acc = part((i - 1) % n)
  for t in range(1, n):
    acc = jax.lax.ppermute(acc, axis, perm) + part((i - 1 - t) % n)
  return acc.reshape(b, m, d)


def _ep8_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
  from ...kernels import gmm_v2
  tn = EP_TILE_N_IN if fuse_act is not None else EP_TILE_N_OUT
  return gmm_v2.TileSizes(
      tile_m=min(EP_TILE_M, dims.size_m),
      tile_k=min(EP_TILE_K, dims.size_k),
      tile_n=min(tn, dims.size_n // (2 if fuse_act else 1)),
  )


def _ep8_expert_compute(recvd, e_in, e_out, tot_recv):
  import jax
  import jax.numpy as jnp
  bf16 = jnp.dtype(jnp.bfloat16)
  gs = jnp.array([tot_recv], dtype=jnp.int32)
  if EP_BACKEND == "ragged1":
    gu = jax.lax.ragged_dot(recvd, e_in, gs)
    g, u = jnp.split(gu, 2, axis=-1)
    return jax.lax.ragged_dot(jax.nn.silu(g) * u, e_out, gs)
  from ...kernels import gmm_v2
  h = gmm_v2.gmm_v2(
      recvd, e_in, gs, fuse_act="silu",
      preferred_element_type=bf16, zero_initialize=False,
      tile_info=_ep8_tile_fn, vmem_limit_bytes=32 * 1024 * 1024,
  )
  return gmm_v2.gmm_v2(
      h, e_out, gs,
      preferred_element_type=bf16, zero_initialize=False,
      tile_info=_ep8_tile_fn, vmem_limit_bytes=32 * 1024 * 1024,
  )


def _moe_ep8(x, s_in, s_out, e_in, e_out, router, *, temperature, axis, nslabs=1):
  """Zero-drop Expert-Parallel (EP=8) MoE layer over sequence-sharded `x`.

  Each chip owns one complete private expert (1, D, 2*FE) / (1, FE, D) and a
  replicated copy of the shared SwiGLU (+1.65 GiB/chip across 20 layers).
  Tokens are dispatched and combined via `jax.lax.ragged_all_to_all` with
  full zero-drop capacity (`cap = S // nslabs`), while `gmm_v2(..., zero_initialize=False)`
  only computes the exact `tot_recv` active rows per chip.
  """
  import jax
  import jax.numpy as jnp
  b, m_local, d = x.shape
  n = jax.lax.axis_size(axis)
  xl = x.reshape(-1, d)
  ml = xl.shape[0]
  i = jax.lax.axis_index(axis)

  def _shared(xin):
    v, g = jnp.split(xin @ s_in, 2, axis=-1)
    return (v * jax.nn.silu(g)) @ s_out

  def _prep(xin):
    xf = xin.astype(jnp.float32)
    xf = xf / jnp.maximum(jnp.linalg.norm(xf, axis=-1, keepdims=True), 1e-12)
    chosen = jnp.argmax(xf @ router / temperature, axis=-1)
    order = jnp.argsort(chosen, stable=False)
    send_sz = jnp.bincount(chosen, length=EXPERTS).astype(jnp.int32)
    if os.environ.get("DT_MOE_HIST") == "1":   # diagnosis: rows per expert over the whole sequence
      jax.debug.print("  [moe-hist] chip {c}: rows per expert {h}", c=i,
                      h=jax.lax.psum(send_sz, axis), ordered=False)
    inv = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype))
    xs = xin.at[order].get(unique_indices=True, mode="promise_in_bounds")
    all_cnt = jax.lax.all_gather(send_sz, axis, axis=0, tiled=False)
    return xs, inv, send_sz, all_cnt

  if nslabs <= 1 or ml % nslabs != 0:
    cap = ml * n
    xs, inv, send_sz, all_cnt = _prep(xl)
    in_off = jnp.cumsum(send_sz) - send_sz
    col_off = jnp.cumsum(all_cnt, axis=0) - all_cnt
    row_off = jnp.cumsum(all_cnt, axis=1) - all_cnt
    recv_sz = all_cnt[:, i]
    recv_buf = jnp.empty((cap, d), dtype=x.dtype)
    recvd = jax.lax.ragged_all_to_all(
        xs, recv_buf, in_off, send_sz, col_off[i, :], recv_sz, axis_name=axis)
    if EP_SHARED_POS == "between":
      sh_out = _shared(xl)
    out_e = _ep8_expert_compute(recvd, e_in, e_out, recv_sz.sum().astype(jnp.int32))
    back_buf = jnp.empty((ml, d), dtype=x.dtype)
    back = jax.lax.ragged_all_to_all(
        out_e, back_buf, col_off[:, i], recv_sz, row_off[:, i], send_sz, axis_name=axis)
    if EP_SHARED_POS != "between":
      sh_out = _shared(xl)
    out = back.at[inv].get(unique_indices=True, mode="promise_in_bounds") + sh_out
    return out.reshape(b, m_local, d)

  ms = ml // nslabs
  cap_slab = ms * n
  slabs = [xl[k * ms:(k + 1) * ms] for k in range(nslabs)]

  def _disp(xs, ssz, ac):
    in_off = jnp.cumsum(ssz) - ssz
    co = jnp.cumsum(ac, axis=0) - ac
    rb = jnp.empty((cap_slab, d), dtype=x.dtype)
    return jax.lax.ragged_all_to_all(
        xs, rb, in_off, ssz, co[i, :], ac[:, i], axis_name=axis), co

  def _comb(out_e, ssz, ac, co):
    ro = jnp.cumsum(ac, axis=1) - ac
    bb = jnp.empty((ms, d), dtype=x.dtype)
    return jax.lax.ragged_all_to_all(
        out_e, bb, co[:, i], ac[:, i], ro[:, i], ssz, axis_name=axis)

  preps = [_prep(slabs[0])]
  recvs = [_disp(preps[0][0], preps[0][2], preps[0][3])]
  backs = []
  for k in range(nslabs):
    if k + 1 < nslabs:
      p_next = _prep(slabs[k + 1])
      preps.append(p_next)
      recvs.append(_disp(p_next[0], p_next[2], p_next[3]))
    yk = _ep8_expert_compute(recvs[k][0], e_in, e_out, preps[k][3][:, i].sum().astype(jnp.int32))
    bk = _comb(yk, preps[k][2], preps[k][3], recvs[k][1])
    backs.append(bk)
  sh_out = _shared(xl)
  parts = [backs[k].at[preps[k][1]].get(unique_indices=True, mode="promise_in_bounds")
           for k in range(nslabs)]
  return (jnp.concatenate(parts, axis=0) + sh_out).reshape(b, m_local, d)


def _dp8_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
  from ...kernels import gmm_v2
  tn = DP_TILE_N_IN if fuse_act is not None else DP_TILE_N_OUT
  return gmm_v2.TileSizes(
      tile_m=min(DP_TILE_M, dims.size_m),
      tile_k=min(DP_TILE_K, dims.size_k),
      tile_n=min(tn, dims.size_n // (2 if fuse_act else 1)),
  )


def _dp8_fp8_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
  from ...kernels import gmm_v2
  return gmm_v2.TileSizes(
      tile_m=min(DP_TILE_M, dims.size_m),
      tile_k=min(1024, dims.size_k),
      tile_n=min(1024, dims.size_n // (2 if fuse_act else 1)),
  )


def _moe_dp8(x, s_in, s_out, e_in, e_out, router, *, temperature, e_in_sc=None, e_out_sc=None):
  """Zero-communication Sequence/Data-Parallel (DP=8) MoE layer for large-HBM TPUs (v7x 95 GiB).

  Every chip holds a complete replica of the 20 MoE layers (+31.4 GiB/chip in bf16,
  or +15.7 GiB/chip with DT_MOE_FP8=1) and processes only its local sequence shard
  `x` of shape `(b, S/tp, D)`:
    - 0 bytes ICI collective communication (no all_gather, reduce_scatter, or all_to_all);
    - 1.00x load balance regardless of routing skew (Agnes-Video-H3-MoE real skew is 4.2x-6.9x);
    - exact zero-drop math without cross-chip bf16 partial-sum rounding.
  """
  import jax
  import jax.numpy as jnp
  b, m_local, d = x.shape
  xl = x.reshape(-1, d)
  xf = xl.astype(jnp.float32)
  xf = xf / jnp.maximum(jnp.linalg.norm(xf, axis=-1, keepdims=True), 1e-12)
  chosen = jnp.argmax(xf @ router / temperature, axis=-1)
  order = jnp.argsort(chosen, stable=False)
  sizes = jnp.bincount(chosen, length=EXPERTS).astype(jnp.int32)
  inv = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype))
  xs = xl.at[order].get(unique_indices=True, mode="promise_in_bounds")
  # gmm_v2 requires rows divisible by TPU sublane size (16 on v7x).
  if (GMM or e_in_sc is not None) and (xl.shape[0] % 16 == 0):
    from ...kernels import gmm_v2
    bf16 = jnp.dtype(jnp.bfloat16)
    tf = _dp8_fp8_tile_fn if e_in_sc is not None else _dp8_tile_fn
    h = gmm_v2.gmm_v2(
        xs, e_in, sizes, rhs_scale=e_in_sc, fuse_act="silu",
        preferred_element_type=bf16, maybe_quantize_lhs=(e_in_sc is not None),
        zero_initialize=False, tile_info=tf, vmem_limit_bytes=48 * 1024 * 1024,
    )
    y = gmm_v2.gmm_v2(
        h, e_out, sizes, rhs_scale=e_out_sc,
        preferred_element_type=bf16, maybe_quantize_lhs=(e_out_sc is not None),
        zero_initialize=False, tile_info=tf, vmem_limit_bytes=48 * 1024 * 1024,
    )
  else:
    gu = jax.lax.ragged_dot(xs, e_in, sizes)
    g, u = jnp.split(gu, 2, axis=-1)
    y = jax.lax.ragged_dot(jax.nn.silu(g) * u, e_out, sizes)
  out = y.at[inv].get(unique_indices=True, mode="promise_in_bounds")
  v, g = jnp.split(xl @ s_in, 2, axis=-1)
  return (out + (v * jax.nn.silu(g)) @ s_out).reshape(b, m_local, d)


def make_ff_class():
  from flax import nnx

  class MoEFeedForward(nnx.Module):
    """Drop-in for the block's `ff` (same input and output)."""

    def __init__(self, s_in, s_out, e_in, e_out, router, temperature, mesh,
                 mode="tp", e_in_sc=None, e_out_sc=None):
      self.s_in, self.s_out = nnx.Param(s_in), nnx.Param(s_out)
      self.e_in, self.e_out = nnx.Param(e_in), nnx.Param(e_out)
      self.router = nnx.Param(router)
      self.e_in_sc = nnx.Param(e_in_sc) if e_in_sc is not None else None
      self.e_out_sc = nnx.Param(e_out_sc) if e_out_sc is not None else None
      self.temperature = temperature
      self.mesh = mesh
      self.mode = mode
      self.seq_sharded_input = mode in ("dp", "ep")

    def __call__(self, x):
      import jax
      from jax.sharding import PartitionSpec as P
      tp = dict(self.mesh.shape)["tp"]
      if self.mode == "dp" and x.shape[0] == 1 and x.shape[1] % tp == 0:
        if self.e_in_sc is not None:
          def _dp_fp8(x_, si_, so_, ei_, eo_, r_, eisc_, eosc_):
            return _moe_dp8(x_, si_, so_, ei_, eo_, r_, temperature=self.temperature,
                            e_in_sc=eisc_, e_out_sc=eosc_)
          fn = jax.shard_map(
              _dp_fp8, mesh=self.mesh,
              in_specs=(P(None, "tp", None), P(), P(), P(), P(), P(), P(), P()),
              out_specs=P(None, "tp", None), check_vma=False)
          return fn(x, self.s_in[...], self.s_out[...], self.e_in[...], self.e_out[...],
                    self.router[...], self.e_in_sc[...], self.e_out_sc[...])
        fn = jax.shard_map(
            functools.partial(_moe_dp8, temperature=self.temperature),
            mesh=self.mesh,
            in_specs=(P(None, "tp", None), P(), P(), P(), P(), P()),
            out_specs=P(None, "tp", None), check_vma=False)
        return fn(x, self.s_in[...], self.s_out[...], self.e_in[...], self.e_out[...],
                  self.router[...])
      if self.mode == "ep" and x.shape[0] == 1 and x.shape[1] % tp == 0:
        fn = jax.shard_map(
            functools.partial(_moe_ep8, temperature=self.temperature, axis="tp",
                              nslabs=EP_SLABS),
            mesh=self.mesh,
            in_specs=(P(None, "tp", None), P(), P(),
                      P("tp", None, None), P("tp", None, None), P()),
            out_specs=P(None, "tp", None), check_vma=False)
        return fn(x, self.s_in[...], self.s_out[...], self.e_in[...], self.e_out[...],
                  self.router[...])
      # gmm_v2 wants each ring block's rows a multiple of 16 (104,000 / 8 =
      # 13,000 crashed on v7x); otherwise fall back to one psum.
      ring = (RING and x.shape[0] == 1 and x.shape[1] % tp == 0
              and (x.shape[1] // tp) % 16 == 0)
      fn = jax.shard_map(
          functools.partial(_moe_ring if ring else _moe_local,
                            temperature=self.temperature, axis="tp"),
          mesh=self.mesh,
          in_specs=(P(), P(None, "tp"), P("tp", None), P(None, None, "tp"),
                    P(None, "tp", None), P()),
          out_specs=P(None, "tp", None) if ring else P(), check_vma=False)
      return fn(x, self.s_in[...], self.s_out[...], self.e_in[...], self.e_out[...],
                self.router[...])
  return MoEFeedForward


def install(transformer, snapshot, base_transformer_dir, mesh, dtype=None, verbose=print):
  """Replace blocks 15-34's `ff` with the Agnes MoE (weights on `mesh`)."""
  from concurrent.futures import ThreadPoolExecutor
  import time
  import jax
  import jax.numpy as jnp
  import torch
  from jax.sharding import NamedSharding, PartitionSpec as P
  from safetensors import safe_open
  t0 = time.perf_counter()
  dtype = dtype or jnp.bfloat16
  snap = _snapshot(snapshot)
  init = json.load(open(os.path.join(snap, "init.json")))
  with safe_open(os.path.join(snap, "step-60.safetensors"), "pt") as f:
    lora_all = {k: f.get_tensor(k).float() for k in f.keys()}
  get_base = _tensor_reader(base_transformer_dir)
  tp = dict(mesh.shape).get("tp", 1)
  _hbm_gib = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 2**30
  use_dp = (DP_ENV == "1") or (DP_ENV == "auto" and not EP and tp > 1 and _hbm_gib >= 64.0)
  if use_dp and tp > 1:
    mode = "dp"
  elif EP and tp == EXPERTS:
    mode = "ep"
  else:
    mode = "tp"
  use_fp8 = (mode == "dp") and MOE_FP8
  build_tp = 1 if mode in ("dp", "ep") else tp
  cls = make_ff_class()
  if mode == "dp":
    specs = (P(), P(), P(), P(), P())
  elif mode == "ep":
    specs = (P(), P(), P("tp", None, None), P("tp", None, None), P())
  else:
    specs = (P(None, "tp"), P("tp", None), P(None, None, "tp"), P(None, "tp", None), P())

  def put(t, spec, dt):
    if FAST_LOAD and jnp.dtype(dt) == jnp.dtype(jnp.bfloat16):
      arr = t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(jnp.bfloat16)
      return jax.device_put(arr, NamedSharding(mesh, spec))
    arr = t.to(torch.float32).numpy()
    return jax.device_put(jnp.asarray(arr, dt), NamedSharding(mesh, spec))

  @jax.jit
  def _quant_fp8_dev(w):
    w32 = w.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(w32), axis=1, keepdims=True), jnp.float32(1e-8))
    scale = amax / jnp.float32(448.0)
    qw = jnp.clip(w32 / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return qw, scale[:, None, :, :]

  def put_fp8_expert(t, spec):
    w_j = put(t, spec, dtype)
    return _quant_fp8_dev(w_j)

  def _build_one(layer):
    lora = {k: v for k, v in lora_all.items() if k.startswith(f"blocks.{layer}.mlp.")}
    return build_layer(layer, get_base, init[str(layer)], lora, build_tp,
                       expert_gate_first=(True if mode in ("dp", "ep") else GMM))

  def _install_layer(layer, ws, temp):
    if use_fp8:
      s_in_j = put(ws[0], specs[0], dtype)
      s_out_j = put(ws[1], specs[1], dtype)
      e_in_j, e_in_sc = put_fp8_expert(ws[2], specs[2])
      e_out_j, e_out_sc = put_fp8_expert(ws[3], specs[3])
      r_j = put(ws[4], specs[4], jnp.float32)
      transformer.transformer_blocks[layer].ff = cls(
          s_in_j, s_out_j, e_in_j, e_out_j, r_j, temp, mesh,
          mode=mode, e_in_sc=e_in_sc, e_out_sc=e_out_sc)
    else:
      dts = (dtype, dtype, dtype, dtype, jnp.float32)
      arrs = [put(w, s, d) for w, s, d in zip(ws, specs, dts)]
      transformer.transformer_blocks[layer].ff = cls(*arrs, temp, mesh, mode=mode)

  if FAST_LOAD:
    prev_threads = torch.get_num_threads()
    workers = min(4, len(MOE_LAYERS))
    torch.set_num_threads(max(1, prev_threads // workers))
    try:
      with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [(layer, pool.submit(_build_one, layer)) for layer in MOE_LAYERS]
        for layer, fut in futs:
          *ws, temp = fut.result()
          _install_layer(layer, ws, temp)
    finally:
      torch.set_num_threads(prev_threads)
  else:
    for layer in MOE_LAYERS:
      *ws, temp = _build_one(layer)
      _install_layer(layer, ws, temp)

  if mode == "dp":
    fp8_s = ", fp8-w8a8" if use_fp8 else ""
    mode_s = f"DP={tp} (0-ICI replicated, gmm={int(GMM)}, tile={DP_TILE_M}x{DP_TILE_K}x{DP_TILE_N_IN}{fp8_s})"
  elif mode == "ep":
    mode_s = f"EP={tp} (slabs={EP_SLABS}, {EP_BACKEND})"
  else:
    mode_s = f"TP={tp} (ring={int(RING)}, gmm={int(GMM)})"
  verbose(f"  [agnes-moe] blocks {MOE_LAYERS[0]}-{MOE_LAYERS[-1]}: shared 3072 + "
          f"{EXPERTS} x 6144 top-1, {mode_s}, LoRA merged ({time.perf_counter() - t0:.1f}s)", flush=True)
  return transformer
