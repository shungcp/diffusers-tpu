"""MiniMax-H3's DiT in native JAX (flax nnx).

`torchax_pipeline.py` traces the diffusers module; this native path writes
the layers out. The model is 33 B at the shipped config -- 50 blocks of
hidden 5376, 56 heads of 128, FFN 14336. Everything computed from tensor
*contents* runs in a host prologue, so the module is jittable with all of
it static.

Running this file checks a small random configuration against diffusers
on CPU:

    JAX_PLATFORMS=cpu python -m diffusers_tpu.models.minimax_h3.jax_dit

Three details of the PyTorch reference that are easy to get wrong:

* **The modulation is not per-sample, it is per `(timestep, modality)`.**
  `adaln_proj` projects one shared `temb` into `6 * hidden * 3` and the
  block indexes the result with `timestep_indices * 3 + token_tags`, so
  text, audio and video rows of the same step read different shift,
  scale and gate. That gather is the reason the block takes
  `adaln_indices` at all.
* **`silu(temb)` runs in float32, the projection in bfloat16.** The
  checkpoint is mixed precision and `time_embedder` is a float32 module.
  Rounding before the activation biases every block identically at every
  step, and that bias accumulates along the trajectory rather than
  cancelling.
* **RoPE rotates only the leading `rotary_dim` channels** and passes the
  rest through. With `rope_freq_dim` 16 over three axes the rotated part
  is 96 of the 128, so a third of every head is untouched.
"""

from __future__ import annotations

import functools
import math
import os

import numpy as np

# The shipped configuration, from the checkpoint's `transformer/config.json`.
CONFIG = dict(
    num_attention_heads=56,
    attention_head_dim=128,
    hidden_size=5376,
    num_layers=50,
    ffn_dim=14336,
    time_embed_dim=2688,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
)
MODALITY_NUM = 3  # text, audio, video

# The full shipped config, as the checkpoint's `config.json` has it.
CONFIG_FULL = dict(
    num_attention_heads=56, attention_head_dim=128, hidden_size=5376,
    num_layers=50, num_refiner_layers=2, ffn_dim=14336, in_channels=24,
    audio_in_channels=32, patch_size=(1, 2, 2), text_dim=5120, freq_dim=256,
    time_embed_hidden_dim=5376, time_embed_dim=2688, rope_freq_dim=16,
    rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)

IMAGE_CONFIG_FULL = dict(
    CONFIG_FULL,
    audio_in_channels=0,
    num_pdd_heads=6,
)


# Rotate all 128 lanes rather than slicing the leading 96 (see
# `apply_rope_full_lane`). Off by default so the CPU equivalence check
# compares the plain transcription.
FULL_LANE_ROPE = os.environ.get("DT_FULL_LANE_ROPE") == "1"

# Round the packed sequence up to a multiple of this before the block
# stack runs, and hand attention the true length. 0 is off. Linears and
# elementwise ops are row-independent, so padding rows cannot affect real
# ones; attention gets k and v sliced back to the true length. On 4 chips
# padding 57,290 tokens to 57,344 cut a step from 4.94 s to 3.64 s.
PAD_SEQUENCE_TO = int(os.environ.get("DT_PAD_SEQUENCE_TO", "0"))

# Sharding of the residual stream between layers (`DT_STREAM_SHARD`):
#
# * unset -- the SPMD partitioner decides (default, fastest);
# * `seq` -- pin it to `P(None, "tp", None)` across the norms, modulation
#   and gated adds, and gather it whole before attention and the two
#   column-parallel linears (as `install_sequence_parallel` does for
#   torchax);
# * `whole` -- replicate it everywhere.
#
# `seq` is ~6% slower on 8 chips: attention and `ff.proj` both want whole
# rows, so pinning the stream doubles the per-layer all-gather instead of
# removing it.
_STREAM = os.environ.get("DT_STREAM_SHARD", "")


# DT_CMM=1: hand-written collective matmul for the qkv and first FF
# projections. Instead of all-gathering the sequence-sharded `x` before a
# column-sharded matmul, the shards of `x` are rotated around the ring in
# a shard_map and each is multiplied as it arrives, interleaving ppermute
# with compute. Mathematically identical to the full matmul.
CMM = os.environ.get("DT_CMM") == "1"


def _cmm(x, weights):
  """`[x @ w.T for w in weights]`, x sharded on sequence, w on output dim.

  `w` is sharded `P("tp", None)`. The output is sharded on its last axis,
  the same layout the partitioner would produce.
  """
  import jax
  import jax.numpy as jnp
  from jax.sharding import PartitionSpec as P

  def local(xl, *wl):
    n = jax.lax.axis_size("tp")
    me = jax.lax.axis_index("tp")
    sl = xl.shape[1]
    accs = [jnp.zeros((xl.shape[0], sl * n, w.shape[0]), xl.dtype) for w in wl]
    xc = xl
    perm = [(j, (j + 1) % n) for j in range(n)]
    for step in range(n):          # unrolled so XLA can overlap the next transfer with this matmul
      src = (me - step) % n        # xc currently holds shard `src`
      nxt = jax.lax.ppermute(xc, "tp", perm) if step + 1 < n else None
      for a_i, w in enumerate(wl):
        y = jax.lax.dot_general(xc, w, (((2,), (1,)), ((), ())))
        accs[a_i] = jax.lax.dynamic_update_slice(accs[a_i], y.astype(xl.dtype),
                                                 (0, src * sl, 0))
      xc = nxt
    return tuple(accs)

  outs = jax.shard_map(
      local, in_specs=(P(None, "tp", None),) + (P("tp", None),) * len(weights),
      out_specs=(P(None, None, "tp"),) * len(weights), check_vma=False)(x, *weights)
  return list(outs)


def _tp_size():
  """Size of the `tp` axis of the mesh in context (1 without one)."""
  import jax
  return dict(jax.sharding.get_abstract_mesh().shape).get("tp", 1)


def _constrain(x, *spec):
  """`with_sharding_constraint`, or the identity when the flag is off."""
  if not _STREAM:
    return x
  import jax
  from jax.sharding import PartitionSpec as P
  if _STREAM == "whole":
    spec = (None,) * len(spec)
  return jax.lax.with_sharding_constraint(x, P(*spec))

# DT_DUMP_BLOCKS=<dir>: debug dump of per-block activation statistics
# (norm, mean, std and a strided slice) for comparing two runs layer by
# layer. Full activations are ~1.3 GB each, so only summaries are saved.
_BLK = {"n": -1}


def _blk_save(tag, x):
  d = os.environ.get("DT_DUMP_BLOCKS")
  if not d:
    return
  import numpy as _np
  if tag == "x0":
    _BLK["n"] += 1
  if _BLK["n"] != int(os.environ.get("DT_DUMP_BLOCK_CALL", "0")):
    return
  os.makedirs(d, exist_ok=True)
  a = _np.asarray(x, _np.float32)
  _np.savez(os.path.join(d, f"{tag}.npz"),
            norm=_np.linalg.norm(a.astype(_np.float64)),
            mean=a.mean(), std=a.std(),
            slice=a.reshape(-1)[::max(1, a.size // 4096)].copy())
  print(f"  [blk] {tag} norm={_np.linalg.norm(a.astype(_np.float64)):.6g}",
        flush=True)


def _build_modules():
  """Imports flax lazily, so this file can be read without it installed."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  class RMSNorm(nnx.Module):
    """Scale is `weight`; the reduction is in float32 whatever `x` is."""

    def __init__(self, dim, eps=1e-5, *, dtype=jnp.bfloat16, rngs=None):
      self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      dtype = x.dtype
      x32 = x.astype(jnp.float32)
      rrms = jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + self.eps)
      return (x32 * rrms * self.weight[...]).astype(dtype)

  class Linear(nnx.Module):
    """`y = x @ W.T (+ b)`, weight stored `(out, in)` as torch has it.

    `precision` is for `TimestepEmbedder`: TPU runs float32 matmuls in
    bfloat16 by default, and at timestep 0 the embedding is a cancelling
    sum that bfloat16 moves by ~4%. It is not set globally because
    `JAX_DEFAULT_MATMUL_PRECISION=highest` breaks the Pallas splash kernel.
    """

    def __init__(self, in_f, out_f, *, bias=False, dtype=jnp.bfloat16,
                 rngs=None, precision=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype))
      self.bias = nnx.Param(jnp.zeros((out_f,), dtype)) if bias else None
      self.weight_scale = nnx.data(None)
      self.precision = precision

    def __call__(self, x):
      w = self.weight[...]
      if w.dtype == jnp.int8 and self.weight_scale is not None and INT8_ACT:
        # W8A8 (DT_INT8_ACT=1): per-row int8 activations into the int8 MXU
        # path (v5e: twice the bf16 rate), int32 accumulation.
        x32 = x.astype(jnp.float32)
        xs = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True),
                         jnp.float32(1e-12)) / 127.0
        xq = jnp.clip(jnp.round(x32 / xs), -127, 127).astype(jnp.int8)
        acc = jax.lax.dot_general(xq, w, (((x.ndim - 1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.int32)
        y = (acc.astype(jnp.float32) * xs * self.weight_scale[...]).astype(x.dtype)
        return y + self.bias[...] if self.bias is not None else y
      if w.dtype == jnp.int8 and self.weight_scale is not None:
        # Weight-only int8 (`load_transformer(int8=True)`): half the bytes
        # to hold and to stream from host; the matmul stays bfloat16.
        y = jax.lax.dot_general(x, w.astype(x.dtype),
                                (((x.ndim - 1,), (1,)), ((), ())))
        y = (y.astype(jnp.float32) * self.weight_scale[...]).astype(x.dtype)
        return y + self.bias[...] if self.bias is not None else y
      if w.dtype == jnp.float8_e4m3fn and self.weight_scale is not None:
        f8_max = jnp.float32(448.0)
        x32 = x.astype(jnp.float32)
        row_max = jnp.maximum(
            jnp.max(jnp.abs(x32), axis=-1, keepdims=True), jnp.float32(1e-12))
        x_scale = row_max / f8_max
        q_x = jnp.clip(x32 / x_scale, -f8_max, f8_max).astype(
            jnp.float8_e4m3fn)
        acc = jax.lax.dot_general(
            q_x, w, (((x.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32)
        y = (acc * x_scale * self.weight_scale[...]).astype(x.dtype)
        return y + self.bias[...] if self.bias is not None else y
      y = jax.lax.dot_general(x, w,
                              (((x.ndim - 1,), (1,)), ((), ())),
                              precision=self.precision)
      return y + self.bias[...] if self.bias is not None else y

  def apply_rope(x, cos, sin):
    """Rotate the leading `cos.shape[-1]` channels, pass the rest through.

    `x` is `(batch, seq, heads, head_dim)`; `cos`/`sin` are `(seq, rotary)`.
    """
    rotary = cos.shape[-1]
    rotated, passed = x[..., :rotary], x[..., rotary:]
    cos = cos.astype(x.dtype)[None, :, None, :]
    sin = sin.astype(x.dtype)[None, :, None, :]
    x1, x2 = jnp.split(rotated, 2, axis=-1)
    turned = jnp.concatenate([-x2, x1], axis=-1)
    return jnp.concatenate([rotated * cos + turned * sin, passed], axis=-1)

  @functools.lru_cache(maxsize=8)
  def rotate_half_matrix(rotary, head_dim):
    """R with `x @ R == cat(-x2, x1, 0)`, x1/x2 the halves of x[:rotary].

    One +-1 per column, so the bf16 matmul is exact.

    Built in numpy and converted at the use site: the `lru_cache` outlives
    a trace, and a cached `jnp` array would leak a tracer into later
    traces (`UnexpectedTracerError`).
    """
    index = np.arange(rotary // 2)
    r = np.zeros((head_dim, head_dim), np.float32)
    r[index + rotary // 2, index] = -1.0
    r[index, index + rotary // 2] = 1.0
    return r

  def apply_rope_full_lane(x, cos, sin):
    """The same rotation over all 128 lanes instead of the leading 96.

    96 of 128 channels rotate, and slicing them, splitting in half and
    concatenating back lands on none of the 128-lane vector register
    boundaries, so XLA spends the time on relayouts rather than on the
    multiply-add. Padding the tables (cos with 1, sin with 0) and
    rotating by a signed permutation matmul keeps every operation on
    whole vregs. The arithmetic is identical (~4% faster on the torchax
    backend).
    """
    rotary, head_dim = cos.shape[-1], x.shape[-1]
    cos = jnp.pad(cos.astype(x.dtype), ((0, 0), (0, head_dim - rotary)),
                  constant_values=1.0)[None, :, None, :]
    sin = jnp.pad(sin.astype(x.dtype), ((0, 0), (0, head_dim - rotary)),
                  constant_values=0.0)[None, :, None, :]
    rotate = jnp.asarray(rotate_half_matrix(rotary, head_dim), x.dtype)
    return x * cos + (x @ rotate) * sin

  class AdaLNModulation(nnx.Module):
    """One `temb` into six modulation tensors, per `(timestep, modality)`."""

    def __init__(self, time_embed_dim, hidden_size, *, dtype=jnp.bfloat16,
                 rngs=None):
      self.hidden_size = hidden_size
      self.linear = Linear(time_embed_dim, 6 * hidden_size * MODALITY_NUM,
                           bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, temb):
      # float32 through the activation, the projection's dtype after it.
      activated = jax.nn.silu(temb.astype(jnp.float32))
      w_dtype = self.linear.weight[...].dtype
      in_dtype = jnp.bfloat16 if w_dtype == jnp.float8_e4m3fn else w_dtype
      out = self.linear(activated.astype(in_dtype))
      out = out.reshape(-1, 6 * self.hidden_size)
      return jnp.split(out, 6, axis=-1)

  class Attention(nnx.Module):
    """Full self-attention over the one packed sequence. No mask, no cross."""

    def __init__(self, hidden_size, heads, dim_head, qk_norm_eps=1e-5, *,
                 dtype=jnp.bfloat16, rngs=None):
      self.heads = heads
      self.head_dim = dim_head
      inner = heads * dim_head
      self.to_q = Linear(hidden_size, inner, dtype=dtype, rngs=rngs)
      self.to_k = Linear(hidden_size, inner, dtype=dtype, rngs=rngs)
      self.to_v = Linear(hidden_size, inner, dtype=dtype, rngs=rngs)
      self.norm_q = RMSNorm(dim_head, qk_norm_eps, dtype=dtype, rngs=rngs)
      self.norm_k = RMSNorm(dim_head, qk_norm_eps, dtype=dtype, rngs=rngs)
      self.to_out = Linear(inner, hidden_size, dtype=dtype, rngs=rngs)

    def __call__(self, x, rope, sdpa=None, true_len=None):
      shape = x.shape[:-1] + (self.heads, self.head_dim)
      turn = apply_rope_full_lane if FULL_LANE_ROPE else apply_rope
      if QKV_CHUNK and rope is not None and x.shape[1] > QKV_CHUNK:
        # DT_QKV_CHUNK=<rows>: projections, q/k norm and rope a slab of rows
        # at a time. Their float32 intermediates are 3.2 GB each at 768p
        # ref2va (118,784 rows), one too many for a 16 GB v5e chip.
        c, b, s_ = QKV_CHUNK, x.shape[0], x.shape[1]
        n = -(-s_ // c)
        pad = n * c - s_
        xp = jnp.pad(x, ((0, 0), (0, pad), (0, 0))) if pad else x
        cos, sin = (jnp.pad(t, ((0, pad), (0, 0))) if pad else t for t in rope)
        xs = xp.reshape(b, n, c, x.shape[2]).swapaxes(0, 1)
        cs = cos.reshape(n, c, -1)
        ss = sin.reshape(n, c, -1)

        def one(args):
          xc, cc, sc = args
          sh = xc.shape[:-1] + (self.heads, self.head_dim)
          qc = turn(self.norm_q(self.to_q(xc).reshape(sh)), cc, sc)
          kc = turn(self.norm_k(self.to_k(xc).reshape(sh)), cc, sc)
          return qc, kc, self.to_v(xc).reshape(sh)
        q, k, v = (t.swapaxes(0, 1).reshape((b, n * c) + shape[2:])[:, :s_]
                   for t in jax.lax.map(one, (xs, cs, ss)))
      else:
        if CMM and rope is not None and x.shape[1] % _tp_size() == 0:
          qx, kx, vx = _cmm(x, [self.to_q.weight[...], self.to_k.weight[...],
                                self.to_v.weight[...]])
        else:
          qx, kx, vx = self.to_q(x), self.to_k(x), self.to_v(x)
        q = self.norm_q(qx.reshape(shape))
        k = self.norm_k(kx.reshape(shape))
        v = vx.reshape(shape)
        if rope is not None:
          q, k = turn(q, *rope), turn(k, *rope)
      dynamic = true_len is not None and not isinstance(true_len, int)
      if dynamic:
        # A traced true length (`jax_server`'s bucketed shapes): keep the
        # padded keys but zero them, and let the kernel mask them by
        # `kv_len` -- one program for every length in the bucket.
        keep = (jnp.arange(x.shape[1]) < true_len)[None, :, None, None]
        k, v = jnp.where(keep, k, 0), jnp.where(keep, v, 0)
      elif true_len is not None and true_len != x.shape[1]:
        # The padded rows are keys to nobody. Queries stay padded: the
        # kernel pads the query axis itself anyway and those output rows
        # are never read.
        k, v = k[:, :true_len], v[:, :true_len]
      # `(b, s, h, d)` -> the kernels want `(b, h, s, d)`.
      q, k, v = (jnp.swapaxes(t, 1, 2) for t in (q, k, v))
      if dynamic:
        out = sdpa(q, k, v, kv_len=true_len)
        out = jnp.swapaxes(out, 1, 2).reshape(x.shape[:-1] + (-1,))
        return self.to_out(out.astype(x.dtype))
      if sdpa is None:
        scale = self.head_dim ** -0.5
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32),
                            k.astype(jnp.float32)) * scale
        out = jnp.einsum("bhqk,bhkd->bhqd",
                         jax.nn.softmax(scores, axis=-1),
                         v.astype(jnp.float32)).astype(q.dtype)
      else:
        out = sdpa(q, k, v)
      out = jnp.swapaxes(out, 1, 2).reshape(x.shape[:-1] + (-1,))
      return self.to_out(out.astype(x.dtype))

  class FeedForward(nnx.Module):
    """SwiGLU: one projection to `2 * ffn_dim`, split, `a * silu(b)`."""

    def __init__(self, hidden_size, ffn_dim, *, dtype=jnp.bfloat16, rngs=None):
      self.proj = Linear(hidden_size, 2 * ffn_dim, dtype=dtype, rngs=rngs)
      self.out = Linear(ffn_dim, hidden_size, dtype=dtype, rngs=rngs)
      self.swiglu_tp = 1

    def __call__(self, x):
      if (
          os.environ.get("DT_PALLAS_FFN") == "1"
          and getattr(self, "swiglu_tp", 1) == _tp_size()
          and x.shape[1] >= 1024
      ):
        from ...kernels import pallas_ffn
        ws_proj = (
            self.proj.weight_scale[...]
            if self.proj.weight_scale is not None
            and self.proj.weight_scale[...] is not None
            else None
        )
        ws_out = (
            self.out.weight_scale[...]
            if self.out.weight_scale is not None
            and self.out.weight_scale[...] is not None
            else None
        )
        bm = int(os.environ.get("DT_PALLAS_FFN_BM", "512"))
        return pallas_ffn.pallas_swiglu_ffn_sharded(
            x,
            self.proj.weight[...],
            self.out.weight[...],
            ws_proj,
            ws_out,
            block_m=bm,
        )
      if FF_CHUNK and x.shape[1] > FF_CHUNK:
        # DT_FF_CHUNK=<rows>: the feed-forward a slab of rows at a time. Its
        # (seq, 2 x 14336) intermediate is 6 GB at 768p/345 frames, which
        # tips a 16 GB v5e chip over.
        n = -(-x.shape[1] // FF_CHUNK)
        pad = n * FF_CHUNK - x.shape[1]
        xp = jnp.pad(x, ((0, 0), (0, pad), (0, 0))) if pad else x
        xs = xp.reshape(x.shape[0], n, FF_CHUNK, x.shape[2]).swapaxes(0, 1)

        def one(xc):
          value, gate = jnp.split(self.proj(xc), 2, axis=-1)
          return self.out(value * jax.nn.silu(gate))
        ys = jax.lax.map(one, xs)
        return ys.swapaxes(0, 1).reshape(x.shape[0], n * FF_CHUNK, -1)[:, :x.shape[1]]
      if CMM and x.shape[1] % _tp_size() == 0 and x.shape[1] > 50000:   # skip the token refiner
        (y,) = _cmm(x, [self.proj.weight[...]])
      else:
        y = self.proj(x)
      tp = getattr(self, "swiglu_tp", 1)
      if tp > 1:
        yr = y.reshape(*y.shape[:-1], tp, 2, -1)
        value, gate = yr[..., 0, :], yr[..., 1, :]
        return self.out((value * jax.nn.silu(gate)).reshape(*x.shape[:-1], -1))
      value, gate = jnp.split(y, 2, axis=-1)
      return self.out(value * jax.nn.silu(gate))

  class Block(nnx.Module):
    """Pre-norm attention and feed-forward, each modulated per row."""

    def __init__(self, hidden_size, num_attention_heads, attention_head_dim,
                 ffn_dim, time_embed_dim, norm_eps, qk_norm_eps, *,
                 dtype=jnp.bfloat16, rngs=None):
      self.norm1 = RMSNorm(hidden_size, norm_eps, dtype=dtype, rngs=rngs)
      self.attn = Attention(hidden_size, num_attention_heads,
                            attention_head_dim, qk_norm_eps, dtype=dtype,
                            rngs=rngs)
      self.norm2 = RMSNorm(hidden_size, norm_eps, dtype=dtype, rngs=rngs)
      self.ff = FeedForward(hidden_size, ffn_dim, dtype=dtype, rngs=rngs)
      self.adaln_proj = AdaLNModulation(time_embed_dim, hidden_size,
                                        dtype=dtype, rngs=rngs)

    def __call__(self, x, temb, adaln_indices, rope, sdpa=None,
                 segments=None, true_len=None, mod=None):
      if mod is None:
        mods = self.adaln_proj(temb)
      else:
        # Precomputed on the host (`load_transformer(adaln_temb=)`): the
        # modulation depends on the timestep levels alone.
        mods = jnp.split(mod.reshape(-1, 6 * self.adaln_proj.hidden_size), 6,
                         axis=-1)
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

      def apply(h, combine):
        """Modulate by run rather than by row.

        Gathering per row with `jnp.take` materialises a `(seq, hidden)`
        tensor six times a block. Instead the last run (the video rows,
        ~98% of the sequence) is broadcast over the whole tensor and the
        few leading runs are written back over the front. Same arithmetic,
        nothing of sequence length allocated.
        """
        if segments is None:
          return combine(h, None)
        *head_runs, (_, _, bulk) = segments
        out = combine(h, bulk)
        if not head_runs:
          return out
        head_width = sum(stop - start for start, stop, _ in head_runs)
        if head_width >= 512:
          pos = jnp.arange(h.shape[1])[None, :, None]
          for start, stop, index in head_runs:
            out = jnp.where((pos >= start) & (pos < stop), combine(h, index), out)
          return out
        head = jnp.concatenate(
            [combine(h[:, start:stop], index)
             for start, stop, index in head_runs], axis=1)
        return out.at[:, :head.shape[1]].set(head)

      def rows(param, index):
        return param[index] if index is not None else jnp.take(
            param, adaln_indices, axis=0)

      # `_constrain` is the identity unless DT_STREAM_SHARD is set; see
      # `_STREAM`. Attention and the two column-parallel linears
      # (`to_q/k/v`, `ff.proj`) want whole rows, everything else here is
      # row-independent and runs eight times narrower when cut.
      x = _constrain(x, None, "tp", None)
      residual = x
      h = apply(self.norm1(x),
                lambda t, i: t * (1.0 + rows(scale_msa, i)) + rows(shift_msa, i))
      attended = self.attn(_constrain(h, None, None, None), rope, sdpa,
                           true_len)
      x = residual + apply(_constrain(attended, None, "tp", None),
                           lambda t, i: t * rows(gate_msa, i))

      residual = x
      h = apply(self.norm2(x),
                lambda t, i: t * (1.0 + rows(scale_mlp, i)) + rows(shift_mlp, i))
      ff = _constrain(self.ff(_constrain(h, None, None, None)),
                      None, "tp", None)
      return residual + apply(ff, lambda t, i: t * rows(gate_mlp, i))

  class TokenRefinerBlock(nnx.Module):
    """Plain pre-norm block over the text stream. No AdaLN, no rotary."""

    def __init__(self, hidden_size, heads, head_dim, ffn_dim, norm_eps,
                 qk_norm_eps, *, dtype=jnp.bfloat16, rngs=None):
      self.norm1 = RMSNorm(hidden_size, norm_eps, dtype=dtype, rngs=rngs)
      self.attn = Attention(hidden_size, heads, head_dim, qk_norm_eps,
                            dtype=dtype, rngs=rngs)
      self.norm2 = RMSNorm(hidden_size, norm_eps, dtype=dtype, rngs=rngs)
      self.ff = FeedForward(hidden_size, ffn_dim, dtype=dtype, rngs=rngs)

    def __call__(self, x, sdpa=None, text_len=None):
      x = x + self.attn(self.norm1(x), None, sdpa, text_len)
      return x + self.ff(self.norm2(x))

  class TokenRefiner(nnx.Module):
    def __init__(self, hidden_size, heads, head_dim, ffn_dim, layers,
                 norm_eps, qk_norm_eps, final_norm_eps, *,
                 dtype=jnp.bfloat16, rngs=None):
      self.refiner_blocks = nnx.List([
          TokenRefinerBlock(hidden_size, heads, head_dim, ffn_dim, norm_eps,
                            qk_norm_eps, dtype=dtype, rngs=rngs)
          for _ in range(layers)])
      self.final_norm = RMSNorm(hidden_size, final_norm_eps, dtype=dtype,
                                rngs=rngs)

    def __call__(self, x, sdpa=None, text_len=None):
      for block in self.refiner_blocks:
        x = block(x, sdpa, text_len)
      return self.final_norm(x)

  class AdaLNOut(nnx.Module):
    """Final norm, shift and scale selected per row by its timestep."""

    def __init__(self, hidden_size, time_embed_dim, eps, *,
                 dtype=jnp.bfloat16, rngs=None):
      self.norm = RMSNorm(hidden_size, eps, dtype=dtype, rngs=rngs)
      self.linear = Linear(time_embed_dim, 2 * hidden_size, bias=True,
                           dtype=dtype, rngs=rngs)

    def __call__(self, x, temb, timestep_indices):
      activated = jax.nn.silu(temb.astype(jnp.float32))
      shift, scale = jnp.split(
          self.linear(activated.astype(self.linear.weight[...].dtype)), 2, -1)
      x = self.norm(x)
      # Optional debug dump (DT_DUMP_BLOCKS) of the final modulation inputs.
      if os.environ.get("DT_DUMP_BLOCKS"):
        jax.debug.callback(functools.partial(_blk_save, "ada_temb"), temb)
        jax.debug.callback(functools.partial(_blk_save, "ada_normed"), x)
        jax.debug.callback(functools.partial(_blk_save, "ada_scale"), scale)
        jax.debug.callback(functools.partial(_blk_save, "ada_shift"), shift)
        jax.debug.callback(functools.partial(_blk_save, "ada_tsidx"),
                           timestep_indices.astype(jnp.float32))
      return (x * (1.0 + jnp.take(scale, timestep_indices, axis=0))
              + jnp.take(shift, timestep_indices, axis=0))

  def rope_angles(position_ids, rope_freq_dim=16, rope_theta=10000.0):
    """`(seq, 3)` coordinates to `cos`/`sin` of `(seq, 6 * rope_freq_dim)`.

    One `inv_freq` serves all three axes; the three blocks are
    concatenated and then concatenated with themselves, so `rotate_half`
    turns `2 * 3 * rope_freq_dim` = 96 of the 128 head channels.
    """
    inv = 1.0 / (rope_theta ** (
        jnp.arange(0, 2 * rope_freq_dim, 2, dtype=jnp.float32)
        / (2 * rope_freq_dim)))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv.reshape(1, 1, -1)
    freqs = jnp.concatenate(
        [freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)

  def timestep_embedding(timestep, channels=256):
    """`Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`.

    Cosines first, then sines -- the flip is not cosmetic, it decides
    which half of the embedding the MLP's first column sees.
    """
    half = channels // 2
    exponent = -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32)
    emb = jnp.exp(exponent / half)
    emb = timestep.astype(jnp.float32)[:, None] * emb[None, :]
    return jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)

  class TimestepEmbedder(nnx.Module):
    """`TimestepEmbedding`: linear, SiLU, linear. float32 in the checkpoint."""

    def __init__(self, in_channels, hidden, out_dim, *, dtype=jnp.float32,
                 rngs=None):
      self.linear_1 = Linear(in_channels, hidden, bias=True, dtype=dtype,
                             rngs=rngs, precision=jax.lax.Precision.HIGHEST)
      self.linear_2 = Linear(hidden, out_dim, bias=True, dtype=dtype,
                             rngs=rngs, precision=jax.lax.Precision.HIGHEST)

    def __call__(self, x):
      return self.linear_2(jax.nn.silu(self.linear_1(x)))

  class Transformer(nnx.Module):
    """The whole stack over one packed sequence.

    The checkpoint is mixed precision and it matters: `proj_in`,
    `audio_proj_in`, `time_embedder`, `proj_out` and `audio_proj_out`
    are float32 while `context_embedder` and the fifty blocks are
    bfloat16. The text stream's dtype is what the packed buffer takes.
    """

    def __init__(self, *, num_attention_heads=56, attention_head_dim=128,
                 hidden_size=5376, num_layers=50, num_refiner_layers=2,
                 ffn_dim=14336, in_channels=24, audio_in_channels=32,
                 patch_size=(1, 2, 2), text_dim=5120, freq_dim=256,
                 time_embed_hidden_dim=5376, time_embed_dim=2688,
                 rope_freq_dim=16, rope_theta=10000.0, norm_eps=1e-5,
                 qk_norm_eps=1e-5, final_norm_eps=1e-5, num_pdd_heads=0,
                 dtype=jnp.bfloat16, rngs=None):
      patch = patch_size[0] * patch_size[1] * patch_size[2]
      video_patch_dim = in_channels * patch
      self.freq_dim = freq_dim
      self.rope_freq_dim = rope_freq_dim
      self.rope_theta = rope_theta
      self.proj_in = Linear(video_patch_dim, hidden_size, bias=True,
                            dtype=jnp.float32, rngs=rngs)
      self.audio_proj_in = (
          Linear(audio_in_channels, hidden_size, bias=True,
                 dtype=jnp.float32, rngs=rngs)
          if audio_in_channels else None
      )
      self.context_embedder = Linear(text_dim, hidden_size, bias=True,
                                     dtype=dtype, rngs=rngs)
      self.time_embedder = TimestepEmbedder(freq_dim, time_embed_hidden_dim,
                                            time_embed_dim, rngs=rngs)
      self.token_refiner = TokenRefiner(
          hidden_size, num_attention_heads, attention_head_dim, ffn_dim,
          num_refiner_layers, norm_eps, qk_norm_eps, final_norm_eps,
          dtype=dtype, rngs=rngs)
      self.transformer_blocks = nnx.List([
          Block(hidden_size, num_attention_heads, attention_head_dim,
                ffn_dim, time_embed_dim, norm_eps, qk_norm_eps,
                dtype=dtype, rngs=rngs)
          for _ in range(num_layers)])
      self.norm_out = AdaLNOut(hidden_size, time_embed_dim, final_norm_eps,
                               dtype=dtype, rngs=rngs)
      if num_pdd_heads > 0:
        self.proj_out = None
        self.pdd_proj_out = nnx.List([
            Linear(hidden_size, video_patch_dim, bias=True,
                   dtype=jnp.float32, rngs=rngs)
            for _ in range(num_pdd_heads)
        ])
      else:
        self.proj_out = Linear(hidden_size, video_patch_dim, bias=True,
                               dtype=jnp.float32, rngs=rngs)
        self.pdd_proj_out = None
      self.audio_proj_out = (
          Linear(hidden_size, audio_in_channels, bias=True,
                 dtype=jnp.float32, rngs=rngs)
          if audio_in_channels else None
      )

    def __call__(self, video, audio, text, timestep, timestep_indices,
                 token_tags, position_ids, video_indices, audio_indices,
                 text_indices, sdpa=None, segments=None, seq_len=None,
                 text_len=None, adaln_mods=None, pdd_step=None):
      """`seq_len` / `text_len` (optional, traced scalars): true lengths of
      the packed sequence / text. When given, inputs use bucketed shapes and
      attention masks by the true length (`jax_server`)."""
      seq = position_ids.shape[0]
      # Everything outside attention is row-independent, so the stack may
      # run on a rounded-up sequence and the extra rows are dropped at
      # the end by `video_indices`/`audio_indices`, which never name them.
      padded = seq
      if PAD_SEQUENCE_TO:
        padded = -(-seq // PAD_SEQUENCE_TO) * PAD_SEQUENCE_TO
      rope = rope_angles(position_ids, self.rope_freq_dim, self.rope_theta)
      if padded != seq:
        if segments is not None:
          # The padded rows sit at the end, where the bulk run is, so
          # extending its stop is all the runs need.
          *rest, (start, _, index) = segments
          segments = (*rest, (start, padded, index))
        extra = padded - seq
        # cos 1 / sin 0 is the identity rotation, so the padded rows stay
        # whatever the zero buffer and their modulation make them.
        rope = (jnp.pad(rope[0], ((0, extra), (0, 0)), constant_values=1.0),
                jnp.pad(rope[1], ((0, extra), (0, 0)), constant_values=0.0))
        timestep_indices = jnp.pad(timestep_indices, (0, extra))
        token_tags = jnp.pad(token_tags, (0, extra))

      video_embeds = self.proj_in(video.astype(jnp.float32))
      text_embeds = self.context_embedder(
          text.astype(self.context_embedder.weight[...].dtype))
      text_embeds = self.token_refiner(text_embeds, sdpa, text_len)

      dtype = text_embeds.dtype
      x = jnp.zeros((text_embeds.shape[0], padded, text_embeds.shape[-1]),
                    dtype)
      x = x.at[:, text_indices].set(text_embeds)
      x = x.at[:, video_indices].set(video_embeds.astype(dtype))
      if self.audio_proj_in is not None and audio is not None:
        audio_embeds = self.audio_proj_in(audio.astype(jnp.float32))
        x = x.at[:, audio_indices].set(audio_embeds.astype(dtype))

      temb = self.time_embedder(
          timestep_embedding(timestep, self.freq_dim))
      adaln_indices = timestep_indices * MODALITY_NUM + token_tags

      if os.environ.get("DT_DUMP_BLOCKS"):
        jax.debug.callback(functools.partial(_blk_save, "x0"), x)
        jax.debug.callback(functools.partial(_blk_save, "temb"), temb)
      _barrier = (_tp_size() <= 2
                  or os.environ.get("DT_BLOCK_BARRIER") == "1")
      for _i, block in enumerate(self.transformer_blocks):
        if _barrier and _i > 0:
          x, temb, adaln_indices = jax.lax.optimization_barrier(
              (x, temb, adaln_indices))
        x = block(x, temb, adaln_indices, rope, sdpa, segments,
                  seq if seq_len is None else seq_len,
                  None if adaln_mods is None else adaln_mods[_i])
        if os.environ.get("DT_DUMP_BLOCKS") and _i in (0, 4, 9, 19, 29, 39,
                                                       49):
          jax.debug.callback(functools.partial(_blk_save, f"b{_i:02d}"), x)

      x = self.norm_out(x, temb, timestep_indices).astype(jnp.float32)
      if os.environ.get("DT_DUMP_BLOCKS"):
        jax.debug.callback(functools.partial(_blk_save, "normout"), x)
      if self.pdd_proj_out is not None:
        idx = 0 if pdd_step is None else pdd_step
        w = jnp.stack([h.weight[...] for h in self.pdd_proj_out])[idx]
        b = jnp.stack([h.bias[...] for h in self.pdd_proj_out])[idx]
        vid_out = jnp.einsum("...i,oi->...o", x, w.astype(x.dtype)) + b.astype(x.dtype)
      else:
        vid_out = self.proj_out(x)
      vid_out = jnp.take(vid_out, video_indices, axis=1)
      if self.audio_proj_out is None:
        return vid_out
      return (vid_out, jnp.take(self.audio_proj_out(x), audio_indices, axis=1))

  return Block, Transformer


# --------------------------------------------------------------------------
# numerical equivalence against diffusers
# --------------------------------------------------------------------------


def _torch_block(config, dtype):
  import torch
  from diffusers.models.transformers import transformer_minimax_h3 as h3

  block = h3.MiniMaxH3TransformerBlock(**config)
  return block.to(dtype).eval()


def test_block(verbose=True):
  """One block, random weights, JAX against PyTorch on CPU."""
  import jax.numpy as jnp
  import numpy as np
  import torch
  from flax import nnx

  # Small enough to run on a laptop, same structure as the real thing.
  config = dict(hidden_size=128, num_attention_heads=4, attention_head_dim=32,
                ffn_dim=256, time_embed_dim=64, norm_eps=1e-5,
                qk_norm_eps=1e-5)
  torch.manual_seed(0)
  reference = _torch_block(config, torch.float32)

  Block, _ = _build_modules()
  ours = Block(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))

  # The checkpoint names map one-to-one apart from the feed-forward, which
  # diffusers nests as `ff.net.0.proj` and `ff.net.2`.
  renames = {"ff.net.0.proj": "ff.proj", "ff.net.2": "ff.out",
             "attn.to_out.0": "attn.to_out"}
  state = dict(reference.named_parameters())
  for name, tensor in state.items():
    target = name
    for old, new in renames.items():
      target = target.replace(old, new)
    node = ours
    for part in target.split(".")[:-1]:
      node = getattr(node, part)
    leaf = target.split(".")[-1]
    getattr(node, leaf)[...] = jnp.asarray(
        tensor.detach().float().numpy(), jnp.float32)

  seq, timesteps = 16, 2
  rotary = 2 * (config["attention_head_dim"] // 4)   # an even, smaller dim
  rng = np.random.default_rng(0)
  x = rng.standard_normal((1, seq, config["hidden_size"]), np.float32)
  temb = rng.standard_normal((timesteps, config["time_embed_dim"]), np.float32)
  cos = rng.standard_normal((seq, rotary), np.float32)
  sin = rng.standard_normal((seq, rotary), np.float32)
  tags = rng.integers(0, MODALITY_NUM, seq)
  steps = rng.integers(0, timesteps, seq)
  indices = (steps * MODALITY_NUM + tags).astype(np.int32)

  with torch.no_grad():
    want = reference(
        torch.from_numpy(x), torch.from_numpy(temb),
        torch.from_numpy(indices).long(),
        (torch.from_numpy(cos), torch.from_numpy(sin)),
    ).numpy()
  got = np.asarray(ours(
      jnp.asarray(x), jnp.asarray(temb), jnp.asarray(indices),
      (jnp.asarray(cos), jnp.asarray(sin)),
  ))

  diff = np.abs(want - got)
  scale = np.abs(want).mean()
  ok = diff.max() < 2e-4 * max(scale, 1.0)
  if verbose:
    print(f"  block {str(want.shape):18s} max|diff| {diff.max():.3e}  "
          f"relative {diff.max() / scale:.3e}  "
          + ("MATCH" if ok else "MISMATCH"))
  return ok


def test_transformer(verbose=True):
  """The whole stack at a small config, JAX against diffusers on CPU."""
  import jax.numpy as jnp
  import numpy as np
  import torch
  from flax import nnx
  from diffusers.models.transformers import transformer_minimax_h3 as h3

  config = dict(num_attention_heads=4, attention_head_dim=32, hidden_size=128,
                num_layers=2, num_refiner_layers=1, ffn_dim=256,
                in_channels=8, audio_in_channels=6, patch_size=(1, 2, 2),
                text_dim=64, freq_dim=32, time_embed_hidden_dim=128,
                time_embed_dim=64, rope_freq_dim=4, rope_theta=10000.0,
                norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
  torch.manual_seed(0)
  reference = h3.MiniMaxH3Transformer3DModel(**config).to(torch.float32).eval()

  _, Transformer = _build_modules()
  ours = Transformer(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))

  renames = (("ff.net.0.proj", "ff.proj"), ("ff.net.2", "ff.out"),
             ("attn.to_out.0", "attn.to_out"))
  for name, tensor in reference.named_parameters():
    target = name
    for old, new in renames:
      target = target.replace(old, new)
    node = ours
    for part in target.split(".")[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    getattr(node, target.split(".")[-1])[...] = jnp.asarray(
        tensor.detach().float().numpy(), jnp.float32)

  n_video, n_audio, n_text, n_steps = 12, 5, 4, 2
  seq = n_video + n_audio + n_text
  patch = config["in_channels"] * 4
  rng = np.random.default_rng(0)
  video = rng.standard_normal((1, n_video, patch), np.float32)
  audio = rng.standard_normal((1, n_audio, config["audio_in_channels"]),
                              np.float32)
  text = rng.standard_normal((1, n_text, config["text_dim"]), np.float32)
  timestep = rng.random(n_steps).astype(np.float32)
  order = rng.permutation(seq)
  text_i = np.sort(order[:n_text]).astype(np.int32)
  video_i = np.sort(order[n_text:n_text + n_video]).astype(np.int32)
  audio_i = np.sort(order[n_text + n_video:]).astype(np.int32)
  tags = np.zeros(seq, np.int32)
  tags[text_i], tags[audio_i] = 1, 2
  steps = rng.integers(0, n_steps, seq).astype(np.int32)
  pos = rng.integers(0, 7, (seq, 3)).astype(np.int32)

  t = torch.from_numpy
  with torch.no_grad():
    want_v, want_a = reference(
        t(video), t(audio), t(text), t(timestep), t(steps).long(),
        t(tags).long(), t(pos).long(), t(video_i).long(), t(audio_i).long(),
        t(text_i).long(), return_dict=False)
  got_v, got_a = ours(
      jnp.asarray(video), jnp.asarray(audio), jnp.asarray(text),
      jnp.asarray(timestep), jnp.asarray(steps), jnp.asarray(tags),
      jnp.asarray(pos), jnp.asarray(video_i), jnp.asarray(audio_i),
      jnp.asarray(text_i))

  ok = True
  for label, want, got in (("video", want_v.numpy(), np.asarray(got_v)),
                           ("audio", want_a.numpy(), np.asarray(got_a))):
    diff = np.abs(want - got)
    scale = np.abs(want).mean()
    here = diff.max() < 2e-4 * max(scale, 1.0)
    ok &= here
    if verbose:
      print(f"  {label:<6}{str(want.shape):18s} max|diff| {diff.max():.3e}  "
            f"relative {diff.max() / scale:.3e}  "
            + ("MATCH" if here else "MISMATCH"))
  return ok


# --------------------------------------------------------------------------
# the checkpoint
# --------------------------------------------------------------------------

# Where our names differ from the checkpoint's. Only the feed-forward and
# the attention output do: diffusers nests them in `net`/`to_out` lists.
_RENAMES = (
    ("ff.net.0.proj", "ff.proj"),
    ("ff.net.2", "ff.out"),
    ("attn.to_out.0", "attn.to_out"),
)


def find_snapshot(repo="MiniMaxAI/MiniMax-H3"):
  """The local snapshot directory of `repo` (see `diffusers_tpu.hub.local_snapshot`)."""
  from ...hub import local_snapshot

  return local_snapshot(repo)


def build_sharded(mesh, dtype=None, config=None):
  """A model whose parameters are only ever allocated sharded.

  Constructing it the plain way allocates every `jnp.zeros` on device 0
  first -- 62 GB of it at the shipped config -- and the chip is out of
  memory before a single weight is read. `nnx.eval_shape` builds the
  tree abstractly, and each parameter is then materialised straight
  into its shard.
  """
  import jax
  import jax.numpy as jnp
  from flax import nnx

  _, Transformer = _build_modules()
  dtype = dtype or jnp.bfloat16
  config = config or CONFIG_FULL
  model = nnx.eval_shape(
      lambda: Transformer(**config, dtype=dtype, rngs=nnx.Rngs(0)))
  state = nnx.state(model, nnx.Param)
  for path, leaf in nnx.to_flat_state(state):
    name = ".".join(str(part) for part in path).removesuffix(".value")
    sharding = jax.sharding.NamedSharding(mesh, _spec_for(name))
    leaf.value = jax.device_put(
        jnp.zeros(leaf.value.shape, leaf.value.dtype), sharding)
  nnx.update(model, state)
  return model


def load_lora(adapter_dir):
  """A PEFT LoRA adapter (e.g. Pika-H3-Flex) as `{checkpoint key: (A, B, scale)}`.

  `adapter_dir` holds `adapter.safetensors` and a `manifest.json` with
  `rank` and `alpha`. Keys are the checkpoint's own (`...to_q.weight`),
  so the merge happens before `_RENAMES`, on the tensor as read.
  """
  import json
  import os

  from safetensors.torch import load_file

  with open(os.path.join(adapter_dir, "manifest.json")) as f:
    manifest = json.load(f)
  scale = float(manifest["alpha"]) / float(manifest["rank"])
  state = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
  out = {}
  for key, a in state.items():
    if ".lora_A." not in key:
      continue
    base = key.split(".lora_A.")[0]
    b = state[key.replace(".lora_A.", ".lora_B.")]
    out[base + ".weight"] = (a, b, scale)
  return out, manifest


def _merge_lora(key, t, lora):
  """`W + scale * B @ A` in float32, back in `W`'s dtype.

  The reference keeps the adapter unmerged and in bfloat16
  (`apply_lora(master_dtype=bfloat16)`); merging in float32 and rounding
  once is the closer of the two to the trained function.
  """
  import torch

  if not lora or key not in lora:
    return t
  a, b, scale = lora[key]
  with torch.no_grad():
    return (t.float() + scale * (b.float() @ a.float())).to(t.dtype)


_INT8_TARGETS = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                 "attn.to_out.0.weight", "ff.net.0.proj.weight",
                 "ff.net.2.weight")


def _quantize_int8(t, rows=1024):
  """Per-output-channel symmetric int8 of an `(out, in)` weight, by slabs."""
  import torch

  q = torch.empty(t.shape, dtype=torch.int8)
  scale = torch.empty((t.shape[0],), dtype=torch.float32)
  for r in range(0, t.shape[0], rows):
    w = t[r:r + rows].float()
    sc = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q[r:r + rows] = torch.round(w / sc[:, None]).clamp(-127, 127).to(torch.int8)
    scale[r:r + rows] = sc
  return q, scale


INT8_ACT = os.environ.get("DT_INT8_ACT", "0") == "1"
FF_CHUNK = int(os.environ.get("DT_FF_CHUNK", "0"))
QKV_CHUNK = int(os.environ.get("DT_QKV_CHUNK", "0"))

# Write-through weight cache: set CACHE_WRITE["dir"] before a load and the
# loaders save each host array as it is made, before it goes to the chips.
# Saving the loaded module afterwards instead copied every
# pinned array back into host memory and kept the copy: +1 GB per GB saved,
# which killed a 47 GB v5e host at 17 GB written.
CACHE_WRITE = {}


def cache_put(name, arr):
  """Save one host array under the module's parameter path (thread-safe)."""
  import os

  import numpy as np

  d = CACHE_WRITE.get("dir")
  if not d:
    return
  os.makedirs(d, exist_ok=True)
  arr = np.asarray(arr)
  dt = str(arr.dtype)
  np.save(os.path.join(d, name + ".npy"),
          arr.view(np.uint16) if dt == "bfloat16" else arr)
  CACHE_WRITE.setdefault("dtypes", {})[name] = dt


def cache_finish(extra=None):
  """Write the index for everything `cache_put` saved; turn writing off."""
  import json
  import os

  d = CACHE_WRITE.pop("dir", None)
  dtypes = CACHE_WRITE.pop("dtypes", {})
  if not d:
    return
  for key, arr in (extra or {}).items():
    CACHE_WRITE["dir"] = d
    cache_put("_extra." + key, arr)
    CACHE_WRITE.pop("dir")
    dtypes["_extra." + key] = CACHE_WRITE.pop("dtypes")["_extra." + key]
  with open(os.path.join(d, "index.json"), "w") as f:
    json.dump({"params": sorted(k for k in dtypes if not k.startswith("_extra.")),
               "extra": sorted(extra or {}), "dtypes": dtypes}, f)


# `load_transformer(adaln_temb=)` leaves its `(steps, blocks, levels, out)`
# modulation here rather than on the nnx module, which holds jax arrays only.
ADALN_MODS = {}


def load_param_cache(model, cache_dir, mesh, spec_fn, memory_kind=None):
  """Fill an abstract module from a weight-cache directory; returns extras.

  Params the module was not built with (int8 `weight_scale`) are created.
  Shapes that do not match the module (placeholders) are replicated.
  """
  import json
  import os

  import jax
  import ml_dtypes
  import numpy as np
  from flax import nnx
  from jax.sharding import NamedSharding, PartitionSpec as P

  with open(os.path.join(cache_dir, "index.json")) as f:
    index = json.load(f)

  def get(fname):
    # One large sequential read per file, not a memory map: on a disk that
    # is limited in operations rather than bytes (Colab's loop device, read-
    # ahead fixed at 128 KB) page-faulting a 20 GB DiT cache in ran at
    # ~20 MB/s. Same bytes either way.
    with open(os.path.join(cache_dir, fname + ".npy"), "rb", buffering=0) as f:
      version = np.lib.format.read_magic(f)
      read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                     else np.lib.format.read_array_header_2_0)
      shape, fortran, dtype = read_header(f)
      arr = np.empty(shape, dtype, order="F" if fortran else "C")
      view, got = memoryview(arr.reshape(-1).view(np.uint8)), 0
      while got < view.nbytes:
        k = f.readinto(view[got:])
        if not k:
          raise IOError(f"short read in {fname}.npy")
        got += k
    if index["dtypes"][fname] == "bfloat16":
      arr = arr.view(ml_dtypes.bfloat16)
    return arr

  # A few files read ahead on threads: each read stays large and
  # sequential, and a fast disk still gets parallel requests (one thread
  # alone loaded the v5e DiT cache in 165 s against 117 s memory-mapped).
  import concurrent.futures as cf
  names = list(index["params"])
  pool = cf.ThreadPoolExecutor(int(os.environ.get("DT_CACHE_READ_THREADS", "4")))
  pending = {n: pool.submit(get, n) for n in names[:8]}
  for i, name in enumerate(names):
    if i + 8 < len(names):
      pending[names[i + 8]] = pool.submit(get, names[i + 8])
    arr = pending.pop(name).result()
    parts = name.split(".")
    node = model
    for part in parts[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    spec = spec_fn(name, arr.ndim) if spec_fn else P()
    if mesh is not None and any(
        ax is not None and d % mesh.devices.size for d, ax in zip(arr.shape, spec)):
      spec = P()
    kind = memory_kind(name) if callable(memory_kind) else memory_kind
    value = jax.device_put(np.ascontiguousarray(arr),
                           NamedSharding(mesh, spec, memory_kind=kind))
    del arr
    current = getattr(node, parts[-1], None)
    if isinstance(current, nnx.Param):
      current.value = value
    else:
      setattr(node, parts[-1], nnx.data(nnx.Param(value)))
  pool.shutdown()
  return {k: np.asarray(get("_extra." + k)) for k in index["extra"]}


def host_temb(transformer_dir, levels_per_step, freq_dim=256):
  """`time_embedder(timestep_embedding(levels))` on the host, per step.

  Float64 through the tiny float32 MLP; returns one `(levels, 2688)`
  float32 array per step. This is what lets `load_transformer` turn the
  26 GB of `adaln_proj` weights into a few hundred MB of modulation.
  """
  import json
  import os

  import numpy as np
  from ...safetensors_io import safe_open

  index = json.load(open(os.path.join(
      transformer_dir, "diffusion_pytorch_model.safetensors.index.json")))
  w = {}
  for k in ("linear_1.weight", "linear_1.bias", "linear_2.weight",
            "linear_2.bias"):
    key = "time_embedder." + k
    with safe_open(os.path.join(transformer_dir, index["weight_map"][key]),
                   framework="np") as r:
      w[k] = r.get_tensor(key).astype(np.float64)
  half = freq_dim // 2
  freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
  out = []
  for levels in levels_per_step:
    e = np.asarray(levels, np.float32)[:, None] * freqs[None, :]
    e = np.concatenate([np.cos(e), np.sin(e)], -1).astype(np.float64)
    h = e @ w["linear_1.weight"].T + w["linear_1.bias"]
    h = h / (1 + np.exp(-h))
    out.append((h @ w["linear_2.weight"].T + w["linear_2.bias"]).astype(np.float32))
  return out


def _adaln_mods(t, temb_steps):
  """`adaln_proj.linear(silu(temb))` without the bias, every step: `(S, L, out)`.

  As the module does it: silu in float32, input rounded to the weight's
  bfloat16, product rounded to bfloat16 (the bias is added later, and
  rounded again, in `load_transformer`).
  """
  import ml_dtypes
  import numpy as np
  import torch

  w = t.float()
  res = []
  for temb in temb_steps:
    a = temb / (1 + np.exp(-temb))
    a = torch.from_numpy(a.astype(ml_dtypes.bfloat16).astype(np.float32))
    res.append((a @ w.T).numpy().astype(ml_dtypes.bfloat16).astype(np.float32))
  return np.stack(res)


def load_transformer(model, transformer_dir, mesh=None, verbose=False,
                     place=None, memory_kind=None, round_bf16=False,
                     lora=None, int8=False, adaln_temb=None):
  """Read the checkpoint into the nnx module, sharding as it goes.

  Every tensor must land and every parameter must be filled: a name that
  does not resolve is an error, not a warning, because a silently
  skipped tensor is the failure this is written to catch. The dtype each
  parameter was built with wins, which is what keeps the mixed-precision
  layout (`proj_in` and the output heads in float32, the stack in
  bfloat16) intact.

  `memory_kind="pinned_host"` parks the weights in host memory while
  still splitting them across the mesh. `place=` puts every weight on one
  device, which only suits a single chip (jit rejects single-device
  arrays against a multi-device mesh).
  """
  import glob
  import os

  import jax
  import jax.numpy as jnp
  from flax import nnx
  from ...safetensors_io import safe_open

  shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {transformer_dir}")

  def resolve(name):
    node = model
    for part in name.split(".")[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    return getattr(node, name.split(".")[-1])

  if (os.environ.get("DT_FAST_LOAD", "1") == "1" and mesh is not None
      and place is None and not round_bf16):
    return _load_transformer_parallel(model, shards, mesh, memory_kind,
                                      resolve, verbose, lora=lora, int8=int8,
                                      adaln_temb=adaln_temb)
  if lora or int8 or adaln_temb is not None:
    raise NotImplementedError("lora= needs the parallel loader (DT_FAST_LOAD=1, a mesh)")

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        name = key
        for old, new in _RENAMES:
          name = name.replace(old, new)
        param = resolve(name)
        # `nnx.eval_shape` leaves each `Param` holding a
        # `ShapeDtypeStruct`, which is not subscriptable; `.value`
        # works for both that and a real array.
        want = param.value if hasattr(param, "value") else param[...]
        value = reader.get_tensor(key)
        if round_bf16 and value.dtype == jnp.float32:
          # Match the torchax backend, whose
          # `from_pretrained(torch_dtype=torch.bfloat16)` also rounds the
          # float32 `proj_in` and output heads, so outputs are comparable.
          value = value.astype(jnp.bfloat16)
        value = value.astype(want.dtype)
        value = value.reshape(want.shape)
        if place is not None:
          value = jax.device_put(value, place)
        elif mesh is not None:
          # `memory_kind` may be a callable deciding per tensor, so only as
          # many weights as needed are offloaded to host memory.
          kind = memory_kind(name) if callable(memory_kind) else memory_kind
          value = jax.device_put(
              value, jax.sharding.NamedSharding(mesh, _spec_for(name),
                                                memory_kind=kind))
        if hasattr(param, "value"):
          param.value = value
        else:
          param[...] = value
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  if verbose:
    print(f"loaded {loaded} tensors from {len(shards)} shards")
  return model


def _load_transformer_parallel(model, shards, mesh, memory_kind, resolve,
                               verbose, lora=None, int8=False,
                               adaln_temb=None):
  """Parallel `load_transformer`: one thread per safetensors file.

  Tensors are read as host numpy (bf16 via a zero-copy ml_dtypes view) and
  `device_put` with a NamedSharding, so each chip receives only its shard.
  The serial `framework="flax"` path instead lands every tensor whole on
  chip 0 before splitting it, which is much slower at cold start.
  """
  import concurrent.futures as cf

  import jax
  import ml_dtypes
  import numpy as np
  import torch
  from flax import nnx
  from ...safetensors_io import safe_open

  want_dtypes = {}

  def target(name):
    param = resolve(name)
    want = param.value if hasattr(param, "value") else param[...]
    return param, want

  def one_shard(shard):
    out = []
    with safe_open(shard, framework="pt") as reader:
      for key in reader.keys():
        name = key
        for old, new in _RENAMES:
          name = name.replace(old, new)
        _, want = target(name)
        t = _merge_lora(key, reader.get_tensor(key), lora)
        if key in (lora or {}):
          merged.append(key)
        kind = memory_kind(name) if callable(memory_kind) else memory_kind
        if adaln_temb is not None and ".adaln_proj.linear." in key:
          b = int(key.split(".")[1])
          if key.endswith(".weight"):
            out.append((name, ("adaln_w", b, _adaln_mods(t, adaln_temb))))
          else:
            out.append((name, ("adaln_b", b, t.float().numpy())))
          continue
        if int8 and key.startswith("transformer_blocks.") and key.endswith(
            _INT8_TARGETS):
          q, sc = _quantize_int8(t)
          cache_put(name, q.numpy().reshape(want.shape))
          cache_put(name[:-len("weight")] + "weight_scale", sc.numpy())
          value = jax.device_put(
              q.numpy().reshape(want.shape),
              jax.sharding.NamedSharding(mesh, _spec_for(name), memory_kind=kind))
          s_name = name[:-len("weight")] + "weight_scale"
          scale = jax.device_put(
              sc.numpy(),
              jax.sharding.NamedSharding(mesh, _spec_for(s_name), memory_kind=kind))
          out.append((name, (value, scale)))
          continue
        if t.dtype == torch.bfloat16:
          arr = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        else:
          arr = t.numpy()
        arr = arr.astype(np.dtype(want.dtype), copy=False).reshape(want.shape)
        cache_put(name, arr)
        value = jax.device_put(
            arr, jax.sharding.NamedSharding(mesh, _spec_for(name),
                                            memory_kind=kind))
        out.append((name, value))
    return out

  merged = []
  # A small host (v5e-1: 47 GB) cannot hold fourteen shards' worth of
  # float32 temporaries at once; DT_LOAD_THREADS bounds them.
  workers = int(os.environ.get("DT_LOAD_THREADS", len(shards)))
  with cf.ThreadPoolExecutor(max_workers=workers) as ex:
    results = list(ex.map(one_shard, shards))
  if lora and len(merged) != len(lora):
    raise ValueError(f"LoRA merged into {len(merged)} of {len(lora)} modules")
  if lora:
    print(f"  [lora] merged {len(merged)} modules", flush=True)
  loaded = 0
  n_scales = 0
  adaln_w, adaln_b = {}, {}
  for out in results:
    for name, value in out:
      param, _ = target(name)
      if isinstance(value, tuple) and value[0] in ("adaln_w", "adaln_b"):
        (adaln_w if value[0] == "adaln_w" else adaln_b)[value[1]] = value[2]
        # A placeholder keeps the module whole without 26 GB behind it.
        cache_put(name, np.zeros((1,) * len(param.value.shape),
                                 np.dtype(param.value.dtype)))
        ph = jax.device_put(np.zeros((1,) * len(param.value.shape),
                                     np.dtype(param.value.dtype)),
                            jax.sharding.NamedSharding(
                                mesh, jax.sharding.PartitionSpec()))
        param.value = ph
        loaded += 1
        continue
      if isinstance(value, tuple):
        value, scale = value
        node = model
        for part in name.split(".")[:-1]:
          node = node[int(part)] if part.isdigit() else getattr(node, part)
        node.weight = nnx.Param(value)
        node.weight_scale = nnx.data(nnx.Param(scale))
        loaded += 1
        n_scales += 1
        continue
      if hasattr(param, "value"):
        param.value = value
      else:
        param[...] = value
      loaded += 1
  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  expected -= n_scales
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  if adaln_temb is not None:
    import ml_dtypes
    blocks = sorted(adaln_w)
    # (S, blocks, L, out): bias added and rounded the way `y + bias` is.
    mods = np.stack([(adaln_w[b] + adaln_b[b].astype(ml_dtypes.bfloat16)
                      .astype(np.float32)[None, None]) for b in blocks], 1)
    ADALN_MODS["last"] = mods.astype(ml_dtypes.bfloat16)
    print(f"  [adaln-host] {len(blocks)} blocks x {mods.shape[0]} steps -> "
          f"{ADALN_MODS['last'].nbytes / 2**20:.0f} MiB of modulation",
          flush=True)
  if verbose:
    print(f"loaded {loaded} tensors from {len(shards)} shards (parallel)"
          + (f", {n_scales} int8" if n_scales else ""))
  return model


def load_transformer_from(model, params, mesh=None, place=None):
  """Fill the nnx module from weights that are already on the chips.

  The traced backend holds the whole transformer as a dict of jax
  arrays keyed by the diffusers module path (the checkpoint names); this
  reuses them instead of reading the checkpoint again. Names go through
  `_RENAMES` as in `load_transformer`, and every parameter must be filled.
  """
  import jax
  from flax import nnx

  def resolve(name):
    node = model
    for part in name.split(".")[:-1]:
      node = node[int(part)] if part.isdigit() else getattr(node, part)
    return getattr(node, name.split(".")[-1])

  filled, skipped = 0, []
  for key, value in params.items():
    name = key
    for old, new in _RENAMES:
      name = name.replace(old, new)
    try:
      param = resolve(name)
    except AttributeError:
      # Non-persistent buffers such as `rope.inv_freq`; this module
      # computes its own.
      skipped.append(key)
      continue
    # `.value` works for both real arrays and the `ShapeDtypeStruct`
    # leaves `nnx.eval_shape` leaves behind.
    want = param.value if hasattr(param, "value") else param[...]
    value = value.astype(want.dtype).reshape(want.shape)
    if place is not None:
      value = jax.device_put(value, place)
    elif mesh is not None:
      value = jax.device_put(
          value, jax.sharding.NamedSharding(mesh, _spec_for(name)))
    if hasattr(param, "value"):
      param.value = value
    else:
      param[...] = value
    filled += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if filled != expected:
    raise ValueError(
        f"filled {filled} of {expected} parameters"
        + (f"; skipped {skipped}" if skipped else ""))
  if skipped:
    print(f"  [native] skipped {len(skipped)} non-persistent buffers: {skipped}",
          flush=True)
  return model


def _spec_for(name):
  """Where each weight is cut over the one `tp` axis.

  The head axis splits `q`, `k`, `v` and the feed-forward's gate; the
  contraction splits `to_out` and the feed-forward's output, so each
  block ends with one all-reduce. Norms and the timestep MLP are small
  and stay whole. Same layout as the torchax path.
  """
  from jax.sharding import PartitionSpec as P

  rows = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
          "ff.proj.weight", "adaln_proj.linear.weight")
  row_scales = ("attn.to_q.weight_scale", "attn.to_k.weight_scale",
                "attn.to_v.weight_scale", "ff.proj.weight_scale",
                "adaln_proj.linear.weight_scale")
  columns = ("attn.to_out.weight", "ff.out.weight")
  if any(name.endswith(suffix) for suffix in rows):
    return P("tp", None)
  if any(name.endswith(suffix) for suffix in row_scales):
    return P("tp")
  if any(name.endswith(suffix) for suffix in columns):
    return P(None, "tp")
  if name.endswith("adaln_proj.linear.bias"):
    return P("tp")
  return P()


def quantize_transformer_fp8(model, mesh=None):
  """Quantize the 6 big Linears per transformer block to float8_e4m3fn."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  f8_max = jnp.float32(448.0)

  @jax.jit
  def _quant_w(w):
    w32 = w.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(w32), axis=1), jnp.float32(1e-12)) / f8_max
    qw = jnp.clip(w32 / scale[:, None], -f8_max, f8_max).astype(jnp.float8_e4m3fn)
    return qw, scale

  targets = (
      ("attn", "to_q"),
      ("attn", "to_k"),
      ("attn", "to_v"),
      ("attn", "to_out"),
      ("ff", "proj"),
      ("ff", "out"),
      ("adaln_proj", "linear"),
  )
  count = 0
  for i, block in enumerate(model.transformer_blocks):
    for parent_attr, leaf_attr in targets:
      lin = getattr(getattr(block, parent_attr), leaf_attr)
      w = lin.weight[...]
      if parent_attr == "adaln_proj" and (
          (mesh is not None and mesh.devices.size > 1) or w.shape[0] == 1):
        # Leave the modulation in bfloat16 when HBM is not the constraint
        # (several chips) or when it is a placeholder (`--adaln-host`): its
        # fp8 error accumulates over 50 blocks (v7x-8 ref2va DINO 0.8057).
        continue
      mem_kind = getattr(w.sharding, "memory_kind", None) if hasattr(w, "sharding") else None
      w_name = f"transformer_blocks.{i}.{parent_attr}.{leaf_attr}.weight"
      s_name = f"transformer_blocks.{i}.{parent_attr}.{leaf_attr}.weight_scale"
      # Compute on device then place back in original memory_kind if offloaded
      if mesh is not None and mem_kind == "pinned_host":
        w_dev = jax.device_put(
            w, jax.sharding.NamedSharding(mesh, _spec_for(w_name), memory_kind="device"))
        qw_dev, scale_dev = _quant_w(w_dev)
        qw = jax.device_put(
            qw_dev, jax.sharding.NamedSharding(mesh, _spec_for(w_name), memory_kind="pinned_host"))
        scale = jax.device_put(
            scale_dev, jax.sharding.NamedSharding(mesh, _spec_for(s_name), memory_kind="pinned_host"))
      else:
        qw, scale = _quant_w(w)
        if mesh is not None:
          qw = jax.device_put(
              qw, jax.sharding.NamedSharding(mesh, _spec_for(w_name), memory_kind=mem_kind))
          scale = jax.device_put(
              scale, jax.sharding.NamedSharding(mesh, _spec_for(s_name), memory_kind=mem_kind))
      lin.weight = nnx.Param(qw)
      lin.weight_scale = nnx.data(nnx.Param(scale))
      count += 1
  print(f"  [fp8] Quantized {count} Linear layers to float8_e4m3fn across "
        f"{len(model.transformer_blocks)} blocks", flush=True)
  return model


def interleave_swiglu_weights(model, mesh=None):
  """Permute `ff.proj` weights from `(2, tp, ffn // tp, in)` to `(tp, 2, ffn // tp, in)`.

  With `P("tp", None)` sharding on `(2 * ffn_dim, hidden)`, the unpermuted layout
  places all `value` rows on the first half of the mesh and all `gate` rows on
  the second half. Interleaving `(tp, 2, ffn_dim // tp, hidden)` gives every chip
  matching `(value, gate)` pairs so `value * silu(gate)` and the sequence-tiled
  Pallas SwiGLU FFN megakernel (`pallas_swiglu_ffn_sharded`) run locally per chip.
  """
  import jax
  import jax.numpy as jnp
  from flax import nnx

  tp = mesh.devices.size if mesh is not None else 1
  if tp <= 1:
    for blk in (*model.token_refiner.refiner_blocks, *model.transformer_blocks):
      blk.ff.swiglu_tp = 1
    return model

  @jax.jit
  def _perm_w(w):
    return jnp.swapaxes(w.reshape(2, tp, -1, w.shape[-1]), 0, 1).reshape(w.shape)

  @jax.jit
  def _perm_s(s):
    return jnp.swapaxes(s.reshape(2, tp, -1), 0, 1).reshape(s.shape)

  for blk in (*model.token_refiner.refiner_blocks, *model.transformer_blocks):
    lin = blk.ff.proj
    w = lin.weight[...]
    if w.shape[0] % (2 * tp) != 0 or getattr(blk.ff, "swiglu_tp", 1) == tp:
      continue
    sh = w.sharding
    mem = getattr(sh, "memory_kind", None)
    if mem == "pinned_host":
      dev_sh = jax.sharding.NamedSharding(mesh, sh.spec, memory_kind="device")
      w_p = jax.device_put(_perm_w(jax.device_put(w, dev_sh)), sh)
    else:
      w_p = jax.device_put(_perm_w(w), sh)
    lin.weight = nnx.Param(w_p)
    if lin.weight_scale is not None and lin.weight_scale[...] is not None:
      s = lin.weight_scale[...]
      s_sh = s.sharding
      if getattr(s_sh, "memory_kind", None) == "pinned_host":
        s_dev = jax.sharding.NamedSharding(mesh, s_sh.spec, memory_kind="device")
        s_p = jax.device_put(_perm_s(jax.device_put(s, s_dev)), s_sh)
      else:
        s_p = jax.device_put(_perm_s(s), s_sh)
      lin.weight_scale = nnx.data(nnx.Param(s_p))
    blk.ff.swiglu_tp = tp
  print(f"  [swiglu] interleaved ff.proj weights across tp={tp}", flush=True)
  return model


def test_checkpoint_names(verbose=True):
  """Every checkpoint key resolves, and every parameter is covered.

  Checked against the index rather than by loading: the real config is
  32 B parameters and the point is the name mapping, not the bytes.
  `nnx.eval_shape` builds the module tree without allocating any of it.
  """
  import glob
  import json
  import os

  import jax
  import jax.numpy as jnp
  from flax import nnx

  try:
    snapshot = find_snapshot()
  except FileNotFoundError as missing:
    if verbose:
      print(f"  checkpoint names: skipped ({missing})")
    return True
  index = glob.glob(os.path.join(snapshot, "transformer", "*.index.json"))
  if not index:
    if verbose:
      print("  checkpoint names: skipped (no index)")
    return True
  keys = sorted(json.load(open(index[0]))["weight_map"])

  _, Transformer = _build_modules()
  abstract = nnx.eval_shape(
      lambda: Transformer(**CONFIG_FULL, dtype=jnp.bfloat16,
                          rngs=nnx.Rngs(0)))
  ours = set()
  for path, _ in nnx.to_flat_state(nnx.state(abstract, nnx.Param)):
    ours.add(".".join(str(part) for part in path).removesuffix(".value"))

  renamed = []
  for key in keys:
    name = key
    for old, new in _RENAMES:
      name = name.replace(old, new)
    renamed.append(name)

  missing = sorted(set(renamed) - ours)
  extra = sorted(ours - set(renamed))
  ok = not missing and not extra
  if verbose:
    print(f"  checkpoint names: {len(keys)} keys, {len(ours)} parameters  "
          + ("MATCH" if ok else "MISMATCH"))
    for name in missing[:4]:
      print(f"    in the checkpoint, not in our model: {name}")
    for name in extra[:4]:
      print(f"    in our model, not in the checkpoint: {name}")
  return ok


def parameter_count():
  """What the shipped config comes to, as a check on the transcription."""
  c = CONFIG
  inner = c["num_attention_heads"] * c["attention_head_dim"]
  per_block = (
      3 * c["hidden_size"] * inner            # q, k, v
      + inner * c["hidden_size"]              # out
      + 2 * c["attention_head_dim"]           # q/k norms
      + 2 * c["hidden_size"]                  # the two norms
      + c["hidden_size"] * 2 * c["ffn_dim"]   # swiglu projection
      + c["ffn_dim"] * c["hidden_size"]       # ff out
      + c["time_embed_dim"] * 6 * c["hidden_size"] * MODALITY_NUM
      + 6 * c["hidden_size"] * MODALITY_NUM   # adaln bias
  )
  return per_block, per_block * c["num_layers"]


def main():
  ok = test_block()
  ok &= test_transformer()
  ok &= test_checkpoint_names()
  per_block, total = parameter_count()
  print(f"  one block {per_block / 1e9:.2f}B, x{CONFIG['num_layers']} = "
        f"{total / 1e9:.1f}B  ({total * 2 / 2**30:.0f} GiB in bfloat16)")
  print("all match" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
