# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side derivation of the fixed-m softmax bound.

Everything here decides WHAT ``m`` the fixed-m splash kernel should be
handed and WHICH heads are allowed to use it: the safety constants that
encode the no-flush proof, the runtime Cauchy-Schwarz reductions, the
sampled max-logit observable used for offline calibration, and the
static operand builder that turns a calibration file into the kernel's
``(2, heads)`` scalar-prefetch array. No Pallas here: ordinary JAX and
NumPy that runs outside the kernel. The kernels import from here, never the
reverse.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import fp8_cast

# log2(e): the kernels exponentiate with exp2, so natural-log logits are
# scaled by it. Shared so the bound is in exactly the kernel's units.
LOG2_E = 1.44269504

# Recenter C in s = exp2(qk - m + C): C=88 maximizes the no-flush window
# (l ~ 2^105, o ~ 2^112 stay in f32). A row flushes (l=0 -> NaN) once
# m - rowmax > C + 126.
FIXED_M_RECENTER = 88.0

# Flat gate: fixed-m only if the head's Cauchy-Schwarz bound <= 213.
# k-smoothing forces rowmax >= 0, so the flush gap ceil(bound) <= 213 <
# C + 126 = 214. Outlier heads exceed it and keep online softmax.
FIXED_M_SAFE_BOUND = 213.0

# Ring gate = half the flat bound. K is un-smoothed here, so only the
# two-sided |qk| <= bound holds and the flush gap is ~2*bound; bound <=
# 106.5 keeps the max-carrying step alive. Safe because the LSE merge
# makes a step-flush graceful (l=0 -> lse=-inf = zero weight), so only
# that step must survive.
FIXED_M_RING_SAFE_BOUND = FIXED_M_SAFE_BOUND / 2.0


def max_row_norm_per_head(x: jax.Array, *, seq_axis: int = 2) -> jax.Array:
  """Max L2 row norm per head over the device-local batch.

  Args:
      x: ``(B, heads, seq, dim)`` (``seq_axis=2``) or ``(B, seq, heads,
          dim)`` (``seq_axis=1``); the d-minor reduce is ~3x faster on
          the seq-major layout.
      seq_axis: which axis is the sequence.

  Returns:
      ``(heads,)`` f32 — max over ``(B, seq)`` of ``|x[..., :]|``.
  """
  # Max over batch too so the prefetch operand stays UNMAPPED under the
  # batch vmap (a mapped operand slows the pallas_call ~1.6x at batch 2).
  return jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1)).max(
      axis=(0, seq_axis)
  )


def fixed_m_query_norms(
    query: jax.Array, scale: float | None = None, *, seq_axis: int = 2
) -> jax.Array:
  """Max SCALED query row norm per head, for the fixed-m bound.

  Args:
      query: un-scaled query; layout per ``max_row_norm_per_head``.
      scale: softmax scale; ``None`` -> ``1/sqrt(dim)``.
      seq_axis: which axis is the sequence.

  Returns:
      ``(heads,)`` f32 max row norm of ``query * scale * log2(e)`` —
      the exp2-softmax scaling the kernels fold into q, folded here as
      a scalar (``|c * q_row| = c * |q_row|`` for the positive scale)
      instead of materializing a scaled copy.
  """
  scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
  return max_row_norm_per_head(query, seq_axis=seq_axis) * (
      scale_factor * LOG2_E
  )


def fixed_m_operands(
    query_scaled: jax.Array, key: jax.Array, *, gate: float, seq_axis: int = 2
) -> jax.Array:
  """Builds the ``(2, heads)`` fixed-m scalar-prefetch operand.

  Args:
      query_scaled: query already scaled by ``scale * log2(e)``.
      key: keys, same layout as ``query_scaled``.
      gate: eligibility bound; heads above it fall back to online
          softmax in-kernel.
      seq_axis: sequence axis, per ``max_row_norm_per_head``.

  Returns:
      Row 0 = integral CS bound (the kernel's m), row 1 = flag.
  """
  qn_max = max_row_norm_per_head(query_scaled, seq_axis=seq_axis)
  mk_h = max_row_norm_per_head(key, seq_axis=seq_axis)
  bound = qn_max * mk_h
  # Row 0 is the integral bound the kernel uses as m directly; integral
  # so exp2(m) is exact.
  return jnp.stack([jnp.ceil(bound), (bound <= gate).astype(jnp.float32)])


def ring_fixed_m_operands(qn_max: jax.Array, mk_h: jax.Array) -> jax.Array:
  """``fixed_m_operands`` for the ring, from norms it already carries.

  Norms are taken as arguments rather than derived from tensors here
  because the K-shard norm ROTATES with its shard across ring steps:
  each shard's norm is reduced once on its home rank and then
  ``ppermute``d alongside K/V, so no rank ever re-reduces a rotated K.

  This is the ring's runtime bound. Prefer the calibrated static
  operand (``static_fixed_m_operands``) when a calibration file exists:
  it arrives as a graph input, whereas this one depends on the call's
  own activations. Which wins over online softmax is shape-dependent.

  Args:
      qn_max: ``(heads,)`` scaled query row norms (stationary local Q).
      mk_h: ``(heads,)`` key row norms of this step's un-smoothed shard.

  Returns:
      ``(2, heads)`` f32 — row 0 = integral ``m``, row 1 = eligibility
      flag against the halved un-smoothed ring gate.
  """
  bound = qn_max * mk_h
  # Row 0 is ``floor(gate)`` for every head, not the head's own
  # ``ceil(bound)``:
  #  * Free: ``m`` cancels in ``o / l``, and an eligible head has
  #    ``bound <= 106.5``, so the worst flush gap is 212.5, inside the
  #    no-flush window of ``FIXED_M_RECENTER + 126 = 214``.
  #  * Faster: when both rows derive from ``bound``, the operand lands on
  #    the critical path of the kernel's scalar prefetch instead of
  #    overlapping the K/V collective-permute (147.4 vs 137.2 ms per fused
  #    call at a 720p/121f ring shard on v7x-8). One constant row is enough.
  m = jnp.full_like(bound, float(math.floor(FIXED_M_RING_SAFE_BOUND)))
  # Row 1 gates on the HALVED un-smoothed ring bound (K is not
  # k-smoothed here).
  return jnp.stack([m, (bound <= FIXED_M_RING_SAFE_BOUND).astype(jnp.float32)])


# Query rows sampled per attention site when MEASURING the true max
# logit (``sampled_max_abs_logit``). The exact quantity maxes over all
# (q row, k row) pairs, i.e. a second full QK^T; sampling the q side
# keeps every k row exact — the direction that matters, since one q row
# already sees the whole key set — and makes the cost linear in the
# sample. At 720p/81f: 4.3 ms at 1024 rows vs 253 ms exact (59x cheaper).
# The sampled bound was 0.91x the exact one at the median and 0.87x at
# worst among heads that matter; the eligibility margin absorbs this.
MAX_LOGIT_SAMPLE_ROWS = 1024


def sampled_max_abs_logit(
    query: jax.Array,
    key: jax.Array,
    scale: float | None = None,
    *,
    seq_axis: int = 2,
    num_samples: int = MAX_LOGIT_SAMPLE_ROWS,
    qk_fp8: bool = False,
    rows_per_chunk: int = 64,
    gather_axes: tuple[str, ...] | None = None,
    reduce_axes: tuple[str, ...] | None = None,
) -> jax.Array:
  """Max ``|q . k|`` per head over sampled q rows and ALL k rows.

  This is the quantity the fixed-m bound actually needs; the
  Cauchy-Schwarz product ``max|q_row| * max|k_row|`` only bounds it, up
  to the q/k cosine, and that slack is what gates heads out.

  Sampled, not exact: the exact max needs a full extra QK^T. Every KEY
  row is still visited, so the result is exact for the head's worst key;
  only the q side is sampled, on a fixed stride (deterministic across
  prompts, so the growth statistics compare like with like, and a slice
  on the sequence axis stays device-local where a gather would move the
  whole query).

  Args:
      query: un-scaled query, ``(B, heads, seq, dim)`` (``seq_axis=2``)
          or ``(B, seq, heads, dim)`` (``seq_axis=1``).
      key: keys, same layout as ``query``.
      scale: softmax scale; ``None`` -> ``1/sqrt(dim)``.
      seq_axis: which axis is the sequence.
      num_samples: q rows to visit; ``<= 0`` or ``>= seq`` visits every
          row, which makes the result exact at full QK^T cost.
      qk_fp8: reproduce the kernel's E4M3 cast of both operands, so the
          measured logits are the ones the served kernel forms.
      rows_per_chunk: q rows per score tile. Caps the transient
          ``(B, heads, rows, seq)`` f32 score block; the total work is
          unaffected.
      gather_axes: when called inside a ``shard_map``, the mesh axes the
          SEQUENCE is split over. The sampled rows are all-gathered
          across them (~20 MB) so every device scores every sampled row
          against its own key shard, which is what makes the result an
          all-pairs max without moving the key. ``num_samples`` is then
          per-shard.
      reduce_axes: when called inside a ``shard_map``, the mesh axes to
          ``pmax`` the per-head result over — the sequence axes plus any
          batch axis. Omitting an axis silently reports one shard's max.

  Returns:
      ``(heads,)`` f32 max ``|q . k * scale * log2(e)|`` — the same
      log2 units as the Cauchy-Schwarz bound it replaces.

  Raises:
      ValueError: If ``seq_axis`` is neither 1 nor 2.
  """
  if seq_axis == 2:
    # To (B, seq, heads, dim): the einsum below wants the head axis
    # free on both operands, and the observer's layout is already this.
    query = jnp.swapaxes(query, 1, 2)
    key = jnp.swapaxes(key, 1, 2)
  elif seq_axis != 1:
    raise ValueError(f"seq_axis must be 1 or 2, got {seq_axis}")
  seq, heads, dim = query.shape[1], query.shape[2], query.shape[3]
  scale_factor = 1.0 / math.sqrt(dim) if scale is None else scale
  wanted = seq if num_samples <= 0 or num_samples >= seq else num_samples
  # ceil so the stride spans the WHOLE sequence: a floor stride would
  # leave the tail rows unvisited at every prompt, a systematic blind
  # spot rather than a sampling error.
  stride = -(-seq // wanted)
  sample = query[:, ::stride]
  if gather_axes:
    # Explicit, not a sharding constraint: constraining the slice to be
    # sequence-replicated propagates backwards and replicates the whole
    # query (~28 GB -> 37 GB, OOM at 720p/81f).
    sample = jax.lax.all_gather(sample, gather_axes, axis=1, tiled=True)
  # Scale AFTER sampling: |c * q| = c * |q| for the positive scale, and
  # scaling the full query would materialize a second copy of it.
  sample = (sample.astype(jnp.float32) * (scale_factor * LOG2_E)).astype(
      jnp.bfloat16
  )
  keys = key.astype(jnp.bfloat16)
  if qk_fp8:
    # E4M3 -> bf16 is value-preserving (3 mantissa bits, in-range
    # exponents), so this reproduces the kernel's rounding exactly while
    # the dot stays on the ordinary bf16 path.
    #
    # The barrier is required: XLA:TPU folds away an adjacent
    # bf16 -> e4m3 -> bf16 pair (with a pad/dynamic_slice in between,
    # the round trip returned x unchanged). Do not remove it.
    sample = jax.lax.optimization_barrier(
        fp8_cast.quantize_saturating(sample)
    ).astype(jnp.bfloat16)
    keys = jax.lax.optimization_barrier(
        fp8_cast.quantize_saturating(keys)
    ).astype(jnp.bfloat16)
  rows = sample.shape[1]
  chunk = min(rows_per_chunk, rows)
  # Zero-pad to a whole number of chunks; a zero q row contributes
  # |0 . k| = 0 and cannot raise the max.
  pad = (-rows) % chunk
  if pad:
    sample = jnp.pad(sample, ((0, 0), (0, pad), (0, 0), (0, 0)))

  def _chunk_max(index: jax.Array, best: jax.Array) -> jax.Array:
    block = jax.lax.dynamic_slice_in_dim(sample, index * chunk, chunk, axis=1)
    # (B, rows, H, D) x (B, seq, H, D) -> (B, H, rows, seq). Under a
    # shard_map ``keys`` is the device's own shard, so this is the local
    # half of the all-pairs max and only the (heads,) result crosses
    # devices (the pmax below).
    scores = jnp.einsum(
        "bqhd,bshd->bhqs", block, keys, preferred_element_type=jnp.float32
    )
    return jnp.maximum(best, jnp.abs(scores).max(axis=(0, 2, 3)))

  best = jax.lax.fori_loop(
      0, (rows + pad) // chunk, _chunk_max, jnp.zeros((heads,), jnp.float32)
  )
  if reduce_axes:
    # Each device saw every sampled row but only its own key shard, so
    # the all-pairs max is the max of the per-device maxima.
    best = jax.lax.pmax(best, reduce_axes)
  return best


# E4M3 round-to-nearest has three mantissa bits, so it can round a
# magnitude UP by up to 2^-4. qk_fp8 casts BOTH q and k, so a bound
# calibrated on the bf16 tensors must be inflated by (1 + 2^-4)^2
# before it bounds the logits the fp8 kernel actually forms.
_QK_FP8_BOUND_INFLATION = (1.0 + 2.0**-4) ** 2


def static_fixed_m_operands(
    qn_max: np.ndarray,
    mk_h: np.ndarray,
    *,
    margin_log2: float,
    qk_fp8: bool = False,
    gate: float = FIXED_M_RING_SAFE_BOUND,
    max_logit: np.ndarray | None = None,
) -> np.ndarray:
  """Builds the fixed-m prefetch operand from CALIBRATED norms.

  The runtime twin (``ring_fixed_m_operands``) computes the flag from the
  call's own activations, which delays the kernel's scalar prefetch and
  loses some overlap with the K/V collective-permute. A graph-input
  operand costs nothing: at a 720p/81f ring shard with qk_fp8, 71.6 ms per
  fused call vs 75.1 for the runtime twin and 74.2 for online softmax.
  The runtime twin remains the fallback when no calibration exists.

  Safety, since a static bound has no runtime check. The two failure
  directions are not symmetric:

  * ``m`` too SMALL is catastrophic. The hot loop forms
    ``exp2(qk - m + 88)``, so a logit more than 16 log2 units above
    ``m`` overflows the accumulator to inf and NaNs the rest of the
    generation (a full row sums ~2^17 terms and the PV matmul adds a
    few more bits: o ~ 2^112 against f32's 2^128).
  * ``m`` too LARGE is graceful. A step underflows to ``l = 0`` once
    ``m - qk > 214``, and the ring's LSE merge maps that to
    ``lse = -inf``, i.e. exactly zero weight for that step rather than
    a NaN.

  So this deliberately does NOT use the calibrated bound as ``m``. It
  uses the LARGEST ``m`` the gate permits — ``floor(gate)`` — for every
  eligible head, which is free (``m`` cancels in ``o / l``, and both
  ``s`` and ``l`` are scale-invariant in f32/bf16) and buys the maximum
  possible tolerance to a prompt the calibration set never saw:

      NaN threshold      = floor(gate) + 16 = 122   (uniform)
      graceful threshold = 214 - floor(gate) = 108  (hit FIRST)

  A head whose true bound drifts past 108 therefore starts dropping
  ring steps before it can reach the NaN at 122. A per-head
  ``m = calibrated + margin`` gets this ordering backwards.

  ``margin_log2`` then controls only ELIGIBILITY, i.e. how much slack
  a head must show at calibration time to be trusted with fixed-m at
  all. Raising it costs throughput (disqualified heads keep online
  softmax) and never costs correctness.

  Two bounds are accepted, both per-head upper bounds on ``|q . k|`` in
  the same log2 units and both spent against the same gate: the measured
  ``max_logit`` (``sampled_max_abs_logit``) when the calibration file
  carries it, else the Cauchy-Schwarz product. The measured one is
  tighter — Cauchy-Schwarz is loose by exactly the q/k cosine — but
  empirical where Cauchy-Schwarz is a proof, so the two are intersected
  (``min``): that can only tighten, and it stops a corrupt measurement
  from widening the gate past what the norms already permit. It is
  empirical twice over (an unseen prompt can exceed it; it samples q
  rows), so ``margin_log2`` must dominate both gaps.

  Args:
      qn_max: ``(heads,)`` calibrated max SCALED query row norm.
      mk_h: ``(heads,)`` calibrated max key row norm.
      margin_log2: eligibility slack in log2 units, on top of the
          calibrated bound. Justify it from the observed bound growth
          across the calibration set.
      qk_fp8: whether the served kernel casts q/k to E4M3, which needs
          the Cauchy-Schwarz bound inflated (see
          ``_QK_FP8_BOUND_INFLATION``). ``max_logit`` is NOT inflated —
          it must already have been measured under the same casting, so
          passing the bf16 measurement with ``qk_fp8=True`` would
          under-bound.
      gate: eligibility ceiling; heads above it keep online softmax.
      max_logit: optional ``(heads,)`` measured max ``|q . k|``.

  Returns:
      ``(2, heads)`` f32 — row 0 = integral ``m``, row 1 = eligibility
      flag. Same layout the kernel's scalar prefetch expects.

  Raises:
      ValueError: If the norm arrays disagree in shape, are not 1-D,
          ``margin_log2`` is negative, or ``max_logit`` is not a finite
          non-negative ``(heads,)`` array.
  """
  qn_max = np.asarray(qn_max, dtype=np.float64)
  mk_h = np.asarray(mk_h, dtype=np.float64)
  if qn_max.ndim != 1 or qn_max.shape != mk_h.shape:
    raise ValueError(
        f"expected matching 1-D (heads,) norms, got {qn_max.shape} and "
        f"{mk_h.shape}"
    )
  if margin_log2 < 0.0:
    raise ValueError(f"margin_log2 must be >= 0, got {margin_log2}")
  bound = qn_max * mk_h
  if qk_fp8:
    bound = bound * _QK_FP8_BOUND_INFLATION
  if max_logit is not None:
    measured = np.asarray(max_logit, dtype=np.float64)
    if measured.shape != bound.shape:
      raise ValueError(f"max_logit must be {bound.shape}, got {measured.shape}")
    if not np.all(np.isfinite(measured)) or np.any(measured < 0.0):
      raise ValueError("max_logit must be finite and non-negative")
    bound = np.minimum(bound, measured)
  # Integral so exp2(m) is exact, as in the runtime path.
  flag = (np.ceil(bound) + margin_log2 <= gate).astype(np.float64)
  # Same m for every eligible head, the largest the gate allows, so the
  # graceful failure comes before the catastrophic one (see docstring).
  m = np.full_like(bound, np.floor(gate))
  return np.stack([m, flag]).astype(np.float32)
