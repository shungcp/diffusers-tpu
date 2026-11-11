"""Use this repo's Pallas splash attention kernel on torch_tpu.

Why not torch_tpu's own `splash_sdpa`: it requires the block size to divide
the sequence length, and H3's packed sequences (14,856 at 480p, 103,976 at
768p) are not multiples of 128, so no valid block exists
(`scripts/probes_aot/splash_shapes.py` reproduces this). This repo's kernel
accepts any block size by masking edge blocks in Mosaic.

On 768p/345 frames (19 steps) generic SDPA attention takes 133.8 s (~47% of
denoise) vs 72 s with this kernel under torchax.
"""

from __future__ import annotations

import functools
import os

import jax


_REGISTERED: dict[tuple, object] = {}

# Block sizes are constants, not derived from shapes: registering a new op
# inside dispatch would happen under Dynamo tracing, where `jax_op`'s
# `inspect.signature(..., eval_str=True)` fails on CPython 3.14. Ops must be
# registered before compiling; dispatch only looks them up.
#
# Block sweep (`scripts/probes_aot/ours_block_sweep.py`, one chip, 768p
# seq=103976): q=2048 kv=4096 is 73.2 ms (0.52x generic SDPA); 2048/2048
# (the torchax setting) is 76.8 ms.
BQ = int(os.environ.get("DT_OURS_BQ", "2048"))
BKV = int(os.environ.get("DT_OURS_BKV", "2048"))
# Shorter sequences (e.g. the 56-token text refiner) use generic SDPA; they
# are negligible in the profile and would otherwise need their own op.
MIN_SEQ = int(os.environ.get("DT_OURS_MIN_SEQ", "2048"))
# `DT_OURS_TRIPWIRE=1`: raise as soon as the main sequence reaches dispatch,
# reporting shapes and tracing state. Debug aid to confirm the op is used.
TRIPWIRE = os.environ.get("DT_OURS_TRIPWIRE") == "1"
QK_FP8 = os.environ.get("DT_OURS_QK_FP8") == "1"
# Sparse op to use, or None for dense. Set by sparse_attn_tt before each DiT call.
SPARSE_OP = None
SPARSE_SEQ: int = 0


def _op_for(bq: int, bkv: int, fixed_m: bool):
  """Register (and cache) one torch op per block-size configuration.

  `jax_op` infers the schema from `jax.Array` annotations, so the kernel is
  wrapped in a function with exactly (query, key, value); a functools.partial
  would leave csa.attention's kwarg-only `kv_len` and raise NotImplementedError.
  """
  key = (bq, bkv, fixed_m, QK_FP8)
  if key in _REGISTERED:
    return _REGISTERED[key]
  import torch
  from diffusers_tpu.kernels import splash as csa

  def fn(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
    return csa.attention(
        query, key, value, scale=None, block_q=bq,
        block_kv=bkv, block_kv_compute=bkv,
        use_fixed_m=fixed_m, qk_fp8=QK_FP8)

  name = (f"pallas::h3_splash_q{bq}_kv{bkv}"
          + ("_fm" if fixed_m else "")
          + ("_fp8" if QK_FP8 else ""))
  op = torch.tpu.pallas.jax_op(name, fn)
  _REGISTERED[key] = op
  return op


def install(verbose=print, fixed_m: bool = True) -> bool:
  """Route diffusers' attention dispatch to this kernel."""
  import torch
  from diffusers.models import attention_dispatch

  try:
    op = _op_for(BQ, BKV, fixed_m)   # the only config used; register before compiling
  except Exception as exc:  # noqa: BLE001
    verbose(f"  [ours] registration failed: {type(exc).__name__}: "
            f"{' '.join(str(exc).split())[:160]}", flush=True)
    return False

  original = attention_dispatch.dispatch_attention_fn
  state = {"n": 0, "fallback": 0}

  def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, **kwargs):
    # The kernel supports no mask, causal or dropout; fall back rather than
    # silently change semantics. Short sequences also fall back (MIN_SEQ).
    # No bookkeeping while tracing: Dynamo cannot trace `print` under
    # fullgraph and dict mutation becomes a side-effect replay.
    # `is_compiling()` constant-folds to True so these branches vanish.
    already_bhsd = kwargs.pop("_already_bhsd", False)
    seq_ax = -2 if already_bhsd else -3
    tracing = torch.compiler.is_compiling()
    if TRIPWIRE and max(query.shape) > 1000:   # skip the 56-token text refiner
      m = None if attn_mask is None else tuple(attn_mask.shape)
      raise RuntimeError(
          f"[ours tripwire] dispatch reached. q={tuple(query.shape)} "
          f"k={tuple(key.shape)} tracing={tracing} mask={m} "
          f"causal={is_causal} dropout={dropout_p}")
    if (attn_mask is not None or dropout_p or is_causal
        or query.shape[seq_ax] < MIN_SEQ or key.shape[seq_ax] < MIN_SEQ):
      if not tracing:
        state["fallback"] += 1
        if state["fallback"] == 1:
          verbose(f"  [ours] fallback: q={tuple(query.shape)} mask="
                  f"{None if attn_mask is None else tuple(attn_mask.shape)}"
                  f" causal={is_causal} dropout={dropout_p}", flush=True)
      if already_bhsd:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
      return original(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    # Match the torchax call exactly (`torchax_pipeline.py`): block 2048/2048,
    # use_fixed_m=True, plus `k = k - mean(k, axis=-2)` beforehand. The two go
    # together: fixed-m's no-flush guarantee presumes a k-smoothed key.
    #
    # Layout: diffusers passes `(B, seq, heads, dim)` but `kernels/splash.py`
    # expects `(B, heads, seq, dim)`, so transpose. Key centering must then
    # run along `dim=-2` (the sequence axis after the transpose).
    if not already_bhsd:
      query = query.transpose(1, 2)
      key = key.transpose(1, 2)
      value = value.transpose(1, 2)
    if fixed_m:
      key = key - key.mean(dim=-2, keepdim=True)
    if not tracing:
      state["n"] += 1
      if state["n"] <= 2:
        verbose(f"  [ours] hit {state['n']}, q={tuple(query.shape)} "
                f"block=({BQ},{BKV}) fallbacks={state['fallback']}",
                flush=True)
    # Sparse attention (`--sparse-window`, see sparse_attn_tt). Read this
    # module's global SPARSE_OP rather than another module's dict: Dynamo
    # only guards on globals the function reads directly, otherwise it would
    # reuse the dense program. ref2va's token_refiner (7,183 tokens) also
    # reaches here but its length differs from SPARSE_SEQ, so it stays dense.
    if SPARSE_OP is not None and query.shape[-2] == SPARSE_SEQ:
      return SPARSE_OP(query, key, value).transpose(1, 2)
    # Back to `(B, seq, heads, dim)` for the caller.
    return op(query, key, value).transpose(1, 2)

  attention_dispatch.dispatch_attention_fn = dispatch
  import diffusers.models.transformers.transformer_minimax_h3 as h3
  if hasattr(h3, "dispatch_attention_fn"):
    h3.dispatch_attention_fn = dispatch

  bhsd_mode = os.environ.get("DT_BHSD_ROPE", "1")
  if bhsd_mode in ("1", "full") and hasattr(h3, "MiniMaxH3AttnProcessor"):
    orig_call = h3.MiniMaxH3AttnProcessor.__call__
    half = 48
    _rot_cpu = torch.zeros(128, 128, dtype=torch.bfloat16)
    _idx = torch.arange(half)
    _rot_cpu[_idx + half, _idx] = -1
    _rot_cpu[_idx, _idx + half] = 1
    _rot_tpu = _rot_cpu.to("tpu") if bhsd_mode == "full" else None

    def _bhsd_call(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
      if attention_mask is not None or hidden_states.shape[-2] < MIN_SEQ:
        return orig_call(self, attn, hidden_states, rotary_emb, attention_mask)
      if attn.fused_projections:
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
      else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
      # Transpose to (B, H, S, D) immediately so norm_q, norm_k and RoPE
      # operate on (S, D) 2D tiles matching TPU's native (8, 128) tile layout.
      query = query.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
      key = key.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
      value = value.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
      query = attn.norm_q(query)
      key = attn.norm_k(key)
      if rotary_emb is not None:
        cos, sin = rotary_emb
        rotary_dim = cos.shape[-1]
        if bhsd_mode == "full" and _rot_tpu is not None and rotary_dim == 96 and query.shape[-1] == 128:
          pad = 32
          cos128 = torch.nn.functional.pad(cos.to(query.dtype), (0, pad), value=1.0)[None, None, :, :]
          sin128 = torch.nn.functional.pad(sin.to(query.dtype), (0, pad), value=0.0)[None, None, :, :]
          query = query * cos128 + (query @ _rot_tpu) * sin128
          key = key * cos128 + (key @ _rot_tpu) * sin128
        else:
          cos_b = cos.to(query.dtype)[None, None, :, :]
          sin_b = sin.to(query.dtype)[None, None, :, :]
          qr, qp = query[..., :rotary_dim], query[..., rotary_dim:]
          q1, q2 = qr.chunk(2, dim=-1)
          query = torch.cat((qr * cos_b + torch.cat((-q2, q1), dim=-1) * sin_b, qp), dim=-1)
          kr, kp = key[..., :rotary_dim], key[..., rotary_dim:]
          k1, k2 = kr.chunk(2, dim=-1)
          key = torch.cat((kr * cos_b + torch.cat((-k2, k1), dim=-1) * sin_b, kp), dim=-1)
      out = dispatch(
          query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          _already_bhsd=True,
      )
      out = out.flatten(2, 3).type_as(query)
      if os.environ.get("DT_POST_ATTN_PIPE") == "1":
        return out
      out = attn.to_out[0](out)
      out = attn.to_out[1](out)
      return out

    h3.MiniMaxH3AttnProcessor.__call__ = _bhsd_call
    verbose(f"  [ours] BHSD-first norm+RoPE enabled (mode={bhsd_mode})", flush=True)

  verbose("  [ours] attention routed to this repo's Pallas kernel", flush=True)
  return True
