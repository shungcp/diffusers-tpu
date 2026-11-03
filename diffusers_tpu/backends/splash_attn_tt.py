"""Give the torch_tpu path a TPU attention kernel instead of generic SDPA.

On 768p/345 frames (19 steps) generic SDPA attention is ~47% of the denoise
time (133.8 s vs 72 s for torchax with this repo's splash kernel).

torch_tpu ships `splash_sdpa` as a drop-in. Patching diffusers'
`dispatch_attention_fn` reaches every block without touching the model.
"""

from __future__ import annotations


def install(verbose=print) -> bool:
  """Route diffusers' attention dispatch at torch_tpu's splash kernel."""
  import torch
  from diffusers.models import attention_dispatch

  try:
    from torch_tpu._internal.ops.splash_attention import splash_sdpa
  except ImportError as exc:  # noqa: BLE001
    # It is a Pallas kernel, so it needs JAX installed even though the
    # runtime is torch_tpu. Without it we silently fall back to generic
    # attention; this message is the only sign.
    verbose(f"  [splash] splash_sdpa unavailable (requires jax): {exc}", flush=True)
    return False

  original = attention_dispatch.dispatch_attention_fn
  state = {"n": 0, "fallback": 0}

  def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, **kwargs):
    # The kernel has no mask/dropout support; fall back rather than
    # silently computing something else.
    if attn_mask is not None or dropout_p:
      state["fallback"] += 1
      if state["fallback"] == 1:
        verbose(f"  [splash] fallback: attn_mask="
                f"{tuple(attn_mask.shape) if attn_mask is not None else None}"
                f" dropout={dropout_p} q={tuple(query.shape)}", flush=True)
      return original(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    state["n"] += 1
    # Block sizes must be clamped to the sequence and kept a multiple of
    # 128 lanes: the text refiner calls this with only 56 tokens. Use 2048
    # (as torchax does) on long sequences, else the largest 128-multiple.
    q_len, kv_len = query.shape[-2], key.shape[-2]
    fit = lambda n: max(128, min(2048, (n // 128) * 128))
    if q_len < 128 or kv_len < 128:
      state["fallback"] += 1
      return original(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    bq, bkv = fit(q_len), fit(kv_len)
    if state["n"] <= 2:
      verbose(f"  [splash] hit {state['n']}, q={tuple(query.shape)} "
              f"block_q={bq} block_kv={bkv} fallbacks={state['fallback']}",
              flush=True)
    return splash_sdpa(query, key, value, is_causal=is_causal,
                       block_q=bq, block_kv=bkv, block_kv_compute=bkv)

  attention_dispatch.dispatch_attention_fn = dispatch
  # The module that calls it imported the name directly.
  import diffusers.models.transformers.transformer_minimax_h3 as h3
  if hasattr(h3, "dispatch_attention_fn"):
    h3.dispatch_attention_fn = dispatch
  verbose("  [splash] attention routed to torch_tpu splash_sdpa", flush=True)
  return True
