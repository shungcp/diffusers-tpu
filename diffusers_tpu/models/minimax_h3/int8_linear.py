#!/usr/bin/env python3
"""An int8 drop-in for the block's four big Linears.

The repository's `Fp8LinearStatic` is the same idea in E4M3, but fp8 has
no native path on v6e and measures slower than bfloat16 there. int8 does
have one, and at MiniMax-H3's shapes it is 1.6-2.2x. This keeps the
structure of the fp8 module -- per-output-channel weight scales, one
calibrated activation scale -- and swaps the numerics.

The attention kernel is left alone: its QK^T contracts over `head_dim`
(128), which is too narrow for int8 to pay, and it measures 0.51x.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from torch import nn

INT8_MAX = 127.0


def _quantise_rows(
    weight: torch.Tensor, rows: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
  """int8 weight and its per-output-channel scale, a slab at a time.

  Done whole, this promotes the weight to float32 and then builds a
  second float32 copy for the rounding. Over the 19.25 G parameters the
  six Linears carry across fifty blocks that is 154 GiB of temporaries,
  and it dominated startup. A slab at a time keeps the peak at the slab.

  Args:
    weight: `(out, in)`, in whatever dtype it was stored.
    rows: Output rows per slab.

  Returns:
    The int8 weight and its `(out,)` float32 scale.
  """
  out_features = weight.shape[0]
  quantised = torch.empty_like(weight, dtype=torch.int8)
  scale = torch.empty(out_features, dtype=torch.float32)
  for start in range(0, out_features, rows):
    slab = weight[start : start + rows].to(torch.float32)
    slab_scale = (slab.abs().amax(dim=1) / INT8_MAX).clamp_min(1e-12)
    slab.div_(slab_scale.unsqueeze(1)).round_().clamp_(-INT8_MAX, INT8_MAX)
    quantised[start : start + rows] = slab.to(torch.int8)
    scale[start : start + rows] = slab_scale
  return quantised, scale


def absmax_scale(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
  """The scale that maps ``x``'s largest magnitude onto int8's range."""
  if dim is None:
    peak = x.abs().max()
  else:
    peak = x.abs().amax(dim=1 - dim if x.dim() == 2 else dim, keepdim=False)
  return (peak / INT8_MAX).clamp_min(torch.finfo(torch.float32).tiny)


class Int8LinearStatic(nn.Module):
  """W8A8 int8 Linear with a fixed activation scale.

  Keeps ``weight`` and ``bias`` as attribute names so sharding rules
  keyed on parameter names still match.
  """

  def __init__(
      self,
      original: nn.Linear,
      input_scale: float,
      compute_dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    if not input_scale > 0 or input_scale == float("inf"):
      raise ValueError(f"input_scale must be positive, got {input_scale}")
    self.in_features = original.in_features
    self.out_features = original.out_features
    self._compute_dtype = compute_dtype

    quantised, scale = _quantise_rows(original.weight.detach())
    # Stored as parameters (not buffers or plain attributes) named `weight`
    # so name-keyed sharding and per-block weight streaming, which walk
    # `named_parameters()`, handle them like the bf16 weight they replace.
    # `requires_grad=False` is what lets a parameter hold int8.
    self.weight = nn.Parameter(quantised, requires_grad=False)
    self.weight_scale = nn.Parameter(
        scale.to(torch.float32), requires_grad=False
    )
    self._input_scale = float(input_scale)
    self.bias = (
        None
        if original.bias is None
        else nn.Parameter(original.bias.detach().to(compute_dtype))
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    import torchax

    env = torchax.default_env()
    x = env.t2j_iso(hidden_states)
    xq = jnp.clip(
        jnp.round(x.astype(jnp.float32) / self._input_scale), -127, 127
    ).astype(jnp.int8)
    # int32 accumulate is the native path; the two scales come back out
    # afterwards, so the product stays exact up to the rounding above.
    acc = jax.lax.dot_general(
        xq,
        env.t2j_iso(self.weight),
        (((xq.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    out = acc.astype(jnp.float32) * (
        self._input_scale * env.t2j_iso(self.weight_scale)
    )
    result = env.j2t_iso(out.astype(jnp.bfloat16))
    return result if self.bias is None else result + self.bias


class Int8LinearDynamic(Int8LinearStatic):
  """Same, but the activation scale is each row's own absmax.

  One scale per tensor is what kept `ff.net.2` in bf16: after the swiglu
  its absmax is 408, while the median row only reaches 17 -- a factor of
  24, so a typical row would land on five of int8's 127 levels. The
  outlier is a *row*, not a channel (measured: row gain 24.2 against
  column gain 9.0), and a row scale factors straight out of the dot
  product because it is constant along the contraction axis. A column
  scale would not; that one has to be folded into the weight.

  The cost is a reduce over the contraction axis and a second pass over
  the activation, which the static version does not pay.
  """

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    import torchax

    env = torchax.default_env()
    x = env.t2j_iso(hidden_states).astype(jnp.float32)
    # Keepdims so the divide broadcasts and the scale stays shaped for
    # the dequantise at the end.
    row_scale = jnp.maximum(
        jnp.max(jnp.abs(x), axis=-1, keepdims=True) / INT8_MAX, 1e-12
    )
    xq = jnp.clip(jnp.round(x / row_scale), -127, 127).astype(jnp.int8)
    acc = jax.lax.dot_general(
        xq,
        env.t2j_iso(self.weight),
        (((xq.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    out = acc.astype(jnp.float32) * row_scale * env.t2j_iso(
        self.weight_scale
    )
    result = env.j2t_iso(out.astype(jnp.bfloat16))
    return result if self.bias is None else result + self.bias


def convert(
    block: nn.Module,
    scales: dict[str, float],
    dynamic: bool = False,
) -> int:
  """Replaces the block's big Linears, returns how many were swapped.

  `dynamic` picks the per-row activation scale; the calibrated value in
  `scales` is then only used to decide the layer is a candidate.
  """
  swapped = 0
  for name, scale in scales.items():
    parent = block
    parts = name.split(".")
    for part in parts[:-1]:
      parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    leaf = parts[-1]
    original = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
    if isinstance(original, nn.Linear):
      cls = Int8LinearDynamic if dynamic else Int8LinearStatic
      replacement = cls(original, scale)
      if leaf.isdigit():
        parent[int(leaf)] = replacement
      else:
        setattr(parent, leaf, replacement)
      swapped += 1
  return swapped
