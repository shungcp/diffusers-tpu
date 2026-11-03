#!/usr/bin/env python3
"""A native TPU v7x (Ironwood) FP8 (E4M3) drop-in for the block's six big Linears.

While v6e lacked native FP8 MXU support (making int8 faster there), TPU v7x
provides native hardware `float8_e4m3fn` MXU execution at 2x the throughput of
`bfloat16` with `float32` accumulation (`preferred_element_type=jnp.float32`).

Uses the same structure and `int8_scales.json` activation peak calibration as
`int8_linear.py` (`peak = int8_scale * 127.0`, mapped onto `F8_MAX = 448.0`),
with `ff.net.2` defaulting to per-row dynamic scaling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from torch import nn

F8_MAX = 448.0
INT8_MAX = 127.0


def _quantise_rows_fp8(
    weight: torch.Tensor, rows: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
  """FP8 (float8_e4m3fn) weight and its per-output-channel float32 scale."""
  out_features = weight.shape[0]
  quantised = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
  scale = torch.empty(out_features, dtype=torch.float32, device=weight.device)
  for start in range(0, out_features, rows):
    slab = weight[start : start + rows].to(torch.float32)
    slab_scale = (slab.abs().amax(dim=1) / F8_MAX).clamp_min(1e-12)
    slab_q = (slab / slab_scale.unsqueeze(1)).clamp(-F8_MAX, F8_MAX).to(
        torch.float8_e4m3fn
    )
    quantised[start : start + rows] = slab_q
    scale[start : start + rows] = slab_scale
  return quantised, scale


class Fp8LinearStatic(nn.Module):
  """W8A8 E4M3 FP8 Linear with a fixed activation scale for TPU v7x."""

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

    quantised, scale = _quantise_rows_fp8(original.weight.detach())
    self.weight = nn.Parameter(quantised, requires_grad=False)
    self.weight_scale = nn.Parameter(
        scale.to(torch.float32), requires_grad=False
    )
    # Map calibrated int8 scale (peak / 127) to E4M3 range (peak / 448).
    self._input_scale = float(input_scale) * (INT8_MAX / F8_MAX)
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
        x.astype(jnp.float32) / self._input_scale, -F8_MAX, F8_MAX
    ).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        xq,
        env.t2j_iso(self.weight),
        (((xq.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out = acc * (self._input_scale * env.t2j_iso(self.weight_scale))
    result = env.j2t_iso(out.astype(jnp.bfloat16))
    return result if self.bias is None else result + self.bias


class Fp8LinearDynamic(Fp8LinearStatic):
  """W8A8 E4M3 FP8 Linear with per-row dynamic activation scale (for ff.net.2)."""

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    import torchax

    env = torchax.default_env()
    x = env.t2j_iso(hidden_states).astype(jnp.float32)
    row_scale = jnp.maximum(
        jnp.max(jnp.abs(x), axis=-1, keepdims=True) / F8_MAX, 1e-12
    )
    xq = jnp.clip(x / row_scale, -F8_MAX, F8_MAX).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        xq,
        env.t2j_iso(self.weight),
        (((xq.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out = acc * row_scale * env.t2j_iso(self.weight_scale)
    result = env.j2t_iso(out.astype(jnp.bfloat16))
    return result if self.bias is None else result + self.bias


def convert(
    block: nn.Module,
    scales: dict[str, float],
    dynamic: bool = False,
) -> int:
  """Replaces the block's big Linears with FP8 equivalents."""
  swapped = 0
  for name, scale in scales.items():
    parent = block
    parts = name.split(".")
    for part in parts[:-1]:
      parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    leaf = parts[-1]
    original = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
    if isinstance(original, nn.Linear):
      cls = Fp8LinearDynamic if dynamic else Fp8LinearStatic
      replacement = cls(original, scale)
      if leaf.isdigit():
        parent[int(leaf)] = replacement
      else:
        setattr(parent, leaf, replacement)
      swapped += 1
  return swapped
