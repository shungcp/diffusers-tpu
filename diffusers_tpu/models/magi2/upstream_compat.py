"""Shared TPU compatibility layer for running upstream Sand.ai ``MAGI-2-preview``
PyTorch DiTs under ``torchax`` (Priority 2) and ``torch_tpu`` (Priority 3).

Provides:
  1. Import-time ``sys.modules`` shims for CUDA/Triton-only packages
     (``triton``, ``flash_attn_interface``, ``magi_attn_extensions``,
     ``magi_compiler``, ``flash_attn.layers.rotary``) so upstream
     ``inference/model/magi2_preview.py`` and ``magi2_refiner.py`` import
     cleanly on TPU hosts without GPU drivers.
  2. Pure-PyTorch ``apply_rotary_emb`` (exact partial/interleaved match to
     ``flash_attn.layers.rotary.apply_rotary_emb``).
  3. Pluggable backend hooks for Attention+Sink, MH-MoE, mHC, Ulysses A2A,
     and Refiner 3D sparse attention.
  4. Fast 8-thread parallel safetensors streaming helpers with exact
     ``expert_bias_ema -> expert_bias`` swap and expert-axis ``EP=8`` slicing
     (``E=256 -> 32 experts/head/chip`` across all 12 heads).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Backend Kernel Hook Registry
# ---------------------------------------------------------------------------


@dataclass
class Magi2TpuHooks:
  """Active TPU kernel callbacks installed by ``torchax`` or ``torch_tpu``."""

  attn_with_sink: Callable[..., Any] | None = None
  grouped_linear: Callable[..., Any] | None = None
  mh_moe_routed: Callable[..., Any] | None = None
  moe_local_head: Callable[..., Any] | None = None
  route_mh_moe: Callable[..., Any] | None = None
  fp32_matmul_highest: Callable[..., Any] | None = None
  mhc_pre: Callable[..., Any] | None = None
  mhc_post: Callable[..., Any] | None = None
  mhc_fused_post_pre: Callable[..., Any] | None = None
  ulysses_scatter_seq_gather_head: Callable[..., Any] | None = None
  ulysses_scatter_head_gather_seq: Callable[..., Any] | None = None
  ep_all_gather: Callable[..., Any] | None = None
  ep_reduce_scatter: Callable[..., Any] | None = None
  refiner_sparse_attn: Callable[..., Any] | None = None
  ep_rank: int = 0
  ep_size: int = 1


_ACTIVE_HOOKS = Magi2TpuHooks()


def get_tpu_hooks() -> Magi2TpuHooks:
  return _ACTIVE_HOOKS


def set_tpu_hooks(hooks: Magi2TpuHooks) -> None:
  global _ACTIVE_HOOKS
  _ACTIVE_HOOKS = hooks


# ---------------------------------------------------------------------------
# Pure-PyTorch RoPE matching flash_attn.layers.rotary.apply_rotary_emb
# ---------------------------------------------------------------------------


def apply_rotary_emb_torch(
    x: Any,
    cos: Any,
    sin: Any,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Any = 0,
    cu_seqlens: Any = None,
    max_seqlen: int | None = None,
) -> Any:
  """Pure-PyTorch partial RoPE matching ``flash_attn.layers.rotary.apply_rotary_emb``.

  Supports both ``(B, S, H, D)`` and packed ``(S, H, D)`` inputs where
  ``cos``/``sin`` have shape ``(S, D_rot // 2)`` (or ``(1, S, D_rot // 2)``).
  """
  import torch

  del inplace, seqlen_offsets, cu_seqlens, max_seqlen
  orig_dtype = x.dtype
  rot_dim = cos.shape[-1] * 2
  x_rot = x[..., :rot_dim].float()
  x_pass = x[..., rot_dim:]

  c = cos.float()
  s = sin.float()
  if x.ndim == 4:
    # x: (B, S, H, D), c/s: (S, D_rot//2) or (B, S, D_rot//2)
    if c.ndim == 2:
      c = c[None, :, None, :]
      s = s[None, :, None, :]
    elif c.ndim == 3:
      c = c[:, :, None, :]
      s = s[:, :, None, :]
  elif x.ndim == 3:
    # x: (S, H, D), c/s: (S, D_rot//2)
    if c.ndim == 2:
      c = c[:, None, :]
      s = s[:, None, :]
  else:
    raise ValueError(f"Unsupported rank {x.ndim} for apply_rotary_emb_torch")

  if interleaved:
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
  else:
    half = rot_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    out_rot = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)

  if x_pass.shape[-1] > 0:
    return torch.cat((out_rot.to(orig_dtype), x_pass), dim=-1)
  return out_rot.to(orig_dtype)


# ---------------------------------------------------------------------------
# Import-Time Shims for Upstream MAGI-2-preview
# ---------------------------------------------------------------------------


def find_upstream_root(explicit: str | Path | None = None) -> Path | None:
  """Locate the cloned ``MAGI-2-preview`` upstream repository if available."""
  cands: list[Path] = []
  if explicit:
    cands.append(Path(explicit).expanduser())
  env_root = os.environ.get("MAGI2_UPSTREAM_ROOT", "").strip()
  if env_root:
    cands.append(Path(env_root).expanduser())
  cands.extend([
      Path("/workspace/shun/magi2_ref/MAGI-2-preview"),
      Path("/usr/local/google/home/shunwang/tpus/magi2_ref/MAGI-2-preview"),
      Path(__file__).resolve().parents[4] / "magi2_ref" / "MAGI-2-preview",
  ])
  for p in cands:
    if (p / "inference" / "model" / "magi2_preview.py").exists():
      return p
  return None


def _register_dummy_module(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:
  mod = types.ModuleType(name)
  if attrs:
    for k, v in attrs.items():
      setattr(mod, k, v)
  sys.modules[name] = mod
  return mod


def _passthrough_decorator(*d_args: Any, **d_kwargs: Any) -> Any:
  if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
    return d_args[0]
  return lambda fn: fn


@contextlib.contextmanager
def disable_torch_compile_during_import():
  """Temporarily replace ``torch.compile`` with a no-op decorator during upstream import."""
  import torch

  orig = torch.compile
  torch.compile = _passthrough_decorator  # type: ignore[assignment]
  try:
    yield
  finally:
    torch.compile = orig


def install_tpu_shims(upstream_root: str | Path | None = None) -> Path | None:
  """Install ``sys.modules`` shims so upstream ``MAGI-2-preview`` imports cleanly on TPU.

  Idempotent and safe to call before importing upstream modules under either
  ``torchax`` or ``torch_tpu``.
  """
  import torch

  os.environ.setdefault("MAGI2_MHC_BACKEND", "torch")

  root = find_upstream_root(upstream_root)
  if root is not None and str(root) not in sys.path:
    sys.path.insert(0, str(root))

  # 1. triton / triton.language
  if "triton" not in sys.modules:
    tl_mod = _register_dummy_module(
        "triton.language",
        {
            "constexpr": Any,
            "float32": "float32",
            "bfloat16": "bfloat16",
            "int32": "int32",
        },
    )
    _register_dummy_module(
        "triton",
        {
            "language": tl_mod,
            "jit": _passthrough_decorator,
            "autotune": _passthrough_decorator,
            "heuristics": _passthrough_decorator,
            "Config": lambda *a, **kw: None,
            "cdiv": lambda a, b: (a + b - 1) // b,
            "next_power_of_2": lambda n: 1 << (max(1, int(n)) - 1).bit_length(),
        },
    )

  # 2. torch.library.triton_op / wrap_triton
  if not hasattr(torch.library, "triton_op"):
    torch.library.triton_op = _passthrough_decorator  # type: ignore[attr-defined]
  if not hasattr(torch.library, "wrap_triton"):
    torch.library.wrap_triton = lambda fn: fn  # type: ignore[attr-defined]

  # 3. flash_attn_interface + magi_attn_extensions.fa3_interface_with_sink
  if "flash_attn_interface" not in sys.modules:
    _register_dummy_module("flash_attn_interface", {"flash_attn_3_cuda": None})

  if "magi_attn_extensions.fa3_interface_with_sink" not in sys.modules:
    def _fa3_sink_dispatch(*args: Any, **kwargs: Any) -> Any:
      if _ACTIVE_HOOKS.attn_with_sink is not None:
        return _ACTIVE_HOOKS.attn_with_sink(*args, **kwargs)
      raise RuntimeError("No TPU attn_with_sink hook installed.")

    _register_dummy_module("magi_attn_extensions", {})
    _register_dummy_module(
        "magi_attn_extensions.fa3_interface_with_sink",
        {
            "FA3VarlenFuncWithSink": object,
            "fa3_func_with_sink": _fa3_sink_dispatch,
            "fa3_varlen_func_with_sink": _fa3_sink_dispatch,
        },
    )

  # 4. magi_compiler
  if "magi_compiler" not in sys.modules:
    @dataclass
    class _DummyCompileConfig:
      backend: str = "tpu"
      dynamic: bool = False

    _register_dummy_module(
        "magi_compiler",
        {
            "magi_compile": _passthrough_decorator,
            "magi_register_custom_op": _passthrough_decorator,
        },
    )
    _register_dummy_module(
        "magi_compiler.api",
        {"magi_register_custom_op": _passthrough_decorator},
    )
    _register_dummy_module(
        "magi_compiler.config",
        {"CompileConfig": _DummyCompileConfig},
    )

  # 5. flash_attn.layers.rotary
  if "flash_attn.layers.rotary" not in sys.modules:
    _register_dummy_module("flash_attn", {})
    _register_dummy_module("flash_attn.layers", {})
    _register_dummy_module(
        "flash_attn.layers.rotary",
        {"apply_rotary_emb": apply_rotary_emb_torch},
    )

  return root


# ---------------------------------------------------------------------------
# Fast 8-Thread Parallel Expert-Axis Sharded Checkpoint Loader (for PyTorch/TPU)
# ---------------------------------------------------------------------------


def load_safetensors_weight_map(checkpoint_dir: str | Path) -> tuple[Path, dict[str, str]]:
  """Read ``model.safetensors.index.json`` (or fall back to single safetensors)."""
  ckpt = Path(checkpoint_dir)
  idx_path = ckpt / "model.safetensors.index.json"
  if idx_path.exists():
    return ckpt, json.loads(idx_path.read_text())["weight_map"]
  single = ckpt / "model.safetensors"
  if single.exists():
    from ...safetensors_io import safe_open
    with safe_open(str(single)) as f:
      return ckpt, {k: "model.safetensors" for k in f.keys()}
  raise FileNotFoundError(f"No safetensors index or file in {ckpt}")


def stream_preview_weights_for_rank(
    checkpoint_dir: str | Path,
    *,
    rank: int,
    ep_size: int = 8,
    num_heads: int = 12,
    num_experts: int = 256,
    max_workers: int = 8,
) -> dict[str, np.ndarray]:
  """Parallel-stream Stage 1 Preview weights for a single EP rank (25.3 GiB/rank).

  Slices routed MoE tensors along the expert axis ``E = 256 -> E_local = 32``
  per head across all 12 heads, and swaps ``expert_bias_ema -> expert_bias``.
  Returns host NumPy arrays (bf16 stored as ``ml_dtypes.bfloat16`` / ``uint16``
  or ``np.float32``) ready for zero-copy ``torch.from_numpy`` / TPU transfer.
  """
  from ...safetensors_io import safe_open

  ckpt, wm = load_safetensors_weight_map(checkpoint_dir)
  e_local = num_experts // ep_size
  e_lo = rank * e_local
  e_hi = e_lo + e_local

  by_file: dict[str, list[str]] = {}
  for k, fn in wm.items():
    by_file.setdefault(fn, []).append(k)

  bias_src = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
  use_ema_bias = bias_src != "main"

  out: dict[str, np.ndarray] = {}

  def _load_one_shard(item: tuple[str, list[str]]) -> dict[str, np.ndarray]:
    fn, keys = item
    local_d: dict[str, np.ndarray] = {}
    with safe_open(str(ckpt / fn)) as f:
      for k in keys:
        if k.endswith(".router.expert_bias_ema"):
          if use_ema_bias:
            target = k[: -len("_ema")]
            local_d[target] = f.get_tensor(k)
          continue
        if k.endswith(".router.expert_bias"):
          if not use_ema_bias:
            local_d[k] = f.get_tensor(k)
          continue
        if k.endswith((".moe_mlp.W_gate", ".moe_mlp.W_up", ".moe_mlp.W_down")):
          # Stored as (H * E, out, in) or (H, E, out, in) -> slice [e_lo:e_hi] along E!
          arr = f.get_tensor(k)
          if arr.ndim == 3 and arr.shape[0] == num_heads * num_experts:
            arr = arr.reshape(num_heads, num_experts, arr.shape[1], arr.shape[2])
            local_d[k] = np.ascontiguousarray(arr[:, e_lo:e_hi, :, :])
          elif arr.ndim == 4 and arr.shape[0] == num_heads and arr.shape[1] == num_experts:
            local_d[k] = np.ascontiguousarray(arr[:, e_lo:e_hi, :, :])
          else:
            local_d[k] = arr
          continue
        local_d[k] = f.get_tensor(k)
    return local_d

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for shard_d in pool.map(_load_one_shard, sorted(by_file.items())):
      out.update(shard_d)
  return out
