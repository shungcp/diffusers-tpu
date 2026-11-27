"""Priority 2 (`torchax`) runner for ``MAGI-2-preview`` 113.93B DiT on TPU.

Executes :mod:`diffusers_tpu.models.magi2.torch_preview_dit` (PyTorch
``nn.Module`` code via ``torch.func.functional_call`` inside
``torchax.default_env()``) under a single outer ``jax.jit(jax.shard_map)`` with
``jax.lax.scan`` over the 36 identical Multi-Head MoE layers (layers 2..37).

Hot TPU v7x kernels are bridged via ``env.t2j_iso`` / ``env.j2t_iso`` through
:class:`diffusers_tpu.models.magi2.upstream_compat.Magi2TpuHooks`:
- **4-stream mHC**: Tokamax Pallas ``mhc_pre`` / ``mhc_fused_post_pre`` /
  ``mhc_post`` with exact MAGI-2 global-max Sinkhorn-Knopp epilogue.
- **Attention + 1-Token Sink**: In-tree Pallas ``splash.residual_attention``
  with base-2 sink + Ulysses CP=8 ``jax.lax.all_to_all``.
- **Modality GroupedLinear (`M=3`)**: ``jax.lax.ragged_dot``.
- **Multi-Head Latent MoE**: Zero-cumsum ``argsort[:cap]`` sliced-AG +
  MegaCore Dual-TC ``gmm_v2`` (supports both BF16 and FP8-W8A8).

Design highlights:
- 8-thread parallel safetensors loader streams 113.93B weights directly onto
  TPU HBM in ~20s with peak host RAM < 10 GiB (routed MoE weights sharded on
  the expert axis ``E = 256 -> 32/chip``).
- Matches :mod:`jax_preview_dit`'s exact call signature
  ``jitted_fn(packed_state, x, rope_or_coords, modality_mapping, time_token_sequence)``
  so it plugs directly into :mod:`jax_pipeline` / :mod:`torchax_pipeline`!
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Magi2PreviewConfig
from .jax_mh_moe import (
    mh_moe_forward_local,
    shard_routed_down,
    shard_routed_gate_up,
)
from .jax_preview_dit import _sdpa_with_sink_single_segment
from .rope_and_ops import (
    mhc_fused_post_pre_tokamax,
    mhc_post_tokamax,
    mhc_pre_tokamax,
    use_tokamax_mhc,
)
from .torch_preview_dit import (
    Magi2PostAdapterTorch,
    Magi2PreAdapterTorch,
    Magi2PreviewLayerTorch,
    Magi2PreviewTransformerTorch,
)
from .upstream_compat import Magi2TpuHooks, load_safetensors_weight_map

try:
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
except ImportError:  # pragma: no cover
  jax = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]
  Mesh = NamedSharding = P = None  # type: ignore[assignment]


_MOE_SHARDED_KEYS = frozenset({
    "mlp.w_gate_up",
    "mlp.w_down",
    "mlp.w_gate_up_scale",
    "mlp.w_down_scale",
})


def _put_replicated(arr_np: np.ndarray, mesh: Any, target_dtype: Any) -> Any:
  arr_j = jnp.asarray(arr_np, dtype=target_dtype)
  if mesh is None:
    return arr_j
  return jax.device_put(arr_j, NamedSharding(mesh, P()))


def _convert_grouped_linear_torch_np(
    w_pt: np.ndarray, num_modality: int
) -> np.ndarray:
  """Convert checkpoint ``(M*out, in)`` to PyTorch ``GroupedLinearTorch`` layout.

  - ``M == 1``: stays ``(out, in)`` for ``F.linear``.
  - ``M > 1``: ``(M, in, out)`` for ``jax.lax.ragged_dot``.
  """
  if num_modality == 1:
    return np.ascontiguousarray(w_pt)
  m_out, in_f = w_pt.shape
  out_f = m_out // num_modality
  return np.ascontiguousarray(
      w_pt.reshape(num_modality, out_f, in_f).transpose(0, 2, 1)
  )


def _convert_norm_np(w_pt: np.ndarray, num_modality: int, dim: int) -> np.ndarray:
  if num_modality == 1:
    return np.ascontiguousarray(w_pt.reshape(dim))
  return np.ascontiguousarray(w_pt.reshape(num_modality, dim))


def _stack_moe_layer_dicts(
    moe_layers: list[dict[str, Any]],
    *,
    mesh: Any | None,
    axis_name: str,
) -> dict[str, Any]:
  """Stack 36 identical MoE layer parameter dicts along axis 0, preserving EP sharding."""
  keys = sorted(moe_layers[0].keys())
  stacked: dict[str, Any] = {}
  for k in keys:
    arrs = [d.pop(k) for d in moe_layers]
    if k in _MOE_SHARDED_KEYS and mesh is not None:
      # Each arr has shape (ep, ...), spec P(axis_name, None, ...)
      # Stack on axis 0 -> shape (36, ep, ...), spec P(None, axis_name, None, ...)
      ndim = arrs[0].ndim
      out_spec = P(None, axis_name, *([None] * (ndim - 1)))
      out_sharding = NamedSharding(mesh, out_spec)
      fn = jax.jit(lambda xs: jnp.stack(xs, axis=0), out_shardings=out_sharding)
      stacked[k] = fn(arrs)
    else:
      stacked[k] = jnp.stack(arrs, axis=0)
      if mesh is not None:
        stacked[k] = jax.device_put(stacked[k], NamedSharding(mesh, P()))
    for a in arrs:
      if hasattr(a, "delete"):
        try:
          a.delete()
        except Exception:
          pass
    del arrs
  return stacked


def load_torchax_preview_params_from_safetensors(
    checkpoint_dir: str | Path,
    config: Magi2PreviewConfig | None = None,
    *,
    mesh: Any | None = None,
    axis_name: str = "tp",
    dtype: Any = None,
    fp8_moe: bool = False,
    max_workers: int = 8,
    verbose: bool = True,
) -> tuple[dict[str, Any], Magi2PreviewConfig]:
  """8-thread parallel streaming loader directly into sharded TPU JAX arrays.

  Returns ``(packed_params, config)`` where ``packed_params`` has PyTorch
  ``Magi2PreviewTransformerTorch`` state_dict parameter names structured as:
  ``{"pre": ..., "mm": {0: ..., 1: ..., 38: ..., 39: ...}, "moe_stacked": ..., "post": ...}``.
  """
  from ...safetensors_io import safe_open

  t0 = time.perf_counter()
  if config is None:
    config = Magi2PreviewConfig.from_checkpoint(checkpoint_dir)
  if dtype is None:
    dtype = jnp.bfloat16

  ep_size = int(mesh.shape[axis_name]) if mesh is not None else 1
  ckpt, wm = load_safetensors_weight_map(checkpoint_dir)

  H = config.moe_num_heads
  E = config.moe_num_experts
  d_head = config.moe_head_dim
  bias_src = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
  use_ema_bias = bias_src != "main"

  pre_d: dict[str, Any] = {}
  post_d: dict[str, Any] = {}
  layers_d: list[dict[str, Any]] = [{} for _ in range(config.num_layers)]
  pending_gate: dict[int, np.ndarray] = {}
  pending_up: dict[int, np.ndarray] = {}

  by_file: dict[str, list[str]] = {}
  for k, fn in wm.items():
    by_file.setdefault(fn, []).append(k)

  def _load_one_shard(item: tuple[str, list[str]]) -> list[tuple[str, np.ndarray]]:
    fn, keys = item
    res: list[tuple[str, np.ndarray]] = []
    with safe_open(str(ckpt / fn)) as f:
      for k in keys:
        if k == "pre_adapter.rope.bands":
          continue
        if k.endswith(".router.expert_bias_ema") and not use_ema_bias:
          continue
        if k.endswith(".router.expert_bias") and use_ema_bias:
          continue
        res.append((k, f.get_tensor(k)))
    return res

  def _assign_one(key: str, arr: np.ndarray) -> None:
    if key.startswith("pre_adapter."):
      sub = key[len("pre_adapter.") :]
      pre_d[sub] = _put_replicated(arr, mesh, jnp.float32)
      return
    if key.startswith("post_adapter."):
      sub = key[len("post_adapter.") :]
      post_d[sub] = _put_replicated(arr, mesh, jnp.float32)
      return

    if not key.startswith("block.layers."):
      raise KeyError(f"Unrecognized key: {key}")

    parts = key.split(".")
    li = int(parts[2])
    sub = ".".join(parts[3:])
    ld = layers_d[li]
    is_moe = config.is_moe_layer(li)
    M = config.num_modality_for_layer(li)

    # Attention
    if sub == "attention.pre_norm.weight":
      ld[sub] = _put_replicated(_convert_norm_np(arr, M, config.hidden_size), mesh, jnp.float32)
    elif sub in ("attention.linear_g.weight", "attention.linear_qkv.weight", "attention.linear_proj.weight"):
      ld[sub] = _put_replicated(_convert_grouped_linear_torch_np(arr, M), mesh, dtype)
    elif sub == "attention.sinks":
      ld[sub] = _put_replicated(arr, mesh, jnp.float32)
    elif sub in ("attention.q_norm.weight", "attention.k_norm.weight"):
      ld[sub] = _put_replicated(_convert_norm_np(arr, M, config.head_dim), mesh, jnp.float32)
    # mHC
    elif sub == "mhc_norm.weight":
      ld[sub] = _put_replicated(_convert_norm_np(arr, M, config.stream_hidden_size), mesh, jnp.float32)
    elif sub.startswith("mhc_"):
      ld[sub] = _put_replicated(arr, mesh, jnp.float32)
    # Dense MLP
    elif not is_moe:
      if sub == "mlp.pre_norm.weight":
        ld[sub] = _put_replicated(_convert_norm_np(arr, M, config.hidden_size), mesh, jnp.float32)
      elif sub in ("mlp.up_gate_proj.weight", "mlp.down_proj.weight"):
        ld[sub] = _put_replicated(_convert_grouped_linear_torch_np(arr, M), mesh, dtype)
      else:
        raise KeyError(f"Unrecognized Dense MLP key: {key}")
    # MoE MLP
    else:
      if sub == "mlp.pre_norm.weight":
        ld[sub] = _put_replicated(_convert_norm_np(arr, 3, config.hidden_size), mesh, jnp.float32)
      elif sub in (
          "mlp.split_linear.weight",
          "mlp.merge_linear.weight",
          "mlp.shared_expert_fc1.weight",
          "mlp.shared_expert_fc2.weight",
      ):
        ld[sub] = _put_replicated(arr, mesh, dtype)
      elif sub in (
          "mlp.modality_specific_shared_expert_fc1.weight",
          "mlp.modality_specific_shared_expert_fc2.weight",
      ):
        ld[sub] = _put_replicated(_convert_grouped_linear_torch_np(arr, 3), mesh, dtype)
      elif sub == "mlp.moe_mlp.gate":
        ld["mlp.gate"] = _put_replicated(arr.reshape(H, E, d_head), mesh, jnp.float32)
      elif sub in ("mlp.moe_mlp.router.expert_bias_ema", "mlp.moe_mlp.router.expert_bias"):
        ld["mlp.expert_bias"] = _put_replicated(arr.reshape(H, E), mesh, jnp.float32)
      elif sub == "mlp.moe_mlp.W_gate":
        w_pair = None
        with gate_lock:
          if li in pending_up:
            w_pair = (arr, pending_up.pop(li))
          else:
            pending_gate[li] = arr
        if w_pair is not None:
          w_sh, sc_sh = shard_routed_gate_up(
              w_pair[0], w_pair[1], num_heads=H, num_experts=E, mesh=mesh,
              axis_name=axis_name, dtype=dtype, fp8=fp8_moe,
          )
          ld["mlp.w_gate_up"] = w_sh
          if fp8_moe and sc_sh is not None:
            ld["mlp.w_gate_up_scale"] = sc_sh
      elif sub == "mlp.moe_mlp.W_up":
        w_pair = None
        with gate_lock:
          if li in pending_gate:
            w_pair = (pending_gate.pop(li), arr)
          else:
            pending_up[li] = arr
        if w_pair is not None:
          w_sh, sc_sh = shard_routed_gate_up(
              w_pair[0], w_pair[1], num_heads=H, num_experts=E, mesh=mesh,
              axis_name=axis_name, dtype=dtype, fp8=fp8_moe,
          )
          ld["mlp.w_gate_up"] = w_sh
          if fp8_moe and sc_sh is not None:
            ld["mlp.w_gate_up_scale"] = sc_sh
      elif sub == "mlp.moe_mlp.W_down":
        w_sh, sc_sh = shard_routed_down(
            arr, num_heads=H, num_experts=E, mesh=mesh,
            axis_name=axis_name, dtype=dtype, fp8=fp8_moe,
        )
        ld["mlp.w_down"] = w_sh
        if fp8_moe and sc_sh is not None:
          ld["mlp.w_down_scale"] = sc_sh
      else:
        raise KeyError(f"Unrecognized MoE key: {key}")

  import threading
  gate_lock = threading.Lock()

  def _load_and_assign(item: tuple[str, list[str]]) -> None:
    for k, arr in _load_one_shard(item):
      _assign_one(k, arr)

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    list(pool.map(_load_and_assign, sorted(by_file.items())))

  if pending_gate or pending_up:
    raise RuntimeError(f"Unpaired W_gate/W_up: gate={sorted(pending_gate)}, up={sorted(pending_up)}")

  mm_dict = {i: layers_d[i] for i in config.mm_layers}
  moe_list = [layers_d[i] for i in config.moe_layers]
  moe_stacked = _stack_moe_layer_dicts(moe_list, mesh=mesh, axis_name=axis_name)

  packed = {
      "pre": pre_d,
      "mm": mm_dict,
      "moe_stacked": moe_stacked,
      "post": post_d,
  }
  if verbose:
    print(
        f"[torchax_magi2] Loaded 113.93B Preview into TPU HBM in "
        f"{time.perf_counter() - t0:.2f}s [ep={ep_size}, fp8={fp8_moe}, dtype={dtype}]",
        flush=True,
    )
  return packed, config


def pack_torch_model_for_torchax(
    model: Magi2PreviewTransformerTorch,
    *,
    mesh: Any | None = None,
    axis_name: str = "tp",
) -> dict[str, Any]:
  """Convert an in-memory ``Magi2PreviewTransformerTorch`` (``ep_size=1``) to packed JAX arrays."""
  cfg = model.config
  ep_size = int(mesh.shape[axis_name]) if mesh is not None else 1

  def _t2np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
      import ml_dtypes
      return t.detach().cpu().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.detach().cpu().numpy()

  def _mod_sd(m: torch.nn.Module, is_moe: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, p in m.named_parameters():
      arr_np = _t2np(p)
      if is_moe and k in ("mlp.w_gate_up", "mlp.w_down"):
        # (H, E, d1, d2) -> (ep, H, E_loc, d1, d2)
        H, E, d1, d2 = arr_np.shape
        e_loc = E // ep_size
        arr_sh = np.ascontiguousarray(
            arr_np.reshape(H, ep_size, e_loc, d1, d2).transpose(1, 0, 2, 3, 4)
        )
        arr_j = jnp.asarray(arr_sh)
        if mesh is not None:
          arr_j = jax.device_put(
              arr_j, NamedSharding(mesh, P(axis_name, None, None, None, None))
          )
        out[k] = arr_j
      else:
        out[k] = _put_replicated(arr_np, mesh, jnp.asarray(arr_np).dtype)
    return out

  pre_d = _mod_sd(model.pre_adapter)
  post_d = _mod_sd(model.post_adapter)
  mm_dict = {i: _mod_sd(model.layers[i]) for i in cfg.mm_layers}
  moe_list = [_mod_sd(model.layers[i], is_moe=True) for i in cfg.moe_layers]
  moe_stacked = _stack_moe_layer_dicts(moe_list, mesh=mesh, axis_name=axis_name)
  return {"pre": pre_d, "mm": mm_dict, "moe_stacked": moe_stacked, "post": post_d}


def _build_packed_param_specs(
    packed: dict[str, Any], axis_name: str
) -> dict[str, Any]:
  """Build matching ``PartitionSpec`` PyTree for ``jax.shard_map``."""
  moe_specs: dict[str, Any] = {}
  for k, v in packed["moe_stacked"].items():
    if k in _MOE_SHARDED_KEYS:
      moe_specs[k] = P(None, axis_name, *([None] * (v.ndim - 2)))
    else:
      moe_specs[k] = P()
  return {
      "pre": jax.tree_util.tree_map(lambda _: P(), packed["pre"]),
      "mm": jax.tree_util.tree_map(lambda _: P(), packed["mm"]),
      "moe_stacked": moe_specs,
      "post": jax.tree_util.tree_map(lambda _: P(), packed["post"]),
  }


def make_torchax_jitted_forward(
    packed_params: dict[str, Any],
    config: Magi2PreviewConfig,
    *,
    sample_lens: tuple[int, ...],
    mesh: Any | None = None,
    axis_name: str = "tp",
    has_time: bool = True,
    embedded_input: bool = False,
    use_splash: bool = True,
    moe_mode: str | None = None,
    capacity_factor: float | None = None,
    use_tokamax: bool | None = None,
    dtype: torch.dtype = torch.bfloat16,
    fp8_moe: bool = False,
):
  """Compile ``Magi2PreviewTransformerTorch`` under ``torchax`` + ``shard_map`` + ``lax.scan``.

  Returns ``(jitted_fn, packed_params)`` with identical invocation signature to
  ``jax_preview_dit.make_jitted_forward``:
    ``out = jitted_fn(packed_params, x, rope_or_coords, modality_mapping, time_token_sequence)``
  """
  os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
  import torchax

  env = torchax.default_env()
  cp_size = int(mesh.shape[axis_name]) if mesh is not None else 1
  tokamax_mhc = use_tokamax_mhc() if use_tokamax is None else bool(use_tokamax)

  # Build meta-device PyTorch template modules (zero host RAM!).
  with torch.device("meta"):
    tpl_full = Magi2PreviewTransformerTorch(
        config, ep_size=cp_size, dtype=dtype, fp8_moe=fp8_moe
    )
    tpl_pre: Magi2PreAdapterTorch = tpl_full.pre_adapter
    tpl_mm: Magi2PreviewLayerTorch = tpl_full.layers[config.mm_layers[0]]  # type: ignore[assignment]
    tpl_moe: Magi2PreviewLayerTorch = tpl_full.layers[config.moe_layers[0]]  # type: ignore[assignment]
    tpl_post: Magi2PostAdapterTorch = tpl_full.post_adapter

  # Build Magi2TpuHooks bridging torchax tensors <-> JAX/Pallas/Tokamax kernels.
  def _grouped_linear_hook(x_t, w_t, gs_t, _mod_t):
    x_j, w_j, gs_j = env.t2j_iso((x_t, w_t, gs_t))
    out_j = jax.lax.ragged_dot(
        x_j, w_j, gs_j, preferred_element_type=jnp.float32
    ).astype(x_j.dtype)
    return env.j2t_iso(out_j)

  def _ulysses_s2h_hook(x_t):
    x_j = env.t2j_iso(x_t)
    # (S_loc, H, D) -> (S_loc, cp_size, H_loc, D) -> (S_total, H_loc, D)
    s_loc, h_tot, d = x_j.shape
    h_loc = h_tot // cp_size
    x_4d = x_j.reshape(s_loc, cp_size, h_loc, d)
    out_j = jax.lax.all_to_all(x_4d, axis_name, split_axis=1, concat_axis=0)
    return env.j2t_iso(out_j.reshape(cp_size * s_loc, h_loc, d))

  def _ulysses_h2s_hook(x_t):
    x_j = env.t2j_iso(x_t)
    # (S_total, H_loc, D) -> (cp_size, S_loc, H_loc, D) -> (S_loc, H, D)
    s_tot, h_loc, d = x_j.shape
    s_loc = s_tot // cp_size
    x_4d = x_j.reshape(cp_size, s_loc, h_loc, d)
    out_j = jax.lax.all_to_all(x_4d, axis_name, split_axis=0, concat_axis=1)
    return env.j2t_iso(out_j.reshape(s_loc, cp_size * h_loc, d))

  def _attn_with_sink_hook(q_t, k_t, v_t, sinks_t, slens, _scale):
    q_j, k_j, v_j, sinks_j = env.t2j_iso((q_t, k_t, v_t, sinks_t))
    s_total, h_loc, d = q_j.shape
    if cp_size > 1:
      rank = jax.lax.axis_index(axis_name)
      sinks_loc = jax.lax.dynamic_slice_in_dim(
          sinks_j, rank * h_loc, h_loc, axis=1
      )
    else:
      sinks_loc = sinks_j

    # (S_total, H_loc, D) -> (1, H_loc, S_total, D)
    q_bhsd = jnp.swapaxes(q_j, 0, 1)[None, :, :, :]
    k_bhsd = jnp.swapaxes(k_j, 0, 1)[None, :, :, :]
    v_bhsd = jnp.swapaxes(v_j, 0, 1)[None, :, :, :]

    outs = []
    offset = 0
    for sl in slens:
      o_seg = _sdpa_with_sink_single_segment(
          q_bhsd[:, :, offset : offset + sl, :],
          k_bhsd[:, :, offset : offset + sl, :],
          v_bhsd[:, :, offset : offset + sl, :],
          sinks_loc,
          use_splash=use_splash,
      )
      outs.append(jnp.swapaxes(o_seg[0], 0, 1))  # (sl, H_loc, D)
      offset += sl
    if offset < s_total:
      outs.append(jnp.zeros((s_total - offset, h_loc, d), dtype=q_j.dtype))
    out_j = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)
    return env.j2t_iso(out_j)

  def _mh_moe_hook(x_t, gate_t, bias_t, wgu_t, wdn_t, wgus_t, wdns_t):
    x_j, gate_j, bias_j, wgu_j, wdn_j = env.t2j_iso(
        (x_t, gate_t, bias_t, wgu_t, wdn_t)
    )
    wgus_j = env.t2j_iso(wgus_t) if wgus_t is not None else None
    wdns_j = env.t2j_iso(wdns_t) if wdns_t is not None else None
    out_j = mh_moe_forward_local(
        x_j,
        gate_j,
        bias_j,
        wgu_j,
        wdn_j,
        num_heads=config.moe_num_heads,
        num_experts=config.moe_num_experts,
        top_k=config.moe_top_k,
        ep_size=cp_size,
        axis_name=axis_name,
        mode=moe_mode,
        capacity_factor=capacity_factor,
        route_norm=config.moe_route_norm,
        route_scale=config.moe_route_scale,
        w_gate_up_scale_local=wgus_j,
        w_down_scale_local=wdns_j,
    )
    return env.j2t_iso(out_j)

  def _mhc_pre_hook(
      s_flat_t, norm_w_t, phi_t, a_pre_t, a_post_t, a_res_t, b_pre_t, b_post_t, b_res_t
  ):
    s_j, nw_j, phi_j, ap_j, apo_j, ar_j, bp_j, bpo_j, br_j = env.t2j_iso((
        s_flat_t, norm_w_t, phi_t, a_pre_t, a_post_t, a_res_t, b_pre_t, b_post_t, b_res_t
    ))
    x_in, pm, cm = mhc_pre_tokamax(
        s_j, nw_j, phi_j, ap_j, bp_j, apo_j, bpo_j, ar_j, br_j,
        n_stream=config.mhc_num_stream,
        eps=config.rms_norm_eps,
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        sinkhorn_eps=config.mhc_sinkhorn_eps,
    )
    return env.j2t_iso(x_in), env.j2t_iso(pm), env.j2t_iso(cm)

  def _mhc_post_hook(s_flat_t, sub_out_t, pm_t, cm_t):
    s_j, sub_j, pm_j, cm_j = env.t2j_iso((s_flat_t, sub_out_t, pm_t, cm_t))
    out_j = mhc_post_tokamax(sub_j, s_j, pm_j, cm_j)
    return env.j2t_iso(out_j)

  def _mhc_fpp_hook(
      s_flat_t, sub_out_t, pm_t, cm_t,
      norm_w_t, phi_t, a_pre_t, a_post_t, a_res_t, b_pre_t, b_post_t, b_res_t,
  ):
    (
        s_j, sub_j, pm_j, cm_j,
        nw_j, phi_j, ap_j, apo_j, ar_j, bp_j, bpo_j, br_j,
    ) = env.t2j_iso((
        s_flat_t, sub_out_t, pm_t, cm_t,
        norm_w_t, phi_t, a_pre_t, a_post_t, a_res_t, b_pre_t, b_post_t, b_res_t,
    ))
    s_mid, x_next, pm_next, cm_next = mhc_fused_post_pre_tokamax(
        sub_j, s_j, pm_j, cm_j,
        nw_j, phi_j, ap_j, bp_j, apo_j, bpo_j, ar_j, br_j,
        n_stream=config.mhc_num_stream,
        eps=config.rms_norm_eps,
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        sinkhorn_eps=config.mhc_sinkhorn_eps,
    )
    return (
        env.j2t_iso(s_mid),
        env.j2t_iso(x_next),
        env.j2t_iso(pm_next),
        env.j2t_iso(cm_next),
    )

  hooks = Magi2TpuHooks(
      attn_with_sink=_attn_with_sink_hook,
      grouped_linear=_grouped_linear_hook,
      mh_moe_routed=_mh_moe_hook,
      mhc_pre=_mhc_pre_hook if tokamax_mhc else None,
      mhc_post=_mhc_post_hook if tokamax_mhc else None,
      mhc_fused_post_pre=_mhc_fpp_hook if tokamax_mhc else None,
      ulysses_scatter_head_gather_seq=_ulysses_s2h_hook if cp_size > 1 else None,
      ulysses_scatter_seq_gather_head=_ulysses_h2s_hook if cp_size > 1 else None,
      ep_size=cp_size,
  )

  def _run_local(st, x_l, rope_l, mod_l, time_l):
    # 1. PreAdapter + timestep injection + 3D RoPE + modality sort (via torchax)
    with env:
      x_t = env.j2t_iso(x_l)
      rope_in_t = env.j2t_iso(rope_l)
      mod_in_t = env.j2t_iso(mod_l)
      time_t = env.j2t_iso(time_l) if (has_time and time_l is not None) else None

      mod_ids_t = torch.where(
          mod_in_t == 3, torch.full_like(mod_in_t, 2), mod_in_t
      ).to(torch.int64)
      if rope_in_t.shape[-1] == 9:
        from .torch_preview_dit import compute_3d_rope_96_torch
        rope_96_t = compute_3d_rope_96_torch(rope_in_t, config.rope_temperature)
      else:
        rope_96_t = rope_in_t.float()

      if embedded_input:
        x_emb_t = x_t.float()
      else:
        pre_t = {k: env.j2t_iso(v) for k, v in st["pre"].items()}
        x_emb_t = torch.func.functional_call(tpl_pre, pre_t, (x_t, mod_ids_t))

      if time_t is not None:
        t_dim = time_t.shape[-1]
        x_emb_t = torch.cat((time_t.float(), x_emb_t[:, t_dim:]), dim=-1)
      x_act_t = x_emb_t.to(dtype)

      perm_t = torch.argsort(mod_ids_t, stable=True)
      inv_perm_t = torch.zeros_like(perm_t).scatter(
          0,
          perm_t,
          torch.arange(perm_t.shape[0], device=perm_t.device, dtype=perm_t.dtype),
      )
      mod_s_t = mod_ids_t.index_select(0, perm_t)
      gs_t = torch.stack([
          (mod_s_t == 0).sum(),
          (mod_s_t == 1).sum(),
          (mod_s_t == 2).sum(),
      ]).to(torch.int32)
      x_s_t = x_act_t.index_select(0, perm_t)

      # Layers 0, 1 (MM Dense)
      for li in (0, 1):
        p_t = {k: env.j2t_iso(v) for k, v in st["mm"][li].items()}
        x_s_t = torch.func.functional_call(
            tpl_mm,
            p_t,
            (x_s_t, rope_96_t, mod_s_t, gs_t, perm_t, inv_perm_t, sample_lens),
            {"hooks": hooks},
        )

      # Extract pure JAX tensors before entering jax.lax.scan (no leaked torchax tracers!).
      x_s_j, rope_96_j, mod_s_j, gs_j, perm_j, inv_perm_j, mod_ids_j = env.t2j_iso((
          x_s_t, rope_96_t, mod_s_t, gs_t, perm_t, inv_perm_t, mod_ids_t
      ))

    # 2. Layers 2..37 (36 identical Multi-Head MoE layers via jax.lax.scan!)
    def _moe_scan_step(carry_j, one_layer_j):
      with env:
        p_t = {}
        for k, v in one_layer_j.items():
          v_use = v[0] if k in _MOE_SHARDED_KEYS else v
          p_t[k] = env.j2t_iso(v_use)
        out_t = torch.func.functional_call(
            tpl_moe,
            p_t,
            (
                env.j2t_iso(carry_j),
                env.j2t_iso(rope_96_j),
                env.j2t_iso(mod_s_j),
                env.j2t_iso(gs_j),
                env.j2t_iso(perm_j),
                env.j2t_iso(inv_perm_j),
                sample_lens,
            ),
            {"hooks": hooks},
        )
        return env.t2j_iso(out_t), None

    x_s_j, _ = jax.lax.scan(_moe_scan_step, x_s_j, st["moe_stacked"])

    # 3. Layers 38, 39 (MM Dense) + unpermute + PostAdapter
    with env:
      x_s_t = env.j2t_iso(x_s_j)
      rope_96_t = env.j2t_iso(rope_96_j)
      mod_s_t = env.j2t_iso(mod_s_j)
      gs_t = env.j2t_iso(gs_j)
      perm_t = env.j2t_iso(perm_j)
      inv_perm_t = env.j2t_iso(inv_perm_j)
      mod_ids_t = env.j2t_iso(mod_ids_j)

      for li in (38, 39):
        p_t = {k: env.j2t_iso(v) for k, v in st["mm"][li].items()}
        x_s_t = torch.func.functional_call(
            tpl_mm,
            p_t,
            (x_s_t, rope_96_t, mod_s_t, gs_t, perm_t, inv_perm_t, sample_lens),
            {"hooks": hooks},
        )
      x_seq_t = x_s_t.index_select(0, inv_perm_t)
      post_t = {k: env.j2t_iso(v) for k, v in st["post"].items()}
      out_t = torch.func.functional_call(tpl_post, post_t, (x_seq_t, mod_ids_t))
      return env.t2j_iso(out_t)

  if mesh is not None and cp_size > 1:
    st_specs = _build_packed_param_specs(packed_params, axis_name)
    in_specs = (
        st_specs,
        P(axis_name, None),
        P(axis_name, None),
        P(axis_name,),
        P(axis_name, None) if has_time else P(),
    )
    out_spec = P(axis_name, None)
    sharded = jax.shard_map(
        _run_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_spec,
        check_vma=False,
    )
    return jax.jit(sharded), packed_params

  return jax.jit(_run_local), packed_params


class _TorchaxPreAdapterShim:
  """Thin PreAdapter wrapper over ``packed_params['pre']`` matching ``jax_pipeline``."""

  def __init__(self, pre_params: dict[str, Any]) -> None:
    self._p = pre_params
    self.text_embedder_bias = pre_params["text_embedder.bias"]

  def embed_video(self, x_video: Any) -> Any:
    return (
        jnp.matmul(
            x_video.astype(jnp.float32),
            self._p["video_embedder.weight"].T,
            precision=jax.lax.Precision.HIGHEST,
        )
        + self._p["video_embedder.bias"]
    )

  def embed_audio(self, x_audio: Any) -> Any:
    return (
        jnp.matmul(
            x_audio.astype(jnp.float32),
            self._p["audio_embedder.weight"].T,
            precision=jax.lax.Precision.HIGHEST,
        )
        + self._p["audio_embedder.bias"]
    )

  def embed_text(self, x_text: Any) -> Any:
    return (
        jnp.matmul(
            x_text.astype(jnp.float32),
            self._p["text_embedder.weight"].T,
            precision=jax.lax.Precision.HIGHEST,
        )
        + self._p["text_embedder.bias"]
    )


class TorchaxPreviewModel:
  """Pipeline-compatible container holding sharded ``torchax`` Preview parameters."""

  def __init__(
      self,
      packed_params: dict[str, Any],
      config: Magi2PreviewConfig,
      *,
      axis_name: str = "tp",
      dtype: Any = None,
      fp8_moe: bool = False,
  ) -> None:
    self.packed_params = packed_params
    self.config = config
    self.axis_name = axis_name
    self.dtype = jnp.bfloat16 if dtype is None else dtype
    self.fp8_moe = fp8_moe
    self.pre_adapter = _TorchaxPreAdapterShim(packed_params["pre"])


def make_torchax_preview_step_fn(
    model: TorchaxPreviewModel,
    *,
    video_thw: tuple[int, int, int],
    n_aud: int,
    sample_lens: tuple[int, int],
    mesh: Any,
    use_splash: bool | None = None,
    moe_mode: str | None = None,
    capacity_factor: float | None = None,
):
  """Fused ``jax.jit`` CFG step around :func:`make_torchax_jitted_forward`."""
  from jax import lax
  from .rope_and_ops import sinusoidal_embedding_1d

  t_v, h_v, w_v = video_thw
  n_vid = t_v * h_v * w_v
  s_cond, s_uncond = int(sample_lens[0]), int(sample_lens[1])
  seg_lens = (s_cond, s_uncond)
  ax = model.axis_name
  cp_size = int(dict(mesh.shape).get(ax, 1)) if mesh is not None else 1
  time_dim = model.config.time_channel_dim
  v_ch = model.config.video_in_channels
  a_ch = model.config.audio_in_channels
  model_dtype = model.dtype
  pt_dtype = torch.float32 if model_dtype == jnp.float32 else torch.bfloat16

  core_fn, core_state = make_torchax_jitted_forward(
      model.packed_params,
      model.config,
      sample_lens=seg_lens,
      mesh=mesh,
      axis_name=ax,
      has_time=(time_dim > 0),
      embedded_input=True,
      use_splash=(True if use_splash is None else bool(use_splash)),
      moe_mode=moe_mode,
      capacity_factor=capacity_factor,
      dtype=pt_dtype,
      fp8_moe=model.fp8_moe,
  )

  seq_sharding = NamedSharding(mesh, P(ax, None)) if (mesh is not None and cp_size > 1) else None
  rep_sharding = NamedSharding(mesh, P()) if (mesh is not None and cp_size > 1) else None

  @jax.jit
  def _step_fn(
      st,
      latent_ncthw,
      audio_btc,
      raw_t_scalar,
      video_cfg,
      audio_cfg,
      txt_cond_emb,
      ref_cond_emb,
      txt_uncond_emb,
      ref_uncond_emb,
      pad_emb,
      rope_all,
      mod_all,
      time_kind_all,
  ):
    pre_s = st["pre"]
    v_flat = jnp.transpose(latent_ncthw[0], (1, 2, 3, 0)).reshape(n_vid, v_ch)
    if seq_sharding is not None and (n_vid % cp_size == 0):
      v_flat = lax.with_sharding_constraint(v_flat, seq_sharding)
    v_emb = (
        jnp.matmul(v_flat.astype(jnp.float32), pre_s["video_embedder.weight"].T, precision=lax.Precision.HIGHEST)
        + pre_s["video_embedder.bias"]
    ).astype(model_dtype)
    a_emb = (
        jnp.matmul(audio_btc[0].astype(jnp.float32), pre_s["audio_embedder.weight"].T, precision=lax.Precision.HIGHEST)
        + pre_s["audio_embedder.bias"]
    ).astype(model_dtype)

    parts = [v_emb, a_emb, txt_cond_emb]
    if ref_cond_emb.shape[0] > 0:
      parts.append(ref_cond_emb)
    parts.extend([v_emb, a_emb, txt_uncond_emb])
    if ref_uncond_emb.shape[0] > 0:
      parts.append(ref_uncond_emb)
    if pad_emb.shape[0] > 0:
      parts.append(pad_emb)
    x_emb = jnp.concatenate(parts, axis=0)
    if seq_sharding is not None:
      x_emb = lax.with_sharding_constraint(x_emb, seq_sharding)

    if time_dim > 0:
      emb_noisy = sinusoidal_embedding_1d(raw_t_scalar, dim=time_dim)
      emb_clean = sinusoidal_embedding_1d(jnp.float32(0.0), dim=time_dim)
      emb_zero = jnp.zeros((time_dim,), dtype=jnp.float32)
      time_all = jnp.where(
          (time_kind_all == 2)[:, None],
          emb_noisy[None, :],
          jnp.where(
              (time_kind_all == 1)[:, None],
              emb_clean[None, :],
              emb_zero[None, :],
          ),
      )
      if seq_sharding is not None:
        time_all = lax.with_sharding_constraint(time_all, seq_sharding)
    else:
      time_all = None

    out_all = core_fn(st, x_emb, rope_all, mod_all, time_all)

    cond_v = jnp.transpose(
        out_all[0:n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch), (3, 0, 1, 2)
    )[None, ...]
    cond_a = out_all[n_vid : n_vid + n_aud, :a_ch][None, ...]

    uncond_v = jnp.transpose(
        out_all[s_cond : s_cond + n_vid, :v_ch].reshape(t_v, h_v, w_v, v_ch),
        (3, 0, 1, 2),
    )[None, ...]
    uncond_a = out_all[s_cond + n_vid : s_cond + n_vid + n_aud, :a_ch][None, ...]

    v_cfg_video = uncond_v + video_cfg * (cond_v - uncond_v)
    v_cfg_audio = uncond_a + audio_cfg * (cond_a - uncond_a)
    if rep_sharding is not None:
      v_cfg_video = lax.with_sharding_constraint(v_cfg_video, rep_sharding)
      v_cfg_audio = lax.with_sharding_constraint(v_cfg_audio, rep_sharding)
    return v_cfg_video, v_cfg_audio

  return _step_fn, core_state

