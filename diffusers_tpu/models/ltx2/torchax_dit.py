"""``torchax`` backend for VideoGym-LTX-2.3 (22B 48-block Audio-Video DiT).

Traces pure-PyTorch :class:`LTX2TransformerTorch` (``ltx2/torch_dit.py``)
through ``torchax`` into the exact same single fused 2D ``(cfg, sp)``
``shard_map`` + Pallas ``splash.attention`` program as ``jax_sp.py``, supporting:
  * Auto ``4x2`` (``cfg=4 x sp=2``) physical-chip-aligned CFG parallelism on ``v7x-8``
  * Partial or full ``resident`` block placement (48/48 on ``v7x`` 95 GiB)
  * Optional on-TPU W8A8 E4M3 FP8 (``--fp8``) and FP8 QK Splash (``--qk-fp8``)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch

from . import jax_sp
from .torch_dit import (
    LTX2TpuHooks,
    LTX2TransformerTorch,
    load_torch_ltx2_dit,
)


def _quantize_w_fp8_tpu(w_nk: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Quantize ``(out, in)`` BF16 weight -> ``(in, out)`` E4M3 FP8 + ``(1, out)`` f32 scale."""
  w_kn = jnp.swapaxes(w_nk, 0, 1)
  f8_max = jnp.float32(448.0)
  wf = w_kn.astype(jnp.float32)
  scale = jnp.maximum(jnp.max(jnp.abs(wf), axis=0, keepdims=True), jnp.float32(1e-12)) / f8_max
  qw = jnp.clip(wf / scale, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
  return qw, scale


_quantize_w_fp8_jit = jax.jit(_quantize_w_fp8_tpu)


def _should_fp8_param(key: str, arr: jax.Array) -> bool:
  if not key.startswith("blocks.") or not key.endswith(".weight"):
    return False
  if ".to_gate_logits." in key or arr.ndim != 2:
    return False
  return min(arr.shape) >= 512


def load_and_place_torchax_dit(
    snapshot_dir: str,
    mesh: Any,
    axis: str = "sp",
    resident: set[int] | tuple[int, ...] = (),
    *,
    fp8: bool = False,
) -> tuple[LTX2TransformerTorch, dict[str, jax.Array], dict[str, Any]]:
  """Load ``LTX2TransformerTorch`` and shard its functional JAX weights on ``mesh``."""
  from jax.sharding import NamedSharding, PartitionSpec as P

  model = load_torch_ltx2_dit(snapshot_dir, dtype=torch.bfloat16, fp8=False, device=None, verbose=False)
  model.eval()
  cfg = dict(model.cfg)
  res_set = set(int(i) for i in resident)
  sp_size = int(mesh.shape[axis])

  def _spec(key: str, shape: tuple[int, ...]) -> Any:
    if key.startswith("blocks."):
      bi = int(key.split(".")[1])
      if bi not in res_set and len(shape) >= 1 and shape[0] % sp_size == 0 and sp_size > 1:
        return P(axis)
    return P()

  params: dict[str, jax.Array] = {}
  for name, p in list(model.named_parameters()):
    arr_cpu = p.detach().contiguous()
    if arr_cpu.dtype == torch.bfloat16:
      raw = arr_cpu.view(torch.uint16).numpy()
      arr = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.bfloat16)
    else:
      arr = jnp.asarray(arr_cpu.numpy())
    if fp8 and _should_fp8_param(name, arr):
      w_dev = jax.device_put(arr, NamedSharding(mesh, P()))
      qw, sc = _quantize_w_fp8_jit(w_dev)
      del w_dev
      params[name] = jax.device_put(qw, NamedSharding(mesh, _spec(name, qw.shape)))
      sc_key = name[:-len(".weight")] + ".scale"
      params[sc_key] = jax.device_put(sc, NamedSharding(mesh, P()))
    else:
      params[name] = jax.device_put(arr, NamedSharding(mesh, _spec(name, arr.shape)))

  # Release CPU weights from nn.Module shell (keeps structure only for functional tracing).
  for mod in model.modules():
    for k in list(mod._parameters.keys()):
      if mod._parameters[k] is not None:
        mod._parameters[k] = torch.nn.Parameter(
            torch.empty(0, dtype=mod._parameters[k].dtype, device="meta"), requires_grad=False
        )
  return model, params, cfg


def make_torchax_forward(
    model: LTX2TransformerTorch,
    params_sample: dict[str, jax.Array],
    mesh: Any,
    axis: str = "sp",
    resident: set[int] | tuple[int, ...] = (),
    stg_blocks: tuple[int, ...] = (),
    *,
    cfg_axis: str | None = None,
    fp8: bool = False,
    qk_fp8: bool = False,
):
  """Return ``forward_fn(params, x, a, ctx, actx, tv, ta, tt, token_t, rope, stg, iso)`` via ``torchax``."""
  import torchax
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  from diffusers_tpu.kernels import block_sizes, splash  # noqa: F401

  if cfg_axis is None and len(mesh.axis_names) == 2:
    cfg_axis, axis = mesh.axis_names[0], mesh.axis_names[1]
  elif axis not in mesh.axis_names:
    axis = mesh.axis_names[-1]
  sp_size = int(mesh.shape[axis])
  c_ax = cfg_axis if (cfg_axis is not None and int(mesh.shape[cfg_axis]) > 1) else None
  s_ax = axis if sp_size > 1 else None
  res_set = set(int(i) for i in resident)
  stg_set = set(int(i) for i in stg_blocks)
  num_layers = len(model.blocks)

  top_keys = [k for k in params_sample if not k.startswith("blocks.")]
  blk_keys: list[list[str]] = [[] for _ in range(num_layers)]
  for k in params_sample:
    if k.startswith("blocks."):
      blk_keys[int(k.split(".")[1])].append(k)

  def _spec(key: str, shape: tuple[int, ...]) -> Any:
    if key.startswith("blocks."):
      bi = int(key.split(".")[1])
      if bi not in res_set and len(shape) >= 1 and shape[0] % sp_size == 0 and sp_size > 1:
        return P(axis)
    return P()

  blk_specs = [{k: _spec(k, params_sample[k].shape) for k in ks} for ks in blk_keys]
  top_specs = {k: P() for k in top_keys}

  env = torchax.default_env()
  t2j = env.t2j_iso
  j2t = env.j2t_iso

  attns = jax_sp._attentions(axis, sp_size=sp_size, qk_fp8=qk_fp8)

  def _video_self_ta(q_t, k_t, v_t):
    return j2t(attns["video_self"](t2j(q_t), t2j(k_t), t2j(v_t)))

  def _v2a_ta(q_t, k_t, v_t):
    return j2t(attns["v2a"](t2j(q_t), t2j(k_t), t2j(v_t)))

  def _fp8_lin_ta(x_t, qw_t, sc_t, b_t):
    xj, qwj, scj, bj = t2j(x_t), t2j(qw_t), t2j(sc_t), t2j(b_t)
    f8_max = jnp.float32(448.0)
    x32 = xj.astype(jnp.float32)
    xs = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True), jnp.float32(1e-12)) / f8_max
    qx = jnp.clip(x32 / xs, -f8_max, f8_max).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        qx, qwj, (((xj.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    y = (acc * xs * scj).astype(xj.dtype) + bj.astype(xj.dtype)
    return j2t(y)

  hooks = LTX2TpuHooks(
      video_self_attn=_video_self_ta,
      v2a_attn=_v2a_ta,
      fp8_linear=_fp8_lin_ta if fp8 else None,
      sp_size=sp_size,
      qk_fp8=qk_fp8,
  )

  def _set_module_params(root: torch.nn.Module, prefix: str, state: dict[str, jax.Array]) -> None:
    plen = len(prefix)
    for k, v in state.items():
      rel = k[plen:]
      parts = rel.split(".")
      sub: Any = root
      for p in parts[:-1]:
        sub = sub[int(p)] if isinstance(sub, torch.nn.ModuleList) else getattr(sub, p)
      sub._parameters[parts[-1]] = j2t(v)

  def body(top_p, blk_p_list, x, a, ctx, actx, tv, ta, tt, token_t, rope, stg, iso):
    with env:
      _set_module_params(model, "", top_p)
      xt, at, ctxt, actxt = j2t(x), j2t(a), j2t(ctx), j2t(actx)
      tvt, tat, ttt = j2t(tv), j2t(ta), j2t(tt)
      tkt = j2t(token_t)
      rv_c, rv_s = j2t(rope["video"][0]), j2t(rope["video"][1])
      ra_c, ra_s = j2t(rope["audio"][0]), j2t(rope["audio"][1])
      rcv_c, rcv_s = j2t(rope["ca_video"][0]), j2t(rope["ca_video"][1])
      rca_c, rca_s = j2t(rope["ca_audio"][0]), j2t(rope["ca_audio"][1])
      iso_keep = (1.0 - j2t(iso).to(xt.dtype)).reshape(-1, 1, 1)
      stg_t = j2t(stg)

      cond = model.conditioning(tvt, tat, ttt)
      xt = model.proj_in(xt)
      at = model.audio_proj_in(at)

      for i in range(num_layers):
        bp = {
            k: jax.lax.all_gather(v, axis, axis=0, tiled=True) if blk_specs[i][k] != P() else v
            for k, v in blk_p_list[i].items()
        }
        _set_module_params(model.blocks[i], f"blocks.{i}.", bp)
        stg_i = stg_t if (i in stg_set) else None
        xt, at = model.blocks[i](
            xt, at, ctxt, actxt,
            cond["temb"], cond["temb_a"], cond["tp"], cond["tpa"],
            cond["cav"], cond["cavg"], cond["caa"], cond["caag"],
            tkt,
            rv_c, rv_s, ra_c, ra_s,
            rcv_c, rcv_s, rca_c, rca_s,
            stg_i, iso_keep,
            hooks=hooks,
        )

      out_v, out_a = model.output_heads(xt, at, cond["emb"], cond["emb_a"], tkt)
      return t2j(out_v), t2j(out_a)

  rows = P(c_ax, s_ax, None)
  aud_spec = P(c_ax, None, None)
  vid_rope = (P(s_ax, None, None),) * 2
  rope_specs = {"video": vid_rope, "ca_video": vid_rope, "audio": (P(),) * 2, "ca_audio": (P(),) * 2}

  mapped = shard_map(
      body,
      mesh=mesh,
      in_specs=(
          top_specs, blk_specs,
          rows, aud_spec, aud_spec, aud_spec,
          P(c_ax, None), P(c_ax), P(c_ax), P(s_ax),
          rope_specs, P(c_ax), P(c_ax),
      ),
      out_specs=(rows, aud_spec),
      check_rep=False,
  )

  def forward_fn(params, x, a, ctx, actx, tv, ta, tt, token_t, rope, stg, iso):
    top_p = {k: params[k] for k in top_keys}
    blk_p_list = [{k: params[k] for k in ks} for ks in blk_keys]
    if stg is None:
      stg = jnp.zeros((x.shape[0],), dtype=jnp.bool_)
    if iso is None:
      iso = jnp.zeros((x.shape[0],), dtype=jnp.bool_)
    return mapped(top_p, blk_p_list, x, a, ctx, actx, tv, ta, tt, token_t, rope, stg, iso)

  return forward_fn
