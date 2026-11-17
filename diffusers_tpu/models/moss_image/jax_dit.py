"""MOSS-Image's DiT and sampling step in native JAX.

The same computation as ``torchax_step.StepCore`` -- [positive, negative] as
one padded batch, per-sample kv lengths, layer skipping in the negative
branch, CFG and the Euler update in float32 -- written as plain functions
over a flat weight dict, so there is no PyTorch dispatch and no traced
``nn.Module``:

* the 42 blocks' weights are stacked and run as ``lax.scan``, cut at the
  skipped layers (layer 22 by default): a per-sample select inside the scan
  body would add a pass over the activations to every layer;
* ``ffn_up_gate`` is split into ``ffn_up`` and ``ffn_gate`` at load (see
  ``torchax_dit.split_up_gate`` for why);
* on a ``('cfg', 'tensor')`` mesh the batch is split over ``cfg``, heads and
  FFN rows over ``tensor``, the residual stream is whole per chip, and
  attention is the splash kernel under ``shard_map``.

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.moss_image.jax_dit
checks a small random configuration against the upstream module.
"""
import functools
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Imported here, not inside the traced function: importing the Pallas kernel
# registers config options, which changes the tracing context and makes the
# second call of a jitted step retrace and recompile.
from diffusers_tpu.kernels import block_sizes, splash

_BLOCK_KEYS = ("norm1", "norm2", "q_norm", "k_norm", "q", "k", "v", "o",
               "ffn_up", "ffn_gate", "ffn_down")


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def _from_torch_state(state, config, dtype, scan=False):
  """Upstream state dict (numpy) -> our flat dict; linears as [in, out].

  Block weights are ``blocks.<i>.<name>`` per layer, or stacked as
  ``blocks.<name>`` [L, ...] with ``scan``.
  """
  depth = config["num_layers"]
  params = {}

  def lin(name, keep=None):
    w = state[name + ".weight"]
    params[name + ".w"] = jnp.asarray(w.T, keep or dtype)
    if name + ".bias" in state:
      params[name + ".b"] = jnp.asarray(state[name + ".bias"], keep or dtype)

  for name in ("video_embedder", "time_embedding.0", "time_embedding.2",
               "shared_block_modulation", "video_head_modulation",
               "final_linear_video", "text_embedder.proj"):
    lin(name)
  params["final_norm_video"] = jnp.asarray(state["final_norm_video.weight"], dtype)
  for t in range(2):
    pre = f"text_embedder.blocks.{t}"
    for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"):
      lin(f"{pre}.{name}")
    for name in ("norm1", "norm2", "q_norm", "k_norm"):
      params[f"{pre}.{name}"] = jnp.asarray(state[f"{pre}.{name}.weight"], dtype)

  stacks = {k: [] for k in _BLOCK_KEYS}
  for i in range(depth):
    pre = f"layers.{i}"
    for name in ("norm1", "norm2", "q_norm", "k_norm"):
      stacks[name].append(state[f"{pre}.{name}.weight"])
    for name in ("q", "k", "v", "o", "ffn_down"):
      stacks[name].append(state[f"{pre}.proj.{name}.weight"].T)
    up, gate = np.split(state[f"{pre}.proj.ffn_up_gate.weight"], 2, axis=0)
    stacks["ffn_up"].append(up.T)
    stacks["ffn_gate"].append(gate.T)
  for name, values in stacks.items():
    if scan:
      params["blocks." + name] = jnp.asarray(np.stack(values), dtype)
    else:
      for i, value in enumerate(values):
        params[f"blocks.{i}.{name}"] = jnp.asarray(value, dtype)
  return params


def load(transformer_dir, dtype=jnp.bfloat16, scan=False):
  """(params, config) from the snapshot's ``transformer/``."""
  from safetensors.numpy import load_file

  with open(os.path.join(transformer_dir, "config.json")) as fh:
    config = json.load(fh)
  state = {}
  for path in sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors"))):
    for k, v in load_file(path).items():
      state[k] = v.astype(np.float32) if v.dtype != np.float32 else v
  return _from_torch_state(state, config, dtype, scan), config


def param_specs(params):
  """PartitionSpecs: block projections over ``tensor``, the rest replicated."""
  out = {}
  for k, v in params.items():
    lead = (None,) * (v.ndim - 2)  # a stacked [L, in, out] has one more axis
    name = k.rsplit(".", 1)[-1]
    if k.startswith("blocks.") and name in ("q", "k", "v", "ffn_up", "ffn_gate"):
      out[k] = P(*lead, None, "tensor")  # output columns
    elif k.startswith("blocks.") and name in ("o", "ffn_down"):
      out[k] = P(*lead, "tensor", None)  # input rows
    else:
      out[k] = P()
  return out


# ---------------------------------------------------------------------------
# layers (upstream names and arithmetic)
# ---------------------------------------------------------------------------


def _rms(x, w):
  xf = x.astype(jnp.float32)
  y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + 1e-6)
  return y.astype(x.dtype) * w


def _situ(hidden, gate):
  return (4.0 * jnp.tanh(gate / 4.0) * jax.nn.sigmoid(gate)) * (25.0 * jnp.tanh(hidden / 25.0))


def _rotate(x, rope):
  """x [B, S, H, D], rope [B, S, D] (sin | cos halves)."""
  sin, cos = jnp.split(rope, 2, -1)
  sin = jnp.concatenate((sin, sin), -1)[:, :, None]
  cos = jnp.concatenate((cos, cos), -1)[:, :, None]
  a, b = jnp.split(x, 2, -1)
  return x * cos + jnp.concatenate((-b, a), -1) * sin


def _linear(x, params, name):
  y = x @ params[name + ".w"].astype(x.dtype)
  if name + ".b" in params:
    y = y + params[name + ".b"].astype(x.dtype)
  return y


def _text_block(params, pre, x, rope, key_mask, heads):
  q, k, v = jnp.split(_linear(_rms(x, params[pre + ".norm1"]), params, pre + ".qkv"), 3, -1)
  shape = (*q.shape[:2], heads, q.shape[-1] // heads)
  q = _rotate(_rms(q.reshape(shape), params[pre + ".q_norm"]), rope)
  k = _rotate(_rms(k.reshape(shape), params[pre + ".k_norm"]), rope)
  v = v.reshape(shape)
  out = jax.nn.dot_product_attention(q.astype(v.dtype), k.astype(v.dtype), v,
                                     mask=key_mask[:, None, None, :])
  x = x + _linear(out.reshape(x.shape), params, pre + ".proj")
  y = _rms(x, params[pre + ".norm2"])
  return x + _linear(_situ(_linear(y, params, pre + ".ffn_up"),
                           _linear(y, params, pre + ".ffn_gate")), params, pre + ".ffn_down")


def _block(w, x, rope, mod, attend, heads):
  """``_Block.forward`` with ``w`` one layer's slice of the stacked weights."""
  shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (mod[:, i][:, None] for i in range(6))
  dtype = w["q"].dtype
  y = (_rms(x, w["norm1"]) * (1 + scale_a) + shift_a).astype(dtype)
  shape = (*y.shape[:2], heads, -1)
  q = _rms((y @ w["q"]).reshape(shape), w["q_norm"])
  k = _rms((y @ w["k"]).reshape(shape), w["k_norm"])
  v = (y @ w["v"]).reshape(shape)
  out = attend(_rotate(q, rope).astype(v.dtype), _rotate(k, rope).astype(v.dtype), v)
  x = x + (out.reshape(y.shape[:2] + (-1,)) @ w["o"]) * gate_a
  y = (_rms(x, w["norm2"]) * (1 + scale_f) + shift_f).astype(dtype)
  return x + (_situ(y @ w["ffn_up"], y @ w["ffn_gate"]) @ w["ffn_down"]) * gate_f


def make_attend(mesh, kv_len):
  """Joint attention: [B, S, H, D] -> [B, S, H, D], valid keys a prefix."""
  if mesh is None or jax.default_backend() == "cpu":
    def attend(q, k, v):
      mask = jnp.arange(k.shape[1])[None, :] < kv_len[:, None]
      return jax.nn.dot_product_attention(q, k, v, mask=mask[:, None, None, :])
    return attend

  def attend(q, k, v):
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))  # [B, H, S, D]
    seq = k.shape[2]
    blocks = block_sizes.resolve_block_sizes(seq, seq, q.shape[-1], {}, "residual",
                                             block_sizes.detect_spec())
    valid = (jnp.arange(seq)[None, :] < kv_len[:, None])[:, None, :, None]
    k = jnp.where(valid, k, 0)
    k = jnp.where(valid, k - k.sum(2, keepdims=True) /
                  kv_len[:, None, None, None].astype(k.dtype), 0)

    def local(q, k, v, n):
      outs = [splash.attention(q[i:i + 1], k[i:i + 1], v[i:i + 1], scale=None,
                               kv_len=n[i], kv_mask_all=seq > 8192, **blocks)
              for i in range(q.shape[0])]
      return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)

    spec = P("cfg", "tensor", None, None)
    out = shard_map(local, mesh=mesh, in_specs=(spec, spec, spec, P("cfg")),
                    out_specs=spec, check_rep=False)(q, k, v, kv_len.astype(jnp.float32))
    return out.transpose(0, 2, 1, 3)

  return attend


def dit(params, latents, sinusoid, context, rope, kv_len, skip, *, config,
        skip_layers=(), mesh=None):
  """Velocity for a padded batch. Shapes as in ``torchax_step.StepCore.dit``."""
  heads = config["num_heads"]
  depth = config["num_layers"]
  batch, _, height, width = latents.shape
  n_img = height * width
  scan = "blocks.q" in params
  dtype = params["blocks.q" if scan else "blocks.0.q"].dtype
  stream = (lambda t: t) if mesh is None else (
      lambda t: jax.lax.with_sharding_constraint(t, NamedSharding(mesh, P("cfg"))))

  time = _linear(sinusoid.astype(params["time_embedding.0.w"].dtype), params, "time_embedding.0")
  time = _linear(jax.nn.silu(time), params, "time_embedding.2").astype(jnp.float32)
  mod = _linear(time.astype(params["shared_block_modulation.w"].dtype), params,
                "shared_block_modulation").reshape(time.shape[0], 6, -1)
  head = _linear(time.astype(params["video_head_modulation.w"].dtype), params,
                 "video_head_modulation").reshape(time.shape[0], 2, -1)
  video = latents.reshape(batch, latents.shape[1], n_img).transpose(0, 2, 1)
  video = _linear(video.astype(dtype), params, "video_embedder")
  rope = rope.astype(video.dtype)
  key_mask = jnp.arange(context.shape[1])[None, :] < (kv_len - n_img)[:, None]
  text = _linear(context.astype(dtype), params, "text_embedder.proj")
  for t in range(2):
    text = _text_block(params, f"text_embedder.blocks.{t}", text, rope[:, n_img:],
                       key_mask, heads)
  x = stream(jnp.concatenate((video, text), 1))
  attend = make_attend(mesh, kv_len)

  def layer_weights(i):
    if scan:
      return {k: params["blocks." + k][i] for k in _BLOCK_KEYS}
    return {k: params[f"blocks.{i}.{k}"] for k in _BLOCK_KEYS}

  def run(x, lo, hi):
    if hi <= lo:
      return x
    if not scan:  # unrolled, like the traced module (default; faster to run)
      for i in range(lo, hi):
        x = stream(_block(layer_weights(i), x, rope, mod, attend, heads))
      return x
    def body(x, w):
      return stream(_block(w, x, rope, mod, attend, heads)), None
    return jax.lax.scan(body, x, {k: params["blocks." + k][lo:hi] for k in _BLOCK_KEYS})[0]

  start = 0
  for layer in sorted(skip_layers):
    x = run(x, start, layer)
    out = stream(_block(layer_weights(layer), x, rope, mod, attend, heads))
    x = jnp.where(skip[:, None, None], x, out)
    start = layer + 1
  x = run(x, start, depth)

  x = _rms(x[:, :n_img], params["final_norm_video"])
  x = x * (1 + head[:, 1, None]) + head[:, 0, None]
  x = _linear(x.astype(dtype), params, "final_linear_video")
  return x.transpose(0, 2, 1).reshape(batch, -1, height, width)


def step(params, latents, sinusoid, context, rope, kv_len, skip, dt, guidance, *,
         config, skip_layers=(), mesh=None):
  """One CFG + Euler step; ``latents`` [1, C, H, W] float32."""
  batch = context.shape[0]
  x = jnp.broadcast_to(latents, (batch,) + latents.shape[1:])
  if mesh is not None and batch > 1:
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("cfg")))
  pred = dit(params, x, sinusoid, context, rope, kv_len, skip, config=config,
             skip_layers=skip_layers, mesh=mesh).astype(jnp.float32)
  if batch > 1:
    pred = pred[1:2] + guidance * (pred[0:1] - pred[1:2])
  return latents + dt * pred


def make_step(transformer_dir, mesh, dtype=jnp.bfloat16, scan=False):
  """(params on ``mesh``, config, jitted ``step``)."""
  params, config = load(transformer_dir, dtype, scan)
  specs = param_specs(params)
  params = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in params.items()}
  fn = jax.jit(functools.partial(step, config=config, mesh=mesh),
               static_argnames=("skip_layers",))
  return params, config, fn


def _check():
  """Small random configuration against the upstream module, CPU float32."""
  import copy

  import torch

  from diffusers_tpu.models.moss_image import remote, torchax_dit as d

  mod = remote.load_module()
  torch.manual_seed(0)
  cfg = dict(dim=384, ffn_dim=1024, num_heads=3, num_layers=5, text_dim=64,
             latent_channels=16, freq_dim=256)
  m = mod.MOSSImageTransformer(**cfg).eval()
  for p in m.parameters():
    p.data.normal_(0, 0.05)
  H, W = 6, 5
  lat = torch.randn(1, 16, H, W)
  pos, neg = torch.randn(1, 9, 64), torch.randn(1, 4, 64)
  t, g, dt = torch.tensor([300.0]), 3.0, -0.05
  with torch.no_grad():
    p_ = m(lat, t, pos).sample
    n_ = m(lat, t, neg, skip_layers=[2]).sample
    ref = lat + dt * (n_ + g * (p_ - n_))
  state = {k: v.numpy() for k, v in m.state_dict().items()}
  params = _from_torch_state(state, dict(cfg), jnp.float32)
  Lp = 16
  ctx = np.concatenate([np.pad(pos.numpy(), ((0, 0), (0, Lp - 9), (0, 0))),
                        np.pad(neg.numpy(), ((0, 0), (0, Lp - 4), (0, 0)))])
  rope = np.concatenate([d.rope_table(m, H, W, 9, Lp).numpy(), d.rope_table(m, H, W, 4, Lp).numpy()])
  out = step(params, jnp.asarray(lat.numpy()), jnp.asarray(d.time_sinusoid(m, t).numpy()),
             jnp.asarray(ctx), jnp.asarray(rope), jnp.array([H * W + 9, H * W + 4]),
             jnp.array([False, True]), jnp.float32(dt), jnp.float32(g), config=dict(cfg),
             skip_layers=(2,))
  print("max |step - upstream|", float(np.abs(np.asarray(out) - ref.numpy()).max()),
        "scale", float(ref.abs().max()))


if __name__ == "__main__":
  jax.config.update("jax_default_matmul_precision", "highest")
  _check()
