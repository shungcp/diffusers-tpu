"""LTX-2.3's text connectors (diffusers ``LTX2TextConnectors``) in JAX.

Gemma-3's 49 hidden states (embeddings + 48 layers) of the left-padded
prompt are RMS-normalized per token and layer, flattened (channel-major),
zeroed on padding and projected to the video (4096) and audio (2048)
widths. Each stream then runs an 8-block 1-D transformer (gated attention,
split RoPE over positions / 4096, GELU-tanh FFN) after moving the valid
tokens to the front and filling the rest with 128 learned registers
(tiled), so the DiT sees no padding: its text mask is all ones.
"""
import numpy as np

import jax
import jax.numpy as jnp

from diffusers_tpu.models.ltx2 import jax_dit as jd


def load(folder, dtype=jnp.bfloat16):
  import json
  import os

  with open(os.path.join(folder, "config.json")) as fh:
    config = json.load(fh)
  params = {"video": {}, "audio": {}}
  per_layer = {"video": {}, "audio": {}}
  for name, value in jd._iter_state(folder):
    v = np.asarray(value)
    if v.ndim == 2 and name.endswith(".weight"):
      v = v.T
    v = v.astype(jnp.dtype(dtype))
    stream = "video" if name.startswith("video_") else "audio"
    rest = name.split("_", 1)[1]  # text_proj_in.* / connector.*
    if rest.startswith("connector.transformer_blocks."):
      _, _, idx, r = rest.split(".", 3)
      per_layer[stream].setdefault(r, {})[int(idx)] = v
    else:
      params[stream][rest.replace("connector.", "")] = v
  for s in params:
    n = len(next(iter(per_layer[s].values())))
    params[s]["blocks"] = [{k: d[i] for k, d in per_layer[s].items()} for i in range(n)]
  return params, config


def place(params, mesh, axis):
  """Split over ``axis``: the 188160-wide input projections by input rows,
  the blocks Megatron-style (their sums all-reduced by XLA); ~4 GB otherwise."""
  from jax.sharding import NamedSharding
  from jax.sharding import PartitionSpec as P

  def specs(stream):
    out = {k: P() for k in stream if k != "blocks"}
    out["text_proj_in.weight"] = P(axis, None)
    out["blocks"] = jd.param_specs({"blocks": stream["blocks"]}, axis)["blocks"]
    return out

  sp = {s: specs(params[s]) for s in params}
  return jax.tree.map(lambda v, s: jax.device_put(v, NamedSharding(mesh, s)), params, sp)


def rope_1d(length, dim, heads, base_seq_len=4096, theta=10000.0):
  coords = (np.arange(length, dtype=np.float32) / np.float32(base_seq_len))
  # rope_table takes [start, end) bounds; equal bounds give the point itself
  return jd.rope_table(np.stack([coords, coords], -1)[None], dim, heads, (1.0,), theta)


def _connector(p, h, heads, rope, attention):
  b, l, d = h.shape

  for layer in p["blocks"]:
    h = h + jd._attn(layer, "attn1", jd._rms(h), None, heads, attention, rope)
    h = h + jd._ff(layer, "ff", jd._rms(h))
  return jd._rms(h)


def forward(params, config, hidden, mask, attention=jd.dense_attention):
  """hidden [B, L, 3840, 49] (Gemma hidden states stacked last), mask [B, L]
  -> (video [B, L, 4096], audio [B, L, 2048]) prompt embeddings."""
  b, l, c, n = hidden.shape
  dtype = params["video"]["text_proj_in.weight"].dtype
  var = jnp.mean(jnp.square(hidden.astype(jnp.float32)), axis=2, keepdims=True)
  hf = (hidden.astype(jnp.float32) * jax.lax.rsqrt(var + 1e-6)).astype(hidden.dtype).reshape(b, l, c * n)
  keep = mask.astype(bool)
  hf = jnp.where(keep[..., None], hf, jnp.zeros_like(hf))
  # front-align the valid tokens (stable), registers after them
  order = jnp.argsort(1 - keep.astype(jnp.int32), axis=1, stable=True)
  nvalid = keep.sum(1, keepdims=True)
  front = jnp.arange(l)[None] < nvalid
  out = []
  for s in ("video", "audio"):
    p = params[s]
    width = config[f"{s}_hidden_dim"]
    heads = config[f"{s}_connector_num_attention_heads"]
    x = hf * np.float32(np.sqrt(width / config["caption_channels"])).astype(hf.dtype)
    x = jd._linear(x, p, "text_proj_in")
    x = jnp.take_along_axis(x, order[..., None], axis=1)
    reg = p["learnable_registers"]
    reg = jnp.tile(reg, (l // reg.shape[0], 1)).astype(x.dtype)
    x = jnp.where(front[..., None], x, reg[None])
    rope = tuple(jnp.asarray(t) for t in rope_1d(l, width, heads, config["connector_rope_base_seq_len"],
                                                   config["rope_theta"]))
    out.append(_connector(p, x, heads, rope, attention))
  return out[0], out[1]
