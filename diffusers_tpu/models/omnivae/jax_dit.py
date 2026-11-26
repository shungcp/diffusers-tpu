"""OmniVAE's T2AV DiT (``BridgedZImageJointModel``) in JAX.

Two stock Z-Image towers (video: 48-channel Wan2.2 latents; audio: 128-channel
DAC latents as a T x 1 x 1 grid) that never share a sequence: each runs
single-stream self-attention over ``[x tokens, caption tokens]``. After each of
the 18 main layers a bridge exchanges information between the two x streams
(audio -> video and video -> audio cross-attention, modulated by the *other*
modality's time embedding, gated per sample for dual CFG).

Matches the official runtime patches: real-valued RoPE on interleaved channel
pairs (theta 256, axes t/h/w = 16/56/56 dims), caption positions (1 + i, 0, 0)
and x positions (cap_len + 1 + f, h, w); the bridge rotates only the time
pairs, with video frames mapped onto the audio timeline
``round(f * (T_a - 1) / (F - 1))``. Padding is handled with key masks, so the
official multiple-of-64 padding needs no counterpart.

Weights: ``transformer_{video,audio}/`` (diffusers safetensors) and
``bridges/bridges.safetensors``; linears stored as ``[in, out]``.
"""
import functools
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

ADALN = 256


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _host(v, dtype):
  v = np.asarray(v, np.float32)
  return v.astype(jnp.dtype(dtype)) if v.dtype != jnp.dtype(dtype) else v


def _prep(state, dtype):
  out = {}
  for name, v in state.items():
    if name.endswith("pad_token"):
      continue
    v = np.asarray(v, np.float32)
    if v.ndim == 2:   # nn.Linear [out, in] -> [in, out]
      v = v.T
    keep32 = name.startswith("t_embedder.")
    out[name] = v if keep32 else v.astype(jnp.dtype(dtype))
  return out


def load(ckpt_dir, dtype=jnp.bfloat16):
  """(video params, audio params, bridge params, video config, audio config),
  host numpy arrays (``t_embedder`` kept float32)."""
  from safetensors.numpy import load_file

  towers = []
  for name in ("transformer_video", "transformer_audio"):
    d = os.path.join(ckpt_dir, name)
    with open(os.path.join(d, "config.json")) as fh:
      cfg = json.load(fh)
    state = {}
    for p in glob.glob(os.path.join(d, "*.safetensors")):
      state.update(load_file(p))
    towers.append((_prep(state, dtype), cfg))
  bridges = _prep(load_file(os.path.join(ckpt_dir, "bridges", "bridges.safetensors")), dtype)
  (vp, vc), (ap, ac) = towers
  return vp, ap, bridges, vc, ac


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _linear(x, params, name):
  w = params[name + ".weight"]
  y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=w.dtype)
  b = params.get(name + ".bias")
  return y if b is None else y + b.astype(w.dtype)


def _rms(x, w, eps=1e-5):
  """diffusers ``RMSNorm``: float32 statistic, back to the weight's dtype, times weight."""
  xf = x.astype(jnp.float32)
  y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + eps)
  return y.astype(w.dtype) * w


def _layer_norm(x, eps=1e-6):
  xf = x.astype(jnp.float32)
  m = xf.mean(-1, keepdims=True)
  y = (xf - m) * jax.lax.rsqrt(jnp.square(xf - m).mean(-1, keepdims=True) + eps)
  return y.astype(x.dtype)


def _rope(x, cos, sin, half=False):
  """Rotation of x [B, S, H, D] (float32); cos/sin [B, S, D/2]. Interleaved
  pairs (the checkpoint layout), or -- with ``half`` -- pairs split into the
  two halves of each head (``permute_rope`` reorders q / k that way)."""
  if half:
    xf = x.astype(jnp.float32)
    c = jnp.concatenate([cos, cos], -1)[:, :, None]
    s = jnp.concatenate([sin, sin], -1)[:, :, None]
    x1, x2 = jnp.split(xf, 2, -1)
    return (xf * c + jnp.concatenate([-x2, x1], -1) * s).astype(x.dtype)
  xf = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
  re, im = xf[..., 0], xf[..., 1]
  c, s = cos[:, :, None], sin[:, :, None]
  return jnp.stack((re * c - im * s, re * s + im * c), -1).reshape(x.shape).astype(x.dtype)


def dense_attention(q, k, v, mask=None):
  """[B, S, H, D] attention; ``mask`` [B, Sk] bool (keys). float32 scores."""
  s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
  if mask is not None:
    s = jnp.where(mask[:, None, None, :], s, -jnp.inf)
  p = jax.nn.softmax(s, -1).astype(v.dtype)
  return jnp.einsum("bhqk,bkhd->bqhd", p, v, preferred_element_type=jnp.float32).astype(v.dtype)


def permute_rope(params, head_dim=128):
  """q / k projections (and their per-head norms) with each head's interleaved
  pairs regrouped into halves (evens, then odds). q and k move together, so
  every score is unchanged; RoPE then runs as rotate-half (``half=True``)."""
  perm = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
  out = dict(params)
  for name, v in params.items():
    if name.endswith(("to_q.weight", "to_k.weight")):
      heads = v.shape[1] // head_dim
      idx = (np.arange(heads)[:, None] * head_dim + perm[None]).ravel()
      out[name] = np.asarray(v)[:, idx]
    elif name.endswith(("norm_q.weight", "norm_k.weight")) and v.shape[0] == head_dim:
      out[name] = np.asarray(v)[perm]
  return out


def time_sinusoid(t, dim=256, max_period=10000.0):
  """``TimestepEmbedder.timestep_embedding``: cos first, then sin (float32)."""
  half = dim // 2
  freqs = jnp.exp(-np.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
  a = jnp.asarray(t, jnp.float32)[:, None] * freqs[None]
  return jnp.concatenate([jnp.cos(a), jnp.sin(a)], -1)


def rope_tables(axes_dims, axes_lens, theta):
  """Per-axis (cos, sin) tables [len, dim/2] as the patched ``precompute_freqs_cis``."""
  out = []
  for d, n in zip(axes_dims, axes_lens):
    f = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    ph = np.outer(np.arange(n, dtype=np.float64), f).astype(np.float32)
    out.append((np.cos(ph), np.sin(ph)))
  return out


def rope_lookup(tables, pos):
  """pos [B, S, 3] int -> (cos, sin) [B, S, head_dim / 2]."""
  cs = [jnp.asarray(c)[pos[..., i]] for i, (c, _) in enumerate(tables)]
  ss = [jnp.asarray(s)[pos[..., i]] for i, (_, s) in enumerate(tables)]
  return jnp.concatenate(cs, -1), jnp.concatenate(ss, -1)


# ---------------------------------------------------------------------------
# tower
# ---------------------------------------------------------------------------


def _block(params, name, x, mask, cos, sin, heads, attention, adaln=None, eps=1e-5, half=False):
  """``ZImageTransformerBlock`` (modulated when ``adaln`` is given)."""
  pre = name + "."
  if adaln is not None:
    mod = _linear(adaln, params, pre + "adaLN_modulation.0")[:, None]       # [B, 1, 4D]
    scale_a, gate_a, scale_f, gate_f = jnp.split(mod, 4, -1)
    scale_a, scale_f = 1.0 + scale_a, 1.0 + scale_f
    gate_a, gate_f = jnp.tanh(gate_a), jnp.tanh(gate_f)
  b, s, d = x.shape
  y = _rms(x, params[pre + "attention_norm1.weight"], eps)
  if adaln is not None:
    y = y * scale_a
  shape = (b, s, heads, d // heads)
  q = _rms(_linear(y, params, pre + "attention.to_q").reshape(shape), params[pre + "attention.norm_q.weight"])
  k = _rms(_linear(y, params, pre + "attention.to_k").reshape(shape), params[pre + "attention.norm_k.weight"])
  v = _linear(y, params, pre + "attention.to_v").reshape(shape)
  q, k = _rope(q, cos, sin, half), _rope(k, cos, sin, half)
  out = _linear(attention(q, k, v, mask).reshape(b, s, d), params, pre + "attention.to_out.0")
  out = _rms(out, params[pre + "attention_norm2.weight"], eps)
  x = x + (gate_a * out if adaln is not None else out)
  y = _rms(x, params[pre + "ffn_norm1.weight"], eps)
  if adaln is not None:
    y = y * scale_f
  h = jax.nn.silu(_linear(y, params, pre + "feed_forward.w1")) * _linear(y, params, pre + "feed_forward.w3")
  out = _rms(_linear(h, params, pre + "feed_forward.w2"), params[pre + "ffn_norm2.weight"], eps)
  return x + (gate_f * out if adaln is not None else out)


def embed_time(params, t):
  """t [B] (model timestep 1 - sigma) -> adaLN input [B, 256] in the weights' dtype."""
  e = time_sinusoid(t * 1000.0)
  e = _linear(e, params, "t_embedder.mlp.0")
  e = _linear(jax.nn.silu(e), params, "t_embedder.mlp.2")
  return e.astype(params["layers.0.attention.to_q.weight"].dtype)


def prepare(params, config, x_tokens, cap, cap_mask, t, x_pos, cap_pos, attention, half=False, parts=1):
  """Embedding and refiners: (unified [B, N + Lc, D], mask, cos, sin, adaln).

  x_tokens [B, N, C]; cap [B, Lc, 1024] (zero past ``cap_mask``);
  x_pos / cap_pos [B, *, 3] int position ids."""
  heads, eps = config["n_heads"], config["norm_eps"]
  tables = rope_tables(config["axes_dims"], config["axes_lens"], config["rope_theta"])
  adaln = embed_time(params, t)
  x = _linear(x_tokens, params, "all_x_embedder.1-1")
  c = _linear(_rms(cap, params["cap_embedder.0.weight"], eps), params, "cap_embedder.1")
  ccos, csin = rope_lookup(tables, cap_pos)
  for i in range(config["n_refiner_layers"]):
    c = _block(params, f"context_refiner.{i}", c, cap_mask, ccos, csin, heads, attention, eps=eps, half=half)
  xcos, xsin = rope_lookup(tables, x_pos)
  for i in range(config["n_refiner_layers"]):
    x = _block(params, f"noise_refiner.{i}", x, None, xcos, xsin, heads, attention, adaln, eps, half)
  u = lambda a, b_: unify(a, b_, parts)
  return (u(x, c), u(jnp.ones(x.shape[:2], bool), cap_mask), u(xcos, ccos), u(xsin, csin), adaln)


def unify(x, c, parts=1):
  """The single-stream sequence: ``[x, caption]``, or -- split over ``parts``
  chips -- ``[x_0, caption, x_1, caption, ...]`` (each chip: its x rows and a
  copy of the caption), so x row boundaries match the chip boundaries."""
  if parts == 1:
    return jnp.concatenate([x, c], 1)
  b, n = x.shape[:2]
  xs = x.reshape(b, parts, n // parts, *x.shape[2:])
  cs = jnp.broadcast_to(c[:, None], (b, parts) + c.shape[1:])
  return jnp.concatenate([xs, cs], 2).reshape(b, -1, *x.shape[2:])


def x_rows(u, n_x, parts=1):
  if parts == 1:
    return u[:, :n_x]
  b = u.shape[0]
  return u.reshape(b, parts, -1, *u.shape[2:])[:, :, :n_x // parts].reshape(b, n_x, *u.shape[2:])


def add_x(u, d, parts=1):
  if parts == 1:
    return u.at[:, :d.shape[1]].add(d)
  b, n = d.shape[:2]
  ur = u.reshape(b, parts, -1, *u.shape[2:])
  return ur.at[:, :, :n // parts].add(d.reshape(b, parts, n // parts, *d.shape[2:])).reshape(u.shape)


def final(params, unified, adaln, n_x, parts=1):
  scale = 1.0 + _linear(jax.nn.silu(adaln), params, "all_final_layer.1-1.adaLN_modulation.1")
  y = _layer_norm(x_rows(unified, n_x, parts)) * scale[:, None]
  return _linear(y, params, "all_final_layer.1-1.linear")


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------


def bridge_rope(audio_config, t_idx, head_dim):
  """Time-only RoPE (cos, sin) [S, head_dim / 2]: the first t-axis pairs rotate
  by ``t_idx`` (audio timeline), the rest are the identity."""
  d, n = audio_config["axes_dims"][0], audio_config["axes_lens"][0]
  (c, s), = rope_tables([d], [n], audio_config["rope_theta"])
  t_idx = np.clip(np.asarray(t_idx), 0, n - 1)
  pad = head_dim // 2 - d // 2
  cos = np.concatenate([c[t_idx], np.ones((len(t_idx), pad), np.float32)], -1)
  sin = np.concatenate([s[t_idx], np.zeros((len(t_idx), pad), np.float32)], -1)
  return cos, sin


def video_time_indices(frames, h, w, audio_len):
  t = np.arange(frames * h * w) // (h * w)
  if frames <= 1:
    return np.zeros_like(t)
  return np.clip(np.round(t.astype(np.float32) * float(audio_len - 1) / float(frames - 1)).astype(np.int64),
                 0, audio_len - 1)


def _cross(params, pre, xq, xkv, qrope, krope, heads, attention, eps, half=False):
  b, sq, d = xq.shape
  hd = d // heads
  q = _rms(_linear(xq, params, pre + "to_q").reshape(b, sq, heads, hd), params[pre + "norm_q.weight"], eps)
  k = _rms(_linear(xkv, params, pre + "to_k").reshape(b, -1, heads, hd), params[pre + "norm_k.weight"], eps)
  v = _linear(xkv, params, pre + "to_v").reshape(b, -1, heads, hd)
  q = _rope(q, jnp.asarray(qrope[0])[None], jnp.asarray(qrope[1])[None], half)
  k = _rope(k, jnp.asarray(krope[0])[None], jnp.asarray(krope[1])[None], half)
  return _linear(attention(q, k, v, None).reshape(b, sq, d), params, pre + "to_out")


def bridge(params, i, vx, ax, vrope, arope, t_v, t_a, heads, attention, gate=None, eps=1e-5, half=False):
  """``BridgeBlock``: (delta_video, delta_audio). ``gate`` [B] scales both
  deltas per sample (dual CFG's bridge-off slot)."""
  pre = f"{i}."

  def adaln(name, t):
    mod = _linear(jax.nn.silu(t), params, pre + name + ".proj")
    s, g = jnp.split(mod, 2, -1)
    return 1.0 + s[:, None], jnp.tanh(g)[:, None]

  s, g = adaln("a2v_adaln", t_a)
  out = _cross(params, pre + "a2v_attn.", _rms(vx, params[pre + "a2v_norm_q.weight"], eps) * s,
               _rms(ax, params[pre + "a2v_norm_kv.weight"], eps), vrope, arope, heads, attention, eps, half)
  dv = g * _rms(out, params[pre + "a2v_norm_out.weight"], eps)
  s, g = adaln("v2a_adaln", t_v)
  # Audio queries over the video keys: a sequence-parallel run keeps the keys
  # where they are and merges partial softmaxes (``attention.kv_rows``).
  out = _cross(params, pre + "v2a_attn.", _rms(ax, params[pre + "v2a_norm_q.weight"], eps) * s,
               _rms(vx, params[pre + "v2a_norm_kv.weight"], eps), arope, vrope, heads,
               getattr(attention, "kv_rows", attention), eps, half)
  da = g * _rms(out, params[pre + "v2a_norm_out.weight"], eps)
  if gate is not None:
    dv = dv * gate[:, None, None].astype(dv.dtype)
    da = da * gate[:, None, None].astype(da.dtype)
  return dv, da


# ---------------------------------------------------------------------------
# joint forward
# ---------------------------------------------------------------------------


def positions(cap_lens, n_cap, grid):
  """(x_pos [B, N, 3], cap_pos [B, Lc, 3]) int32 for per-sample caption lengths."""
  f, h, w = grid
  cl = np.asarray(cap_lens)[:, None]
  ff, hh, ww = np.meshgrid(np.arange(f), np.arange(h), np.arange(w), indexing="ij")
  base = np.stack([ff.ravel(), hh.ravel(), ww.ravel()], -1)[None]          # [1, N, 3]
  x_pos = np.broadcast_to(base, (len(cap_lens),) + base.shape[1:]).copy()
  x_pos[..., 0] += cl + 1
  i = np.arange(n_cap)[None]
  cap_pos = np.zeros((len(cap_lens), n_cap, 3), np.int64)
  cap_pos[..., 0] = np.where(i < cl, 1 + i, 0)
  return x_pos.astype(np.int32), cap_pos.astype(np.int32)


def joint_forward(vp, ap, bp, vconf, aconf, video, audio, t, cap, cap_mask, v_pos, v_cap_pos,
                  a_pos, a_cap_pos, vrope, arope, gate=None, attention=dense_attention, half=False):
  """video [B, 48, F, H, W], audio [B, 128, T], t [B] -> (video pred, audio pred), float32.

  cap [B, Lc, 1024] / cap_mask [B, Lc]: the prompt embeddings (both towers
  take the same ones); ``*_pos``: from ``positions``; ``vrope`` / ``arope``:
  ``bridge_rope`` tables."""
  b, cv, f, h, w = video.shape
  ta = audio.shape[-1]
  dtype = vp["layers.0.attention.to_q.weight"].dtype
  vx = video.astype(dtype).transpose(0, 2, 3, 4, 1).reshape(b, f * h * w, cv)
  ax = audio.astype(dtype).transpose(0, 2, 1)
  nv, na = vx.shape[1], ax.shape[1]
  cap = cap.astype(dtype)
  # Sequence parallel (TPU): ``attention.parts`` chips, video rows split.
  parts = getattr(attention, "parts", 1)
  if nv % parts or (nv // parts + cap.shape[1]) % 8:
    parts = 1
  rows = getattr(attention, "rows", lambda x: x) if parts > 1 else (lambda x: x)
  vattn = functools.partial(attention, layout=(nv // parts, cap.shape[1])) if parts > 1 else attention
  vu, vmask, vcos, vsin, vt = prepare(vp, vconf, vx, cap, cap_mask, t, v_pos, v_cap_pos, attention, half, parts)
  vu = rows(vu)
  au, amask, acos, asin, at = prepare(ap, aconf, ax, cap, cap_mask, t, a_pos, a_cap_pos, attention, half)
  vh, ah = vconf["n_heads"], aconf["n_heads"]
  for i in range(vconf["n_layers"]):
    vu = _block(vp, f"layers.{i}", vu, vmask, vcos, vsin, vh, vattn, vt, vconf["norm_eps"], half)
    au = _block(ap, f"layers.{i}", au, amask, acos, asin, ah, attention, at, aconf["norm_eps"], half)
    dv, da = bridge(bp, i, x_rows(vu, nv, parts), au[:, :na], vrope, arope, vt, at, vh, attention, gate,
                    half=half)
    vu = rows(add_x(vu, dv, parts))
    au = au.at[:, :na].add(da)
  vout = final(vp, vu, vt, nv, parts).astype(jnp.float32).reshape(b, f, h, w, cv).transpose(0, 4, 1, 2, 3)
  aout = final(ap, au, at, na).astype(jnp.float32).transpose(0, 2, 1)
  return vout, aout
