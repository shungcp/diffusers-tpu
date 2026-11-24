"""MOVA's dual-tower DiT in JAX: the video tower, the audio tower and the bridge.

Both towers are DiffSynth's Wan DiT (``mova/diffusion/models/wan_*_dit.py``):

* video: Wan2.2-A14B-like, two experts with the same layout (dim 5120, 40
  layers, 40 heads x 128, FFN 13824), a (1, 2, 2) patch over 36 input
  channels (16 noise + 4 mask + 16 first-frame latent) and a 3-D RoPE (time /
  height / width over 44 / 42 / 42 of each head's dims, interleaved pairs);
* audio: Wan-1.3B-like (dim 1536, 30 layers, 12 heads), a 1-D patch and a
  1-D RoPE over the whole head.

The bridge (``interactionv2.DualTowerConditionalBridge``, strategy "full")
runs before each of the first 30 layers: the video tokens cross-attend to
the audio tokens (a2v) and the audio tokens to the video tokens (v2a), both
from the states *before* the layer, each through LayerNorm(y), q/k RMSNorm
over the full width and a rotate-half RoPE on time positions aligned
between the towers (video frame f sits at f * audio_fps / (video_fps / 4)).

Precision is the official run's: every weight in bf16, the timestep
embedding computed in float32 and then cast to bf16, everything else in the
weights' dtype (LayerNorm / RMSNorm reduce in float32; RoPE angles float64
on the host). Loaded in float32 the same code is the float32 reference.
"""
import functools
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def _read_state(folder):
  from safetensors.numpy import load_file

  state = {}
  for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
    for k, v in load_file(path).items():
      state[k] = v
  return state


def _np32(v):
  return np.asarray(v, np.float32) if v.dtype != np.float32 else v


def _host(value, dtype):
  """A host (numpy) array in ``dtype``: nothing lands on a device until
  ``place`` puts it there already sharded (a 14B expert does not fit one chip)."""
  return np.asarray(value).astype(jnp.dtype(dtype))


def load_tower(folder, dtype=jnp.bfloat16):
  """(params, config) of a Wan tower as host arrays; linears [in, out]."""
  with open(os.path.join(folder, "config.json")) as fh:
    config = json.load(fh)
  state = _read_state(folder)
  params = {}
  for name, value in state.items():
    value = _np32(value)
    keep = dtype
    if name.startswith(("time_embedding.", "time_projection.")):
      keep = jnp.float32  # computed in float32, cast to bf16 afterwards (as upstream)
    if name == "patch_embedding.weight":
      value = value.reshape(value.shape[0], -1).T  # [C*kt*kh*kw, dim]
    elif value.ndim == 2 and name.endswith(".weight"):
      value = value.T
    elif name.endswith("modulation"):
      value = value.reshape(value.shape[-2], value.shape[-1])
    params[name] = _host(value, keep)
  return params, config


def load_bridge(folder, dtype=jnp.bfloat16):
  with open(os.path.join(folder, "config.json")) as fh:
    config = json.load(fh)
  params = {}
  for name, value in _read_state(folder).items():
    value = _np32(value)
    params[name] = _host(value.T if value.ndim == 2 else value, dtype)
  return params, config


# ---------------------------------------------------------------------------
# small layers
# ---------------------------------------------------------------------------


def _linear_default(x, params, name):
  """In the weights' dtype, like a bf16 ``nn.Linear`` (the MXU still accumulates in
  float32). Under tensor parallelism the row-split projections' partial sums
  are then all-reduced in bf16: in float32 they were 39% of a step (150 x 1.8 GB)."""
  w = params[name + ".weight"]
  y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=w.dtype)
  return y + params[name + ".bias"].astype(w.dtype)


_linear = _linear_default


def _layer_norm(x, eps=1e-6, w=None, b=None):
  xf = x.astype(jnp.float32)
  mean = xf.mean(-1, keepdims=True)
  y = (xf - mean) * jax.lax.rsqrt(jnp.square(xf - mean).mean(-1, keepdims=True) + eps)
  if w is not None:
    y = y * w.astype(jnp.float32) + b.astype(jnp.float32)
  return y.astype(x.dtype)


def _rms(x, w, eps=1e-6):
  xf = x.astype(jnp.float32)
  y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + eps)
  return (y * w.astype(jnp.float32)).astype(x.dtype)


def time_sinusoid(t, dim=256):
  t = np.asarray(t, np.float64).reshape(-1)
  half = dim // 2
  s = np.outer(t, np.power(10000.0, -np.arange(half, dtype=np.float64) / half))
  return np.concatenate([np.cos(s), np.sin(s)], 1).astype(np.float32)


def _freqs(dim, end, theta=10000.0):
  """``precompute_freqs_cis`` angles (float64) [end, dim // 2]."""
  inv = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2].astype(np.float64) / dim))
  return np.outer(np.arange(end, dtype=np.float64), inv)


def video_rope(t, h, w, head_dim):
  """(cos, sin) [t*h*w, head_dim // 2] for the 3-D RoPE (interleaved pairs)."""
  ft = _freqs(head_dim - 2 * (head_dim // 3), t)
  fh = _freqs(head_dim // 3, h)
  fw = _freqs(head_dim // 3, w)
  ang = np.concatenate([
      np.broadcast_to(ft[:, None, None], (t, h, w, ft.shape[-1])),
      np.broadcast_to(fh[None, :, None], (t, h, w, fh.shape[-1])),
      np.broadcast_to(fw[None, None, :], (t, h, w, fw.shape[-1])),
  ], -1).reshape(t * h * w, -1)
  return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def audio_rope(length, head_dim):
  """(cos, sin) [length, head_dim // 2]: ``precompute_freqs_cis_1d`` (whole head)."""
  ang = _freqs(head_dim, length)
  return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def bridge_rope(grid, audio_len, video_fps, audio_fps, head_dim, theta=10000.0):
  """``build_aligned_freqs``: rotate-half (cos, sin) [L, head_dim] for both towers."""
  f, h, w = grid
  inv = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
  video_pos = np.repeat(np.arange(f, dtype=np.float32) * np.float32(audio_fps / (video_fps / 4.0)), h * w)
  audio_pos = np.arange(audio_len, dtype=np.float32)

  def table(pos):
    a = (pos[:, None] * inv[None]).astype(np.float32)
    a = np.concatenate([a, a], -1)
    return np.cos(a), np.sin(a)

  return table(video_pos), table(audio_pos)


def _rope_pairs(x, cos, sin):
  """Interleaved-pair rotation of x [B, S, H, D] in float32 (upstream: float64 complex)."""
  xf = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
  even, odd = xf[..., 0], xf[..., 1]
  c, s = cos[None, :, None], sin[None, :, None]
  return jnp.stack((even * c - odd * s, even * s + odd * c), -1).reshape(x.shape).astype(x.dtype)


def _rope_half(x, cos, sin):
  """rotate-half RoPE of x [B, S, H, D]; cos/sin [S, D] cast to x's dtype (upstream)."""
  c, s = cos.astype(x.dtype)[None, :, None], sin.astype(x.dtype)[None, :, None]
  x1, x2 = jnp.split(x, 2, -1)
  return x * c + jnp.concatenate((-x2, x1), -1) * s


def dense_attention(q, k, v):
  """[B, S, H, D] -> [B, S, H, D]; float32 scores (the CPU reference path)."""
  s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
  p = jax.nn.softmax(s, -1).astype(v.dtype)
  return jnp.einsum("bhqk,bkhd->bqhd", p, v, preferred_element_type=jnp.float32).astype(v.dtype)


# ---------------------------------------------------------------------------
# towers
# ---------------------------------------------------------------------------


def embed_time(params, config, sinusoid):
  """(t [B, dim], t_mod [B, 6, dim]) from ``time_sinusoid(t)``, computed in
  float32 then cast to the weights' dtype (as upstream)."""
  dim = config["dim"]
  te = _linear(jnp.asarray(sinusoid), params, "time_embedding.0").astype(jnp.float32)
  te = _linear(jax.nn.silu(te), params, "time_embedding.2").astype(jnp.float32)
  tm = _linear(jax.nn.silu(te), params, "time_projection.1").astype(jnp.float32).reshape(-1, 6, dim)
  dtype = params["head.head.weight"].dtype
  return te.astype(dtype), tm.astype(dtype)


def embed_text(params, context):
  y = _linear(context, params, "text_embedding.0")
  return _linear(jax.nn.gelu(y.astype(jnp.float32), approximate=True).astype(y.dtype), params,
                 "text_embedding.2")


def block(params, i, x, ctx, t_mod, cos, sin, heads, attention, linear=None, cross_attention=None):
  """``DiTBlock.forward`` for layer ``i``."""
  _linear = linear or _linear_default
  cross_attention = cross_attention or attention
  pre = f"blocks.{i}."
  mod = params[pre + "modulation"][None] + t_mod
  shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (mod[:, j][:, None] for j in range(6))
  b, s, d = x.shape
  shape = (b, -1, heads, d // heads)
  y = _layer_norm(x) * (1 + scale_a) + shift_a
  q = _rms(_linear(y, params, pre + "self_attn.q"), params[pre + "self_attn.norm_q.weight"])
  k = _rms(_linear(y, params, pre + "self_attn.k"), params[pre + "self_attn.norm_k.weight"])
  v = _linear(y, params, pre + "self_attn.v")
  q = _rope_pairs(q.reshape(shape), cos, sin)
  k = _rope_pairs(k.reshape(shape), cos, sin)
  out = attention(q, k, v.reshape(shape)).reshape(b, s, d)
  x = x + gate_a * _linear(out, params, pre + "self_attn.o")
  y = _layer_norm(x, w=params[pre + "norm3.weight"], b=params[pre + "norm3.bias"])
  q = _rms(_linear(y, params, pre + "cross_attn.q"), params[pre + "cross_attn.norm_q.weight"])
  k = _rms(_linear(ctx, params, pre + "cross_attn.k"), params[pre + "cross_attn.norm_k.weight"])
  v = _linear(ctx, params, pre + "cross_attn.v")
  out = cross_attention(q.reshape(shape), k.reshape(b, -1, heads, d // heads),
                        v.reshape(b, -1, heads, d // heads)).reshape(b, s, d)
  x = x + _linear(out, params, pre + "cross_attn.o")
  y = _layer_norm(x) * (1 + scale_f) + shift_f
  h = _linear(y, params, pre + "ffn.0")
  h = jax.nn.gelu(h.astype(jnp.float32), approximate=True).astype(h.dtype)
  return x + gate_f * _linear(h, params, pre + "ffn.2")


def head(params, x, t):
  m = params["head.modulation"][None] + t[:, None]  # [B, 2, dim]
  y = _layer_norm(x) * (1 + m[:, 1:2]) + m[:, 0:1]
  return _linear(y, params, "head.head")


def bridge_attend(params, name, x, y, x_rope, y_rope, heads, attention, linear=None):
  """``ConditionalCrossAttentionBlock``: x attends to LayerNorm(y)."""
  _linear = linear or _linear_default
  y = _layer_norm(y, w=params[name + ".y_norm.weight"], b=params[name + ".y_norm.bias"])
  pre = name + ".inner."
  q = _rms(_linear(x, params, pre + "q"), params[pre + "norm_q.weight"])
  k = _rms(_linear(y, params, pre + "k"), params[pre + "norm_k.weight"])
  v = _linear(y, params, pre + "v")
  b, sx, d = q.shape
  q = _rope_half(q.reshape(b, sx, heads, -1), *x_rope)
  k = _rope_half(k.reshape(b, k.shape[1], heads, -1), *y_rope)
  out = attention(q, k, v.reshape(b, v.shape[1], heads, -1)).reshape(b, sx, d)
  return _linear(out, params, pre + "o")


def patchify_video(latents, patch):
  """[B, C, T, H, W] -> [B, T*h*w, C*pt*ph*pw] in the conv kernel's order."""
  b, c, t, hh, ww = latents.shape
  pt, ph, pw = patch
  x = latents.reshape(b, c, t // pt, pt, hh // ph, ph, ww // pw, pw)
  x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
  return x.reshape(b, (t // pt) * (hh // ph) * (ww // pw), c * pt * ph * pw)


def unpatchify_video(x, grid, patch, out_dim):
  """'b (f h w) (x y z c) -> b c (f x) (h y) (w z)'."""
  b = x.shape[0]
  f, h, w = grid
  pt, ph, pw = patch
  x = x.reshape(b, f, h, w, pt, ph, pw, out_dim)
  return x.transpose(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, out_dim, f * pt, h * ph, w * pw)


def dual_step(video, vconf, audio, aconf, bridge, bconf, visual_latents, audio_latents, context,
              t, audio_t, video_fps, *, attention=dense_attention, video_layer=None):
  """``MOVA.inference_single_step``: (video velocity [B,16,T,H,W], audio velocity [B,128,L]).

  ``t`` / ``audio_t`` are ``time_sinusoid`` embeddings [1, 256]."""
  dtype = video["head.head.weight"].dtype
  vt, vtm = embed_time(video, vconf, t)
  at, atm = embed_time(audio, aconf, audio_t)
  vctx = embed_text(video, context.astype(dtype))
  actx = embed_text(audio, context.astype(dtype))
  patch = tuple(vconf["patch_size"])
  b, _, tt, hh, ww = visual_latents.shape
  grid = (tt // patch[0], hh // patch[1], ww // patch[2])
  vx = jnp.dot(patchify_video(visual_latents.astype(dtype), patch), video["patch_embedding.weight"],
               preferred_element_type=jnp.float32)
  vx = (vx + video["patch_embedding.bias"].astype(jnp.float32)).astype(dtype)
  ax = jnp.dot(audio_latents.astype(dtype).transpose(0, 2, 1), audio["patch_embedding.weight"],
               preferred_element_type=jnp.float32)
  ax = (ax + audio["patch_embedding.bias"].astype(jnp.float32)).astype(dtype)
  vhd = vconf["dim"] // vconf["num_heads"]
  ahd = aconf["dim"] // aconf["num_heads"]
  vcos, vsin = (jnp.asarray(a) for a in video_rope(*grid, vhd))
  acos, asin = (jnp.asarray(a) for a in audio_rope(ax.shape[1], ahd))
  (bvc, bvs), (bac, bas) = bridge_rope(grid, ax.shape[1], video_fps, bconf["audio_fps"], bconf["head_dim"])
  vrope, arope = (jnp.asarray(bvc), jnp.asarray(bvs)), (jnp.asarray(bac), jnp.asarray(bas))
  vheads, aheads = vconf["num_heads"], aconf["num_heads"]
  bvh = bconf["visual_hidden_dim"] // bconf["head_dim"]
  bah = bconf["audio_hidden_dim"] // bconf["head_dim"]
  shared = min(vconf["num_layers"], aconf["num_layers"])
  if video_layer is not None:  # sequence parallel (``make_video_layer``)
    for i in range(vconf["num_layers"]):
      vp = {k: v for k, v in video.items() if k.startswith(f"blocks.{i}.")}
      bp = ({k: v for k, v in bridge.items() if f"_conditioners.{i}." in k} if i < shared else None)
      vx, v2a = video_layer(i, vp, bp, vx, ax, vctx, vtm, vcos, vsin, vrope, arope)
      if i < shared:
        ax = block(audio, i, ax + v2a, actx, atm, acos, asin, aheads, attention)
    vout = unpatchify_video(head(video, vx, vt), grid, patch, vconf["out_dim"])
    aout = head(audio, ax, at).transpose(0, 2, 1)
    return vout, aout
  for i in range(shared):
    a2v = bridge_attend(bridge, f"audio_to_video_conditioners.{i}", vx, ax, vrope, arope, bvh, attention)
    v2a = bridge_attend(bridge, f"video_to_audio_conditioners.{i}", ax, vx, arope, vrope, bah, attention)
    vx, ax = vx + a2v, ax + v2a
    vx = block(video, i, vx, vctx, vtm, vcos, vsin, vheads, attention)
    ax = block(audio, i, ax, actx, atm, acos, asin, aheads, attention)
  for i in range(shared, vconf["num_layers"]):
    vx = block(video, i, vx, vctx, vtm, vcos, vsin, vheads, attention)
  vout = unpatchify_video(head(video, vx, vt), grid, patch, vconf["out_dim"])
  aout = head(audio, ax, at).transpose(0, 2, 1)
  return vout, aout


def _check(snapshot, ref_path):
  """inference_single_step against the official code (float32, CPU), both experts."""
  ref = np.load(ref_path)
  jax.config.update("jax_default_matmul_precision", "highest")
  audio, aconf = load_tower(os.path.join(snapshot, "audio_dit"), jnp.float32)
  bridge, bconf = load_bridge(os.path.join(snapshot, "dual_tower_bridge"), jnp.float32)
  rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
  for tag, folder in (("hi", "video_dit"), ("lo", "video_dit_2")):
    video, vconf = load_tower(os.path.join(snapshot, folder), jnp.float32)
    t = float(ref[f"t_{tag}"])
    sin_t = jnp.asarray(time_sinusoid([t]))
    v, a = dual_step(video, vconf, audio, aconf, bridge, bconf, jnp.asarray(ref["visual_in"]),
                     jnp.asarray(ref["audio_in"]), jnp.asarray(ref["context"]), sin_t, sin_t,
                     float(ref["fps"]))
    print(f"{folder} (t={t}): video rel L2 {rel(np.asarray(v), ref[f'out_v_{tag}']):.2e}, "
          f"audio rel L2 {rel(np.asarray(a), ref[f'out_a_{tag}']):.2e}", flush=True)
    del video


if __name__ == "__main__":
  import sys

  from diffusers_tpu.hub import local_snapshot

  _check(local_snapshot("OpenMOSS-Team/MOVA-360p"), sys.argv[1])


# ---------------------------------------------------------------------------
# TPU: tensor-parallel layout and the splash kernel
# ---------------------------------------------------------------------------

_COLUMN = ("self_attn.q", "self_attn.k", "self_attn.v", "cross_attn.q", "cross_attn.k",
           "cross_attn.v", "ffn.0", "inner.q", "inner.k", "inner.v")
_ROW = ("self_attn.o", "cross_attn.o", "ffn.2", "inner.o")


def param_specs(params):
  """Heads / FFN columns over ``tensor`` (Megatron layout); the rest replicated.

  The q/k RMSNorms run over the full width, so their statistic is all-reduced
  (small); everything else of a block stays on its chip until ``o`` / ``ffn.2``.
  """
  from jax.sharding import PartitionSpec as P

  specs = {}
  for name, value in params.items():
    base = name.rsplit(".", 1)[0]
    if any(base.endswith(c) for c in _COLUMN):
      specs[name] = P(None, "tensor") if name.endswith(".weight") else P("tensor")
    elif any(base.endswith(r) for r in _ROW) and name.endswith(".weight"):
      specs[name] = P("tensor", None)
    else:
      specs[name] = P()
  return specs


def place(params, mesh):
  from jax.sharding import NamedSharding

  specs = param_specs(params)
  return {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in params.items()}


def _splash_local(q, k, v):
  """Device-local splash attention, [B, H, S, D]. Fixed-m softmax (exact:
  heads outside the safe bound fall back to the online softmax), which needs
  zero-mean keys -- softmax is invariant to that shift. 43120 x 43120 with
  2 x 10 heads on a v6e: 45.6 -> 35.1 ms."""
  from diffusers_tpu.kernels import block_sizes, splash

  blocks = block_sizes.resolve_block_sizes(q.shape[2], k.shape[2], q.shape[-1], {},
                                           "residual-fixedm", block_sizes.detect_spec())
  k = k - jnp.mean(k, axis=2, keepdims=True)
  return splash.attention(q, k, v, scale=None, use_fixed_m=True, **blocks)


def make_attention(mesh):
  """[B, S, H, D] attention: the splash kernel per chip, heads over ``tensor``."""
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  from diffusers_tpu.kernels import splash  # noqa: F401  (import outside any trace)

  local = _splash_local

  def runner(spec):
    return shard_map(local, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec,
                     check_rep=False)

  split = runner(P("cfg", "tensor", None, None))
  # Heads that do not divide over ``tensor`` (the audio tower's 12 on 8 chips)
  # run whole on every chip of their cfg group: 403 rows, a negligible cost.
  whole = runner(P("cfg", None, None, None))
  parts = mesh.shape["tensor"]

  def attention(q, k, v):
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    run = split if q.shape[1] % parts == 0 else whole
    return run(q, k, v).transpose(0, 2, 1, 3)

  return attention


def make_video_layer(mesh, vconf, bconf):
  """One video layer (with its bridge cross-attentions) sequence parallel over
  ``tensor``: each chip keeps a quarter of the rows of the residual stream and
  runs every token-wise op on them with whole weights, all-gathered layer by
  layer from their tensor-parallel shards (they do not depend on activations,
  so the scheduler can fetch them early). Self-attention swaps rows for heads
  with an all-to-all each way (Ulysses); the cross-attentions to text / audio
  keep all heads per chip; video -> audio combines each chip's partial softmax
  over its rows of keys. Per layer this moves ~0.5 GB of weights and four
  0.2 GB all-to-alls a chip, against three all-reduces of the 0.9 GB
  residual stream under tensor parallelism.

  Returns ``layer(i, video_params_i, bridge_params_i | None, vx, ax, vctx, vtm,
  vcos, vsin, vrope, arope) -> (vx, v2a | None)``."""
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  from diffusers_tpu.kernels import splash  # noqa: F401  (import outside any trace)

  T = "tensor"
  heads = vconf["num_heads"]
  if heads % mesh.shape[T]:
    raise ValueError(f"sequence parallel needs the {heads} video heads to divide over "
                     f"tensor={mesh.shape[T]}")
  bvh = bconf["visual_hidden_dim"] // bconf["head_dim"]
  bah = bconf["audio_hidden_dim"] // bconf["head_dim"]

  def linear(x, params, name):
    base = name
    w, b = params[name + ".weight"], params[name + ".bias"]
    if any(base.endswith(c) for c in _COLUMN):
      w = jax.lax.all_gather(w, T, axis=1, tiled=True)
      b = jax.lax.all_gather(b, T, axis=0, tiled=True)
    elif any(base.endswith(r) for r in _ROW):
      w = jax.lax.all_gather(w, T, axis=0, tiled=True)
    y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=w.dtype)
    return y + b.astype(w.dtype)

  def local(q, k, v):  # [B, S, H, D], every head on this chip
    out = _splash_local(*(t.transpose(0, 2, 1, 3) for t in (q, k, v)))
    return out.transpose(0, 2, 1, 3)

  def ulysses(q, k, v):  # rows -> heads, attend, heads -> rows
    to_heads = lambda t: jax.lax.all_to_all(t, T, 2, 1, tiled=True)
    out = local(to_heads(q), to_heads(k), to_heads(v))
    return jax.lax.all_to_all(out, T, 1, 2, tiled=True)

  def video_to_audio(bp, name, ax, vx, arope, vrope):
    """Audio queries over every chip's rows of video keys: partial softmax
    statistics combined with pmax / psum (exact)."""
    y = _layer_norm(vx, w=bp[name + ".y_norm.weight"], b=bp[name + ".y_norm.bias"])
    pre = name + ".inner."
    q = _rms(linear(ax, bp, pre + "q"), bp[pre + "norm_q.weight"])
    k = _rms(linear(y, bp, pre + "k"), bp[pre + "norm_k.weight"])
    v = linear(y, bp, pre + "v")
    b, sq, d = q.shape
    q = _rope_half(q.reshape(b, sq, bah, -1), *arope)
    k = _rope_half(k.reshape(b, k.shape[1], bah, -1), *vrope)
    v = v.reshape(b, v.shape[1], bah, -1)
    sc = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
    m = jax.lax.pmax(jnp.max(sc, -1, keepdims=True), T)
    e = jnp.exp(sc - m)
    l = jax.lax.psum(jnp.sum(e, -1, keepdims=True), T)              # [b, h, q, 1]
    o = jax.lax.psum(jnp.einsum("bhqk,bkhd->bqhd", e.astype(v.dtype), v,
                                preferred_element_type=jnp.float32), T)
    o = (o / jnp.transpose(l, (0, 2, 1, 3))).astype(v.dtype).reshape(b, sq, d)
    return linear(o, bp, pre + "o")

  def body(i, vp, bp, vx, ax, vctx, vtm, vcos, vsin, vrope, arope):
    v2a = None
    if bp is not None:
      v2a = video_to_audio(bp, f"video_to_audio_conditioners.{i}", ax, vx, arope, vrope)
      vx = vx + bridge_attend(bp, f"audio_to_video_conditioners.{i}", vx, ax, vrope, arope, bvh,
                              local, linear=linear)
    vx = block(vp, i, vx, vctx, vtm, vcos, vsin, heads, ulysses, linear=linear,
               cross_attention=local)
    return vx, v2a

  rows, batch, rep = P("cfg", T, None), P("cfg"), P()
  cache = {}

  def layer(i, vp, bp, vx, ax, vctx, vtm, vcos, vsin, vrope, arope):
    key = (i, bp is not None)
    if key not in cache:
      specs = lambda d: {k: v for k, v in param_specs(d).items()}
      in_specs = (specs(vp), specs(bp) if bp is not None else None, rows, batch, batch, rep,
                  P(T, None), P(T, None), (P(T, None), P(T, None)), (rep, rep))
      out_specs = (rows, batch if bp is not None else None)
      fn = lambda *a, i=i: body(i, *a)
      cache[key] = shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                             check_rep=False)
    return cache[key](vp, bp, vx, ax, vctx, vtm, vcos, vsin, vrope, arope)

  return layer
