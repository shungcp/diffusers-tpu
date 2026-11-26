"""LTX-2.3's audio-video DiT (diffusers ``LTX2VideoTransformer3DModel``) in JAX.

48 blocks, each with a video stream (dim 4096, 32 heads x 128) and an audio
stream (dim 2048, 32 heads x 64):

    self-attention (video, audio)           RMSNorm + AdaLN shift/scale, gate
    prompt cross-attention (video, audio)   LTX-2.3: q and the prompt modulated too
    audio->video and video->audio attention RoPE on time positions only
    feed-forward (GELU-tanh, 4x)            AdaLN shift/scale, gate

Every attention has q / k RMSNorm over all heads, "split" RoPE (each head's
first half rotates against its second half) and a per-head output gate
``2 sigmoid(W x)``. Timesteps are per video token (the conditioning first
frame is at t = 0), so the video modulation is per token too.

Spatio-temporal guidance (STG) replaces the self-attention of the listed
blocks by its value projection on the batch rows whose ``perturb`` is 1.

Precision follows diffusers in bf16: weights and activations bf16, norms and
RoPE in float32, timestep embeddings from float32 sinusoids. Loaded in
float32 the same code is the float32 reference.
"""
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

STREAMS = ("video", "audio")


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def _iter_state(folder):
  """(name, float32 or bf16 numpy array) for every tensor, without torch copies."""
  import ml_dtypes
  import torch
  from safetensors import safe_open

  for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
    with safe_open(path, "pt") as f:
      for k in f.keys():
        t = f.get_tensor(k)
        if t.dtype == torch.bfloat16:
          yield k, t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        else:
          yield k, t.float().numpy()


def load(folder, dtype=jnp.bfloat16, num_layers=None):
  """(params, config): top-level tensors by name, one dict per block under
  ``blocks``. Linears are [in, out]."""
  with open(os.path.join(folder, "config.json")) as fh:
    config = json.load(fh)
  n = config["num_layers"] if num_layers is None else num_layers
  top, per_layer = {}, {}
  for name, value in _iter_state(folder):
    v = np.asarray(value)
    if v.ndim == 2 and name.endswith(".weight"):
      v = v.T
    v = v.astype(jnp.dtype(dtype))
    if name.startswith("transformer_blocks."):
      _, idx, rest = name.split(".", 2)
      if int(idx) < n:
        per_layer.setdefault(rest, [None] * n)[int(idx)] = v
    else:
      top[name] = v
  top["blocks"] = [{k: v[i] for k, v in per_layer.items()} for i in range(n)]
  config = dict(config, num_layers=n)
  return top, config


# ---------------------------------------------------------------------------
# small layers
# ---------------------------------------------------------------------------


def _linear(x, p, name):
  w = p[name + ".weight"]
  y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=jnp.float32).astype(w.dtype)
  return y + p[name + ".bias"].astype(w.dtype)


def _rms(x, eps=1e-6, w=None):
  xf = x.astype(jnp.float32)
  y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), -1, keepdims=True) + eps)
  if w is not None:
    y = y * w.astype(jnp.float32)
  return y.astype(x.dtype)


def _layer_norm(x, eps=1e-6):
  xf = x.astype(jnp.float32)
  m = xf.mean(-1, keepdims=True)
  y = (xf - m) * jax.lax.rsqrt(jnp.square(xf - m).mean(-1, keepdims=True) + eps)
  return y.astype(x.dtype)


def _sinusoid(t, dim=256):
  """diffusers ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
  half = dim // 2
  freqs = jnp.exp(-np.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
  arg = t.astype(jnp.float32)[:, None] * freqs[None]
  return jnp.concatenate([jnp.cos(arg), jnp.sin(arg)], -1)


def _adaln_single(p, name, t, dtype):
  """``LTX2AdaLayerNormSingle``: (linear(silu(emb)), emb) for flat timesteps ``t``."""
  e = _sinusoid(t).astype(dtype)
  e = _linear(e, p, name + ".emb.timestep_embedder.linear_1")
  e = _linear(jax.nn.silu(e), p, name + ".emb.timestep_embedder.linear_2")
  return _linear(jax.nn.silu(e), p, name + ".linear"), e


# ---------------------------------------------------------------------------
# RoPE (host side; tables are per request shape)
# ---------------------------------------------------------------------------


def video_coords(frames, height, width, fps, scale=(8, 32, 32), causal_offset=1):
  """[3, N, 2] pixel-space [start, end) bounds per latent token (frame / h / w)."""
  f, h, w = np.meshgrid(np.arange(frames, dtype=np.float32), np.arange(height, dtype=np.float32),
                        np.arange(width, dtype=np.float32), indexing="ij")
  grid = np.stack([f, h, w]).reshape(3, -1)
  c = np.stack([grid, grid + 1], -1) * np.asarray(scale, np.float32)[:, None, None]
  c[0] = np.maximum(c[0] + causal_offset - scale[0], 0) / np.float32(fps)
  return c.astype(np.float32)


def audio_coords(frames, scale=4, hop=160, rate=16000, causal_offset=1):
  """[1, N, 2] seconds [start, end) per audio latent frame."""
  g = np.arange(frames, dtype=np.float32)
  start = np.maximum(g * scale + causal_offset - scale, 0) * hop / rate
  end = np.maximum((g + 1) * scale + causal_offset - scale, 0) * hop / rate
  return np.stack([start, end], -1)[None].astype(np.float32)


def rope_table(coords, dim, heads, max_pos, theta=10000.0):
  """(cos, sin) [N, heads, dim // heads // 2] for split RoPE (diffusers
  ``LTX2AudioVideoRotaryPosEmbed`` with ``rope_type="split"``)."""
  nd = coords.shape[0]
  mid = ((coords[..., 0] + coords[..., 1]) / np.float32(2.0)).astype(np.float32)  # [nd, N]
  grid = np.stack([mid[i] / np.float32(max_pos[i]) for i in range(nd)], -1)  # [N, nd]
  pw = np.power(theta, np.linspace(0.0, 1.0, dim // (2 * nd), dtype=np.float64))
  freqs = (pw * np.pi / 2.0).astype(np.float32)
  f = (grid[..., None] * np.float32(2) - np.float32(1)) * freqs  # [N, nd, F]
  f = np.swapaxes(f, -1, -2).reshape(f.shape[0], -1)  # [N, F * nd], frequency-major
  pad = dim // 2 - f.shape[-1]
  cos, sin = np.cos(f), np.sin(f)
  if pad:
    cos = np.concatenate([np.ones_like(cos[:, :pad]), cos], -1)
    sin = np.concatenate([np.zeros_like(sin[:, :pad]), sin], -1)
  n = f.shape[0]
  return cos.reshape(n, heads, -1).astype(np.float32), sin.reshape(n, heads, -1).astype(np.float32)


def ropes(config, frames, height, width, fps, audio_frames):
  """The four RoPE tables a forward needs: video / audio self-attention and
  the time-only ones of the audio<->video attentions."""
  c = config
  vc = video_coords(frames, height, width, fps, tuple(c["vae_scale_factors"]), c["causal_offset"])
  ac = audio_coords(audio_frames, c["audio_scale_factor"], c["audio_hop_length"], c["audio_sampling_rate"],
                    c["causal_offset"])
  dim, adim = c["num_attention_heads"] * c["attention_head_dim"], c["audio_num_attention_heads"] * c[
      "audio_attention_head_dim"]
  mp, amp = c["pos_embed_max_pos"], c["audio_pos_embed_max_pos"]
  cmp_ = max(mp, amp)
  return {
      "video": rope_table(vc, dim, c["num_attention_heads"], (mp, c["base_height"], c["base_width"]), c["rope_theta"]),
      "audio": rope_table(ac, adim, c["audio_num_attention_heads"], (amp,), c["rope_theta"]),
      "ca_video": rope_table(vc[:1], c["audio_cross_attention_dim"], c["num_attention_heads"], (cmp_,), c["rope_theta"]),
      "ca_audio": rope_table(ac, c["audio_cross_attention_dim"], c["audio_num_attention_heads"], (cmp_,),
                             c["rope_theta"]),
  }


def _apply_rope(x, rope):
  """x [B, S, H, D]; rope (cos, sin) [S, H, D/2]: first half rotates against the second."""
  cos, sin = (r[None] for r in rope)
  xf = x.astype(jnp.float32)
  r = x.shape[-1] // 2
  a, b = xf[..., :r], xf[..., r:]
  return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], -1).astype(x.dtype)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def dense_attention(q, k, v, bias=None):
  """[B, S, H, D]; ``bias`` additive [B, Sk] (float32). float32 scores."""
  s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / np.sqrt(q.shape[-1])
  if bias is not None:
    s = s + bias[:, None, None, :]
  p = jax.nn.softmax(s, -1).astype(v.dtype)
  return jnp.einsum("bhqk,bkhd->bqhd", p, v, preferred_element_type=jnp.float32).astype(v.dtype)


def _attn(p, name, x, ctx, heads, attention, q_rope=None, k_rope=None, bias=None, value_only=None):
  """``LTX2Attention`` (gated, q/k RMSNorm across heads, split RoPE).
  ``value_only`` [B] bool: rows whose output is the value projection (STG)."""
  b, s, _ = x.shape
  ctx = x if ctx is None else ctx
  gate = 2.0 * jax.nn.sigmoid(_linear(x, p, name + ".to_gate_logits"))  # [B, S, H]
  v = _linear(ctx, p, name + ".to_v")
  q = _rms(_linear(x, p, name + ".to_q"), w=p[name + ".norm_q.weight"])
  k = _rms(_linear(ctx, p, name + ".to_k"), w=p[name + ".norm_k.weight"])
  q, k, vh = (t.reshape(t.shape[0], t.shape[1], heads, -1) for t in (q, k, v))
  if q_rope is not None:
    q = _apply_rope(q, q_rope)
    k = _apply_rope(k, k_rope if k_rope is not None else q_rope)
  o = attention(q, k, vh, bias)
  if value_only is not None:
    o = jnp.where(value_only[:, None, None, None], vh, o)
  o = (o * gate[..., None].astype(o.dtype)).reshape(b, s, -1)
  return _linear(o, p, name + ".to_out.0")


def _ff(p, name, x):
  h = _linear(x, p, name + ".net.0.proj")
  return _linear(jax.nn.gelu(h, approximate=True), p, name + ".net.2")


def _mod(table, temb):
  """table [n, d] + temb [B, T, n * d] -> n arrays [B, T, d] (T = 1 or tokens)."""
  n, d = table.shape
  vals = table[None, None].astype(temb.dtype) + temb.reshape(temb.shape[0], temb.shape[1], n, d)
  return [vals[:, :, i] for i in range(n)]


# ---------------------------------------------------------------------------
# block / model
# ---------------------------------------------------------------------------


def _kind(attention, kind):
  """``attention``: one function for all, or a dict by kind (video_self,
  audio_self, video_text, audio_text, a2v, v2a)."""
  return attention[kind] if isinstance(attention, dict) else attention


def _tokens(mods, cond):
  """Video modulation [B, K, d] for the K distinct timesteps -> per token
  (a gather each consumer fuses), instead of [B, N, 9 d] through every block."""
  idx = cond.get("token_t")
  return mods if idx is None else [jnp.take(m, idx, axis=1) for m in mods]


def block(p, x, a, ctx, actx, cond, rope, bias, cfg, attention, stg=None, iso=None):
  """One ``LTX2VideoTransformerBlock`` (LTX-2.3 options). ``cond``: the
  modulation inputs (see ``forward``); ``stg`` [B] bool or None; ``iso`` [B]
  bool: rows without the audio<->video attentions (modality isolation)."""
  vh, ah = cfg["num_attention_heads"], cfg["audio_num_attention_heads"]
  sh, sc, g, sh_m, sc_m, g_m, sh_q, sc_q, g_q = _tokens(_mod(p["scale_shift_table"], cond["temb"]), cond)
  ash, asc, ag, ash_m, asc_m, ag_m, ash_q, asc_q, ag_q = _mod(p["audio_scale_shift_table"], cond["temb_audio"])

  # 1. self-attention
  h = _rms(x) * (1 + sc) + sh
  x = x + _attn(p, "attn1", h, None, vh, _kind(attention, "video_self"), rope["video"], value_only=stg) * g
  h = _rms(a) * (1 + asc) + ash
  a = a + _attn(p, "audio_attn1", h, None, ah, _kind(attention, "audio_self"), rope["audio"], value_only=stg) * ag

  # 2. prompt cross-attention (LTX-2.3: q and the prompt both modulated)
  sh_kv, sc_kv = _mod(p["prompt_scale_shift_table"], cond["temb_prompt"])
  ash_kv, asc_kv = _mod(p["audio_prompt_scale_shift_table"], cond["temb_prompt_audio"])
  h = _rms(x) * (1 + sc_q) + sh_q
  c = ctx * (1 + sc_kv) + sh_kv
  x = x + _attn(p, "attn2", h, c, vh, _kind(attention, "video_text"), bias=bias) * g_q
  h = _rms(a) * (1 + asc_q) + ash_q
  c = actx * (1 + asc_kv) + ash_kv
  a = a + _attn(p, "audio_attn2", h, c, ah, _kind(attention, "audio_text"), bias=bias) * ag_q

  # 3. audio <-> video
  nx, na = _rms(x), _rms(a)
  vt = p["video_a2v_cross_attn_scale_shift_table"]
  at = p["audio_a2v_cross_attn_scale_shift_table"]
  v_a2v_sc, v_a2v_sh, v_v2a_sc, v_v2a_sh = _mod(vt[:4], cond["ca_video"])
  (a2v_gate,) = _mod(vt[4:], cond["ca_video_gate"])
  a_a2v_sc, a_a2v_sh, a_v2a_sc, a_v2a_sh = _mod(at[:4], cond["ca_audio"])
  (v2a_gate,) = _mod(at[4:], cond["ca_audio_gate"])
  hq = nx * (1 + v_a2v_sc) + v_a2v_sh
  hk = na * (1 + a_a2v_sc) + a_a2v_sh
  keep = 1 if iso is None else (~iso).astype(x.dtype)[:, None, None]
  x = x + keep * (a2v_gate * _attn(p, "audio_to_video_attn", hq, hk, ah, _kind(attention, "a2v"), rope["ca_video"],
                                   rope["ca_audio"]))
  hq = na * (1 + a_v2a_sc) + a_v2a_sh
  hk = nx * (1 + v_v2a_sc) + v_v2a_sh
  a = a + keep * (v2a_gate * _attn(p, "video_to_audio_attn", hq, hk, ah, _kind(attention, "v2a"), rope["ca_audio"],
                                   rope["ca_video"]))

  # 4. feed-forward
  x = x + _ff(p, "ff", _rms(x) * (1 + sc_m) + sh_m) * g_m
  a = a + _ff(p, "audio_ff", _rms(a) * (1 + asc_m) + ash_m) * ag_m
  return x, a


def conditioning(params, t_video, t_audio, sigma, audio_sigma, dtype):
  """Modulation inputs of every block (LTX-2.3, ``use_cross_timestep=True``).
  t_video [B, K]: the K distinct video timesteps (scaled by 1000; the tokens
  pick theirs with ``token_t``), t_audio / sigma / audio_sigma [B]."""
  b = t_video.shape[0]
  temb, emb = _adaln_single(params, "time_embed", t_video.reshape(-1), dtype)
  temb_a, emb_a = _adaln_single(params, "audio_time_embed", t_audio, dtype)
  tp, _ = _adaln_single(params, "prompt_adaln", sigma, dtype)
  tpa, _ = _adaln_single(params, "audio_prompt_adaln", audio_sigma, dtype)
  cav, _ = _adaln_single(params, "av_cross_attn_video_scale_shift", audio_sigma, dtype)
  cavg, _ = _adaln_single(params, "av_cross_attn_video_a2v_gate", audio_sigma, dtype)
  caa, _ = _adaln_single(params, "av_cross_attn_audio_scale_shift", sigma, dtype)
  caag, _ = _adaln_single(params, "av_cross_attn_audio_v2a_gate", sigma, dtype)
  r = lambda t: t.reshape(b, -1, t.shape[-1])
  return {"temb": r(temb), "emb": r(emb), "temb_audio": r(temb_a), "emb_audio": r(emb_a),
          "temb_prompt": r(tp), "temb_prompt_audio": r(tpa), "ca_video": r(cav), "ca_video_gate": r(cavg),
          "ca_audio": r(caa), "ca_audio_gate": r(caag)}


def forward(params, cfg, x, a, ctx, actx, t_video, t_audio, sigma, rope, text_mask=None, stg=None,
            stg_blocks=(), attention=dense_attention, token_t=None, constrain=None, blocks_fn=None, isolate=None):
  """Velocity for video tokens x [B, N, 128] and audio tokens a [B, Na, 128].
  ctx / actx: the connectors' video / audio prompt embeddings; text_mask [B, L]
  (1 = keep); stg [B] bool rows perturbed in ``stg_blocks``. t_video: [B, N]
  per token, or [B, K] distinct values with ``token_t`` [N] their index."""
  dtype = params["proj_in.weight"].dtype
  cond = dict(conditioning(params, t_video, t_audio, sigma, sigma, dtype), token_t=token_t)
  bias = None if text_mask is None else (1.0 - text_mask.astype(jnp.float32)) * -10000.0
  x = _linear(x, params, "proj_in")
  a = _linear(a, params, "audio_proj_in")
  ctx, actx = ctx.astype(dtype), actx.astype(dtype)
  rope = {k: tuple(jnp.asarray(t) for t in v) for k, v in rope.items()}
  if blocks_fn is not None:  # e.g. jax_sp: the blocks sequence parallel in one shard_map
    x, a = blocks_fn(params["blocks"], x, a, ctx, actx, cond, rope, stg, isolate)
    blocks = []
  else:
    blocks = params["blocks"]
  # Unrolled: a scan over stacked weights copies every layer's slice.
  for i, p in enumerate(blocks):
    vo = stg if i in set(stg_blocks) else None
    if constrain is not None:
      x = constrain(x)
    x, a = block(p, x, a, ctx, actx, cond, rope, bias, cfg, attention, vo, isolate)
  def out(h, table, emb, name):
    sh, sc = table[0].astype(emb.dtype) + emb, table[1].astype(emb.dtype) + emb
    return _linear(_layer_norm(h) * (1 + sc) + sh, params, name)

  emb = cond["emb"] if token_t is None else jnp.take(cond["emb"], token_t, axis=1)
  x = out(x, params["scale_shift_table"], emb, "proj_out")
  a = out(a, params["audio_scale_shift_table"], cond["emb_audio"], "audio_proj_out")
  return x, a


# ---------------------------------------------------------------------------
# TPU: tensor-parallel layout and the splash kernel
# ---------------------------------------------------------------------------

_COLUMN = (".to_q", ".to_k", ".to_v", ".to_gate_logits", ".net.0.proj")
_ROW = (".to_out.0", ".net.2")


def param_specs(params, axis="tp", fsdp=None):
  """Megatron layout over ``axis``: heads and FFN columns split, the output
  projections split by rows (their sums all-reduced); the rest replicated.
  ``fsdp``: also split the other dimension of the big matrices over that axis
  (all-gathered layer by layer while the rows are split over it).
  """
  from jax.sharding import PartitionSpec as P

  def spec(name, v, lead):
    base = name.rsplit(".", 1)[0]
    pre = (None,) * lead
    if base.endswith(_COLUMN) or base.endswith((".norm_q", ".norm_k")):
      return P(*pre, fsdp, axis) if v.ndim - lead == 2 else P(*pre, axis)
    if base.endswith(_ROW) and name.endswith(".weight"):
      return P(*pre, axis, fsdp)
    return P()

  out = {k: spec(k, v, 0) for k, v in params.items() if k != "blocks"}
  out["blocks"] = [{k: spec(k, v, 0) for k, v in blk.items()} for blk in params["blocks"]]
  return out


def place(params, mesh, axis="tp", fsdp=None):
  from jax.sharding import NamedSharding

  specs = param_specs(params, axis, fsdp)
  return jax.tree.map(lambda v, s: jax.device_put(v, NamedSharding(mesh, s)), params, specs)


def make_attention(mesh, axis="tp", seq=None):
  """[B, S, H, D] attention: the splash kernel per chip with heads over
  ``axis`` (and query rows over ``seq``: keys gathered) for the long
  self-attention, dense (XLA) for the short ones."""
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P

  from diffusers_tpu.kernels import block_sizes, splash

  qspec, spec = P(None, axis, seq, None), P(None, axis, None, None)

  def local(q, k, v):
    blocks = block_sizes.resolve_block_sizes(q.shape[2], k.shape[2], q.shape[-1], {}, "residual",
                                             block_sizes.detect_spec())
    return splash.attention(q, k, v, scale=None, **blocks)

  run = shard_map(local, mesh=mesh, in_specs=(qspec, spec, spec), out_specs=qspec, check_rep=False)

  def attention(q, k, v, bias=None):
    if bias is None and min(q.shape[1], k.shape[1]) >= 1024 and q.shape[1] % 128 == 0 and k.shape[1] % 128 == 0:
      q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
      return run(q, k, v).transpose(0, 2, 1, 3)
    return dense_attention(q, k, v, bias)

  return attention
