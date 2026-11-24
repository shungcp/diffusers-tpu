"""MOSS-SoundEffect-v2's audio DiT (Wan 1.3B layout) and sampling step in JAX.

The DiT is DiffSynth's ``WanAudioModel`` with a 1-D patch: 30 blocks of
self-attention (1-D RoPE over the whole head, interleaved pairs), cross-
attention to 512 text tokens (no mask -- the pipeline zeroes the padded
rows), and a GELU-tanh FFN, each block modulated six ways by the timestep.
Weights are read from the Hub checkpoint's diffusers-style names.

Precision follows the official GPU run, which is CUDA autocast(bf16):
matmuls in bf16, the timestep embedding and modulation in float32, the
norms in float32, and the residual stream in float32 (``x + gate * out``
with a float32 gate). The CPU reference runs everything in float32.

One sampling step is one program, as for MOSS-Image: the DiT on
[positive, negative] as one batch (both contexts are 512 rows, so there is
no padding to mask), CFG and the Euler update in float32. On a
``('cfg', 'tensor')`` mesh the batch is split over ``cfg``; the second axis
splits the sequence (``seq_parallel``, keys and values gathered) or the
heads and FFN columns (tensor parallel). Attention is the splash kernel;
the CPU check uses a plain float32 softmax.

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m diffusers_tpu.models.moss_soundeffect.jax_dit \
        --check dit.npz
compares one forward against the official code's output (float32).
"""
import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# At import, not inside a traced function (importing the Pallas kernel
# registers config options, which would make a jitted step retrace once).
from diffusers_tpu.kernels import block_sizes, splash

def _halves_permutation(dim, heads):
  """Per head, the interleaved pairs (2i, 2i+1) regrouped as [evens | odds]."""
  head_dim = dim // heads
  within = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
  return (np.arange(heads)[:, None] * head_dim + within[None]).reshape(-1)


def load(transformer_dir, dtype=jnp.bfloat16):
  """(params, config); linears as [in, out], one entry per layer (``blocks.<i>.*``).

  The self-attention's q/k output features are permuted within each head from
  interleaved pairs to [evens | odds], so RoPE rotates two contiguous halves
  instead of a strided (..., 64, 2) view (``q . k`` is unchanged when both are
  permuted alike; the RMSNorm over all 1536 features commutes with it).
  """
  from safetensors.numpy import load_file

  with open(os.path.join(transformer_dir, "config.json")) as fh:
    config = json.load(fh)
  state = load_file(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))
  depth = config["num_layers"]
  params = {}

  def lin(dst, src, keep=None):
    params[dst + ".w"] = jnp.asarray(state[src + ".weight"].T, keep or dtype)
    params[dst + ".b"] = jnp.asarray(state[src + ".bias"], keep or jnp.float32)

  ce = "condition_embedder"
  lin("text1", f"{ce}.text_embedder.linear_1")
  lin("text2", f"{ce}.text_embedder.linear_2")
  # The timestep path runs in float32 (autocast(float32) in the official code).
  lin("time1", f"{ce}.time_embedder.linear_1", jnp.float32)
  lin("time2", f"{ce}.time_embedder.linear_2", jnp.float32)
  lin("tproj", f"{ce}.time_proj", jnp.float32)
  params["patch.w"] = jnp.asarray(state["patch_embedding.weight"][:, :, 0].T, dtype)
  params["patch.b"] = jnp.asarray(state["patch_embedding.bias"], jnp.float32)
  lin("out", "proj_out")
  params["head_mod"] = jnp.asarray(state["scale_shift_table"][0], jnp.float32)

  names = {"sa.q": "attn1.to_q", "sa.k": "attn1.to_k", "sa.v": "attn1.to_v",
           "sa.o": "attn1.to_out.0", "ca.q": "attn2.to_q", "ca.k": "attn2.to_k",
           "ca.v": "attn2.to_v", "ca.o": "attn2.to_out.0", "ffn.up": "ffn.net.0.proj",
           "ffn.down": "ffn.net.2"}
  stack = lambda f: np.stack([f(i) for i in range(depth)])
  for ours, theirs in names.items():
    params[f"blocks.{ours}.w"] = jnp.asarray(
        stack(lambda i: state[f"blocks.{i}.{theirs}.weight"].T), dtype)
    params[f"blocks.{ours}.b"] = jnp.asarray(
        stack(lambda i: state[f"blocks.{i}.{theirs}.bias"]), jnp.float32)
  perm = _halves_permutation(config["dim"], config["num_heads"])
  for name in ("sa.q", "sa.k"):
    params[f"blocks.{name}.w"] = params[f"blocks.{name}.w"][:, :, perm]
    params[f"blocks.{name}.b"] = params[f"blocks.{name}.b"][:, perm]
  for ours, theirs in (("sa.norm_q", "attn1.norm_q"), ("sa.norm_k", "attn1.norm_k"),
                       ("ca.norm_q", "attn2.norm_q"), ("ca.norm_k", "attn2.norm_k")):
    w = stack(lambda i: state[f"blocks.{i}.{theirs}.weight"])
    params[f"blocks.{ours}"] = jnp.asarray(w[:, perm] if ours.startswith("sa.") else w, jnp.float32)
  params["blocks.norm3.w"] = jnp.asarray(stack(lambda i: state[f"blocks.{i}.norm2.weight"]),
                                         jnp.float32)
  params["blocks.norm3.b"] = jnp.asarray(stack(lambda i: state[f"blocks.{i}.norm2.bias"]),
                                         jnp.float32)
  params["blocks.mod"] = jnp.asarray(stack(lambda i: state[f"blocks.{i}.scale_shift_table"][0]),
                                     jnp.float32)
  # One array per layer, not stacked: indexing a stacked weight inside the
  # step copied each layer's slice (1.7 ms a step on one chip).
  for name in [k for k in params if k.startswith("blocks.")]:
    stacked = params.pop(name)
    for i in range(depth):
      params[f"blocks.{i}.{name[len('blocks.'):]}"] = stacked[i]
  return params, config


def param_specs(params):
  """Q/K/V/FFN-up columns and O/FFN-down rows over ``tensor``; the rest replicated.

  The q/k RMSNorms normalize over the whole 1536-wide projection, so their
  input is gathered (one small all-gather a block) rather than split by head.
  """
  specs = {}
  for k, v in params.items():
    name = k.split(".", 2)[2] if k.startswith("blocks.") else ""
    if name[:-2] in ("sa.q", "sa.k", "sa.v", "ca.q", "ca.k", "ca.v", "ffn.up"):
      specs[k] = P(None, "tensor") if name.endswith(".w") else P("tensor")
    elif name.endswith(".w") and name[:-2] in ("sa.o", "ca.o", "ffn.down"):
      specs[k] = P("tensor", None)
    else:
      specs[k] = P()
  return specs


# ---------------------------------------------------------------------------


def _linear(x, params, name, out_dtype=None):
  w = params[name + ".w"]
  y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=jnp.float32)
  y = y + params[name + ".b"]
  return y.astype(out_dtype or w.dtype)


def _heads(x, params, name, heads):
  """[B, S, D] -> [B, H, S, Dh] straight from the projection (no transpose)."""
  w = params[name + ".w"]
  w = w.reshape(w.shape[0], heads, -1)
  y = jnp.einsum("bsd,dhk->bhsk", x.astype(w.dtype), w, preferred_element_type=jnp.float32)
  return y + params[name + ".b"].reshape(heads, 1, -1)


def _unheads(x, params, name, out_dtype):
  """[B, H, S, Dh] -> [B, S, D] through the output projection."""
  w = params[name + ".w"]
  w = w.reshape(x.shape[1], -1, w.shape[-1])
  y = jnp.einsum("bhsk,hkd->bsd", x.astype(w.dtype), w, preferred_element_type=jnp.float32)
  return (y + params[name + ".b"]).astype(out_dtype)


def _rms_heads(x, w, eps):
  """RMSNorm over all features of a [B, H, S, Dh] tensor (the official norm is
  over the full projection width, not per head)."""
  xf = x.astype(jnp.float32)
  ms = jnp.mean(jnp.square(xf), axis=(1, 3), keepdims=True)
  return xf * jax.lax.rsqrt(ms + eps) * w.reshape(x.shape[1], 1, -1)


def _layer_norm(x, eps, w=None, b=None):
  xf = x.astype(jnp.float32)
  mean = xf.mean(-1, keepdims=True)
  var = jnp.square(xf - mean).mean(-1, keepdims=True)
  y = (xf - mean) * jax.lax.rsqrt(var + eps)
  if w is not None:
    y = y * w + b
  return y


def rope_table(length, head_dim, theta=10000.0):
  """(cos, sin) [length, head_dim // 2], float32 (``precompute_freqs_cis_1d``)."""
  freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
  angles = np.outer(np.arange(length, dtype=np.float64), freqs)
  return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rope(x, cos, sin):
  """x [B, H, S, D] with features as [evens | odds] (see ``load``); float32."""
  xf = x.astype(jnp.float32)
  even, odd = jnp.split(xf, 2, -1)
  c, s = cos[None, None], sin[None, None]
  return jnp.concatenate((even * c - odd * s, even * s + odd * c), -1)


def _attention(q, k, v):
  """[B, H, S, D] each -> [B, H, S, D]; float32 scores and softmax."""
  scale = 1.0 / np.sqrt(q.shape[-1])
  s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
  p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
  return jnp.einsum("bhqk,bhkd->bhqd", p, v, preferred_element_type=jnp.float32).astype(v.dtype)


def _blocks(sq, skv):
  """Splash tiles for these short sequences: the whole query in one block.

  Swept on a v6e (2 x 12 heads): self-attention 1500 x 1500 0.272 ms with
  the autotuner's 768/1280 against 0.206 ms at 1536/768; cross-attention
  1500 x 512 0.192 ms (768/512) against 0.125 ms (1536/512). 60 calls a step.
  """
  up = lambda n: -(-n // 128) * 128
  if sq > 2048:
    return block_sizes.resolve_block_sizes(sq, skv, 128, {}, "residual", block_sizes.detect_spec())
  bkv = 768 if skv > 1024 else up(skv)
  return {"block_q": up(sq), "block_kv": bkv, "block_kv_compute": bkv}


def make_attention(mesh, seq_parallel=False):
  """[B, H, S, D] attention through the splash kernel; no masks in this model.

  On a mesh the batch is over ``cfg``; the second axis splits the heads, or
  with ``seq_parallel`` the query rows (keys and values whole on every chip:
  the partitioner gathers them).
  """

  def local(q, k, v):
    return splash.attention(q, k, v, scale=None, **_blocks(q.shape[2], k.shape[2]))

  if mesh is not None:
    if seq_parallel:
      qspec, kvspec = P("cfg", None, "tensor", None), P("cfg", None, None, None)
    else:
      qspec = kvspec = P("cfg", "tensor", None, None)
    local = shard_map(local, mesh=mesh, in_specs=(qspec, kvspec, kvspec), out_specs=qspec,
                      check_rep=False)

  return local


def time_sinusoid(timestep, dim=256):
  """``sinusoidal_embedding_1d`` in float64 on the host -> float32 [B, dim]."""
  t = np.asarray(timestep, np.float64).reshape(-1)
  half = dim // 2
  sinusoid = np.outer(t, np.power(10000.0, -np.arange(half, dtype=np.float64) / half))
  return np.concatenate([np.cos(sinusoid), np.sin(sinusoid)], 1).astype(np.float32)


def dit(params, config, latents, sinusoid, context, cos, sin, *, mesh=None,
        attention=_attention, seq_parallel=False):
  """Velocity [B, 128, F] for latents [B, 128, F], text [B, 512, 2048]."""
  heads = config["num_heads"]
  dim = config["dim"]
  eps = config["eps"]
  spec = P("cfg", "tensor") if seq_parallel else P("cfg")
  stream = (lambda t: t) if mesh is None else (
      lambda t: jax.lax.with_sharding_constraint(t, NamedSharding(mesh, spec)))
  batch = latents.shape[0]

  t = _linear(sinusoid, params, "time1", jnp.float32)
  t = _linear(jax.nn.silu(t), params, "time2", jnp.float32)
  t_mod = _linear(jax.nn.silu(t), params, "tproj", jnp.float32).reshape(-1, 6, dim)
  ctx = _linear(context, params, "text1")
  ctx = _linear(jax.nn.gelu(ctx.astype(jnp.float32), approximate=True), params, "text2")
  x = _linear(latents.transpose(0, 2, 1), params, "patch").astype(jnp.float32)
  x = stream(x)

  def block(x, w):
    mod = w["mod"][None] + t_mod  # [B, 6, D] float32
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (mod[:, i][:, None] for i in range(6))
    bf = w["sa.v.w"].dtype

    y = _layer_norm(x, eps) * (1 + scale_a) + shift_a
    q = _rope(_rms_heads(_heads(y, w, "sa.q", heads), w["sa.norm_q"], eps), cos, sin).astype(bf)
    k = _rope(_rms_heads(_heads(y, w, "sa.k", heads), w["sa.norm_k"], eps), cos, sin).astype(bf)
    v = _heads(y, w, "sa.v", heads).astype(bf)
    x = x + gate_a * _unheads(attention(q, k, v), w, "sa.o", jnp.float32)

    y = _layer_norm(x, eps, w["norm3.w"], w["norm3.b"])
    q = _rms_heads(_heads(y, w, "ca.q", heads), w["ca.norm_q"], eps).astype(bf)
    k = _rms_heads(_heads(ctx, w, "ca.k", heads), w["ca.norm_k"], eps).astype(bf)
    v = _heads(ctx, w, "ca.v", heads).astype(bf)
    x = x + _unheads(attention(q, k, v), w, "ca.o", jnp.float32)

    y = _layer_norm(x, eps) * (1 + scale_f) + shift_f
    h = _linear(y, w, "ffn.up", jnp.float32)
    h = jax.nn.gelu(h, approximate=True)
    x = x + gate_f * _linear(h, w, "ffn.down", jnp.float32)
    return stream(x)

  for i in range(config["num_layers"]):
    pre = f"blocks.{i}."
    x = block(x, {k[len(pre):]: v for k, v in params.items() if k.startswith(pre)})

  head = params["head_mod"][None] + t[:, None]  # [B_t, 2, D]; B_t may be 1 for a CFG batch
  shift, scale = head[:, 0][:, None], head[:, 1][:, None]
  y = _layer_norm(x, eps) * (1 + scale) + shift
  out = _linear(y, params, "out", jnp.float32)
  return out.transpose(0, 2, 1)


def step(params, latents, sinusoid, context, cos, sin, dt, guidance, *, config, mesh=None,
         seq_parallel=False):
  """One CFG + Euler step; ``latents`` [1, 128, F] float32, ``context`` [B, 512, 2048]."""
  batch = context.shape[0]
  x = jnp.broadcast_to(latents, (batch,) + latents.shape[1:])
  if mesh is not None and batch > 1:
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("cfg")))
  attention = (_attention if jax.default_backend() == "cpu"
               else make_attention(mesh, seq_parallel))
  pred = dit(params, config, x, sinusoid, context, cos, sin, mesh=mesh, attention=attention,
             seq_parallel=seq_parallel)
  if batch > 1:
    pred = pred[1:2] + guidance * (pred[0:1] - pred[1:2])
  return latents + dt * pred


def make_step(transformer_dir, mesh, dtype=jnp.bfloat16, seq_parallel=False):
  """(params on ``mesh``, config, jitted ``step``).

  ``seq_parallel``: the mesh's second axis splits the sequence instead of the
  heads; the weights are then whole on every chip.
  """
  params, config = load(transformer_dir, dtype)
  if mesh is not None:
    specs = {k: P() for k in params} if seq_parallel else param_specs(params)
    params = {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in params.items()}
  fn = jax.jit(functools.partial(step, config=config, mesh=mesh, seq_parallel=seq_parallel))
  return params, config, fn


def _check(ref_path, transformer_dir):
  """One forward against the official ``model_fn_wan_video`` output (float32)."""
  ref = np.load(ref_path)
  jax.config.update("jax_default_matmul_precision", "highest")
  params, config = load(transformer_dir, jnp.float32)
  frames = ref["latents"].shape[-1]
  cos, sin = rope_table(frames, config["dim"] // config["num_heads"])
  out = dit(params, config, jnp.asarray(ref["latents"]), jnp.asarray(time_sinusoid(ref["timestep"])),
            jnp.asarray(ref["context"]), jnp.asarray(cos), jnp.asarray(sin))
  out, want = np.asarray(out, np.float64), ref["out"].astype(np.float64)
  rel = np.linalg.norm(out - want) / np.linalg.norm(want)
  print(f"DiT vs official: rel L2 {rel:.2e}, max |diff| {np.abs(out - want).max():.2e}")


if __name__ == "__main__":
  import argparse

  from diffusers_tpu.hub import local_snapshot

  p = argparse.ArgumentParser()
  p.add_argument("--check", required=True, help="dit.npz from the official code")
  a = p.parse_args()
  _check(a.check, os.path.join(local_snapshot("OpenMOSS-Team/MOSS-SoundEffect-v2.0"), "transformer"))
