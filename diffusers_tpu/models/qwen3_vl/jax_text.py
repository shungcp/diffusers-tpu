"""Qwen3-VL's language stack in JAX (flax nnx), used as a text encoder.

MiniMax-H3 reuses this tower to turn a prompt into the hidden states the
DiT is conditioned on. The defaults and the self-test below use the
Qwen-Image-2.1 checkpoint's text encoder config.

A plain Qwen3 decoder (36 layers, 4096 wide, 32 query heads over 8 KV
heads, SwiGLU, RMSNorm) with three additions:

* **Per-head RMSNorm on q and k**, over the head dimension only, applied
  after the projection and before RoPE.
* **Interleaved mRoPE.** Positions are three-dimensional (frame, height,
  width) and the frequencies are interleaved between the axes rather than
  sliced into contiguous blocks: with ``mrope_section = [24, 20, 20]``, slots
  ``1, 4, ... 58`` come from the height axis, ``2, 5, ... 59`` from width,
  and everything else (including all four above 59) from the frame axis.
  For a text-only prompt all three axes carry the same position, so this
  collapses to ordinary RoPE. ``rope_theta`` is 5e6, not the usual 1e4.
* **Deepstack injection.** Features from vision layers 8, 16 and 24 are
  added into the language stream at image-token positions, in language
  layers 0, 1 and 2 respectively. Text-only prompts never use it.

Run as an encoder, so unlike transformers there is no KV cache, and the
output is a layer's hidden states before the final RMSNorm.

Self-test against transformers:

    JAX_PLATFORMS=cpu python jax_text.py
"""

import glob
import math
import os

import numpy as np

# From text_config in the shipped text_encoder/config.json.
DEFAULTS = dict(num_layers=36, hidden_size=4096, head_dim=128, num_heads=32,
                num_kv_heads=8, intermediate_size=12288, vocab_size=151936,
                rms_norm_eps=1e-6, rope_theta=5_000_000.0,
                mrope_section=(24, 20, 20))


MODEL_ID = "Qwen/Qwen-Image-2.1"


def find_snapshot(model_id=MODEL_ID):
  """The local snapshot directory, downloading only what is missing.

  Goes through `huggingface_hub` rather than globbing a cache path, so it
  honours `HF_HOME` / `HF_HUB_CACHE` and works on a machine that has never
  seen this model.
  """
  from huggingface_hub import snapshot_download

  return snapshot_download(model_id)


# Text attention in float32 at highest precision (DT_TEXT_ATTN_F32, on by default).
TEXT_ATTN_F32 = os.environ.get("DT_TEXT_ATTN_F32", "1") == "1"
# Causal text attention a block of queries at a time (DT_TEXT_ATTN_CHUNK=<rows>),
# for one small chip and a long ref2va prompt.
TEXT_ATTN_CHUNK = int(os.environ.get("DT_TEXT_ATTN_CHUNK", "0"))
# Long causal prompts (ref2va's ~7.5k tokens) go through JAX's splash kernel,
# heads sharded over this mesh axis. The dense path materializes float32
# scores: text/vision encode 1.0 -> 0.6 s warm, video DINO 0.955 against it
# (c345; noise floor ~0.948). DT_TEXT_SPLASH=0 restores the dense path.
TEXT_SPLASH = os.environ.get("DT_TEXT_SPLASH", "1") == "1"
TEXT_SPLASH_MIN = int(os.environ.get("DT_TEXT_SPLASH_MIN", "2048"))
TEXT_SPLASH_AXIS = "tp"
TEXT_SPLASH_BLK = int(os.environ.get("DT_TEXT_SPLASH_BLK", "512"))
if TEXT_SPLASH:
  # Imported here, not inside the first trace: a Pallas import during
  # tracing changes the trace context and every later call retraces (c287).
  from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
      splash_attention_kernel as _sk_preload)


_SPLASH_KERNELS = {}


def _causal_kernel(n, g, blk):
  """One splash kernel per shape: building the mask info for a 7.7k causal
  mask is host work, and the 50 layers would otherwise each redo it while
  tracing (server's first ref2va request: text 7 -> 24 s, c346)."""
  key = (n, g, blk)
  if key not in _SPLASH_KERNELS:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm)
    import jax
    # Concrete even when first reached inside a trace: the kernel keeps its
    # mask arrays, and a traced copy cached here leaks into the next trace.
    with jax.ensure_compile_time_eval():
      _SPLASH_KERNELS[key] = sk.make_splash_mqa_single_device(
          sm.MultiHeadMask([sm.CausalMask((n, n))] * g),
          block_sizes=sk.BlockSizes(block_q=blk, block_kv=blk, block_kv_compute=blk))
  return _SPLASH_KERNELS[key]


def _splash_causal(q, k, v, scale):
  """Device-local causal GQA attention. q `(s, hq, d)`, k/v `(s, hkv, d)`."""
  import jax
  import jax.numpy as jnp
  s, hq, d = q.shape
  hkv = k.shape[1]
  g = hq // hkv
  blk = TEXT_SPLASH_BLK
  sp = (-s) % blk
  n = s + sp
  # Padding goes at the end: causal rows never reach a padded key.
  q, k, v = (jnp.pad(t, ((0, sp), (0, 0), (0, 0))) for t in (q, k, v))
  kernel = _causal_kernel(n, g, blk)
  qh = (q * scale).reshape(n, hkv, g, d).transpose(1, 2, 0, 3)   # (hkv, g, n, d)
  kh, vh = k.transpose(1, 0, 2), v.transpose(1, 0, 2)              # (hkv, n, d)
  out = jax.vmap(kernel)(qh, kh, vh)                              # (hkv, g, n, d)
  return out.transpose(2, 0, 1, 3).reshape(n, hq, d)[:s]


def _build_modules():
  """Imports flax lazily so the file can be read without it installed."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, *, rngs=None):
      self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
      self.eps = eps

    def __call__(self, x):
      x32 = x.astype(jnp.float32)
      variance = jnp.mean(x32**2, axis=-1, keepdims=True)
      return (self.weight[...] * (x32 * jax.lax.rsqrt(variance + self.eps))
              ).astype(x.dtype)

  class Linear(nnx.Module):
    """Bias-free; weight stored as torch does, (out, in)."""

    def __init__(self, in_f, out_f, *, dtype, rngs=None):
      self.weight = nnx.Param(jnp.zeros((out_f, in_f), dtype))
      self.weight_scale = nnx.data(None)

    def __call__(self, x):
      w = self.weight[...]
      if w.dtype == jnp.int8 and self.weight_scale is not None:
        # Weight-only int8 (MiniMax-H3 `jax_text.load(int8=True)`).
        y = x @ w.astype(x.dtype).T
        return (y.astype(jnp.float32) * self.weight_scale[...]).astype(x.dtype)
      return x @ w.T

  def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

  def apply_rope(x, cos, sin):
    """`x` is [batch, seq, heads, head_dim]; cos/sin are [seq, head_dim]."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + rotate_half(x) * sin

  class Attention(nnx.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, eps, *, dtype, rngs):
      self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
      self.scale = head_dim**-0.5
      self.q_proj = Linear(hidden, heads * head_dim, dtype=dtype, rngs=rngs)
      self.k_proj = Linear(hidden, kv_heads * head_dim, dtype=dtype, rngs=rngs)
      self.v_proj = Linear(hidden, kv_heads * head_dim, dtype=dtype, rngs=rngs)
      self.o_proj = Linear(heads * head_dim, hidden, dtype=dtype, rngs=rngs)
      # Over the head dimension only, so no reshape is needed afterwards.
      self.q_norm = RMSNorm(head_dim, eps)
      self.k_norm = RMSNorm(head_dim, eps)

    def __call__(self, x, cos, sin, mask=None):
      b, s, _ = x.shape
      q = self.q_norm(self.q_proj(x).reshape(b, s, self.heads, self.head_dim))
      k = self.k_norm(self.k_proj(x).reshape(b, s, self.kv_heads, self.head_dim))
      v = self.v_proj(x).reshape(b, s, self.kv_heads, self.head_dim)
      q = apply_rope(q, cos, sin).astype(v.dtype)
      k = apply_rope(k, cos, sin).astype(v.dtype)
      # jax's attention does grouped queries itself when the head counts
      # differ, so no explicit repeat.
      if TEXT_SPLASH and mask is None and b == 1 and s >= TEXT_SPLASH_MIN:
        from jax.sharding import PartitionSpec as P
        spec = P(None, TEXT_SPLASH_AXIS, None)
        local = lambda a, c, e: _splash_causal(a, c, e, self.scale)
        out = jax.shard_map(local, in_specs=(spec, spec, spec), out_specs=spec,
                            check_vma=False)(q[0], k[0], v[0])[None]
      elif TEXT_ATTN_CHUNK and mask is None and s > TEXT_ATTN_CHUNK:
        # Causal attention a block of queries at a time (DT_TEXT_ATTN_CHUNK):
        # one 16 GB v5e chip cannot hold the 64 x 7518^2 float32 scores of a
        # ref2va prompt (27.5 GB of temporaries in the whole-tower program).
        c = TEXT_ATTN_CHUNK
        n = -(-s // c)
        qp = jnp.pad(q, ((0, 0), (0, n * c - s), (0, 0), (0, 0)))
        qc = qp.reshape(b, n, c, self.heads, self.head_dim).swapaxes(0, 1)
        cols = jnp.arange(s)[None, :]
        dt = jnp.float32 if TEXT_ATTN_F32 else v.dtype
        kf, vf = k.astype(dt), v.astype(dt)

        def one(args):
          qi, start = args
          rows = start + jnp.arange(c)[:, None]
          m = (cols <= rows)[None, None]
          with jax.default_matmul_precision("highest" if TEXT_ATTN_F32 else "default"):
            return jax.nn.dot_product_attention(qi.astype(dt), kf, vf, mask=m,
                                                scale=self.scale)
        out = jax.lax.map(one, (qc, jnp.arange(n) * c))
        out = out.swapaxes(0, 1).reshape(b, n * c, self.heads, self.head_dim)[:, :s]
        out = out.astype(v.dtype)
      elif TEXT_ATTN_F32:
        # float32 at highest precision: the bfloat16 path loses accuracy on
        # long prompts (7,518 tokens with a ref2va image) at the same speed.
        with jax.default_matmul_precision("highest"):
          out = jax.nn.dot_product_attention(
              q.astype(jnp.float32), k.astype(jnp.float32),
              v.astype(jnp.float32), mask=mask, is_causal=mask is None,
              scale=self.scale).astype(v.dtype)
      else:
        out = jax.nn.dot_product_attention(q, k, v, mask=mask,
                                           is_causal=mask is None,
                                           scale=self.scale)
      return self.o_proj(out.reshape(b, s, -1))

  class MLP(nnx.Module):
    def __init__(self, hidden, intermediate, *, dtype, rngs):
      self.gate_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.up_proj = Linear(hidden, intermediate, dtype=dtype, rngs=rngs)
      self.down_proj = Linear(intermediate, hidden, dtype=dtype, rngs=rngs)

    def __call__(self, x):
      return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

  class DecoderLayer(nnx.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, intermediate, eps,
                 *, dtype, rngs):
      self.input_layernorm = RMSNorm(hidden, eps)
      self.self_attn = Attention(hidden, heads, kv_heads, head_dim, eps,
                                 dtype=dtype, rngs=rngs)
      self.post_attention_layernorm = RMSNorm(hidden, eps)
      self.mlp = MLP(hidden, intermediate, dtype=dtype, rngs=rngs)

    def __call__(self, x, cos, sin, mask=None):
      x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
      return x + self.mlp(self.post_attention_layernorm(x))

  class TextModel(nnx.Module):
    """Qwen3-VL's language model, run as an encoder.

    Returns hidden states *before* `norm`. The final RMSNorm is loaded so
    the module round-trips the checkpoint, but `__call__` does not apply
    it: the DiT conditions on the unnormalised output, and transformers 5.x
    ties `hidden_states[-1]` to the normalised one (upstream works around
    this with a forward hook).
    """

    def __init__(self, num_layers=36, hidden_size=4096, head_dim=128,
                 num_heads=32, num_kv_heads=8, intermediate_size=12288,
                 vocab_size=151936, rms_norm_eps=1e-6, rope_theta=5e6,
                 mrope_section=(24, 20, 20), *, dtype, rngs):
      self.embed_tokens = nnx.Param(jnp.zeros((vocab_size, hidden_size), dtype))
      self.layers = nnx.List([
          DecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim,
                       intermediate_size, rms_norm_eps, dtype=dtype, rngs=rngs)
          for _ in range(num_layers)])
      self.norm = RMSNorm(hidden_size, rms_norm_eps)

    def __call__(self, input_ids, cos, sin, mask=None, inputs_embeds=None,
                 deepstack=None, visual_positions=None, stop_at_layer=None,
                 scan_layers=None):
      """`visual_positions` is an int index array, not a boolean mask:
      boolean indexing needs a concrete count, which a traced function
      does not have.

      `stop_at_layer=n` returns after `n` decoder layers, i.e.
      transformers' `hidden_states[n]`. MiniMax-H3 conditions on
      `hidden_states[50]` of its 64-layer tower (see its pipeline's
      `components.text_encoder_layer`); running all 64 layers gives
      visibly wrong conditioning. `None` runs every layer.
      """
      x = self.embed_tokens[...][input_ids] if inputs_embeds is None \
          else inputs_embeds
      # `scan_layers=(layer_graph, stacked_states, start)`: layers from
      # `start` up to `stop_at_layer` run as one `scan` over their stacked
      # weights (one layer to compile instead of ~48; the H3-Image server).
      start = scan_layers[2] if scan_layers is not None else None
      for index, layer in enumerate(self.layers):
        if stop_at_layer is not None and index >= stop_at_layer:
          break
        if start is not None and index >= start:
          from flax import nnx as _nnx
          graph, stacked = scan_layers[0], scan_layers[1]
          x, _ = jax.lax.scan(
              lambda h, st: (_nnx.merge(graph, st)(h, cos, sin, mask), None), x, stacked)
          break
        x = layer(x, cos, sin, mask)
        if deepstack is not None and index < len(deepstack):
          # Add the vision features back in at the image-token positions.
          x = x.at[:, visual_positions].add(deepstack[index].astype(x.dtype))
      return x

  return TextModel


# --------------------------------------------------------------------------
# host prologue: interleaved mRoPE
# --------------------------------------------------------------------------


def mrope_cos_sin(position_ids, head_dim=128, rope_theta=5_000_000.0,
                  mrope_section=(24, 20, 20)):
  """Build the rotary tables for one sequence, on the host.

  Args:
    position_ids: `(3, seq)` -- the frame, height and width position of every
      token. A text-only prompt passes the same `arange` three times.

  Returns:
    `cos`, `sin`, each `(seq, head_dim)`.

  Each axis gets its own
  `freqs` over all `head_dim / 2` slots, and then the slots are handed out
  round-robin: slot `3k + 1` to height and `3k + 2` to width for as many `k`
  as that axis's section allows, everything else to frame. The result is
  duplicated, not interleaved, into the full `head_dim` -- `rotate_half`
  pairs `i` with `i + head_dim / 2`.
  """
  position_ids = np.asarray(position_ids, np.float64)
  if position_ids.shape[0] != 3:
    raise ValueError(f"expected (3, seq) positions, got {position_ids.shape}")
  inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2) / head_dim))
  half = head_dim // 2
  slot_axis = np.zeros(half, np.int64)
  for axis, offset in ((1, 1), (2, 2)):
    slot_axis[offset:mrope_section[axis] * 3:3] = axis
  # Positions repeat (a ref2va prompt has ~7.5k tokens but a few hundred
  # distinct values): take cos/sin of each distinct value once and gather.
  # The same float64 products, so the tables are unchanged.
  uniq, inv = np.unique(position_ids, return_inverse=True)
  inv = inv.reshape(position_ids.shape)                       # (3, seq)
  angles = uniq[:, None] * inv_freq[None, :]                  # (U, dim/2)
  rows = inv[slot_axis].T                                     # (seq, dim/2)
  cols = np.arange(half)[None, :]
  cos = np.cos(angles).astype(np.float32)[rows, cols]
  sin = np.sin(angles).astype(np.float32)[rows, cols]
  return np.concatenate([cos, cos], -1), np.concatenate([sin, sin], -1)


# --------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------

_PREFIX = "model.language_model."


def _resolve(model, path):
  obj = model
  for part in path:
    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
  return obj


def load_text_model(model, text_encoder_dir, mesh=None, spec_fn=None):
  """Read the language half of the checkpoint into the nnx module.

  Skips `lm_head` -- nothing here generates tokens -- and the vision tower,
  which lives in its own module.
  """
  import jax
  from flax import nnx
  from safetensors import safe_open
  from jax.sharding import NamedSharding

  shards = sorted(glob.glob(os.path.join(text_encoder_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {text_encoder_dir}")

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if not key.startswith(_PREFIX):
          continue                      # lm_head, model.visual.*
        name = key[len(_PREFIX):]
        if name == "embed_tokens.weight":
          param = model.embed_tokens
        else:
          param = _resolve(model, name.split("."))
        value = reader.get_tensor(key).astype(param[...].dtype)
        if value.shape != param[...].shape:
          raise ValueError(
              f"{key}: checkpoint {value.shape} vs model {param[...].shape}")
        if mesh is not None and spec_fn is not None:
          value = jax.device_put(
              value, NamedSharding(mesh, spec_fn(name.split("."), value.ndim)))
        param[...] = value
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return model


_COLUMN = frozenset({"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"})
_ROW = frozenset({"o_proj", "down_proj"})


def param_spec(keys, ndim):
  """Tensor-parallel layout, matching the DiT's conventions."""
  from jax.sharding import PartitionSpec as P

  parent = next((k for k in reversed(keys)
                 if k not in ("weight", "value", ".value")), "")
  if ndim == 2 and parent in _COLUMN:
    return P("tensor", None)
  if ndim == 2 and parent in _ROW:
    return P(None, "tensor")
  return P()


# --------------------------------------------------------------------------
# numerical equivalence against transformers
# --------------------------------------------------------------------------


def main():
  import jax.numpy as jnp
  import torch
  from flax import nnx
  from transformers import Qwen3VLForConditionalGeneration

  text_dir = os.path.join(find_snapshot(), "text_encoder")

  TextModel = _build_modules()
  model = TextModel(**DEFAULTS, dtype=jnp.float32, rngs=nnx.Rngs(0))
  load_text_model(model, text_dir)

  torch_model = Qwen3VLForConditionalGeneration.from_pretrained(
      text_dir, dtype=torch.float32).eval()

  rng = np.random.default_rng(0)
  seq = 24
  input_ids = rng.integers(0, 150000, size=(1, seq))
  positions = np.tile(np.arange(seq), (3, 1))  # text-only: all axes agree
  cos, sin = mrope_cos_sin(positions, DEFAULTS["head_dim"],
                           DEFAULTS["rope_theta"], DEFAULTS["mrope_section"])

  # transformers returns hidden_states[-1] tied to the normalised output, so
  # neutralise the final norm the way the diffusers pipeline does.
  language = torch_model.model.language_model
  handle = language.norm.register_forward_hook(lambda m, a, out: a[0])
  try:
    with torch.no_grad():
      ref = torch_model(input_ids=torch.tensor(input_ids),
                        attention_mask=torch.ones((1, seq), dtype=torch.long),
                        output_hidden_states=True).hidden_states[-1].numpy()
  finally:
    handle.remove()

  got = np.asarray(model(jnp.asarray(input_ids), jnp.asarray(cos),
                         jnp.asarray(sin)))
  diff = np.abs(ref - got)
  rel = diff.max() / max(np.abs(ref).max(), 1e-9)
  print(f"hidden {ref.shape}  max|diff| {diff.max():.3e}  relative {rel:.3e}  "
        f"mean|diff| {diff.mean():.3e}")
  ok = rel < 1e-4
  print("MATCH" if ok else "MISMATCH")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
