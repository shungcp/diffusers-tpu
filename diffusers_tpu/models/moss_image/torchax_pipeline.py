"""MOSS-Image text-to-image on TPU via torchax.

The upstream modules run unchanged under torchax; what changes is who drives
them:

* the denoising loop is ours (``Generator``): one jitted program per step
  (``torchax_step.StepCore``) runs the DiT on [positive, negative] as one
  batch, combines CFG and takes the Euler step, with no host sync until the
  decode. On a ``('cfg', 'tensor')`` mesh with ``--cfg 2`` the two branches
  run side by side on two halves of the chips. ``--loop upstream`` drives the
  upstream ``MOSSImagePipeline.__call__`` instead, with ``transformer``
  swapped for ``DiTPrologue`` (one call per branch), as a reference.
* ``text_encoder`` -> ``TextSplit``: the Qwen3.5-9B embedding lookup and RoPE
  tables on the host, the 32 decoder layers (24 Gated DeltaNet, 8 attention)
  plus the final norm on device. The ids are right-padded to a bucket so
  prompts of similar length share one program; the model is causal, so the
  valid rows do not see the padding and are returned unchanged.
* ``vae.decode`` -> one compiled program (``post_quant_conv`` + decoder).

    PYTHONPATH=. python -m diffusers_tpu.models.moss_image.torchax_pipeline \\
        --prompt "a red fox in the snow" --size 1024 --runs 2
"""
import argparse
import contextlib
import math
import os
import re
import time
from types import SimpleNamespace

import jax
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import torchax

from diffusers_tpu.models.moss_image import remote
from diffusers_tpu.models.moss_image.torchax_dit import (
    DiTCore, rope_table, split_up_gate, time_sinusoid)
from diffusers_tpu.models.moss_image.torchax_step import StepCore, make_masked_splash

# At tensor=4: 24 heads -> 6 per chip, FFN 8192 -> 2048. The fused
# ``ffn_up_gate`` is split into ``ffn_up`` and ``ffn_gate`` first
# (``split_up_gate``); ``--fused-ffn`` keeps it, and its row split then moves
# the gate between chips.
DIT_SHARDINGS = {
    r".*layers\.\d+\.proj\.(q|k|v)\.weight": ("tensor", None),
    r".*layers\.\d+\.proj\.o\.weight": (None, "tensor"),
    r".*layers\.\d+\.proj\.(ffn_up_gate|ffn_up|ffn_gate)\.weight": ("tensor", None),
    r".*layers\.\d+\.proj\.ffn_down\.weight": (None, "tensor"),
}

# Qwen3.5-9B: the MLPs (12288 intermediate) and the 8 attention layers'
# projections (16 query heads with an output gate, 4 key/value heads), over
# every chip (both mesh axes). The Gated DeltaNet layers stay replicated for
# now: their q/k/v share one projection and one depthwise conv over the
# concatenation.
_ALL = ("cfg", "tensor")
TEXT_SHARDINGS = {
    r".*layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight": (_ALL, None),
    r".*layers\.\d+\.mlp\.down_proj\.weight": (None, _ALL),
    r".*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight": (_ALL, None),
    r".*layers\.\d+\.self_attn\.o_proj\.weight": (None, _ALL),
}


@contextlib.contextmanager
def torchax_paused():
  torchax.disable_globally()
  try:
    yield
  finally:
    torchax.enable_globally()


def _shard(weights, rules, mesh):
  for name, tensor in weights.items():
    for pattern, spec in rules.items():
      if re.fullmatch(pattern, name):
        tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P(*spec)))
        break
    else:
      tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
  return weights


def _move_module(module, env):
  """Weights (and non-persistent buffers) onto the default device."""
  with jax.default_device(jax.devices("cpu")[0]):
    state = env.to_xla(module.state_dict())
    module.load_state_dict(state, assign=True)
  from torchax.interop import extract_all_buffers

  _, buffers = extract_all_buffers(module)
  for qual, tensor in buffers.items():
    if hasattr(tensor, "apply_jax_"):
      continue
    *prefix, attr = qual.split(".")
    owner = module
    for p in prefix:
      owner = getattr(owner, p)
    try:
      setattr(owner, attr, env.to_xla(tensor))
    except Exception:  # pylint: disable=broad-except
      pass


def _host(env, t):
  import torchax.tensor as tx

  if isinstance(t, tx.Tensor):
    return env.j2t_copy(t._elem)
  return t


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


def make_splash_sdpa(mesh, env, use_fixed_m=False, block_q=None, block_kv=None):
  """SDPA for the joint self-attention: the custom splash kernel per chip.

  Heads are split over the mesh and the output stays split by heads, which
  is the layout ``proj['o']`` (sharded on its contraction) consumes: the
  block's only exchange is then the all-reduce after ``o``. The kernel pads
  the sequence (4096 image rows + a prompt's length) itself.
  """
  import functools

  import jax.numpy as jnp
  from jax.experimental.shard_map import shard_map

  from diffusers_tpu.kernels import block_sizes
  from diffusers_tpu.kernels import splash

  spec = P(None, "tensor", None, None)
  pinned = {"block_q": block_q, "block_kv": block_kv, "block_kv_compute": block_kv}
  mode = "residual-fixedm" if use_fixed_m else "residual"
  hw = block_sizes.detect_spec()
  resolved = {}

  def sdpa(query, key, value):
    jq, jk, jv = env.t2j_iso((query, key, value))
    shape = (jq.shape[2], jk.shape[2], jq.shape[-1])
    if shape not in resolved:
      resolved[shape] = block_sizes.resolve_block_sizes(*shape, pinned, mode, hw)
      print(f"  [splash] seq={shape[0]} -> {resolved[shape]}", flush=True)
    # k-smoothing: softmax-invariant; the fixed-m softmax relies on it.
    jk = jk - jnp.mean(jk, axis=2, keepdims=True)
    local = functools.partial(splash.attention, scale=None, use_fixed_m=use_fixed_m,
                              **resolved[shape])
    out = shard_map(local, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec,
                    check_rep=False)(jq, jk, jv)
    return env.j2t_iso(out)

  return sdpa


class DiTPrologue:
  """Stands in for ``pipe.transformer`` under ``--loop upstream``: one jitted
  ``DiTCore`` call per sample, the text padding stripped on the host."""

  def __init__(self, model, core, env):
    self._model = model
    self._core = core
    self._env = env
    self._ropes = {}
    self._lengths = {}

  def __getattr__(self, name):  # .config, .layers, ...
    return getattr(self._model, name)

  def _lengths_of(self, mask):
    key = id(mask)
    hit = self._lengths.get(key)
    if hit is None or hit[0] is not mask:
      lengths = _host(self._env, mask).sum(-1).tolist()
      hit = (mask, [int(n) for n in lengths])
      self._lengths[key] = hit
    return hit[1]

  def _rope(self, height, width, length):
    key = (height, width, length)
    if key not in self._ropes:
      with torchax_paused():
        table = rope_table(self._model, height, width, length)
      self._ropes[key] = self._env.to_xla(table)
    return self._ropes[key]

  def __call__(self, hidden_states, timestep, encoder_hidden_states,
               encoder_attention_mask=None, return_dict=True, skip_layers=None):
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch, _, height, width = hidden_states.shape
    if encoder_attention_mask is None:
      lengths = [encoder_hidden_states.shape[1]] * batch
    else:
      lengths = self._lengths_of(encoder_attention_mask)
    with torchax_paused():
      sinusoid = time_sinusoid(self._model, _host(self._env, timestep).float())
    sinusoid = self._env.to_xla(sinusoid)
    skip = tuple(sorted(skip_layers or ()))
    outs = []
    for i in range(batch):
      context = encoder_hidden_states[i:i + 1, :lengths[i]]
      outs.append(self._core(hidden_states[i:i + 1], sinusoid, context,
                             self._rope(height, width, lengths[i]), skip_layers=skip))
    sample = outs[0] if batch == 1 else torch.cat(outs)
    return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def _drain(steps):
  """Runs a ``denoise_steps`` iterator to the end; its return value."""
  while True:
    try:
      next(steps)
    except StopIteration as done:
      return done.value


def run_concurrently(gens, requests, **common):
  """One request per generator, on disjoint meshes, overlapped.

  Each generator's steps are dispatched in turn -- step 0 of every request,
  then step 1, ... -- and nothing waits until all are dispatched, so the
  meshes run side by side. Returns the uint8 images.
  """
  iters = []
  for gen, req in zip(gens, requests):
    gen.encode(req["prompt"])
    if common.get("guidance_scale", 3.0) != 1.0:
      gen.encode(common.get("negative_prompt", ""))
    kwargs = dict(common, seed=req.get("seed", 42))
    iters.append(gen.denoise_steps(req["prompt"], **kwargs))
  results = [None] * len(iters)
  live = list(range(len(iters)))
  while live:
    for i in list(live):
      try:
        next(iters[i])
      except StopIteration as done:
        results[i] = done.value
        live.remove(i)
  return [gen.decode(latents) for gen, latents in zip(gens, results)]


class Generator:
  """Text-to-image with our own loop: encode, ``steps`` jitted steps, decode."""

  def __init__(self, pipe, model, step, env, mesh, text_bucket=64):
    self.pipe = pipe
    self._model = model
    self._step = step
    self._env = env
    self._mesh = mesh
    self._bucket = text_bucket
    self._prompts = {}
    self.times = {}
    self.jax_decoder = None  # vae_dir: decode with jax_vae (decoders made lazily per dtype)
    self.vae_dtype = "auto"
    self._decoders = {}
    self._dec_seen, self._dec_threads = set(), {}

  def _put(self, array, spec):
    return self._env.j2t_iso(jax.device_put(array, NamedSharding(self._mesh, spec)))

  def _put_torch(self, t, spec=P()):
    """A host torch tensor onto this generator's mesh (bf16 stays bf16)."""
    import jax.numpy as jnp

    if t.dtype == torch.bfloat16:
      array = jax.device_put(t.view(torch.int16).numpy(), NamedSharding(self._mesh, spec))
      return self._env.j2t_iso(jax.lax.bitcast_convert_type(array, jnp.bfloat16))
    return self._put(t.numpy(), spec)

  def encode(self, prompt):
    """(embeds [1, bucket, 4096] on device, true length L), cached per prompt.

    The pipeline's ``encode_prompt`` formatting, but the encoder's padded
    output is kept as is: slicing it to each prompt's exact length (and the
    pipeline's zero-fill) are eager ops that compile once per new length,
    0.3 s a request. Rows past ``L`` are masked downstream and never read.
    """
    hit = self._prompts.get(prompt)
    if hit is None:
      tokenizer = self.pipe.tokenizer
      formatted = tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}], tokenize=False,
          add_generation_prompt=True, enable_thinking=True)
      ids = tokenizer(formatted, return_tensors="pt", truncation=True,
                      max_length=4096).input_ids
      text = self.pipe.text_encoder
      embeds, cos, sin, length = text.host_inputs(ids)
      hit = (text._core(*(self._put_torch(t) for t in (embeds, cos, sin))), length)
      self._prompts[prompt] = hit
    return hit

  def _inputs(self, prompts, height, width, encoded=None):
    """Padded contexts, per-sample RoPE, valid lengths; batch on ``cfg``."""
    if encoded is None:
      encoded = [self.encode(p) for p in prompts]
    longest = max(n for _, n in encoded)
    padded = math.ceil(longest / self._bucket) * self._bucket
    rows = []
    for embeds, _ in encoded:
      if embeds.shape[1] < padded:  # a shorter prompt's encoder bucket
        embeds = torch.cat([embeds, embeds.new_zeros(
            1, padded - embeds.shape[1], embeds.shape[-1])], 1)
      rows.append(embeds[:, :padded])
    context = torch.cat(rows) if len(rows) > 1 else rows[0]
    batch_spec = P("cfg") if len(rows) > 1 else P()
    context = self._env.j2t_iso(jax.device_put(
        self._env.t2j_iso(context), NamedSharding(self._mesh, batch_spec)))
    with torchax_paused():
      rope = torch.cat([rope_table(self._model, height // 16, width // 16, n, padded)
                        for _, n in encoded])
    n_img = (height // 16) * (width // 16)
    kv_len = np.array([n_img + n for _, n in encoded], np.int32)
    return (context, self._put(rope.numpy(), batch_spec),
            self._put(kv_len, batch_spec))

  def denoise_steps(self, prompt, negative_prompt="", height=1024, width=1024, steps=50,
                    guidance_scale=3.0, uncond_skip_layers=(22,), seed=42):
    """Yields after dispatching each step; returns the final latents."""
    cfg = guidance_scale != 1.0
    prompts = [prompt, negative_prompt] if cfg else [prompt]
    context, rope, kv_len = self._inputs(prompts, height, width)
    skip = self._put(np.array([False, True][:len(prompts)]),
                     P("cfg") if cfg else P())
    skip_layers = tuple(sorted(uncond_skip_layers or ())) if cfg else ()
    scheduler = self.pipe.scheduler
    scheduler.set_timesteps(steps, device="cpu")
    timesteps = scheduler.timesteps.float()
    sigmas = scheduler.sigmas.float()
    with torchax_paused():
      sinusoids = [time_sinusoid(self._model, t).numpy() for t in timesteps]
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, 128, height // 16, width // 16), generator=gen,
                          dtype=torch.float32)
    latents = self._put(latents.numpy(), P())
    guidance = self._put(np.float32(guidance_scale), P())
    for i in range(len(timesteps)):
      dt = self._put(np.float32(sigmas[i + 1] - sigmas[i]), P())
      latents = self._step(latents, self._put(sinusoids[i], P()), context, rope,
                           kv_len, skip, dt, guidance, skip_layers=skip_layers)
      yield
    return latents

  def denoise(self, *args, **kwargs):
    """The latents after every step; dispatched asynchronously, not waited on."""
    return _drain(self.denoise_steps(*args, **kwargs))

  def _decoder_for(self, shape):
    """(params, jitted decoder) for latents of ``shape`` (dtype by size)."""
    import jax.numpy as jnp

    from diffusers_tpu.models.moss_image import jax_vae

    dtype = self.vae_dtype
    if dtype == "auto":
      # fp32 like upstream, unless one chip's share of the image passes
      # 2 MP: 1 chip at 2048x2048 needs 9.5 GiB of fp32 scratch beside the
      # resident models (8.8 free). bf16 needs 5.5 (57 dB PSNR to fp32).
      pixels = shape[-1] * shape[-2] * 256 / self._mesh.devices.size
      dtype = "bf16" if pixels > 2 * 1024 * 1024 else "fp32"
    if dtype not in self._decoders:
      self._decoders[dtype] = jax_vae.make_decoder(
          self.jax_decoder, mesh=self._mesh,
          dtype=jnp.bfloat16 if dtype == "bf16" else jnp.float32)
    return self._decoders[dtype]

  def _token_length(self, prompt):
    lengths = self.__dict__.setdefault("_token_lengths", {})
    if prompt in lengths:
      return lengths[prompt]
    tokenizer = self.pipe.tokenizer
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    n = int(tokenizer(formatted, return_tensors="pt", truncation=True,
                      max_length=4096).input_ids.shape[1])
    lengths[prompt] = n
    return n

  def _precompile_step(self, prompt, kwargs):
    """Compile the denoising step for a new (size, text bucket) in a thread,
    beside the text encoder, from the token counts alone. Compile only: the
    same program is lowered from the step module's functional call and
    written to the compilation cache, nothing runs on the chips, so a busy
    server is not interrupted. `DT_MOSS_PAR_STEP=0` turns it off."""
    import os
    import threading

    if os.environ.get("DT_MOSS_PAR_STEP", "1") != "1" or type(self) is not Generator:
      return None
    height, width = kwargs.get("height", 1024), kwargs.get("width", 1024)
    gs = kwargs.get("guidance_scale", 3.0)
    cfg = gs != 1.0
    prompts = [prompt, kwargs.get("negative_prompt", "")] if cfg else [prompt]
    lengths = [self._token_length(p) for p in prompts]
    padded = math.ceil(max(lengths) / self._bucket) * self._bucket
    skip_layers = tuple(sorted(kwargs.get("uncond_skip_layers", (22,)) or ())) if cfg else ()
    key = (height, width, padded, cfg, skip_layers)   # the program's shapes only
    seen = self.__dict__.setdefault("_step_seen", set())
    if key in seen:
      return None
    seen.add(key)
    zeros = self._put_torch(torch.zeros(1, padded, 4096, dtype=torch.bfloat16))
    context, rope, kv_len = self._inputs(prompts, height, width,
                                         encoded=[(zeros, n) for n in lengths])
    skip = self._put(np.array([False, True][:len(prompts)]), P("cfg") if cfg else P())
    scheduler = self.pipe.scheduler
    scheduler.set_timesteps(kwargs.get("steps", 50), device="cpu")
    with torchax_paused():
      sinusoid = time_sinusoid(self._model, scheduler.timesteps.float()[0]).numpy()
    latents = self._put(np.zeros((1, 128, height // 16, width // 16), np.float32), P())
    args = (latents, self._put(sinusoid, P()), context, rope, kv_len, skip,
            self._put(np.float32(-0.02), P()), self._put(np.float32(gs), P()))

    core = self._step
    if not hasattr(core, "functional_call"):   # e.g. the native JAX step
      return None
    import functools

    from torchax.interop import jax_view
    jx = lambda t: t._elem if hasattr(t, "_elem") else t
    fn = jax.jit(jax_view(functools.partial(core.functional_call, "forward")),
                 **(core._extra_jit_args or {}))
    params = jax.tree.map(jx, core.params)
    buffers = jax.tree.map(jx, core.buffers)
    jargs = tuple(jx(a) for a in args)

    def run():
      fn.lower(params, buffers, *jargs, skip_layers=skip_layers).compile()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th

  def _precompile_decode(self, height, width):
    """Compile the decoder for a new image size in a thread, beside the text
    encoder and the denoising steps (it depends only on the size); the
    decode then finds it in the compilation cache. `DT_MOSS_PAR_DECODE=0`
    turns it off."""
    import os
    import threading

    shape = (1, 128, height // 16, width // 16)
    if (self.jax_decoder is None or shape in self._dec_seen
        or os.environ.get("DT_MOSS_PAR_DECODE", "1") != "1"):
      return
    self._dec_seen.add(shape)
    params, fn = self._decoder_for(shape)
    aval = jax.ShapeDtypeStruct(shape, np.float32, sharding=NamedSharding(self._mesh, P()))

    def run():
      fn.lower(params, aval).compile()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    self._dec_threads[shape] = th

  def decode(self, latents):
    """The pipeline's decode: BN de-normalisation, 2x2 unpatchify, VAE.

    uint8 [H, W, 3]. With ``jax_decoder`` set the whole of it is one native
    JAX program (``jax_vae.py``); otherwise the diffusers VAE, traced.
    """
    if self.jax_decoder is not None:
      import jax.numpy as jnp

      from diffusers_tpu.models.moss_image import jax_vae

      params, fn = self._decoder_for(latents.shape)
      th = self._dec_threads.pop(tuple(latents.shape), None)
      if th is not None:
        th.join()
      return np.asarray(fn(params, self._env.t2j_iso(latents)))[0]
    vae = self.pipe.vae
    z = latents.to(dtype=vae.dtype)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    image = self.pipe.image_processor.postprocess(images, output_type="np")[0]
    return (image * 255).round().clip(0, 255).astype(np.uint8)

  def __call__(self, prompt, **kwargs):
    def sync(t):
      jax.block_until_ready(t._elem if hasattr(t, "_elem") else t)
      return t

    t0 = time.perf_counter()
    self._precompile_decode(kwargs.get("height", 1024), kwargs.get("width", 1024))
    step_th = self._precompile_step(prompt, kwargs)
    self.encode(prompt)
    if kwargs.get("guidance_scale", 3.0) != 1.0:
      self.encode(kwargs.get("negative_prompt", ""))
    sync(self._prompts[prompt][0])
    if step_th is not None:
      step_th.join()
    t1 = time.perf_counter()
    latents = sync(self.denoise(prompt, **kwargs))
    t2 = time.perf_counter()
    image = self.decode(latents)
    t3 = time.perf_counter()
    self.times = {"text": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return image


class DMDGenerator(Generator):
  """The 8-step DMD sampler (``dmd.py``): no CFG, one prompt, ``steps`` <= 8."""

  def __init__(self, *args, mix, codes, **kwargs):
    super().__init__(*args, **kwargs)
    self._mix = mix
    self._codes = codes

  def denoise_steps(self, prompt, negative_prompt="", height=1024, width=1024, steps=8,
                    guidance_scale=1.0, uncond_skip_layers=(), seed=42):
    del negative_prompt, guidance_scale, uncond_skip_layers  # no CFG
    steps = min(steps, len(self._codes))  # every prefix of the schedule is trained
    context, rope, kv_len = self._inputs([prompt], height, width)
    skip = self._put(np.array([False]), P())
    with torchax_paused():
      sinusoids = [time_sinusoid(self._model, t).numpy() for t in self._codes[:steps]]
    gen = torch.Generator("cpu").manual_seed(seed)
    z = torch.randn((1, 128, height // 16, width // 16), generator=gen, dtype=torch.float32)
    z = self._put(z.numpy(), P())
    x0 = self._put(np.zeros(z.shape, np.float32), P())
    mix = self._put(self._mix.numpy(), P())
    for i in range(steps):
      x0 = self._step(z, x0, mix, self._put(sinusoids[i], P()), context, rope, kv_len, skip)
      yield
    return x0

  def __call__(self, prompt, **kwargs):
    kwargs["guidance_scale"] = 1.0  # so the base class skips the negative prompt
    return super().__call__(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Qwen3.5 text encoder
# ---------------------------------------------------------------------------


class TextCore(torch.nn.Module):
  """The decoder layers and final norm, causal, without a cache."""

  def __init__(self, language_model):
    super().__init__()
    self.layers = language_model.layers
    self.norm = language_model.norm

  def forward(self, inputs_embeds, cos, sin):
    hidden = inputs_embeds
    for layer in self.layers:
      # No masks: attention falls back to is_causal, the Gated DeltaNet is
      # causal by construction, so right padding cannot reach valid rows.
      hidden = layer(hidden, position_embeddings=(cos, sin),
                     attention_mask=None, position_ids=None,
                     past_key_values=None, use_cache=False)
    return self.norm(hidden)


class TextSplit:
  """Stands in for ``pipe.text_encoder`` (``Qwen3_5Model``)."""

  def __init__(self, model, core, env, bucket=256, pad_id=0):
    self._model = model
    self._core = core
    self._env = env
    self._bucket = bucket
    self._pad_id = pad_id

  def __getattr__(self, name):
    return getattr(self._model, name)

  def host_inputs(self, input_ids):
    """Padded embeddings and RoPE tables on the host, and the true length."""
    lm = self._model.language_model
    ids = _host(self._env, input_ids).to(torch.long)
    length = ids.shape[1]
    padded = min(max(self._bucket, math.ceil(length / self._bucket) * self._bucket), 4096)
    if padded > length:
      ids = torch.cat([ids, ids.new_full((ids.shape[0], padded - length), self._pad_id)], 1)
    with torchax_paused():
      embeds = lm.embed_tokens(ids)
      pos = torch.arange(padded).view(1, 1, -1).expand(3, ids.shape[0], -1)
      cos, sin = lm.rotary_emb(embeds, pos)
    return embeds, cos, sin, length

  def __call__(self, input_ids=None, attention_mask=None, **kwargs):
    del attention_mask, kwargs  # one unpadded prompt per call upstream
    embeds, cos, sin, length = self.host_inputs(input_ids)
    to = self._env.to_xla
    hidden = self._core(to(embeds), to(cos), to(sin))
    hidden = hidden[:, :length]
    return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden,))


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class VaeDecode(torch.nn.Module):

  def __init__(self, vae):
    super().__init__()
    self.post_quant_conv = vae.post_quant_conv
    self.decoder = vae.decoder

  def forward(self, z):
    return self.decoder(self.post_quant_conv(z))


def _patch_vae_decode(vae, compiled):
  from diffusers.models.autoencoders.vae import DecoderOutput

  def decode(z, return_dict=True, generator=None):
    del generator
    sample = compiled(z)
    return DecoderOutput(sample=sample) if return_dict else (sample,)

  vae.decode = decode


# ---------------------------------------------------------------------------


def build_pipeline(repo, env, mesh, text_bucket=256, vae_eager=False,
                   split_ffn=True, attention="splash", fixed_m=False,
                   loop="ours", dmd=False, dit_backend="torchax"):
  """``Generator`` (``loop="ours"``) or the patched upstream pipeline.

  ``dmd`` installs the 8-step DMD adapter (``dmd.py``) on ``repo`` (which
  should then be MOSS-Image-SFT) and returns a ``DMDGenerator``.
  """
  from diffusers import AutoencoderKLFlux2
  from transformers import AutoModel, AutoTokenizer

  # Loading is plain torch (an earlier replica's build left torchax on).
  with contextlib.suppress(RuntimeError):  # "empty mode stack": it was off
    torchax.disable_globally()
  # Everything built here lands on ``mesh`` (replicas build one at a time).
  env._mesh = mesh
  env.default_device_or_sharding = NamedSharding(mesh, P())
  mod = remote.load_module(repo)
  part = lambda name: remote.component(repo, name)
  transformer = mod.MOSSImageTransformer.from_pretrained(
      part("transformer"), torch_dtype=torch.bfloat16).eval()
  tokenizer = AutoTokenizer.from_pretrained(part("tokenizer"))
  text_encoder = AutoModel.from_pretrained(
      part("text_encoder"), dtype=torch.bfloat16, attn_implementation="sdpa").eval()
  vae = AutoencoderKLFlux2.from_pretrained(part("vae"), torch_dtype=torch.float32).eval()
  scheduler = mod.MOSSImageScheduler.from_pretrained(part("scheduler"))
  pipe = mod.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae)
  cls = pipe.__class__
  pipe.__class__ = type(f"{cls.__name__}TPU", (cls,), {
      "_execution_device": property(lambda _s: torch.device("jax"))})

  torchax.enable_globally()

  if dit_backend == "jax" and loop == "ours" and not dmd:
    from diffusers_tpu.models.moss_image import jax_dit

    params, _, fn = jax_dit.make_step(remote.component(repo, "transformer"), mesh)

    def core(latents, sinusoid, context, rope, kv_len, skip, dt, guidance, skip_layers=()):
      j = env.t2j_iso
      out = fn(params, j(latents), j(sinusoid), j(context), j(rope), j(kv_len), j(skip),
               j(dt), j(guidance), skip_layers=skip_layers)
      return env.j2t_iso(out)
  else:
    core = None
  if split_ffn and core is None:
    split_up_gate(transformer)
  rules = dict(DIT_SHARDINGS)
  if dmd:
    from diffusers_tpu.models.moss_image import dmd as dmd_mod

    with torchax_paused():  # host-side: read and install the adapter
      mix, codes = dmd_mod.apply(transformer, dmd_mod.load_adapter())
    rules.update(dmd_mod.LORA_SHARDINGS)
  if core is not None:
    pass  # the native JAX step holds its own weights; the module stays on the host
  elif loop == "upstream":
    _move_module(transformer, env)
    sdpa = (make_splash_sdpa(mesh, env, use_fixed_m=fixed_m)
            if attention == "splash" else None)
    core = torchax.compile(DiTCore(transformer, sdpa), torchax.CompileOptions(
        jax_jit_kwargs={"static_argnames": ("skip_layers",)}))
  elif dmd:
    _move_module(transformer, env)
    sdpa = make_masked_splash(mesh, env, use_fixed_m=fixed_m)
    core = torchax.compile(dmd_mod.DMDStep(StepCore(transformer, sdpa, env, mesh)))
  else:
    _move_module(transformer, env)
    sdpa = make_masked_splash(mesh, env, use_fixed_m=fixed_m)
    core = torchax.compile(StepCore(transformer, sdpa, env, mesh), torchax.CompileOptions(
        jax_jit_kwargs={"static_argnames": ("skip_layers",)}))
  if hasattr(core, "params"):
    core.params = _shard(core.params, rules, mesh)
    core.buffers = _shard(core.buffers, rules, mesh)
  if loop == "upstream":
    pipe.transformer = DiTPrologue(transformer, core, env)

  lm = text_encoder.language_model
  _move_module(lm.layers, env)
  _move_module(lm.norm, env)
  text_core = torchax.compile(TextCore(lm))
  text_core.params = _shard(text_core.params, TEXT_SHARDINGS, mesh)
  text_core.buffers = _shard(text_core.buffers, TEXT_SHARDINGS, mesh)
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  pipe.text_encoder = TextSplit(text_encoder, text_core, env, text_bucket, pad_id)

  _move_module(vae, env)
  if not vae_eager:
    _patch_vae_decode(vae, torchax.compile(VaeDecode(vae)))
  if loop == "upstream":
    return pipe
  if dmd:
    return DMDGenerator(pipe, transformer, core, env, mesh, mix=mix, codes=codes)
  return Generator(pipe, transformer, core, env, mesh)


def bench(gen, env, prompt, negative_prompt, height, width, skip, reps=10,
          profile_dir=None):
  """Per-stage device time for ``Generator``, each stage synchronised."""

  def sync(t):
    jax.block_until_ready(t._elem if hasattr(t, "_elem") else t)
    return t

  def timed(label, fn, n):
    sync(fn())  # compile / warm
    t0 = time.perf_counter()
    for _ in range(n):
      out = fn()
    sync(out)
    dt = (time.perf_counter() - t0) / n
    print(f"{label:34s} {dt * 1e3:8.2f} ms", flush=True)
    return dt

  pipe = gen.pipe
  timed("text encode (positive)", lambda: pipe.encode_prompt(prompt)[0], 3)
  print(f"text tokens: positive {gen.encode(prompt)[1]}, negative {gen.encode(negative_prompt)[1]}")
  context, rope, kv_len = gen._inputs([prompt, negative_prompt], height, width)
  skip_flags = gen._put(np.array([False, True]), P("cfg"))
  latents = gen._put(np.random.randn(1, 128, height // 16, width // 16).astype(np.float32), P())
  sin = gen._put(np.random.randn(1, 256).astype(np.float32), P())
  dt = gen._put(np.float32(-0.02), P())
  g = gen._put(np.float32(3.0), P())

  def step():
    return gen._step(latents, sin, context, rope, kv_len, skip_flags, dt, g,
                     skip_layers=tuple(skip))

  c = timed("one CFG step (DiT x2 + CFG + Euler)", step, reps)
  z = env.to_xla(torch.randn(1, 32, height // 8, width // 8))
  timed("VAE decode (diffusers, torchax)", lambda: pipe.vae.decode(z, return_dict=False)[0], 3)
  from diffusers_tpu.models.moss_image import jax_vae
  vae_dir = remote.component(remote.DEFAULT_REPO, "vae")
  lat = gen._put(np.random.randn(1, 128, height // 16, width // 16).astype(np.float32), P())._elem
  for label, kwargs in (("JAX fp32, one chip's work", dict(mesh=None)),
                        ("JAX fp32, rows over chips", dict(mesh=gen._mesh)),
                        ("JAX bf16, rows over chips", dict(mesh=gen._mesh, dtype=jax.numpy.bfloat16))):
    params, fn = jax_vae.make_decoder(vae_dir, **kwargs)
    x = lat if kwargs["mesh"] is not None else jax.device_put(lat, jax.devices()[0])
    if kwargs["mesh"] is None:
      params = jax.device_put(params, jax.devices()[0])
    timed(f"VAE decode ({label})", lambda: fn(params, x), 3)
  print(f"=> 50 steps = {50 * c:.2f} s")
  if profile_dir:
    jax.profiler.start_trace(profile_dir, create_perfetto_trace=True)
    for _ in range(3):
      out = step()
    sync(out)
    jax.profiler.stop_trace()
    print(f"trace: {profile_dir}")
    jax.profiler.start_trace(profile_dir + "_vae", create_perfetto_trace=True)
    for _ in range(3):
      out = pipe.vae.decode(z, return_dict=False)[0]
    sync(out)
    jax.profiler.stop_trace()
    print(f"trace: {profile_dir}_vae")


def serve_replicas(gens, args, skip):
  """``--requests`` over several replicas: ``len(gens)`` requests at a time."""
  import json

  from PIL import Image

  os.makedirs(args.out, exist_ok=True)
  with open(args.requests) as fh:
    requests = [json.loads(line) for line in fh if line.strip()]
  height, width = (int(x) for x in (str(args.size).split("x") * 2)[:2])
  common = dict(negative_prompt=args.negative_prompt, height=height, width=width,
                steps=args.steps, guidance_scale=args.guidance_scale,
                uncond_skip_layers=skip)
  n = len(gens)
  t_all = time.perf_counter()
  batches = []
  for start in range(0, len(requests), n):
    group = requests[start:start + n]
    for gen, req in zip(gens, group):
      gen._prompts.pop(req["prompt"], None)
    t0 = time.perf_counter()
    images = run_concurrently(gens[:len(group)], group, **common)
    dt = time.perf_counter() - t0
    batches.append((dt, len(group)))
    for index, (req, image) in enumerate(zip(group, images)):
      Image.fromarray(image).save(os.path.join(args.out, f"{req.get('tag', start + index)}.png"))
    print(f"requests {start}-{start + len(group) - 1}: {dt:.2f}s for {len(group)}", flush=True)
  total = time.perf_counter() - t_all
  warm = batches[1:] or batches
  per_image = sum(d for d, _ in warm) / sum(k for _, k in warm)
  print(f"{len(requests)} requests in {total:.1f}s; warm {per_image:.2f}s per image "
        f"({sum(d for d, _ in warm) / len(warm):.2f}s per batch of {n})", flush=True)


def serve(gen, args, skip):
  """Resident: one request per JSON line, ``{"prompt", "seed"?, "size"?, "tag"?}``.

  The first request of a new size (or prompt-length bucket) compiles; each
  request encodes its own prompt (the empty negative prompt is encoded once).
  """
  import json

  from PIL import Image

  os.makedirs(args.out, exist_ok=True)
  with open(args.requests) as fh:
    requests = [json.loads(line) for line in fh if line.strip()]
  times = []
  t_all = time.perf_counter()
  for index, req in enumerate(requests):
    size = str(req.get("size", args.size))
    height, width = (int(x) for x in (size.split("x") * 2)[:2])
    gen._prompts.pop(req["prompt"], None)  # a new prompt every request
    t0 = time.perf_counter()
    image = gen(req["prompt"], negative_prompt=args.negative_prompt, height=height,
                width=width, steps=req.get("steps", args.steps),
                guidance_scale=args.guidance_scale, uncond_skip_layers=skip,
                seed=req.get("seed", args.seed))
    dt = time.perf_counter() - t0
    times.append(dt)
    tag = req.get("tag", f"{index:04d}")
    Image.fromarray(image).save(os.path.join(args.out, f"{tag}.png"))
    split = ", ".join(f"{k} {v:.2f}" for k, v in gen.times.items())
    print(f"request {tag} {height}x{width}: {dt:.2f}s ({split})", flush=True)
  total = time.perf_counter() - t_all
  warm = times[1:] or times
  print(f"{len(times)} requests in {total:.1f}s; warm mean {sum(warm) / len(warm):.2f}s",
        flush=True)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", default=remote.DEFAULT_REPO)
  p.add_argument("--prompt", default="a red fox in the snow")
  p.add_argument("--negative-prompt", default="")
  p.add_argument("--size", default="1024", help="N or HxW")
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--guidance-scale", type=float, default=3.0)
  p.add_argument("--uncond-skip-layers", default="22",
                 help="comma-separated; empty disables")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--chips", type=int, default=0, help="0 = all")
  p.add_argument("--replicas", type=int, default=1,
                 help="independent model copies on disjoint chips (with --requests)")
  p.add_argument("--text-bucket", type=int, default=256)
  p.add_argument("--vae-eager", action="store_true")
  p.add_argument("--vae", choices=("jax", "torch"), default="jax",
                 help="native JAX decoder (rows over the chips) or diffusers' through torchax")
  p.add_argument("--vae-dtype", choices=("auto", "fp32", "bf16"), default="auto",
                 help="JAX decoder precision; auto = fp32 unless a chip's share passes 2 MP")
  p.add_argument("--attention", choices=("splash", "xla"), default="splash")
  p.add_argument("--cfg", type=int, default=0,
                 help="mesh axis for the CFG batch: 2 runs the two branches on two halves of "
                      "the chips; 0 = 2 when the chip count is even, else 1")
  p.add_argument("--loop", choices=("ours", "upstream"), default="ours")
  p.add_argument("--dit", choices=("torchax", "jax"), default="torchax",
                 help="the step program: the upstream module traced, or native JAX (jax_dit.py)")
  p.add_argument("--dmd", action="store_true",
                 help="8-step DMD adapter on MOSS-Image-SFT (no CFG; --steps <= 8, default 8)")
  p.add_argument("--fixed-m", action="store_true", help="fixed-m softmax in splash")
  p.add_argument("--fused-ffn", action="store_true",
                 help="keep ffn_up_gate fused (its row split moves the gate between chips)")
  p.add_argument("--bench", action="store_true",
                 help="time text encode, one DiT call per branch and the decode")
  p.add_argument("--profile", default=None,
                 help="with --bench: trace three CFG steps into this dir")
  p.add_argument("--output", default="moss_image_torchax.png")
  p.add_argument("--requests", default=None,
                 help="JSONL of requests to serve with the models resident")
  p.add_argument("--out", default="out", help="directory for --requests images")
  p.add_argument("--compile-cache-dir", default="~/.cache/moss_image_jax")
  return p.parse_args()


def main():
  args = parse_args()
  print(args)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  height, width = (int(x) for x in (args.size.split("x") * 2)[:2])
  if args.dmd:
    from diffusers_tpu.models.moss_image import dmd as dmd_mod

    if args.repo == remote.DEFAULT_REPO:
      args.repo = dmd_mod.BASE
    if args.steps == 50:
      args.steps = 8
    args.guidance_scale = 1.0

  devices = jax.devices()[:args.chips or None]
  env = torchax.default_env()
  per = len(devices) // args.replicas
  gens = []
  t0 = time.perf_counter()
  for r in range(args.replicas):
    group = devices[r * per:(r + 1) * per]
    cfg = args.cfg or (2 if per % 2 == 0 and args.loop == "ours"
                       and args.guidance_scale != 1.0 else 1)
    mesh = Mesh(mesh_utils.create_device_mesh((cfg, per // cfg), group), ("cfg", "tensor"))
    print(f"replica {r}: chips {[d.id for d in group]}, mesh cfg={cfg} x tensor={per // cfg}, "
          f"{devices[0].device_kind}")
    gen = build_pipeline(args.repo, env, mesh, args.text_bucket, args.vae_eager,
                         not args.fused_ffn, args.attention, args.fixed_m, args.loop,
                         args.dmd, args.dit)
    if args.vae == "jax" and args.loop == "ours":
      gen.jax_decoder = remote.component(args.repo, "vae")
      gen.vae_dtype = args.vae_dtype
    gens.append(gen)
  gen = gens[0]
  print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)
  if args.replicas > 1:
    if not args.requests:
      raise SystemExit("--replicas needs --requests")
    serve_replicas(gens, args, [int(x) for x in args.uncond_skip_layers.split(",") if x.strip()])
    return

  skip = [int(x) for x in args.uncond_skip_layers.split(",") if x.strip()]
  if args.bench:
    bench(gen, env, args.prompt, args.negative_prompt, height, width, skip,
          profile_dir=args.profile)
    return
  if args.requests:
    serve(gen, args, skip)
    return
  image = None
  for i in range(args.runs):
    t0 = time.perf_counter()
    if args.loop == "ours":
      image = gen(args.prompt, negative_prompt=args.negative_prompt, height=height,
                  width=width, steps=args.steps, guidance_scale=args.guidance_scale,
                  uncond_skip_layers=skip, seed=args.seed)
      split = " (" + ", ".join(f"{k} {v:.2f}" for k, v in gen.times.items()) + ")"
    else:
      out = gen(args.prompt, height=height, width=width,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt, uncond_skip_layers=skip,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                output_type="np")
      image = np.asarray(out.images[0])
      split = ""
    dt = time.perf_counter() - t0
    print(f"run {i}: {dt:.2f}s{split}" + (" (includes compilation)" if i == 0 else ""),
          flush=True)

  from PIL import Image

  if image.dtype != np.uint8:
    image = (image * 255).round().clip(0, 255).astype(np.uint8)
  Image.fromarray(image).save(args.output)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
