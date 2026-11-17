"""MOSS-Image's DiT through torchax: host prologue plus a traceable core.

The upstream ``MOSSImageTransformer.forward`` cannot be traced as written:

* it validates the mask with ``torch.all(...)`` inside an ``if``, and selects
  each sample's text with ``encoder_hidden_states[i, mask[i].bool()]`` -- a
  shape that depends on tensor contents;
* the timestep sinusoid is computed in float64, which JAX does not carry
  unless x64 is enabled (it would silently become float32);
* RoPE tables are rebuilt from a float32 buffer on every call.

None of these depends on the latents, so the host does them: it strips the
text padding, builds the sinusoid in float64 and the RoPE table in float32,
and hands the core plain tensors. The core is the rest of ``forward`` line for
line, for one sample, and holds the original submodules under their original
names so weight-name sharding rules still match.

Run this file to check the core on TPU against the upstream module on CPU:

    PYTHONPATH=. python -m diffusers_tpu.models.moss_image.torchax_dit --size 512
"""
import argparse
import time

import torch


def split_up_gate(model):
  """Replaces each block's fused ``ffn_up_gate`` [hidden; gate] linear with
  two linears, ``ffn_up`` and ``ffn_gate``.

  The layer computes ``hidden, gate = up_gate(y).chunk(2, -1)``. Split by
  rows over the chips as stored, the first half of the chips holds only
  hidden rows and the second only gate rows, so every block moves half the
  FFN activation between chips (a collective-permute, 17% of a step at
  1024x1024 on a v6e-4). Two linears, each split by rows, give every chip
  the hidden and gate rows that pair with each other and with its slice of
  ``ffn_down``, with no reshape of the activation (an interleaved fused
  matrix needs one, and with a batch of two the reshape cost a relayout and
  an unfused activation: 13 ms a step).
  """
  for layer in model.layers:
    fused = layer.proj["ffn_up_gate"]
    hidden, gate = fused.weight.data.chunk(2, 0)
    for name, w in (("ffn_up", hidden), ("ffn_gate", gate)):
      linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=False,
                               dtype=w.dtype, device=w.device)
      linear.weight.data = w.contiguous()
      layer.proj[name] = linear
    del layer.proj["ffn_up_gate"]


def make_attention(sdpa, rotate):
  """Upstream ``_attention`` with the SDPA call replaced by ``sdpa``."""

  def attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return sdpa(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)

  return attention


def _block(layer, x, rope, modulation, attention, situ):
  """``_Block.forward``, for either FFN layout (fused or ``split_up_gate``)."""
  shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
  y = layer.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
  dtype = layer.proj["q"].weight.dtype
  y = y.to(dtype)
  q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
  k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
  x = x + layer.proj["o"](attention(q, k, layer.proj["v"](y), rope, layer.heads)) * gate_a[:, None]
  y = layer.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
  y = y.to(dtype)
  if "ffn_up" in layer.proj:
    hidden, gate = layer.proj["ffn_up"](y), layer.proj["ffn_gate"](y)
  else:
    hidden, gate = layer.proj["ffn_up_gate"](y).chunk(2, -1)
  return x + layer.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class DiTCore(torch.nn.Module):
  """``MOSSImageTransformer.forward`` for one sample, from the embeddings on."""

  def __init__(self, model, sdpa=None):
    super().__init__()
    import inspect

    upstream = inspect.getmodule(type(model))
    self._attention = (upstream._attention if sdpa is None
                       else make_attention(sdpa, upstream._rotate))
    self._situ = upstream._situ
    self.video_embedder = model.video_embedder
    self.text_embedder = model.text_embedder
    self.time_embedding = model.time_embedding
    self.shared_block_modulation = model.shared_block_modulation
    self.layers = model.layers
    self.final_norm_video = model.final_norm_video
    self.final_linear_video = model.final_linear_video
    self.video_head_modulation = model.video_head_modulation

  def forward(self, latents, time_sinusoid, context, rope, skip_layers=()):
    """latents [1, C, H, W] fp32; time_sinusoid [1, freq_dim] fp32;
    context [1, L, text_dim] (valid tokens only); rope [1, H*W + L, head_dim]."""
    _, _, height, width = latents.shape
    n_img = height * width
    time = self.time_embedding(
        time_sinusoid.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(
        time.to(self.shared_block_modulation.weight.dtype)).view(1, 6, -1)
    head = self.video_head_modulation(
        time.to(self.video_head_modulation.weight.dtype)).view(1, 2, -1)
    video = latents.flatten(2).transpose(1, 2)
    video = self.video_embedder(video.to(self.video_embedder.weight.dtype))
    rope = rope.to(video.dtype)
    text = self.text_embedder(context, rope[:, n_img:])
    x = torch.cat((video, text), 1)
    for index, layer in enumerate(self.layers):
      if index not in skip_layers:
        x = _block(layer, x, rope, mod, self._attention, self._situ)
    x = self.final_norm_video(x[:, :n_img].to(self.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(1, -1, height, width)


def time_sinusoid(model, timestep, batch=1):
  """The float64 sinusoid of ``forward``, as float32 [batch, freq_dim]."""
  times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).expand(batch)
  half = model.config.freq_dim // 2
  freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
  phase = times.double()[:, None] * freq
  return torch.cat((phase.cos(), phase.sin()), -1).float()


def rope_table(model, height, width, text_length, pad_to=None):
  """float32 [1, H*W + L, head_dim], ``_RoPE.forward`` on the host.

  Rebuilt from the config rather than called on ``model.rope``: once the
  model's buffers are on the device, its ``bands`` are a device tensor.
  ``pad_to`` extends the text rows (positions continue) for a prompt padded
  to that length; the image rows still carry the true ``text_length``.
  """
  if pad_to is not None and pad_to != text_length:
    table = rope_table(model, height, width, text_length)
    tail = rope_table(model, 1, 1, pad_to)[:, 1 + text_length:]
    return torch.cat((table, tail), 1)
  cfg = model.config
  num_bands = cfg.dim // cfg.num_heads // 8
  bands = 1 / (cfg.rope_base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))
  h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
  image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w),
                      -1).flatten(0, 1).float()
  image[:, 2] -= (height - 1) / 2
  image[:, 3] -= (width - 1) / 2
  text = torch.zeros(text_length, 4, dtype=torch.float32)
  text[:, 1] = torch.arange(text_length)
  phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
  return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _parity(args):
  import jax
  import numpy as np
  import torchax

  from diffusers_tpu.models.moss_image import remote

  if args.highest:
    jax.config.update("jax_default_matmul_precision", "highest")
  mod = remote.load_module()
  path = remote.snapshot() + "/transformer"
  h = w = args.size // 16
  gen = torch.Generator().manual_seed(0)
  latents = torch.randn(1, 128, h, w, generator=gen)
  context = torch.randn(1, args.text_len, 4096, generator=gen).to(torch.bfloat16)
  timestep = torch.tensor([700.0])

  def upstream(dtype):
    model = mod.MOSSImageTransformer.from_pretrained(path, torch_dtype=dtype).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
      out = model(latents, timestep, context.to(dtype)).sample.float()
    print(f"cpu {dtype}: {time.perf_counter() - t0:.1f}s")
    return model, out

  model32, ref32 = upstream(torch.float32)
  model, ref16 = upstream(torch.bfloat16)
  if args.tpu_fp32:
    model, context = model32, context.float()

  sin = time_sinusoid(model, timestep)
  rope = rope_table(model, h, w, args.text_len)
  core = DiTCore(model)
  with torch.no_grad():
    host16 = core(latents, sin, context, rope).float()

  torchax.enable_globally()
  env = torchax.default_env()
  with env:
    core.to("jax")
    compiled = torchax.compile(core)
    ins = [env.to_xla(t) for t in (latents, sin, context, rope)]
    for i in range(2):
      t0 = time.perf_counter()
      out = compiled(*ins)
      res = torch.from_numpy(np.asarray(jax.device_get(out._elem)).astype(np.float32))
      print(f"tpu run {i}: {time.perf_counter() - t0:.2f}s")
  torchax.disable_globally()

  def stats(name, a, b):
    a, b = a.flatten().double(), b.flatten().double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / b.norm()).item()
    print(f"{name:28s} cos={cos:.6f} rel_l2={rel:.3e}")

  stats("cpu bf16 vs cpu fp32", ref16, ref32)
  stats("core bf16 (cpu) vs upstream", host16, ref16)
  stats("tpu bf16 vs cpu fp32", res, ref32)
  stats("tpu bf16 vs cpu bf16", res, ref16)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--size", type=int, default=512)
  p.add_argument("--text-len", type=int, default=77)
  p.add_argument("--tpu-fp32", action="store_true",
                 help="run the TPU core with the float32 weights")
  p.add_argument("--highest", action="store_true",
                 help="float32 matmuls at full precision on TPU")
  _parity(p.parse_args())
