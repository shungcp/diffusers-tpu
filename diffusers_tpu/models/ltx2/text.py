"""Gemma-3-12B text encoder for LTX-2.3 (transformers' own model through torchax).

The prompt is tokenized left-padded to ``max_length`` (1024) exactly as the
diffusers pipeline does (Gemma's positions run over the padding too, so the
padded layout is part of the result). The conditioning is all 49 hidden
states (embeddings, then every decoder layer; the last one after the final
norm), stacked on a trailing axis: [B, L, 3840, 49].

All Gemma-3 layers see the same mask here: causal + key padding (the 1024
sliding window covers the whole 1024-token prompt).
"""
import contextlib

import numpy as np

MAX_LENGTH = 1024


def _core_class():
  import torch

  class Core(torch.nn.Module):
    """The decoder stack: all hidden states for given embeddings / mask / RoPE."""

    def __init__(self, lm):
      super().__init__()
      self.layers = lm.layers
      self.norm = lm.norm
      self.types = lm.config.layer_types

    def forward(self, embeds, mask, cos_g, sin_g, cos_l, sin_l):
      h = embeds
      outs = [h]
      for layer, kind in zip(self.layers, self.types):
        pe = (cos_l, sin_l) if kind == "sliding_attention" else (cos_g, sin_g)
        h = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=None, past_key_values=None)
        outs.append(h)
      outs[-1] = self.norm(h)
      return torch.stack(outs, -1)

  return Core


# Megatron layout of the Gemma decoder over ``axis`` (torch Linear weights are [out, in]).
TP_RULES = {
    r"layers\.\d+\.self_attn\.(q|k|v)_proj\.weight": ("tp", None),
    r"layers\.\d+\.self_attn\.o_proj\.weight": (None, "tp"),
    r"layers\.\d+\.mlp\.(gate|up)_proj\.weight": ("tp", None),
    r"layers\.\d+\.mlp\.down_proj\.weight": (None, "tp"),
}


class TextEncoder:
  """``encode(prompt) -> (hidden [1, L, 3840, 49], mask [1, L] numpy)``; the
  hidden states a jax array on the mesh (numpy without one)."""

  def __init__(self, folder, tokenizer_folder, dtype=None, env=None, mesh=None, max_length=MAX_LENGTH):
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
    self.tokenizer.padding_side = "left"
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    if env is not None:
      import torchax
      with contextlib.suppress(RuntimeError):
        torchax.disable_globally()
    model = Gemma3ForConditionalGeneration.from_pretrained(folder, dtype=dtype or torch.bfloat16,
                                                           attn_implementation="sdpa").eval()
    self.lm = model.model.language_model
    self.core = _core_class()(self.lm)
    self.env, self.mesh, self.max_length = env, mesh, max_length
    self.cache = {}
    if env is not None:
      import torchax
      from jax.sharding import NamedSharding
      from jax.sharding import PartitionSpec as P

      from diffusers_tpu.models.moss_image.torchax_pipeline import _move_module, _shard

      torchax.enable_globally()
      env._mesh = mesh
      env.default_device_or_sharding = NamedSharding(mesh, P())
      _move_module(self.core, env)
      self.core = torchax.compile(self.core)
      every = tuple(mesh.axis_names)  # split over all chips: 24 GB in bf16
      rules = {k: tuple(every if a == "tp" else a for a in v) for k, v in TP_RULES.items()}
      self.core.params = _shard(self.core.params, rules, mesh)
      self.core.buffers = _shard(self.core.buffers, {}, mesh)
      torchax.disable_globally()

  def _inputs(self, prompt):
    import torch

    enc = self.tokenizer([prompt.strip()], padding="max_length", max_length=self.max_length, truncation=True,
                         add_special_tokens=True, return_tensors="pt")
    ids, keep = enc.input_ids, enc.attention_mask
    with torch.no_grad():
      embeds = self.lm.embed_tokens(ids)
      n = ids.shape[1]
      pos = torch.arange(n).unsqueeze(0)
      cos_g, sin_g = self.lm.rotary_emb(embeds, pos, "full_attention")
      cos_l, sin_l = self.lm.rotary_emb(embeds, pos, "sliding_attention")
      causal = torch.ones(n, n, dtype=torch.bool).tril()
      allowed = causal[None, None] & keep.bool()[:, None, None, :]
      mask = torch.zeros(allowed.shape, dtype=embeds.dtype).masked_fill(~allowed, torch.finfo(embeds.dtype).min)
    return (embeds, mask, cos_g, sin_g, cos_l, sin_l), keep.numpy()

  def encode(self, prompt):
    import torch

    if prompt in self.cache:
      return self.cache[prompt]
    args, keep = self._inputs(prompt)
    with torch.no_grad():
      if self.env is None:
        out = self.core(*args).float().numpy()
      else:
        import torchax

        torchax.enable_globally()
        try:
          h = self.core(*[self.env.to_xla(a) for a in args])
          out = self.env.t2j_iso(h)  # stays on the chips
        finally:
          torchax.disable_globally()
    self.cache[prompt] = (out, keep)
    return out, keep
