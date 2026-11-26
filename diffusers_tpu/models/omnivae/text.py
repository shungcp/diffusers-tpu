"""OmniVAE prompt handling and the Qwen3.5-0.8B text encoder.

A prompt is wrapped in one of ten T2AV task templates, then in the chat
template (``add_generation_prompt``, ``enable_thinking``):
``<|im_start|>user\\n{templated}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``.
The negative prompt is ``""`` through the chat template only. The conditioning
is ``hidden_states[-2]``: the output of the 23rd of 24 decoder layers (before
the final norm), valid tokens only (up to 512). The model is causal, so the
unpadded prompt gives the same rows as the official right-padded batch.
"""
import contextlib
import os
import random

import numpy as np

T2AV_TEMPLATES = (
    "Generate a synchronized video and audio clip of: {text}",
    "Produce a video with matching sound depicting: {text}",
    "Create a short film with synchronized audio showing: {text}",
    "Render an audiovisual recording of: {text}",
    "Synthesize video and audio jointly for: {text}",
    "Generate a cinematic clip with synchronized sound of: {text}",
    "Compose a paired video and audio scene of: {text}",
    "Make a sound film clip showing: {text}",
    "Generate an audio-visual scene depicting: {text}",
    "Produce a video clip with realistic accompanying audio of: {text}",
)
MAX_TOKENS = 512


def wrap_prompt(prompt, template=None, seed=0):
  """The T2AV task template around ``prompt``: ``template`` (an index) or one
  drawn with ``random.Random(seed)`` (the official CLI's choice)."""
  t = T2AV_TEMPLATES[template] if template is not None else random.Random(int(seed)).choice(T2AV_TEMPLATES)
  return t.format(text=str(prompt).replace("{", "{{").replace("}", "}}"))


def chat(tokenizer, text):
  return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)


def _core_class():
  import torch

  class Core(torch.nn.Module):

    def __init__(self, language_model, n_layers):
      super().__init__()
      self.layers = language_model.layers[:n_layers]

    def forward(self, embeds, cos, sin):
      h = embeds
      for layer in self.layers:
        # No mask: full attention falls back to is_causal; the Gated DeltaNet
        # is causal by construction.
        h = layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None,
                  past_key_values=None, use_cache=False)
      return h

  return Core


class TextEncoder:
  """``encode(text) -> [L, 1024]`` float32 numpy; cached per text.

  ``env``/``mesh`` given: the layers run through torchax on the mesh
  (replicated; 0.75B), padded to ``bucket`` tokens so prompts share programs."""

  def __init__(self, folder, dtype=None, env=None, mesh=None, bucket=128):
    import torch
    from transformers import AutoModel, AutoTokenizer

    self.tokenizer = AutoTokenizer.from_pretrained(folder)
    self.tokenizer.padding_side = "right"
    if env is not None:
      import torchax
      with contextlib.suppress(RuntimeError):
        torchax.disable_globally()
    model = AutoModel.from_pretrained(folder, dtype=dtype or torch.bfloat16,
                                      attn_implementation="sdpa").eval()
    self.lm = model.language_model if hasattr(model, "language_model") else model
    n = len(self.lm.layers) - 1      # hidden_states[-2]
    self.core = _core_class()(self.lm, n)
    self.env, self.mesh, self.bucket = env, mesh, bucket
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
      self.core.params = _shard(self.core.params, {}, mesh)
      self.core.buffers = _shard(self.core.buffers, {}, mesh)
      torchax.disable_globally()

  def ids(self, text):
    enc = self.tokenizer([text], truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    return enc.input_ids

  def encode(self, text):
    import torch

    if text in self.cache:
      return self.cache[text]
    ids = self.ids(text)
    n = ids.shape[1]
    padded = n
    if self.env is not None:
      padded = min(MAX_TOKENS, -(-n // self.bucket) * self.bucket)
      if padded > n:
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.cat([ids, ids.new_full((1, padded - n), pad)], 1)
    with torch.no_grad():
      embeds = self.lm.embed_tokens(ids)
      pos = torch.arange(padded).view(1, 1, -1).expand(3, 1, -1)
      cos, sin = self.lm.rotary_emb(embeds, pos)
      if self.env is None:
        h = self.core(embeds, cos, sin)
        out = h[0, :n].float().numpy()
      else:
        import torchax

        torchax.enable_globally()
        try:
          to = self.env.to_xla
          h = self.core(to(embeds), to(cos), to(sin))
          out = np.asarray(self.env.t2j_iso(h)[0, :n].astype(np.float32))
        finally:
          torchax.disable_globally()
    self.cache[text] = out
    return out
