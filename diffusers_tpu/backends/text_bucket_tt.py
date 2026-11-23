"""Text-length buckets for the torch_tpu H3 DiT (`--text-bucket N`, default 256).

A new prompt length changes the packed sequence, so the DiT programs (and, for
lossy runs, the sparse attention program) compile again: step 0 took 27-34 s
for every new length at 768p. Here prompts whose text rows round up to the
same multiple of N share one set of programs:

* Text rows are padded to a multiple of N. Attention masks keys at or past
  the true length at run time (a one-element integer `attention_mask`, see
  `ours_splash_tt`); padded queries/keys/values are zeroed, so the padding's
  content never reaches real rows (randn-filled vs zero-filled padding gave
  bit-identical output, c800/c802).
* The packed sequence is reordered: the target video rows (the last run of
  equal modulation index, ~98% of t2va rows) move to the front, everything
  else (text, conditions, audio) follows, padding last. The video then starts
  at row 0 with a length fixed by the video shape, so
  - each block broadcasts one modulation row over the video and gathers
    per row only over the short tail (what segmod does, without baking
    text-dependent row runs into the program; segmod cannot bucket);
  - the sparse block table (`sparse_attn_tt`, "tail" layout) depends only on
    the video shape and the bucketed length, so lossy runs bucket too.
* The token refiner masks by the true text length as well.

Outputs are not bit-identical to the unbucketed path (shapes and summation
order change; the difference is chaos-level, DINO within or near the
semantics-free 1e-3 perturbation floor, and judged visually equivalent on
11-22). README config (768p, v6e-8), new prompt in the same bucket: t2va
lossless 152.6 -> 128.3 s, lossy 89.4 -> 42.4 s; ref2va lossless 191.3 ->
166.9 s, lossy 103.2 -> 59.1 s; steady state per step +0-2% (c840).
"""
import os

import torch

EAGER_KV: dict = {"n": None}
# Rows of the broadcast video run (0 = per-row gather everywhere) and its
# modulation index; read by the block forward while tracing.
BULK: dict = {"V": 0, "i": None}
# Below this share of rows in the video run (ref2va: text + reference rows +
# audio are 13%), recomputing the tail beside the broadcast was slower than
# the per-row gather (8.91 vs 8.10 s a step, c813); keep the gather then.
_MIN_BULK = 0.95
_orig_block = None


def _install_block(transformer, verbose):
  import diffusers.models.transformers.transformer_minimax_h3 as h3
  bcls = h3.MiniMaxH3TransformerBlock

  def block_forward(self, hidden_states, temb, adaln_indices, rotary_emb,
                    attention_mask=None):
    V = BULK["V"]
    if V <= 0:
      return _orig_block(self, hidden_states, temb, adaln_indices, rotary_emb,
                         attention_mask)
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = self.adaln_proj(temb)
    bulk = adaln_indices[:1]
    tail = adaln_indices[V:]
    tail_pos = torch.arange(V, hidden_states.shape[1], device=hidden_states.device)
    pick = lambda p: p.index_select(0, bulk)
    # The tail's parameter rows as a one-hot matmul: exact (one nonzero term)
    # and on the MXU, unlike a per-row gather with runtime indices.
    oh = torch.nn.functional.one_hot(tail, scale_msa.shape[0]).to(scale_msa.dtype)
    tsel = lambda p: oh @ p

    def mod(x, scale, shift):
      out = x * (1.0 + pick(scale)) + pick(shift)
      t = x[:, V:] * (1.0 + tsel(scale)) + tsel(shift)
      return out.index_copy(1, tail_pos, t)

    def res(r, gate, y):
      out = r + pick(gate) * y
      t = r[:, V:] + tsel(gate) * y[:, V:]
      return out.index_copy(1, tail_pos, t)

    h = res(hidden_states, gate_msa,
            self.attn(mod(self.norm1(hidden_states), scale_msa, shift_msa),
                      rotary_emb, attention_mask))
    return res(h, gate_mlp, self.ff(mod(self.norm2(h), scale_mlp, shift_mlp)))

  global _orig_block
  _orig_block = bcls.forward
  bcls.forward = block_forward
  # AotBlocks captured each block's forward at install time (`_aot_fwd`);
  # patching the class alone does not reach the compiled programs (c819).
  n_fix = 0
  for b in transformer.transformer_blocks:
    if "_aot_fwd" in b.__dict__:
      b.__dict__["_aot_fwd"] = block_forward
      n_fix += 1
  from diffusers_tpu.backends.aot_dit import AotBlocks
  AotBlocks.KEY_EXTRA.append(lambda: ("bulkV", BULK["V"]))
  verbose(f"  [text-bucket] video run broadcast in {n_fix} blocks", flush=True)


def install(transformer, n: int, verbose=print) -> bool:
  if n <= 0:
    return False
  from diffusers_tpu.backends.aot_dit import AotBlocks
  if AotBlocks.PRE_TRACE:
    verbose("  [text-bucket] off: segment-wise modulation bakes row runs into the program",
            flush=True)
    return False
  import diffusers.models.transformers.transformer_minimax_h3 as h3
  from diffusers.models.modeling_utils import get_parameter_dtype

  rcls = h3.MiniMaxH3TokenRefinerBlock

  def refiner_forward(self, hidden_states):
    kv = EAGER_KV["n"]
    mask = None if kv is None else torch.tensor([kv], dtype=torch.int32,
                                                 device=hidden_states.device)
    hidden_states = hidden_states + self.attn(self.norm1(hidden_states), None, mask)
    return hidden_states + self.ff(self.norm2(hidden_states))
  rcls.forward = refiner_forward

  _install_block(transformer, verbose)
  cls = type(transformer)

  def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
              timestep, timestep_indices, token_tags, position_ids,
              video_indices, audio_indices, text_indices,
              attention_kwargs=None, return_dict=True):
    seq = position_ids.shape[0]
    n_txt = encoder_hidden_states.shape[1]
    dev = position_ids.device
    # The packed layout is rebuilt on the host: every op below has a shape
    # that follows the prompt length, so on the chip each new length compiled
    # its own small eager programs (c886: 33 a new prompt). The tensors are
    # small; they go back to the chip once at the end. Values are unchanged.
    position_ids, token_tags, timestep_indices = (
        position_ids.cpu(), token_tags.cpu(), timestep_indices.cpu())
    text_indices, video_indices, audio_indices = (
        text_indices.cpu(), video_indices.cpu(), audio_indices.cpu())
    # Video run (the last run of equal modulation index) to the front.
    ai = timestep_indices * h3.MINIMAX_H3_MODALITY_NUM + token_tags
    chg = (ai[1:] != ai[:-1]).nonzero()
    s_last = int(chg[-1]) + 1 if chg.numel() else 0
    cpu = torch.device("cpu")   # explicit: the caller runs under `with torch.device("tpu")`
    perm = torch.cat([torch.arange(s_last, seq, device=cpu), torch.arange(0, s_last, device=cpu)])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(seq, device=cpu)
    position_ids = position_ids.index_select(0, perm)
    token_tags = token_tags.index_select(0, perm)
    timestep_indices = timestep_indices.index_select(0, perm)
    text_indices = inv.index_select(0, text_indices).to(text_indices.dtype)
    video_indices = inv.index_select(0, video_indices).to(video_indices.dtype)
    audio_indices = inv.index_select(0, audio_indices).to(audio_indices.dtype)
    n_bulk = seq - s_last
    try:
      from diffusers_tpu.backends import sparse_attn_tt as _sp
      lay = _sp.STATE.get("layout")
      if lay is not None and lay[0] != "tail":
        _sp.STATE["layout"] = ("tail", n_bulk, lay[1], lay[2])
    except Exception:  # noqa: BLE001
      pass
    BULK["V"] = n_bulk if n_bulk >= _MIN_BULK * seq else 0
    if not BULK.get("said"):
      BULK["said"] = True
      verbose(f"  [text-bucket] video run {n_bulk} of {seq} rows "
              f"({100 * n_bulk / seq:.1f}%): "
              f"{'broadcast' if BULK['V'] else 'per-row gather'}", flush=True)

    extra = -(-n_txt // n) * n - n_txt
    if extra:
      ehs = encoder_hidden_states.cpu()
      encoder_hidden_states = torch.cat(
          [ehs, ehs.new_zeros((ehs.shape[0], extra, ehs.shape[2]))], dim=1).to(dev)
      text_indices = torch.cat([text_indices, torch.arange(
          seq, seq + extra, dtype=text_indices.dtype, device=cpu)])
      position_ids = torch.cat([position_ids, position_ids[-1:].expand(extra, -1)])
      token_tags = torch.cat([token_tags, token_tags.new_ones(extra)])
      t_text = timestep_indices.index_select(0, text_indices[:1])
      timestep_indices = torch.cat([timestep_indices, t_text.expand(extra)])
    position_ids, token_tags, timestep_indices = (
        position_ids.to(dev), token_tags.to(dev), timestep_indices.to(dev))
    text_indices, video_indices, audio_indices = (
        text_indices.to(dev), video_indices.to(dev), audio_indices.to(dev))
    kv = torch.tensor([seq], dtype=torch.int32, device=dev)
    sequence_length = position_ids.shape[0]

    rotary_emb = self.rope(position_ids)
    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
    text_embeds = self.context_embedder(
        encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
    EAGER_KV["n"] = n_txt
    try:
      text_embeds = self.token_refiner(text_embeds)
    finally:
      EAGER_KV["n"] = None
    x = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    x = x.index_copy(1, text_indices, text_embeds)
    x = x.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    x = x.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
    adaln_indices = timestep_indices * h3.MINIMAX_H3_MODALITY_NUM + token_tags
    for block in self.transformer_blocks:
      x = block(x, temb, adaln_indices, rotary_emb, attention_mask=kv)
    x = self.norm_out(x, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
    video_output = self.proj_out(x).index_select(1, video_indices)
    audio_output = self.audio_proj_out(x).index_select(1, audio_indices)
    if not return_dict:
      return (video_output, audio_output)
    return h3.MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

  cls.forward = forward
  verbose(f"  [text-bucket] text rows padded to a multiple of {n}, masked by "
          f"true length at run time", flush=True)
  return True
