"""Qwen-Image-2.1 transformer block, re-expressed for the cached denoising steps.

Same arithmetic as diffusers' `QwenImage21TransformerBlock.forward` with
`kv_cache_mode="cached"`, but the prefix keys/values arrive as tensor arguments
instead of inside a `QwenImage21KVLayerCache` object, so one AOT-compiled
program serves all 32 blocks. Lives at module scope because torch.compile
refuses to package classes defined inside a function.
"""
import torch


class CachedCore(torch.nn.Module):
  def __init__(self, block):
    super().__init__()
    self.block = block

  def forward(self, hidden_states, modulation, rotary_emb, target_token_mask,
              cached_k, cached_v):
    import diffusers.models.transformers.transformer_qwenimage21 as q21mod

    b, attn = self.block, self.block.attn
    mod1, mod2 = modulation.chunk(2, dim=-1)
    x, gate1 = b._modulate(b.img_norm1(hidden_states), mod1, target_token_mask)
    q = attn.to_q(x).unflatten(-1, (attn.heads, -1))
    k = attn.to_k(x).unflatten(-1, (attn.heads, -1))
    v = attn.to_v(x).unflatten(-1, (attn.heads, -1))
    q = attn.norm_q(q).to(v.dtype)
    k = attn.norm_k(k).to(v.dtype)
    q = q21mod.apply_rotary_emb_qwen(q, rotary_emb, use_real=False)
    k = q21mod.apply_rotary_emb_qwen(k, rotary_emb, use_real=False)
    k = torch.cat([cached_k, k], dim=1)
    v = torch.cat([cached_v, v], dim=1)
    h = q21mod.dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0)
    h = h.flatten(2, 3).type_as(q)
    h = attn.to_out[1](attn.to_out[0](h))
    hidden_states = hidden_states + gate1.tanh() * h
    x2, gate2 = b._modulate(b.img_norm2(hidden_states), mod2, target_token_mask)
    return hidden_states + gate2.tanh() * b.img_mlp(x2)


class CachedGroup(torch.nn.Module):
  """Several consecutive cached blocks in one program: one launch for the group.

  `forward(hidden_states, modulation, rotary_emb, target_token_mask, k0, v0,
  k1, v1, ...)` with one prefix key/value pair per block.
  """

  def __init__(self, blocks):
    super().__init__()
    self.cores = torch.nn.ModuleList(CachedCore(b) for b in blocks)

  def forward(self, hidden_states, modulation, rotary_emb, target_token_mask,
              *kvs):
    for i, core in enumerate(self.cores):
      fwd = core.__dict__.get("_aot_fwd") or type(core).forward
      hidden_states = fwd(core, hidden_states, modulation, rotary_emb,
                          target_token_mask, kvs[2 * i], kvs[2 * i + 1])
    return hidden_states


def _a2a(x, group_name, world):
  """all_to_all_single along dim 0 (equal splits), usable inside a compiled graph."""
  out = torch.ops._c10d_functional.all_to_all_single(
      x.contiguous(), [x.shape[0] // world] * world, [x.shape[0] // world] * world,
      group_name)
  return torch.ops._c10d_functional.wait_tensor(out)


class UlyssesGroup(torch.nn.Module):
  """All cached blocks in one program, Ulysses sequence parallel.

  Each rank keeps 1/world of the target rows and full (replicated) block
  weights: the linears run locally with no collective, and only attention
  trades sequence for heads with two all-to-alls. Tensor parallel's two
  all-reduces per block are gone. The prefix keys/values come from the
  tensor-parallel prefill, where this rank owns head group `rank` -- the same
  group the all-to-all hands it.
  """

  def __init__(self, full_blocks, rank, world, group_name):
    super().__init__()
    self.blocks = torch.nn.ModuleList(full_blocks)
    self.rank, self.world, self.group_name = rank, world, group_name

  def forward(self, hidden_states, modulation, rotary_emb, target_token_mask, *kvs):
    import diffusers.models.transformers.transformer_qwenimage21 as q21mod

    w, r = self.world, self.rank
    bsz, seq, dim = hidden_states.shape
    part = seq // w
    x = hidden_states[:, r * part:(r + 1) * part]
    rope = rotary_emb[r * part:(r + 1) * part]
    tmask = None if target_token_mask is None else target_token_mask[r * part:(r + 1) * part]
    for i, b in enumerate(self.blocks):
      attn = b.attn
      heads = attn.heads
      hd = heads // w
      mod1, mod2 = modulation.chunk(2, dim=-1)
      xm, gate1 = b._modulate(b.img_norm1(x), mod1, tmask)
      q = attn.to_q(xm).unflatten(-1, (heads, -1))
      k = attn.to_k(xm).unflatten(-1, (heads, -1))
      v = attn.to_v(xm).unflatten(-1, (heads, -1))
      q = attn.norm_q(q).to(v.dtype)
      k = attn.norm_k(k).to(v.dtype)
      q = q21mod.apply_rotary_emb_qwen(q, rope, use_real=False)
      k = q21mod.apply_rotary_emb_qwen(k, rope, use_real=False)

      def to_heads(t):
        # [B, part, H, D] -> [W(head group), B, part, H/W, D] -> exchange -> [B, seq, H/W, D]
        t = t.unflatten(2, (w, hd)).permute(2, 0, 1, 3, 4)
        t = _a2a(t, self.group_name, w)              # dim 0 now indexes the source rank
        return t.permute(1, 0, 2, 3, 4).reshape(bsz, seq, hd, t.shape[-1])

      q, k, v = to_heads(q), to_heads(k), to_heads(v)
      k = torch.cat([kvs[2 * i], k], dim=1)
      v = torch.cat([kvs[2 * i + 1], v], dim=1)
      h = q21mod.dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0)
      # [B, seq, H/W, D] -> [W(seq chunk), B, part, H/W, D] -> exchange -> [B, part, H, D]
      h = h.unflatten(1, (w, part)).permute(1, 0, 2, 3, 4)
      h = _a2a(h, self.group_name, w)                # dim 0 now indexes the head group
      h = h.permute(1, 2, 0, 3, 4).reshape(bsz, part, heads * h.shape[-1]).type_as(q)
      h = attn.to_out[1](attn.to_out[0](h))
      x = x + gate1.tanh() * h
      xm2, gate2 = b._modulate(b.img_norm2(x), mod2, tmask)
      x = x + gate2.tanh() * b.img_mlp(xm2)
    out = torch.ops._c10d_functional.all_gather_into_tensor(
        x.transpose(0, 1).contiguous(), w, self.group_name)
    out = torch.ops._c10d_functional.wait_tensor(out)
    return out.transpose(0, 1)


class ExtractCore(torch.nn.Module):
  """The prefill (`kv_cache_mode="extract"`) block for the plain text-to-image
  case: a single causal text segment `[0, prefix)` followed by the target rows.
  Returns `(hidden_states, prefix_keys, prefix_values)` so the caller stores the
  cache outside the compiled program."""

  def __init__(self, block):
    super().__init__()
    self.block = block

  def forward(self, hidden_states, modulation, rotary_emb, target_token_mask,
              text_mask):
    import diffusers.models.transformers.transformer_qwenimage21 as q21mod

    b, attn = self.block, self.block.attn
    prefix = text_mask.shape[-1]
    mod1, mod2 = modulation.chunk(2, dim=-1)
    x, gate1 = b._modulate(b.img_norm1(hidden_states), mod1, target_token_mask)
    q = attn.to_q(x).unflatten(-1, (attn.heads, -1))
    k = attn.to_k(x).unflatten(-1, (attn.heads, -1))
    v = attn.to_v(x).unflatten(-1, (attn.heads, -1))
    q = attn.norm_q(q).to(v.dtype)
    k = attn.norm_k(k).to(v.dtype)
    q = q21mod.apply_rotary_emb_qwen(q, rotary_emb, use_real=False)
    k = q21mod.apply_rotary_emb_qwen(k, rotary_emb, use_real=False)
    text = q21mod.dispatch_attention_fn(q[:, :prefix], k[:, :prefix], v[:, :prefix],
                                        attn_mask=text_mask, dropout_p=0.0)
    image = q21mod.dispatch_attention_fn(q[:, prefix:], k, v, attn_mask=None,
                                         dropout_p=0.0)
    h = torch.cat([text, image], dim=1).flatten(2, 3).type_as(q)
    h = attn.to_out[1](attn.to_out[0](h))
    hidden_states = hidden_states + gate1.tanh() * h
    x2, gate2 = b._modulate(b.img_norm2(hidden_states), mod2, target_token_mask)
    hidden_states = hidden_states + gate2.tanh() * b.img_mlp(x2)
    return hidden_states, k[:, :prefix].clone(), v[:, :prefix].clone()
