"""Segment-wise modulation instead of a per-row gather (torch_tpu).

diffusers' block does 6 `index_select(0, adaln_indices)` per layer
(scale/shift/gate twice), each materialising a full-sequence tensor. The rows
form contiguous runs and the last run (video rows) is ~98% of the sequence,
so we compute the whole tensor with the last run's parameter row broadcast
and rewrite only the short head:

    out = f(hidden, params[last_run])
    out[:, :head] = f(hidden[:, :head], params[head_runs])

Same arithmetic (torchax's `install_segmented_modulation` measured 1.21x on a
15 s clip, within one bfloat16 ULP).

The runs must be Python constants before compiling, so they are computed
eagerly from the real `adaln_indices`. The block class's `forward` is
replaced; `aot_dit._flat_forward` uses `type(module).forward`, so the
compiled program picks up this version.
"""

from __future__ import annotations

import os

_RUNS: tuple | None = None
_SAID = []


def _compute_runs(adaln_indices) -> tuple:
  """Compress per-row segment ids into contiguous `(start, end, index)` runs."""
  idx = adaln_indices.detach().to("cpu").tolist()
  runs = []
  start = 0
  for i in range(1, len(idx) + 1):
    if i == len(idx) or idx[i] != idx[start]:
      runs.append((start, i, int(idx[start])))
      start = i
  return tuple(runs)


def install(transformer, verbose=print) -> bool:
  """Replace the block class's forward with a segment-wise modulation version."""
  import torch

  blocks = getattr(transformer, "transformer_blocks", None)
  if not blocks:
    verbose("  [segmod] transformer_blocks not found", flush=True)
    return False
  cls = type(blocks[0])
  original = cls.forward

  # `adaln_indices` is generated inside the transformer, so a pre-hook on the
  # transformer cannot see it. Runs are computed eagerly (outside compiled
  # regions) from the block's real `adaln_indices` via `warmup()`, which the
  # AOT pre-trace hook calls before each compile.
  def warmup(adaln_indices, force=False):
    global _RUNS
    if _RUNS is not None and not force:
      return
    # Recompute before every compile: the modulation index per run differs
    # between step 1 and later steps, and AotBlocks compiles one program for
    # each. Freezing the first result bakes wrong indices into later programs.
    _RUNS = _compute_runs(adaln_indices)
    head = sum(e - s for s, e, _ in _RUNS[:-1])
    last = _RUNS[-1][1] - _RUNS[-1][0]
    verbose(f"  [segmod] {len(_RUNS)} runs, {head} head rows, "
            f"last run {last} rows ({100 * last / _RUNS[-1][1]:.1f}%)", flush=True)

  install.warmup = warmup

  # AOT pre-trace hook; block positional args are
  # (hidden_states, temb, adaln_indices, rotary_emb, ...).
  try:
    from diffusers_tpu.backends.aot_dit import AotBlocks

    def _maybe_warmup(a, k):
      # PRE_TRACE is a class-level list shared by all AotBlocks instances,
      # including VAE blocks (`--aot-vae`) that have no adaln_indices. If none
      # is found this is not a DiT block; skip it.
      if "adaln_indices" in k:
        return warmup(k["adaln_indices"], force=True)
      for x in list(a) + list(k.values()):
        if (isinstance(x, torch.Tensor) and x.dim() == 1
            and x.dtype == torch.int64):
          return warmup(x, force=True)

    AotBlocks.PRE_TRACE.append(_maybe_warmup)
  except Exception as exc:  # noqa: BLE001
    verbose(f"  [segmod] could not install AOT pre-trace hook: {type(exc).__name__}",
            flush=True)

  def by_runs(x, runs, fn):
    """Apply `fn` to `x` run by run; equivalent to a per-row parameter gather.

    As in torchax: compute the whole tensor broadcasting the last run's single
    parameter row (~98% of rows), then recompute the short head and write it back.
    """
    *head_runs, (_, _, bulk) = runs
    if not head_runs:
      return fn(x, bulk)
    # `DT_SEGMOD_CAT=1`: compute each run separately and concatenate.
    # The default recomputes the head and writes it back; cheap for t2va
    # (head ~1% of rows, 146.4 -> ~143 s) but slower than the gather for
    # ref2va (head ~13%, ~188 -> ~205 s).
    if os.environ.get("DT_SEGMOD_CAT") == "1":
      return torch.cat([fn(x[:, s:e], i) for s, e, i in runs], dim=1)
    # `DT_SEGMOD_HEAD=1`: gather per row only over the H head rows; the last
    # run (video rows) still broadcasts one parameter row. Two ops per
    # modulation regardless of run count; ref2va has 6 runs (some only a few
    # rows), where per-run slicing adds ~30 tiny ops per block.
    if os.environ.get("DT_SEGMOD_HEAD") == "1":
      H = head_runs[-1][1]
      idx = torch.tensor([i for s, e, i in head_runs for _ in range(e - s)],
                         dtype=torch.long, device=x.device)
      return torch.cat([fn(x[:, :H], idx), fn(x[:, H:], bulk)], dim=1)
    out = fn(x, bulk)
    head = torch.cat([fn(x[:, s:e], i) for s, e, i in head_runs], dim=1)
    return out.index_copy(
        1, torch.arange(head.shape[1], device=x.device), head)

  def by_runs_add(residual, x, gate, runs, head_idx):
    """Fuse `residual + gate * x` before `index_copy` to avoid materialising `gate * x`."""
    *head_runs, (_, _, bulk) = runs
    if not head_runs:
      return residual + gate[bulk] * x
    out = residual + gate[bulk] * x
    head = torch.cat([
        residual[:, s:e] + gate[i] * x[:, s:e]
        for s, e, i in head_runs
    ], dim=1)
    return out.index_copy(1, head_idx, head)

  def forward(self, hidden_states, temb, adaln_indices, rotary_emb,
              attention_mask=None, **kw):
    if _RUNS is None and not torch.compiler.is_compiling():
      warmup(adaln_indices)
    runs = _RUNS
    if runs is None:          # not warmed up while tracing: original (correct, slower)
      return original(self, hidden_states, temb, adaln_indices,
                      rotary_emb, attention_mask, **kw)
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = self.adaln_proj(temb)

    # `DT_SEGMOD_PART=msa|mlp`: segment only half of the modulations and keep
    # the per-row gather for the other half. The gathers run on the async
    # stream overlapping all-reduce; removing all of them on ref2va exposed
    # the all-reduce (1.2 s -> 21.7 s exclusive, net 17 s slower).
    part = os.environ.get("DT_SEGMOD_PART", "all")
    fused_res = os.environ.get("DT_FUSED_RES_MOD", "1") == "1"
    head_len = runs[-1][0] if len(runs) > 1 else 0
    head_idx = (torch.arange(head_len, device=hidden_states.device)
                if (fused_res and head_len > 0) else None)

    def mod(x, fn, which):
      if part == "all" or part == which:
        return by_runs(x, runs, fn)
      return fn(x, adaln_indices)          # original per-row gather

    residual = hidden_states
    norm_hidden_states = mod(
        self.norm1(hidden_states),
        lambda x, i: x * (1.0 + scale_msa[i]) + shift_msa[i], "msa")
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    if fused_res and (part == "all" or part == "msa") and head_idx is not None:
      hidden_states = by_runs_add(residual, attn_output, gate_msa, runs, head_idx)
    else:
      hidden_states = residual + mod(
          attn_output, lambda x, i: gate_msa[i] * x, "msa")

    residual = hidden_states
    norm_hidden_states = mod(
        self.norm2(hidden_states),
        lambda x, i: x * (1.0 + scale_mlp[i]) + shift_mlp[i], "mlp")
    ff_output = self.ff(norm_hidden_states)
    if fused_res and (part == "all" or part == "mlp") and head_idx is not None:
      hidden_states = by_runs_add(residual, ff_output, gate_mlp, runs, head_idx)
    else:
      hidden_states = residual + mod(
          ff_output, lambda x, i: gate_mlp[i] * x, "mlp")
    return hidden_states

  cls.forward = forward
  verbose("  [segmod] block modulation is now segment-wise (no per-row gather)",
          flush=True)
  return True
