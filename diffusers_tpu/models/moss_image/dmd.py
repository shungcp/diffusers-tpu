"""MOSS-Image-SFT 8-step data-space DMD (``lodestones/MOSS-Image-SFT-8-steps-dmd``).

A community distillation of MOSS-Image-SFT (50 steps, CFG 3) to 8 network
evaluations without CFG. It is not a different architecture but an adapter on
the SFT transformer, plus its own sampler:

* a rank-128 LoRA on every block's ``q, k, v, o, ffn_up_gate, ffn_down``,
  applied **unmerged** -- the delta is below bf16's rounding step, so merging
  it into the bf16 weights would discard most of it;
* fp32 replacements for ``time_embedding`` and the ``shared_block_modulation``
  / ``video_head_modulation`` biases (the t=1 time vector is folded into the
  biases); those three modules run in fp32;
* the data-space sampler: from fixed noise ``z`` and ``x0 = 0``,
  ``x = z + Q x0``, ``v = DiT(x, t_r)``, ``x0 = z - v`` for the 8 step codes
  (``t_r`` = 1000 ... 24); the last ``x0`` is decoded like any latent.

All of it comes from ``generator.safetensors``.
"""
import os

import torch

from diffusers_tpu.hub import local_snapshot

REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
BASE = "OpenMOSS-Team/MOSS-Image-SFT"

# Column-parallel linears split their output rows (B's rows follow); the
# row-parallel ones split their input (A's columns follow).
LORA_SHARDINGS = {
    r".*layers\.\d+\.proj\.(q|k|v|ffn_up|ffn_gate)\.lora_B": ("tensor", None),
    r".*layers\.\d+\.proj\.(o|ffn_down)\.lora_A": (None, "tensor"),
}


class LoRALinear(torch.nn.Linear):
  """``y = W x + B (A x)``, the adapter unmerged; same weight name as before."""

  def __init__(self, base, lora_a, lora_b):
    super().__init__(base.in_features, base.out_features, bias=base.bias is not None,
                     device="meta")
    self.weight = base.weight
    self.bias = base.bias
    self.lora_A = torch.nn.Parameter(lora_a.to(base.weight.dtype), requires_grad=False)
    self.lora_B = torch.nn.Parameter(lora_b.to(base.weight.dtype), requires_grad=False)

  def forward(self, x):
    y = super().forward(x)
    delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
    return (y.float() + delta.float()).to(y.dtype)


def load_adapter(repo=REPO):
  from safetensors.torch import load_file

  return load_file(os.path.join(local_snapshot(repo), "generator.safetensors"))


def apply(model, adapter):
  """Installs the adapter on a (``split_up_gate``-d or fused) SFT transformer.

  Returns ``(Q [128, 128], step_codes [8])`` for the sampler.
  """
  for index, layer in enumerate(model.layers):
    proj = layer.proj
    for name in ("q", "k", "v", "o", "ffn_down"):
      a = adapter[f"layers.{index}.proj.{name}.lora_A"]
      b = adapter[f"layers.{index}.proj.{name}.lora_B"]
      proj[name] = LoRALinear(proj[name], a, b)
    a = adapter[f"layers.{index}.proj.ffn_up_gate.lora_A"]
    b = adapter[f"layers.{index}.proj.ffn_up_gate.lora_B"]
    if "ffn_up" in proj:  # split layout: B's rows are [hidden; gate]
      b_up, b_gate = b.chunk(2, 0)
      proj["ffn_up"] = LoRALinear(proj["ffn_up"], a, b_up)
      proj["ffn_gate"] = LoRALinear(proj["ffn_gate"], a, b_gate)
    else:
      proj["ffn_up_gate"] = LoRALinear(proj["ffn_up_gate"], a, b)
  # fp32 conditioning: the time embedder and both modulation layers.
  for module in (model.time_embedding, model.shared_block_modulation,
                 model.video_head_modulation):
    module.float()
  state = {k: v for k, v in adapter.items() if k.startswith(
      ("time_embedding.", "shared_block_modulation.", "video_head_modulation."))}
  missing = [k for k in state if k not in dict(model.named_parameters())]
  if missing:
    raise KeyError(f"adapter tensors with no home: {missing}")
  model.load_state_dict(state, strict=False)
  return adapter["latent_mix.Q"].float(), adapter["step_codes"].float()


class DMDStep(torch.nn.Module):
  """One data-space step: ``x0 <- z - DiT(z + Q x0, t)``, for one prompt."""

  def __init__(self, core):
    super().__init__()
    self.core = core  # a StepCore; only its ``dit`` is used

  def forward(self, z, x0, mix, sinusoid, context, rope, kv_len, skip):
    x = z + torch.einsum("oc,bchw->bohw", mix, x0)
    v = self.core.dit(x, sinusoid, context, rope, kv_len, skip, ()).float()
    return z - v
