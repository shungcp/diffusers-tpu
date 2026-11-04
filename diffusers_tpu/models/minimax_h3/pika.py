"""Pika-H3-Flex on the PyTorch backends (torchax, torch_tpu).

The adapters are PEFT LoRAs on MiniMax-H3's attention and feed-forward
projections; they are merged into the transformer's weights before any
sharding, and the pipeline's two schedulers are swapped for Pika's
six-step re-noising sampler. Plain torch: no jax import.
"""

import torch


def merge_pika_lora(model: torch.nn.Module, adapter_dir: str) -> dict:
  """Merge a Pika-H3-Flex adapter into the transformer's weights, in float32.

  Returns the adapter manifest (ladder, shifts, trunk).
  """
  import json
  import os

  from safetensors.torch import load_file

  with open(os.path.join(adapter_dir, "manifest.json")) as f:
    manifest = json.load(f)
  scale = float(manifest["alpha"]) / float(manifest["rank"])
  state = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
  modules = dict(model.named_modules())
  merged = 0
  with torch.no_grad():
    for key, a in state.items():
      if ".lora_A." not in key:
        continue
      b = state[key.replace(".lora_A.", ".lora_B.")]
      lin = modules[key.split(".lora_A.")[0]]
      lin.weight.copy_((lin.weight.float() + scale * (b.float() @ a.float()))
                       .to(lin.weight.dtype))
      merged += 1
  print(f"  [pika] merged {merged} LoRA modules", flush=True)
  return manifest


def pika_scheduler(base, manifest: dict, shift_key: str, seed: int):
  """A copy of `base` (the pipeline's H3 scheduler) running Pika's sampler.

  The fixed knots / 1000 through this modality's shift, whatever
  `num_inference_steps` the pipeline asks for; each step takes
  x0 = x + sigma * v and re-noises it to the next level with fresh noise.
  """
  import numpy as np

  cls = type(base)

  class PikaRenoise(cls):
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
      q = np.asarray(manifest["sampling"]["knots"], np.float64) / 1000.0
      s = float(manifest["sampling"][shift_key])
      ladder = (s * q / (1 + (s - 1) * q)).astype(np.float32)
      super().set_timesteps(device=device, sigmas=torch.from_numpy(ladder))
      self._gen = torch.Generator().manual_seed(seed + 1)

    def step(self, model_output, timestep, sample, return_dict=True):
      if self._step_index is None:
        self._step_index = (self.index_for_timestep(timestep)
                            if self._begin_index is None else self._begin_index)
      i = self._step_index
      sigma, nxt = float(self.sigmas[i]), float(self.sigmas[i + 1])
      x0 = sample.float() + sigma * model_output.float()
      if nxt > 0.0:
        eps = torch.randn(x0.shape, generator=self._gen, dtype=torch.float32,
                          device="cpu")
        x0 = (1.0 - nxt) * x0 + nxt * eps.to(x0.device)
      self._step_index += 1
      out = x0.to(sample.dtype)
      if not return_dict:
        return (out,)
      from diffusers.schedulers.scheduling_minimax_h3 import (
          MiniMaxH3SchedulerOutput)
      return MiniMaxH3SchedulerOutput(prev_sample=out)

  sched = PikaRenoise.from_config(base.config)
  return sched
