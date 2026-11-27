"""Priority 2 (`torchax`) CLI entry point for ``MAGI-2-preview`` on TPU.

Runs the full 3-stage MAGI-2-preview pipeline (Qwen3.5-27B text encoder +
Stage 1 113.93B Preview DiT via :mod:`torchax_preview_dit` + optional Stage 2
6.72B 1080p Refiner + Stage 3 TurboVAED & Oobleck stereo audio VAE).

Example::

    python -u -m diffusers_tpu.models.magi2.torchax_pipeline \\
        --resolution 540p --steps 100 --warmup --save-latents \\
        --out /workspace/shun/magi2_outputs --tag magi2_torchax_540p
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> None:
  os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
  raw = list(sys.argv[1:] if argv is None else argv)
  if "--preview-backend" not in raw:
    raw = ["--preview-backend", "torchax", *raw]
  from .jax_pipeline import main as jax_pipeline_main

  jax_pipeline_main(raw)


if __name__ == "__main__":
  main()
