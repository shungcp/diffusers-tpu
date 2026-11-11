#!/bin/bash
# MiniMax-H3-Image (6-step distilled 2K/4K t2i & edit), torchax backend.
#
#   bash scripts/run_h3_image_torchax.sh                   # 2048x2048 t2i
#   bash scripts/run_h3_image_torchax.sh --size 4736x3552  # 4K (4:3) t2i
#   bash scripts/run_h3_image_torchax.sh --size 4096x4096 --reference ref.png
set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.torchax_image_pipeline \
  --chips 8 --steps 6 --seed 42 --warmup \
  --out out --tag h3_image_torchax "$@"
