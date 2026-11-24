#!/bin/bash
# MOVA-360p image + text -> video with synchronized audio (50 steps, CFG 5), JAX + torchax.
#
#   bash scripts/run_mova.sh --image ref.jpg --prompt "..."                  # 352x640, 193 frames (8 s), all chips
#   bash scripts/run_mova.sh --image ref.jpg --prompt "..." --cfg 2          # CFG branches on two chip groups (8 chips)
#
# Writes --output (default mova.mp4, H.264 + AAC 48 kHz). Env: PYTHON, HF_HOME.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.mova.pipeline "$@"
