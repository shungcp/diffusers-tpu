#!/bin/bash
# OmniVAE text -> video + audio (t2av_recon_distill_avclip: 256x256, 121 frames, 48 kHz audio; 50 steps), JAX + torchax.
#
#   bash scripts/run_omnivae.sh --prompt "..."                      # all chips (video rows split over them), dual CFG
#   bash scripts/run_omnivae.sh --prompt "..." --cfg-mode simple    # 2 forwards a step instead of 3
#   bash scripts/run_omnivae.sh --prompt "..." --chips 1
#
# Writes --output (default omnivae.mp4, H.264 + AAC). Env: PYTHON, OMNIVAE_RELEASE_ROOT (or HF_HOME).
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.omnivae.pipeline "$@"
