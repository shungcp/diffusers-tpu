#!/bin/bash
# MOVA-360p / MOVA-720p image + text -> video with synchronized audio (50 steps, CFG 5), JAX + torchax.
#
#   bash scripts/run_mova.sh --image ref.jpg --prompt "..."                  # 352x640, 193 frames (8 s), all chips
#   bash scripts/run_mova.sh --image ref.jpg --prompt "..." --cfg 2          # CFG branches on two chip groups (8 chips)
#   bash scripts/run_mova.sh --repo OpenMOSS-Team/MOVA-720p --height 720 --width 1280 --image ref.jpg --prompt "..."
#   bash scripts/run_mova.sh --repo OpenMOSS-Team/MOVA-720p --height 720 --width 1280 --fp8 --qk-fp8 --image ref.jpg --prompt "..."
#   bash scripts/run_mova.sh --repo OpenMOSS-Team/MOVA-720p --height 720 --width 1280 --lossy --image ref.jpg --prompt "..."
#
# Writes --output (default mova.mp4, H.264 + AAC 48 kHz). Env: PYTHON, HF_HOME, LIBTPU_INIT_ARGS.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_force_async_all_to_all=true}"
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.mova.pipeline "$@"
