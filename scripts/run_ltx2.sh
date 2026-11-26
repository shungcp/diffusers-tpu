#!/bin/bash
# VideoGym-LTX-2.3 image + text -> video (+ audio), JAX + torchax. Official settings by default:
# 768x512 (a 512x512 puzzle centered on a black canvas), 129 frames at 24 fps, 30 steps, CFG 3 + STG 1 (block 28),
# guidance rescale 0.7.
#
#   bash scripts/run_ltx2.sh --image canvas.png --prompt "..."
#
# Writes --output (default ltx2.mp4). Env: PYTHON, HF_HOME.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.ltx2.pipeline "$@"
