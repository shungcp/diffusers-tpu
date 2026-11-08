#!/bin/bash
# MiniMax-H3 t2va through torchax (PyTorch model code, JAX execution), 8 chips, 768x1344, 345 frames.
#
#   bash scripts/run_h3_torchax.sh            # lossy: step cache 0.20 + sparse window 16
#   bash scripts/run_h3_torchax.sh --cache-threshold 0 --sparse-window 0 --block-kv 4096   # lossless
#   bash scripts/run_h3_torchax.sh --workflow ref2va --reference image.png --cache-threshold 0.15
#
# Env: PYTHON, HF_HOME. Later arguments override the defaults below.
set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_ici_ag_pipelining=true \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_tpu_scoped_vmem_limit_kib=98304 \
--xla_tpu_enable_offloading_copy_to_sparsecore=false \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.torchax_pipeline \
  --num-frames 345 --steps 20 \
  --fixed-m --warmup --scoped-vmem-kib 98304 --full-lane-rope --timestep-pad \
  --jax-vae \
  --cache-threshold 0.20 --sparse-window 16 --sparse-dense-steps 6 \
  --out out --tag torchax "$@"
