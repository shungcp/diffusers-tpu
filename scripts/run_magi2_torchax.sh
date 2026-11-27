#!/bin/bash
# MAGI-2-preview (113.93B Multi-Head Latent MoE) through torchax on TPU v7x-8.
#
#   bash scripts/run_magi2_torchax.sh --resolution 540p --steps 100 --warmup
#   bash scripts/run_magi2_torchax.sh --resolution 540p --fp8 --steps 100 --warmup
#
# Env: PYTHON, MAGI2_CKPT_ROOT, JAX_COMPILATION_CACHE_DIR.
set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_use_associative_scan_emitter=false \
--xla_tpu_force_async_all_to_all=true \
--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"}
export DT_MAGI2_AG_CAP_FACTOR=${DT_MAGI2_AG_CAP_FACTOR:-2.0}
export MAGI2_CKPT_ROOT=${MAGI2_CKPT_ROOT:-/workspace/shun/magi2_preview_weights}
exec "${PYTHON:-python3}" -u -m diffusers_tpu.models.magi2.torchax_pipeline \
  --model-dir "$MAGI2_CKPT_ROOT" \
  --preview-backend torchax \
  "$@"
