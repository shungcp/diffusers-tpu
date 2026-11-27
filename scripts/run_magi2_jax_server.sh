#!/bin/bash
# Sand.ai MAGI-2-preview TPU-Resident Native JAX Server on TPU v7x-8.
#
# Keeps Stage 0 (Qwen3.5-27B + lazy Wan2.2), Stage 1 (113.93B Preview DiT),
# optional Stage 2 (6.72B 1080p Refiner), and Stage 3 (8-chip TurboVAED + Oobleck)
# resident in TPU HBM (~56.5 GiB / 94.74 GiB) across all requests in --requests JSONL.
#
#   bash scripts/run_magi2_jax_server.sh --requests reqs.jsonl --lossy
#   bash scripts/run_magi2_jax_server.sh --requests reqs.jsonl --resolution 1080p --enable-refiner
set -eu

CC=${MAGI2_COMPILE_CACHE:-$HOME/.cache/magi2/jax_cc_server}
mkdir -p "$CC"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_use_associative_scan_emitter=false --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_force_async_all_to_all=true --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"}

exec "${PYTHON:-python}" -u -m diffusers_tpu.models.magi2.jax_server \
  --compile-cache "$CC" \
  "$@"
