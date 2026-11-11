#!/bin/bash
# MiniMax-H3-Image (6-step distilled 2K/4K t2i & edit), JAX backend.
#
#   bash scripts/run_h3_image_jax.sh                       # 2048x2048 t2i
#   bash scripts/run_h3_image_jax.sh --size 4736x3552      # 4K (4:3) t2i
#   bash scripts/run_h3_image_jax.sh --size 4096x4096 --reference ref.png  # 4K edit
set -eu
CC=${H3_IMAGE_COMPILE_CACHE:-$HOME/.cache/h3_image/jax_cc}
mkdir -p "$CC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_scoped_vmem_limit_kib=65536"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.jax_image_pipeline \
  --chips 8 --steps 6 --seed 42 --warmup --adaln-host \
  --compile-cache "$CC" --out out --tag h3_image_jax "$@"
