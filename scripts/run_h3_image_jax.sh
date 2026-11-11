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
# An image is a short sequence (4k-16k rows): on v6e keep the residual stream
# whole so each block costs two all-reduces (with the attention output kept
# split by heads and SwiGLU rows interleaved, both keyed on this), instead of
# the partitioner's sequence-sharded stream and its gathers (v6e-8,
# 4736x3552: denoise 7.42 -> 2.99 s). Measured on v6e only, where all-reduces
# do not overlap compute; other TPUs keep the partitioner's choice unless set.
ACCEL=${TPU_ACCELERATOR_TYPE:-$(curl -s -m 2 -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || true)}
case "$ACCEL" in v6e*) export DT_STREAM_SHARD=${DT_STREAM_SHARD:-whole} ;; esac
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_scoped_vmem_limit_kib=65536"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.jax_image_pipeline \
  --chips 8 --steps 6 --seed 42 --warmup --adaln-host --full-lane-rope --segments \
  --compile-cache "$CC" --out out --tag h3_image_jax "$@"
