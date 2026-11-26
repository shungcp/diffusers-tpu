#!/bin/bash
# Sand.ai MAGI-2-preview on TPU v7x-8, native JAX backend.
#
#   bash scripts/run_magi2_jax.sh 272p|540p|1080p [jax_pipeline args...]
#   bash scripts/run_magi2_jax.sh t2v  [272p|540p|1080p] [jax_pipeline args...]
#   bash scripts/run_magi2_jax.sh i2v  [272p|540p|1080p] --image ref.png [jax_pipeline args...]
#
# Optional flags (passed through or overriding defaults):
#   --fp8                  W8A8 E4M3 MoE quantization on TPU v7x
#   --moe-mode ag|ep|ref   Multi-Head Latent MoE dispatch mode (default: auto -> ag)
#   --steps N              Stage 1 denoising steps (default: 100)
#   --no-refiner           Skip Stage 2 1080p Refiner
#   --warmup --runs N      Steady-state benchmarking keeping Stage 1 resident
#   --text-embeds-cache F  Reuse precomputed Qwen3.5-27B text embeddings (.npz)
#   --profile-dir DIR      Capture TPU XProf trace on steady step
#
# Env:
#   PYTHON               Python interpreter (default: python)
#   MAGI2_CKPT_ROOT      Checkpoint root (preview/, refiner/, text_encoder/, vae/,
#                        turbo_vae/, stable-audio-open-1.0/)
#   MAGI2_COMPILE_CACHE  JAX persistent compile cache dir
#                        (default: ~/.cache/magi2/jax_cc_<res>)
#   LIBTPU_INIT_ARGS     XLA/TPU compiler flags (defaults to DVFS p-state 7 +
#                        latency-hiding scheduler + async collectives)
set -eu

WF="t2v"
RES="540p"
if [ $# -gt 0 ]; then
  case "$1" in
    t2v|i2v)
      WF="$1"; shift
      if [ $# -gt 0 ]; then
        case "$1" in
          272p|540p|1080p) RES="$1"; shift ;;
        esac
      fi
      ;;
    272p|540p|1080p)
      RES="$1"; shift
      ;;
    -h|--help)
      exec "${PYTHON:-python}" -u -m diffusers_tpu.models.magi2.jax_pipeline --help
      ;;
  esac
fi

if [ "$WF" = i2v ]; then
  case " $* " in
    *" --image "*) ;;
    *) echo "i2v workflow requires --image <path>" >&2; exit 2 ;;
  esac
fi

CC=${MAGI2_COMPILE_CACHE:-$HOME/.cache/magi2/jax_cc_${RES}}
mkdir -p "$CC"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_use_associative_scan_emitter=false --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true --xla_tpu_force_async_all_to_all=true --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"}

exec "${PYTHON:-python}" -u -m diffusers_tpu.models.magi2.jax_pipeline \
  --resolution "$RES" \
  --seed 42 \
  --torch-noise \
  --compile-cache "$CC" \
  "$@"
