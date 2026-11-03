#!/bin/bash
# MiniMax-H3 on a TPU slice, JAX backend (default: 8 chips, 768x1344, 345 frames, 20 steps).
#
#   bash scripts/run_h3_jax.sh t2va   [--lossy] [jax_pipeline args...]
#   bash scripts/run_h3_jax.sh ref2va --reference image.png [--lossy] [args...]
#
# --lossy: step cache (t2va 0.20 / ref2va 0.15) + temporal-window sparse attention (window 16,
#          first 6 calls dense). Later arguments override the defaults below.
# Env: PYTHON (interpreter, default python), H3_COMPILE_CACHE (persistent compile cache,
#      default ~/.cache/h3/jax_cc_<workflow>), HF_HOME (Hugging Face cache with MiniMaxAI/MiniMax-H3).
set -eu
WF=${1:?usage: run_h3_jax.sh t2va|ref2va [--lossy] [args...]}; shift
case "$WF" in t2va|ref2va) ;; *) echo "workflow must be t2va or ref2va" >&2; exit 2 ;; esac
REST=(); IS_LOSSY=0
for x in "$@"; do [ "$x" = --lossy ] && IS_LOSSY=1 || REST+=("$x"); done
set -- ${REST[@]+"${REST[@]}"}
LOSSY=()
if [ $IS_LOSSY = 1 ]; then
  if [ "$WF" = t2va ]; then
    LOSSY=(--cache-threshold 0.20 --sparse-window 16 --sparse-dense-steps 6)
  else
    LOSSY=(--cache-threshold 0.15 --sparse-window 16 --sparse-dense-steps 6)
  fi
fi
if [ "$WF" = ref2va ]; then
  case " $* " in *" --reference "*) ;; *) echo "ref2va needs --reference <image>" >&2; exit 2 ;; esac
fi
CC=${H3_COMPILE_CACHE:-$HOME/.cache/h3/jax_cc_$WF}
mkdir -p "$CC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.jax_pipeline \
  --height 768 --width 1344 --num-frames 345 --steps 20 --chips 8 \
  --prompt "a red fox trotting through snow" --seed 42 \
  --warmup --torch-noise --vae-tile-overlap-px 32 --compile-cache "$CC" \
  ${LOSSY[@]+"${LOSSY[@]}"} "$@"
