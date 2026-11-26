#!/bin/bash
# MAGI-2-preview (113.93B Multi-Head Latent MoE) through torch_tpu (8-process, 1/chip) on TPU v7x-8.
#
#   bash scripts/run_magi2_torchtpu.sh --mode bench --resolution 272p
#   bash scripts/run_magi2_torchtpu.sh --mode bench --resolution 540p --fp8
#   bash scripts/run_magi2_torchtpu.sh --mode denoise --resolution 540p --steps 100 \
#       --text-embeds-cache /workspace/shun/magi2_outputs/prompt_fox_embeds.npz
#   # Full end-to-end MP4 (Stage 0 Qwen3.5 encode -> Stage 1 torch_tpu 8-rank DiT -> Stage 3 8-chip TurboVAED+Oobleck):
#   bash scripts/run_magi2_torchtpu.sh --mode denoise --resolution 540p --fp8 --steps 100 \
#       --prompt "A barista smiling in a cozy cafe saying 'Careful, it is still very hot.'" \
#       --enhance --decode-mp4 --out /workspace/shun/magi2_outputs --tag magi2_tt_cafe
#
# Needs the torch_tpu venv (/workspace/venvs/venv_tt) for Stage 1 and optional JAX_PYTHON
# (/workspace/venvs/venv_052/bin/python) for Stage 0/3 when --prompt/--decode-mp4 is used.
# Env: PYTHON, JAX_PYTHON, MAGI2_CKPT_ROOT, DT_AOT_PERSIST (default 1), MAGI2_TT_CACHE.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)

export MAGI2_CKPT_ROOT=${MAGI2_CKPT_ROOT:-/workspace/shun/magi2_preview_weights}
TT_CACHE=${MAGI2_TT_CACHE:-$HOME/.cache/magi2/tt_aot}
if [ "${DT_AOT_PERSIST:-1}" != "0" ]; then
  mkdir -p "$TT_CACHE/aot" "$TT_CACHE/inductor" "$TT_CACHE/xla" "$TT_CACHE/jax"
  if [ "${DT_AOT_PERSIST:-1}" = "1" ]; then
    export DT_AOT_PERSIST="$TT_CACHE/aot"
  fi
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$TT_CACHE/inductor"}
  export XLA_PERSISTENT_CACHE_DIR=${XLA_PERSISTENT_CACHE_DIR:-"$TT_CACHE/xla"}
  export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-"$TT_CACHE/jax"}
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
fi

# Parse optional E2E Stage 0/3 wrapper flags while forwarding Stage 1 args to torchtpu_magi2.py
PROMPT=""
PROMPT_FILE=""
ENHANCE=0
DECODE_MP4=0
RES="272p"
OUT_DIR="/workspace/shun/magi2_outputs"
TAG="magi2_torchtpu"
TEXT_CACHE=""
PASS_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --enhance) ENHANCE=1; shift ;;
    --decode-mp4) DECODE_MP4=1; shift ;;
    --resolution) RES="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
    --out) OUT_DIR="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
    --tag) TAG="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
    --text-embeds-cache) TEXT_CACHE="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

JAX_PY=${JAX_PYTHON:-/workspace/venvs/venv_052/bin/python}
if [ ! -x "$JAX_PY" ]; then
  JAX_PY=${PYTHON:-python3}
fi

# Optional Stage 0 (Qwen3.5-27B text encode) before torch_tpu acquires libtpu
if [ -z "$TEXT_CACHE" ] && { [ -n "$PROMPT" ] || [ -n "$PROMPT_FILE" ]; }; then
  mkdir -p "$OUT_DIR"
  TEXT_CACHE="$OUT_DIR/${TAG}_text_embeds.npz"
  echo "[run_magi2_torchtpu] Stage 0: Encoding prompt to $TEXT_CACHE via $JAX_PY ..."
  S0_ARGS=(--model-dir "$MAGI2_CKPT_ROOT" --resolution "$RES" --text-embeds-cache "$TEXT_CACHE" --encode-only --compile-cache "${MAGI2_COMPILE_CACHE:-$TT_CACHE/jax}")
  if [ -n "$PROMPT_FILE" ]; then
    S0_ARGS+=(--prompt-file "$PROMPT_FILE")
  else
    S0_ARGS+=(--prompt "$PROMPT")
  fi
  [ "$ENHANCE" = "1" ] && S0_ARGS+=(--enhance)
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.magi2.jax_pipeline "${S0_ARGS[@]}"
  PASS_ARGS+=(--text-embeds-cache "$TEXT_CACHE")
fi

# Allocate 8 consecutive free ports for the TPU slice builder below the ephemeral range.
BASE=""
for b in $(seq 20000 20 31000 | shuf | head -200); do
  busy=0
  for j in $(seq 0 10); do
    (exec 3<>/dev/tcp/127.0.0.1/$((b + j))) 2>/dev/null && { busy=1; break; }
  done
  [ $busy = 0 ] && { BASE=$b; break; }
done
[ -n "$BASE" ] || { echo "no free port range" >&2; exit 1; }
export TORCH_TPU_SLICEBUILDER_ADDRESSES=$(for j in 0 1 2 3 4 5 6 7; do printf "localhost:%d," $((BASE + j)); done | sed 's/,$//')

export TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY:-"2,2,1,2"}
export TORCH_TPU_DEVICES_PER_HOST=${TORCH_TPU_DEVICES_PER_HOST:-8}
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--slicebuilder_use_insecure_grpc=true \
--xla_tpu_use_associative_scan_emitter=false \
--xla_tpu_force_async_all_to_all=true \
--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"}
export TPU_PREMAPPED_BUFFER_SIZE=${TPU_PREMAPPED_BUFFER_SIZE:-2147483648}
export DT_MAGI2_AG_CAP_FACTOR=${DT_MAGI2_AG_CAP_FACTOR:-2.0}
if [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}
fi
export PYTHONPATH="$R/compat/torchvision_stub:$R:/workspace/shun/tokamax${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON:-python3}" -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port $((BASE + 10)) \
  -m diffusers_tpu.backends.torchtpu_magi2 \
  --model-dir "$MAGI2_CKPT_ROOT" \
  "${PASS_ARGS[@]}"

# Optional Stage 3 (8-chip TurboVAED + Oobleck stereo audio -> playable MP4) after torch_tpu releases libtpu
if [ "$DECODE_MP4" = "1" ]; then
  LAT_NPZ="$OUT_DIR/${TAG}_stage1_latents.npz"
  if [ ! -f "$LAT_NPZ" ]; then
    echo "[run_magi2_torchtpu] ERROR: expected latents at $LAT_NPZ (pass --mode denoise with --decode-mp4)" >&2
    exit 1
  fi
  echo "[run_magi2_torchtpu] Stage 3: Decoding $LAT_NPZ -> $OUT_DIR/${TAG}.mp4 (8-chip TurboVAED + Oobleck) ..."
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.magi2.jax_pipeline \
    --model-dir "$MAGI2_CKPT_ROOT" \
    --resolution "$RES" \
    --load-stage1-latents "$LAT_NPZ" \
    --no-refiner \
    --out "$OUT_DIR" \
    --tag "$TAG" \
    --compile-cache "${MAGI2_COMPILE_CACHE:-$TT_CACHE/jax}"
fi
