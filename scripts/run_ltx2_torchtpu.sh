#!/bin/bash
# VideoGym-LTX-2.3 (22B Audio-Video DiT) through torch_tpu (8-process, 4x2 CFG-parallel) on TPU v7x-8.
#
#   bash scripts/run_ltx2_torchtpu.sh --image canvas.png --prompt "Move the red dot along the white path to the green square."
#   bash scripts/run_ltx2_torchtpu.sh --image canvas.png --prompt "..." --fp8 --output ltx2_tt_fp8.mp4
#   bash scripts/run_ltx2_torchtpu.sh --mode bench --fp8
#
# Uses JAX_PYTHON (/workspace/venvs/venv_052/bin/python) for Stage 0 (Gemma-3 + video encode) and
# Stage 3 (on-TPU uint8 video decode + audio vocoder), and PYTHON (/workspace/venvs/venv_tt/bin/python)
# for Stage 1 (8-rank torch_tpu 48-block DiT denoise with DT_AOT_PERSIST).
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)

REPO=${LTX2_REPO:-OpenMOSS-Team/VideoGym-LTX-2.3}
TT_CACHE=${LTX2_TT_CACHE:-$HOME/.cache/ltx2/tt_aot}
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

MODE="denoise"
IMAGE=""
PROMPT=""
NEG_PROMPT=""
HEIGHT=512
WIDTH=768
FRAMES=129
FPS=24
STEPS=30
SEED=10
CRF=""
OUTPUT="ltx2_torchtpu.mp4"
STAGE0_NPZ=""
LATENTS_NPZ=""
PASS_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --negative-prompt) NEG_PROMPT="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --crf) CRF="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --stage0-npz) STAGE0_NPZ="$2"; shift 2 ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

JAX_PY=${JAX_PYTHON:-/workspace/venvs/venv_052/bin/python}
[ -x "$JAX_PY" ] || JAX_PY=${PYTHON:-python3}
TT_PY=${PYTHON:-/workspace/venvs/venv_tt/bin/python}
[ -x "$TT_PY" ] || TT_PY=python3

OUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUT_DIR"
STEM=$(basename "$OUTPUT" .mp4)
[ -n "$LATENTS_NPZ" ] || LATENTS_NPZ="$OUT_DIR/${STEM}_latents.npz"

# Stage 0: Gemma-3 + connectors + video VAE encode (skips 44 GB DiT load!)
if [ "$MODE" = "denoise" ] && [ -z "$STAGE0_NPZ" ]; then
  if [ -z "$IMAGE" ] || [ -z "$PROMPT" ]; then
    echo "[run_ltx2_torchtpu] ERROR: --image and --prompt required for --mode denoise (or pass --stage0-npz)" >&2
    exit 1
  fi
  STAGE0_NPZ="$OUT_DIR/${STEM}_stage0.npz"
  echo "[run_ltx2_torchtpu] Stage 0: Encoding prompt + image to $STAGE0_NPZ via $JAX_PY ..."
  S0_ARGS=(
    --repo "$REPO" --image "$IMAGE" --prompt "$PROMPT" --negative-prompt "$NEG_PROMPT"
    --height "$HEIGHT" --width "$WIDTH" --frames "$FRAMES" --fps "$FPS"
    --steps "$STEPS" --seed "$SEED" --encode-only "$STAGE0_NPZ"
    --compile-cache-dir "$TT_CACHE/jax"
  )
  [ -n "$CRF" ] && S0_ARGS+=(--crf "$CRF")
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.ltx2.pipeline "${S0_ARGS[@]}"
fi

# Resolve HF repo to local snapshot dir if needed for torch_tpu DiT loader
SNAP_DIR="$REPO"
if [ ! -d "$SNAP_DIR/transformer" ]; then
  SNAP_DIR=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$JAX_PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', local_files_only=True))" 2>/dev/null || echo "$REPO")
fi

# Allocate 8 consecutive free ports for TPU slice builder
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
--xla_tpu_dvfs_p_state=7 \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_force_async_all_to_all=true \
--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true \
--xla_tpu_enable_dot_strength_reduction=true"}
export TPU_PREMAPPED_BUFFER_SIZE=${TPU_PREMAPPED_BUFFER_SIZE:-2147483648}
if [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}
fi
unset TORCH_DEVICE_BACKEND_AUTOLOAD || true
export PYTHONPATH="$R/compat/torchvision_stub:$R${PYTHONPATH:+:$PYTHONPATH}"

# Stage 1: 8-rank torch_tpu 48-block DiT denoise (or bench)
TT_ARGS=(--mode "$MODE" --repo "$SNAP_DIR" --latents-out "$LATENTS_NPZ")
[ -n "$STAGE0_NPZ" ] && TT_ARGS+=(--stage0-npz "$STAGE0_NPZ")
"$TT_PY" -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port $((BASE + 10)) \
  -m diffusers_tpu.backends.torchtpu_ltx2 \
  "${TT_ARGS[@]}" "${PASS_ARGS[@]}"

# Stage 3: On-TPU uint8 video decode + audio decode -> MP4 (skips Gemma-3 & 44 GB DiT!)
if [ "$MODE" = "denoise" ]; then
  echo "[run_ltx2_torchtpu] Stage 3: Decoding $LATENTS_NPZ -> $OUTPUT via $JAX_PY ..."
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.ltx2.pipeline \
    --repo "$SNAP_DIR" --image "${IMAGE:-/dev/null}" --prompt "${PROMPT:-}" \
    --height "$HEIGHT" --width "$WIDTH" --frames "$FRAMES" --fps "$FPS" \
    --steps "$STEPS" --decode-latents "$LATENTS_NPZ" --output "$OUTPUT" \
    --compile-cache-dir "$TT_CACHE/jax"
fi
