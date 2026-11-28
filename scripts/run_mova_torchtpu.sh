#!/bin/bash
# MOVA (360p / 720p) through native torch_tpu (8-process, 2x4 CFG x Ulysses SP) on TPU v7x-8.
#
#   bash scripts/run_mova_torchtpu.sh --repo /path/to/MOVA-720p --height 720 --width 1280 \
#     --fp8 --lossy --image ref.jpg --prompt "..." --output mova_tt_720p.mp4
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)

REPO=${MOVA_REPO:-OpenMOSS-Team/MOVA-720p}
TT_CACHE=${MOVA_TT_CACHE:-$HOME/.cache/mova/tt_aot}
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
HEIGHT=720
WIDTH=1280
FRAMES=193
FPS=24
STEPS=50
SHIFT=5.0
SEED=42
OUTPUT="mova_torchtpu.mp4"
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
    --shift) SHIFT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
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

# Stage 0: UMT5 + Wan2.1 VAE first-frame encode (skips 31B DiT load!)
if [ "$MODE" = "denoise" ] && [ -z "$STAGE0_NPZ" ]; then
  if [ -z "$IMAGE" ] || [ -z "$PROMPT" ]; then
    echo "[run_mova_torchtpu] ERROR: --image and --prompt required for --mode denoise (or pass --stage0-npz)" >&2
    exit 1
  fi
  STAGE0_NPZ="$OUT_DIR/${STEM}_stage0.npz"
  echo "[run_mova_torchtpu] Stage 0: Encoding prompt + image to $STAGE0_NPZ via $JAX_PY ..."
  S0_ARGS=(
    --repo "$REPO" --image "$IMAGE" --prompt "$PROMPT"
    --height "$HEIGHT" --width "$WIDTH" --frames "$FRAMES" --fps "$FPS"
    --steps "$STEPS" --shift "$SHIFT" --seed "$SEED"
    --encode-only "$STAGE0_NPZ" --compile-cache-dir "$TT_CACHE/jax"
  )
  [ -n "$NEG_PROMPT" ] && S0_ARGS+=(--negative-prompt "$NEG_PROMPT")
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.mova.pipeline "${S0_ARGS[@]}"
fi

SNAP_DIR="$REPO"
if [ ! -d "$SNAP_DIR/video_dit" ]; then
  SNAP_DIR=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$JAX_PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', local_files_only=True))" 2>/dev/null || echo "$REPO")
fi

# Allocate 8 consecutive free ports for TPU slice builder
BASE=""
for b in $(seq 20000 20 31000 | shuf | head -200); do
  if ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE ":(${b}|$((b+1))|$((b+2))|$((b+3))|$((b+4))|$((b+5))|$((b+6))|$((b+7)))$"; then
    BASE=$b
    break
  fi
done
[ -n "$BASE" ] || BASE=26800

ADDRS="localhost:${BASE}"
for i in 1 2 3 4 5 6 7; do
  ADDRS="${ADDRS},localhost:$((BASE + i))"
done
RDZV_PORT=$((BASE + 11))

export TORCH_TPU_SLICEBUILDER_ADDRESSES="$ADDRS"
export TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY:-"2,2,1,2"}
export TORCH_TPU_DEVICES_PER_HOST=8
export JAX_PLATFORMS=cpu
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"

TT_ARGS=(--repo "$SNAP_DIR" --mode "$MODE" --latents-out "$LATENTS_NPZ")
[ -n "$STAGE0_NPZ" ] && TT_ARGS+=(--stage0-npz "$STAGE0_NPZ")
[ ${#PASS_ARGS[@]} -gt 0 ] && TT_ARGS+=("${PASS_ARGS[@]}")

echo "[run_mova_torchtpu] Stage 1: 8-rank torch_tpu DiT ($MODE) via $TT_PY ..."
"$TT_PY" -m torch.distributed.run \
  --nproc_per_node=8 --nnodes=1 --rdzv_endpoint="localhost:${RDZV_PORT}" \
  -m diffusers_tpu.backends.torchtpu_mova "${TT_ARGS[@]}"

if [ "$MODE" = "denoise" ]; then
  echo "[run_mova_torchtpu] Stage 3: Decoding $LATENTS_NPZ -> $OUTPUT via $JAX_PY ..."
  env -u JAX_PLATFORMS TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$R" "$JAX_PY" -u -m diffusers_tpu.models.mova.pipeline \
    --repo "$REPO" --height "$HEIGHT" --width "$WIDTH" --frames "$FRAMES" --fps "$FPS" \
    --decode-latents "$LATENTS_NPZ" --output "$OUTPUT" --compile-cache-dir "$TT_CACHE/jax"
fi
