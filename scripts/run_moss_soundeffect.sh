#!/bin/bash
# MOSS-SoundEffect-v2 text-to-audio (100 steps, CFG 4, 48 kHz mono), JAX + torchax.
#
#   bash scripts/run_moss_soundeffect.sh --prompt "..." --seconds 10          # all chips: CFG branches x sequence halves
#   bash scripts/run_moss_soundeffect.sh --prompt "..." --chips 1
#   bash scripts/run_moss_soundeffect.sh --requests reqs.jsonl --out out      # resident: one JSON line per request
#   bash scripts/run_moss_soundeffect.sh --requests reqs.jsonl --replicas 4   # four model copies (most throughput)
#
# Writes --output (default moss_soundeffect.wav). Env: PYTHON, HF_HOME.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.moss_soundeffect.pipeline "$@"
