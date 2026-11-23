#!/bin/bash
# torch_tpu overlap probe on 8 ranks (scripts/probe_tt_overlap.py). Env: PYTHON.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
BASE=$(( 20000 + RANDOM % 10000 ))
export TORCH_TPU_SLICEBUILDER_ADDRESSES=$(for j in 0 1 2 3 4 5 6 7; do printf "localhost:%d," $((BASE + j)); done | sed 's/,$//')
exec "${PYTHON:-python}" -m torch.distributed.run --nproc_per_node 8 --master_port $((BASE + 10)) \
  "$R/scripts/probe_tt_${PROBE:-overlap}.py" "$@"
