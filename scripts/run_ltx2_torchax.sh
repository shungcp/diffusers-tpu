#!/bin/bash
# VideoGym-LTX-2.3 (22B Audio-Video DiT) through torchax (PyTorch DiT traced into JAX shard_map + Pallas Splash).
#
#   bash scripts/run_ltx2_torchax.sh --image canvas.png --prompt "Move the red dot along the white path to the green square."
#   bash scripts/run_ltx2_torchax.sh --image canvas.png --prompt "..." --fp8 --output ltx2_torchax_fp8.mp4
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$R/scripts/run_ltx2.sh" --dit-backend torchax "$@"
