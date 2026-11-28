#!/bin/bash
# MOVA (360p / 720p) through torchax (PyTorch Dual-Tower DiT traced into fused 40L JAX shard_map + Pallas Splash).
#
#   bash scripts/run_mova_torchax.sh --repo OpenMOSS-Team/MOVA-720p --height 720 --width 1280 --fp8 --lossy --image ref.jpg --prompt "..."
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$R/scripts/run_mova.sh" --dit-backend torchax "$@"
