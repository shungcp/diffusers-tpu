#!/bin/bash
# MOSS-Image text-to-image (MOSS-Image-Base, 50 steps, CFG 3), torchax backend.
#
#   bash scripts/run_moss_image.sh --prompt "..."                    # 1024x1024, all chips
#   bash scripts/run_moss_image.sh --prompt "..." --size 2048        # 2048x2048
#   bash scripts/run_moss_image.sh --prompt "..." --chips 1          # one chip
#   bash scripts/run_moss_image.sh --requests reqs.jsonl --out out   # resident: one JSON line per request
#   bash scripts/run_moss_image.sh --requests reqs.jsonl --replicas 2  # two model copies, half the chips each
#   bash scripts/run_moss_image.sh --dmd --prompt "..."              # 8-step DMD adapter on MOSS-Image-SFT
#
# With an even chip count the positive and negative CFG branches run side by
# side on two halves of the chips (--cfg 1 turns it off). Writes --output
# (default moss_image.png). Env: PYTHON, HF_HOME.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.moss_image.torchax_pipeline \
  --output moss_image.png "$@"
