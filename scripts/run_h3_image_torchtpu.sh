#!/bin/bash
# MiniMax-H3-Image (6-step distilled 2K/4K t2i & edit), torch_tpu backend (8 chips/devices).
#
#   bash scripts/run_h3_image_torchtpu.sh                  # 2048x2048 t2i
#   bash scripts/run_h3_image_torchtpu.sh --size 4736x3552 # 4K (4:3) t2i
#   bash scripts/run_h3_image_torchtpu.sh --size 4096x4096 --reference ref.png
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
BASE=""
for b in $(seq 20000 20 31000 | shuf | head -200); do
  busy=0
  for j in $(seq 0 10); do (exec 3<>/dev/tcp/127.0.0.1/$((b + j))) 2>/dev/null && { busy=1; break; }; done
  [ $busy = 0 ] && { BASE=$b; break; }
done
[ -n "$BASE" ] || { echo "no free port range" >&2; exit 1; }
export TORCH_TPU_SLICEBUILDER_ADDRESSES=$(for j in 0 1 2 3 4 5 6 7; do printf "localhost:%d," $((BASE + j)); done | sed 's/,$//')
export PYTHONPATH=$R/compat/torchvision_stub:$R/compat/diffusers_h3_image${PYTHONPATH:+:$PYTHONPATH}
export DT_SPMD_SAFE=1 DT_OURS_BQ=2048 DT_OURS_BKV=4096
exec "${PYTHON:-python}" -m torch.distributed.run --nproc_per_node 8 --master_port $((BASE + 10)) \
  -m diffusers_tpu.backends.torchtpu_h3_image \
  --steps 6 --seed 42 --aot-group 10 --aot-vae-group 12 \
  --out out --tag h3_image_torchtpu "$@"
