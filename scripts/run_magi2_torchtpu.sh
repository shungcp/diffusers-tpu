#!/bin/bash
# MAGI-2-preview (113.93B Multi-Head Latent MoE) through torch_tpu (8-process, 1/chip) on TPU v7x-8.
#
#   bash scripts/run_magi2_torchtpu.sh --mode bench --resolution 272p
#   bash scripts/run_magi2_torchtpu.sh --mode bench --resolution 540p --fp8
#   bash scripts/run_magi2_torchtpu.sh --mode denoise --resolution 540p --steps 100 \
#       --text-embeds-cache /workspace/shun/magi2_outputs/prompt_fox_embeds.npz
#
# Needs the torch_tpu venv (/workspace/venvs/venv_tt). Env: PYTHON, MAGI2_CKPT_ROOT.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)

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
export MAGI2_CKPT_ROOT=${MAGI2_CKPT_ROOT:-/workspace/shun/magi2_preview_weights}
if [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}
fi
export PYTHONPATH="$R/compat/torchvision_stub:$R:/workspace/shun/tokamax${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON:-python3}" -m torch.distributed.run \
  --nproc_per_node 8 \
  --master_port $((BASE + 10)) \
  -m diffusers_tpu.backends.torchtpu_magi2 \
  --model-dir "$MAGI2_CKPT_ROOT" \
  "$@"
