#!/bin/bash
# MiniMax-H3 through torch_tpu (eager PyTorch on TPU, one process per chip), 8 chips, 768x1344, 345 frames.
#
#   bash scripts/run_h3_torchtpu.sh t2va   [torchtpu_h3 args...]
#   bash scripts/run_h3_torchtpu.sh ref2va --reference image.png [args...]
#   lossy: add --cache-threshold 0.20 --sparse-window 16 --sparse-dense-steps 6 (ref2va: 0.15)
#
# Needs the torch_tpu environment (requirements/torchtpu.txt). Env: PYTHON, HF_HOME,
# TORCH_TPU_TOPOLOGY (auto-detected: v6e-8 "2,4,1", v7x-8 "2,2,1,2").
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
WF=${1:?usage: run_h3_torchtpu.sh t2va|ref2va [args...]}; shift
EXTRA=()
case "$WF" in
  # Text encoder tensor-parallel on the chips (8 x 7.3 GiB) instead of on
  # the host, freed after encoding so the lossless denoise still fits:
  # Pika 768p t2va 71.1 -> 51.9 s, lossy 52.5 -> 49.8 s, video DINO 0.967.
  # The VAE stays resident (1-2 s less a request); ref2va, which also holds
  # the reference encoder, does not fit that way and moves it per request.
  t2va) EXTRA=(--shard-text --free-text-after --seg-mod) ;;
  ref2va) EXTRA=(--free-text-after --vae-on-demand)
    case " $* " in *" --reference "*) ;; *) echo "ref2va needs --reference <image>" >&2; exit 2 ;; esac ;;
  *) echo "workflow must be t2va or ref2va" >&2; exit 2 ;;
esac
# 8 consecutive free ports for the slice builder, below the kernel's ephemeral range so an
# outgoing connection cannot grab one between probing and binding.
BASE=""
for b in $(seq 20000 20 31000 | shuf | head -200); do
  busy=0
  for j in $(seq 0 10); do (exec 3<>/dev/tcp/127.0.0.1/$((b + j))) 2>/dev/null && { busy=1; break; }; done
  [ $busy = 0 ] && { BASE=$b; break; }
done
[ -n "$BASE" ] || { echo "no free port range" >&2; exit 1; }
export TORCH_TPU_SLICEBUILDER_ADDRESSES=$(for j in 0 1 2 3 4 5 6 7; do printf "localhost:%d," $((BASE + j)); done | sed 's/,$//')
# torch_tpu environments usually lack torchvision; diffusers only needs a few names from it.
export PYTHONPATH=$R/compat/torchvision_stub${PYTHONPATH:+:$PYTHONPATH}
export DT_SPMD_SAFE=1 DT_DECODE_SHM=1 DT_OURS_BQ=2048 DT_OURS_BKV=4096
# Lossy runs compile a dense and a sparse DiT; call each compiled artifact directly so they do not
# share one Dynamo frame cache. Lossless runs do not need it (and are ~2 s faster without).
case " $* " in *" --sparse-window "*) export DT_AOT_CALL_ART=1 ;; esac
exec "${PYTHON:-python}" -m torch.distributed.run --nproc_per_node 8 --master_port $((BASE + 10)) \
  -m diffusers_tpu.backends.torchtpu_h3 --workflow "$WF" --runs 2 \
  --num-frames 345 --max-duration 15 --height 768 --width 1344 --steps 20 --vae-clip 0 \
  --shard-decode --shard-adaln --aot-dit --ours-attn --text-layers 51 --aot-vae \
  --no-offload --aot-group 10 --prefetch 1 \
  --out out --tag "torchtpu_$WF" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
