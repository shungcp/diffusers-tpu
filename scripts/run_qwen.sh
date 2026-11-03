#!/bin/bash
# Qwen-Image-2.1 text-to-image (1024x1024, 20 steps by default) on a TPU slice.
#
#   bash scripts/run_qwen.sh jax      --prompt "..." [--num_inference_steps 4] [--image cond.png]
#   bash scripts/run_qwen.sh torchax  --prompt "..."
#   bash scripts/run_qwen.sh torchtpu --prompt "..."     # 8 chips (CHIPS=1 for one), torch_tpu environment
#
# Writes --output (default qwen_image_21_<backend>.png). Env: PYTHON, HF_HOME.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
BACKEND=${1:?usage: run_qwen.sh jax|torchax|torchtpu [args...]}; shift
case "$BACKEND" in
  jax)
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    exec "${PYTHON:-python}" -u -m diffusers_tpu.models.qwen_image_21.jax_pipeline \
      --output qwen_image_21_jax.png "$@" ;;
  torchax)
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    exec "${PYTHON:-python}" -u -m diffusers_tpu.backends.torchax_qwen21 \
      --output qwen_image_21_torchax.png "$@" ;;
  torchtpu)
    # Eight processes: tensor-parallel prefill, then every cached step as one AOT
    # program in Ulysses sequence parallel; VAE decoder AOT, Qwen3-VL sharded on the
    # chips. CHIPS=1 runs a single chip.
    export PYTHONPATH=$R/compat/torchvision_stub${PYTHONPATH:+:$PYTHONPATH}
    CHIPS=${CHIPS:-8}
    if [ "$CHIPS" = 1 ]; then
      exec "${PYTHON:-python}" -u -m diffusers_tpu.backends.torchtpu_qwen21 \
        --output qwen_image_21_torchtpu.png "$@"
    fi
    BASE=""
    for b in $(seq 20000 20 31000 | shuf | head -200); do
      busy=0
      for j in $(seq 0 10); do (exec 3<>/dev/tcp/127.0.0.1/$((b + j))) 2>/dev/null && { busy=1; break; }; done
      [ $busy = 0 ] && { BASE=$b; break; }
    done
    [ -n "$BASE" ] || { echo "no free port range" >&2; exit 1; }
    export TORCH_TPU_SLICEBUILDER_ADDRESSES=$(for j in $(seq 0 $((CHIPS - 1))); do printf "localhost:%d," $((BASE + j)); done | sed 's/,$//')
    exec "${PYTHON:-python}" -m torch.distributed.run --nproc_per_node "$CHIPS" --master_port $((BASE + 10)) \
      -m diffusers_tpu.backends.torchtpu_qwen21 --ulysses --aot-prefill --aot-vae --ours-attn \
      --output qwen_image_21_torchtpu.png "$@" ;;
  *) echo "backend must be jax, torchax or torchtpu" >&2; exit 2 ;;
esac
