#!/bin/bash
# Resident server (JAX backend): load once, then serve every request in a JSONL file.
#
#   bash scripts/run_h3_jax_server.sh requests.jsonl [--warmup warmup.jsonl] [--out DIR] [jax_server args...]
#
# One request per line: {"tag", "prompt", "reference"?, "seed"?, "lossy"?, "num_frames"?, "height"?, "width"?}
# Env: PYTHON, H3_COMPILE_CACHE (default ~/.cache/h3/jax_cc_server), HF_HOME.
set -eu
REQ=${1:?usage: run_h3_jax_server.sh requests.jsonl [args...]}; shift
CC=${H3_COMPILE_CACHE:-$HOME/.cache/h3/jax_cc_server}
mkdir -p "$CC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=true"}
exec "${PYTHON:-python}" -u -m diffusers_tpu.models.minimax_h3.jax_server \
  --requests "$REQ" --compile-cache "$CC" "$@"
