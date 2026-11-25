# JAX / torchax MiniMax-H3 and H3-Image: environment variables / 环境变量

`scripts/run_h3_jax.sh`, `run_h3_jax_server.sh`, `run_h3_torchax.sh` and the
`run_h3_image_*.sh` scripts need none of these; defaults are the tested
configuration. Most have a command-line flag; prefer it.

这些启动脚本不需要设置下列任何变量,默认值就是测过的配置;多数有对应的命令行参数,优先用参数。

## Options / 可选功能

| Variable | Default | Meaning |
|---|---|---|
| `DT_PALLAS_FFN` (`--pallas-ffn`), `DT_PALLAS_FFN_BM`, `DT_PALLAS_FFN_ALLREDUCE` | off | Sequence-tiled Pallas SwiGLU FFN (opt-in; slower except short sequences) |
| `DT_INT8_ACT` | off | W8A8: int8 activations with `--int8-weights` |
| `DT_QK_FP8` | off | Q and K in float8 in the attention kernel (image pipeline; `--qk-fp8`) |
| `DT_SERVER_PARK` | off | Server: park the DiT on host during a reference's text stage instead of shrinking the vision chunk |
| `DT_WARM_VIDEO_DECODE` | 1 | Server: compile the video decoder at start-up (0 skips; the first request pays it) |
| `DT_WEIGHT_CACHE` | off | Image pipeline: cache converted weights on disk |
| `DT_TX_VISION_ON_HOST` | off | torchax: vision tower on the host, language model on the TPU |
| `DT_STREAM_SHARD` | per pipeline | Residual-stream layout (`whole` / `seq`) |
| `DT_FULL_LANE_ROPE`, `DT_PAD_SEQUENCE_TO` | from flags | Set by `--full-lane-rope`, `--pad-sequence-to` |
| `DT_DECODE_CHUNK_BATCH`, `DT_DECODE_TILE_SHARD` | from flags | Video decode batching / tile sharding |
| `DT_MOE_DP` | auto | Agnes MoE: experts replicated on every device (auto: HBM >= 64 GiB, i.e. v7x); 0 = tensor parallel |
| `DT_MOE_RING`, `DT_MOE_GMM` | 1, 1 | Agnes MoE tensor parallel: ring reduce-scatter (0 = one psum); gmm_v2 (0 = `jax.lax.ragged_dot`) |
| `DT_MOE_EP` (`DT_MOE_EP_SLABS`, `DT_MOE_EP_BACKEND`) | off | Agnes MoE expert parallel (one expert a device, ragged all-to-all; loses to skewed routing) |
| `DT_MOE_FP8` | off | Agnes MoE DP: experts in float8 W8A8 (halves their HBM) |
| `DT_MOE_FAST_LOAD` | 1 | Agnes MoE: threaded LoRA merge at load (bit-identical; 0 = serial) |
| `DT_MOE_*_TILE_*` | tuned | gmm_v2 tiles for the EP / DP paths |

## Memory and compile tuning / 内存与编译调参

`DT_QKV_CHUNK`, `DT_FF_CHUNK` (row slabs for long sequences on few chips),
`DT_ENC_GROUP`, `DT_ENC_VMEM_KIB`, `DT_STEP_VMEM_KIB`, `DT_STEP_FAST_KIB`, `DT_TV_VMEM_KIB`,
`DT_VISION_ATTN_CHUNK`, `DT_VAE_1CHIP_TILE_BATCH`, `DT_LOAD_THREADS`, `DT_CACHE_READ_THREADS`,
`DT_SEGMOD_SELECT_MIN`, `DT_SWIGLU_INTERLEAVE`, `DT_ATTN_KEEP_HEADS`, `DT_SPLASH_UNROLL`,
`DT_VAE_ENC_PRECISION`, `DT_PRE_DENSE`, `DT_KEEP_DIT`, `DT_VISIBLE_CHIPS`.

## Diagnostics / 诊断

| Variable | Meaning |
|---|---|
| `DT_EMB_NOISE=<eps>` (`DT_EMB_NOISE_SEED`) | Perturb the text conditioning to measure the noise floor (server, jax_pipeline) |
| `DT_MOE_HIST=1` | Agnes MoE expert parallel: print rows per expert per layer |
| `DT_PAD_ROW_FILL=randn` | Fill padded text rows with noise (bucket leak check, server) |
| `DT_DUMP_BLOCKS=<dir>`, `DT_DUMP_BLOCK_CALL`, `DT_DUMP_COND`, `DT_TEXT_DUMP`, `DT_TEXT_DIAG`, `DT_ENC_CHECK` | Dumps and checks for layer-by-layer comparisons |
| `DT_SERVER_STEP_TIMES`, `DT_SERVER_TEXT_PROFILE=<dir>`, `DT_DECODE_TIMING`, `DT_CLOCK_LEAVES` | Timing and profiles |
| `DT_QK_BARRIER=0`, `DT_BLOCK_BARRIER`, `DT_PROBE_PAD_KV` | Ablations / timing probes |
