# torch_tpu MiniMax-H3: environment variables / 环境变量

`scripts/run_h3_torchtpu.sh` needs none of these; defaults are the tested
configuration. Prefer the command-line flags where one exists.

`scripts/run_h3_torchtpu.sh` 不需要设置下列任何变量,默认值就是测过的配置;有命令行参数的优先用参数。

## Tuning / 调参

| Variable | Default | Flag | Meaning |
|---|---|---|---|
| `DT_TEXT_CACHE` | 8 | `--text-cache` | Text conditionings kept for repeated prompts (0 = off) |
| `DT_REF_CACHE` | 8 | `--ref-cache` | Encoded reference latents kept for repeated references (0 = off) |
| `DT_POST_ATTN_PIPE` | on for v7x-8, else off | `--[no-]post-attn-pipe` | Two-slab all-reduce/compute overlap; helps on v7x-8, not on v6e-8 |
| `DT_VAE_ENC_NO_TILE` | 1 on v7x-8, else 0 | `--[no-]vae-enc-tile` | Encode the reference frame whole instead of in tiles (needs v7x's HBM) |
| `DT_OURS_KV_BKV` | 4096 on v7x-8, by length elsewhere | `--kv-block` | Kv block of the runtime-length attention kernel |
| `DT_OURS_BQ` / `DT_OURS_BKV` | 2048 / 4096 | | Attention q / kv blocks |
| `DT_OURS_QK_FP8` | off | | Q and K in float8 inside the attention kernel (lossy) |
| `DT_DIT_PRECOMPILE` / `DT_DIT_PRECOMPILE_START` | 1 / prefetch | | Compile a new text bucket's DiT program in the background (0 = off) |
| `DT_AOT_PERSIST` | `~/.cache/diffusers_tpu/aot_programs` | | Where compiled DiT programs are saved and reloaded (empty = off) |
| `DT_SPMD_SAFE` | 1 | | Compile each block's all-reduces into the block's program |

## Diagnostics / 诊断

| Variable | Meaning |
|---|---|
| `DT_STEP_CLOCK=1` | Per-step wall clock (syncs every step) |
| `DT_PHASE_STAMP=1` | Epoch timestamps at text encoder / step boundaries / decode |
| `DT_RANK_STAMPS=1` | Per-rank host timestamps around decode |
| `DT_DECODE_TIMING=1`, `DT_OFFLOAD_TIMING=1`, `DT_VAE_MOVE_TIMING=1`, `DT_PREFETCH_PROBE=1` | Timing of decode, layer offload, VAE moves, prefetch |
| `DT_EMB_NOISE=<eps>` (`DT_EMB_NOISE_SEED`) | Multiply the text conditioning by (1 + eps·N(0,1)) to measure the run-to-run noise floor |
| `DT_MATMUL_PRECISION` | Override matmul precision to study numerical drift |
| `DT_OURS_TRIPWIRE=1` | Raise if the main sequence reaches the attention dispatcher unexpectedly |
| `DT_FIXED_TSLOTS=pad`, `DT_VL_HOST_META=<list>`, `DT_BHSD_ROPE=<mode>`, `DT_SEGMOD_PART=msa\|mlp` | Alternative modes kept for comparisons |

## H3-Image (`scripts/run_h3_image_torchtpu.sh`)

| Variable | Meaning |
|---|---|
| `DT_SEGMOD_WHERE` / `DT_SEGMOD_F32` | Masked segment-wise modulation in float32; set to 1 by the entry point |
| `DT_TEXT_CHECK=1`, `DT_ENC_CHECK=1` | Compare the compiled text tower / reference encoder against eager |
| `DT_TEXT_TIMING=1`, `DT_RANK_TIMES=1` | Per-stage text-encoding times (syncs), per-rank request times |
| `DT_PROFILE_ALL_RANKS=1` | With `--profile`, trace every rank instead of rank 0 |
