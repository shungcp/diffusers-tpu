# TPU v7x (Ironwood) Optimization Guide & Benchmarks / TPU v7x 硬件架构与优化指南

## 1. Hardware Architecture & Device Accounting / 硬件架构与 Device 计数口径

Understanding the physical vs. logical topology of **TPU v7x (Ironwood)** is essential for interpreting multi-device benchmarks, memory capacity, and kernel tiling:
理解 **TPU v7x (Ironwood)** 的物理拓扑与逻辑 Device 映射关系，是分析多设备性能、HBM 容量与算子分块的基础：

| Metric / 指标 | TPU v7x Specification / TPU v7x 规格 | Notes for `diffusers-tpu` / 代码与配置要点 |
| :--- | :--- | :--- |
| **Chip vs. TensorCore (Device)** | **1 physical chip = 2 TensorCores = 2 logical devices (ranks)**<br>1 个物理 Chip 含 2 个独立 TensorCore，软件栈（JAX / `torch_tpu`）将每个 TensorCore 视为 1 个独立 Device（或 1 个进程 Rank） | **`v7x-8` = 4 physical chips = 8 TensorCores (8 devices / 8 ranks)**<br>`TORCH_TPU_TOPOLOGY="2,2,1,2"`（`2×2×1` 物理 chip 网格 × 每 chip `2` 个 core） |
| **HBM Capacity & Bandwidth** | **192 GiB HBM3e per physical chip** (**~95 GiB usable per TensorCore device**)<br>每物理 Chip 192 GiB HBM3e（每个逻辑 Device / TensorCore 可用约 **95 GiB**） | 无需 `--offload`：单个逻辑 Device（95 GiB）即可完整常驻 35B DiT 权重；在 `v7x-8` 上可同时常驻完整 DiT + 8-way TP 分片 Qwen3-VL Text Encoder（`7.3 GiB/device`）+ VAE Decoder |
| **MXU Systolic Array** | **`256 × 256` per TensorCore** (`bf16` / `fp8` E4M3 / `int8`)<br>每个 TensorCore 配备 **`256 × 256`** 脉动阵列（与 TPU v6e Trillium 相同，区别于 v4/v5e/v5p 的 `128 × 128`） | 矩阵乘（MatMul / Pallas FlashAttention / SwiGLU FFN）的序列分块（`block_q=2048`, `block_kv=4096`, 2-Slab 切分点 `S0=49152=192×256`）均需对齐 **`256`** 的整数倍，避免尾块填充浪费 MXU 周期 |
| **VPU 2D Register Tile Layout** | **`(8, 128)` (`8 sublanes × 128 lanes`)**<br>向量处理单元（VPU）与 HBM 2D 切片的基础寄存器布局为 `8 × 128` | 逐元素归一化（RMSNorm）与 RoPE 若作用在 `(B, S, H=7, D=128)` 上，次末维 `H=7 < 8` 会触发跨 sublane 广播拷贝；提前转置为 `(B, H, S, D)` 可使最内两维 `(S, 128)` 对齐 `(8, 128)` VPU 寄存器布局 |
| **Scoped VMEM Capacity** | **64 MiB per TensorCore** (`--xla_tpu_scoped_vmem_limit_kib=65536`) | 允许更大的 Pallas FlashAttention 分块（`BQ=2048, BKV=4096`）与多层算子融合驻留片上 VMEM |
| **SparseCore (SC)** | Dedicated SparseCores per chip for embedding / gather / scatter & optional collective offload | 支持异步 DMA 与特定集合通信卸载，但对规则稠密 All-Gather/All-Reduce 需结合 `LIBTPU_INIT_ARGS` 实测取舍 |

---

## 2. `torch_tpu` Optimization Ledger on `v7x-8` / `v7x-8` 上的 `torch_tpu` 阶梯优化台账

Workload / 测试负载：**Pika-H3-Flex 6-step text-to-video with audio (`t2va`, `768×1344`, `345` frames, `14.4 s` clip)** on **TPU `v7x-8` (4 chips / 8 TensorCore devices)**, warm steady-state request (`run 1` of `--runs 2`).
测试任务：在 **TPU `v7x-8`（4 物理 Chip / 8 TensorCore Devices）** 上运行 Pika-H3-Flex 6 步带音轨文生视频（`t2va`, `768×1344`, `345` 帧），统计 `--runs 2` 的第二次热请求（`run 1`）端到端墙钟时间与阶段拆解。

| Stage / 阶段 | Optimization Summary / 核心优化机制 | **`run 1` Total** | `text_encoder` | **`denoise/other`** | **`vae_decode` (slowest)** | Decode & Host Sub-timings (`rank 0`) | Pixel Verification (`vs ref`) | Commit / Env Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **Step 01 Baseline** | 8-device TP Text Encoder + Resident VAE + `aot_group=10` | **`48.1 s`**<br>*(slowest `49.2s`)* | `1.8 ~ 3.2 s` | `34.4 ~ 34.7 s` | `8.9 ~ 11.6 s` | • TPU decode `3.9s` + `/dev/shm` `2.0s`<br>• 8-rank CPU blend & round-trip to TPU `3.0~5.7s` | Reference (`ref`) | `567ac8d` |
| **Step 02 Trial**<br>*(Rejected)* | Per-linear sequence chunking (`DT_TP_CHUNKS=2/4/8` in `tp_h3.py`) | `c=2`: `44.2 s`<br>`c=4`: `46.9 s`<br>`c=8`: `47.3 s` | `1.2 ~ 1.5 s` | `c=2`: `34.2 s`<br>`c=4`: **`36.5 s`**<br>`c=8`: **`36.5 s`** | `9.0 ~ 10.1 s` | • Single linear (`1.09~1.88ms`) too small to hide `9.0ms` AR<br>• Two `torch.cat` copies add **`+9.88 ms/block`**; `103989/4=25998` breaks `256×256` MXU alignment | `PSNR 19.5~22.1 dB` | ❌ Rejected |
| **Step 04 (`old`)** | Raw 16-bit `bf16` (`t.view(int16)`) exchange over `/dev/shm` | **`44.3 s`** | `1.4 s` | `34.2 s` | **`9.0 s`** | • `gather`: `2.0 -> 1.8s`<br>• `blend`: `2.9s` (still round-trips to TPU) | **`identical=True`** | `c841add` |
| **Step 04 (`base`)** | Host postprocessing (`DT_DECODE_HOST_POST=1`) + Rank 0-only 32-thread blend (`DT_DECODE_RANK0_ONLY=1`) | **`40.3 ~ 42.2 s`** | `1.2 ~ 2.6 s` | `33.2 ~ 33.6 s` | **`6.0 s`** | • `gather`: `1.8 ~ 1.9s`<br>• `blend`: `0.2s`<br>• `postprocess 0.2s + to np 0.9~1.0s` | **`identical=True`** | `c841add` |
| **+ Opt-1**<br>(`DT_OPT1=1`) | **Pipelined `/dev/shm` write inside decode loop + Rank 0 in-RAM keep + 8-thread parallel read + 32-thread `bf16` permute & zero-copy `.numpy()`** | **`38.1 s`**<br>*(`-2.2~4.1 s`)* | `1.2 s` | **`32.4 s`** | **`4.7 s`**<br>*(`-1.3 s`)* | • `gather`: `1.9s ->` **`0.5 s`** (`-1.4s`)<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** (`-1.0s`) | **`identical=True`**<br>`max\|d\|=0` | `8de71db`<br>*(Default on)* |
| **+ Opt-2**<br>(`DT_RESIDENT_TEXT=1`) | **Keep 8-way TP sharded Text Encoder (`7.3 GiB/device`) resident in v7x's 95 GiB HBM (`--no-offload`), skipping per-request 369-tensor CPU↔TPU transfers & `tt_compile.clear_cache()`** | **`35.7 s`**<br>*(`-2.4 s`)* | `1.2 s` | **`29.8 s`**<br>*(`-2.6 s`)* | **`4.7 s`** | • `gather`: **`0.6 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | **`identical=True`**<br>`max\|d\|=0` | `8de71db`<br>*(Default on v7x)* |
| **+ Opt-3**<br>(`DT_DECODE_TAIL_SPLIT=1`) | **Split the 4 tail decode chunks (`16..19` of 20) 50/50 across all 8 devices (Ranks `0..3` top 14 tiles, Ranks `4..7` bottom 14 tiles, stitched on TPU)** | **`35.1 s`**<br>*(`-0.6 s`)* | `1.2 s` | **`29.8 s`** | **`4.1 s`**<br>*(other 7 devices `3.9s`)* | • `tail-chunk 16 (1/2 tiles)`: **`0.9 s`**<br>• `gather`: **`0.3 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | **`identical=True`**<br>`max\|d\|=0` | `8de71db` / `91d89fe`<br>*(Default on)* |
| **+ Opt-4 + Opt-5a**<br>(`DT_FUSED_RES_MOD=1`<br>`DT_BHSD_ROPE=1`) | **Denoise block optimizations (100% bit-identical):**<br>1. `by_runs_add`: fuse `residual + gate * x` before `index_copy`, saving two `1.06 GB` full-sequence HBM writes/block;<br>2. Transpose Q/K/V to `(B, H, S, D)` immediately after `unflatten` to align `(S, 128)` with the `(8, 128)` VPU tile before `norm_q/k` and RoPE, dropping `.contiguous()` | **`34.6 s`** (`0.0.51`)<br>**`32.6 s`** (`0.0.52`, `33.4 s` 3-run mean) | `1.4 s` (`0.0.51`)<br>**`0.4 s`** (`0.0.52`) | **`29.2 s`** (`0.0.51`)<br>**`29.3 s`** (`0.0.52`) | **`4.1 s`** (`0.0.51`)<br>**`2.9 s`** (`0.0.52`) | • `tail-chunk 16`: **`0.9 s`**<br>• `gather`: **`0.3 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | **`identical=True`**<br>`max\|d\|=0`<br>`mean\|d\|=0.0000` | `1016192`<br>*(Bit-exact)* |
| **+ Opt-5b (Default)**<br>(`DT_BHSD_ROPE=full`) | Replace `chunk+cat` RoPE with 128-lane MXU matmul `x * cos128 + (x @ R) * sin128` + `libtpu 0.0.52` + single-compile `AotBlocks` (`1dbdde3`) | **`34.3 s`** (`0.0.51`)<br>**`32.1 s`** (`0.0.52`, `32.5 s` 3-run mean; cold compile `121.3 -> 94.7 ->` **`69.6 s`**) | `1.2 s` (`0.0.51`)<br>**`0.4 s`** (`0.0.52`) | **`29.0 s`** (`0.0.51`)<br>**`28.9 s`** (`0.0.52`) | **`4.1 s`** (`0.0.51`)<br>**`2.9 s`** (`0.0.52`) | Same as above | `identical=False`<br>`PSNR=19.2 dB` | `1016192` / `1dbdde3`<br>*(Default on)* |
| **+ Opt-6 (2-Slab Pipe)**<br>(`DT_POST_ATTN_PIPE=1`) | **2-Slab Post-Attention Sub-block Pipeline (`S0 = 49152 = 192×256`):**<br>Split `to_out -> AR1 -> gate_msa -> norm2 -> mod_mlp -> SwiGLU FFN -> AR2 -> gate_mlp` at `S0=49152` (aligned to `256×256` MXU and `4096`), overlapping `AR1(1)` with `Slab 0 norm2+FFN` and `AR2(0)` with `Slab 1 norm2+FFN` (100% video rows, zero `index_copy`), with only 1 `torch.cat` per block | **`32.0 s`** (`0.0.51`)<br>**`29.5 s`** (`0.0.52`, `30.4 s` 3-run mean; **`-18.6 s` / `-38.7%`**) | **`1.1 s`** (`0.0.51`)<br>**`0.4 s`** (`0.0.52`) | **`26.8 s`** (`0.0.51`)<br>**`26.3 s`** (`0.0.52`, `4.38s/step`) | **`4.1 s`** (`0.0.51`)<br>**`2.9 s`** (`0.0.52`) | • `tail-chunk 16`: **`0.9 s`**<br>• `gather`: **`0.3 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | `identical=False`<br>`mean\|d\|=9.90`<br>`PSNR=21.8 dB`<br>*(1-ULP bf16 tile rounding)* | `52aba21` / `4492ce4`<br>(`DT_POST_ATTN_PIPE=1`) |

---

## 3. JAX & `torchax` Benchmarks on TPU v7x (1, 2, 4, 8 Devices) / JAX 与 `torchax` 在 v7x 上的性能

### 3.1 Pika-H3-Flex (6 steps, with audio) — JAX (`--fp8 --qk-fp8 --adaln-host`)

| Workflow & Resolution | 1 device (0.5 chip) | 2 devices (1 chip) | 4 devices (2 chips) | 8 devices / `v7x-8` (4 chips) |
| :--- | :---: | :---: | :---: | :---: |
| **text-to-video, `768×1344`, 345 frames** | **192.7 s** (`179.5 + 12.1`) | **101.4 s** (`91.1 + 9.2`) | **56.6 s** (`49.5 + 6.7`) | **29.3 s** (`25.2 + 3.6`) |
| **reference-to-video, `768×1344`, 345 frames** | — | — | — | **33.1 s** (`29.0 + 3.6`) |
| **text-to-video, `480×832`, 362 frames (15 s)** | — | — | — | **8.4 s** (`6.1 + 2.0`) |
| **text-to-video, `480×832`, 141 frames (6 s)** | **12.4 s** (`9.4 + 2.1`) | — | — | **3.1 s** (`2.2 + 0.8`) |

> **Comparison across backends on `v7x-8` (`768×1344`, 345 frames, Pika-H3-Flex `t2va`)**:
> - **JAX (`--fp8 --qk-fp8 --adaln-host`)**: **`29.3 s`** (`denoise 25.2 s + vae_decode 3.6 s`)
> - **`torch_tpu` (`bf16`, `DT_POST_ATTN_PIPE=1`, `libtpu 0.0.52`)**: **`29.5 s`** steady-state (`denoise/other 26.3 s + vae_decode 2.9 s + text_encoder 0.4 s`; **`30.4 s`** mean of 3 warm requests; `32.0 s` on `libtpu 0.0.51`)
> - **`torch_tpu` (`bf16`, default `DT_BHSD_ROPE=full`, `libtpu 0.0.52`)**: **`32.1 s`** steady-state (`denoise/other 28.9 s + vae_decode 2.9 s + text_encoder 0.4 s`; **`32.5 s`** mean of 3 warm requests; `34.3 s` on `libtpu 0.0.51`)
> - **`torch_tpu` (`bf16`, bit-identical `DT_BHSD_ROPE=1`)**: **`32.6 s`** steady-state on `libtpu 0.0.52` (`denoise/other 29.3 s + vae_decode 2.9 s + text_encoder 0.4 s`; **`33.4 s`** mean of 3 warm requests; **`34.6 s`** on `libtpu 0.0.51`)

---

## 4. TPU v7x Runtime & XLA Flags (`LIBTPU_INIT_ARGS`) Exploration / v7x 编译与运行时 Flag 探索与实测对比

### 4.1 Key XLA / `libtpu` & Runtime Flags on TPU v7x

| Flag / Environment Variable | Target Subsystem | Explored Behavior & Impact on TPU v7x | Status |
| :--- | :--- | :--- | :---: |
| `--xla_tpu_dvfs_p_state=7` | Clock / DVFS | Locks TPU v7x TensorCore frequency to highest P-state (`p_state=7`). | Enabled by default across all 3 backends |
| `--xla_tpu_enable_latency_hiding_scheduler=true` | XLA Latency Hiding Scheduler (LHS) | Schedules async collectives and DMA copies across independent compute instructions. On `v7x` (`libtpu 0.0.51+`), LHS + `DT_SPMD_SAFE=1` already executes `all-reduce` asynchronously on the ICI stream without extra collective flags. | Enabled by default across all 3 backends |
| `--xla_tpu_enable_async_collective_fusion=true`<br>`--xla_tpu_enable_async_collective_fusion_multiple_steps=true`<br>`--xla_tpu_overlap_compute_collective_tc=true`<br>`--xla_tpu_enable_ici_ag_pipelining=true` | Async Collective Fusion & TensorCore Overlap | Helps `JAX` / `torchax` whole-model SPMD graphs fuse multi-step collectives. On `torch_tpu`, unsplit `to_out -> AR1 -> norm2 -> FFN -> AR2` has a strict data dependency, so compiler flags alone cannot overlap `AR1`/`AR2` with `FFN` (`denoise/other` stays `29.2 s`); requires `DT_POST_ATTN_PIPE=1` (`26.8 s`). Note: `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` must be omitted as it aborts PjRt initialization on `ghostlite`/v6e. | Enabled in `JAX` & `torchax`; not needed in `torch_tpu` |
| `--xla_tpu_aggressive_opt_barrier_removal=ENABLED` | Barrier Removal | Removes optimization barriers to allow more aggressive instruction reordering (`32.0 -> 31.9 s` with `DT_POST_ATTN_PIPE=1`), but alters `bf16` reduction order and breaks bit-identical output on the unsplit path (`identical=True -> False, PSNR=20.3 dB`). | Enabled in `torchax`; omitted in `torch_tpu` to preserve bit-exact default |
| `--xla_tpu_scoped_vmem_limit_kib=65536` | Scoped VMEM Allocation | Raises per-fusion scoped VMEM limit to 64 MiB (matching v7x's 64 MiB VMEM per TensorCore). Setting it globally in `torchax` or `torch_tpu` slows down the 3D VAE decoder (`7.8 -> 12.2 s` in `torchax`) or eager-mode Qwen3-VL text encoder (`1.4 -> 4.2 s` in `torch_tpu`). In `torchax_pipeline.py`, it is scoped per-program to the DiT only. | Scoped to DiT in `torchax`; global `65536` in `JAX`; default in `torch_tpu` |
| `--xla_tpu_enable_offloading_copy_to_sparsecore=false`<br>`--xla_tpu_enable_sparse_core_collective_offload_all_gather=false` | SparseCore Offload Control | Prevents XLA from offloading regular dense `all-gather` and copy ops onto SparseCore when direct ICI TensorCore transfers have lower latency. | Set in `run_h3_torchax.sh` |
| `TPU_PREMAPPED_BUFFER_SIZE` | Host-to-Device PCIe/DMA Premapped Buffer | Default `0` caps H2D transfer bandwidth at `5.4 GB/s` (vs `26.1 GB/s` at `>=64 GiB`). `torchtpu_h3.py` automatically sets this to `min(192 GiB, RAM / 2 / local_ranks)`. | Auto-configured in `torchtpu_h3.py` |
| `DT_SPMD_SAFE=1` | `torch_tpu` `SplitCompiler` | By default `torch_tpu` splits the compiled FX graph at every collective (`all_reduce`), exposing ~11 ms per collective. Wrapping `_flat_group` in `torch.tpu.distributed.spmd_safe` compiles all 10 grouped blocks and their 20 `all_reduce`s into a single XLA HLO graph. | Enabled by default in `run_h3_torchtpu.sh` |
| `DT_OURS_BQ=2048` / `DT_OURS_BKV=4096` | Pallas FlashAttention Tiling | Matches TPU v7x's `256×256` MXU (`2048 = 8×256`, `4096 = 16×256`) and 64 MiB VMEM. | Enabled by default in `run_h3_torchtpu.sh` & `JAX` |

### 4.2 Measured `LIBTPU_INIT_ARGS` Ablation on `v7x-8` (`torch_tpu`, Pika 768p 345f `t2va`)

| `LIBTPU_INIT_ARGS` Profile | `DT_POST_ATTN_PIPE` | **`run 1` Total** | `text_encoder` | **`denoise/other`** | `vae_decode` | Pixel Verification (`vs ref`) | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`BASE_FLAGS` (Default in `torchtpu_h3.py`)**<br>`p_state=7`, `lhs=true`, `spmd_rng_unsafe=true`, `dot_strength_reduction=true` | `0` (Default) | **`34.6 s`** | `1.4 s` | **`29.2 s`** | `4.1 s` | **`identical=True`** (`max\|d\|=0`) | **Best bit-identical default** |
| **`BASE_FLAGS` + `DT_POST_ATTN_PIPE=1`** | `1` (2-Slab) | **`32.0 s`** | **`1.1 s`** | **`26.8 s`** | `4.1 s` | `PSNR = 21.8 dB` (`mean\|d\|=9.90`) | **Best fast pipeline (`-2.6 s`)** |
| **`FLAGS_ASYNC_TC` (`torchax` / `JAX` extended flags)**<br>`+ async_collective_fusion`, `overlap_compute_collective_tc`, `ici_ag_pipelining`, `lhs_prioritize_async_depth`, `aggressive_opt_barrier_removal`, `scoped_vmem=65536` | `0` | `37.4 s` | `4.2 s` *(eager regression)* | `29.2 s` *(no gain)* | `4.0 s` | `identical=False`, `PSNR = 20.3 dB` | ❌ Compiler flags alone cannot overlap data-dependent unsplit `AR1`/`AR2`; global `scoped_vmem=65536` slows eager text encoder (`+2.8 s`) and barrier removal breaks bit-exactness |
| **`FLAGS_ASYNC_TC` + `DT_POST_ATTN_PIPE=1`** | `1` (2-Slab) | **`31.9 s`** | `1.2 s` | **`26.6 s`** | `4.1 s` | `identical=False`, `PSNR = 20.2 dB` | Marginal `-0.1 s` over `BASE_FLAGS + PIPE=1`, but lowers PSNR (`21.8 -> 20.2 dB`) |

---

## 5. `MiniMax-H3-Image` (6-Step 2K & 4K) on TPU `v7x-8` vs. GPU Baselines / `MiniMax-H3-Image` 性能对比与优化

`MiniMax-H3-Image` (`MiniMaxAI/MiniMax-H3-Image`) pairs the 33.4B `Qwen3-VL` conditioner (`stop_at_layer=51`) with a 50-layer DiT (`hidden=5376, heads=56×128, ffn=14336`, 6 step-specific output heads `pdd_proj_out.{0..5}`) and a 24-layer 2D ViT tiled VAE decoder (`tile_px=256, overlap_px=64`).

### 5.1 Official GPU Baselines vs. TPU `v7x-8` (6 Steps, CFG-Free)

| Hardware / Stack | Precision | 2K (`2048×2048`) `t2i` | 2K (`2048×2048`) `edit` | 4K (`4736×3552` / `4096×4096`) `t2i` | 4K (`4096×4096`) `edit` | Stage Breakdown (`text + denoise + vae`) / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1 × RTX PRO 6000 Blackwell** (SGLang) | BF16 / FP8 | `3.434 s` / `2.338 s` | — | `18.544 s` / `14.445 s` | — | Warm pipeline median (excl. text encoder) |
| **1 × NVIDIA H200** (SGLang / vLLM-Omni) | BF16 | `1.934 s` (`3.20 s` e2e) | — | `10.133 s` | — | SGLang warm pipeline / vLLM-Omni client e2e |
| **2 × NVIDIA H200** (SGLang Ulysses2 / vLLM-Omni) | BF16 | `1.123 s` (`1.50 s` e2e) | — | `5.465 s` | — | 2-way Ulysses SP / TP |
| **4 × NVIDIA H200** (vLLM-Omni Ulysses SP4) | BF16 | `0.88 s` (`TP4: 1.00 s`) | `1.35 s` | `3.84 s` | — | Resident Qwen3-VL + DiT + VAE |
| **2 × NVIDIA B300** (SGLang Ulysses2) | BF16 | `0.604 s` | `0.931 s` | `2.654 s` | `3.146 s` | Warm pipeline median (excl. text encoder) |
| **TPU `v6e-8` (`JAX` / `torchax` / `torch_tpu`)** | BF16 | `0.63 s` / `0.64 s` / `0.68 s` | `1.15 s` / `1.17 s` / `1.32 s` | `2.77 s` / `2.80 s` / `2.95 s` | `3.38 s` (`JAX`) | Warm e2e (incl. text encode + 6 DiT steps + decode) |
| **TPU `v7x-8` (`JAX`, `DT_STREAM_SHARD=whole`)** | **BF16** | **`0.517 s`** (`0.529 s` e2e) | **`0.770 s`**¹ (`0.909 s` e2e; `0.817 s` warm) | **`2.178 s`** (`2.191 s` e2e) | **`2.505 s`**¹ (`2.653 s` e2e; `2.561 s` warm) | `0.012 s + 0.490 s / 2.103 s + 0.023 s / 0.066 s` |
| **TPU `v7x-8` (`JAX`, sequence-sharded)** | **FP8** | `0.833 s` (`0.852 s` e2e) | `0.929 s`¹ (`1.322 s` e2e) | `3.053 s` (`3.074 s` e2e) | — | `0.019 s + 0.806 s / 2.975 s + 0.024 s / 0.066 s` |
| **TPU `v7x-8` (`torchax`)** | **BF16** | **`0.533 s`** (`0.546 s` e2e) | **`0.792 s`**¹ (`0.933 s` e2e; `0.839 s` warm) | **`2.195 s`** (`2.208 s` e2e) | — | `0.013 s + 0.508 s / 2.115 s + 0.022 s / 0.067 s` |
| **TPU `v7x-8` (`torch_tpu`)** | **BF16** | **`0.551 s`** (`0.578 s` e2e) | **`0.821 s`**¹ (`1.031 s` e2e; `0.903 s` warm) | **`2.284 s`** (`2.309 s` e2e) | — | `0.026 s + 0.524 s / 2.179 s + 0.024 s / 0.092 s` |
| **TPU `v7x-8` (`torch_tpu`, `DT_POST_ATTN_PIPE=1`)** | **BF16** | **`0.551 s`** (`0.578 s` e2e) | — | **`1.951 s`** (`1.977 s` e2e) | — | `0.026 s +` **`1.845 s` (4K)** `+ 0.093 s` |

¹ `denoise + 2D VAE decode` (`0.746 s + 0.024 s = 0.770 s` at 2K edit, `2.442 s + 0.063 s = 2.505 s` at 4K edit on `JAX`; `0.798 s + 0.023 s = 0.821 s` on `torch_tpu`); full end-to-end (`0.909 s` at 2K, `2.653 s` at 4K on `JAX`; `1.031 s` on `torch_tpu`) also includes Qwen3-VL vision+text (`0.092 s` on `JAX`, `0.126 s` on `torch_tpu` with `DT_VIS_COMPILE=1`) and sharded 3D causal reference VAE encoding (`0.045 s` at 2K, `0.055 s` at 4K on `JAX`; `0.072 s` on `torch_tpu`). With `--requests reqs.jsonl` resident serving on `v7x-8`, a new 2048×2048 `t2i` shape compiles and runs in **`7.865 s`** (down from `45.360 s`) and a new `edit` shape in **`10.529 s`** (down from `77.895 s`); warm 2048×2048 `t2i` requests run in **`0.558 s`** (`0.584 s` with `--scan-layers`) and warm one-reference 2048×2048 edits run in **`0.929 s`** (`0.074 s` text + `0.035 s` prepare + `0.776 s` denoise + `0.024 s` decode).

### 5.2 `MiniMax-H3-Image` Optimization Progression on `v7x-8`

| Step | Optimization / 核心优化项 | 2K (`2048×2048`) | 4K (`4736×3552` / `4096×4096`) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Opt-0** | Initial `JAX` baseline (`--adaln-host` inter-block barrier, unaligned `seq`, sublane RoPE) | `2.202 s` (`FP8: 1.937 s`) | `7.771 s` (`FP8: 6.947 s`) | `1.00×` |
| **Opt-1** | Remove inter-block `optimization_barrier` on 8-device TP (`_tp_size() > 2`) + `--segments` + `--full-lane-rope` + `--pad-sequence-to 128` (sequence-sharded residual stream) | `0.861 s` (`FP8: 0.833 s`) | `3.155 s` (`FP8: 3.053 s`) | `2.56×` (2K) / `2.46×` (4K) |
| **Opt-2** | Whole residual stream (`DT_STREAM_SHARD=whole`) + head-sharded attention output (`DT_ATTN_KEEP_HEADS=1`) + per-chip SwiGLU row interleaving (`DT_SWIGLU_INTERLEAVE=1`) + full-sequence `jnp.where` segmented modulation (`DT_SEGMOD_SELECT_MIN=1`): 2 plain `all_reduce`s and 1 fused residual+norm pass per block | **`0.517 s`** (`0.529 s` e2e; `denoise 0.490 s`) | **`2.178 s`** (`2.191 s` e2e; `denoise 2.103 s`) | **`4.26×` (2K) / `3.57×` (4K)** |
| **Opt-3** | `torchax` hybrid pipeline with custom Pallas Splash Attention + `keep_heads` + `_interleave_swiglu` + `select_min=1` masked segmented modulation + JAX 2D ViT tile-sharded VAE (`0.022 s` at 2K, `0.067 s` at 4K) | `1.057 s` -> **`0.533 s`** (`0.546 s` e2e) | `2.941 s` -> **`2.195 s`** (`2.208 s` e2e) | **`4.13×` (2K) / `3.54×` (4K)** |
| **Opt-4** | Sharded JIT-compiled 3D causal reference VAE encoder (`tiled_encode` with `with_sharding_constraint`: `0.045 s` at 2K, `0.055 s` at 4K) + `jnp.where` segmented modulation + `DT_STREAM_SHARD=whole` in `edit` mode + `--scan-layers` / `--parallel-compile` / scoped VMEM in `--requests` serving (new shape `45.4 ->` **`7.9 s`** `t2i`, `77.9 ->` **`10.5 s`** `edit`; warm `edit` **`0.929 s`**) | `2.787 s` -> **`0.909 s` e2e** (`0.817 s` warm; `denoise+vae` **`0.770 s`**) | `4.37 s` (`v6e-8`) -> **`2.653 s` e2e** (`2.561 s` warm; `denoise+vae` **`2.505 s`**) | **`3.07×` (2K Edit)** |
| **Opt-5** | `torch_tpu` (`libtpu 0.0.52`) 8-rank TP (`AotBlocks(group=10)` single-compile + `tpu:0` device fix + 128-lane BHSD RoPE + `DT_SEGMOD_WHERE=1` + `DT_SEGMOD_F32=1` + compiled 27-layer Qwen3-VL vision tower `DT_VIS_COMPILE=1` (`0.030 s`) + 256-bucket compiled text tower + 2-matmul/2D-FF compiled ViT tile decode `0.024 s` / `0.092 s` + `DT_POST_ATTN_PIPE=1` 2-slab overlap on 4K; `edit` `2.914 ->` **`1.031 s` e2e**, cold start `51.6 ->` **`16.7 s`**) | `1.287 s` -> **`0.578 s` e2e** (`0.551 s` warm, `denoise` **`0.524 s`**, cold `25.8 s`) | `3.709 s` -> **`1.977 s` e2e** (`1.951 s` warm, `denoise` **`1.845 s`**, cold `36.6 s`) | **`3.99×` (2K) / `3.98×` (4K)** |

---

## 6. `MOSS-Image` (50-Step CFG 3 & 8-Step DMD) on TPU `v7x-8` vs. `v6e-4` & GPU / `MOSS-Image` 性能对比

`MOSS-Image` (`OpenMOSS-Team/MOSS-Image-Base`, 5B 42-layer DiT with 24 heads × 128 dim, `Qwen3.5-9B` hybrid Gated-DeltaNet + full-attention text encoder, and `AutoencoderKLFlux2` VAE) runs on a 2D `("cfg", "tensor")` mesh where the two CFG branches execute side-by-side (`cfg=2`, skipping layer 22 in the unconditional branch via `jnp.where`/`torch.where`), the FFN `ffn_up_gate` is split into separate `ffn_up` and `ffn_gate` linears to eliminate inter-chip collective-permutes, Splash attention uses a single masked loop (`kv_mask_all=True`) for `seq > 8192` (`2048×2048`), and the VAE decoder runs as a native NHWC JAX program (`jax_vae.py`, `9.44 ms` in `fp32` / `8.01 ms` in `bf16` at `1024×1024` across 8 devices vs `122.58 ms` traced through `torchax`).

| Workload / Mode | 1 Device (`v7x` / `v6e`) | 2 Devices (`v7x` / `v6e`) | 4 Devices (`v7x-4` / `v6e-4`) | **8 Devices (`v7x-8` / `v6e-8`)** | GPU Reference | Stage Breakdown on `v7x-8` (`text + denoise + decode`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`1024×1024` (50 steps, CFG 3, `--dit torchax`)** | **`9.97 s`** / `13.0 s` | **`4.96 s`** / `5.64 s` | **`2.87 s`** / `4.03 s` | **`2.68 s`** / `2.56 s` | `17.1 s` (H100 diffusers) | `0.021 s + 2.66 s (53.1 ms/step) + 0.02 s` |
| **`1024×1024` (50 steps, CFG 3, `--dit jax`)** | — | — | **`2.70 s`** / `3.96 s` | **`2.58 s`** (cold `27.8 s`) | `17.1 s` (H100 diffusers) | `0.021 s + 2.57 s (51.4 ms/step) + 0.02 s` |
| **`2048×2048` (50 steps, CFG 3, `--dit torchax`)** | **`63.0 s`** / `82.2 s` | **`31.3 s`** / — | **`16.36 s`** / `24.9 s` | **`12.84 s`** / `13.8 s` | `≈139 s` (RTX 5090 DiT only) | `0.021 s + 12.79 s (255.8 ms/step) + 0.05 s` |
| **`2048×2048` (50 steps, CFG 3, `--dit jax`)** | — | — | **`16.24 s`** / `24.48 s` | **`12.74 s`** (cold `42.5 s`) | `≈139 s` (RTX 5090 DiT only) | `0.021 s + 12.69 s (253.8 ms/step) + 0.05 s` |
| **8-Step DMD (`--dmd`, `1024×1024`, no CFG)** | — | — | **`0.61 s`** / `0.48 s` | **`0.38 s`** / `0.46 s` | — | `0.021 s + 0.36 s (45.0 ms/step, tensor=8) + 0.02 s` |
| **8-Step DMD (`--dmd`, `2048×2048`, no CFG)** | — | — | **`2.82 s`** / — | **`1.58 s`** (cold `54.3 s`) | — | `0.021 s + 1.54 s (192.5 ms/step, tensor=8) + 0.05 s` |
| **Resident Serving (`--requests`, `1024×1024`, `--replicas 1`)** | — | — | **`2.89 s`/req** / `4.05 s`/req | **`2.71 s`/req** / `2.43 s`/req | — | `0.03 s text + 2.66 s denoise + 0.02 s decode` (fresh prompt) |
| **Resident Serving (`--requests`, `1024×1024`, `--replicas 2`)** | — | — | **`2.50 s`/img** / `2.85 s`/img | **`1.52 s`/img** (`3.04 s` / batch of 2) | — | Two 4-device (`cfg=2 × tensor=2`) replicas overlapped |
| **Resident Serving (`--requests`, `1024×1024`, `--replicas 4`)** | — | — | — | **`1.28 s`/img** / `1.46 s`/img | — | Four 2-device (`cfg=2 × tensor=1`) replicas (`5.12 s` / batch of 4) |

---

## 7. `torch_tpu` Warm-Start Persistence (`DT_AOT_PERSIST`) on `v7x-8` / 跨进程编译产物持久化与压缩

Measured on `v7x-8` (`torch_tpu 0.2.0`, `libtpu 0.0.52`, fresh OS processes with `/dev/shm/torch_tpu_cache` wiped between runs) using `b9a487b` (`DT_FIXED_TSLOTS=1` + pure-torch Pallas `register_fake`) and `DT_AOT_PERSIST=<dir>` (`0c9d592`, `3dcddf3`, `d6a4afa`), where `aot_compile` + `save_compiled_function` / `load_compiled_function` skips TorchDynamo and AOTAutograd tracing entirely:

| Workload (`v7x-8`, 8 ranks, `DT_SPMD_SAFE=1`) | Cold Compile + Save (`run 0` / `run 1`) | Warm Tier-3 Only (`DT_AOT_PERSIST=OFF`, `run 0` / `run 1`) | **Warm Saved Programs (`DT_AOT_PERSIST=ON`, `run 0` / `run 1`)** | Cross-Process Pixel Check |
| :--- | :---: | :---: | :---: | :---: |
| **`MiniMax-H3` Pika `t2va` (`480×832×121`, 6 steps)** | `35.4 s` / `6.0 s` | `17.0 s` / `4.7 s` | **`6.1 s` / `4.7 s`** (`-10.9 s` / **`2.79×`** faster `run 0`) | **`max\|d\| = 0`** (`sha_eq = True`) |
| **`MiniMax-H3-Image` `2048×2048` (6 steps)** | `40.4 s` / `0.614 s` | `14.6 s` (`denoise 6.69 s`) / `0.613 s` | **`6.74 s`** (`denoise` **`1.29 s`**, **`5.2×`** faster) / `0.610 s` | **`max\|d\| = 0`** (`exact = True`) |
| **`MiniMax-H3-Image` `4736×3552` (6 steps)** | `36.7 s` / `2.365 s` | `27.1 s` (`denoise 8.17 s`) / `2.337 s` | **`18.9 s`** (`denoise` **`2.98 s`**, `+0.77 s` vs `run 1`) / `2.344 s` | **`max\|d\| = 0`** (`exact = True`) |

- **Storage & compression notes (`dedup_stats.py`)**:
  - Sharing one `rank 0` archive across all 8 ranks (`3dcddf3`) cuts `DT_AOT_PERSIST` disk usage by **8×** immediately (`H3 t2va` `829.3 MiB -> 105.5 MiB`; `H3-Image 2K+4K` `2,645.7 MiB -> 330.7 MiB`).
  - Because `save_compiled_function` embeds uncompressed PJRT executable bytes inside a Python pickle, plain `zstd -3` (4 MiB window) only achieves `~4.4×`, whereas **`zstd -3 --long=27`** (128 MiB window) compresses single rank-0 files by **`16×–25×`** (`330.7 MiB -> 13.4 MiB` on `H3-Image 2K+4K`) and **FastCDC `1 KiB` + `zstd --long=27`** compresses cross-shape archives down to **`9.76 MiB`** (`33.9×` on rank-0; `164.9×–329.5×` if per-rank copies were stored).

---

## 8. `MAGI-2-preview` (113.93B Multi-Head Latent MoE + 1080p Refiner) on TPU `v7x-8` / `MAGI-2-preview` 端到端优化台账

Full end-to-end 4-phase optimization of Sand.ai **MAGI-2-preview** (113.93B / 5.95B active Stage 1 Multi-Head Latent MoE DiT with 4-stream mHC + Attention Sink, 6.72B 30-layer Stage 2 1080p Refiner, Qwen3.5-27B text encoder, Wan2.2-TI2V-5B I2V encoder, TurboVAED video decoder, and AutoencoderOobleck stereo audio VAE) across **Native JAX**, **torchax**, and **torch_tpu** on TPU `v7x-8` (`jax 0.12.0.1`, `torch_tpu`, `libtpu 0.0.52`, `94.74 GiB` HBM per logical device).

### 8.1 Stage-by-Stage Optimization Ledger (Phases 1–4)

| Phase / Commit | Stage & Target | Baseline (`v7x-8`) | Optimized (`v7x-8`) | Speedup | Parity / QA Verification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 1** (`dc341a0`) | Stage 0a `Wan2_2_VAEEncoder` (`@jax.jit`) | `70.41 s` cold / `12.58 s` warm eager | **`13.67 s`** cold / **`0.29 s`** cache hit / **`22.7 ms`** warm | **`554×`** warm | `rel_l2 = 1.48e-3` (lossless) |
| **Phase 1** (`dc341a0`) | Stage 0b `Qwen3.5-27B` `--uncond-cache` (default ON) | `20.80 s` per request | **`1.62 ms`** disk cache hit | **`12,840×`** | `max\|d\| = 0.0` (bit-exact) |
| **Phase 1** (`dc341a0`) | Stage 3a `TurboVAED` 8-chip `decode_sharded` (`540p`, 125f) | `1.29 s` warm (1-chip) | **`0.72 s`** warm (8-chip chunk-parallel) | **`1.80×`** | **`identical=True`, `max\|d\|=0`, `PSNR=∞`** |
| **Phase 1** (`dc341a0`) | Stage 3a `TurboVAED` 8-chip `decode_sharded` (`1080p`, 249f) | `11.97 s` warm (1-chip) | **`3.88 s`** warm (8-chip chunk-parallel) | **`3.09×`** | **`identical=True`, `max\|d\|=0`, `PSNR=∞`** |
| **Phase 2** (`e900499`) | Stage 2 `1080p` Refiner (`CP=8`, GQA+W-coal+tail Pallas BF16) | `87.35 s/step` (`437 s` / 5 steps) | **`10.69 s/step`** (`53.97 s` / 5 steps) | **`8.17×`** | 1-step `cos = 0.999998` |
| **Phase 2** (`e900499`) | Stage 2 `1080p` Refiner (**default `DP=2 × CP=4` CFG-parallel BF16**) | `87.35 s/step` (`437 s` / 5 steps) | **`10.56 s/step`** (`53.38 s` / 5 steps) | **`8.27×`** | 1-step `cos = 0.999999`; 5-step vs CP=8 `cos = 0.999985` |
| **Phase 2** (`e900499`) | Stage 2 `1080p` Refiner (opt-in `--refiner-qk-fp8`) | `87.35 s/step` | **`10.15 s/step`** (`50.75 s` / 5 steps warm) | **`8.61×`** | 5-step vs BF16 `cos = 0.998875` |
| **Phase 3** (`53b5c95`) | Stage 1 `540p` 100-step I2V `--lossy` (Audio+Video TeaCache `0.30` + late-step half-seq `cond-only` CFG skip `after=0.65`) | `318–338 s` lossless (`3.18–3.38 s/step`) | **`129.76 s`** (`1.266 s/step` eff.; `29` full + `17` cond-only @ `1.97 s` + `54` cached); warm E2E **`181.23 s`** | **`2.51×–2.61×`** | Case 2 Barista QA vs lossless: Video **`PSNR = 22.62 dB`** (`mean\|d\|=7.60/255`), Dialogue Audio **`-23.1 vs -23.5 dBFS`**, 50 ms speech envelope **corr `0.9793`** |
| **Phase 4** (`ffc4f17`) | Native JAX Resident Server (`jax_server.py`, `--requests` JSONL) | Cold reload per request (`33–320 s` startup + re-JIT) | **All 4 models BF16 resident in `41.80 GiB / 94.74 GiB` HBM/chip** (`33.65 s` warm startup once); req #2 zero-reload via `bind_sample_lens` (`s3v=0.19 s`, `s3a=0.04 s`) | **`316×` s3v / `363×` s3a** warm; **0 s** reload | Verified 2 distinct prompt lengths (`cafe` + `fox`) with zero extra HBM weight copies |
| **Phase 4** (`ffc4f17`) | `torch_tpu` E2E (`run_magi2_torchtpu.sh --prompt --enhance --decode-mp4`) + shared rank-0 `DT_AOT_PERSIST` (`272p FP8`) | `32.57 s` cold compile / `9.18 s` warm Inductor+XLA; manual 3-step CLI | **`1.45 s`** compile via `DT_AOT_PERSIST` (`178 MiB` rank-0 bins: `mm 35M`, `moe 141M`, `post 2.7M`); **`0.717 s/step`** denoise + full MP4 mux | **`22.5×`** vs cold / **`6.3×`** vs warm Inductor+XLA | Identical cross-process latent stats (`video std=1.9196`, `audio std=0.6711`) |

### 8.2 Three-Backend Stage 1 + Stage 2 Performance Summary (`v7x-8`, 10 s Audio+Video, CFG=2)

| Configuration (`v7x-8`) | Native JAX | `torchax` | `torch_tpu` (`8-rank`, `spmd_safe`) |
| :--- | :---: | :---: | :---: |
| Stage 1 `272p BF16` (`S=29,664`) | `0.960 s/step` | `0.960 s/step` | **`0.923 s/step`** |
| Stage 1 `272p FP8-W8A8` | — | `0.963 s/step` | **`0.717–0.829 s/step`** (`1.45 s` warm `DT_AOT_PERSIST` compile) |
| Stage 1 `540p BF16` (`S=115,680`, lossless) | `3.415 s/step` (`67.7 GiB` peak) | `3.728 s/step` | **`3.331 s/step`** |
| Stage 1 `540p FP8-W8A8` (lossless) | `3.430 s/step` (`42.7 GiB` HBM, `PSNR 20.83 dB`) | `3.745 s/step` | **`3.248 s/step`** |
| Stage 1 `540p BF16 --lossy` (100 steps I2V E2E) | **`1.266 s/step` eff. (`129.76 s` denoise / `181.23 s` warm E2E)** | — | — |
| Stage 2 `1080p` Refiner (`1088×1920 @ 25 fps`, 5 steps, default BF16 `DP=2×CP=4`) | **`10.56 s/step` (`53.38 s` / 5 steps, `8.27×` lossless)** | — | — |
| Stage 2 `1080p` Refiner (`--refiner-qk-fp8`, 5 steps) | **`10.15 s/step` (`50.75 s` warm, `8.61×`)** | — | — |

### 8.3 Key Engineering Decisions & Data-Driven Drops / 关键工程决策与实测否决项

- **Stage 2 Refiner Kernel (`jax_refiner_dit.py`)**: Micro-profiling proved legacy `_masked_pallas_block_grid_attention` accounted for **88.5%** (`41.48 s/pass`) of total Refiner time while Dense MatMuls took only `0.51 s/pass` and CP=8 all-to-all took `0.53 s/pass`. Fusing all 4 GQA query heads per KV head (`bq=512`), coalescing 5 contiguous W-blocks per DMA (`bkv=640` via `pl.Element`), keeping tail KV resident in VMEM, adding a dedicated `bkv=1024` tail-Q kernel (`32×` fewer grid steps), and running CFG branches in parallel (`DP=2 × CP=4`) delivered **`8.27×` lossless** speedup (`87.35 -> 10.56 s/step`).
- **Stage 1 Zero-HBM Multi-Length Compilation (`bind_sample_lens`)**: Because MAGI-2 Stage 1 attention has no text pad mask (injecting zero-padded text tokens would corrupt attention), we added `core_fn.bind_sample_lens(sample_lens)` sharing the single resident 113.93B packed parameter state with **zero extra HBM** — powering both Phase 3b half-sequence `cond-only` CFG skip (`1.97 s/step`) and Phase 4 resident multi-request serving (`41.80 GiB` HBM total for all 4 models in BF16).
- **`torch_tpu` `DT_AOT_PERSIST` Closure Hygiene (`torchtpu_magi2.py`)**: Hoisting `_top_run_mm_layer`, `_top_run_moe_layer`, `_top_run_group`, and `_top_run_post` to module top-level (avoiding closures over `self.hooks` / `torch._C._DispatchModule`) enabled clean single-copy rank-0 `.bin` serialization (`178 MiB` total for all 8 ranks), reducing warm cross-process compile from `32.57 s` cold (`9.18 s` warm Inductor/XLA) down to **`1.45 s`**.
- **Intentionally Rejected Optimizations (Measured, Not Worth Keeping)**:
  1. **Refiner Dense MatMul FP8**: Dense MatMuls are only `0.51 s/pass` (`1.1%` of Refiner); FP8 gave `1.03×` total gain -> skipped to preserve lossless BF16.
  2. **Stage 1 Splash Attention `qk_fp8`**: Measured `0.705 s/step` (`8.81 ms/layer`) vs `0.682 s/step` (`8.52 ms/layer`) in BF16 on MHA (`Hq=Hkv=24`) due to Q/K cast overhead -> dropped; kept lossless BF16 Splash (with redundant `jnp.repeat` removed).
  3. **Stage 1 Dense/Attn/Shared `GroupedLinear` FP8**: Measured `0.077 s/step` vs `0.096 s/step` BF16 (only `19 ms/step`, `0.6%`) -> kept lossless BF16.
  4. **Stage 1 `moe_mode=ep` (`ragged_all_to_all`)**: Real-weight 540p 4-step benchmark measured **`4.92 s/step`** vs **`3.45 s/step`** for `moe_mode=ag` (`1.42×` slower on 8-chip ICI due to variable-length routing imbalance + metadata sync) -> kept `ag` as default.

---

## 9. `VideoGym-LTX-2.3` (22B Audio-Video DiT) on TPU `v7x-8` / `VideoGym-LTX-2.3` 在 `v7x-8` 上的优化台账

Verified on `v7x-8` (`jax 0.12.0.1`, `diffusers 0.42.0.dev0`, `libtpu 0.0.52`, `94.74 GiB` HBM/device) against the `v6e-8` handoff baseline (`3aa98c9`: `15.03 s` E2E, `481 ms/fwd`, native JAX video decoder) at official model-card settings (`768×512×129 @ 24 fps`, `5.33 s` clip, 30 Euler steps, 4 CFG/STG/Modality rows, `maze_canvas.png`).

### 9.1 Single-Forward DiT Sweep (`rows=4`, `6,528` video + `134` audio + `1,024` text tokens)

| Configuration (`v7x-8` vs `v6e-8`) | Mean / Min (ms/fwd) | DiT HBM | xplane 2-fwd Breakdown (`overlap.py`) | Parity vs `1×8 BF16` |
| :--- | :---: | :---: | :--- | :--- |
| `v6e-8` baseline (`1×8, resident=20`, BF16) | `481.0` | `18.0 GiB` | compute `314`, AG `126` (`0` overlap), idle `35` | — |
| `v7x-8` `1×8, resident=48`, BF16 | **`250.6 / 249.3`** (`1.92×`) | `35.65 GiB` | `window 493: compute 446, comm 0, idle 47` | ref |
| `v7x-8` `2×4, resident=48`, BF16 | `254.8 / 253.4` | `35.64 GiB` | `window 500: compute 444, comm 2, idle 54` | `vid 7.66e-3, aud 1.51e-2` |
| **`v7x-8` `4×2` (`cfg=4, sp=2`), `resident=48`, BF16 (auto default)** | **`207.2 / 206.0`** (**`2.33×`**) | `35.63 GiB` | **`window 407: compute 390 (95.8% util), comm 9, idle 8`** | `vid 7.73e-3, aud 1.90e-2` |
| `v7x-8` `1×8, resident=48`, FP8-W8A8 | `228.9 / 226.4` (`2.10×`) | `18.44 GiB` | — | `vid 2.33e-2, aud 7.26e-2` |
| **`v7x-8` `4×2, resident=48`, FP8-W8A8 (`--fp8`)** | **`184.4 / 182.2`** (**`2.64×`**) | **`18.41 GiB`** | **`window 358: compute 333 (93.0% util), comm 9, idle 16`** | `vid 2.33e-2, aud 7.27e-2` |
| `v7x-8` `4×2, resident=48`, FP8 + `--qk-fp8` | `183.1 / 180.8` | `18.41 GiB` | `-1.3 ms` vs `--fp8` alone (negligible gain) | — |

- **8-virtual-device CPU FP32 exactness**: `1×8` (`vid_rel=2.21e-7`), `2×4` (`2.19e-7`), `4×2` (`2.80e-7`) all match single-device `jax_dit.forward` to FP32 machine epsilon.
- **Why `4×2` (`cfg=4, sp=2`) beats `1×8` by `21%` on `v7x-8` (`250.6 -> 206.0 ms/fwd`, 100% lossless)**:
  1. `mesh_utils.create_device_mesh((4, 2))` places `sp=2` across the **2 logical cores of the same physical `v7x` chip**, keeping all 240 collectives per forward (Ulysses `all_to_all` + `v2a` `pmax`/`psum`) **100% intra-chip** (zero inter-chip ICI during all 48 DiT blocks; `idle` drops `47 -> 8 ms`).
  2. Each core runs `B=1` row instead of `B=4`, cutting redundant audio-stream (`audio_attn1/2`, `audio_ff`, `v2a`) and text K/V projection (`attn2.to_k/v`) compute by **`4×`** (`compute` drops `223 -> 195 ms/fwd`).

### 9.2 Full 30-Step End-to-End Maze Pipeline (`768×512×129 @ 24 fps`, `5.33 s` video + audio)

| Configuration | Warm E2E (s) | Encode (s) | Denoise (30 steps) | Decode (s) | Peak HBM | Load (s) | 30-Step Maze QA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `v6e-8` (`3aa98c9`, `1×8, res=20`, BF16) | `15.03` | `0.08` | `14.10` (`0.470/step`) | `0.84` | `27.6 GiB` | — | Maze solved |
| `v7x-8` `1×8, res=48`, BF16 (CPU `uint8` decode) | `8.79` (`1.71×`) | `0.04` | `7.37` (`0.246/step`) | `1.39` | `44.2 GiB` | `48.0` | Ref (`v7x` `1×8`) |
| `v7x-8` `4×2, res=48`, BF16 (CPU `uint8` decode) | `7.19` (`2.09×`) | `0.04` | `6.07` (`0.202/step`) | `1.08` | `44.2 GiB` | `50.7` | **`PSNR = 45.51 dB`** vs `1×8` |
| **`v7x-8` `4×2, res=48`, BF16 (auto default, on-TPU `uint8` decode)** | **`6.60`** (**`2.28×`**) | `0.04` | **`6.06`** (`0.202/step`) | **`0.49`** | `44.2 GiB` | `48–51` | **Byte-exact MP4** (`209,797 B`) vs CPU `uint8`; **`45.51 dB`** vs `1×8` |
| **`v7x-8` `4×2, res=48`, `--fp8` (on-TPU FP8 load + on-TPU `uint8` decode)** | **`5.82`** (**`2.58×`**) | `0.04` | **`5.29`** (**`0.176/step`**) | **`0.49`** | **`27.0 GiB`** | **`44.3`** (`4.95×` vs `219 s` CPU quant) | **`PSNR = 40.00 dB`** vs `1×8`; visually indistinguishable maze solve; warm cache `run 0 = 29.64 s` |

- **Decode breakdown (`0.493 s` total)**: native JAX `uint8` video decoder (`jax_vae.make_decoder(..., to_uint8=True)`, cutting PCIe D2H `4×` from `608 -> 152 MB` and eliminating CPU `np.round`) takes **`0.262 s` (`min 0.255 s`, `506 FPS`!)** + torchax audio VAE/vocoder **`0.231 s` (`min 0.230 s`)**.

### 9.3 `c947` Open Question on `v7x-8` (`1×8, resident=20`, BF16)

| Experiment (`resident=20` on `v7x-8`) | HLO Collectives | Mean (ms/fwd) | HBM | xplane 2-fwd (`overlap.py`) | Verdict |
| :--- | :--- | :---: | :---: | :--- | :--- |
| Baseline (default flags, 34 `all_gather`/block) | `all-gather=952, all-gather-start=0, all-to-all-start=192` | **`296.3`** (`min 293.8`) | `18.03 GiB` | `window 588: compute 311, comm 549, overlap 287 (92.3%!), exposed 262, idle 14` | **`v7x` backend overlaps 92.3% of compute even without HLO `all-gather-start`** (`+45.7 ms` vs `res=48`, unlike `+126 ms` zero-overlap on `v6e`) |
| Packed `--prefetch` (1 flat `all_gather`/block hoisted to block `i`) | `all-gather=28, all-gather-start=0, all-to-all-start=192` | `413.9` | `22.65 GiB` | `window 817: compute 491, comm 221, overlap 107, exposed 114, idle 213` | **Slower (`+117.6 ms`)**: packing/slicing added `+180 ms` compute & `+200 ms` idle — dropped |
| `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` | `FAILED_PRECONDITION` (`only supported on TPU v5e`) | — | — | — | **Unsupported on `v7x` as well as `v6e`** |
| `--xla_tpu_enable_ici_ag_pipelining=true` + prefetch | `all-gather=28, all-gather-start=0` | `414.3` | `22.65 GiB` | No delta | No effect |

