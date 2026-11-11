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
| **+ Opt-4 + Opt-5a**<br>(`DT_FUSED_RES_MOD=1`<br>`DT_BHSD_ROPE=1`) | **Denoise block optimizations (100% bit-identical):**<br>1. `by_runs_add`: fuse `residual + gate * x` before `index_copy`, saving two `1.06 GB` full-sequence HBM writes/block;<br>2. Transpose Q/K/V to `(B, H, S, D)` immediately after `unflatten` to align `(S, 128)` with the `(8, 128)` VPU tile before `norm_q/k` and RoPE, dropping `.contiguous()` | **`34.6 s`**<br>*(`-0.5 s`, cumulative **`-13.5 s`**)* | `1.4 s` | **`29.2 s`**<br>*(`-0.6 s`, `4.87s/step`)* | **`4.1 s`**<br>*(other 7 devices `3.9s`)* | • `tail-chunk 16`: **`0.9 s`**<br>• `gather`: **`0.3 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | **`identical=True`**<br>`max\|d\|=0`<br>`mean\|d\|=0.0000` | `1016192`<br>*(Default on)* |
| **+ Opt-5b (Optional)**<br>(`DT_BHSD_ROPE=full`) | Replace `chunk+cat` RoPE with 128-lane MXU matmul `x * cos128 + (x @ R) * sin128` | **`34.3 s`**<br>*(`-0.3 s`)* | `1.2 s` | **`29.0 s`**<br>*(`4.83s/step`)* | **`4.1 s`** | Same as above | `identical=False`<br>`PSNR=19.2 dB` | `1016192`<br>*(Opt-in)* |
| **+ Opt-6 (2-Slab Pipe)**<br>(`DT_POST_ATTN_PIPE=1`) | **2-Slab Post-Attention Sub-block Pipeline (`S0 = 49152 = 192×256`):**<br>Split `to_out -> AR1 -> gate_msa -> norm2 -> mod_mlp -> SwiGLU FFN -> AR2 -> gate_mlp` at `S0=49152` (aligned to `256×256` MXU and `4096`), overlapping `AR1(1)` with `Slab 0 norm2+FFN` and `AR2(0)` with `Slab 1 norm2+FFN` (100% video rows, zero `index_copy`), with only 1 `torch.cat` per block | **`32.0 s`**<br>*(`-2.6 s`, cumulative **`-16.1 s` / `-33.5%`**)* | **`1.1 s`** | **`26.8 s`**<br>*(`-2.4 s`, `4.47s/step`)* | **`4.1 s`**<br>*(other 7 devices `3.9s`)* | • `tail-chunk 16`: **`0.9 s`**<br>• `gather`: **`0.3 s`**<br>• `blend`: **`0.2 s`**<br>• `postprocess 0.2s + to np` **`0.0 s`** | `identical=False`<br>`mean\|d\|=9.90`<br>`PSNR=21.8 dB`<br>*(1-ULP bf16 tile rounding)* | `52aba21`<br>(`DT_POST_ATTN_PIPE=1`) |

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
> - **`torch_tpu` (`bf16`, `DT_POST_ATTN_PIPE=1`)**: **`32.0 s`** (`denoise/other 26.8 s + vae_decode 4.1 s + text_encoder 1.1 s`)
> - **`torch_tpu` (`bf16`, default bit-identical)**: **`34.6 s`** (`denoise/other 29.2 s + vae_decode 4.1 s + text_encoder 1.4 s`)

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

| Hardware / Stack | Precision | 2K (`2048×2048`) `t2i` | 2K (`2048×2048`) `edit` | 4K (`4736×3552` / `4096×4096`) `t2i` | Stage Breakdown (`text + denoise + vae`) / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1 × RTX PRO 6000 Blackwell** (SGLang) | BF16 / FP8 | `3.434 s` / `2.338 s` | — | `18.544 s` / `14.445 s` | Warm pipeline median (excl. text encoder) |
| **1 × NVIDIA H200** (SGLang / vLLM-Omni) | BF16 | `1.934 s` (`3.20 s` e2e) | — | `10.133 s` | SGLang warm pipeline / vLLM-Omni client e2e |
| **2 × NVIDIA H200** (SGLang Ulysses2 / vLLM-Omni) | BF16 | `1.123 s` (`1.50 s` e2e) | — | `5.465 s` | 2-way Ulysses SP / TP |
| **4 × NVIDIA H200** (vLLM-Omni Ulysses SP4) | BF16 | `0.88 s` (`TP4: 1.00 s`) | `1.35 s` | `3.84 s` | Resident Qwen3-VL + DiT + VAE |
| **2 × NVIDIA B300** (SGLang Ulysses2) | BF16 | `0.604 s` | `0.931 s` | `2.654 s` (`edit: 3.146 s`) | Warm pipeline median (excl. text encoder) |
| **TPU `v7x-8` (`JAX`)** | **BF16** | **`0.861 s`** (`0.879 s` e2e) | **`1.089 s`** (`1.496 s` w/ ref VAE) | **`3.155 s`** (`3.174 s` e2e) | `0.018 s + 0.834 s / 3.081 s + 0.024 s / 0.064 s` |
| **TPU `v7x-8` (`JAX`)** | **FP8** | **`0.833 s`** (`0.852 s` e2e) | **`0.929 s`** (`1.322 s` w/ ref VAE) | **`3.053 s`** (`3.074 s` e2e) | `0.019 s + 0.806 s / 2.975 s + 0.024 s / 0.066 s` |
| **TPU `v7x-8` (`torchax`)** | **BF16** | **`1.057 s`** (`1.075 s` e2e) | — | **`2.941 s`** (`2.961 s` e2e) | `0.018 s + 1.031 s / 2.865 s + 0.023 s / 0.064 s` |
| **TPU `v7x-8` (`torch_tpu`)** | **BF16** | **`0.944 s`** (`1.287 s` e2e) | — | **`3.121 s` / `3.358 s`** (`3.709 s` e2e) | `denoise` **`0.721 s` (2K) / `2.442 s` (4K)** + `vae 0.219 s / 0.662 s` |
| **TPU `v7x-8` (`torch_tpu`, `DT_POST_ATTN_PIPE=1`)** | **BF16** | **`0.944 s`** (`1.343 s` e2e) | — | **`2.673 s` / `2.758 s`** (`3.086 s` e2e) | `denoise` **`2.034 s` / `2.049 s` (4K)** + `vae 0.623 s / 0.696 s` |

### 5.2 `MiniMax-H3-Image` Optimization Progression on `v7x-8`

| Step | Optimization / 核心优化项 | 2K (`2048×2048`) | 4K (`4736×3552` / `4096×4096`) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Opt-0** | Initial `JAX` baseline (`--adaln-host` inter-block barrier, unaligned `seq`, sublane RoPE) | `2.202 s` (`FP8: 1.937 s`) | `7.771 s` (`FP8: 6.947 s`) | `1.00×` |
| **Opt-1** | Remove inter-block `optimization_barrier` on 8-device TP (`_tp_size() > 2`) + 8-way sharded `adaln_proj` + `--segments` + `--full-lane-rope` + `--pad-sequence-to 128` | **`0.861 s`** (`FP8: 0.833 s`) | **`3.155 s`** (`FP8: 3.053 s`) | **`2.56×` (2K) / `2.46×` (4K)** |
| **Opt-2** | `torchax` hybrid pipeline with custom Pallas Splash Attention (`jquery.shape[2] == jkey.shape[2]` self-attention routing) + segmented modulation + JAX 2D ViT tile-sharded VAE (`0.023 s` at 2K, `0.064 s` at 4K) | **`1.057 s`** (`1.075 s` e2e) | **`2.941 s`** (`2.961 s` e2e) | **`2.64×` (4K)** |
| **Opt-3** | JIT-compiled 3D causal reference VAE encoder (`_enc_one_tile`) + elementwise `jnp.where` for long `head_runs` (`>= 512` tokens) in `edit` mode | `2.787 s` -> **`1.496 s`** (BF16) / `2.908 s` -> **`1.322 s`** (FP8; `denoise+vae` **`0.929 s`**) | — | **`2.20×` (2K Edit)** |
| **Opt-4** | `torch_tpu` 8-rank TP (`AotBlocks(group=10)` + BHSD norm/RoPE + fused segmented mod + 8-rank batched 2D ViT VAE decode + `DT_POST_ATTN_PIPE=1` 2-slab overlap on 4K) | **`0.944 s`** (`denoise` **`0.721 s`**) | `3.121 s` -> **`2.673 s`** (`denoise` `2.442 s` -> **`2.034 s`**) | **`2.91×` (4K)** |

