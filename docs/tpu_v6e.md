# TPU v6e (Trillium) Optimization Guide & Benchmarks / TPU v6e 硬件架构与优化指南

## 1. Hardware Architecture & Device Accounting / 硬件架构与 Device 计数口径

| Metric / 指标 | TPU v6e (Trillium) Specification / TPU v6e 规格 | Notes for `diffusers-tpu` / 代码与配置要点 |
| :--- | :--- | :--- |
| **Chip vs. TensorCore (Device)** | **1 physical chip = 1 TensorCore = 1 logical device (rank)**<br>每个物理 Chip 含 1 个 TensorCore，对应 1 个逻辑 Device / Rank | **`v6e-8` = 8 physical chips = 8 TensorCores (8 devices / 8 ranks)**<br>`TORCH_TPU_TOPOLOGY="2,4,1"` |
| **HBM Capacity** | **32 GiB HBM2e per chip (device)**<br>每个 Chip / Device 拥有 **32 GiB HBM** | 在 `v6e-8` 上，8-way TP 分片后的 35B DiT（`~7.5 GiB/chip`）与 VAE Decoder 可常驻 HBM（`--no-offload`），但 8-way TP 分片后的 Qwen3-VL Text Encoder（`7.3 GiB/chip`）在编码完成后需通过 `--free-text-after` 释放回主机内存，为 `768p 345f` 去噪激活值腾出 HBM 空间 |
| **MXU Systolic Array** | **`256 × 256` per TensorCore** (`bf16` / `int8`)<br>配备 **`256 × 256`** 脉动阵列（较 v5e 的 `128 × 128` 翻倍） | Pallas FlashAttention 与线性层分块需对齐 `256` 的整数倍（如 `block_q=2048, block_kv=4096`） |
| **VPU 2D Register Tile Layout** | **`(8, 128)` (`8 sublanes × 128 lanes`)** | 与 v7x 相同，逐元素算子与 RoPE 在 `(B, H, S, D=128)` 布局下可直接对齐 `(8, 128)` 寄存器 tile |

---

## 2. Benchmarks on TPU `v6e-8` (8 Chips) / `v6e-8` 性能汇总

### 2.1 MiniMax-H3 — `768×1344`, 345 frames (14.4 s), 20 steps

| Workflow / 模式 | JAX | `torchax` | `torch_tpu` |
| :--- | :---: | :---: | :---: |
| **text-to-video, lossless / 文生视频, 无损** | **110.0 s** | 110.0 s | 135.2 s |
| **text-to-video, lossy / 文生视频, 有损** | **34.0 s** | 34.8 s | 43.2 s |
| **reference-to-video, lossless / 参考图生视频, 无损** | **139.2 s** | 186.7 s | 189.0 s |
| **reference-to-video, lossy / 参考图生视频, 有损** | **51.2 s** | 51.6 s | 78.0 s |

### 2.2 Pika-H3-Flex — 6 steps, `768×1344`, 345 frames (with audio)

| Workflow / 模式 | JAX | `torchax` | `torch_tpu` |
| :--- | :---: | :---: | :---: |
| **text-to-video (`t2va`) / 文生视频** | **36.5 s** (`38.3 s` w/o fused uint8 decode) | **38.5 s** | **46.6 s** |
| **reference-to-video (`ref2va`) / 参考图生视频** | **51.1 s** | — | — |

---

## 3. Recent `torch_tpu` Progression on `v6e-8` / `v6e-8` 上的 `torch_tpu` 优化演进

| Date / Commit | Optimization / 优化项 | Pika 768p `t2va` | H3 20-step Lossy `t2va` | H3 20-step Lossy `ref2va` | Pixel Verification |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2026-11-08** | Baseline with on-host text encoder | `71.1 s` | `52.5 s` | — | Reference |
| **2026-11-08** | 8-chip TP Text Encoder (`--shard-text --free-text-after`) + resident VAE | `51.3 s` | `49.8 s` | `82.2 s` | `DINO 0.967` |
| **2026-11-10** (`c841add`) | Raw `bf16` `/dev/shm` chunk exchange + Rank 0-only 32-thread host postprocessing (`decode: 9.0 -> 6.0 s`) | `48.7 s` | `46.1 ~ 46.3 s` | `79.7 ~ 80.0 s` | **`identical=True`** (`max\|d\|=0`) |
| **2026-11-10** (`91d89fe`) | Pipelined `/dev/shm` writes + 8-thread parallel gather + zero-copy `.numpy()` (`Opt-1`) + 8-rank tail-chunk spatial tile split (`Opt-3`) | **`47.4 s`** | **`44.5 s`** | **`78.0 s`** | **`identical=True`** (`max\|d\|=0`) |
| **2026-11-11** (`1016192`) | Fused `residual + gate * x` in segmented modulation + BHSD-first `norm_q`/`norm_k`/RoPE | **`46.6 s`** | **`43.2 s`** | `78.0 s` (unchanged) | **`identical=True`** |

> **Why `Opt-2` (`DT_RESIDENT_TEXT=1`) is disabled on `v6e-8`**:
> Unlike `v7x-8` (95 GiB HBM per TensorCore device), each `v6e` chip has 32 GiB HBM. Keeping the `7.3 GiB` sharded Text Encoder resident alongside the `7.5 GiB` sharded DiT and VAE leaves insufficient HBM headroom for the `103,989`-token `768p` denoising activations, so `torchtpu_h3.py` enables `DT_RESIDENT_TEXT` automatically only on `v7x` (`TORCH_TPU_TOPOLOGY == "2,2,1,2"`).

> **`DT_POST_ATTN_PIPE=1` on `v6e-8`**: the two-slab post-attention pipeline (`52aba21`) runs out of HBM in the first 768p request (`RuntimeProgramAllocationFailure`), and the process then hangs instead of exiting; keep it off on `v6e`. On `v6e` the in-graph all-reduce is synchronous, so the slabs would not overlap anyway.

## 4. MiniMax-H3-Image on `v6e-8` / `v6e-8` 上的 MiniMax-H3-Image

Warm requests, end to end (text encoding, six DiT steps, decode). / 暖请求,端到端(文本编码、六步 DiT、解码)。

| Size / 尺寸 | JAX | `torchax` | `torch_tpu` |
| :--- | :---: | :---: | :---: |
| 2048x2048 t2i | **0.68 s** | 0.74 s | 0.75 s |
| 4736x3552 t2i | **3.08 s** | 3.12 s | 3.36 s |
| 2048x2048 edit, 1 reference / 编辑 | **1.94 s** | 2.10 s | 3.22 s |
| 4096x4096 edit, 1 reference / 编辑 | **4.37 s** | — | — |

An image is a 4k-16k-row sequence, so eight-way tensor parallelism is communication-bound. What made the JAX transformer 2.4x faster at 4736x3552 (denoise 7.42 -> 2.99 s), in order:
图像是 4k-16k 行的短序列,8 路张量并行受通信限制。JAX transformer 在 4736x3552 上快 2.4 倍(去噪 7.42 -> 2.99 秒)依次来自:

| Change / 改动 | Denoise / 去噪 |
| :--- | :---: |
| Partitioner's default (sequence-sharded stream) / 分区器默认(按序列切的残差流) | 7.42 s |
| `DT_STREAM_SHARD=whole` (residual stream replicated) / 残差流整份复制 | 4.24 s |
| `--full-lane-rope`, `--segments` | 4.01 s |
| Attention output left split by heads (`DT_ATTN_KEEP_HEADS`) / 注意力输出保持按头切 | 3.74 s |
| SwiGLU rows interleaved per chip (`DT_SWIGLU_INTERLEAVE`) / SwiGLU 行按 chip 交错 | **2.99 s** |

Each block then holds two plain all-reduces and nothing else crosses the mesh; on v6e those do not overlap compute (about 40% of a step). Tried and dropped: sequence-parallel stream with the same fixes (4.54 s), Pallas SwiGLU FFN (3.22 s), hand-written collective matmul and XLA flag variants (no change).
此后每个 block 只有两次普通 all-reduce,再无其他跨 chip 交换;v6e 上 all-reduce 不与计算重叠(约占一步的 40%)。试过但放弃:同样修正下的序列并行残差流(4.54 秒)、Pallas SwiGLU FFN(3.22 秒)、手写 collective matmul 与 XLA 标志组合(无变化)。

Fewer chips / 更少的 chip: 4 chips with `--text-int8` 0.93 s / 4.38 s; one v5e chip with `--offload --int8-weights --free-text-after` 11.4 s / 43.7 s (2048x2048 / 4736x3552).
4 个 chip 加 `--text-int8`:0.93 / 4.38 秒;单个 v5e chip 加 `--offload --int8-weights --free-text-after`:11.4 / 43.7 秒(2048x2048 / 4736x3552)。
