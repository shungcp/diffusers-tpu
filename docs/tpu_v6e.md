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
