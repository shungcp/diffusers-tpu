# TPU v5e (`v5litepod-1` / Colab) Optimization Guide & Benchmarks / TPU v5e 硬件架构与优化指南

## 1. Hardware Architecture & Constraints / 硬件架构与资源约束

| Metric / 指标 | TPU v5e Specification / TPU v5e 规格 | Notes for `diffusers-tpu` / 代码与配置要点 |
| :--- | :--- | :--- |
| **Chip vs. TensorCore (Device)** | **1 physical chip = 1 TensorCore = 1 logical device** | 单卡环境（`v5litepod-1` 或免费 Google Colab TPU v5e-1 Runtime） |
| **HBM Capacity** | **16 GiB HBM per chip** | 35B DiT 原始 `bf16` 权重（~70 GB）无法常驻 HBM，必须开启 `--offload --int8-weights --adaln-host`，将 DiT 权重量化为 `int8`（`19.6 GB`）驻留主机内存按层流式加载 |
| **MXU Systolic Array** | **`128 × 128` per TensorCore** (`bf16` / `int8`) | 注意：v5e 的 MXU 为 **`128 × 128`**（而 v6e/v7x 为 `256 × 256`）；开启 `DT_INT8_ACT=1` 可同时量化激活值以调用 `int8` 矩阵单元，去噪提速 10–22% |
| **VPU 2D Register Tile Layout** | **`(8, 128)` (`8 sublanes × 128 lanes`)** | 与 v6e/v7x 保持一致 |

---

## 2. Benchmarks on One TPU v5e Chip (`v5litepod-1`) / 单卡 TPU v5e 性能汇总

Workload / 测试负载：**Pika-H3-Flex 6 steps**, `--chips 1 --offload --int8-weights --adaln-host`.

| Workflow / 模式 | `480×832`, 141 frames (6 s) | `480×832`, 362 frames (15 s) | `768×1344`, 345 frames (14.4 s) |
| :--- | :---: | :---: | :---: |
| **text-to-video (`t2va`) / 文生视频** | **55 s** | **272 s** (**251 s** w/ `DT_INT8_ACT=1`) | **748 s** |
| **reference-to-video (`ref2va`) / 参考图生视频** | **121 s** | **314 s** (**288 s** w/ `DT_INT8_ACT=1`) | **1005 s** |

---

## 3. Key Optimizations & Flags on TPU v5e / 单卡 v5e 关键优化与参数

| Flag / Feature | Mechanism & Effect / 作用机制与实测效果 |
| :--- | :--- |
| `--int8-weights` | Halves DiT weights in host RAM to `19.6 GB`, halving PCIe H2D streaming time every denoising step. |
| `--adaln-host` | Precomputes the timestep modulation on the host CPU so `26 GB` of `adaln_proj` weights (`2688 × (6 × 5376 × 3)` across 50 blocks) never cross PCIe to the chip, with zero change to outputs. |
| `DT_WEIGHT_CACHE=<dir>` | Caches converted `int8` weights on disk using sequential reads (`824a60d` instead of `mmap` page faults), reducing startup load time from `375 s` to `178 s`. |
| `DT_FF_CHUNK=4096`<br>`DT_QKV_CHUNK=16384` | Required at `768×1344, 345 frames` (`--block-q 1024 --block-kv 2048` for `t2va`, `--block-q 512 --block-kv 1024` for `ref2va`) to bound peak activation HBM within 16 GiB. |
| `DT_INT8_ACT=1` | Quantizes activations to `int8` as well to run matmuls on the `int8` MXU (`10–22%` faster denoise). |
| **Background Host Audio Decode** | Decodes the BigVGAN audio waveform on the host CPU concurrently while the TPU decodes the video latents, adding `0.0 s` wall-clock overhead on `v5litepod-1`. |
