# diffusers-tpu

Fast inference for open diffusion models on Cloud TPU, built on top of [huggingface/diffusers](https://github.com/huggingface/diffusers).
基于 [huggingface/diffusers](https://github.com/huggingface/diffusers),在 Cloud TPU 上高速运行开源扩散模型。

Two models are supported today: **MiniMax-H3** (video + audio) and **Qwen-Image-2.1** (image).
目前支持两个模型:**MiniMax-H3**(视频 + 音频)和 **Qwen-Image-2.1**(图像)。

MiniMax-H3 also runs with the six-step **Pika-H3-Flex** adapters.
MiniMax-H3 还可以搭配 **Pika-H3-Flex** 六步适配器运行。

Every model can run through three backends, so you can pick the one that fits your stack.
每个模型都能用三种后端运行,可以按自己的技术栈选择。

| Backend / 后端 | What it is / 是什么 | Status / 状态 |
|---|---|---|
| **JAX** | The model rewritten in flax, with our Pallas attention kernels / 用 flax 重写的模型,配合自研 Pallas 注意力 kernel | Fastest, recommended / 最快,推荐 |
| torchax | The diffusers PyTorch code, executed through JAX / 直接运行 diffusers 的 PyTorch 代码,经 JAX 执行 | No rewrite needed / 无需重写 |
| torch_tpu | Eager PyTorch on TPU, one process per chip / 在 TPU 上逐算子运行 PyTorch,每个 chip 一个进程 | For PyTorch-native stacks / 适合纯 PyTorch 技术栈 |

---

## News / 更新

Newest first. Each entry links a date to what changed.
最新的在最上面。每条记录写明日期和变化。

- **2026-11-07** — Back-to-back requests on the resident server: 33 s -> 15.3 s each (480p reference-to-video on a v6e-8). Saving each clip's frames took 18 s, outside the timed request, because they reached the host as a strided array; the chips now hand back a contiguous one (0.2 s). Output unchanged.
  **2026-11-07** — 常驻服务连续处理请求:每条从 33 秒降到 15.3 秒(v6e-8 上 480p 参考图生视频)。原来每段视频的帧存盘要 18 秒(不在计时范围内),原因是帧以跨步数组的形式回到主机;现在 chip 直接交回连续数组(0.2 秒)。输出不变。

- **2026-11-06** — Faster video decode (tile extraction and blending fused, frames converted to uint8 on the chips): 768p Pika text-to-video 40.6 -> 38.3 s on a v6e-8 and 31.3 -> 29.3 s on a v7x-8; output unchanged (DINO 0.9999+). Weights are now read with sequential reads instead of memory maps.
  **2026-11-06** — 视频解码加速(切块和拼接融合,帧在 chip 上转成 uint8):v6e-8 上 768p Pika 文生视频 40.6 -> 38.3 秒,v7x-8 上 31.3 -> 29.3 秒;输出不变(DINO 0.9999 以上)。权重改为顺序读取,不再用内存映射。

- **2026-11-04** — TPU v7x results on 1, 2, 4 and 8 devices (a v7x chip has two TensorCores, each a JAX device with 95 GB of HBM, so no `--offload` is needed even at 768p on one device), plus a sequence-tiled Pallas SwiGLU FFN kernel (`--pallas-ffn`): Pika-H3-Flex generates a 14-second 768p clip in 36.0 s on a v7x-8 (8 devices, 4 chips; 13.5 s at 480p), 63.3 s on 4 devices, 108.4 s on 2 devices, and 197.6 s on a single device.
  **2026-11-04** — 新增 TPU v7x 在 1、2、4、8 个 device 上的性能数据(一个 v7x chip 有两个 TensorCore,每个是一个 JAX device,各有 95 GB HBM,所以单个 device 跑 768p 也无需 `--offload`)及序列分块 Pallas SwiGLU FFN kernel(`--pallas-ffn`):Pika-H3-Flex 生成 14 秒 768p 视频在 v7x-8(8 个 device,4 个 chip)上为 36.0 秒(480p 为 13.5 秒),4 个 device 为 63.3 秒,2 个 device 为 108.4 秒,单个 device 为 197.6 秒。

- **2026-11-04** — The JAX backend now decodes the soundtrack as well (its videos used to be silent); `scripts/frames_to_mp4.py` muxes it in. It adds about 0.1 s on a v6e-8 and nothing on one chip.
  **2026-11-04** — JAX 后端现在也解码音轨(之前生成的视频没有声音),`scripts/frames_to_mp4.py` 会把音轨合进 mp4。在 v6e-8 上多约 0.1 秒,单 chip 上不增加时间。

- **2026-11-03** — Pika-H3-Flex six-step adapters: a 14-second 768p clip in 40.3 s on a v6e-8. MiniMax-H3 now also runs end to end on a single TPU v5e chip.
  **2026-11-03** — 支持 Pika-H3-Flex 六步适配器:在 v6e-8 上 40.3 秒生成一段 14 秒的 768p 视频。MiniMax-H3 现在也能在单个 TPU v5e chip 上端到端运行。

- **2026-11-03** — First release: MiniMax-H3 and Qwen-Image-2.1, each on JAX, torchax and torch_tpu.
  **2026-11-03** — 首次发布:MiniMax-H3 和 Qwen-Image-2.1,各支持 JAX、torchax、torch_tpu 三种后端。

---

## Results / 性能

Measured 2026-11-02 to 2026-11-04 on a TPU v6e-8 (8 chips); warm requests, with the model already loaded and compiled.
测于 2026-11-02 至 2026-11-04,TPU v6e-8(8 个 chip);均为热请求时间,模型已加载并编译完成。

Software: jax 0.12.0.1, libtpu 0.0.52 (0.0.51 for torch_tpu; JAX numbers re-measured on 0.0.52, 2026-11-07), torch_tpu 0.2.0, diffusers `cc8644b` (see `requirements/`).
软件版本:jax 0.12.0.1、libtpu 0.0.52(torch_tpu 用 0.0.51;JAX 数字已于 2026-11-07 在 0.0.52 上重测)、torch_tpu 0.2.0、diffusers `cc8644b`(见 `requirements/`)。

### MiniMax-H3 — 768x1344, 345 frames (14.4 s), 20 steps

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| text-to-video, lossless / 文生视频,无损 | **110.0 s** | 117.8 s | 139.0 s |
| text-to-video, lossy / 文生视频,有损 | **34.0 s** | 42.9 s | 52.5 s |
| reference-to-video, lossless / 参考图生视频,无损 | **139.2 s** | 186.7 s | 189 s |
| reference-to-video, lossy / 参考图生视频,有损 | **51.2 s** | 77.3 s | 86 s |

JAX times include decoding the audio; the two lossless JAX numbers were measured before it was added (about 0.1 s more).
JAX 的时间包含音频解码;两个无损 JAX 数字是加入音频解码之前测的(会多约 0.1 秒)。

"Lossy" skips a DiT call while the timestep embedding barely changes, and lets each video frame attend only to nearby frames.
"有损"是指:时间步嵌入几乎不变时跳过一次 DiT 计算,并且让每一帧只关注相邻的帧。

The visual quality stays close to lossless; check it on your own prompts before relying on it.
画质与无损版本接近;正式使用前,请先用自己的提示词检查效果。

The resident server keeps the model on the chips between requests: 34.0 s (text) and 47.5 s (reference) per lossy request.
常驻服务在请求之间把模型保留在 chip 上:每个有损请求 34.0 秒(文生)和 47.5 秒(参考图生)。

### Qwen-Image-2.1 — 1024x1024

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| 20 steps / 20 步 | **0.74 s** | 1.28 s | 1.18 s |
| 4 steps / 4 步 | **0.28 s** | 0.49 s | 0.55 s |

All three backends use all 8 chips for Qwen-Image.
Qwen-Image 的三种后端都使用全部 8 个 chip。

### Pika-H3-Flex — six steps, JAX backend

Measured 2026-11-04 on a TPU v6e-8, same size as the MiniMax-H3 table (768x1344, 345 frames), audio included.
测于 2026-11-04,TPU v6e-8,尺寸与上面 MiniMax-H3 的表相同(768x1344,345 帧),含音频解码。

| | Pika-H3-Flex, 6 steps / 6 步 | MiniMax-H3, lossless / 无损 |
|---|---|---|
| text-to-video / 文生视频 | **36.5 s** | 110.0 s |
| reference-to-video / 参考图生视频 | **51.1 s** | 139.2 s |

Our six-step sampler matches Pika's own code within numerical noise, checked with the same initial noise.
我们的六步采样与 Pika 官方代码一致,差异在数值噪声范围内(用相同的初始噪声核对过)。

### Short clips in six steps / 六步生成短片 — 480x832, 362 frames (15 s)

A resident server (`--runs`) keeps the model and decoder on the chips; these are its steady-state times per request.
常驻服务(`--runs`)把模型和解码器保留在 chip 上;下面是它每个请求的稳态时间。

| Pika-H3-Flex, v6e-8 | per request / 每个请求 |
|---|---|
| text-to-video / 文生视频 | **10.1 s** |
| reference-to-video, same reference image / 参考图生视频,参考图不变 | **12.7 s** |
| reference-to-video, a new reference image every request / 参考图生视频,每次换新参考图 | **13.8 s** |

Reference-to-video uses `--sparse-window 16 --sparse-dense-steps 2`; a new image adds preparing it on the host (0.25 s), the text/vision encode (0.6 s) and the reference encode (0.45 s), all included in 13.8 s.
参考图生视频使用 `--sparse-window 16 --sparse-dense-steps 2`;换新图时要加上主机上准备图片(0.25 秒)、文本/视觉编码(0.6 秒)和参考图编码(0.45 秒),都已计入 13.8 秒。

Without the sparse attention, reference-to-video takes 15.0 s with the same image and 16.4 s with a new one.
不用稀疏注意力时,参考图生视频在参考图不变时为 15.0 秒,每次换新图时为 16.4 秒。

The same sparse attention brings text-to-video to 9.9 s.
文生视频加上同样的稀疏注意力是 9.9 秒。

Through the resident server, with a new prompt or reference image on every request, text-to-video takes 10.6 s and reference-to-video 14.6–15.4 s (`sparse_dense_steps` 3).
通过常驻服务、每个请求换新的提示词或参考图时,文生视频为 10.6 秒,参考图生视频为 14.6–15.4 秒(`sparse_dense_steps` 设为 3)。

### Fewer chips / 更少的 chip

Measured 2026-11-02 to 2026-11-03; "4 chips" means four chips of the same v6e-8 host.
测于 2026-11-02 至 2026-11-03;"4 chip"指同一台 v6e-8 主机上的 4 个 chip。

| | 1 chip | 4 chips / 4 chip | 8 chips / 8 chip |
|---|---|---|---|
| MiniMax-H3 text-to-video, lossy / 文生视频,有损 | 30.7 s ¹ | 63.7 s | 34.0 s |
| MiniMax-H3 text-to-video, lossless / 文生视频,无损 | 65.0 s ¹ | 208.4 s | 110.0 s |
| Qwen-Image-2.1, 20 steps, JAX / 20 步,JAX | 8.1 s ² | 1.34 s ³ | 0.74 s |
| Qwen-Image-2.1, 20 steps, torchax / 20 步,torchax | does not fit / 放不下 | 1.66 s | 1.28 s |
| Qwen-Image-2.1, 20 steps, torch_tpu / 20 步,torch_tpu | 8.8 s | does not fit / 放不下 | 1.18 s |
| MiniMax-H3 reference-to-video, lossy / 参考图生视频,有损 | — | 92.7 s | 51.2 s |
| Pika-H3-Flex text-to-video / 文生视频 | 55 s ⁴ | 70.8 s | 36.5 s |

¹ 480x832, 121 frames, `--chips 1 --offload`. ¹ 480x832,121 帧,`--chips 1 --offload`。

² With `--text-encoder host`; the prompt is encoded once on the host (4.3 s) and not counted. ² 使用 `--text-encoder host`;提示词在主机上编码一次(4.3 秒),不计入。

³ With `DT_QWEN_ULYSSES=0`. ³ 使用 `DT_QWEN_ULYSSES=0`。

⁴ On a **TPU v5e** chip (16 GB), 480x832, 141 frames, with `--int8-weights --adaln-host`. ⁴ 在 **TPU v5e** chip(16 GB)上,480x832,141 帧,使用 `--int8-weights --adaln-host`。

MiniMax-H3 on 4 chips needs `--offload` (text-to-video 69.8 s lossy, 208.4 s lossless; reference-to-video 98.7 s lossy, 267.3 s lossless), or `--adaln-host --text-offload` (63.7 s and 92.7 s lossy).
MiniMax-H3 在 4 chip 上需要 `--offload`(文生视频有损 69.8 秒、无损 208.4 秒;参考图生视频有损 98.7 秒、无损 267.3 秒),或者 `--adaln-host --text-offload`(有损 63.7 秒和 92.7 秒)。

The 4-chip and 8-chip MiniMax-H3 and Pika-H3-Flex numbers use 768x1344, 345 frames.
4 chip 和 8 chip 的 MiniMax-H3、Pika-H3-Flex 数字都是 768x1344、345 帧。

### One TPU v5e chip / 单个 TPU v5e chip

Measured 2026-11-04 on a v5litepod-1 (16 GB HBM, 47 GB host memory), Pika-H3-Flex, six steps.
测于 2026-11-04,v5litepod-1(16 GB HBM,47 GB 主机内存),Pika-H3-Flex 六步。

| per request / 每个请求 | 480x832, 141 frames / 141 帧 | 480x832, 362 frames / 362 帧 | 768x1344, 345 frames / 345 帧 |
|---|---|---|---|
| text-to-video / 文生视频 | 55 s | 272 s (251 s ¹) | 748 s |
| reference-to-video / 参考图生视频 | 121 s | 314 s (288 s ¹) | 1005 s |

¹ With `DT_INT8_ACT=1`. ¹ 使用 `DT_INT8_ACT=1`。

Run with `--chips 1 --offload --int8-weights --adaln-host`: the DiT is held in host memory as int8 (19.6 GB) and streamed in.
运行时加上 `--chips 1 --offload --int8-weights --adaln-host`:DiT 以 int8(19.6 GB)放在主机内存,流式送入 chip。

`--int8-weights` halves the weights and changes the output a little; `--adaln-host` precomputes the timestep modulation, so 26 GB of weights never reach the chip, and does not change the output.
`--int8-weights` 把权重减半,结果略有变化;`--adaln-host` 预先算好时间步调制,于是有 26 GB 权重根本不用上 chip,不改变结果。

Set `DT_WEIGHT_CACHE=<dir>` to keep the converted weights on disk: loading drops from 375 s to 178 s on the next start, and the transformer and text-encoder checkpoint shards are no longer read.
设置 `DT_WEIGHT_CACHE=<目录>` 把转换好的权重存在磁盘上:下次启动时加载从 375 秒降到 178 秒,而且不再读取 transformer 和文本编码器的原始权重分片。

Add `--cache-only` to build the cache and exit. It also runs without a TPU (`JAX_PLATFORMS=cpu`, about 95 GB of memory), and the files are identical to the ones a TPU run writes, so the conversion can happen on any big-memory machine and the cache be copied over.
加上 `--cache-only` 只构建缓存然后退出。它也能在没有 TPU 的机器上运行(`JAX_PLATFORMS=cpu`,约需 95 GB 内存),生成的文件与 TPU 上写出的逐字节相同,所以可以在任意大内存机器上转换,再把缓存拷过去。

On one chip the audio is decoded on the host, in the background while the chip decodes the video, so it adds no time.
单 chip 时音频在主机上解码,与 chip 解码视频同时进行,不增加时间。

768x1344, 345 frames needs `DT_FF_CHUNK=4096 --block-q 1024 --block-kv 2048` for text-to-video, and `DT_FF_CHUNK=4096 DT_QKV_CHUNK=16384 --block-q 512 --block-kv 1024` for reference-to-video.
768x1344、345 帧的文生视频需要 `DT_FF_CHUNK=4096 --block-q 1024 --block-kv 2048`,参考图生视频需要 `DT_FF_CHUNK=4096 DT_QKV_CHUNK=16384 --block-q 512 --block-kv 1024`。

`DT_INT8_ACT=1` also quantizes activations and uses the int8 matrix units: denoising is 10-22% faster, and the output changes more.
`DT_INT8_ACT=1` 会把激活也量化,使用 int8 矩阵单元:去噪快 10-22%,结果的变化也更大。

### TPU v7x — 1, 2, 4 and 8 devices / TPU v7x(1、2、4、8 个 device)

Measured 2026-11-04 on TPU v7x, JAX backend with `--fp8 --qk-fp8 --adaln-host`. A v7x chip has two TensorCores, and JAX sees each as one device with 95 GB of HBM; "8 devices" is a v7x-8, which is 4 chips. With 95 GB per device, the full 35B DiT and 768x1344, 345-frame sequence stay resident in HBM on 1, 2, 4 and 8 devices without `--offload`.
测于 2026-11-04,TPU v7x,JAX 后端配合 `--fp8 --qk-fp8 --adaln-host`。一个 v7x chip 有两个 TensorCore,JAX 把每个视为一个 device,各有 95 GB HBM;"8 个 device"即 v7x-8,也就是 4 个 chip。由于每个 device 有 95 GB,完整的 35B DiT 与 768x1344、345 帧序列在 1、2、4、8 个 device 上均可常驻 HBM,无需 `--offload`。

#### Pika-H3-Flex (6 steps / 6 步)

Measured 2026-11-06 with audio, on the code of this commit. 8 and 4 devices: steady state (`--runs 2`, second request); 1 and 2 devices: a single run (`--runs 1`).
测于 2026-11-06,含音频,即本提交的代码。8 和 4 个 device:稳态(`--runs 2` 的第二次请求);1 和 2 个 device:单次运行(`--runs 1`)。

| | 1 device | 2 devices / 2 个 device | 4 devices / 4 个 device | 8 devices (v7x-8) / 8 个 device |
|---|---|---|---|---|
| text-to-video, 768x1344, 345 frames / 文生视频,768p 345 帧 | **192.7 s** (179.5 + 12.1) | **101.4 s** (91.1 + 9.2) | **56.6 s** (49.5 + 6.7) | **29.3 s** (25.2 + 3.6) |
| reference-to-video, 768x1344, 345 frames / 参考图生视频,768p 345 帧 | — | — | — | **33.1 s** (29.0 + 3.6) |
| text-to-video, 480x832, 362 frames (15 s) / 文生视频,480p 362 帧 | — | — | — | **8.4 s** (6.1 + 2.0) |
| text-to-video, 480x832, 141 frames (6 s) / 文生视频,480p 141 帧 | **12.4 s** (9.4 + 2.1) | — | — | **3.1 s** (2.2 + 0.8) |

Parentheses show `(denoise + video decode)` in seconds. On a v7x-8, a 15-second 480p clip with sound takes 8.4 s.
括号内为 `(去噪 + 视频解码)` 秒数。在 v7x-8 上,一段 15 秒带声音的 480p 视频只需 8.4 秒。

#### MiniMax-H3 — 768x1344, 345 frames (14.4 s), 20 steps

Measured 2026-11-04, before the audio decode and the faster video decode; 8-device lossy text-to-video is now 32.0 s (27.9 + 3.6) steady state with audio.
测于 2026-11-04,早于音频解码和视频解码加速;8 个 device 有损文生视频现在为 32.0 秒(27.9 + 3.6,稳态,含音频)。

| | 2 devices / 2 个 device | 4 devices / 4 个 device | 8 devices (v7x-8) / 8 个 device |
|---|---|---|---|
| text-to-video, lossy / 文生视频,有损 ¹ | **111.1 s** (93.9 + 12.9) | **68.5 s** (54.0 + 9.7) | **44.8 s** (33.6 + 6.6) |
| text-to-video, lossless / 文生视频,无损 | **305.8 s** (289.0 + 12.8) | **170.3 s** (156.6 + 9.6) | **90.3 s** (79.4 + 6.6) ³ |
| reference-to-video, lossy / 参考图生视频,有损 ¹ | **137.4 s** (119.1 + 12.8) | **83.0 s** (67.6 + 9.7) | **52.6 s** (41.4 + 6.6) |
| reference-to-video, lossless / 参考图生视频,无损 | — | — | **101.9 s** (91.2 + 6.6) |

¹ Measured with `--sparse-window 2 --sparse-dense-steps 6 --cache-threshold 0.08` (11 of 19 DiT calls run). ¹ 使用 `--sparse-window 2 --sparse-dense-steps 6 --cache-threshold 0.08` 测得(19 次 DiT 调用中执行 11 次)。

² `--pallas-ffn` (a sequence-tiled Pallas SwiGLU FFN kernel) is opt-in. It was slower than XLA on 1 and 2 devices and on long sequences, and where it is faster (8 devices, 480x832, 141 frames: 0.2 s) the output drifts (DINO 0.943). ² `--pallas-ffn`(序列分块 Pallas SwiGLU FFN kernel)需要手动开启。它在 1、2 个 device 和长序列上比 XLA 慢;唯一更快的场景(8 个 device、480x832、141 帧,快 0.2 秒)输出会漂移(DINO 0.943)。

³ On a v7x-8 (8 devices), `torchax` lossless text-to-video takes 128.5 s. ³ 在 v7x-8(8 个 device)上,`torchax` 无损文生视频为 128.5 秒。

---

## Setup / 安装

Use Python 3.12 for the JAX and torchax backends.
JAX 和 torchax 后端请使用 Python 3.12。

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements/jax.txt && pip install -e .
```

Download the model weights into the Hugging Face cache (set `HF_HOME` to choose where).
把模型权重下载到 Hugging Face 缓存(用 `HF_HOME` 指定位置)。

```bash
hf download MiniMaxAI/MiniMax-H3
hf download Qwen/Qwen-Image-2.1
```

The torch_tpu backend needs its own environment with Python 3.14: `requirements/torchtpu.txt`.
torch_tpu 后端需要单独的 Python 3.14 环境:`requirements/torchtpu.txt`。

---

## Run MiniMax-H3 / 运行 MiniMax-H3

Generate a video from text with the JAX backend.
用 JAX 后端,从文字生成视频。

```bash
bash scripts/run_h3_jax.sh t2va --prompt "a red fox trotting through snow"
```

Add `--lossy` for the faster mode, and `--num-frames 121` for a shorter clip.
加上 `--lossy` 使用更快的模式,加上 `--num-frames 121` 生成更短的片段。

```bash
bash scripts/run_h3_jax.sh t2va --lossy --num-frames 121
```

Generate a video from a reference image.
从一张参考图生成视频。

```bash
bash scripts/run_h3_jax.sh ref2va --reference ref.png --prompt "..." --lossy
```

Serve many requests with one loaded model; each line of the file is one request.
用一个加载好的模型服务多个请求;文件里每一行是一个请求。

```bash
# requests.jsonl
#   {"tag": "a", "prompt": "...", "lossy": true}
#   {"tag": "b", "prompt": "...", "reference": "ref.png", "num_frames": 241}
bash scripts/run_h3_jax_server.sh requests.jsonl --out out
```

The other two backends have their own scripts.
另外两种后端有各自的脚本。

```bash
bash scripts/run_h3_torchax.sh
bash scripts/run_h3_torchtpu.sh t2va
```

Frames are saved to `out/<tag>_frames.npy` and the soundtrack to `out/<tag>_audio.npy` (32 kHz stereo; `--no-audio` skips it); this turns them into mp4 files with sound.
帧保存在 `out/<tag>_frames.npy`,音轨保存在 `out/<tag>_audio.npy`(32 kHz 立体声;`--no-audio` 可跳过);下面的命令把它们转成带声音的 mp4。

```bash
python scripts/frames_to_mp4.py out out/*_frames.npy
```

---

## Run Qwen-Image-2.1 / 运行 Qwen-Image-2.1

Pick a backend and give a prompt; the image is written to a PNG file.
选一种后端并给出提示词;图像会写入 PNG 文件。

```bash
bash scripts/run_qwen.sh jax     --prompt "a red fox in the snow, photorealistic"
bash scripts/run_qwen.sh jax     --prompt "..." --num_inference_steps 4
bash scripts/run_qwen.sh torchax --prompt "..."
bash scripts/run_qwen.sh torchtpu --prompt "..."
```

---

## Run Pika-H3-Flex / 运行 Pika-H3-Flex

Download the two adapters; the base model is the MiniMax-H3 you already have. The command prints the folder it downloaded to.
下载两个适配器;底座就是已经下载好的 MiniMax-H3。这条命令会输出下载到的文件夹。

```bash
PIKA=$(hf download Pika-Labs/Pika-H3-Flex --include "fl2va/*" --include "ref2va/*" --quiet)
```

Point `--pika-adapter` at the `fl2va` folder for text-to-video, or the `ref2va` folder for reference-to-video.
文生视频时把 `--pika-adapter` 指向 `fl2va` 文件夹,参考图生视频时指向 `ref2va` 文件夹。

```bash
bash scripts/run_h3_jax.sh t2va --pika-adapter $PIKA/fl2va --prompt "..."
bash scripts/run_h3_jax.sh ref2va --pika-adapter $PIKA/ref2va --reference character.png --prompt "..."
```

The resident server takes the adapters too, one per workflow; each request line can set `sparse_dense_steps`.
常驻服务也能加载适配器,每种工作流一个;每行请求可以设置 `sparse_dense_steps`。

```bash
bash scripts/run_h3_jax_server.sh requests.jsonl --pika-fl2va $PIKA/fl2va --pika-ref2va $PIKA/ref2va
```

The torchax and torch_tpu backends take the same flag; for torchax, turn its default lossy options off for the six-step ladder.
torchax 和 torch_tpu 后端用同一个参数;torchax 六步采样时要关掉它默认的有损选项。

```bash
bash scripts/run_h3_torchax.sh --pika-adapter $PIKA/fl2va --cache-threshold 0 --sparse-window 0 --prompt "..."
bash scripts/run_h3_torchtpu.sh t2va --pika-adapter $PIKA/fl2va --prompt "..."
```

| Pika-H3-Flex, 768x1344, 345 frames | JAX | torchax | torch_tpu |
|---|---|---|---|
| text-to-video / 文生视频 | **36.5 s** | 47.3 s | 71.1 s |
| reference-to-video / 参考图生视频 | **51.1 s** | 67.0 s | 79.4 s |

The adapter is merged into the weights while loading, so the denoising step costs the same as the base model.
适配器在加载时就合并进权重,所以每一步去噪的开销和底座模型一样。

The adapters are released under the MiniMax H3 Community License by Pika Labs.
这两个适配器由 Pika Labs 以 MiniMax H3 社区许可证发布。

---

## Good to know / 使用须知

The first run compiles for a few minutes; compiled programs are cached under `~/.cache/h3/`, so later runs start much faster.
第一次运行要编译几分钟;编译结果缓存在 `~/.cache/h3/`,之后启动会快很多。

`--fp8` and `--qk-fp8` use float8 math, which helps on TPU v7x but is slower on v6e.
`--fp8` 和 `--qk-fp8` 使用 float8 计算,在 TPU v7x 上有帮助,在 v6e 上反而更慢。

`--pallas-ffn` enables the sequence-tiled Pallas SwiGLU FFN kernel (`diffusers_tpu/kernels/pallas_ffn.py`) on TPU v7x. It is opt-in: it was slower than XLA on 1 and 2 devices and on long sequences, and where it was faster (8 devices, 480x832, 141 frames: 0.2 s) the output drifts (DINO 0.943), because each device scales its FP8 activations separately.
`--pallas-ffn` 在 TPU v7x 上启用序列分块 Pallas SwiGLU FFN kernel(`diffusers_tpu/kernels/pallas_ffn.py`),需要手动开启:它在 1、2 个 device 和长序列上比 XLA 慢;唯一更快的场景(8 个 device、480x832、141 帧,快 0.2 秒)输出会漂移(DINO 0.943),因为每个 device 各自计算 FP8 激活的缩放因子。

A single v6e chip can run MiniMax-H3 at 480p with `--chips 1 --offload`, streaming weights from host memory.
单个 v6e chip 可以用 `--chips 1 --offload` 运行 480p 的 MiniMax-H3,权重从主机内存流式加载。

A single TPU v5e chip also runs MiniMax-H3 end to end; see "One TPU v5e chip" under Results.
单个 TPU v5e chip 也能端到端运行 MiniMax-H3;见"性能"一节里的"单个 TPU v5e chip"。

Every entry point lists all of its options with `--help`.
每个入口都可以用 `--help` 查看全部选项。

---

## Adding a model / 添加新模型

The attention kernels (`diffusers_tpu/kernels`) and the torch_tpu integration (`diffusers_tpu/backends`) do not depend on any model.
注意力 kernel(`diffusers_tpu/kernels`)和 torch_tpu 集成(`diffusers_tpu/backends`)不依赖任何具体模型。

Each model lives in its own folder under `diffusers_tpu/models/`.
每个模型放在 `diffusers_tpu/models/` 下自己的文件夹里。

[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) explains how to bring up another diffusers model.
[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) 介绍了如何接入另一个 diffusers 模型。

---

## License / 许可证

The code is released under Apache-2.0.
代码以 Apache-2.0 许可证发布。

The model weights follow their own licenses on the Hugging Face hub.
模型权重遵循它们在 Hugging Face 上各自的许可证。
