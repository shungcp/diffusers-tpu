# diffusers-tpu

Fast inference for open diffusion models on Cloud TPU, built on top of [huggingface/diffusers](https://github.com/huggingface/diffusers).
基于 [huggingface/diffusers](https://github.com/huggingface/diffusers),在 Cloud TPU 上高速运行开源扩散模型。

Two models are supported today: **MiniMax-H3** (video + audio) and **Qwen-Image-2.1** (image).
目前支持两个模型:**MiniMax-H3**(视频 + 音频)和 **Qwen-Image-2.1**(图像)。

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

- **2026-11-03** — First release: MiniMax-H3 and Qwen-Image-2.1, each on JAX, torchax and torch_tpu.
  **2026-11-03** — 首次发布:MiniMax-H3 和 Qwen-Image-2.1,各支持 JAX、torchax、torch_tpu 三种后端。

---

## Results / 性能

Measured 2026-11-02 to 2026-11-03 on a TPU v6e-8 (8 chips); warm requests, with the model already loaded and compiled.
测于 2026-11-02 至 2026-11-03,TPU v6e-8(8 个 chip);均为热请求时间,模型已加载并编译完成。

Software: jax 0.12.0.1, libtpu 0.0.51, torch_tpu 0.2.0, diffusers `cc8644b` (see `requirements/`).
软件版本:jax 0.12.0.1、libtpu 0.0.51、torch_tpu 0.2.0、diffusers `cc8644b`(见 `requirements/`)。

### MiniMax-H3 — 768x1344, 345 frames (14.4 s), 20 steps

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| text-to-video, lossless / 文生视频,无损 | **110.0 s** | 117.8 s | 139.0 s |
| text-to-video, lossy / 文生视频,有损 | **36.9 s** | 42.9 s | 52.5 s |
| reference-to-video, lossless / 参考图生视频,无损 | **142.6 s** | 186.7 s | 189 s |
| reference-to-video, lossy / 参考图生视频,有损 | **55.1 s** | 77.3 s | 86 s |

"Lossy" skips a DiT call while the timestep embedding barely changes, and lets each video frame attend only to nearby frames.
"有损"是指:时间步嵌入几乎不变时跳过一次 DiT 计算,并且让每一帧只关注相邻的帧。

The visual quality stays close to lossless; check it on your own prompts before relying on it.
画质与无损版本接近;正式使用前,请先用自己的提示词检查效果。

The resident server keeps the model on the chips between requests: 36.4 s (text) and 54.8 s (reference) per lossy request.
常驻服务在请求之间把模型保留在 chip 上:每个有损请求 36.4 秒(文生)和 54.8 秒(参考图生)。

### Qwen-Image-2.1 — 1024x1024

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| 20 steps / 20 步 | **0.74 s** | 1.28 s | 1.18 s |
| 4 steps / 4 步 | **0.28 s** | 0.49 s | 0.55 s |

All three backends use all 8 chips for Qwen-Image.
Qwen-Image 的三种后端都使用全部 8 个 chip。

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
huggingface-cli download MiniMaxAI/MiniMax-H3
huggingface-cli download Qwen/Qwen-Image-2.1
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

Frames are saved to `out/<tag>_frames.npy`; this turns them into mp4 files.
帧保存在 `out/<tag>_frames.npy`;下面的命令把它们转成 mp4。

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

## Good to know / 使用须知

The first run compiles for a few minutes; compiled programs are cached under `~/.cache/h3/`, so later runs start much faster.
第一次运行要编译几分钟;编译结果缓存在 `~/.cache/h3/`,之后启动会快很多。

`--fp8` and `--qk-fp8` use float8 math, which helps on TPU v7x but is slower on v6e.
`--fp8` 和 `--qk-fp8` 使用 float8 计算,在 TPU v7x 上有帮助,在 v6e 上反而更慢。

A single v6e chip can run MiniMax-H3 at 480p with `--chips 1 --offload`, streaming weights from host memory.
单个 v6e chip 可以用 `--chips 1 --offload` 运行 480p 的 MiniMax-H3,权重从主机内存流式加载。

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
