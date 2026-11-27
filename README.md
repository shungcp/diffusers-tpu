# diffusers-tpu

Fast inference for open diffusion models on Cloud TPU, built on top of [huggingface/diffusers](https://github.com/huggingface/diffusers).

基于 [huggingface/diffusers](https://github.com/huggingface/diffusers),在 Cloud TPU 上高速运行开源扩散模型。

Nine models are supported today: **MiniMax-H3** (video + audio), **MiniMax-H3-Image** (image and image editing), **Qwen-Image-2.1** (image), **MOSS-Image** (image, torchax backend), **MOSS-SoundEffect-v2** (text-to-audio, JAX), **MOVA** (image + text to video with audio, JAX), **OmniVAE T2AV** (text to video with audio, JAX), **VideoGym-LTX-2.3** (image + text to video, LTX-2.3, JAX) and **MAGI-2-preview** (113.93B Multi-Head Latent MoE video + audio + 1080p refiner, JAX / torchax / torch_tpu).

目前支持九个模型:**MiniMax-H3**(视频 + 音频)、**MiniMax-H3-Image**(图像与图像编辑)、**Qwen-Image-2.1**(图像)、**MOSS-Image**(图像,torchax 后端)、**MOSS-SoundEffect-v2**(文生音效,JAX)、**MOVA**(图 + 文生成带同步音频的视频,JAX)、**OmniVAE T2AV**(文生带同步音频的视频,JAX)、**VideoGym-LTX-2.3**(图 + 文生视频,LTX-2.3,JAX)和 **MAGI-2-preview**(113.93B 多头潜空间 MoE 音视频生成 + 1080p 超分,JAX / torchax / torch_tpu 三后端)。

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

- **2026-11-27** — **VideoGym-LTX-2.3 on TPU `v7x-8`** ([usage](#run-videogym-ltx-23--运行-videogym-ltx-23), [v7x ledger](docs/tpu_v7x.md#9-videogym-ltx-23-22b-audio-video-dit-on-tpu-v7x-8--videogym-ltx-23-在-v7x-8-上的优化台账)): auto `--resident 48` (all 44 GB DiT weights resident per 95 GiB device, eliminating per-block weight all-gathers) + auto 2D **`4×2` (`cfg=4, sp=2`) CFG-parallel** mesh (keeping all 240 Ulysses/v2a collectives 100% intra-chip across each physical chip's 2 cores and cutting redundant audio & text K/V projection compute by `4×`, `95.8%` compute utilization) + **on-TPU `uint8` video decode** (`4×` smaller PCIe D2H, cutting decode `1.08–1.40 s -> 0.49 s`: video `0.255 s` / `506 FPS` + audio `0.230 s`). Full 30-step `768×512×129 @ 24 fps` (`5.33 s` clip) E2E drops from `15.03 s` (`v6e-8`) to **`6.60 s` lossless BF16** (`0.202 s/step`, `44.2 GiB` peak, `PSNR 45.51 dB` vs `1×8`) and **`5.82 s` with `--fp8`** (on-TPU W8A8 E4M3 quantization in `44.3 s` load, `0.176 s/step`, `27.0 GiB` peak, `PSNR 40.00 dB`, warm cache `run 0 = 29.64 s`, `2.58×` vs `v6e-8`). Also verified `c947` on `v7x-8`: `fuse_all_gather=true` is unsupported (`v5e`-only), while default `--resident 20` already achieves `92.3%` compute overlap (`296.3 ms/fwd`) and manual packed prefetch is slower (`413.9 ms/fwd`).

  **2026-11-27** — **VideoGym-LTX-2.3 TPU `v7x-8` 优化**([用法](#run-videogym-ltx-23--运行-videogym-ltx-23),[优化台账](docs/tpu_v7x.md#9-videogym-ltx-23-22b-audio-video-dit-on-tpu-v7x-8--videogym-ltx-23-在-v7x-8-上的优化台账)):自动识别 `v7x` 95 GiB 显存默认启用 `--resident 48`(全 48 层常驻、消除权重 all-gather)+ 自动 2D **`4×2`(`cfg=4, sp=2`)CFG 并行**(将 48 层全部 240 次 Ulysses/v2a 通信留在单颗物理芯片 2 核内实现零跨芯 ICI,同时将音频流与文本 K/V 投影计算降为 `1/4`,计算利用率达 `95.8%`)+ **TPU 端融合 `uint8` 视频解码**(PCIe 回传缩 4 倍,总解码 `1.08–1.40 s -> 0.49 s`:视频 `0.255 s`/`506 FPS` + 音频 `0.230 s`)。完整 30 步 `768×512×129 @ 24 fps`(`5.33 s` 音视频)端到端从 `v6e-8` 的 `15.03 s` 降至 **无损 BF16 `6.60 s`**(`0.202 s/步`,峰值 `44.2 GiB`,对比 `1×8` `PSNR 45.51 dB`)与 **`--fp8` `5.82 s`**(TPU 端 W8A8 E4M3 加载 `44.3 s`,`0.176 s/步`,峰值 `27.0 GiB`,`PSNR 40.00 dB`,热缓存首跑 `29.64 s`,较 `v6e-8` 提速 **`2.58×`**)。同时在 `v7x-8` 完成 `c947` 结论验证:`fuse_all_gather=true` 同样仅支持 `v5e`,而默认 `--resident 20` 在 `v7x` 底层已实现 `92.3%` 计算重叠(`296.3 ms/fwd`),手动打包预取反而更慢(`413.9 ms/fwd`)。
- **2026-11-26** — **MAGI-2-preview** (Sand.ai 113.93B / 5.95B active Multi-Head Latent MoE Stage 1 DiT + 4-stream mHC + Attention Sink, Qwen3.5-27B text encoder, Stage 2 1080p 6.72B Refiner DiT, TurboVAED / Wan2.2-TI2V-5B video VAE, and AutoencoderOobleck stereo audio VAE) on TPU v7x-8 across Native JAX, torchax, and torch_tpu ([usage](#run-magi-2-preview--运行-magi-2-preview), [v7x ledger](docs/tpu_v7x.md#8-magi-2-preview-11393b-multi-head-latent-moe--1080p-refiner-on-tpu-v7x-8--magi-2-preview-端到端优化台账)). Stage 1 runs Ulysses CP=8 + EP=8 (32 local experts/head/chip) with Pallas Splash+Sink (`V7_SPEC`), MegaCore Dual-TC `gmm_v2` (BF16 and W8A8 E4M3 FP8), and Tokamax Pallas 4-stream mHC. Verified on v7x-8 (10 s audio+video, CFG=2): **Native JAX** Stage 1 `272p BF16` **0.960 s/step**, `540p BF16` lossless **3.18–3.42 s/step** (`67.7 GiB` peak HBM; T2V / I2V / Ref2VA with synchronized English speech verified), `540p FP8` **3.430 s/step** (`42.7 GiB` HBM), and controllable **`--lossy`** (audio+video TeaCache + half-seq late-step `cond-only` CFG skip) 100-step `540p` I2V in **129.76 s** (**1.266 s/step** effective, **2.51×–2.61×** speedup; warm E2E **181.23 s**; PSNR `22.62 dB`, 50 ms speech envelope corr `0.9793`); **Stage 2 `1080p` Refiner** (`1088x1920 @ 25 fps`, 5 steps, 522,752 tokens) accelerated **8.27× lossless** from **87.35 -> 10.56 s/step** (**53.38 s** for 5 steps, `cos=0.999999` vs legacy BF16; opt-in `--refiner-qk-fp8` **10.15 s/step** / **8.61×**) via GQA 4-head fused (`bq=512`) + W-coalesced (`bkv=640`) + tail-Q (`bkv=1024`) Pallas kernels and `DP=2 × CP=4` CFG-parallel; **Stage 3a `TurboVAED`** 8-chip chunk-parallel decode cuts 540p warm `1.29 -> 0.72 s` (`1.80×`) and 1080p warm `11.97 -> 3.88 s` (`3.09×`, 100% bit-exact `max|d|=0`, `PSNR=∞`); **Stage 0a `Wan2.2` I2V encode** JIT `70.4 s -> 0.29 s` cache-hit (`22.7 ms` warm) and **Stage 0b `Qwen3.5-27B`** static negative-prompt disk cache `20.8 s -> 1.62 ms` (bit-exact); **Resident JAX Server** (`run_magi2_jax_server.sh`) keeps all 4 BF16 models in **41.80 GiB / 94.74 GiB** HBM/chip (`33.65 s` warm startup once, zero-reload across `--requests` JSONL via `bind_sample_lens`, warm `s3v=0.19 s` / `s3a=0.04 s`) plus offline PE `--enhance`; **torchax** matches Native JAX (`272p BF16` **0.960 s/step**, `540p BF16` **3.728 s/step**, `540p FP8` **3.745 s/step**); **torch_tpu** (`torchrun --nproc_per_node 8`, full E2E `--prompt --enhance --decode-mp4` + shared rank-0 `DT_AOT_PERSIST` `178 MiB` cutting cross-process compile `32.57 s -> 1.45 s` / **22.5×**) reaches **0.923 / 0.717–0.829 s/step** (`272p BF16 / FP8`) and **3.331 / 3.248 s/step** (`540p BF16 / FP8`).

  **2026-11-26** — TPU v7x-8 支持 **MAGI-2-preview**(Sand.ai 113.93B / 5.95B 激活多头潜空间 MoE 一阶段 DiT + 4 流 mHC + Attention Sink、Qwen3.5-27B 文本编码器、二阶段 1080p 6.72B Refiner DiT、TurboVAED / Wan2.2 视频 VAE 与 AutoencoderOobleck 立体声音频 VAE),覆盖原生 JAX、torchax、torch_tpu 三后端([用法](#run-magi-2-preview--运行-magi-2-preview),[完整台账](docs/tpu_v7x.md#8-magi-2-preview-11393b-multi-head-latent-moe--1080p-refiner-on-tpu-v7x-8--magi-2-preview-端到端优化台账))。v7x-8 实测(10 秒音视频,CFG=2):**原生 JAX** 一阶段 `272p BF16` **0.960 秒/步**、`540p BF16` 无损 **3.18–3.42 秒/步**(峰值 HBM `67.7 GiB`,已完成 T2V / I2V / Ref2VA 含同步人声对白端到端验证)、`540p FP8` **3.430 秒/步**(`42.7 GiB` HBM),开启可控 **`--lossy`**(音视频 TeaCache + 后期半序列 `cond-only` CFG 跳过)后 100 步 `540p` I2V 去噪仅 **129.76 秒**(等效 **1.266 秒/步**,加速 **2.51×–2.61×**,热启动端到端 **181.23 秒**,视频 PSNR `22.62 dB`,50ms 语音包络相关性 `0.9793`);**二阶段 `1080p` Refiner**(`1088x1920 @ 25 fps`,5 步,`522,752 tokens`)通过 GQA 4-Head 融合(`bq=512`)+ W 维 5 块合并连续 DMA(`bkv=640`)+ Tail-Q 超块(`bkv=1024`)自研 Pallas 核及 `DP=2 × CP=4` CFG 并行实现 **8.27 倍无损提速**,从 **87.35 降至 10.56 秒/步**(5 步总计 **53.38 秒**,`cos=0.999999`,可选 `--refiner-qk-fp8` **10.15 秒/步** / **8.61×**);**Stage 3a `TurboVAED`** 8 芯分片并行解码使 540p 热解码 `1.29 -> 0.72 秒`(`1.80×`)、1080p 热解码 `11.97 -> 3.88 秒`(`3.09×`,100% bit-exact `max|d|=0`,`PSNR=∞`);**Stage 0a `Wan2.2` I2V 编码** JIT 后 `70.4 -> 0.29 秒`(热 `22.7 ms`),**Stage 0b `Qwen3.5-27B`** 静态负向 prompt 磁盘缓存 `20.8 秒 -> 1.62 ms`(bit-exact);**原生 JAX 常驻服务**(`run_magi2_jax_server.sh`)四模型 BF16 常驻仅占 **41.80 GiB / 94.74 GiB** HBM/卡(热启动 **33.65 秒** 一次,多请求 `--requests` 零重载,复用编译后 `s3v=0.19 秒` / `s3a=0.04 秒`)并内置离线 PE `--enhance`;**torchax** `272p BF16` **0.960 秒/步**、`540p BF16 / FP8` **3.728 / 3.745 秒/步**;**torch_tpu**(`torchrun --nproc_per_node 8`,一键端到端 `--prompt --enhance --decode-mp4` + 共享 rank-0 `DT_AOT_PERSIST` `178 MiB` 使跨进程编译 `32.57 秒 -> 1.45 秒` / **22.5×**)达到 `272p BF16 / FP8` **0.923 / 0.717–0.829 秒/步**、`540p BF16 / FP8` **3.331 / 3.248 秒/步**。
- **2026-11-26** — **VideoGym-LTX-2.3** (OpenMOSS `OpenMOSS-Team/VideoGym-LTX-2.3`: LTX-2.3's 22B audio-video DiT post-trained for visual reasoning puzzles, with Gemma-3-12B, the text connectors, the LTX-2 video VAE, the audio VAE and vocoder): image + text to video at the model card's settings, 768x512 (the 512x512 puzzle centered on a black canvas), 129 frames at 24 fps, 30 steps, CFG 3 with spatio-temporal guidance and guidance rescale 0.7. On a v6e-8 a clip takes **15.0 s** end to end (0.47 s a step; 27.0 GiB peak a chip). The DiT, the connectors and the video decoder are native JAX, Gemma-3, the video encoder and the audio decoder run through torchax; in bfloat16 the pipeline's error against the official float32 run is that of the official bfloat16 run (latents 1.8e-2 vs 2.2e-2 after 3 steps, frames 46.6 vs 45.1 dB). The 6,528 video tokens are few and the 44 GB of weights many, so the blocks run sequence parallel instead of tensor parallel (which all-reduces the residual stream four times a block: 1.26 s a forward against 0.39 s): each chip keeps 1/8 of the rows, 20 blocks stay whole on every chip and the other 28 are all-gathered block by block, the self-attention goes through Ulysses all-to-alls and the video-to-audio attention merges per-chip partial softmaxes. Each step is one forward over four rows (negative, positive, positive with STG, positive with the audio/video attentions off): the last row is diffusers' default audio modality guidance (audio CFG 7, modality 3), which the audio, and through it the video, follow ([usage](#run-videogym-ltx-23--运行-videogym-ltx-23)).

  **2026-11-26** — 支持 **VideoGym-LTX-2.3**(OpenMOSS `OpenMOSS-Team/VideoGym-LTX-2.3`:LTX-2.3 的 22B 音视频 DiT 在视觉推理谜题上做的后训练,配 Gemma-3-12B、文本 connectors、LTX-2 视频 VAE、音频 VAE 与 vocoder):按模型卡设置图 + 文生视频,768x512(512x512 谜题居中、两侧黑边),129 帧 24 fps,30 步,CFG 3 加时空引导(STG)与 guidance rescale 0.7。v6e-8 上一条端到端 **15.0 秒**(每步 0.47 秒;每 chip 峰值 27.0 GiB)。DiT、connectors 与视频解码器为原生 JAX,Gemma-3、视频编码器与音频解码器经 torchax 运行;bf16 下对官方 fp32 的误差与官方 bf16 同级(3 步后 latent 1.8e-2 对 2.2e-2,帧 46.6 对 45.1 dB)。视频只有 6528 个 token 而权重有 44 GB,所以各块用序列并行而不是张量并行(张量并行每块要对残差流做四次 all-reduce:每次前向 1.26 秒,序列并行 0.39 秒):每个 chip 持有 1/8 的行,20 个块在每个 chip 上整份常驻,其余 28 个逐块 all-gather,自注意力走 Ulysses all-to-all,视频到音频的注意力合并各 chip 的部分 softmax。每步一次前向、四行(负、正、正 + STG、正且关掉音视频互注意力):最后一行是 diffusers 默认的音频模态引导(音频 CFG 7、modality 3),音频以及通过它的视频都与之一致。
- **2026-11-26** — **MOVA-720p**: same architecture and configs as MOVA-360p, only the weights differ, so it runs with `--repo OpenMOSS-Team/MOVA-720p`. At the official 1280x720, 193 frames (8 s), 50 steps, CFG 5, a clip takes **788 s** end to end on a v6e-8 (15.7 s a step, 25.0 GiB peak a chip). Two CFG groups of four chips do not fit a v6e chip at 720p (both experts on four chips plus the activations), so above 480x832 on chips under 64 GB the CFG branches share all chips ([usage](#run-mova--运行-mova)).

  **2026-11-26** — 支持 **MOVA-720p**:架构和配置与 MOVA-360p 完全相同,只是权重不同,用 `--repo OpenMOSS-Team/MOVA-720p` 运行。官方 1280x720、193 帧(8 秒)、50 步、CFG 5,v6e-8 上一条端到端 **788 秒**(每步 15.7 秒,每 chip 峰值 25.0 GiB)。720p 下 CFG 两组各 4 chip 放不进 v6e(两个专家分到 4 个 chip 再加激活),所以分辨率高于 480x832 且 chip 显存小于 64 GB 时,CFG 两支共用全部 chip。
- **2026-11-25** — **OmniVAE T2AV** (OpenMOSS `OpenMOSS-Team/OmniVAE`, final configuration `t2av_recon_distill_avclip`: two 1B Z-Image towers for video and audio joined by 18 cross-attention bridges, Qwen3.5-0.8B, decoded by OmniVAE's Wan2.2-style video decoder and its DAC audio decoder): 256x256, 121 frames (5 s) with 48 kHz audio, 50 steps, the release evaluation's dual CFG (3 forwards a step, scales 4). On a v6e-4 a clip takes **4.90 s** end to end (95 ms a step; 17.6 s on one chip), the simple CFG of the official CLI 61 ms a step; on a v6e-8 **3.73 s** (72 ms a step; simple CFG 49 ms). The model is small (2.6B) and has 15 heads, so every chip keeps all weights and the video rows are split over the chips: each chip holds its quarter of the video tokens plus a copy of the caption, keys are gathered for self-attention, and video-to-audio bridge attention merges per-chip partial softmaxes. In float32 the pipeline matches the official code step by step (latents within 6e-6 after 4 steps, both CFG modes; video decoder 3e-7, audio 122 dB); in bfloat16 its error is that of the official bfloat16 run. RoPE runs rotate-half on q / k columns regrouped per head (exact). Two differences from the official code's intent, both following what it actually runs: the scheduler shift is 3 for video and audio (the code's 5 / 3 never reaches the scheduler), and the prompt template is drawn with the seed, without the duration suffix. The Wan2.2 decoder (`models/wan_vae/jax_wan22_vae.py`) is shared ([usage](#run-omnivae--运行-omnivae)).

  **2026-11-25** — 支持 **OmniVAE T2AV**(OpenMOSS `OpenMOSS-Team/OmniVAE`,最终配置 `t2av_recon_distill_avclip`:视频、音频两个 1B Z-Image 塔,由 18 个交叉注意力桥连接,Qwen3.5-0.8B 文本编码器,用 OmniVAE 的 Wan2.2 式视频解码器和 DAC 音频解码器解码):256x256、121 帧(5 秒)加 48 kHz 音频,50 步,发布评测用的 dual CFG(每步 3 次前向,各尺度 4)。v6e-4 上一条端到端 **4.90 秒**(每步 95 毫秒;单 chip 17.6 秒),官方 CLI 的 simple CFG 每步 61 毫秒;v6e-8 上 **3.73 秒**(每步 72 毫秒;simple CFG 49 毫秒)。模型小(2.6B)且只有 15 个头,所以每个 chip 持有整份权重,视频行切分到各 chip:每个 chip 放自己那 1/4 视频 token 和一份 caption,自注意力 gather key,视频到音频的桥接注意力合并各 chip 的部分 softmax。fp32 下整条流程与官方逐步一致(两种 CFG 4 步后 latent 误差 ≤ 6e-6;视频解码器 3e-7,音频 122 dB);bf16 下误差与官方 bf16 同级。RoPE 改为 rotate-half(按头重排 q / k 列,精确等价)。与官方代码"意图"不同、但与其实际运行一致的两处:调度器 shift 视频、音频都是 3(代码里的 5 / 3 没有真正设到调度器上);提示词模板按种子选取,不加时长后缀。Wan2.2 解码器(`models/wan_vae/jax_wan22_vae.py`)为公共模块。
- **2026-11-24** — **MOVA-360p** (OpenMOSS image + text to video with synchronized audio: two 14B Wan2.2-style video experts, a 1.3B audio tower, a cross-modal bridge, UMT5-XXL, the Wan2.1 VAE and DAC): 352x640, 193 frames (8 s), 50 steps, CFG 5. On a v6e-4 a clip takes **164.1 s** end to end (3.25 s a step; both experts resident, 24.1 GiB peak a chip; the video layers run sequence parallel: whole weights all-gathered layer by layer, Ulysses all-to-all around the self-attention, instead of three all-reduces of the residual stream a layer under tensor parallelism, 4.22 s a step); on a v6e-8, with the CFG branches on two groups of four chips, **91.2 s** (1.79 s a step) against 9.0 s a step on an H100 with offload (official README). Everything but the text encoder is native JAX (the UMT5 is transformers' own through torchax); in float32 the pipeline matches the official code (frames within 2/255, audio SNR 45.8 dB after 4 steps), in bfloat16 its error is that of the official bfloat16 run. The Wan2.1 VAE (`models/wan_vae/`) is shared and can run chunked in time with the official cache semantics ([usage](#run-mova--运行-mova)).

  **2026-11-24** — 支持 **MOVA-360p**(OpenMOSS 图 + 文生成带同步音频的视频:两个 14B Wan2.2 式视频专家、1.3B 音频塔、跨模态桥接、UMT5-XXL、Wan2.1 VAE、DAC):352x640、193 帧(8 秒)、50 步、CFG 5。v6e-4 上一条端到端 **164.1 秒**(每步 3.25 秒;两个专家常驻、每 chip 峰值 24.1 GiB;视频层序列并行:权重逐层 all-gather 成整份,自注意力前后各一次 Ulysses all-to-all,代替张量并行下每层三次残差流 all-reduce(每步 4.22 秒));v6e-8 上 CFG 两支各占 4 chip,**91.2 秒**(每步 1.79 秒),官方 README 中 H100(offload)每步 9.0 秒。除文本编码器外全部为原生 JAX(UMT5 用 transformers 自带实现经 torchax);fp32 下与官方代码一致(4 步后帧差 ≤ 2/255、音频 SNR 45.8 dB),bf16 下误差与官方 bf16 同级。Wan2.1 VAE(`models/wan_vae/`)为公共模块,可按官方缓存语义分时间块运行。
- **2026-11-25** — **Agnes-Video-H3-MoE** on the JAX backend (`--moe <snapshot>`): DiT blocks 15–34 of MiniMax-H3 use its shared + 8-expert top-1 feed-forward (experts copied from the dense FFN, rank-64 LoRA merged at load; the other 30 blocks stay dense). Checked against the released algorithm on one layer (fp32 on the CPU, relative error 6e-7). The real routing is very skewed (in most layers one expert takes 55–80% of the tokens), so the layer is split so that skew does not matter: on a v6e-8 every expert's width over the 8 chips, with megablox `gmm_v2` and a hand-written ring reduce-scatter overlapping the next block's compute; on a v7x-8 the experts replicated on every device (DP-8, no communication). 480x832, 124 frames, 50 steps, per request resident: v6e-8 **26.4 s** (dense MiniMax-H3 27.5 s), v7x-8 **17.5 s** (dense 19.8 s); 768x1344x345, 20 steps: v6e-8 110.2 s (dense 104.3 s), v7x-8 **79.2 s** (dense 80.0 s). The weights are Agnes-AI's, under the MiniMax H3 Community License; this repository only loads them. The v7x-8 work was done on the v7x-8.

  **2026-11-25** — JAX 后端支持 **Agnes-Video-H3-MoE**(`--moe <快照>`):MiniMax-H3 DiT 第 15–34 块换成它的共享 + 8 专家 top-1 前馈(专家从稠密 FFN 拷出,rank-64 LoRA 在加载时合并;其余 30 块仍是稠密)。单层与发布算法对照(CPU fp32,相对误差 6e-7)。真实路由极不均(多数层一个专家拿走 55–80% 的 token),所以切分方式选成不受偏斜影响:v6e-8 上每个专家的宽度切到 8 个 chip,用 megablox `gmm_v2`,跨 chip 求和写成与下一块计算重叠的环形 reduce-scatter;v7x-8 上专家在每个 device 各放一份(DP-8,无通信)。480x832、124 帧、50 步、常驻每请求:v6e-8 **26.4 秒**(稠密 MiniMax-H3 27.5 秒),v7x-8 **17.5 秒**(稠密 19.8 秒);768x1344x345、20 步:v6e-8 110.2 秒(稠密 104.3 秒),v7x-8 **79.2 秒**(稠密 80.0 秒)。权重属于 Agnes-AI,按 MiniMax H3 Community License 发布;本仓库只负责加载。v7x-8 部分由 v7x-8 那边完成并实测。
- **2026-11-24** — torch_tpu defaults on a TPU v7x-8: the two-slab post-attention pipeline is on (`--no-post-attn-pipe` turns it off) and the runtime-length attention kernel uses kv block 4096 at every length. 768p steady state 93.0 -> **85.1 s** text-to-video lossless, 29.9 -> **26.9 s** lossy, 105.0 -> **96.2 s** reference-to-video lossless, 35.2 -> **31.9 s** lossy (DINO against the old default at or near the run-to-run noise floor). v6e-8 output is unchanged. The reference is still encoded in tiles, as released; `--no-vae-enc-tile` is faster on v7x but changes the latents. Measured on the v7x-8.

  **2026-11-24** — v7x-8 上 torch_tpu 的默认配置:默认开启两段注意力后流水(`--no-post-attn-pipe` 关闭),运行时长度注意力 kernel 在所有长度上用 kv 块 4096。768p 稳态:文生视频无损 93.0 -> **85.1 秒**、有损 29.9 -> **26.9 秒**,参考图生视频无损 105.0 -> **96.2 秒**、有损 35.2 -> **31.9 秒**(与旧默认的 DINO 在运行间噪声地板附近)。v6e-8 输出不变。参考图仍按发布方式分块编码;`--no-vae-enc-tile` 在 v7x 上更快,但会改变潜变量。数据在 v7x-8 上测得。
- **2026-11-24** — torch_tpu, three fixes, all with the same frames by default. (1) The Pika sampler read each step's sigma from the chip, so the host waited for all queued work before it could enqueue the next step; it now reads a host copy: Pika ref2va 480x832x121 on a v6e-8 9.2 -> **8.0 s** (text-to-video unchanged at 3.8 s). (2) With the text encoder resident (`--no-offload --shard-text`, v7x), a request in a new text bucket waited the full 60 s timeout for the background DiT compile to start: ref2va 480p 82.9 -> **25.9 s** on a v7x-8. (3) The 2-slab post-attention pipeline (`--post-attn-pipe`) now also runs in the text-bucket path: 768p text-to-video 95.8 -> **87.8 s** on a v7x-8 (off by default; no gain on a v6e-8). (2) and (3) were found and measured on the v7x-8.

  **2026-11-24** — torch_tpu 三处修复,默认配置输出逐位不变。(1) Pika 采样器每步从 chip 读 sigma,主机要等排队的计算全部完成才能发下一步;改读主机副本:v6e-8 上 Pika 参考图生视频 480x832x121 9.2 -> **8.0 秒**(文生视频不变,3.8 秒)。(2) 文本编码器常驻时(`--no-offload --shard-text`,v7x),换到新文本桶的请求要白等 60 秒超时,后台 DiT 编译才开始:v7x-8 上 ref2va 480p 82.9 -> **25.9 秒**。(3) 两段注意力后流水(`--post-attn-pipe`)现在也用于分桶路径:v7x-8 上 768p 文生视频 95.8 -> **87.8 秒**(默认关;v6e-8 上无收益)。(2)(3) 由 v7x-8 那边发现并实测。
- **2026-11-22** — **MOSS-SoundEffect-v2** (OpenMOSS text-to-audio: a 1.3B Wan-layout audio DiT, the DAC VAE and Qwen3-1.7B): 100 steps with CFG 4, 48 kHz. On a v6e-4 a 10 s clip takes **0.95 s** end to end (the CFG branches on two pairs of chips, each pair splitting the 1,500-row sequence: 9.5 ms a step), 1.2 s on 2 chips and 2.1 s on one; four one-chip replicas serve **0.58 s** a clip. The DiT and the DAC decoder are native JAX and the text encoder is the stock transformers Qwen3 through torchax, so none of the official code or its pins is needed; in float32 the whole pipeline matches the official code step by step (waveform SNR 95 dB after 10 steps, 89 dB after 100). The DAC decoder (`models/dac/`) is shared, for MOVA and OmniVAE next ([usage](#run-moss-soundeffect-v2--运行-moss-soundeffect-v2)).

  **2026-11-22** — 支持 **MOSS-SoundEffect-v2**(OpenMOSS 文生音效:1.3B Wan 结构音频 DiT、DAC VAE、Qwen3-1.7B):100 步、CFG 4、48 kHz。v6e-4 上 10 秒音频端到端 **0.95 秒**(CFG 两支各占两个 chip,每对 chip 切分 1500 行序列:每步 9.5 ms),2 chip 1.2 秒,单 chip 2.1 秒;四个单 chip 副本吞吐每条 **0.58 秒**。DiT 与 DAC 解码器为原生 JAX,文本编码器是 transformers 自带的 Qwen3 经 torchax 运行,不需要官方代码及其依赖版本;fp32 下整条流程与官方代码逐步一致(10 步后波形 SNR 95 dB,100 步后 89 dB)。DAC 解码器(`models/dac/`)为公共模块,接下来给 MOVA 和 OmniVAE 复用。
- **2026-11-22** — torch_tpu text-length buckets, on by default (`--text-bucket 256`; `0` turns it off and is bit-identical to before). A new prompt length used to recompile the DiT programs (27–34 s at 768p) and, in lossy runs, the sparse attention program (another 24 s). Text rows are now padded to a multiple of 256 and masked by the true length at run time, and the target video rows move to the front of the packed sequence, so the video starts at row 0 for every prompt: each block broadcasts one modulation row over the video (what `--seg-mod` did, which baked text-dependent rows into the program), and the sparse block table no longer depends on the text. 768x1344x345, 20 steps, v6e-8, a new prompt in a seen bucket: text-to-video **152.6 -> 128.3 s** lossless, **89.4 -> 42.4 s** lossy; reference-to-video **191.3 -> 166.9 s**, **103.2 -> 59.1 s**; Pika 480p text-to-video 33.5 -> **8.1 s**; a restarted process with saved programs serves a new prompt at steady-state speed (lossy 70.0 -> **41.6 s**). Steady state within 2%. Not bit-identical to the unbucketed path (shapes and summation order change); DINO near the semantics-free 1e-3 perturbation floor and judged visually equivalent over 18 prompt/seed pairs (6 prompts). Also: `--text-layers 51` now applies to the default text path (13 of 64 Qwen3-VL layers were streamed and run for nothing; bit-identical, 0.3–0.8 s a request), and `torchtpu_h3` writes the soundtrack (`*_audio.npy`) beside the frames.

  **2026-11-22** — torch_tpu 文本长度分桶,默认开启(`--text-bucket 256`;设为 `0` 关闭,与此前逐位一致)。此前提示词长度一变就要重编 DiT 程序(768p 27–34 秒),有损模式还要再编一次稀疏注意力程序(再 24 秒)。现在文本行补齐到 256 的倍数、运行时按真实长度屏蔽,并把目标视频行移到打包序列最前面,视频对任何提示词都从第 0 行开始:每个 block 对视频行广播一行调制参数(`--seg-mod` 的做法,但它把随文本变化的行段写死进程序),稀疏块表也不再随文本变化。768x1344x345、20 步、v6e-8,同一桶内的新提示词:文生视频无损 **152.6 -> 128.3 秒**、有损 **89.4 -> 42.4 秒**;参考图生视频 **191.3 -> 166.9 秒**、**103.2 -> 59.1 秒**;Pika 480p 文生视频 33.5 -> **8.1 秒**;进程重启后(程序已存盘)新提示词即为稳态速度(有损 70.0 -> **41.6 秒**)。稳态差异在 2% 以内。与不分桶不逐位一致(形状与求和顺序变化);DINO 接近无语义 1e-3 扰动的噪声地板,18 组提示词/种子(6 条提示词)肉眼对比画质相当。另外:`--text-layers 51` 现在对默认文本路径生效(Qwen3-VL 64 层里 13 层此前白白流式拷贝和计算;逐位一致,每请求 0.3–0.8 秒);`torchtpu_h3` 同时写出音轨(`*_audio.npy`)。
- **2026-11-21** — torch_tpu per-request time (steady state, not compilation). Reference-to-video no longer copies weights that never change back to the host after every use: the 9.7 GiB VAE and the parked text-encoder modules swap back to a kept host copy. The text encoder's layers are streamed as 1/8 per chip and all-gathered over ICI, instead of every chip pulling each whole layer through the shared host link. The last reference images and prompt conditionings are cached (`DT_REF_CACHE`, `DT_TEXT_CACHE`, 8 each), so a request repeating a reference or a prompt (e.g. a new seed) skips those encoders. 768x1344x345, 20 steps, v6e-8: reference-to-video 167.2 -> **161.8 s** lossless and 66.3 -> **60.9 s** lossy for a new prompt and reference, **151.8 s** / **50.7 s** when both repeat; the torch_tpu column above now also includes the changes since 2026-11-10 (text-to-video 135.2 -> 118.9 s, 43.2 -> 38.6 s). Output bit-identical. Two compilation changes: Qwen3-VL's index metadata (3D RoPE positions, deepstack rows) is computed on the host, so a new prompt length no longer compiles dozens of data-dependent programs (480p reference-to-video text stage 24 -> 7–14 s); `DT_DIT_PRECOMPILE=1` (off by default) compiles a new prompt length's DiT program during the text stage. Fixes: in the JAX server's text buckets the padded queries took part in the attention kernel's fixed-m choice and changed real rows at rounding level, now zeroed; saved DiT programs are keyed by every backend/kernel source and the compiler flags (a narrower key once reused a stale program).

  **2026-11-21** — torch_tpu 每请求耗时(稳态推理,与编译无关)。参考图生视频不再把从不改变的权重在每次用完后拷回主机:9.7 GiB 的 VAE 和停放的文本编码器模块改为换回保留的主机副本。文本编码器的各层改为每个 chip 拷 1/8、再经 ICI all-gather 拼回,不再每个 chip 都经共享的主机链路拷整层。最近的参考图与提示词条件各缓存 8 份(`DT_REF_CACHE`、`DT_TEXT_CACHE`),参考图或提示词重复的请求(例如只换种子)跳过对应编码器。768x1344x345、20 步、v6e-8:新提示词新参考图时参考图生视频无损 167.2 -> **161.8 秒**、有损 66.3 -> **60.9 秒**,两者都重复时 **151.8 秒** / **50.7 秒**;上表 torch_tpu 一列同时计入了 2026-11-10 以来的改动(文生视频 135.2 -> 118.9 秒、43.2 -> 38.6 秒)。输出逐位一致。两项编译相关改动:Qwen3-VL 的索引元数据(3D RoPE 位置、deepstack 行号)改在主机上算,新长度的提示词不再编译几十个数据相关的小程序(480p 参考图生视频文本阶段 24 -> 7–14 秒);`DT_DIT_PRECOMPILE=1`(默认关)在文本阶段提前编译新长度的 DiT 程序。修复:JAX 服务端文本分桶时补齐行的查询参与了注意力 kernel 的 fixed-m 选择、在舍入级改变真实行,现已置零;存盘的 DiT 程序以全部 backend/kernel 源码和编译参数为键(过窄的键曾复用到过期程序)。

- **2026-11-18** — torch_tpu start-up and cache size. All denoising steps now share one DiT program (the first step had one distinct timestep and later steps two, so the time embedding changed shape and a second copy of every grouped program was compiled): 8-chip Pika 480x832x121 cold first run 69 -> **44 s**, ref2va 114 -> **86 s**, 15 -> 7 programs. Compiled DiT programs are now saved and loaded by later processes, skipping Dynamo (`DT_AOT_PERSIST`, on by default in `run_h3_torchtpu.sh`): a fresh process's first t2va run **7–9 s** against 7–8 s steady state (was 20 s), ref2va 29–31 s against 26–29 s, bit-identical in lossless and lossy runs. With one saved copy shared by all ranks, a shape's cache shrinks from 2.45 GB to **0.37 GB** ([details](docs/tpu_v6e.md)). Fixed a hang decoding frame counts whose last chunk has fewer tile rows than chips (e.g. 175 frames at 480p).

  **2026-11-18** — torch_tpu 启动与缓存体积。所有去噪步共用一个 DiT 程序(第 1 步只有一个不同的时间步、之后两个,时间嵌入形状不同,每个分组程序多编一份):8 chip Pika 480x832x121 冷启动首次运行 69 -> **44 秒**,ref2va 114 -> **86 秒**,程序 15 -> 7 个。编好的 DiT 程序现在会存盘,之后的进程直接读回、跳过 Dynamo(`DT_AOT_PERSIST`,`run_h3_torchtpu.sh` 默认打开):新进程第一次 t2va **7–9 秒**,稳态 7–8 秒(原来 20 秒),ref2va 29–31 秒对稳态 26–29 秒,无损与有损模式都逐位一致。所有 rank 共用一份存档后,每个形状的缓存从 2.45 GB 降到 **0.37 GB**([详情](docs/tpu_v6e.md))。修复了最后一段 tile 行数少于 chip 数时解码卡住的问题(例如 480p 175 帧)。
- **2026-11-17** — **MOSS-Image** (OpenMOSS, 5B DiT, Qwen3.5-9B text encoder, FLUX.2 VAE) on the torchax backend. On a v6e-4, 50 steps with CFG 3: 1024x1024 in **4.03 s** end to end, 2048x2048 in **24.9 s**, one chip 1024x1024 in **13.0 s**; resident serving of new prompts 4.05 s a request, or **2.85 s** an image with two 2-chip replicas; the community 8-step DMD adapter **0.48 s** ([usage](#run-moss-image--运行-moss-image)). On a v7x-8 (8 devices): 1024x1024 in **2.68 s** (**2.58 s** with `--dit jax`; 4 devices **2.87 s** / **2.70 s**, 2 devices **4.96 s**, 1 device **9.97 s**), 2048x2048 in **12.84 s** (**12.74 s** with `--dit jax`; 4 devices **16.36 s** / **16.24 s**, 2 devices **31.3 s**, 1 device **63.0 s**); resident serving 2.71 s a request, **1.52 s** an image with two 4-device replicas, or **1.28 s** an image with four 2-device replicas; 8-step DMD **0.38 s** at 1024x1024 and **1.58 s** at 2048x2048. The CFG branches run as one batch on two halves of the chips; the VAE decoder is native JAX (15x the traced one); `kernels/splash.py` gains `kv_mask_all`, one masked body for a runtime kv length (2048x2048 step 723 -> 495 ms; off by default).

  **2026-11-17** — torchax 后端支持 **MOSS-Image**(OpenMOSS,5B DiT、Qwen3.5-9B 文本编码器、FLUX.2 VAE)。v6e-4 上 50 步 CFG 3:1024x1024 端到端 **4.03 秒**,2048x2048 **24.9 秒**,单 chip 1024x1024 **13.0 秒**;常驻服务每条新提示词 4.05 秒,两个 2-chip 副本吞吐 **2.85 秒**/张;社区 8 步 DMD 适配器 **0.48 秒**。v7x-8(8 个 device)上:1024x1024 **2.68 秒**(`--dit jax` **2.58 秒**;4 device **2.87 秒** / **2.70 秒**,2 device **4.96 秒**,单 device **9.97 秒**),2048x2048 **12.84 秒**(`--dit jax` **12.74 秒**;4 device **16.36 秒** / **16.24 秒**,2 device **31.3 秒**,单 device **63.0 秒**);常驻服务每条 2.71 秒,两个 4-device 副本 **1.52 秒**/张,四个 2-device 副本 **1.28 秒**/张;8 步 DMD 1024x1024 **0.38 秒**、2048x2048 **1.58 秒**。CFG 两支作为一个批次在两半 chip 上并行;VAE 解码器为原生 JAX(比追踪版快 15 倍);`kernels/splash.py` 新增 `kv_mask_all`,运行时 kv 长度用单一带 mask 循环体(2048x2048 每步 723 -> 495 ms;默认关闭)。
- **2026-11-17** — torch_tpu: the DiT blocks were compiled twice per shape (an unused `aot_compile` artifact besides the `torch.compile` wrapper), and the extra one could never hit the AOTAutograd cache (its key holds a per-process uuid). Lossless runs now compile once: 8-chip Pika 480x832x121 first run 85 -> **66 s** cold and 37.6 -> **18 s** with a warm cache (on a v7x-8, 768p 345f Pika first run 95 -> **70 s** cold); H3-Image cold 58 -> 42 s (t2i), 82 -> 65 s (edit) on a v6e-8, and on a v7x-8 45 -> **26 s** (2048x2048 t2i, warm **0.58 s**), 52 -> **37 s** (4736x3552 t2i), 52 -> **17 s** (2048x2048 edit); bit-identical output. Lossy runs (`--sparse-window`) no longer need the artifact either: each program gets its own function object, so dense and sparse programs of the same shapes keep separate Dynamo frame caches (warm start 26.5 -> **19-21 s**, bit-identical). Each rank now addresses its chip as `tpu:0` instead of `tpu:<rank>`: torch_tpu 0.3.0 rejects the old index, and on 0.2.0 it made the torch_tpu H3-Image output drift from the JAX reference (DINO 0.910 -> **0.971** against JAX; pixel cosine **0.984**, PSNR **22.4 dB** on v7x-8; video unaffected). torch_tpu 0.3.0 runs but is no faster here, so 0.2.0 stays pinned.

  **2026-11-17** — torch_tpu:DiT block 每个形状编译了两遍(`torch.compile` 包装之外还有一个用不上的 `aot_compile` 产物),多出的那一遍的 AOTAutograd 缓存 key 带每进程不同的 uuid,永远命中不了。无损模式现在只编译一遍:8 chip Pika 480x832x121 首次运行冷启动 85 -> **66 秒**,带缓存 37.6 -> **18 秒**(v7x-8 上 768p 345 帧 Pika 首次运行冷启动 95 -> **70 秒**);v6e-8 上 H3-Image 冷启动 58 -> 42 秒(文生图)、82 -> 65 秒(编辑),v7x-8 上冷启动 45 -> **26 秒**(2048x2048 文生图,暖请求 **0.58 秒**)、52 -> **37 秒**(4736x3552 文生图)、52 -> **17 秒**(2048x2048 编辑);输出逐位一致。有损模式(`--sparse-window`)也不再需要该产物:每个程序一份独立的函数对象,同形状的稠密与稀疏程序各有自己的 Dynamo 帧缓存(暖启动 26.5 -> **19-21 秒**,逐位一致)。各 rank 改用 `tpu:0` 而不是 `tpu:<rank>` 指代自己的 chip:torch_tpu 0.3.0 会拒绝旧写法,在 0.2.0 上它让 torch_tpu 的 H3-Image 出图偏离 JAX 参考(与 JAX 的 DINO 0.910 -> **0.971**;v7x-8 上像素余弦 **0.984**、PSNR **22.4 dB**;视频不受影响)。torch_tpu 0.3.0 能跑但这里没有更快,仍固定 0.2.0。
- **2026-11-16** — New request shapes compile much sooner on a v6e-8 and a v7x-8. Video server (480x832x362 on v6e-8): a new text-to-video shape **151 -> 54 s**, a prompt of a new length 84 -> **11 s**, a new ref2va shape 191 -> **~93 s**, a new reference aspect 60 -> **32 s**, same per-request time. The DiT was compiled twice per shape (step 0's latents came from the host with another sharding), the sparse call waited for the dense one, the text bucket was 64 tokens, the video decoder compiled inside the first request, the audio decoder compiled serially per frame count. H3-Image server: on a v6e-8 a new text-to-image shape 31 -> **10 s** and 0.632 -> 0.598 s a request, a new edit shape 17–29 -> **13–15 s** (on a v7x-8, a new text-to-image shape 45.4 -> **7.9 s** and **0.58 s** a request, a new edit shape 77.9 -> **10.5 s** and **0.93 s** a request with a cached reference): the script's 64 MiB scoped VMEM let the TPU backend also generate whole-activation variants of the DiT's norm+matmul fusions (up to 1.5M instructions), so each program now gets its own ceiling. 4-chip numbers re-measured on a real v6e-4.

  **2026-11-16** — v6e-8 与 v7x-8 上新的请求形状编译快得多。视频服务端(v6e-8 上 480x832x362):新的文生视频形状 **151 -> 54 秒**,新长度的提示词 84 -> **11 秒**,新的 ref2va 形状 191 -> **约 93 秒**,新的参考图宽高比 60 -> **32 秒**,每个请求的耗时不变。原因:DiT 每个形状编译两次(第 0 步的 latent 从主机来,sharding 不同)、稀疏程序等稠密程序编完才编、文本桶 64、视频解码器在第一个请求里才编译、音频解码器按帧数串行编译。H3-Image 服务端:v6e-8 上新的文生图形状 31 -> **10 秒**、每个请求 0.632 -> 0.598 秒,新的编辑形状 17–29 -> **13–15 秒**(v7x-8 上新的文生图形状 45.4 -> **7.9 秒**、每请求 **0.58 秒**,新的编辑形状 77.9 -> **10.5 秒**、复用参考图每请求 **0.93 秒**):脚本设的 64 MiB scoped VMEM 让 TPU 后端为 DiT 的归一化+矩阵乘融合额外生成整块激活的版本(最多 150 万条指令),现在每个程序单独设上限。4 chip 的数字已在真正的 v6e-4 上复测。
- **2026-11-12** — The tiled reference encode, once compiled, had every chip encode all the tiles (a `device_put` does not split a batch inside `jit`); it is now split by a sharding constraint: H3-Image edit on a v6e-8 JAX 1.22 -> **1.15 s**, torchax 1.24 -> 1.17 s, and on a v7x-8 JAX 1.29 -> **0.91 s** (`2.65 s` at 4096x4096), torchax 1.91 -> **0.93 s** (Pika ref2va unchanged, bit-identical). `torch_tpu` also compiles the 27-layer Qwen3-VL vision tower (`DT_VIS_COMPILE=1`, cosine 0.9993) and skips a duplicate reference resize (`1.12 ->` **1.03 s** on a v7x-8, text+vision `0.21 -> 0.13 s`). `torch_tpu` uses the 128-lane RoPE for video too: on a v6e-8 Pika 768p 43.1 -> 42.3 s (mean of three warm requests), ref2va 169.4 -> 166.4 s; on a v7x-8 (`libtpu 0.0.52`) Pika 768p 33.4 -> **32.5 s** (mean of three warm requests; **32.1 s** steady state = `0.4 + 28.9 + 2.9`, and **29.5 s** = `0.4 + 26.3 + 2.9` with `DT_POST_ATTN_PIPE=1`).

  **2026-11-12** — 参考图分块编码在编译后让每个 chip 都编了全部 tile(`jit` 里 `device_put` 不会切分批次),改用 sharding constraint 切分:v6e-8 上 H3-Image 编辑 JAX 1.22 -> **1.15 秒**,torchax 1.24 -> 1.17 秒,v7x-8 上 JAX 1.29 -> **0.91 秒**(4096x4096 为 **2.65 秒**),torchax 1.91 -> **0.93 秒**(Pika ref2va 不变,逐位一致)。`torch_tpu` 同时将 27 层 Qwen3-VL 视觉塔编译为一个程序(`DT_VIS_COMPILE=1`,余弦相似度 0.9993)并省去重复缩放(v7x-8 上编辑 `1.12 ->` **1.03 秒**,文本+视觉 `0.21 -> 0.13 秒`)。`torch_tpu` 视频也改用 128 通道 RoPE:v6e-8 上 Pika 768p 43.1 -> 42.3 秒(三次暖请求平均),ref2va 169.4 -> 166.4 秒;v7x-8(`libtpu 0.0.52`)上 Pika 768p 33.4 -> **32.5 秒**(三次暖请求平均;稳态 **32.1 秒** = `0.4 + 28.9 + 2.9`,开启 `DT_POST_ATTN_PIPE=1` 稳态 **29.5 秒** = `0.4 + 26.3 + 2.9`)。
- **2026-11-12** — MiniMax-H3-Image serving and torch_tpu. `run_h3_image_jax.sh --requests reqs.jsonl` keeps the models resident and serves text-to-image and edit requests, the text padded to 64-token buckets (masked) so prompts in a bucket share one compiled program: 2048x2048 **0.67 s** a request, an edit with a known reference size 1.35 s ([usage](#run-minimax-h3-image--运行-minimax-h3-image)). torch_tpu moves to libtpu 0.0.52 (the same attention kernel 5% faster; Pika 768p 45.1 -> 43.6 s, ref2va 181.8 -> 170.1 s), its image decoder writes attention as two matmuls and a softmax and keeps the FF 2-D (a 261-row tile made every 3-D view a copy; 4736x3552 decode 0.131 -> 0.076 s), and the ranks meet after each request so the first request after warm-up no longer waits for the slowest rank's compile (edit 1.7–2.1 -> 1.31 s). torch_tpu images on a v6e-8: 2048x2048 0.68 s, 4736x3552 2.95 s, edit 1.32 s.

  **2026-11-12** — MiniMax-H3-Image 服务与 torch_tpu。`run_h3_image_jax.sh --requests reqs.jsonl` 让模型常驻,处理文生图与编辑请求;文本按 64 token 分桶补齐(补齐部分被屏蔽),同一桶的提示词共用一个编译程序:2048x2048 每条 **0.67 秒**,参考图尺寸已编译过的编辑 1.35 秒。torch_tpu 升级到 libtpu 0.0.52(同一注意力 kernel 快 5%;Pika 768p 45.1 -> 43.6 秒,ref2va 181.8 -> 170.1 秒);图像解码器的注意力改写成两次矩阵乘加 softmax、FF 保持二维(261 行的 tile 让每个三维视图都是一次拷贝;4736x3552 解码 0.131 -> 0.076 秒);每条请求结束各 rank 汇合一次,warm-up 后的第一条请求不再等最慢 rank 的编译(编辑 1.7–2.1 -> 1.31 秒)。v6e-8 上 torch_tpu 图像:2048x2048 0.68 秒,4736x3552 2.95 秒,编辑 1.32 秒。
- **2026-11-12** — MiniMax-H3-Image JAX: the 53-row text head is also selected in place rather than written back with a dynamic-update-slice, which had split the gated residual and the norm into three full-sequence passes a block. v6e-8: 4736x3552 3.05 -> **2.77 s**, 2048x2048 0.67 -> **0.63 s**. torch_tpu images default to the 128-lane RoPE (`DT_BHSD_ROPE=full`) and do the masked modulation in float32 with one rounding (`DT_SEGMOD_F32=1`; torch's per-op bfloat16 rounding sat inside the FF projection's prologue): 4736x3552 3.19 -> 3.05 s, edit 1.44 -> 1.36 s.

  **2026-11-12** — MiniMax-H3-Image JAX:53 行的文本头部也改为原地选择,不再用 dynamic-update-slice 写回(它把门控残差与 norm 拆成每个 block 三次整序列遍历)。v6e-8:4736x3552 3.05 -> **2.77 秒**,2048x2048 0.67 -> **0.63 秒**。torch_tpu 的图像默认改用 128 通道 RoPE(`DT_BHSD_ROPE=full`),掩码调制改在 float32 里算、最后舍入一次(`DT_SEGMOD_F32=1`;torch 逐算子的 bfloat16 舍入落在 FF 投影的前序里):4736x3552 3.19 -> 3.05 秒,编辑 1.44 -> 1.36 秒。
- **2026-11-12** — MiniMax-H3-Image on torchax and torch_tpu: the segmented modulation selects each head run (text, and an edit's reference rows) with a mask over the whole sequence instead of slicing it out and `index_copy`ing it back, as the JAX backend now does. v6e-8, torchax: 2048x2048 0.74 -> **0.64 s**, 4736x3552 3.12 -> **2.80 s**, edit 1.72 -> 1.24 s; torch_tpu: 4736x3552 3.36 -> 3.19 s, edit 1.74 -> 1.44 s. Video keeps slicing.

  **2026-11-12** — torchax 与 torch_tpu 上的 MiniMax-H3-Image:分段调制改为在整条序列上用掩码选出各头部段(文本,以及编辑的参考图行),不再切出来再 `index_copy` 写回,与 JAX 后端现在的做法一致。v6e-8 上 torchax:2048x2048 0.74 -> **0.64 秒**,4736x3552 3.12 -> **2.80 秒**,编辑 1.72 -> 1.24 秒;torch_tpu:4736x3552 3.36 -> 3.19 秒,编辑 1.74 -> 1.44 秒。视频仍用切片。
- **2026-11-12** — MiniMax-H3-Image editing: the tiled reference encode runs as one compiled program on every backend instead of op by op (1024x1024 reference: JAX 0.57 -> 0.13 s, torch_tpu 0.67 -> 0.09 s), and torch_tpu compiles the text tower after the vision merge. Edit 2048x2048: JAX 1.96 -> 1.52 s (**1.22 s** with the v7x segment-modulation change), torchax 2.14 -> 1.72 s, torch_tpu 2.39 -> 1.74 s on a v6e-8 ([numbers](#minimax-h3-image--six-steps)).

  **2026-11-12** — MiniMax-H3-Image 编辑:三种后端的参考图分块编码都改为一个编译好的程序,不再逐个算子执行(1024x1024 参考图:JAX 0.57 -> 0.13 秒,torch_tpu 0.67 -> 0.09 秒);torch_tpu 在视觉特征合并之后编译文本塔。v6e-8 上编辑 2048x2048:JAX 1.96 -> 1.52 秒(加上 v7x 的分段调制改动为 **1.22 秒**),torchax 2.14 -> 1.72 秒,torch_tpu 2.39 -> 1.74 秒。
- **2026-11-11** — MiniMax-H3-Image (six-step text-to-image and editing) on all three backends; diffusers moves to `875373a`, which ships the model, for every model here (JAX, torchax and Qwen-Image outputs unchanged). On a v6e-8 the JAX backend generates a 2048x2048 image in **0.68 s** and a 4736x3552 one in **3.08 s** ([numbers](#minimax-h3-image--six-steps)). An image is a short sequence, so its JAX transformer keeps the residual stream whole, leaves the attention output split by heads and gives each chip matching SwiGLU rows: every block then needs two plain all-reduces (4736x3552 denoise 7.42 -> 2.99 s). torchax takes the same two steps for its attention output and SwiGLU rows (4736x3552 4.00 -> 3.12 s). torch_tpu stitches the decoded tiles in one compiled program instead of op by op (decode 0.66 -> 0.14 s at 4736x3552) and compiles the text tower with the ids padded to a bucket of 256 (text 0.4-0.7 -> 0.03 s, cosine 0.999999 to op by op).

  **2026-11-11** — MiniMax-H3-Image(六步文生图与图像编辑)支持三种后端;diffusers 升级到包含该模型的 `875373a`,本仓库所有模型同步升级(JAX、torchax 和 Qwen-Image 输出不变)。v6e-8 上 JAX 后端生成 2048x2048 图像 **0.68 秒**,4736x3552 **3.08 秒**([数字](#minimax-h3-image--six-steps))。图像序列短,其 JAX transformer 让残差流整份常驻、注意力输出保持按头切分、SwiGLU 行按 chip 交错:每个 block 只需两次普通 all-reduce(4736x3552 去噪 7.42 -> 2.99 秒)。torchax 的注意力输出与 SwiGLU 行做了同样两项处理(4736x3552 4.00 -> 3.12 秒)。torch_tpu 把解码后的 tile 拼接编译成一个程序,不再逐算子执行(4736x3552 解码 0.66 -> 0.14 秒),并把文本塔按 256 分桶补齐后编译(文本 0.4-0.7 -> 0.03 秒,与逐算子执行的余弦相似度 0.999999)。
- **2026-11-11** — **MiniMax-H3-Image** on TPU v7x-8 (`scripts/run_h3_image_{jax,torchax,torchtpu}.sh`): 2048x2048 takes **0.53 s** e2e (`0.52 s` warm, denoise `0.49 s`) on JAX (**0.56 s**/request with `--requests`), **0.55 s** on torchax, and **0.58 s** (`0.55 s` warm) on torch_tpu; 2048x2048 editing takes **0.91 s** e2e (`0.82 s` warm, **0.77 s** denoise+decode; **0.93 s**/request with `--requests`) on JAX, **0.93 s** on torchax, and **1.03 s** (`0.90 s` warm, **0.82 s** denoise+decode) on torch_tpu; 4736x3552 / 4096x4096 takes **1.98 s** e2e (`1.95 s` warm, denoise **1.85 s** with `DT_POST_ATTN_PIPE=1`) / **2.31 s** on torch_tpu, **2.19 s** (`2.18 s` warm, denoise `2.10 s`) on JAX, **2.21 s** on torchax, and **2.65 s** (`2.56 s` warm, **2.51 s** denoise+decode) for 4096x4096 editing on JAX ([GPU comparison & details](docs/tpu_v7x.md#5-minimax-h3-image-6-step-2k--4k-on-tpu-v7x-8-vs-gpu-baselines--minimax-h3-image-性能对比与优化)).

  **2026-11-11** — **MiniMax-H3-Image**(6 步 2K/4K 文生图与多参考图编辑)在 TPU v7x-8 上的性能(`scripts/run_h3_image_{jax,torchax,torchtpu}.sh`):2048x2048 文生图 JAX 仅需 **0.53 秒**(热态 `0.52 秒`,去噪 `0.49 秒`;`--requests` 常驻服务每条 **0.56 秒**),torchax **0.55 秒**,torch_tpu **0.58 秒**(热态 `0.55 秒`);2048x2048 编辑 JAX **0.91 秒**(热态 `0.82 秒`,去噪+解码 **0.77 秒**;`--requests` 常驻服务每条 **0.93 秒**),torchax **0.93 秒**,torch_tpu **1.03 秒**(热态 `0.90 秒`,去噪+解码 **0.82 秒**);4736x3552 / 4096x4096 文生图 torch_tpu **1.98 秒**(`DT_POST_ATTN_PIPE=1`,热态 `1.95 秒`,去噪 **1.85 秒**)/ **2.31 秒**,JAX **2.19 秒**(热态 `2.18 秒`,去噪 `2.10 秒`),torchax **2.21 秒**,4096x4096 编辑 JAX **2.65 秒**(热态 `2.56 秒`,去噪+解码 **2.51 秒**)([GPU 对比与详情](docs/tpu_v7x.md#5-minimax-h3-image-6-step-2k--4k-on-tpu-v7x-8-vs-gpu-baselines--minimax-h3-image-性能对比与优化))。
- **2026-11-11** — torch_tpu 2-slab post-attention pipeline (`DT_POST_ATTN_PIPE=1`): splits `to_out -> all_reduce -> norm2 -> FFN -> all_reduce` into two 4096-aligned slabs so each slab's `all_reduce` overlaps the other's `norm2 + FFN` compute; 768p Pika-H3-Flex 34.6 -> **32.0 s** on a v7x-8 on `libtpu 0.0.51`, and 32.1 -> **29.5 s** on `libtpu 0.0.52` (denoise 28.9 -> **26.3 s**, decode **2.9 s**, PSNR 21.8 dB; off on v6e-8 where 32 GB HBM is full; [details](docs/tpu_v7x.md)).

  **2026-11-11** — torch_tpu 注意力后 2-slab 流水线(`DT_POST_ATTN_PIPE=1`):将 `to_out -> all_reduce -> norm2 -> FFN -> all_reduce` 切为两个 4096 对齐分块,使两块的 `all_reduce` 与对方的 `norm2 + FFN` 计算重叠;v7x-8 上 768p Pika-H3-Flex 在 `libtpu 0.0.51` 下从 34.6 秒降至 **32.0 秒**,在 `libtpu 0.0.52` 下从 32.1 秒降至 **29.5 秒**(去噪 28.9 -> **26.3 秒**,解码 **2.9 秒**,PSNR 21.8 dB;v6e-8 每 chip 32 GB HBM 不足故保持关闭;[详情](docs/tpu_v7x.md))。
- **2026-11-11** — torch_tpu DiT block: fuse `residual + gate * x` before `index_copy` in segmented modulation and transpose Q/K/V to `(B, H, S, D)` before `norm_q`/`norm_k` and RoPE. Same frames; 768p Pika-H3-Flex 35.1 -> **34.6 s** on a v7x-8, and on a v6e-8 Pika 47.6 -> **46.6 s**, lossy text-to-video 45.1 -> **43.2 s**.

  **2026-11-11** — torch_tpu DiT block:在分段调制 `index_copy` 前融合 `residual + gate * x`,并在 `norm_q`/`norm_k` 与 RoPE 前将 Q/K/V 转置为 `(B, H, S, D)`。帧逐位不变;v7x-8 上 768p Pika-H3-Flex 35.1 -> **34.6 秒**,v6e-8 上 Pika 47.6 -> **46.6 秒**、有损文生视频 45.1 -> **43.2 秒**。
- **2026-11-10** — torch_tpu on TPU v7x-8: keep the sharded text encoder resident in v7x's 95 GiB HBM (`--no-offload`) instead of parking it per request: 768p Pika-H3-Flex 38.1 -> 35.7 s (denoise/other 32.4 -> 29.8 s), same frames.

  **2026-11-10** — TPU v7x-8 上的 torch_tpu:利用每 device 95 GiB HBM 将分片文本编码器常驻(`--no-offload`),不再每条请求来回卸载:768p Pika-H3-Flex 38.1 -> 35.7 秒(`denoise/other` 32.4 -> 29.8 秒),帧不变。
- **2026-11-10** — torch_tpu video decode: pipelined `/dev/shm` writes with parallel host reads, zero-copy `.numpy()`, and 8-rank tail-chunk tile splitting. Same frames; Pika-H3-Flex 40.3 -> **35.1 s** on a v7x-8 (decode 6.0 -> 4.1 s) and 48.7 -> **47.4 s** on a v6e-8 (lossy t2va 46.3 -> 44.5 s, ref2va 80.0 -> 78.0 s).

  **2026-11-10** — torch_tpu 视频解码:分片边算边写 `/dev/shm` 配合主机并行读取、零拷贝 `.numpy()` 及末尾 4 块跨 8 卡分摊。帧不变;Pika-H3-Flex 在 v7x-8 上 40.3 -> **35.1 秒**(解码 6.0 -> 4.1 秒),在 v6e-8 上 48.7 -> **47.4 秒**(有损文生 46.3 -> 44.5 秒,参考图生 80.0 -> 78.0 秒)。
- **2026-11-10** — torch_tpu video decode: the clip is assembled and postprocessed on the host by rank 0 alone, instead of on every rank and then sent back to the chip and out again; chunks cross `/dev/shm` as raw bf16. Same frames. 768p Pika-H3-Flex 44.3 -> 40.3 s on a v7x-8 (decode 9.0 -> 6.0 s); on a v6e-8 Pika 51.3 -> 48.7 s, lossy text-to-video 48.3 -> 46.1 s, lossy reference-to-video 82.2 -> 79.7 s.

  **2026-11-10** — torch_tpu 视频解码:片段只由 rank 0 在主机上拼接和后处理,不再每个 rank 各做一遍、再送回 chip 又拷出来;片段经 `/dev/shm` 以 bf16 原样交换。帧逐位不变。v7x-8 上 768p Pika-H3-Flex 44.3 -> 40.3 秒(解码 9.0 -> 6.0 秒);v6e-8 上 Pika 51.3 -> 48.7 秒,有损文生视频 48.3 -> 46.1 秒,有损参考图生视频 82.2 -> 79.7 秒。
- **2026-11-09** — torchax picks the attention KV block the way the JAX backend does (4096 past 241 frames): 768p Pika-H3-Flex 39.3 -> 38.5 s, lossy text-to-video 35.1 -> 34.8 s on a v6e-8. A different block changes the bf16 accumulation order, so the output changes too (video DINO 0.936 against 2048; another bit-level change, dropping an optimization barrier, gives 0.968), with the same scene and quality.

  **2026-11-09** — torchax 的注意力 KV 分块改为与 JAX 后端相同的自动选择(超过 241 帧用 4096):v6e-8 上 768p Pika-H3-Flex 39.3 -> 38.5 秒,有损文生视频 35.1 -> 34.8 秒。分块不同会改变 bf16 累加顺序,输出也随之变化(视频 DINO 对 2048 为 0.936;另一种逐位级改动,去掉一个 optimization barrier,为 0.968),画面内容和质量相同。
- **2026-11-09** — torchax decodes the audio with the JAX BigVGAN, beside the video decode: it no longer adds 0.7 s, and the waveform matches diffusers' float32 decode at 105 dB SNR (the traced decode at default precision: 19.7 dB). Lossy reference-to-video 52.8 s, Pika-H3-Flex 39.4 s on a v6e-8.

  **2026-11-09** — torchax 的音频改用 JAX BigVGAN 解码,与视频解码同时进行:不再多花 0.7 秒,波形对 diffusers float32 解码 SNR 105 dB(原先默认精度的追踪解码只有 19.7 dB)。v6e-8 上有损参考图生视频 52.8 秒,Pika-H3-Flex 39.4 秒。
- **2026-11-09** — Colab notebook for one free TPU v5e chip, plus a copy with the outputs of an end-to-end run ([details](#free-colab-notebook--免费-colab-notebook)).

  **2026-11-09** — 新增单个免费 TPU v5e chip 的 Colab notebook,以及一份端到端跑过的带输出版本([详情](#free-colab-notebook--免费-colab-notebook))。
- **2026-11-08** — torchax reference-to-video: the AdaLN modulation of the 15.5k text, reference and audio rows no longer gathers them across chips every block; DiT calls match the JAX backend, lossy 768p 68.5 -> 58.0 s on a v6e-8.

  **2026-11-08** — torchax 参考图生视频:文本、参考图和音频共 1.55 万行的 AdaLN 调制不再每个 block 都跨 chip 汇总;DiT 调用与 JAX 后端持平,v6e-8 上 768p 有损 68.5 -> 58.0 秒。

- **2026-11-08** — torchax decodes the video with the JAX backend's decoder (`--jax-vae`, now the default): 768p Pika-H3-Flex 47.3 -> 40.9 s, lossless MiniMax-H3 117.8 -> 110.0 s on a v6e-8; video DINO 0.999 against diffusers' VAE.

  **2026-11-08** — torchax 改用 JAX 后端的视频解码器(`--jax-vae`,现为默认):v6e-8 上 768p Pika-H3-Flex 47.3 -> 40.9 秒,MiniMax-H3 无损 117.8 -> 110.0 秒;视频与 diffusers VAE 的 DINO 为 0.999。

- **2026-11-08** — torch_tpu text-to-video runs the text encoder tensor-parallel on the chips instead of on the host: Pika-H3-Flex 768p 71.1 -> 51.3 s on a v6e-8 (the VAE also stays resident for text-to-video).

  **2026-11-08** — torch_tpu 文生视频改为在 chip 上张量并行运行文本编码器,不再放在主机上:v6e-8 上 Pika-H3-Flex 768p 71.1 -> 51.3 秒(文生视频的 VAE 也改为常驻)。

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

Hardware-specific architecture notes, optimization ledgers, and XLA flag guides are in `docs/`:

各代 TPU 的硬件架构说明、完整优化演进台账与 XLA 编译参数指南见 `docs/` 目录:

- **TPU v7x (Ironwood, 4 chips = 8 TensorCore devices, 95 GiB HBM/device, 256×256 MXU)**: [`docs/tpu_v7x.md`](docs/tpu_v7x.md)
- **TPU v6e (Trillium, 8 chips = 8 TensorCore devices, 32 GiB HBM/chip, 256×256 MXU)**: [`docs/tpu_v6e.md`](docs/tpu_v6e.md)
- **TPU v5e (`v5litepod-1` / Colab, 1 chip, 16 GiB HBM, 128×128 MXU)**: [`docs/tpu_v5e.md`](docs/tpu_v5e.md)

Measured 2026-11-02 to 2026-11-04 on a TPU v6e-8 (8 chips); warm requests, with the model already loaded and compiled. The torch_tpu column was re-measured on 2026-11-21 (mean of two warm requests, each with a new prompt and reference, so neither request cache applies).

测于 2026-11-02 至 2026-11-04,TPU v6e-8(8 个 chip);均为热请求时间,模型已加载并编译完成。torch_tpu 一列于 2026-11-21 重测(两次暖请求平均,按每次都是新提示词、新参考图计,不命中两个请求缓存)。

Software: jax 0.12.0.1, libtpu 0.0.52 (torch_tpu moved from 0.0.51 on 2026-11-12: same attention kernel 5% faster), torch_tpu 0.2.0, diffusers `875373a` (see `requirements/`).

软件版本:jax 0.12.0.1、libtpu 0.0.52(torch_tpu 于 2026-11-12 从 0.0.51 升级:同一注意力 kernel 快 5%)、torch_tpu 0.2.0、diffusers `875373a`(见 `requirements/`)。

### MiniMax-H3 — 768x1344, 345 frames (14.4 s), 20 steps

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| text-to-video, lossless / 文生视频,无损 | **110.0 s** | 110.0 s | 124.0 s |
| text-to-video, lossy / 文生视频,有损 | **34.0 s** | 34.8 s | 39.5 s |
| reference-to-video, lossless / 参考图生视频,无损 | **139.2 s** | 186.7 s | 164.9 s |
| reference-to-video, lossy / 参考图生视频,有损 | **51.2 s** | 51.6 s | 57.7 s |

JAX times include decoding the audio; the two lossless JAX numbers were measured before it was added (about 0.1 s more). torch_tpu was measured on 2026-11-23 with text buckets on (the default). Steady-state times vary by about 3% between boots of the same machine; compare runs from one boot.

JAX 的时间包含音频解码;两个无损 JAX 数字是加入音频解码之前测的(会多约 0.1 秒)。torch_tpu 测于 2026-11-23,文本分桶开启(默认)。同一台机器不同开机之间稳态时间约有 3% 波动;比较请用同一次开机的结果。

"Lossy" skips a DiT call while the timestep embedding barely changes, and lets each video frame attend only to nearby frames.

"有损"是指:时间步嵌入几乎不变时跳过一次 DiT 计算,并且让每一帧只关注相邻的帧。

The visual quality stays close to lossless; check it on your own prompts before relying on it.

画质与无损版本接近;正式使用前,请先用自己的提示词检查效果。

The resident server keeps the model on the chips between requests: 34.0 s (text) and 47.5 s (reference) per lossy request.

常驻服务在请求之间把模型保留在 chip 上:每个有损请求 34.0 秒(文生)和 47.5 秒(参考图生)。

#### A new prompt / 新提示词

The table above is steady state: the same prompt again, everything compiled. A prompt of a new length changes the packed sequence; whether that recompiles depends on the entry point. Text-to-video, 768x1344x345, 20 steps, v6e-8:

上表是稳态(同一提示词再来一次,程序都已编好)。新长度的提示词会改变打包序列,是否重新编译取决于入口。文生视频,768x1344x345,20 步,v6e-8:

| | lossless / 无损 | lossy / 有损 | compile / 编译 |
|---|---|---|---|
| torch_tpu, same process / 同进程 | 126.4 s (152.6 s before text buckets / 分桶前) | 41.8 s (89.4 s) | none within a 256-token bucket; a new bucket compiles in the background / 同一 256 token 桶内不编译,跨桶在后台编译 |
| JAX server, same process / JAX 服务端,同进程 | 107.9 s | 34.3 s | none within a bucket / 桶内不编译 |
| JAX CLI, new process / JAX 命令行,新进程 | 107.6 s + 105.5 s load and compile before the request / 请求前加载与编译 | — | every process / 每个进程 |
| torchax, new process / torchax,新进程 | 111.0 s + 535.6 s warm-up compile / 预热编译 | — | every process / 每个进程 |

Reference-to-video on torch_tpu, same process: 191.3 -> 166.7 s lossless, 103.2 -> 60.6 s lossy.

torch_tpu 参考图生视频同进程:无损 191.3 -> 166.7 秒,有损 103.2 -> 60.6 秒。

### Qwen-Image-2.1 — 1024x1024

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| 20 steps / 20 步 | **0.74 s** | 1.28 s | 1.18 s |
| 4 steps / 4 步 | **0.28 s** | 0.49 s | 0.55 s |

All three backends use all 8 chips for Qwen-Image.

Qwen-Image 的三种后端都使用全部 8 个 chip。

### MiniMax-H3-Image — six steps

v6e-8, warm requests (text encoder, transformer and decoder resident), end to end including text encoding and decode.

v6e-8,暖请求(文本编码器、transformer 与解码器常驻),端到端,含文本编码与解码。

| | JAX | torchax | torch_tpu |
|---|---|---|---|
| 2048x2048 (4,149 rows / 行) | **0.63 s** | 0.64 s | 0.68 s |
| 4736x3552 (16,481 rows / 行) | **2.77 s** | 2.80 s | 2.95 s |
| edit 2048x2048, one reference / 编辑,一张参考图 | **1.15 s** | 1.17 s | 1.32 s |
| edit 4096x4096, one reference / 编辑,一张参考图 | **3.38 s** | | |

On 4 chips (a v6e-4) add `--chips 4 --text-int8`: the text encoder is kept on chip in int8 (first 50 layers) beside the bf16 transformer. 2048x2048 in 0.88 s, 4736x3552 in 4.22 s, a one-reference 2048x2048 edit in 1.52 s (measured on a v6e-4). `--text-offload` keeps it in host memory instead and streams it each request (+0.5 s).

4 个 chip(v6e-4)上加 `--chips 4 --text-int8`:文本编码器以 int8(前 50 层)常驻 chip,与 bf16 transformer 并存。2048x2048 0.88 秒,4736x3552 4.22 秒,一张参考图的 2048x2048 编辑 1.52 秒(在 v6e-4 上实测)。`--text-offload` 则把它放在主机内存、每次请求流式送入(多 0.5 秒)。

On one TPU v5e chip (16 GB HBM, 47 GB host memory) add `--chips 1 --offload --int8-weights --free-text-after`: the prompt is encoded, the text encoder freed, and the int8 DiT streamed from host memory each step. 2048x2048 in 11.4 s, 4736x3552 in 43.7 s; `--int8-weights` changes the output a little.

单个 TPU v5e chip(16 GB HBM、47 GB 主机内存)上加 `--chips 1 --offload --int8-weights --free-text-after`:先编码提示词并释放文本编码器,int8 DiT 每步从主机内存流式送入。2048x2048 11.4 秒,4736x3552 43.7 秒;`--int8-weights` 会让结果略有变化。

v7x-8 (4 chips = 8 TensorCore devices, 95 GiB HBM/device), warm requests, end to end including text encoding and decode (see [`docs/tpu_v7x.md`](docs/tpu_v7x.md) for GPU baselines on 1×/2×/4× H200 and 2× B300):

v7x-8(4 个 chip = 8 个 TensorCore device,每 device 95 GiB HBM),暖请求,端到端含文本编码与解码(与 1×/2×/4× H200 及 2× B300 的 GPU 对比见 [`docs/tpu_v7x.md`](docs/tpu_v7x.md)):

| Workflow / 模式 (TPU v7x-8) | JAX | torchax | torch_tpu |
|---|---:|---:|---:|
| 2048x2048 (4,149 rows / 行) | **0.53 s** (`0.52 s` warm) | 0.55 s (`0.53 s` warm) | 0.58 s (`0.55 s` warm) |
| 4736x3552 / 4096x4096 (16,481 rows / 行) | 2.19 s (`2.18 s` warm) | 2.21 s (`2.20 s` warm) | **1.98 s** (`DT_POST_ATTN_PIPE=1`, denoise **1.85 s**) / 2.31 s |
| edit 2048x2048, one reference / 编辑,一张参考图 | **0.91 s** (**0.77 s** denoise+decode) ¹ | 0.93 s (0.79 s ¹) | 1.03 s (0.82 s ¹) |
| edit 4096x4096, one reference / 编辑,一张参考图 | **2.65 s** (**2.51 s** denoise+decode) ¹ | — | — |

¹ Full end-to-end time includes Qwen3-VL vision+text encoding and sharded 3D reference VAE encoding (`0.05–0.06 s`); parenthesized time is `denoise + 2D VAE decode`. With `--requests reqs.jsonl` resident serving on `v7x-8`, a new 2048x2048 shape compiles and runs in **7.9 s** (`t2i`) / **10.5 s** (`edit`), warm 2048x2048 text-to-image requests run in **0.56–0.58 s**, and warm one-reference edits run in **0.93 s**.

Six steps make the images chaotic in bfloat16: changing only an attention block size moves the latents to a cosine of 0.85 by the fifth step, so backends agree on scene and quality, not on pixels (the diffusers documentation says the same).

六步采样在 bfloat16 下对微小差异很敏感:只改注意力分块大小,第五步的 latents 余弦相似度就降到 0.85,因此各后端的画面内容与质量一致,但像素不同(diffusers 文档也这样说明)。

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

#### Free Colab notebook / 免费 Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shungcp/diffusers-tpu/blob/main/notebooks/h3_colab_v5e.ipynb)

[`notebooks/h3_colab_v5e.ipynb`](notebooks/h3_colab_v5e.ipynb) runs Pika-H3-Flex text-to-video with audio (480x832, 141 frames) on the free Colab TPU v5e-1 runtime. It contains code only: it downloads the weights from the official Hugging Face repositories, and you must comply with the MiniMax H3 Community License, including its territory restrictions. [`h3_colab_v5e_executed.ipynb`](notebooks/h3_colab_v5e_executed.ipynb) is the same notebook with the outputs of one end-to-end run, video included.

[`notebooks/h3_colab_v5e.ipynb`](notebooks/h3_colab_v5e.ipynb) 在免费 Colab TPU v5e-1 上运行 Pika-H3-Flex 带音频文生视频(480x832,141 帧)。notebook 只含代码:权重从 Hugging Face 官方仓库下载,使用者须遵守 MiniMax H3 Community License,包括其中的地域限制。[`h3_colab_v5e_executed.ipynb`](notebooks/h3_colab_v5e_executed.ipynb) 是同一个 notebook 端到端跑过一次的带输出版本,含视频。

In that run, with the int8 cache restored from a Cloud Storage bucket: cache restore ~10 min, weight download ~8 min, the generation step 66 min, of which `generate` itself was 81.5 s (denoise 48.5 s, video decode 5.1 s, audio 13.5 s). Most of the 66 min was reading the cache page by page from Colab's slow disk; the run predates commit `824a60d`, which reads it sequentially. The audio, decoded on the host, took 13.5 s there and did not hide behind the video decode as it does on a v5litepod-1.

那次运行从 GCS 恢复 int8 缓存:恢复缓存约 10 分钟,下载权重约 8 分钟,生成那一步 66 分钟,其中 `generate` 本身 81.5 秒(去噪 48.5 秒、视频解码 5.1 秒、音频 13.5 秒)。66 分钟里大部分是从 Colab 慢盘按页读取缓存;那次运行早于提交 `824a60d`,该提交改为顺序读取。音频在主机上解码,在 Colab 上用了 13.5 秒,没有像在 v5litepod-1 上那样藏到视频解码后面。

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

Parentheses show `(denoise + video decode)` in seconds. On a v7x-8, a 15-second 480p clip with sound takes 8.4 s. On `torch_tpu` (bf16, `libtpu 0.0.52`), 768p 345-frame text-to-video on a v7x-8 takes **32.1 s** (28.9 + 2.9; **32.5 s** mean of three warm requests) with default 128-lane RoPE (`DT_BHSD_ROPE=full`), **32.6 s** (29.3 + 2.9; **33.4 s** mean of three warm requests; **34.6 s** on `libtpu 0.0.51`) with bit-identical `DT_BHSD_ROPE=1`, and **29.5 s** (26.3 + 2.9; **30.4 s** mean of three warm requests; **32.0 s** on `libtpu 0.0.51`) with `DT_POST_ATTN_PIPE=1`.

括号内为 `(去噪 + 视频解码)` 秒数。在 v7x-8 上,一段 15 秒带声音的 480p 视频只需 8.4 秒。在 `torch_tpu`(bf16,`libtpu 0.0.52`)下,v7x-8 上 768p 345 帧文生视频在默认 128 通道 RoPE(`DT_BHSD_ROPE=full`)下稳态为 **32.1 秒**(28.9 + 2.9;三次暖请求平均 **32.5 秒**),逐位一致的 `DT_BHSD_ROPE=1` 下为 **32.6 秒**(29.3 + 2.9;三次暖请求平均 **33.4 秒**;`libtpu 0.0.51` 下为 **34.6 秒**),开启 `DT_POST_ATTN_PIPE=1` 为 **29.5 秒**(26.3 + 2.9;三次暖请求平均 **30.4 秒**;`libtpu 0.0.51` 下为 **32.0 秒**)。

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

#### torch_tpu on a v7x-8 — 768x1344, 345 frames, 20 steps / v7x-8 上的 torch_tpu

Measured 2026-11-24 with the v7x-8 defaults (two-slab post-attention pipeline, kv block 4096; bf16, text buckets on), same settings as the v6e-8 table above. "Steady" repeats the prompt; "new prompt" is a new length in the same 256-token bucket, same process.

测于 2026-11-24,使用 v7x-8 默认配置(两段注意力后流水、kv 块 4096;bf16,文本分桶开启),设置与上面 v6e-8 的表相同。"稳态"是同一提示词再来一次;"新提示词"是同一进程里同一 256 token 桶内的新长度。

| | steady / 稳态 | new prompt / 新提示词 | v6e-8 steady / v6e-8 稳态 |
|---|---|---|---|
| text-to-video, lossless / 文生视频,无损 | **85.1 s** | 88.6 s | 124.0 s |
| text-to-video, lossy / 文生视频,有损 | **26.9 s** | 30.6 s | 39.5 s |
| reference-to-video, lossless / 参考图生视频,无损 | **96.2 s** | 101.1 s | 164.9 s |
| reference-to-video, lossy / 参考图生视频,有损 | **31.9 s** | 37.6 s | 57.7 s |

The JAX server (bf16, `--block-kv 4096`) on the same v7x-8 takes 88.0 s lossless and 28.3 s lossy for text-to-video. Without the pipeline (`--no-post-attn-pipe`) torch_tpu takes 93.0 / 29.9 / 105.0 / 35.2 s.

同一台 v7x-8 上 JAX 服务端(bf16,`--block-kv 4096`)文生视频无损 88.0 秒、有损 28.3 秒。关掉流水(`--no-post-attn-pipe`)时 torch_tpu 为 93.0 / 29.9 / 105.0 / 35.2 秒。

### MOSS-Image — 50 steps, CFG 3, torchax backend

v6e-4 (4 chips), v6e-8 (8 chips), and v7x-8 (up to 8 devices), warm, end to end including text encoding and decode. GPU numbers are third-party (OpenMOSS publishes none); the H100 one runs the official diffusers pipeline with the same settings.

v6e-4(4 个 chip)、v6e-8(8 个 chip)与 v7x-8(至多 8 个 device),暖请求,端到端含文本编码与解码。GPU 数字来自第三方(OpenMOSS 未公布速度);H100 那一行是官方 diffusers pipeline、相同参数。

| | 1 chip (`v6e` / `v7x`) | 2 chips (`v6e` / `v7x`) | 4 chips (`v6e` / `v7x`) | 8 chips (`v6e-8` / `v7x-8`) | GPU |
|---|---|---|---|---|---|
| 1024x1024 | 13.0 s / **9.97 s** | 5.64 s / **4.96 s** | 4.03 s / **2.87 s** (**2.70 s** `--dit jax`) | **2.56 s** / **2.68 s** (**2.58 s** `--dit jax`) | H100 (diffusers) 17.1 s |
| 2048x2048 | 82.2 s / **63.0 s** | — / **31.3 s** | 24.9 s / **16.36 s** (**16.24 s** `--dit jax`) | **13.8 s** / **12.84 s** (**12.74 s** `--dit jax`) | RTX 5090 (bf16, DiT only) ≈139 s |

Resident serving of new prompts (`--requests`): on v6e, 4.05 s a request on 4 chips, 2.43 s on 8 (**2.89 s** on 4 v7x devices, **2.71 s** on 8 v7x devices); `--replicas 2` (two model copies, two chips each) 2.85 s an image on 4 chips (**2.50 s** on 4 v7x devices, **1.52 s** on 8 v7x devices with two 4-device copies); on 8 chips `--replicas 4` gives the most throughput, 1.46 s an image (5.8 s latency) on v6e-8 and **1.28 s** an image (5.12 s latency) on v7x-8. The community 8-step DMD adapter (`--dmd`, no CFG): 0.48 s at 1024x1024 on v6e-4, 0.46 s on v6e-8, and **0.38 s** on v7x-8 (**1.58 s** at 2048x2048 on v7x-8). A new image size compiles its text, denoising and decode programs side by side (the decoder and the step are compiled from their shapes alone while the text encoder compiles; the decoder at a 16 MiB scoped-VMEM ceiling): first request on 8 chips with an empty cache 74.6 -> **40 s** at 1024x1024, 87.4 -> **55 s** at 2048x2048.

常驻服务(`--requests`,每条新提示词):v6e 上 4 chip 每条 4.05 秒,8 chip 2.43 秒(v7x 4 device **2.89 秒**,8 device **2.71 秒**);4 chip 上 `--replicas 2`(两份模型各 2 chip)每张 2.85 秒(v7x 4 device **2.50 秒**,8 device 两份 4-device 模型每张 **1.52 秒**);8 chip 上 `--replicas 4` 吞吐最高,v6e-8 每张 1.46 秒(延迟 5.8 秒),v7x-8 每张 **1.28 秒**(延迟 5.12 秒)。社区 8 步 DMD 适配器(`--dmd`,无 CFG):1024x1024 v6e-4 0.48 秒、v6e-8 0.46 秒、v7x-8 **0.38 秒**(v7x-8 上 2048x2048 **1.58 秒**)。新图像尺寸的文本、去噪、解码三个程序并行编译(解码器与去噪一步只按形状编译,与文本编码器同时进行;解码器用 16 MiB scoped VMEM):8 chip、空缓存时首个请求 1024x1024 74.6 -> **40 秒**,2048x2048 87.4 -> **55 秒**。

### MOSS-SoundEffect-v2 — 100 steps, CFG 4, a 10 s clip

v6e-4, warm, end to end (prompt encoding, 100 steps, decode). The model always denoises a 30 s latent (1,500 frames) and crops, so the time does not depend on the requested length. OpenMOSS publishes no speed numbers.

v6e-4,暖请求,端到端(提示词编码、100 步、解码)。模型总是按 30 秒的 latent(1500 帧)去噪再裁剪,所以耗时与请求的时长无关。OpenMOSS 未公布速度。

| | 1 chip | 2 chips / 2 chip | 4 chips / 4 chip | 4 one-chip replicas / 4 个单 chip 副本 |
|---|---|---|---|---|
| per clip / 每条 | 2.1 s | 1.2 s | **0.95 s** | **0.58 s** (throughput / 吞吐) |
| per step / 每步 | 21 ms | 11.2 ms | 9.5 ms | |

The second mesh axis splits the sequence, not the heads (`--split seq`, the default): each chip keeps the whole model, computes its half of the rows and gathers the keys and values (two small all-gathers a block). Splitting the heads (tensor parallelism, `--split heads`) costs two all-reduces of the whole residual stream a block and is slower than 2 chips (16.9 ms a step).

第二个 mesh 轴切分序列而不是头(`--split seq`,默认):每个 chip 保留完整模型,计算一半的行,并 gather 键和值(每个 block 两次小的 all-gather)。按头切(张量并行,`--split heads`)每个 block 要对整条残差流做两次 all-reduce,比 2 chip 还慢(每步 16.9 ms)。

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

A new shape (size, frame count, reference aspect) compiles once and is cached on disk; the text is padded to 256-token buckets so prompts of different lengths share programs. References of any size: `--ref-aspect-buckets default` center-crops each reference to the nearest of 13 common ratios (a few percent of one side at most; a reference already at one of them is unchanged), and `--warmup-ref-buckets` compiles every bucket at startup (13 buckets in ~10 min on a v6e-8, then cached on disk): odd sizes such as 1300x1000 or 1080x2400 then run at steady state (12–17 s) instead of compiling for 30–60 s. Or, without changing any reference, `--ref-pad-buckets` pads the reference's vision patches, text, encoder tiles and DiT rows to buckets (masked): the same 13 ratios then compile in ~7 min and an odd size runs at steady state (13–20 s), each request ~3–4% slower from the padding; the output differs from the unpadded run only by bf16 rounding. Off switches for the compile-time measures: `DT_COMMIT_LATENTS`, `DT_PAR_SPARSE`, `DT_PAR_ENC`, `DT_PRE_DENSE` (`t2va` by default, `1` for both), `DT_PRE_SPARSE`, `DT_WARM_VIDEO_DECODE`, `DT_BG_AUDIO_COMPILE`, `DT_ENC_VMEM_KIB` (`0` turns each off).

新形状(尺寸、帧数、参考图宽高比)只编译一次并缓存到磁盘;文本补齐到 256 token 的桶,不同长度的提示词共用程序。任意尺寸的参考图:`--ref-aspect-buckets default` 把参考图中心裁剪到 13 种常见宽高比中最近的一种(最多裁掉一边的几个百分点;本来就是这些比例的参考图不变),`--warmup-ref-buckets` 在启动时把每个桶都编译好(v6e-8 上 13 个桶约 10 分钟,之后缓存在磁盘):1300x1000、1080x2400 这类尺寸随即以稳态运行(12–17 秒),不再编译 30–60 秒。若不想改动参考图,`--ref-pad-buckets` 把参考图的视觉 patch、文本、编码 tile 与 DiT 行补齐到桶(补齐部分屏蔽):同样 13 种比例约 7 分钟编完,奇怪尺寸以稳态运行(13–20 秒),每个请求因补齐慢约 3–4%;输出与不补齐时只差 bf16 舍入。编译加速各项的开关:`DT_COMMIT_LATENTS`、`DT_PAR_SPARSE`、`DT_PAR_ENC`、`DT_PRE_DENSE`(默认 `t2va`,`1` 两者都开)、`DT_PRE_SPARSE`、`DT_WARM_VIDEO_DECODE`、`DT_BG_AUDIO_COMPILE`、`DT_ENC_VMEM_KIB`(设 `0` 关闭)。

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

### Agnes-Video-H3-MoE

Download `Agnes-AI/Agnes-Video-H3-MoE` next to `MiniMaxAI/MiniMax-H3` and pass its snapshot (or its HF cache directory) to the JAX pipeline. The split is picked from the chip's HBM (`DT_MOE_DP=auto`); `DT_MOE_DP=0/1`, `DT_MOE_EP=1`, `DT_MOE_RING=0` and `DT_MOE_FP8=1` override it.

下载 `Agnes-AI/Agnes-Video-H3-MoE`(与 `MiniMaxAI/MiniMax-H3` 放在同一缓存),把它的快照(或 HF 缓存目录)传给 JAX 管线。切分方式按 chip 的 HBM 自动选择(`DT_MOE_DP=auto`);`DT_MOE_DP=0/1`、`DT_MOE_EP=1`、`DT_MOE_RING=0`、`DT_MOE_FP8=1` 可以覆盖。

```bash
hf download Agnes-AI/Agnes-Video-H3-MoE
python -m diffusers_tpu.models.minimax_h3.jax_pipeline --height 480 --width 832 --num-frames 124 --steps 50 \
  --moe $HF_HOME/hub/models--Agnes-AI--Agnes-Video-H3-MoE --prompt "..." --runs 2
```

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

## Run MiniMax-H3-Image / 运行 MiniMax-H3-Image

```bash
bash scripts/run_h3_image_jax.sh --size 2048x2048 --prompt "..."
bash scripts/run_h3_image_jax.sh --size 4736x3552 --prompt "..."
bash scripts/run_h3_image_jax.sh --size 4096x4096 --reference ref.png --prompt "Keep everything in <Picture 1> ..."
bash scripts/run_h3_image_torchax.sh --size 2048x2048 --prompt "..."
bash scripts/run_h3_image_torchtpu.sh --size 2048x2048 --prompt "..."     # torch_tpu environment
```

Many prompts with the models resident (JAX): one JSON object per line, `{"prompt": ..., "seed": ..., "reference": ["ref.png"]}` (`reference` for edits, which need the process started with `--reference`). Prompts are padded to 64-token buckets, so only a new bucket (or a new reference size) compiles.

模型常驻、处理多条提示词(JAX):每行一个 JSON,`{"prompt": ..., "seed": ..., "reference": ["ref.png"]}`(`reference` 表示编辑请求,需要进程以 `--reference` 启动)。提示词按 64 token 分桶补齐,只有新的桶(或新的参考图尺寸)才会编译。

```bash
bash scripts/run_h3_image_jax.sh --size 2048x2048 --prompt "..." --requests reqs.jsonl
bash scripts/run_h3_image_jax.sh --size 2048x2048 --reference ref.png --prompt "..." --requests reqs.jsonl
```

A reference reused across requests is preprocessed and VAE-encoded once (a repeated edit 1.23 -> 1.10 s on a v6e-8, same output). The reference encode compiles per tile count (`--enc-tile-bucket 32`) rather than per image size. A new reference shape still compiles the vision tower and the transformer once (~80–120 s on a v6e-8, cached on disk for later processes). A new shape compiles in ~10–20 s on a v6e-8 instead of ~80–120 s: its programs compile in parallel (`--parallel-compile`), the transformer, text and vision layers run as scans over stacked weights so each compiles one layer (`--scan-layers`, on with 8+ chips; an extra copy of those weights in HBM, ~1–2% slower a request), and each program gets its own scoped-VMEM ceiling instead of the script's 64 MiB, which leaves room for whole-activation tiles that the backend code-generates and then discards (2048x2048 text-to-image step 29 -> 9 s and 0.632 -> 0.599 s a request; text+vision tower 19 -> 7 s; reference encode 24 -> 8 s; `DT_STEP_VMEM_KIB`, `DT_TV_VMEM_KIB`, `DT_ENC_VMEM_KIB` override, `0` keeps the global value for the latter two). A new shape whose activation exceeds 64 MiB (4736x3552, long edits) is served by a 32 MiB step program at once while the faster 64 MiB one compiles in the background (`DT_STEP_FAST_KIB`, `0` off; 7040-row edit first request 21.6 -> 17.7 s). To have more reference shapes share one program, pad coarser at ~8% per edit request: `--text-bucket-edit 512 --ref-bucket 1024 --vision-bucket 1024` (defaults 64 / 512 / off keep every request at full speed).

同一张参考图被多个请求复用时,预处理与 VAE 编码只做一次(v6e-8 上重复编辑 1.23 -> 1.10 秒,输出不变)。参考图编码按 tile 数编译(`--enc-tile-bucket 32`),不再按图像尺寸。新的参考图形状仍要编译一次视觉塔与 transformer(v6e-8 上约 80–120 秒,磁盘缓存供之后的进程复用)。v6e-8 上新形状约 10–20 秒编译完(原来 80–120 秒):各程序并行编译(`--parallel-compile`);transformer、文本与视觉各层改为对堆叠权重的 scan,每个只编译一层(`--scan-layers`,8 个及以上 chip 时开启;HBM 多一份这些权重,每个请求慢约 1–2%);每个程序单独设 scoped VMEM 上限,不用脚本的 64 MiB——上限装得下整块激活作为一个分块时,后端会多生成一个这样的版本再丢弃(2048x2048 文生图 step 29 -> 9 秒、每请求 0.632 -> 0.599 秒;文本+视觉塔 19 -> 7 秒;参考图编码 24 -> 8 秒;可用 `DT_STEP_VMEM_KIB`、`DT_TV_VMEM_KIB`、`DT_ENC_VMEM_KIB` 覆盖,后两者设 `0` 沿用全局值)。激活超过 64 MiB 的新形状(4736x3552、长编辑)先用 32 MiB 的 step 程序立即服务,同时后台编译更快的 64 MiB 程序,编好后切换(`DT_STEP_FAST_KIB`,`0` 关闭;7040 行编辑首个请求 21.6 -> 17.7 秒)。若希望更多参考图形状共用程序,可用更粗的补齐,每个编辑请求约慢 8%:`--text-bucket-edit 512 --ref-bucket 1024 --vision-bucket 1024`(默认 64 / 512 / 关,每个请求保持全速)。

The text encoder is the same file as MiniMax-H3's (the Hub deduplicates it), and the prompt is used verbatim.

文本编码器与 MiniMax-H3 是同一个文件(Hub 会去重),提示词原样使用。

---

## Run MOSS-Image / 运行 MOSS-Image

```bash
hf download OpenMOSS-Team/MOSS-Image-Base
bash scripts/run_moss_image.sh --prompt "..."                  # 1024x1024, all chips
bash scripts/run_moss_image.sh --prompt "..." --size 2048
bash scripts/run_moss_image.sh --prompt "..." --chips 1
bash scripts/run_moss_image.sh --requests reqs.jsonl --out out  # {"prompt": ..., "seed": ..., "tag": ...} per line
bash scripts/run_moss_image.sh --requests reqs.jsonl --replicas 2
```

The model code is MOSS-Image's own `pipeline_moss_image.py` from the Hub snapshot; the denoising loop is ours: each step (both CFG branches, the CFG combine and the Euler update) is one compiled program, so nothing waits on the host until the decode.

模型代码就是 Hub 快照里 MOSS-Image 自己的 `pipeline_moss_image.py`;去噪循环是我们的:每一步(CFG 两支、CFG 合并与 Euler 更新)是一个编译好的程序,解码之前不等主机。

`--dit jax` runs the same step as native JAX (`models/moss_image/jax_dit.py`): about 2–6% faster (on a v6e-4: 1024x1024 3.96 s, 2048x2048 24.48 s; on a v7x-8: 1024x1024 **2.58 s** on 8 devices / **2.70 s** on 4 devices, 2048x2048 **12.74 s** on 8 devices / **16.24 s** on 4 devices); the time left is the all-reduces and the attention kernel.

`--dit jax` 用原生 JAX 跑同一个步骤(`models/moss_image/jax_dit.py`):快约 2–6%(v6e-4 上 1024x1024 3.96 秒、2048x2048 24.48 秒;v7x-8 上 1024x1024 8 device **2.58 秒** / 4 device **2.70 秒**、2048x2048 8 device **12.74 秒** / 4 device **16.24 秒**);剩下的时间是 all-reduce 和注意力 kernel。

The 8-step DMD adapter (`lodestones/MOSS-Image-SFT-8-steps-dmd`, a community distillation of MOSS-Image-SFT) runs with `--dmd`:

8 步 DMD 适配器(`lodestones/MOSS-Image-SFT-8-steps-dmd`,MOSS-Image-SFT 的社区蒸馏版)用 `--dmd` 运行:

```bash
hf download OpenMOSS-Team/MOSS-Image-SFT --include "transformer/*" --include "scheduler/*"
hf download lodestones/MOSS-Image-SFT-8-steps-dmd generator.safetensors
bash scripts/run_moss_image.sh --dmd --prompt "..."
```

---

## Run VideoGym-LTX-2.3 / 运行 VideoGym-LTX-2.3

```bash
hf download OpenMOSS-Team/VideoGym-LTX-2.3      # ~96 GB (Gemma-3-12B in float32 is half of it)
bash scripts/run_ltx2.sh --image canvas.png --prompt "Move the red dot along the white path to the green square."
# On TPU v7x-8, run_ltx2.sh auto-selects resident=48 & 4x2 CFG-parallel (6.60 s BF16); add --fp8 for 5.82 s E2E:
bash scripts/run_ltx2.sh --image canvas.png --prompt "..." --fp8
```

`--image` is the 768x512 canvas (a 512x512 puzzle centered, black borders, as in the model card); it is re-compressed as H.264 CRF 33 like diffusers / ltx-pipelines (`--crf 0` skips that). Output: H.264 mp4 with the model's (untrained, near-silent) audio. `--frames` (8k + 1), `--steps`, `--guidance`, `--stg`, `--seed` as needed; `--resident` sets how many DiT blocks stay whole on every chip (20 on v6e, auto 48 on v7x). Warm `--compile-cache-dir` first run takes `36 s` (`v6e-8`) / `29.6 s` (`v7x-8`). Full `v7x-8` optimization ledger: [docs/tpu_v7x.md (Section 9)](docs/tpu_v7x.md#9-videogym-ltx-23-22b-audio-video-dit-on-tpu-v7x-8--videogym-ltx-23-在-v7x-8-上的优化台账).

`--image` 为 768x512 画布(512x512 谜题居中、两侧黑边,同模型卡);与 diffusers / ltx-pipelines 一样先做 H.264 CRF 33 重压缩(`--crf 0` 跳过)。输出 H.264 mp4,附带模型的音频(未训练,接近静音)。可按需设置 `--frames`(8k + 1)、`--steps`、`--guidance`、`--stg`、`--seed`;`--resident` 为每个 chip 上整份常驻的 DiT 块数(`v6e` 默认 20,`v7x` 自动 48)。复用 `--compile-cache-dir` 时首跑仅需 `36 秒`(`v6e-8`) / `29.6 秒`(`v7x-8`)。完整 `v7x-8` 优化台账见 [docs/tpu_v7x.md 第 9 节](docs/tpu_v7x.md#9-videogym-ltx-23-22b-audio-video-dit-on-tpu-v7x-8--videogym-ltx-23-在-v7x-8-上的优化台账)。

| VideoGym-LTX-2.3 (`768×512×129 @ 24 fps`, 30 steps, 4 rows) | Warm E2E | Denoise (30 steps) | Decode (`video + audio`) | Peak HBM |
|---|---|---|---|---|
| `v6e-8` (`1×8, res=20`, BF16) | `15.03 s` | `14.10 s` (`0.470 s/step`) | `0.84 s` | `27.6 GiB` |
| `v7x-8` (`1×8, res=48`, BF16) | `8.79 s` | `7.37 s` (`0.246 s/step`) | `1.39 s` (CPU `uint8`) | `44.2 GiB` |
| **`v7x-8` auto default (`4×2, res=48`, BF16 lossless)** | **`6.60 s`** (`2.28×`) | **`6.06 s`** (`0.202 s/step`) | **`0.49 s`** (`0.255 s` vid + `0.230 s` aud) | `44.2 GiB` |
| **`v7x-8` `--fp8` (`4×2, res=48`, W8A8 E4M3)** | **`5.82 s`** (`2.58×`) | **`5.29 s`** (`0.176 s/step`) | **`0.49 s`** (`PSNR 40.00 dB` vs `1×8`) | **`27.0 GiB`** |

---

## Run OmniVAE / 运行 OmniVAE

```bash
hf download OpenMOSS-Team/OmniVAE --local-dir omnivae_release \
  --include "models/dit/t2av/t2av_recon_distill_avclip/*" "models/vae/audio_video/recon_distill_avclip/*" \
            "models/vae/audio_only/recon_distill_avclip_ft_decoder/*" "models/text_encoder/*"   # ~17 GB of the 96 GB bundle
export OMNIVAE_RELEASE_ROOT=$PWD/omnivae_release
bash scripts/run_omnivae.sh --prompt "A dog barks twice at a passing bicycle on a quiet street."   # all chips, dual CFG
bash scripts/run_omnivae.sh --prompt "..." --cfg-mode simple --seed 7
```

Output: H.264 + AAC (48 kHz) mp4. `--frames`, `--height`, `--width`, `--steps`, `--guidance`, `--template` (T2AV template index) as needed.

输出 H.264 + AAC(48 kHz)mp4。可按需设置 `--frames`、`--height`、`--width`、`--steps`、`--guidance`、`--template`(T2AV 模板序号)。

---

## Run MOVA / 运行 MOVA

```bash
hf download OpenMOSS-Team/MOVA-360p
bash scripts/run_mova.sh --image ref.jpg --prompt "A man in a blue blazer speaks in a quiet room ..."   # 352x640, 8 s, all chips
bash scripts/run_mova.sh --image ref.jpg --prompt "..." --cfg 2      # 8 chips: CFG branches on two groups of 4
hf download OpenMOSS-Team/MOVA-720p
bash scripts/run_mova.sh --repo OpenMOSS-Team/MOVA-720p --height 720 --width 1280 --image ref.jpg --prompt "..."   # 720p
```

The reference image is center-cropped to the output size (352x640 by default) as in the official script. Output: H.264 + AAC (48 kHz) mp4. `--frames`, `--steps`, `--cfg-scale`, `--seed` as in the official CLI.

参考图按官方脚本居中裁剪到输出尺寸(默认 352x640)。输出 H.264 + AAC(48 kHz)mp4。`--frames`、`--steps`、`--cfg-scale`、`--seed` 与官方一致。

---

## Run MOSS-SoundEffect-v2 / 运行 MOSS-SoundEffect-v2

```bash
hf download OpenMOSS-Team/MOSS-SoundEffect-v2.0
bash scripts/run_moss_soundeffect.sh --prompt "Rain on a tin roof, distant thunder" --seconds 10     # all chips
bash scripts/run_moss_soundeffect.sh --prompt "清晨小鸟叽叽喳喳地叫着" --seconds 8 --chips 1
bash scripts/run_moss_soundeffect.sh --requests reqs.jsonl --replicas 4 --out out   # {"prompt", "seconds", "seed", "tag"} per line
```

Prompts are cleaned like the official prompter (`ftfy` + HTML unescaping + whitespace); `ftfy` is not a dependency here, so its default fixes (full-width punctuation, curly quotes, ligatures, control characters, NFC) are reproduced except mojibake repair. The DAC `.pth` is read with `torch.load(weights_only=True)`: it holds a state dict and its constructor arguments only.

提示词按官方 prompter 清洗(`ftfy` + HTML 反转义 + 空白归一);这里不依赖 `ftfy`,复现了它默认的各项修正(全角标点、弯引号、连字、控制字符、NFC),只有乱码修复除外。DAC 的 `.pth` 用 `torch.load(weights_only=True)` 读取:里面只有 state dict 和构造参数。

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
| text-to-video / 文生视频 | **36.5 s** | 38.5 s | 46.6 s |
| reference-to-video / 参考图生视频 | **51.1 s** | 67.0 s | 79.4 s |

The adapter is merged into the weights while loading, so the denoising step costs the same as the base model.

适配器在加载时就合并进权重,所以每一步去噪的开销和底座模型一样。

The adapters are released under the MiniMax H3 Community License by Pika Labs.

这两个适配器由 Pika Labs 以 MiniMax H3 社区许可证发布。

---

## Run MAGI-2-preview / 运行 MAGI-2-preview

Sand.ai's 113.93B (5.95B active) Multi-Head Latent MoE audio+video model with 4-stream mHC, Attention Sink, Qwen3.5-27B text encoder, Stage 2 1080p 6.72B Refiner, TurboVAED / Wan2.2 video VAE, and AutoencoderOobleck stereo audio VAE. Official resolution presets are `272p` (`256x448`, 12.5 fps), `540p` (`512x896`, 12.5 fps), and `1080p` (`1088x1920`, 25 fps via Stage 2 Refiner). Full `v7x-8` optimization ledger: [docs/tpu_v7x.md (Section 8)](docs/tpu_v7x.md#8-magi-2-preview-11393b-multi-head-latent-moe--1080p-refiner-on-tpu-v7x-8--magi-2-preview-端到端优化台账).

Sand.ai 113.93B(5.95B 激活)多头潜空间 MoE 音视频生成模型,包含 4 流 mHC、Attention Sink、Qwen3.5-27B 文本编码器、二阶段 1080p 6.72B Refiner、TurboVAED / Wan2.2 视频 VAE 与 AutoencoderOobleck 立体声音频 VAE。官方分辨率预设为 `272p`(`256x448`,12.5 fps)、`540p`(`512x896`,12.5 fps)与 `1080p`(`1088x1920`,二阶段 Refiner 25 fps)。完整 `v7x-8` 优化台账见 [docs/tpu_v7x.md 第 8 节](docs/tpu_v7x.md#8-magi-2-preview-11393b-multi-head-latent-moe--1080p-refiner-on-tpu-v7x-8--magi-2-preview-端到端优化台账)。

```bash
hf download sand-ai/MAGI-2-preview --local-dir /path/to/magi2_preview_weights
export MAGI2_CKPT_ROOT=/path/to/magi2_preview_weights

# Priority 1: Native JAX (T2V / I2V / Ref2VA, 272p / 540p / 1080p; --lossy 2.6x, --enhance PE expander, --fp8)
bash scripts/run_magi2_jax.sh 540p --prompt "A red fox trotting through snow at golden hour" --enhance
bash scripts/run_magi2_jax.sh 540p --lossy --prompt-file assets/sample_enhanced_t2v.json
bash scripts/run_magi2_jax.sh 1080p --prompt-file assets/sample_enhanced_t2v.json   # Stage 1 + 8.27x Stage 2 Refiner
bash scripts/run_magi2_jax.sh i2v 540p --lossy --image ref.png --prompt 'A barista smiling saying "Careful, it is still very hot."' --enhance

# Resident Native JAX Server (all 4 BF16 models resident in 41.8 GiB/chip, zero reload across JSONL requests)
bash scripts/run_magi2_jax_server.sh --requests reqs.jsonl --resolution 540p --lossy

# Priority 2: torchax (PyTorch DiT traced through JAX/shard_map/lax.scan)
bash scripts/run_magi2_torchax.sh 540p --fp8 --prompt "..."

# Priority 3: torch_tpu (8-rank torchrun, spmd_safe + Pallas jax_op + DT_AOT_PERSIST + E2E MP4)
bash scripts/run_magi2_torchtpu.sh --mode bench --resolution 540p --fp8
bash scripts/run_magi2_torchtpu.sh --mode denoise --resolution 540p --fp8 --steps 100 \
  --prompt "A red fox trotting through snow at golden hour" --enhance --decode-mp4
```

| MAGI-2-preview on TPU v7x-8 (10 s, CFG=2) | Native JAX | torchax | torch_tpu |
|---|---|---|---|
| Stage 1 `272p BF16` (`113.93B`, `S=29,664`) | `0.960 s/step` | `0.960 s/step` | **`0.923 s/step`** |
| Stage 1 `272p FP8-W8A8` | — | `0.963 s/step` | **`0.717–0.829 s/step`** (`1.45 s` warm `DT_AOT_PERSIST`) |
| Stage 1 `540p BF16` (`113.93B`, `S=115,680`, lossless) | `3.18–3.42 s/step` (`67.7 GiB` HBM) | `3.728 s/step` | **`3.331 s/step`** |
| Stage 1 `540p FP8-W8A8` (lossless) | `3.430 s/step` (`42.7 GiB` HBM) | `3.745 s/step` | **`3.248 s/step`** |
| Stage 1 `540p BF16 --lossy` (100 steps I2V, TeaCache + late `cond-only` CFG skip) | **`1.266 s/step` eff.** (`129.76 s` denoise / `181.23 s` warm E2E, **`2.61×`**, PSNR `22.62 dB`, audio corr `0.9793`) | — | — |
| Stage 2 `1080p` Refiner (`6.72B`, 5 steps, default BF16 `DP=2×CP=4` lossless) | **`10.56 s/step`** (`53.38 s` / 5 steps, **`8.27×`** vs `87.35 s`, `cos=0.999999`) | — | — |
| Stage 2 `1080p` Refiner (`--refiner-qk-fp8`, opt-in) | **`10.15 s/step`** (`50.75 s` warm / 5 steps, **`8.61×`**) | — | — |
| Stage 3a `TurboVAED` 8-chip sharded decode (`540p` / `1080p` warm) | **`0.72 s`** (`1.80×`) / **`3.88 s`** (`3.09×`, bit-exact `PSNR=∞`) | — | Uses JAX Stage 3 via `--decode-mp4` |
| Stage 0a `Wan2.2` I2V ref encode / Stage 0b `Qwen3.5` `--uncond-cache` | **`0.29 s`** cache (`22.7 ms` warm) / **`1.62 ms`** disk hit | — | Uses JAX Stage 0 via `--prompt --enhance` |
| Resident Multi-Request Server (`run_magi2_jax_server.sh`) | **`41.80 GiB` HBM/chip** (all 4 BF16 models, `33.65 s` warm startup once, `0 s` reload) | — | — |

> **Audio note / 音频提示:** For rich synchronized dialogue and Foley, either pass `--enhance` (offline PE JSON expander extracting quoted dialogue into `dynamic_layer.timeline_segments`) or pass an upstream PE-structured JSON/markdown prompt via `--prompt-file`; short visual-only prompts are strongly attenuated by the upstream 429-token `DEFAULT_NEGATIVE_PROMPT` (which includes `music, instrumental`) under `audio_cfg=7.0`. Leave `--adaptive-volume-scaling` off (default) unless you want loud clips attenuated to `-31 dBFS`.
>
> 如需清晰同步的人声对白与音效,可附加 `--enhance`(内置离线 PE 剧本扩展器,自动提取引号台词至 `dynamic_layer.timeline_segments`),或通过 `--prompt-file` 传入官方 PE 结构化 JSON/Markdown 剧本;极短的纯画面提示词在 `audio_cfg=7.0` 和官方 429-token 默认负向词(内含 `music, instrumental`)下会被压成近乎静音。默认不开启 `--adaptive-volume-scaling`(仅衰减至 `-31 dBFS`)。

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
