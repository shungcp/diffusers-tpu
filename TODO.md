# Known small wins not taken / 已知但没做的小收益

Measured, judged not worth the complexity at the time, and left here for anyone who wants to squeeze them out. Numbers are from a TPU v6e-8 unless noted.

测过、当时判断不值得为之增加复杂度的优化,记在这里供以后参考。除注明外数字均为 TPU v6e-8。

## torch_tpu, MiniMax-H3 / Pika-H3-Flex

- **Cache the reference image preprocessing for a repeated reference (~0.3 s a request).** At the start of a reference-to-video request the chip waits while the host resizes the reference with PIL (129 ms, diffusers `before_encoder`), runs the Qwen2-VL image processor (82 ms), hashes the pixel values for the text-conditioning cache key (93 ms; SHA-1 is already the fastest of hashlib here) and copies the pixels over (56 ms). Keyed by the original image bytes, the processed inputs could be reused when the same reference comes again: Pika ref2va 480x832x121 8.0 s -> ~7.7 s, 768p lossy ~0.6%. Needs patching two upstream diffusers steps; a new reference gets nothing.

  **重复参考图时缓存其预处理(每请求约 0.3 秒)。** 参考图生视频请求开头 chip 空等主机:PIL 缩放参考图 129 ms(diffusers `before_encoder`)、Qwen2-VL 图像处理器 82 ms、为文本条件缓存的键对像素做哈希 93 ms(SHA-1 已是本机 hashlib 里最快的)、像素拷上 chip 56 ms。以原图字节为键缓存处理结果,同一参考图再来时可复用:Pika ref2va 480x832x121 8.0 -> 约 7.7 秒,768p 有损约 0.6%。需要给上游 diffusers 两个步骤打补丁;新参考图没有收益。

- **Skip the reference-cache key's device-to-host copy (~21 ms a reference-to-video request).** The encoded-reference cache hashes the reference pixels, which are already on the chip by then, so every request copies a 2048x2048 frame back. Hashing on the host would need the pixels before upstream moves them to the chip.

  **省掉参考图缓存键的拷回(每个参考图生视频请求约 21 ms)。** 参考编码缓存对参考像素做哈希,而此时像素已在 chip 上,每个请求都把 2048x2048 的一帧拷回主机。要在主机上算,得在上游把像素搬上 chip 之前拿到它们。

- **The last small compiles for a new reference-to-video prompt (15 tiny eager programs).** A new prompt length still compiles 15 one-op programs (`arange`, `index_put_`, `zeros`, ...), probably in the Qwen3-VL text path with an image (not traced; text-to-video: 1). The request time already equals the steady state within noise.

  **新提示词参考图生视频剩下的小编译(15 个单算子程序)。** 新长度提示词仍会编译 15 个单算子小程序(`arange`、`index_put_`、`zeros` 等),推测在带图的 Qwen3-VL 文本路径里(未追踪;文生视频只剩 1 个)。请求时间已与稳态相差在噪声以内。

- **Sequence parallel within tensor parallel (13% slower today).** Splitting the residual stream by sequence makes the elementwise work 1/8, but torch_tpu's reduce-scatter + all-gather took 2.37 s a step against 1.25 s for the all-reduce (768p t2va 6.34 -> 7.19 s/step). Worth retrying if reduce-scatter gets faster.

  **张量并行内再做序列并行(目前慢 13%)。** 残差流按序列切后逐元素运算降到 1/8,但 torch_tpu 的 reduce-scatter + all-gather 每步 2.37 秒,而 all-reduce 是 1.25 秒(768p 文生视频每步 6.34 -> 7.19 秒)。reduce-scatter 变快后值得重试。

- **Ulysses sequence parallel with whole weights on each device (TPU v7x only, not tried).** Removes the per-block all-reduces (all-reduce is ~20% of a 768p step on v6e) for four all-to-alls, about 8x fewer bytes. The 66 GB DiT does not fit a v6e chip's 32 GB; a v7x device has 95 GB. Qwen-Image already does this (`DT_QWEN_ULYSSES`, 20 steps 1.04 -> 0.74 s).

  **Ulysses 序列并行 + 每个 device 整份权重(仅 TPU v7x,未试)。** 用 4 次 all-to-all 取代每块的 all-reduce(v6e 上 768p 每步约 20% 是 all-reduce),通信字节约少 8 倍。66 GB 的 DiT 放不进 v6e chip 的 32 GB;v7x 每个 device 95 GB。Qwen-Image 已这样做(`DT_QWEN_ULYSSES`,20 步 1.04 -> 0.74 秒)。

- **SparseCore all-reduce offload (4% slower today).** `xla_tpu_enable_sparse_core_collective_offload_all_reduce`: 768p t2va 6.33 -> 6.59 s/step. Recheck with newer libtpu.

  **SparseCore 承担 all-reduce(目前慢 4%)。** `xla_tpu_enable_sparse_core_collective_offload_all_reduce`:768p 文生视频每步 6.33 -> 6.59 秒。新版 libtpu 再测。

- **Two-slab post-attention pipeline on v6e (no gain).** `--post-attn-pipe` overlaps each block's all-reduce with the other slab's compute: 95.8 -> 87.8 s at 768p on a v7x-8, unchanged on a v6e-8 (124.6 / 123.1 vs 124.8 / 122.7 s), so it stays off by default.

  **v6e 上的两段注意力后流水(无收益)。** `--post-attn-pipe` 让每块的 all-reduce 与另一段的计算重叠:v7x-8 上 768p 95.8 -> 87.8 秒,v6e-8 上不变(124.6 / 123.1 对 124.8 / 122.7 秒),所以默认关。

## JAX, Agnes-Video-H3-MoE

- **Hide the MoE input all-gather on v6e-8 (768p: ~121 ms a step).** Tensor parallel over the experts' width needs every token on every chip, and unlike the dense FFN (XLA collective matmul) the all-gather in front of the MoE layer is exposed. Rotating the input blocks around the same ring as the reduce-scatter was slower (two permutes in one direction did not overlap). On v7x-8 DP-8 avoids it.

  **在 v6e-8 上藏住 MoE 前面的 all-gather(768p 每步约 121 ms)。** 按专家宽度做张量并行要求每个 chip 看到所有 token;稠密 FFN 由 XLA collective matmul 藏住通信,MoE 层前面的 all-gather 是暴露的。把输入块放进同一个环里轮转反而更慢(两路同方向 permute 没有重叠)。v7x-8 上 DP-8 不需要它。

- **Expert parallel for a model with balanced routing.** `DT_MOE_EP=1` (ragged all-to-all, one expert a device) was 1.4–2x faster than tensor parallel per layer with even routing, but Agnes's routing (mean max/avg 4.2x) makes it slower end to end.

  **路由均衡的模型可用专家并行。** `DT_MOE_EP=1`(ragged all-to-all,每 device 一个专家)在路由均匀时单层比张量并行快 1.4–2 倍,但 Agnes 的路由(最忙专家平均是均值的 4.2 倍)让它端到端更慢。

- **Row gather as per-row DMA (vllm-torchtpu's fused EP MoE v3 idea).** Correct but 6x slower than XLA's gather at 15k rows on v6e (issue-bound at ~0.28 µs a row); suits decode-size batches only.

  **按行 DMA 做行 gather(vllm-torchtpu fused EP MoE v3 的思路)。** 结果正确,但 v6e 上 1.5 万行时比 XLA gather 慢 6 倍(每行发起约 0.28 µs);只适合解码那样的小批量。

- **Read the expert-sorted rows inside the grouped matmul (no separate gather).** The two row gathers around the experts cost ~1.0 ms a MoE layer at 15k rows on a v6e chip (floor ~0.4 ms). tokamax's `gmm_fuser` cannot fuse a gather (jax 0.12 fuser rules; contiguous LHS slices), so this needs our own kernel; tokamax's SparseCore `ragged_gather` is the other option, untested on v6e. At most ~0.4% of a 768p step.

  **在分组矩阵乘内部按索引读按专家排好的行(去掉单独的 gather)。** 专家前后两次行 gather 在 v6e 单 chip 1.5 万行时每层约 1.0 ms(下限约 0.4 ms)。tokamax 的 `gmm_fuser` 融不了 gather(jax 0.12 fuser 规则;LHS 是连续切片),得自己写 kernel;另一条路是 tokamax 的 SparseCore `ragged_gather`,v6e 上没测。最多占 768p 一步的约 0.4%。

- **gmm_v2 tile_m 512.** As tokamax's gmm_v2 does: 0.81 -> 0.73 ms per expert pair at 13,008 rows, bit-identical, but end to end at 768p within noise (denoise 104.7 vs 104.9 s), so not adopted.

  **gmm_v2 tile_m 改 512。** 照 tokamax 的 gmm_v2:13,008 行时每对专家矩阵乘 0.81 → 0.73 ms,输出逐位相同;但 768p 端到端在噪声内(去噪 104.7 对 104.9 秒),未采用。

## JAX, VideoGym-LTX-2.3

- **Overlap the per-block weight all-gathers with compute (v6e-8: up to ~27% of a step).** With 20 of 48 blocks resident, a 4-row forward is 481 ms: 314 ms compute, 126 ms all-gather with zero overlap, 35 ms idle. XLA keeps every all-gather synchronous (952 plain `all-gather` in the HLO) under every libtpu flag set tried; the all-gather continuation fusion flag is not supported on v6e. Gathering block i+1 inside block i by hand is the next thing to try. On v7x all 48 blocks fit each device, so there is nothing to gather.

  **把每块的权重 all-gather 与计算重叠(v6e-8:最多约占一步的 27%)。** 48 块中常驻 20 块时,4 行前向 481 ms:计算 314 ms,all-gather 126 ms 且与计算零重叠,空闲 35 ms。试过的各组 libtpu flag 下 XLA 都把 all-gather 保持为同步(HLO 中 952 个普通 `all-gather`);all-gather continuation fusion 的 flag 在 v6e 上不支持。下一步可试手动在第 i 块里提前 gather 第 i+1 块。v7x 上 48 块都能整份放进每个 device,没有需要 gather 的。

- **Weight gathers as collective-permute rings instead (tried, slower).** The permutes do run async (244 ms of overlap in two forwards), but a ring of them moves the weights at half the all-gather's speed: 613 ms (one way, 7 hops) / 580 ms (both ways, 4 hops, chunks written in place) against 481 ms. Code on the local branch `exp/ltx-ring-gather`.

  **改用 collective-permute 环做权重 gather(试过,更慢)。** permute 确实异步(两次前向重叠 244 ms),但环形搬权重只有 all-gather 一半的速度:613 ms(单向 7 跳)/ 580 ms(双向 4 跳、直接写入目标位置),对 481 ms。代码在本地分支 `exp/ltx-ring-gather`。

- **More resident blocks (24: end to end 15.39 -> 15.15 s at 29.4 GiB peak, 1.8 GB short of the limit; 28: 4-row forward 481 -> 449 ms).** The DiT alone peaks at 23.1 GiB a chip with 28 resident blocks, which does not leave room on a 32 GB v6e chip for Gemma-3 and the 129-frame VAE decode (27.6 GiB peak with 20).

  **更多常驻块(24 块:端到端 15.39 -> 15.15 秒,峰值 29.4 GiB,离上限只剩 1.8 GB;28 块:4 行前向 481 -> 449 ms)。** 常驻 28 块时仅 DiT 每 chip 峰值 23.1 GiB,32 GB 的 v6e chip 上放不下 Gemma-3 和 129 帧 VAE 解码(常驻 20 块时峰值 27.6 GiB)。

## How these were found / 怎么找到的

Host stalls show up as chip idle time between XLA ops in an xprof trace (`--profile`); `--profile-host` adds the CPU activity to the torch_tpu profiler, which records the host's Python stacks in the same trace, which names the work the chip is waiting for. Removing two per-step host syncs this way gave Pika ref2va 480p 9.2 -> 8.0 s (2026-11-24).

主机造成的停顿在 xprof 剖面(`--profile`)里表现为 XLA 算子之间的 chip 空闲;加 `--profile-host` 给 torch_tpu 剖面器加上 CPU activity,同一份剖面里会记下主机的 Python 调用栈,能看出 chip 在等什么。用这个办法去掉两处每步主机同步后,Pika ref2va 480p 9.2 -> 8.0 秒(2026-11-24)。
