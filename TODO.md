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

## How these were found / 怎么找到的

Host stalls show up as chip idle time between XLA ops in an xprof trace (`--profile`); `--profile-host` adds the CPU activity to the torch_tpu profiler, which records the host's Python stacks in the same trace, which names the work the chip is waiting for. Removing two per-step host syncs this way gave Pika ref2va 480p 9.2 -> 8.0 s (2026-11-24).

主机造成的停顿在 xprof 剖面(`--profile`)里表现为 XLA 算子之间的 chip 空闲;加 `--profile-host` 给 torch_tpu 剖面器加上 CPU activity,同一份剖面里会记下主机的 Python 调用栈,能看出 chip 在等什么。用这个办法去掉两处每步主机同步后,Pika ref2va 480p 9.2 -> 8.0 秒(2026-11-24)。
