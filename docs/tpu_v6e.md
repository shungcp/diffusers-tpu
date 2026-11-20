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

Start-up (2026-11-18, Pika `t2va` 480x832x121 and `ref2va`, fresh process): two changes, both on by default.

| Change / 改动 | `t2va` cold run 0 / 冷启动 | `t2va` warm run 0 / 暖启动 | `ref2va` cold / 冷启动 | DiT programs / 程序数 | Cache / 缓存 | Output / 输出 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Before / 之前 | 69–71 s | 20–22 s | 106–114 s | 15 | 2.45 GB | — |
| Plain-torch fake for the Pallas attention op (`DT_OURS_FAKE`) | −1.0 s | −0.5 s | — | — | — | bit-identical / 逐位一致 |
| Fixed timestep slots by row role (`DT_FIXED_TSLOTS`) | **42–45 s** | **14–16 s** | **80–86 s** | **7** | **1.41 GB** | DINO 0.977 (padding alone: 0.969) |

`jax_op`'s default fake kernel re-exports the Pallas kernel every time Dynamo traces a call. The first denoising step had one distinct timestep and later steps two, so the time embedding changed shape and torch_tpu compiled (and traced, in every new process) a second copy of each grouped DiT program; giving every row role its own slot keeps one program for all steps. The remaining warm-start gap to run 1 is Dynamo tracing of that program (~5.5 s; PyTorch's AOTAutograd cache flag makes no difference, torch_tpu's own cache already hits).

`jax_op` 的默认 fake 每次 Dynamo 追踪都会把 Pallas kernel 重新导出一遍。第 1 步只有一个不同的时间步、之后两个,时间嵌入形状不同,torch_tpu 给每个分组 DiT 程序多编(新进程里也多追踪)一份;按行的角色固定槽位后,所有步共用一个程序。暖启动与 run 1 剩下的差距是追踪这个程序(约 5.5 秒;打开 PyTorch 的 AOTAutograd 缓存开关没有变化,torch_tpu 自己的缓存已命中)。

Saving the compiled programs (`DT_AOT_PERSIST=<dir>`, off by default): the first process writes each grouped program with `aot_compile` + `save_compiled_function`; later processes load it with `torch.compiler.load_compiled_function`, skipping Dynamo and AOTAutograd. File names hash the shape, torch / torch_tpu versions, every `DT_*` setting, `XLA_FLAGS` / `LIBTPU_INIT_ARGS` and every source under `diffusers_tpu/backends` and `diffusers_tpu/kernels`, so a change writes new files instead of loading a stale program. A narrower hash once reused a stale DiT program whose output differed from a fresh compile (c770–c772); after upgrading anything else (diffusers, torch_tpu internals), clear the directory.

| Fresh process / 新进程 | run 0 | run 1 | Output / 输出 |
| :--- | :---: | :---: | :---: |
| `t2va`, cache only / 只有缓存 | 19.4–21.8 s | 7.7 s | — |
| `t2va`, saved programs / 读存档 | **8.0–9.9 s** | 7.6–8.0 s | bit-identical / 逐位一致 |
| `ref2va`, saved programs / 读存档 | **30.4–31.7 s** | 28.7 s | bit-identical / 逐位一致 |

Rank 0 writes one copy that every rank loads (the per-rank copies differed only in a few bytes). With saved programs the AOTAutograd entries (`TORCHINDUCTOR_CACHE_DIR`) and tier-3's large files are no longer read, so one shape needs only the program files plus tier-3's small files: `t2va` 0.14 + 0.23 = 0.37 GB, `ref2va` 0.12 + 0.43 = 0.54 GB (was 2.45 GB), warm run 0 unchanged and output bit-identical (c681).

把编好的程序存盘(`DT_AOT_PERSIST=<目录>`,默认关):第一个进程用 `aot_compile` + `save_compiled_function` 写下每个分组程序,之后的进程用 `torch.compiler.load_compiled_function` 读回,跳过 Dynamo 和 AOTAutograd。文件名包含形状、torch / torch_tpu 版本、全部 `DT_*` 设置、`XLA_FLAGS` / `LIBTPU_INIT_ARGS` 以及 `diffusers_tpu/backends`、`diffusers_tpu/kernels` 下全部源码的哈希,任一变化都会写新文件,不会读到旧程序。早先哈希范围较窄时,曾复用到一份输出与现编不同的旧 DiT 程序(c770–c772);升级其他依赖(diffusers、torch_tpu 内部)后请清空该目录。rank 0 写一份、所有 rank 共用(各 rank 的副本只差几个字节)。有了存档就不再读 AOTAutograd 条目和 tier-3 的大文件,一个形状只需存档加 tier-3 小文件:`t2va` 0.14 + 0.23 = 0.37 GB,`ref2va` 0.12 + 0.43 = 0.54 GB(原来 2.45 GB),暖启动 run 0 不变、输出逐位一致(c681)。

> **Why `Opt-2` (`DT_RESIDENT_TEXT=1`) is disabled on `v6e-8`**:
> Unlike `v7x-8` (95 GiB HBM per TensorCore device), each `v6e` chip has 32 GiB HBM. Keeping the `7.3 GiB` sharded Text Encoder resident alongside the `7.5 GiB` sharded DiT and VAE leaves insufficient HBM headroom for the `103,989`-token `768p` denoising activations, so `torchtpu_h3.py` enables `DT_RESIDENT_TEXT` automatically only on `v7x` (`TORCH_TPU_TOPOLOGY == "2,2,1,2"`).

> **`DT_POST_ATTN_PIPE=1` on `v6e-8`**: the two-slab post-attention pipeline (`52aba21`) runs out of HBM in the first 768p request (`RuntimeProgramAllocationFailure`), and the process then hangs instead of exiting; keep it off on `v6e`. On `v6e` the in-graph all-reduce is synchronous, so the slabs would not overlap anyway.

## 4. MiniMax-H3-Image on `v6e-8` / `v6e-8` 上的 MiniMax-H3-Image

Warm requests, end to end (text encoding, six DiT steps, decode). / 暖请求,端到端(文本编码、六步 DiT、解码)。

| Size / 尺寸 | JAX | `torchax` | `torch_tpu` |
| :--- | :---: | :---: | :---: |
| 2048x2048 t2i | **0.63 s** | 0.64 s | 0.69 s |
| 4736x3552 t2i | **2.77 s** | 2.80 s | 2.95 s |
| 2048x2048 edit, 1 reference / 编辑 | **1.15 s** | 1.18 s | 1.32 s |
| 4096x4096 edit, 1 reference / 编辑 | **3.38 s** | — | — |

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

Later fixes (2026-11-12), each found in a profile / 之后的修正(2026-11-12),每项都由剖面定位:

| Fix / 修正 | Effect (v6e-8) / 效果 |
| :--- | :--- |
| The head runs of the segmented modulation (text; an edit's reference rows) selected with a mask instead of written back: a `dynamic-update-slice` or `index_copy` broke the gated residual and the norm into three full-sequence passes a block / 分段调制的头部段用掩码选择,不再写回:`dynamic-update-slice` / `index_copy` 把门控残差与 norm 拆成每个 block 三次整序列遍历 | JAX 4736 3.05 -> 2.77 s; torchax 3.12 -> 2.80 s; edit denoise 1.24 -> 0.94 s |
| The tiled reference encode compiled, and split over the chips by a sharding constraint (a `device_put` inside `jit` left every chip encoding all tiles) / 参考图分块编码编译成一个程序,并用 sharding constraint 按 chip 切分(`jit` 里的 `device_put` 让每个 chip 编全部 tile) | edit 1.96 -> 1.15 s |
| torch_tpu: masked modulation in float32 (torch's per-op bfloat16 rounding sat in the FF projection's prologue), 128-lane RoPE, libtpu 0.0.52 (same attention kernel 5% faster), decoder attention as two matmuls and FF kept 2-D (261-row tiles made every 3-D view a copy) / torch_tpu:掩码调制用 float32、128 通道 RoPE、libtpu 0.0.52、解码器注意力改两次矩阵乘且 FF 保持二维 | 4736 3.36 -> 2.95 s; decode 0.131 -> 0.076 s |

What is left: about 1.05 s of the 2.77 s at 4736x3552 is the two all-reduces a block, measured at the link's rate (16,481 rows: 1.78 ms each, 174 GB/s bus); a reduce-scatter + all-gather of the same bytes is 2.3x slower, so a sequence-parallel stream does not pay on v6e.

剩下的:4736x3552 的 2.77 秒里约 1.05 秒是每个 block 的两次 all-reduce,实测已跑在链路带宽上(16,481 行每次 1.78 ms,总线 174 GB/s);同字节的 reduce-scatter + all-gather 慢 2.3 倍,所以 v6e 上序列并行残差流不划算。

Fewer chips / 更少的 chip: 4 chips with `--text-int8` 0.88 s / 4.22 s (edit 1.52 s; measured on a v6e-4); one v5e chip with `--offload --int8-weights --free-text-after` 11.4 s / 43.7 s (2048x2048 / 4736x3552).

4 个 chip 加 `--text-int8`:0.88 / 4.22 秒(编辑 1.52 秒;在 v6e-4 上实测);单个 v5e chip 加 `--offload --int8-weights --free-text-after`:11.4 / 43.7 秒(2048x2048 / 4736x3552)。

## 5. `torch_tpu` Compile Cache on Disk / `torch_tpu` 编译缓存的磁盘占用

Pika-H3-Flex `t2va` 480x832, 121 frames, 2 steps, 8 chips, torch_tpu 0.2.0 + libtpu 0.0.52 (2026-11-17). One shape writes 1.99 GB to `TORCHINDUCTOR_CACHE_DIR` (`aotautograd/`, 30 files) plus 0.46 GB of tier-3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`). The AOTAutograd directory holds each compiled graph eight times, once per rank: two copies of the same graph differ in 21 bytes (two 8-byte ids, one varint, `forward_time_taken_ns`), so each gets its own content-hash file name.

Pika-H3-Flex `t2va` 480x832、121 帧、2 步、8 chip,torch_tpu 0.2.0 + libtpu 0.0.52(2026-11-17)。一个形状在 `TORCHINDUCTOR_CACHE_DIR` 写 1.99 GB(`aotautograd/`,30 个文件),tier-3 另写 0.46 GB。AOTAutograd 目录里每个编译图存了 8 份,每个 rank 一份:同一个图的两份只差 21 字节(两个 8 字节 id、一个 varint、`forward_time_taken_ns`),内容哈希不同,所以各占一个文件名。

| Cache on disk / 磁盘缓存 | Size / 体积 | Warm run 0 / 暖启动 run 0 | run 1 / run 2 | Output / 输出 |
| :--- | :---: | :---: | :---: | :---: |
| None (cold) / 无(冷启动) | — | 62–64 s | 9.1 s | — |
| Tier-3 only / 只有 tier-3 | 0.46 GB | 50.0 s | 7.6 s | — |
| AOTAutograd (8 copies per graph) + tier-3 / 每图 8 份 | 2.45 GB | 19.4–20.5 s | 7.0 / 6.1 s | reference / 基准 |
| AOTAutograd, **one file per graph** + tier-3 / **每图只留 1 份** | **0.71 GB** | 18.6–19.0 s | 6.8 / 5.8 s | bit-identical / 逐位一致 |

Deleting all but one file in every `aotautograd/<key>/` directory keeps every rank's warm start hitting the cache (no files rewritten, same compile counts) at a third of the size. Tested on one shape and one host so far.

把每个 `aotautograd/<key>/` 目录删到只剩一个文件后,8 个 rank 暖启动照样命中(没有重新写文件,编译次数不变),体积降到三分之一。目前只在一个形状、一台主机上验证过。

`LIBTPU_INIT_ARGS` flags tried for cache size, each added alone to the defaults / 为缓存体积试过的 libtpu 参数(每个单独加在默认参数上):

| Flag / 参数 | AOTAutograd size / 体积 | Cold / Warm run 0 / 冷 / 暖 run 0 | Warm run 1 / 暖 run 1 | Output / 输出 |
| :--- | :---: | :---: | :---: | :---: |
| (defaults) / 默认 | 1.99 GB | 63.1 / 18.6 s | 6.6 s | — |
| `--xla_tpu_save_original_hlo_module=false` | 1.98 GB | 64.0 / 19.0 s | 7.3 s | bit-identical / 逐位一致 |
| `--xla_xprof_register_llo_debug_info=false` | 1.99 GB | 64.3 / 18.1 s | 6.9 s | bit-identical / 逐位一致 |
| `--xla_tpu_compiler_metadata_on_load=strip_and_compress` | 0.25 GB | 64.7 / 19.8 s | **15.2 s** | bit-identical / 逐位一致 |
| `--xla_tpu_keep_hlo_proto_literals_up_to=1` | already the default in libtpu 0.0.52 / 0.0.52 默认即为 1 | | | |

`strip_and_compress` shrinks the directory only because writes fail: an executable stripped at load can no longer be serialized (`Cannot serialize a TpuExecutable whose compiler metadata was stripped at load time`), so 26 of 32 AOTAutograd writes and 3,756 in-process cache writes are dropped, and each process recompiles 404–461 times instead of 45–94. Not usable on this path. The other two flags change neither size nor speed beyond run-to-run noise (±0.5–1 s).

`strip_and_compress` 让目录变小只是因为写入失败:加载时去掉元数据的可执行文件无法再序列化(`Cannot serialize a TpuExecutable whose compiler metadata was stripped at load time`),32 次 AOTAutograd 写入失败 26 次,进程内缓存写入失败 3756 次,每个进程编译次数从 45–94 涨到 404–461。这条路径不能用。另外两个参数对体积和速度都没有超出单次运行噪声(±0.5–1 秒)的影响。

Three shapes in one cache (480x832x121, 480x832x175, 544x960x121; 2026-11-18) / 三个形状共用一个缓存目录:

| Layout / 存法 | Size / 体积 | Notes / 说明 |
| :--- | :---: | :---: |
| As written (8 copies per graph) + tier-3 / 原样(每图 8 份) | 6.58 GB | — |
| One file per graph + tier-3 / 每图 1 份 | 1.75 GB (0.69 + 1.05) | warm run 0 18–23 s, bit-identical / 逐位一致 |
| One file per graph, no tier-3 / 每图 1 份、无 tier-3 | 0.69 GB | warm run 0 28.5–30.3 s (vs 20.7 s), rewrites 0.2 GB of tier-3 / 暖启动慢约 10 秒并回写 tier-3 |

A fresh process that only loads the cache from disk produces frames bit-identical to the process that compiled them (Pallas attention included).

只从磁盘加载缓存的新进程,输出与当场编译的进程逐位一致(含 Pallas 注意力)。

Filesystem dedup on the one-file-per-graph cache of the three shapes (ZFS kernel module 2.2.2 with 2.1.5 userland, on a loop device; all files read back with matching SHA-1) / 每图 1 份的三形状缓存上的文件系统去重:

| Filesystem / 文件系统 | Size / 体积 | Cold read / 冷读 |
| :--- | :---: | :---: |
| ext4 | 1.75 GB | 5.9 s |
| ZFS dedup, recordsize 4K / 16K / 128K | 1.68 / 1.65 / 1.76 GB | 12.8 / 7.0 / 6.5 s |
| ZFS zstd | 0.51 GB | 27.7 s |
| ZFS dedup 128K + zstd | 0.50 GB | 26.8 s |

Once the per-rank copies are gone, block dedup finds almost nothing across shapes (1.02–1.14x); zstd saves 3.4x but read back about 4.7x slower here. btrfs + duperemove 0.11.2 deduplicated nothing, not even two copies of a graph that differ in 21 bytes.

去掉每 rank 副本后,块去重在不同形状之间几乎找不到重复(1.02–1.14 倍);zstd 省 3.4 倍,但这里读回慢约 4.7 倍。btrfs + duperemove 0.11.2 没有去掉任何重复,连只差 21 字节的两份也没有。

ZFS dedup reaches the same size without deleting anything: the as-written cache (6.58 GB, 8 copies per graph) takes 1.68 GB with dedup at recordsize 16K (4.07x, 98.5k table entries, ~10–17 MB of RAM; cold read 18.1 s) and 0.51 GB with dedup 128K + zstd (cold read 35.5 s), so the copies need not be pruned by hand when the cache lives on ZFS. Pruning instead relies on any copy serving every rank, which held on the shapes tested but is not a documented torch_tpu guarantee.

ZFS 去重不删文件也能达到同样体积:原样写入的缓存(6.58 GB,每图 8 份)在记录大小 16K 去重下占 1.68 GB(4.07 倍,去重表 9.85 万条,约 10–17 MB 内存;冷读 18.1 秒),去重 128K + zstd 占 0.51 GB(冷读 35.5 秒)。缓存放在 ZFS 上就不必手动删副本。手动删除依赖"任意一份都能给所有 rank 用",在测过的形状上成立,但不是 torch_tpu 承诺的行为。

The two big programs are stored twice: tier-3's two >50 MB files are the same executables as the two big AOTAutograd entries (113.4 MB block-identical, shifted by 1,017 bytes, which is why block dedup misses them). Deleting those two tier-3 files keeps warm starts at 18.4–18.8 s with bit-identical output and they are not rewritten, so one shape needs 0.48 GB (0.25 AOTAutograd + 0.23 tier-3). No weights are baked in: in the dumped HLO of a 10-block program the 125 entry parameters include every weight shard (e.g. `bf16[896,5376]` x30) and all 282 constants are scalars; no 64-byte probe from the 1,262 checkpoint/adapter tensors occurs anywhere in the cache.

两个大程序存了两遍:tier-3 里两个 >50 MB 的文件与 AOTAutograd 的两个大条目是同一份可执行文件(113.4 MB 逐块相同,错开 1017 字节,所以块去重找不到)。删掉这两个 tier-3 文件后暖启动 18.4–18.8 秒、输出逐位一致、不会被写回,一个形状只需 0.48 GB(AOTAutograd 0.25 + tier-3 0.23)。权重没有被烧进程序:10-block 程序导出的 HLO 里 125 个入口参数包含全部权重分片(如 `bf16[896,5376]` 30 个),282 个常量全是标量;1262 个权重/adapter 张量取的 64 字节探针在缓存里一个都没出现。

