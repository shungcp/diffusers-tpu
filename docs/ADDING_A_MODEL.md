# Adding a model

The repository is split into model-independent pieces and one package per model.

```
diffusers_tpu/
  kernels/            Pallas attention for TPU, model independent
    splash.py           dense splash attention: fixed-m softmax, runtime kv length (padded keys)
    sparse_splash.py    block-sparse variant driven by a (static or runtime) kv-block table
    mesh_attention.py   head-sharded attention over a device mesh
    block_sizes.py, fixed_m_bound.py, fp8_cast.py ...
  backends/           torch_tpu integration: routing torch attention to the kernels above,
                      AOT block compilation, per-chip process launch
  hub.py              offline lookup of a model in the Hugging Face cache
  models/
    qwen3_vl/         Qwen3-VL text and vision towers in flax (shared by models that use them)
    minimax_h3/       MiniMax-H3: JAX DiT/VAE/pipeline/server, torchax runner
```

## Recommended order for a new diffusers model

1. **torchax first.** If the model exists in `diffusers`, it can run on TPU through torchax with
   no model rewrite. Copy the structure of `models/minimax_h3/torchax_pipeline.py`: shard the
   transformer weights over the mesh, route `scaled_dot_product_attention` to
   `kernels.splash.attention` (with `kernels.mesh_attention` for head sharding), pad the sequence
   to a multiple of 128, and enable `--warmup` plus the persistent compilation cache.
2. **Measure before optimizing.** Profile one denoising step. On video DiTs attention usually
   dominates (68% of a step for MiniMax-H3 at 104k tokens); sweep `block_q` / `block_kv` for the
   new sequence length.
3. **Lossy options transfer directly** to any DiT with timestep conditioning and a temporal axis:
   - step cache: skip a DiT call while the timestep-embedding drift stays below a threshold;
   - temporal-window sparse attention: build a kv-block table where video rows see only nearby
     frames and text/audio/reference rows stay dense (see `Sparsity` in `models/minimax_h3/h3_tpu.py`
     and the table builder in `models/minimax_h3/jax_pipeline.py`).
   Always check quality (e.g. per-frame DINOv2 similarity against a lossless run) on several prompts:
   the threshold that is safe for one prompt is not safe for all.
4. **A JAX rewrite for the models that matter.** A flax/nnx rewrite of the DiT and VAE
   (`models/minimax_h3/jax_dit.py`, `jax_vae.py`) is faster still: weights become jit arguments,
   one compiled program per shape, no PyTorch dispatch. Reuse `models/qwen3_vl` if the model uses a
   Qwen3-VL encoder, and the server skeleton in `jax_server.py` (resident DiT, text-length buckets,
   runtime kv length so prompts of different length share one compiled program).
5. **Validate against the reference** pipeline output with the same initial noise before trusting
   any speed number.
