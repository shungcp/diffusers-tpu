"""Pallas attention kernels for TPU.

Adapted from AI-Hypercomputer/DiffusionServing (Apache 2.0, Google LLC).
Carried in-tree rather than using the stock
``jax.experimental.pallas.ops.tpu.splash_attention``, which is slower on
MiniMax-H3's 103,976-token packed sequence.

Each kernel at its own best block sizes, 8 chips, attention only, best of 3:

| kernel | best | at |
|---|---|---|
| this one | **148.9 ms** | `block_q=1024`, `block_kv=4096` |
| stock | **169.6 ms** | `block_q=4096`, `block_kv=1024` |

The stock kernel also rejected 20 of 37 block-size combinations (it requires
the block to divide the sequence length); this one accepts all of them.
"""
