"""Native JAX / TPU (and torchax / torch_tpu) implementation of Sand.ai MAGI-2.

Supports `sand-ai/MAGI-2-preview` (113.93B total / 5.95B active Multi-Head Latent MoE
DiT + 6.72B dense GQA Refiner + Qwen3.5-27B text encoder + Wan2.2 VAE encoder +
TurboVAED decoder + Oobleck stereo audio decoder) and is config-driven for drop-in
compatibility with future MAGI-2 flagship (480B) checkpoints.
"""
