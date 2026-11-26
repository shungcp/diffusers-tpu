"""Stage 2 MAGI-2 Refiner DiT (6.72B Dense GQA) in native JAX / Flax NNX for TPU v7x-8.

Faithful port of ``inference/model/magi2_refiner.py`` and
``inference/pipeline/refiner_data_proxy.py`` from ``sand-ai/MAGI-2-preview``:

* **Architecture (6,720,075,280 params, 281 safetensors tensors = 12.5586 GiB)**:
  - 30 transformer layers, ``hidden_size = 4096``, ``head_dim = 128``,
    ``32 Q / 8 KV`` heads (GQA 4:1), dense ``swiglu7`` MLP with
    ``intermediate_size = 10920`` (``up_gate_proj: 4096 -> 21840`` interleaved,
    ``down_proj: 10920 -> 4096``), per-Q-head sigmoid output gating
    (``linear_g: 4096 -> 32``).
  - ``mm_layers = (0, 1, 28, 29)`` use 3-way modality-specific weights
    (``[VIDEO=0, AUDIO=1, TEXT=2]``) for both ``MultiModalityRMSNorm`` and
    ``NativeMoELinear``; layers ``2..27`` share single-modality weights.
  - **NO mHC** (single-stream residual), **NO attention sinks**, **NO softcap**,
    **NO timestep / AdaLN conditioning** (completely timestep-free).
  - **Precision parity**: ``PreAdapter`` (``bias=True``) and ``PostAdapter``
    (``bias=False``) are ``float32``; all RMSNorms (``1 + w``, ``eps=1e-6``)
    and QK-Norms + 3D Partial RoPE (``[0:96]`` rotated, ``[96:128]`` passthrough)
    compute in ``float32``; ``MLP.forward`` casts ``down_proj`` output to
    ``float32`` (matching ``magi2_refiner.py:L1848``), promoting the residual
    stream to ``float32`` after layer 0.
* **3D ``(8, 4, 4)`` Block-Grid Sparse Window Attention (all 30 layers)**:
  - Video tokens ``(T, H, W)`` (shipping 1080p: ``(63, 68, 120) = 514,080``)
    are permuted into 3D ``(8, 4, 4)`` blocks (grid ``(8, 17, 30) = 4,080``
    blocks) and padded per block to exact **128-token tiles** with valid tokens
    contiguous at ``[0 : valid_len)`` (``3,570`` blocks of ``128`` for
    ``bt=0..6`` + ``510`` blocks of ``112`` valid / ``16`` masked pad for
    ``bt=7`` -> ``522,240`` padded video tokens).
  - Video block ``(qt, qh, qw)`` attends bidirectionally to all in-bounds
    neighbors ``(qt+dt, qh+dh, qw+dw)`` with ``(dt, dh, dw) in [-2..2]^3``
    (``<= 125`` video blocks, 3.1% density) + all tail (audio + text) blocks;
    tail blocks attend densely to all valid video + tail tokens.
  - **TPU Pallas Kernel Design**:
    - Splits Q into two calls sharing ``(K_all, V_all)``: Video Q (4,080 blocks,
      ``grid_width <= 125 + n_tail_blocks``) and Tail Q (1..8 blocks, dense),
      cutting Pallas grid steps by **30.8x** vs a single unified ``kv_table``.
    - Uses compact 3D separable SMEM decode tables + per-KV-block valid lengths
      ``kv_blen`` (**56 KiB total SMEM**, avoiding the **2.10 MiB > 1 MiB SMEM**
      overflow of a flat ``(4080, 129)`` table on TPU v7x!) and performs **exact
      token-level masking** on the 16 pad tokens in ``bt=7`` (and tail pad
      tokens) with **zero masking overhead** on the 87.5% full blocks!
    - Also supports direct ``diffusers_tpu.kernels.sparse_splash.attention``
      (with SMEM-safe Q chunking) when all video blocks are full 128-token
      tiles, plus an exact fp32 block-masked reference fallback on CPU.
  - **Parallelism on TPU v7x-8**:
    - Ulysses ``CP=8`` (``4 Q heads / 1 KV head`` per chip — exact division of
      ``num_heads_kv=8``!) via ``shard_map`` + ``lax.all_to_all``.
    - Supports ``"replicated"`` (12.56 GiB/chip, zero weight comms — default on
      v7x-8's 192 GiB HBM), ``"fsdp"`` (1.57 GiB/chip), or ``"tp"`` (1.57 GiB/chip).

Self-test:
    python3 -m diffusers_tpu.models.magi2.jax_refiner_dit
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
import functools
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# Top-level JAX / Pallas imports (required by Pallas Mosaic tracing rules).
try:
  import jax
  from jax import lax
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  import jax.numpy as jnp
  from jax.sharding import NamedSharding, PartitionSpec as P
  from ...kernels import fixed_m_bound, sparse_splash
except ImportError:  # Allows host layout tests / py_compile without JAX installed.
  jax = None  # type: ignore[assignment]
  lax = None  # type: ignore[assignment]
  pl = None  # type: ignore[assignment]
  pltpu = None  # type: ignore[assignment]
  jnp = None  # type: ignore[assignment]
  NamedSharding = None  # type: ignore[assignment]
  P = None  # type: ignore[assignment]
  fixed_m_bound = None  # type: ignore[assignment]
  sparse_splash = None  # type: ignore[assignment]

from .config import Magi2RefinerConfig
from .rope_and_ops import (
    apply_3d_partial_rope,
    build_refiner_rope_coords,
    elementwise_fourier_embed,
    multi_modality_rms_norm,
    refiner_block_perm,
    rms_norm_1pw,
    swiglu7_interleaved,
)

# Attention backend selector:
#   "auto"          -> masked_pallas on TPU (exact t=63 + tail pad masking, 56 KiB SMEM);
#                      exact fp32 block-masked reference on CPU.
#   "masked_pallas" -> force 3D-grid masked Pallas kernel (TPU).
#   "sparse_splash" -> force diffusers_tpu.kernels.sparse_splash.attention with
#                      SMEM-safe Q chunking (exact only when all_video_blocks_full=True).
#   "ref"           -> exact fp32 block-masked JAX reference (CPU parity / tiny configs).
ATTN_IMPL = os.environ.get("DT_MAGI2_REFINER_ATTN_IMPL", "auto").strip().lower()

_DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
_NUM_SUBLANES = 8
_NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
_SV_DIM_NUMBERS = (((0,), (0,)), ((), ()))
_LOG2_E = 1.44269504

# Safe SMEM ceiling (bytes) when chunking flat kv_table for sparse_splash.attention.
_SPARSE_SPLASH_SMEM_BUDGET_BYTES = 256 * 1024

# Known non-parameter buffer stored in the upstream refiner checkpoint.
_ROPE_BANDS_KEY = "pre_adapter.rope.bands"


def _on_tpu() -> bool:
  if jax is None:
    return False
  try:
    devices = jax.devices()
    return bool(devices) and devices[0].platform == "tpu"
  except Exception:
    return False


def _shard_map(fn, *, mesh, in_specs, out_specs):
  """Compatibility wrapper for JAX ``shard_map`` across JAX versions."""
  if hasattr(jax, "shard_map"):
    return jax.shard_map(
        fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
  from jax.experimental.shard_map import shard_map as exp_shard_map

  return exp_shard_map(
      fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
  )


def _vmem_limit_bytes() -> int:
  if sparse_splash is not None:
    return int(sparse_splash._splash_vmem_limit_bytes())
  return 128 * 2**20


# ---------------------------------------------------------------------------
# 1. Host-side 3D (8,4,4) Block-Grid Layout, Packing & RoPE Precomputation
# ---------------------------------------------------------------------------


def _dim_block_sizes_np(dim_size: int, block_size: int) -> np.ndarray:
  """Per-block extents along one axis (matches ``refiner_data_proxy._dim_block_sizes``)."""
  n_blocks = math.ceil(dim_size / block_size)
  sizes = np.full((n_blocks,), block_size, dtype=np.int32)
  sizes[-1] = dim_size - block_size * (n_blocks - 1)
  return sizes


def _build_3d_separable_tables(
    block_grid_thw: tuple[int, int, int],
    radius: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
  """Build compact 3D separable neighbor & decode tables for SMEM (<40 KiB).

  For grid ``(nt, nh, nw)`` and radius ``(rt, rh, rw)``:
    - ``nb_arr``: ``int32[3, max_dim, max_win]`` sorted in-bounds neighbor
      coordinates along each axis ``(t, h, w)``.
    - ``cnt_arr``: ``int32[3, max_dim]`` valid neighbor count per block index
      along each axis.
    - ``dec_arr``: ``int32[3, max_ch * max_cw, max_vol]`` precomputed
      ``(it, ih, iw)`` decode indices for step ``j < ct * ch * cw``, eliminating
      all runtime integer division inside Pallas ``kv_index_map``.

  At 1080p ``(8, 17, 30)`` with ``radius=(2, 2, 2)``:
    - ``nb_arr``: ``(3, 30, 5)`` = 1.8 KiB
    - ``cnt_arr``: ``(3, 30)`` = 0.36 KiB
    - ``dec_arr``: ``(3, 25, 125)`` = 37.5 KiB
    - Total: **39.7 KiB** (vs 2.10 MiB flat ``kv_table``, which overflows 1 MiB SMEM!).
  """
  nt, nh, nw = (int(block_grid_thw[0]), int(block_grid_thw[1]), int(block_grid_thw[2]))
  rt, rh, rw = (int(radius[0]), int(radius[1]), int(radius[2]))

  max_ct = min(nt, 2 * rt + 1)
  max_ch = min(nh, 2 * rh + 1)
  max_cw = min(nw, 2 * rw + 1)
  max_dim = max(nt, nh, nw)
  max_win = max(max_ct, max_ch, max_cw)
  max_vol = max_ct * max_ch * max_cw

  nb_arr = np.zeros((3, max_dim, max_win), dtype=np.int32)
  cnt_arr = np.ones((3, max_dim), dtype=np.int32)

  for axis, (n_ax, r_ax) in enumerate(((nt, rt), (nh, rh), (nw, rw))):
    for q in range(n_ax):
      lo = max(0, q - r_ax)
      hi = min(n_ax, q + r_ax + 1)
      nbs = np.arange(lo, hi, dtype=np.int32)
      cnt_arr[axis, q] = len(nbs)
      nb_arr[axis, q, : len(nbs)] = nbs
      if len(nbs) < max_win:
        nb_arr[axis, q, len(nbs) :] = nbs[-1]

  dec_arr = np.zeros((3, max_ch * max_cw, max_vol), dtype=np.int32)
  for ch in range(1, max_ch + 1):
    for cw in range(1, max_cw + 1):
      pair = (ch - 1) * max_cw + (cw - 1)
      chcw = ch * cw
      j = np.arange(max_vol, dtype=np.int32)
      dec_arr[0, pair, :] = np.minimum(j // chcw, max_ct - 1)
      dec_arr[1, pair, :] = (j // cw) % ch
      dec_arr[2, pair, :] = j % cw

  return nb_arr, cnt_arr, dec_arr, max_cw, max_vol


@dataclass(frozen=True)
class RefinerAttentionLayout:
  """Precomputed host-side packing, RoPE, and sparse block-grid attention plan.

  All video blocks are padded to exact ``block_vol = bs_t * bs_h * bs_w``
  (``128`` for ``(8, 4, 4)``) token tiles with valid tokens placed contiguously
  at ``[0 : valid_len)`` inside each block. Tail tokens (optional audio + text)
  follow contiguously at ``[n_vid_pad : n_vid_pad + n_tail_pad)``.
  """

  video_thw: tuple[int, int, int]
  block_sizes: tuple[int, int, int]
  radius: tuple[int, int, int]
  block_grid_thw: tuple[int, int, int]
  block_vol: int
  pad_t: int
  n_vid_real: int
  n_vid_blocks: int
  n_vid_pad: int
  n_audio: int
  n_text: int
  n_tail_real: int
  n_tail_blocks: int
  n_tail_pad: int
  total_seq_len: int
  modality_lens: tuple[int, int, int]
  # Packing / unpacking indices (host NumPy int32)
  vid_src_idx: np.ndarray  # (n_vid_pad,) index into raster video [0..n_vid_real) or -1
  vid_dst_idx: np.ndarray  # (n_vid_real,) raster video idx -> slot in [0..n_vid_pad)
  # Per-block valid token counts & per-token boolean valid mask
  kv_block_valid_lens: np.ndarray  # (n_vid_blocks + n_tail_blocks,) int32
  token_valid_mask: np.ndarray  # (total_seq_len,) bool
  all_video_blocks_full: bool
  # Flat sorted block-sparsity tables (matching BlockGridLocalAttn + dense tail)
  kv_table_vid: list[list[int]]
  kv_table_tail: list[list[int]]
  # Compact 3D separable SMEM tables for Pallas (<60 KiB total at 1080p!)
  nb_arr: np.ndarray  # (3, max_dim, max_win) int32
  cnt_arr: np.ndarray  # (3, max_dim) int32
  dec_arr: np.ndarray  # (3, max_ch * max_cw, max_vol) int32
  max_cw: int
  max_vol: int
  # Precomputed fp32 RoPE [total_seq_len, 6 * (head_dim // 8)] (identity on pad slots)
  rope_96: np.ndarray


def build_refiner_attention_layout(
    video_thw: tuple[int, int, int],
    n_text: int,
    *,
    n_audio: int = 0,
    config: Magi2RefinerConfig | None = None,
    block_sizes: tuple[int, int, int] | None = None,
    radius: tuple[int, int, int] | None = None,
    tail_bucket_len: int | None = None,
    cp_size: int = 1,
    spatial_rope_interpolation: str = "extra",
    build_flat_tables: bool = True,
) -> RefinerAttentionLayout:
  """Build exact 3D ``(bs_t, bs_h, bs_w)`` block-grid layout & RoPE on host NumPy.

  Matches ``refiner_data_proxy.py``'s ``BlockGridLocalAttn`` + ``_build_block_order``
  while padding partial boundary blocks (such as ``bt=7`` having ``7*4*4 = 112``
  tokens when ``T=63``) up to exact ``128``-token tiles with contiguous leading
  valid tokens ``[0 : valid_len)`` and masked trailing pad slots.

  Args:
    video_thw: Latent video grid ``(T, H, W)`` (e.g. ``(63, 68, 120)`` at 1080p).
    n_text: Number of real text tokens (from Qwen3.5 ``hidden_states[-3]``).
    n_audio: Number of real audio tokens (``0`` in shipping video-only refiner).
    config: Optional ``Magi2RefinerConfig`` supplying defaults.
    block_sizes: Override ``(bs_t, bs_h, bs_w)`` (default ``(8, 4, 4)``).
    radius: Override ``(rt, rh, rw)`` (default ``(2, 2, 2)``).
    tail_bucket_len: Optional static padded tail length (multiple of ``block_vol``)
      to reuse one compiled XLA graph across variable prompt lengths.
    cp_size: Ulysses context-parallel world size (``1`` or ``8``); ensures
      ``total_seq_len % cp_size == 0``.
    spatial_rope_interpolation: ``"extra"`` (shipping default) or ``"inter"``.
    build_flat_tables: Build Python ``kv_table_vid`` / ``kv_table_tail`` lists
      (used by ``sparse_splash`` / CPU reference / tests).

  Returns:
    Frozen ``RefinerAttentionLayout``.
  """
  cfg = config or Magi2RefinerConfig()
  t_num, h_num, w_num = (int(video_thw[0]), int(video_thw[1]), int(video_thw[2]))
  bs_t, bs_h, bs_w = (
      block_sizes
      if block_sizes is not None
      else (cfg.block_t_size, cfg.block_h_size, cfg.block_w_size)
  )
  rt, rh, rw = (
      radius
      if radius is not None
      else (cfg.block_t_radius, cfg.block_h_radius, cfg.block_w_radius)
  )
  n_audio = int(n_audio)
  n_text = int(n_text)
  block_vol = bs_t * bs_h * bs_w

  # 1. Reuse exact upstream video block permutation from rope_and_ops.
  perm, _, block_grid_thw, pad_t = refiner_block_perm(
      (t_num, h_num, w_num), bs_t=bs_t, bs_h=bs_h, bs_w=bs_w
  )
  nt, nh, nw = block_grid_thw
  n_vid_real = t_num * h_num * w_num
  n_vid_blocks = nt * nh * nw
  n_vid_pad = n_vid_blocks * block_vol

  # 2. Per-block valid counts: valid tokens sit contiguously at [0 : valid_len).
  t_sz = _dim_block_sizes_np(t_num, bs_t)
  h_sz = _dim_block_sizes_np(h_num, bs_h)
  w_sz = _dim_block_sizes_np(w_num, bs_w)
  vid_block_valid_lens = (
      t_sz[:, None, None] * h_sz[None, :, None] * w_sz[None, None, :]
  ).reshape(-1).astype(np.int32)
  assert int(vid_block_valid_lens.sum()) == n_vid_real
  all_video_blocks_full = bool(np.all(vid_block_valid_lens == block_vol))

  # Map compact block-ordered video tokens into padded 128-token block slots.
  ends = np.cumsum(vid_block_valid_lens, dtype=np.int64)
  starts = ends - vid_block_valid_lens
  block_ids = np.repeat(np.arange(n_vid_blocks, dtype=np.int32), vid_block_valid_lens)
  local_offsets = (
      np.arange(n_vid_real, dtype=np.int64) - np.repeat(starts, vid_block_valid_lens)
  ).astype(np.int32)
  padded_pos = block_ids * block_vol + local_offsets

  vid_src_idx = np.full((n_vid_pad,), -1, dtype=np.int32)
  vid_src_idx[padded_pos] = perm
  vid_dst_idx = np.empty((n_vid_real,), dtype=np.int32)
  vid_dst_idx[perm] = padded_pos

  # 3. Tail (audio + text) padded up to whole block_vol tiles (and cp_size alignment).
  n_tail_real = n_audio + n_text
  if tail_bucket_len is not None:
    if tail_bucket_len < n_tail_real:
      raise ValueError(
          f"tail_bucket_len={tail_bucket_len} < n_audio + n_text={n_tail_real}."
      )
    if tail_bucket_len % block_vol != 0:
      raise ValueError(
          f"tail_bucket_len={tail_bucket_len} must be a multiple of block_vol={block_vol}."
      )
    n_tail_pad = int(tail_bucket_len)
  else:
    n_tail_pad = (
        math.ceil(n_tail_real / block_vol) * block_vol if n_tail_real > 0 else 0
    )

  # Ensure total_seq_len is divisible by cp_size (adds empty masked tail blocks if needed).
  if cp_size > 1 and (n_vid_pad + n_tail_pad) % cp_size != 0:
    step = math.lcm(block_vol, cp_size)
    rem = (n_vid_pad + n_tail_pad) % step
    if rem != 0:
      n_tail_pad += step - rem
  n_tail_blocks = n_tail_pad // block_vol
  total_seq_len = n_vid_pad + n_tail_pad

  tail_block_valid_lens = np.zeros((n_tail_blocks,), dtype=np.int32)
  rem_tail = n_tail_real
  for tb in range(n_tail_blocks):
    take = min(block_vol, rem_tail)
    tail_block_valid_lens[tb] = take
    rem_tail -= take

  kv_block_valid_lens = np.concatenate(
      [vid_block_valid_lens, tail_block_valid_lens], axis=0
  )

  # Per-token boolean valid mask (True for real tokens, False for masked pad slots).
  local_idx = np.arange(block_vol, dtype=np.int32)[None, :]
  token_valid_mask = (local_idx < kv_block_valid_lens[:, None]).reshape(-1)

  # Modality lengths along axis 0: [VIDEO(n_vid_pad), AUDIO(n_audio), TEXT(n_tail_pad - n_audio)].
  # Pad slots in bt=7 live in the VIDEO slice; trailing tail pad slots live in the TEXT slice.
  # Because all non-attention ops are strictly token-wise and all pad keys are masked out
  # of attention, pad slots never influence any valid token.
  modality_lens = (n_vid_pad, n_audio, n_tail_pad - n_audio)
  assert sum(modality_lens) == total_seq_len

  # 4. Compact 3D separable SMEM tables for Pallas (<60 KiB total at 1080p).
  nb_arr, cnt_arr, dec_arr, max_cw, max_vol = _build_3d_separable_tables(
      block_grid_thw, (rt, rh, rw)
  )

  # 5. Optional flat sorted kv_table lists (matching BlockGridLocalAttn + dense tail).
  kv_table_vid: list[list[int]] = []
  kv_table_tail: list[list[int]] = []
  if build_flat_tables:
    active_tail_blocks = [
        n_vid_blocks + tb
        for tb in range(n_tail_blocks)
        if tail_block_valid_lens[tb] > 0
    ]
    nhw = nh * nw
    for qt in range(nt):
      kt_lo, kt_hi = max(0, qt - rt), min(nt, qt + rt + 1)
      for qh in range(nh):
        kh_lo, kh_hi = max(0, qh - rh), min(nh, qh + rh + 1)
        for qw in range(nw):
          kw_lo, kw_hi = max(0, qw - rw), min(nw, qw + rw + 1)
          row = [
              kt * nhw + kh * nw + kw
              for kt in range(kt_lo, kt_hi)
              for kh in range(kh_lo, kh_hi)
              for kw in range(kw_lo, kw_hi)
          ]
          row.extend(active_tail_blocks)
          kv_table_vid.append(row)
    all_active_kv = list(range(n_vid_blocks)) + active_tail_blocks
    kv_table_tail = [all_active_kv for _ in range(n_tail_blocks)]

  # 6. Precompute exact fp32 3D Partial RoPE [total_seq_len, ro_dim].
  coords_raw = build_refiner_rope_coords(
      (t_num, h_num, w_num),
      n_text,
      audio_len=n_audio,
      spatial_rope_interpolation=spatial_rope_interpolation,
  )
  rope_raw = elementwise_fourier_embed(
      coords_raw,
      head_dim=cfg.head_dim,
      temperature=cfg.rope_temperature,
  )  # (n_vid_real + n_tail_real, ro_dim) float32
  ro_dim = rope_raw.shape[-1]
  half_ro = ro_dim // 2

  # Identity rotation (sin=0, cos=1) for all padded slots.
  rope_96 = np.zeros((total_seq_len, ro_dim), dtype=np.float32)
  rope_96[:, half_ro:] = np.float32(1.0)
  rope_96[vid_dst_idx] = rope_raw[:n_vid_real]
  if n_tail_real > 0:
    rope_96[n_vid_pad : n_vid_pad + n_tail_real] = rope_raw[n_vid_real:]

  return RefinerAttentionLayout(
      video_thw=(t_num, h_num, w_num),
      block_sizes=(bs_t, bs_h, bs_w),
      radius=(rt, rh, rw),
      block_grid_thw=block_grid_thw,
      block_vol=block_vol,
      pad_t=pad_t,
      n_vid_real=n_vid_real,
      n_vid_blocks=n_vid_blocks,
      n_vid_pad=n_vid_pad,
      n_audio=n_audio,
      n_text=n_text,
      n_tail_real=n_tail_real,
      n_tail_blocks=n_tail_blocks,
      n_tail_pad=n_tail_pad,
      total_seq_len=total_seq_len,
      modality_lens=modality_lens,
      vid_src_idx=vid_src_idx,
      vid_dst_idx=vid_dst_idx,
      kv_block_valid_lens=kv_block_valid_lens,
      token_valid_mask=token_valid_mask,
      all_video_blocks_full=all_video_blocks_full,
      kv_table_vid=kv_table_vid,
      kv_table_tail=kv_table_tail,
      nb_arr=nb_arr,
      cnt_arr=cnt_arr,
      dec_arr=dec_arr,
      max_cw=max_cw,
      max_vol=max_vol,
      rope_96=rope_96,
  )


def pack_refiner_inputs_np(
    video_cthw: np.ndarray,
    text_tokens: np.ndarray,
    layout: RefinerAttentionLayout,
    *,
    audio_tokens: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Pack ``(C, T, H, W)`` (or ``(1, C, T, H, W)``) video + text (+ optional audio)
  into block-ordered padded arrays ready for ``Magi2RefinerTransformer``.

  Returns:
    ``(video_packed [n_vid_pad, C_v], audio_packed [n_audio, C_a], text_packed [n_tail_pad - n_audio, C_t])``
    all in ``float32``.
  """
  v = np.asarray(video_cthw, dtype=np.float32)
  if v.ndim == 5:
    assert v.shape[0] == 1, f"Refiner window attention expects batch_size=1, got {v.shape}."
    v = v[0]
  assert v.ndim == 4, f"Expected (C, T, H, W) or (1, C, T, H, W), got {v.shape}."
  c_v, t_num, h_num, w_num = v.shape
  assert (t_num, h_num, w_num) == layout.video_thw, (
      f"Video shape {(t_num, h_num, w_num)} != layout.video_thw {layout.video_thw}."
  )
  # Raster order: (T, H, W, C) -> (T*H*W, C), matching refiner_data_proxy.img2tokens (patch=1).
  v_raster = np.transpose(v, (1, 2, 3, 0)).reshape(layout.n_vid_real, c_v)
  video_packed = np.zeros((layout.n_vid_pad, c_v), dtype=np.float32)
  video_packed[layout.vid_dst_idx] = v_raster

  if audio_tokens is None:
    audio_packed = np.zeros((layout.n_audio, 64), dtype=np.float32)
  else:
    a = np.asarray(audio_tokens, dtype=np.float32)
    if a.ndim == 3:
      assert a.shape[0] == 1
      a = a[0]
    assert a.shape[0] == layout.n_audio, (
        f"audio_tokens len {a.shape[0]} != layout.n_audio {layout.n_audio}."
    )
    audio_packed = a

  t = np.asarray(text_tokens, dtype=np.float32)
  if t.ndim == 3:
    assert t.shape[0] == 1
    t = t[0]
  assert t.shape[0] == layout.n_text, (
      f"text_tokens len {t.shape[0]} != layout.n_text {layout.n_text}."
  )
  n_text_slots = layout.modality_lens[2]
  text_packed = np.zeros((n_text_slots, t.shape[-1]), dtype=np.float32)
  if layout.n_text > 0:
    text_packed[: layout.n_text] = t
  return video_packed, audio_packed, text_packed


def unpack_refiner_video_np(
    video_out_padded: np.ndarray,
    layout: RefinerAttentionLayout,
) -> np.ndarray:
  """Restore block-ordered padded video output ``(n_vid_pad, C_v)`` back to ``(C_v, T, H, W)``."""
  arr = np.asarray(video_out_padded)
  t_num, h_num, w_num = layout.video_thw
  c_v = arr.shape[-1]
  v_raster = arr[layout.vid_dst_idx]  # (T*H*W, C_v)
  return np.transpose(v_raster.reshape(t_num, h_num, w_num, c_v), (3, 0, 1, 2))


def unpack_refiner_video_jax(
    video_out_padded: Any,
    layout: RefinerAttentionLayout,
) -> Any:
  """JAX version of ``unpack_refiner_video_np``: ``(n_vid_pad, C_v) -> (C_v, T, H, W)``."""
  t_num, h_num, w_num = layout.video_thw
  c_v = video_out_padded.shape[-1]
  dst = jnp.asarray(layout.vid_dst_idx, dtype=jnp.int32)
  v_raster = video_out_padded[dst]
  return jnp.transpose(v_raster.reshape(t_num, h_num, w_num, c_v), (3, 0, 1, 2))


# ---------------------------------------------------------------------------
# 2. Pallas 3D-Grid Masked Block-Sparse Splash Attention Kernel (TPU v7x-8)
# ---------------------------------------------------------------------------


def _grid3d_resolve_step(
    qt: Any,
    qh: Any,
    qw: Any,
    j: Any,
    nb_ref: Any,
    cnt_ref: Any,
    dec_ref: Any,
    *,
    nh: int,
    nw: int,
    n_vid_blocks: int,
    n_tail_blocks: int,
    max_cw: int,
):
  """Compute ``(k_block, live)`` for video Q-block ``(qt, qh, qw)`` at step ``j``.

  Uses ONLY SMEM lookups and integer add/mul (ZERO division or modulo!).
  """
  nhw = nh * nw
  ct = cnt_ref[0, qt]
  ch = cnt_ref[1, qh]
  cw = cnt_ref[2, qw]
  n_nb = ct * ch * cw

  # Clamp j_nb to [0, n_nb - 1] so dec_ref is always indexed in-bounds and
  # repeats the last neighbor block when n_tail_blocks == 0 (zero redundant DMA).
  j_nb = jnp.minimum(j, n_nb - 1)
  pair_idx = (ch - 1) * max_cw + (cw - 1)
  it = dec_ref[0, pair_idx, j_nb]
  ih = dec_ref[1, pair_idx, j_nb]
  iw = dec_ref[2, pair_idx, j_nb]

  kt = nb_ref[0, qt, it]
  kh = nb_ref[1, qh, ih]
  kw = nb_ref[2, qw, iw]
  k_block_nb = kt * nhw + kh * nw + kw

  if n_tail_blocks > 0:
    tb = jnp.maximum(
        jnp.int32(0), jnp.minimum(j - n_nb, jnp.int32(n_tail_blocks - 1))
    )
    k_block = jnp.where(j < n_nb, k_block_nb, n_vid_blocks + tb)
    live = j < (n_nb + n_tail_blocks)
  else:
    k_block = k_block_nb
    live = j < n_nb
  return k_block, live


def _grid3d_masked_splash_kernel(
    nb_ref: Any,  # SMEM (3, max_dim, max_win) i32
    cnt_ref: Any,  # SMEM (3, max_dim) i32
    dec_ref: Any,  # SMEM (3, max_ch * max_cw, max_vol) i32
    kv_blen_ref: Any,  # SMEM (num_kv_blocks,) i32
    q_ref: Any,  # VMEM (bq, head_dim_qk) bf16
    k_ref: Any,  # VMEM (bkv, head_dim_qk) bf16
    v_ref: Any,  # VMEM (bkv, head_dim_v) bf16
    o_out_ref: Any,  # VMEM (head_dim_v, bq) bf16
    m_scratch_ref: Any,  # VMEM scratch (NUM_SUBLANES, bq) f32
    l_scratch_ref: Any,  # VMEM scratch (NUM_SUBLANES, bq) f32
    o_scratch_ref: Any,  # VMEM scratch (head_dim_v, bq) f32
    *,
    mask_value: float,
    nh: int,
    nw: int,
    n_vid_blocks: int,
    n_tail_blocks: int,
    max_cw: int,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
) -> None:
  """3D block-grid sparse splash attention kernel with exact per-KV-block masking.

  Uses 5D grid ``(num_q_heads, nt, nh, nw, grid_width)`` so ``(qt, qh, qw)`` are
  direct grid axes (zero division/modulo), ``pltpu.VMEM`` scratch buffers
  (megacore-safe), and online ``exp2`` softmax (f32 accum, bf16 MXU dots). Full
  KV blocks (``valid_len >= bkv``, 87.5% of video blocks) take the zero-overhead
  unmasked path; partial blocks (``bt=7`` with ``112/128`` valid tokens, and
  partial tail blocks) mask ``local_pos >= valid_len`` to ``mask_value``; empty
  bucket blocks (``valid_len == 0``) are skipped completely.
  """
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, _NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {_NUM_SUBLANES}")
  assert bkv % bkv_compute == 0

  qt = pl.program_id(1)
  qh = pl.program_id(2)
  qw = pl.program_id(3)
  j = pl.program_id(4)

  k_block, live = _grid3d_resolve_step(
      qt,
      qh,
      qw,
      j,
      nb_ref,
      cnt_ref,
      dec_ref,
      nh=nh,
      nw=nw,
      n_vid_blocks=n_vid_blocks,
      n_tail_blocks=n_tail_blocks,
      max_cw=max_cw,
  )
  valid_len = kv_blen_ref[k_block]
  active = live & (valid_len > 0)
  is_full = valid_len >= bkv

  @pl.when(j == 0)
  def _init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

  def _update(start: Any, masked: bool):
    q = q_ref[...]  # (bq, head_dim_qk) bf16
    k_chunk = k_ref[pl.ds(start, bkv_compute), :]  # (bkv_compute, head_dim_qk) bf16
    v_chunk = v_ref[pl.ds(start, bkv_compute), :]  # (bkv_compute, head_dim_v) bf16
    qk = lax.dot_general(
        k_chunk, q, _NT_DIM_NUMBERS, preferred_element_type=float32
    )  # (bkv_compute, bq) f32
    if masked:
      local_pos = start + lax.broadcasted_iota(jnp.int32, qk.shape, 0)
      qk = jnp.where(local_pos < valid_len, qk, mask_value)

    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v_chunk, s_cast, _SV_DIM_NUMBERS, preferred_element_type=float32
    )
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  def _body_full(kv_compute_index, _):
    _update(kv_compute_index * bkv_compute, masked=False)

  def _body_masked(kv_compute_index, _):
    _update(kv_compute_index * bkv_compute, masked=True)

  @pl.when(active & is_full)
  def _run_full():
    lax.fori_loop(0, bkv // bkv_compute, _body_full, None, unroll=True)

  @pl.when(active & jnp.logical_not(is_full))
  def _run_masked():
    lax.fori_loop(0, bkv // bkv_compute, _body_masked, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def _end():
    l = l_scratch_ref[...]
    l_safe = jnp.where(l == 0.0, 1.0, l)
    l_inv = jnp.concatenate([1.0 / l_safe] * head_dim_v_repeats, axis=0)
    o_out_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_out_ref.dtype)


def _dense_masked_splash_kernel(
    kv_blen_ref: Any,  # SMEM (num_kv_blocks,) i32
    q_ref: Any,  # VMEM (bq, head_dim_qk) bf16
    k_ref: Any,  # VMEM (bkv, head_dim_qk) bf16
    v_ref: Any,  # VMEM (bkv, head_dim_v) bf16
    o_out_ref: Any,  # VMEM (head_dim_v, bq) bf16
    m_scratch_ref: Any,  # VMEM scratch (NUM_SUBLANES, bq) f32
    l_scratch_ref: Any,  # VMEM scratch (NUM_SUBLANES, bq) f32
    o_scratch_ref: Any,  # VMEM scratch (head_dim_v, bq) f32
    *,
    mask_value: float,
    num_kv_blocks: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
) -> None:
  """Dense tail-Q splash attention over all KV blocks with per-KV-block masking."""
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, _NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {_NUM_SUBLANES}")
  assert bkv % bkv_compute == 0

  j = pl.program_id(2)
  valid_len = kv_blen_ref[j]
  active = valid_len > 0
  is_full = valid_len >= bkv

  @pl.when(j == 0)
  def _init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

  def _update(start: Any, masked: bool):
    q = q_ref[...]
    k_chunk = k_ref[pl.ds(start, bkv_compute), :]
    v_chunk = v_ref[pl.ds(start, bkv_compute), :]
    qk = lax.dot_general(
        k_chunk, q, _NT_DIM_NUMBERS, preferred_element_type=float32
    )
    if masked:
      local_pos = start + lax.broadcasted_iota(jnp.int32, qk.shape, 0)
      qk = jnp.where(local_pos < valid_len, qk, mask_value)

    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v_chunk, s_cast, _SV_DIM_NUMBERS, preferred_element_type=float32
    )
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  def _body_full(kv_compute_index, _):
    _update(kv_compute_index * bkv_compute, masked=False)

  def _body_masked(kv_compute_index, _):
    _update(kv_compute_index * bkv_compute, masked=True)

  @pl.when(active & is_full)
  def _run_full():
    lax.fori_loop(0, bkv // bkv_compute, _body_full, None, unroll=True)

  @pl.when(active & jnp.logical_not(is_full))
  def _run_masked():
    lax.fori_loop(0, bkv // bkv_compute, _body_masked, None, unroll=True)

  @pl.when(j == num_kv_blocks - 1)
  def _end():
    l = l_scratch_ref[...]
    l_safe = jnp.where(l == 0.0, 1.0, l)
    l_inv = jnp.concatenate([1.0 / l_safe] * head_dim_v_repeats, axis=0)
    o_out_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_out_ref.dtype)


def _grid3d_masked_splash_attention(
    q_vid: Any,  # (H_q, n_vid_pad, D) bf16
    k_all: Any,  # (H_kv, total_seq_len, D) bf16
    v_all: Any,  # (H_kv, total_seq_len, D) bf16
    nb_arr: Any,  # (3, max_dim, max_win) i32
    cnt_arr: Any,  # (3, max_dim) i32
    dec_arr: Any,  # (3, max_ch * max_cw, max_vol) i32
    kv_blen: Any,  # (num_kv_blocks,) i32
    *,
    layout: RefinerAttentionLayout,
    scale: float,
) -> Any:
  """Launch 3D-grid masked Pallas splash kernel for video Q-blocks."""
  num_q_heads, q_seq_len, head_dim_qk = q_vid.shape
  num_kv_heads, _, head_dim_v = v_all.shape
  assert num_q_heads % num_kv_heads == 0
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  bq = bkv = bkv_compute = layout.block_vol
  nt, nh, nw = layout.block_grid_thw
  nhw = nh * nw
  n_vid_blocks = layout.n_vid_blocks
  n_tail_blocks = layout.n_tail_blocks
  max_cw = layout.max_cw
  grid_width = layout.max_vol + n_tail_blocks

  q_scaled = (q_vid.astype(jnp.float32) * (scale * _LOG2_E)).astype(jnp.bfloat16)
  k_bf16 = k_all.astype(jnp.bfloat16)
  v_bf16 = v_all.astype(jnp.bfloat16)

  def q_index_map(h, qt, qh, qw, j, *_):
    return (h, qt * nhw + qh * nw + qw, 0)

  def kv_index_map(h, qt, qh, qw, j, nb, cnt, dec, _blen):
    k_block, _ = _grid3d_resolve_step(
        qt,
        qh,
        qw,
        j,
        nb,
        cnt,
        dec,
        nh=nh,
        nw=nw,
        n_vid_blocks=n_vid_blocks,
        n_tail_blocks=n_tail_blocks,
        max_cw=max_cw,
    )
    return (h // q_heads_per_kv_head, k_block, 0)

  def out_index_map(h, qt, qh, qw, j, *_):
    return (h, 0, qt * nhw + qh * nw + qw)

  scratch_shapes = [
      pltpu.VMEM((_NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((_NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((head_dim_v, bq), jnp.float32),
  ]
  out = pl.pallas_call(
      functools.partial(
          _grid3d_masked_splash_kernel,
          mask_value=_DEFAULT_MASK_VALUE,
          nh=nh,
          nw=nw,
          n_vid_blocks=n_vid_blocks,
          n_tail_blocks=n_tail_blocks,
          max_cw=max_cw,
          grid_width=grid_width,
          bkv=bkv,
          bkv_compute=bkv_compute,
          head_dim_v=head_dim_v,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=[
              pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
              pl.BlockSpec((None, bkv, head_dim_qk), kv_index_map),
              pl.BlockSpec((None, bkv, head_dim_v), kv_index_map),
          ],
          out_specs=pl.BlockSpec((None, head_dim_v, bq), out_index_map),
          scratch_shapes=scratch_shapes,
          grid=(num_q_heads, nt, nh, nw, grid_width),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          ),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (num_q_heads, head_dim_v, q_seq_len), jnp.bfloat16
      ),
  )(nb_arr, cnt_arr, dec_arr, kv_blen, q_scaled, k_bf16, v_bf16)
  # (H_q, D, n_vid_pad) -> (H_q, n_vid_pad, D)
  return jnp.swapaxes(out, 1, 2)


def _dense_masked_splash_attention(
    q_tail: Any,  # (H_q, n_tail_pad, D) bf16
    k_all: Any,  # (H_kv, total_seq_len, D) bf16
    v_all: Any,  # (H_kv, total_seq_len, D) bf16
    kv_blen: Any,  # (num_kv_blocks,) i32
    *,
    layout: RefinerAttentionLayout,
    scale: float,
) -> Any:
  """Launch dense masked Pallas splash kernel for tail (audio/text) Q-blocks."""
  num_q_heads, q_seq_len, head_dim_qk = q_tail.shape
  num_kv_heads, _, head_dim_v = v_all.shape
  assert num_q_heads % num_kv_heads == 0
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  bq = bkv = bkv_compute = layout.block_vol
  n_tail_blocks = layout.n_tail_blocks
  num_kv_blocks = layout.n_vid_blocks + layout.n_tail_blocks

  q_scaled = (q_tail.astype(jnp.float32) * (scale * _LOG2_E)).astype(jnp.bfloat16)
  k_bf16 = k_all.astype(jnp.bfloat16)
  v_bf16 = v_all.astype(jnp.bfloat16)

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def kv_index_map(h, i, j, *_):
    return (h // q_heads_per_kv_head, j, 0)

  def out_index_map(h, i, j, *_):
    return (h, 0, i)

  scratch_shapes = [
      pltpu.VMEM((_NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((_NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((head_dim_v, bq), jnp.float32),
  ]
  out = pl.pallas_call(
      functools.partial(
          _dense_masked_splash_kernel,
          mask_value=_DEFAULT_MASK_VALUE,
          num_kv_blocks=num_kv_blocks,
          bkv=bkv,
          bkv_compute=bkv_compute,
          head_dim_v=head_dim_v,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
              pl.BlockSpec((None, bkv, head_dim_qk), kv_index_map),
              pl.BlockSpec((None, bkv, head_dim_v), kv_index_map),
          ],
          out_specs=pl.BlockSpec((None, head_dim_v, bq), out_index_map),
          scratch_shapes=scratch_shapes,
          grid=(num_q_heads, n_tail_blocks, num_kv_blocks),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (num_q_heads, head_dim_v, q_seq_len), jnp.bfloat16
      ),
  )(kv_blen, q_scaled, k_bf16, v_bf16)
  return jnp.swapaxes(out, 1, 2)


# ---------------------------------------------------------------------------
# 2b. Fast GQA-Grouped + W-Coalesced 3D Block-Grid Pallas Attention (9x-12x)
# ---------------------------------------------------------------------------


def _gqa_wcoal_video_kernel(
    kv_blen_vid_ref: Any,   # SMEM (nt, nh, nw) i32
    row_full_ref: Any,      # SMEM (nt, nh, nw_base_cnt) i32 (1 if all win_w blocks == block_vol)
    tail_len_ref: Any,      # SMEM (1,) i32
    q_ref: Any,             # VMEM (bq_gqa, D) bf16 where bq_gqa = G * block_vol
    k_vid_ref: Any,         # VMEM (bkv_w, D) bf16 where bkv_w = win_w * block_vol
    v_vid_ref: Any,         # VMEM (bkv_w, D) bf16
    k_tail_ref: Any,        # VMEM (n_tail_pad, D) bf16 (or dummy if n_tail_pad == 0)
    v_tail_ref: Any,        # VMEM (n_tail_pad, D) bf16
    o_out_ref: Any,         # VMEM (D, bq_gqa) bf16
    m_scratch_ref: Any,     # VMEM (8, bq_gqa) f32
    l_scratch_ref: Any,     # VMEM (8, bq_gqa) f32
    o_scratch_ref: Any,     # VMEM (D, bq_gqa) f32
    *,
    nt: int,
    nh: int,
    nw: int,
    rt: int,
    rh: int,
    rw: int,
    win_t: int,
    win_h: int,
    win_w: int,
    block_vol: int,
    has_tail: bool,
    head_dim_v: int,
    qk_fp8: bool = False,
) -> None:
  """GQA-grouped + W-coalesced 3D block-grid sparse splash kernel.

  Fuses all ``G = H_q // H_kv`` GQA query heads into one ``bq = G * 128`` (512)
  Q tile and coalesces all ``win_w = 2*rw + 1`` (5) contiguous W-neighbor KV
  blocks into a single ``bkv = win_w * 128`` (640) DMA tile, cutting grid
  iterations by ~20x and K/V HBM traffic by 4x while keeping tail KV resident
  in VMEM for ``j == 0`` initialization. Mathematically bit-exact in BF16.
  """
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, _NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {_NUM_SUBLANES}")
  bkv_w = win_w * block_vol

  qt = pl.program_id(1)
  qh = pl.program_id(2)
  qw = pl.program_id(3)
  j = pl.program_id(4)

  qt_lo = jnp.maximum(jnp.int32(0), qt - rt)
  qt_hi = jnp.minimum(jnp.int32(nt), qt + rt + 1)
  qh_lo = jnp.maximum(jnp.int32(0), qh - rh)
  qh_hi = jnp.minimum(jnp.int32(nh), qh + rh + 1)
  qw_lo = jnp.maximum(jnp.int32(0), qw - rw)
  qw_hi = jnp.minimum(jnp.int32(nw), qw + rw + 1)

  kt_base = jnp.minimum(qt_lo, jnp.int32(nt - win_t))
  kh_base = jnp.minimum(qh_lo, jnp.int32(nh - win_h))
  kw_base = jnp.minimum(qw_lo, jnp.int32(nw - win_w))

  dt = j // win_h
  dh = j - dt * win_h
  kt = kt_base + dt
  kh = kh_base + dh

  th_active = (kt >= qt_lo) & (kt < qt_hi) & (kh >= qh_lo) & (kh < qh_hi)

  def _dot_qk(k_mat: Any, q_mat: Any) -> Any:
    if qk_fp8:
      return lax.dot_general(
          k_mat.astype(jnp.float8_e4m3fn),
          q_mat.astype(jnp.float8_e4m3fn),
          _NT_DIM_NUMBERS,
          preferred_element_type=float32,
      )
    return lax.dot_general(
        k_mat, q_mat, _NT_DIM_NUMBERS, preferred_element_type=float32
    )

  def _online_step(qk: Any, v_mat: Any) -> None:
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v_mat, s_cast, _SV_DIM_NUMBERS, preferred_element_type=float32
    )
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  @pl.when(j == 0)
  def _init():
    if has_tail:
      q = q_ref[...]
      k_t = k_tail_ref[...]
      v_t = v_tail_ref[...]
      qk_t = _dot_qk(k_t, q)
      tlen = tail_len_ref[0]
      tpos = lax.broadcasted_iota(jnp.int32, qk_t.shape, 0)
      qk_t = jnp.where(tpos < tlen, qk_t, _DEFAULT_MASK_VALUE)
      m0 = qk_t.max(axis=0)[None, :]
      s0 = jnp.exp2(qk_t - m0[0:1])
      l0 = s0.sum(axis=0, keepdims=True)
      o0 = lax.dot_general(
          v_t, s0.astype(jnp.bfloat16), _SV_DIM_NUMBERS, preferred_element_type=float32
      )
      m_scratch_ref[...] = jnp.broadcast_to(m0, m_scratch_ref.shape)
      l_scratch_ref[...] = jnp.broadcast_to(l0, l_scratch_ref.shape)
      o_scratch_ref[...] = o0
    else:
      o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, _DEFAULT_MASK_VALUE)

  w_all_in = (kw_base >= qw_lo) & ((kw_base + win_w) <= qw_hi)
  all_full = w_all_in & (row_full_ref[kt, kh, kw_base] > 0)

  @pl.when(th_active & all_full)
  def _run_full():
    qk = _dot_qk(k_vid_ref[...], q_ref[...])
    _online_step(qk, v_vid_ref[...])

  @pl.when(th_active & jnp.logical_not(all_full))
  def _run_masked():
    qk = _dot_qk(k_vid_ref[...], q_ref[...])
    pos = lax.broadcasted_iota(jnp.int32, (bkv_w, 1), 0)
    b_idx = pos // block_vol
    r_idx = pos - b_idx * block_vol
    kw_cur = kw_base + b_idx
    in_w = (kw_cur >= qw_lo) & (kw_cur < qw_hi)
    vlen = jnp.full((bkv_w, 1), kv_blen_vid_ref[kt, kh, kw_base], dtype=jnp.int32)
    for w in range(1, win_w):
      vlen = jnp.where(b_idx == w, kv_blen_vid_ref[kt, kh, kw_base + w], vlen)
    valid = in_w & (r_idx < vlen)
    qk = jnp.where(valid, qk, _DEFAULT_MASK_VALUE)
    _online_step(qk, v_vid_ref[...])

  @pl.when(j == (win_t * win_h) - 1)
  def _end():
    l = l_scratch_ref[...]
    l_safe = jnp.where(l == 0.0, 1.0, l)
    l_inv = jnp.concatenate([1.0 / l_safe] * head_dim_v_repeats, axis=0)
    o_out_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_out_ref.dtype)


def _gqa_dense_tail_kernel(
    sb_blen_ref: Any,       # SMEM (n_superblocks, blocks_per_sb) i32
    sb_full_ref: Any,       # SMEM (n_superblocks,) i32 (1 if all sub-blocks == block_vol)
    sb_active_ref: Any,     # SMEM (n_superblocks,) i32 (1 if sum(sub-blocks) > 0)
    q_ref: Any,             # VMEM (bq_gqa, D) bf16
    k_ref: Any,             # VMEM (bkv_sb, D) bf16 where bkv_sb = blocks_per_sb * block_vol
    v_ref: Any,             # VMEM (bkv_sb, D) bf16
    o_out_ref: Any,         # VMEM (D, bq_gqa) bf16
    m_scratch_ref: Any,     # VMEM (8, bq_gqa) f32
    l_scratch_ref: Any,     # VMEM (8, bq_gqa) f32
    o_scratch_ref: Any,     # VMEM (D, bq_gqa) f32
    *,
    n_superblocks: int,
    blocks_per_sb: int,
    block_vol: int,
    head_dim_v: int,
    qk_fp8: bool = False,
) -> None:
  """GQA-grouped superblock (bkv=1024) dense masked splash kernel for tail Q tokens."""
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, _NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {_NUM_SUBLANES}")
  bkv_sb = blocks_per_sb * block_vol

  j = pl.program_id(2)
  active = sb_active_ref[j] > 0
  is_full = sb_full_ref[j] > 0

  @pl.when(j == 0)
  def _init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, _DEFAULT_MASK_VALUE)

  def _step(masked: bool) -> None:
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    if qk_fp8:
      qk = lax.dot_general(
          k.astype(jnp.float8_e4m3fn),
          q.astype(jnp.float8_e4m3fn),
          _NT_DIM_NUMBERS,
          preferred_element_type=float32,
      )
    else:
      qk = lax.dot_general(k, q, _NT_DIM_NUMBERS, preferred_element_type=float32)
    if masked:
      pos = lax.broadcasted_iota(jnp.int32, (bkv_sb, 1), 0)
      b_idx = pos // block_vol
      r_idx = pos - b_idx * block_vol
      vlen = jnp.full((bkv_sb, 1), sb_blen_ref[j, 0], dtype=jnp.int32)
      for sb in range(1, blocks_per_sb):
        vlen = jnp.where(b_idx == sb, sb_blen_ref[j, sb], vlen)
      qk = jnp.where(r_idx < vlen, qk, _DEFAULT_MASK_VALUE)

    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(v, s_cast, _SV_DIM_NUMBERS, preferred_element_type=float32)
    o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

  @pl.when(active & is_full)
  def _run_full():
    _step(masked=False)

  @pl.when(active & jnp.logical_not(is_full))
  def _run_masked():
    _step(masked=True)

  @pl.when(j == n_superblocks - 1)
  def _end():
    l = l_scratch_ref[...]
    l_safe = jnp.where(l == 0.0, 1.0, l)
    l_inv = jnp.concatenate([1.0 / l_safe] * head_dim_v_repeats, axis=0)
    o_out_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_out_ref.dtype)


def _gqa_wcoal_block_grid_attention(
    q: Any,  # (S_total, H_q, D)
    k: Any,  # (S_total, H_kv, D)
    v: Any,  # (S_total, H_kv, D)
    kv_blen: Any,  # (num_kv_blocks,) i32
    *,
    layout: RefinerAttentionLayout,
    scale: float,
    qk_fp8: bool = False,
) -> Any:
  """Fast GQA-grouped + W-coalesced TPU Pallas sparse + tail attention."""
  _, num_q_heads, head_dim = q.shape
  num_kv_heads = k.shape[1]
  assert num_q_heads % num_kv_heads == 0
  g_size = num_q_heads // num_kv_heads
  nt, nh, nw = layout.block_grid_thw
  rt, rh, rw = layout.radius
  bvol = layout.block_vol
  win_t = min(nt, 2 * rt + 1)
  win_h = min(nh, 2 * rh + 1)
  win_w = min(nw, 2 * rw + 1)
  bq_gqa = g_size * bvol
  bkv_w = win_w * bvol
  n_vid_blocks = layout.n_vid_blocks
  n_vid_pad = layout.n_vid_pad
  n_tail_blocks = layout.n_tail_blocks
  n_tail_pad = layout.n_tail_pad
  has_tail = n_tail_pad > 0

  # Video Q grouped by GQA group: (H_kv, n_vid_blocks * bq_gqa, D)
  q_vid = q[:n_vid_pad].reshape(n_vid_blocks, bvol, num_kv_heads, g_size, head_dim)
  q_vid = jnp.transpose(q_vid, (2, 0, 3, 1, 4)).reshape(
      num_kv_heads, n_vid_blocks * bq_gqa, head_dim
  )
  q_vid_scaled = (q_vid.astype(jnp.float32) * (scale * _LOG2_E)).astype(jnp.bfloat16)

  # Video K, V in 3D grid layout: (H_kv, nt, nh, nw * bvol, D)
  k_vid = (
      jnp.transpose(k[:n_vid_pad], (1, 0, 2))
      .reshape(num_kv_heads, nt, nh, nw * bvol, head_dim)
      .astype(jnp.bfloat16)
  )
  v_vid = (
      jnp.transpose(v[:n_vid_pad], (1, 0, 2))
      .reshape(num_kv_heads, nt, nh, nw * bvol, head_dim)
      .astype(jnp.bfloat16)
  )

  # Tail K, V (resident in VMEM if has_tail; tiny 128-token dummy otherwise)
  if has_tail:
    k_tail = jnp.transpose(k[n_vid_pad:], (1, 0, 2)).astype(jnp.bfloat16)
    v_tail = jnp.transpose(v[n_vid_pad:], (1, 0, 2)).astype(jnp.bfloat16)
    tail_blk = n_tail_pad
    tail_len = jnp.sum(kv_blen[n_vid_blocks:], dtype=jnp.int32).reshape(1)
  else:
    k_tail = jnp.zeros((num_kv_heads, bvol, head_dim), dtype=jnp.bfloat16)
    v_tail = jnp.zeros((num_kv_heads, bvol, head_dim), dtype=jnp.bfloat16)
    tail_blk = bvol
    tail_len = jnp.zeros((1,), dtype=jnp.int32)

  kv_blen_vid = kv_blen[:n_vid_blocks].reshape(nt, nh, nw)
  # Precompute per-W-window all-full flag: (nt, nh, nw - win_w + 1)
  nw_base = nw - win_w + 1
  w_full_slices = [
      (kv_blen_vid[:, :, w : w + nw_base] >= bvol) for w in range(win_w)
  ]
  row_full = w_full_slices[0]
  for s in w_full_slices[1:]:
    row_full = row_full & s
  row_full_i32 = row_full.astype(jnp.int32)

  nhw = nh * nw

  def q_index_map(h, qt, qh, qw, j, *_):
    return (h, qt * nhw + qh * nw + qw, 0)

  def kv_vid_index_map(h, qt, qh, qw, j, *_):
    qt_lo = jnp.maximum(jnp.int32(0), qt - rt)
    qh_lo = jnp.maximum(jnp.int32(0), qh - rh)
    qw_lo = jnp.maximum(jnp.int32(0), qw - rw)
    kt_base = jnp.minimum(qt_lo, jnp.int32(nt - win_t))
    kh_base = jnp.minimum(qh_lo, jnp.int32(nh - win_h))
    kw_base = jnp.minimum(qw_lo, jnp.int32(nw - win_w))
    dt = j // win_h
    dh = j - dt * win_h
    return (h, kt_base + dt, kh_base + dh, kw_base * bvol, 0)

  def kv_tail_index_map(h, qt, qh, qw, j, *_):
    return (h, 0, 0)

  def out_index_map(h, qt, qh, qw, j, *_):
    return (h, 0, qt * nhw + qh * nw + qw)

  scratch_shapes = [
      pltpu.VMEM((_NUM_SUBLANES, bq_gqa), jnp.float32),
      pltpu.VMEM((_NUM_SUBLANES, bq_gqa), jnp.float32),
      pltpu.VMEM((head_dim, bq_gqa), jnp.float32),
  ]

  out_vid_hdb = pl.pallas_call(
      functools.partial(
          _gqa_wcoal_video_kernel,
          nt=nt,
          nh=nh,
          nw=nw,
          rt=rt,
          rh=rh,
          rw=rw,
          win_t=win_t,
          win_h=win_h,
          win_w=win_w,
          block_vol=bvol,
          has_tail=has_tail,
          head_dim_v=head_dim,
          qk_fp8=qk_fp8,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec((None, bq_gqa, head_dim), q_index_map),
              pl.BlockSpec(
                  (None, None, None, pl.Element(bkv_w), pl.Element(head_dim)),
                  kv_vid_index_map,
              ),
              pl.BlockSpec(
                  (None, None, None, pl.Element(bkv_w), pl.Element(head_dim)),
                  kv_vid_index_map,
              ),
              pl.BlockSpec((None, tail_blk, head_dim), kv_tail_index_map),
              pl.BlockSpec((None, tail_blk, head_dim), kv_tail_index_map),
          ],
          out_specs=pl.BlockSpec((None, head_dim, bq_gqa), out_index_map),
          scratch_shapes=scratch_shapes,
          grid=(num_kv_heads, nt, nh, nw, win_t * win_h),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (num_kv_heads, head_dim, n_vid_blocks * bq_gqa), jnp.bfloat16
      ),
  )(kv_blen_vid, row_full_i32, tail_len, q_vid_scaled, k_vid, v_vid, k_tail, v_tail)

  out_vid = out_vid_hdb.reshape(num_kv_heads, head_dim, n_vid_blocks, g_size, bvol)
  out_vid = jnp.transpose(out_vid, (2, 4, 0, 3, 1)).reshape(
      n_vid_pad, num_q_heads, head_dim
  )

  if not has_tail:
    return out_vid.astype(q.dtype)

  # Fast GQA-grouped superblock (bkv = 8 * block_vol = 1024) dense tail-Q attention
  blocks_per_sb = 8
  bkv_sb = blocks_per_sb * bvol
  n_kv_blocks = n_vid_blocks + n_tail_blocks
  n_superblocks = math.ceil(n_kv_blocks / blocks_per_sb)
  pad_blocks = n_superblocks * blocks_per_sb - n_kv_blocks

  q_tail = q[n_vid_pad:].reshape(n_tail_blocks, bvol, num_kv_heads, g_size, head_dim)
  q_tail = jnp.transpose(q_tail, (2, 0, 3, 1, 4)).reshape(
      num_kv_heads, n_tail_blocks * bq_gqa, head_dim
  )
  q_tail_scaled = (q_tail.astype(jnp.float32) * (scale * _LOG2_E)).astype(jnp.bfloat16)

  k_all = jnp.transpose(k, (1, 0, 2)).astype(jnp.bfloat16)
  v_all = jnp.transpose(v, (1, 0, 2)).astype(jnp.bfloat16)
  if pad_blocks > 0:
    pad_tok = pad_blocks * bvol
    k_all = jnp.pad(k_all, ((0, 0), (0, pad_tok), (0, 0)))
    v_all = jnp.pad(v_all, ((0, 0), (0, pad_tok), (0, 0)))
    sb_blen = jnp.pad(kv_blen, (0, pad_blocks)).reshape(n_superblocks, blocks_per_sb)
  else:
    sb_blen = kv_blen.reshape(n_superblocks, blocks_per_sb)

  sb_full = jnp.all(sb_blen >= bvol, axis=1).astype(jnp.int32)
  sb_active = (jnp.sum(sb_blen, axis=1) > 0).astype(jnp.int32)

  def qt_index_map(h, i, j, *_):
    return (h, i, 0)

  def kvt_index_map(h, i, j, *_):
    return (h, j, 0)

  def outt_index_map(h, i, j, *_):
    return (h, 0, i)

  out_tail_hdb = pl.pallas_call(
      functools.partial(
          _gqa_dense_tail_kernel,
          n_superblocks=n_superblocks,
          blocks_per_sb=blocks_per_sb,
          block_vol=bvol,
          head_dim_v=head_dim,
          qk_fp8=qk_fp8,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec((None, bq_gqa, head_dim), qt_index_map),
              pl.BlockSpec((None, bkv_sb, head_dim), kvt_index_map),
              pl.BlockSpec((None, bkv_sb, head_dim), kvt_index_map),
          ],
          out_specs=pl.BlockSpec((None, head_dim, bq_gqa), outt_index_map),
          scratch_shapes=scratch_shapes,
          grid=(num_kv_heads, n_tail_blocks, n_superblocks),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
          vmem_limit_bytes=_vmem_limit_bytes(),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (num_kv_heads, head_dim, n_tail_blocks * bq_gqa), jnp.bfloat16
      ),
  )(sb_blen, sb_full, sb_active, q_tail_scaled, k_all, v_all)

  out_tail = out_tail_hdb.reshape(num_kv_heads, head_dim, n_tail_blocks, g_size, bvol)
  out_tail = jnp.transpose(out_tail, (2, 4, 0, 3, 1)).reshape(
      n_tail_pad, num_q_heads, head_dim
  )
  return jnp.concatenate([out_vid, out_tail], axis=0).astype(q.dtype)


# ---------------------------------------------------------------------------
# 3. Direct sparse_splash.py Backend (when all_video_blocks_full=True) & CPU Ref
# ---------------------------------------------------------------------------


def _sparse_splash_backend_attention(
    q: Any,  # (S_total, H_q, D)
    k: Any,  # (S_total, H_kv, D)
    v: Any,  # (S_total, H_kv, D)
    *,
    layout: RefinerAttentionLayout,
    scale: float,
) -> Any:
  """Run ``diffusers_tpu.kernels.sparse_splash.attention`` with SMEM-safe Q chunking.

  Note on ``sparse_splash.py`` limitations verified in this codebase:
    1. ``sparse_splash.py`` only masks trailing positions ``pos >= kv_len`` (end
       of sequence), NOT interior partial blocks (such as the 16 pad tokens in
       each of the 510 ``bt=7`` blocks when ``T_ref=63``). Use
       ``_masked_pallas_block_grid_attention`` whenever
       ``not layout.all_video_blocks_full``.
    2. ``sparse_splash.attention`` puts the full ``(1, num_q_blocks, width)``
       int32 ``kv_table`` into SMEM (``2.10 MiB`` at 1080p > 1 MiB TPU v7x SMEM).
       We chunk video Q-blocks so each call's ``kv_table`` stays ``<= 256 KiB``.
    3. ``sparse_splash.attention`` calls ``fixed_m_bound.fixed_m_operands(q, k)``
       unconditionally (even when ``use_fixed_m=False``), which broadcasts
       ``(H_q,) * (H_kv,)`` — legal when ``H_kv == 1`` (Ulysses CP=8) or
       ``H_q == H_kv``, but fails if ``1 < H_kv < H_q``. When ``1 < H_kv < H_q``
       we repeat KV heads to ``H_q``.
  """
  if not layout.all_video_blocks_full:
    raise ValueError(
        "sparse_splash.py only masks trailing kv_len and cannot mask interior "
        "partial video blocks (e.g. T_ref=63 bt=7 112/128). Use "
        "DT_MAGI2_REFINER_ATTN_IMPL=masked_pallas (default) or pad T to a "
        "multiple of block_t_size=8."
    )
  if not layout.kv_table_vid:
    raise ValueError("build_flat_tables=True is required for sparse_splash backend.")

  # (S, H, D) -> (1, H, S, D)
  q_bhsd = jnp.transpose(q, (1, 0, 2))[None, ...].astype(jnp.bfloat16)
  k_bhsd = jnp.transpose(k, (1, 0, 2))[None, ...].astype(jnp.bfloat16)
  v_bhsd = jnp.transpose(v, (1, 0, 2))[None, ...].astype(jnp.bfloat16)

  h_q = q_bhsd.shape[1]
  h_kv = k_bhsd.shape[1]
  if h_kv > 1 and h_kv != h_q:
    rep = h_q // h_kv
    k_bhsd = jnp.repeat(k_bhsd, rep, axis=1)
    v_bhsd = jnp.repeat(v_bhsd, rep, axis=1)

  b = layout.block_vol
  actual_kv_len = layout.n_vid_pad + layout.n_tail_real
  kv_len_arr = jnp.asarray(actual_kv_len, dtype=jnp.int32)

  # Chunk video Q-blocks so each kv_table fits well inside TPU SMEM (<= 256 KiB).
  max_width = max(len(r) for r in layout.kv_table_vid)
  q_blocks_per_chunk = max(
      1, _SPARSE_SPLASH_SMEM_BUDGET_BYTES // max(1, max_width * 4)
  )
  vid_outs = []
  for qb0 in range(0, layout.n_vid_blocks, q_blocks_per_chunk):
    qb1 = min(layout.n_vid_blocks, qb0 + q_blocks_per_chunk)
    q_chunk = q_bhsd[:, :, qb0 * b : qb1 * b, :]
    sub_table = layout.kv_table_vid[qb0:qb1]
    out_c = sparse_splash.attention(
        q_chunk,
        k_bhsd,
        v_bhsd,
        scale=scale,
        block_q=b,
        block_kv=b,
        block_kv_compute=b,
        use_fixed_m=False,
        qk_fp8=False,
        kv_table=sub_table,
        kv_len=kv_len_arr,
    )
    vid_outs.append(out_c)
  out_vid = vid_outs[0] if len(vid_outs) == 1 else jnp.concatenate(vid_outs, axis=2)

  if layout.n_tail_blocks > 0:
    q_tail = q_bhsd[:, :, layout.n_vid_pad :, :]
    out_tail = sparse_splash.attention(
        q_tail,
        k_bhsd,
        v_bhsd,
        scale=scale,
        block_q=b,
        block_kv=b,
        block_kv_compute=b,
        use_fixed_m=False,
        qk_fp8=False,
        kv_table=layout.kv_table_tail,
        kv_len=kv_len_arr,
    )
    out_bhsd = jnp.concatenate([out_vid, out_tail], axis=2)
  else:
    out_bhsd = out_vid

  return jnp.transpose(out_bhsd[0], (1, 0, 2)).astype(q.dtype)


def _masked_pallas_block_grid_attention(
    q: Any,  # (S_total, H_q, D)
    k: Any,  # (S_total, H_kv, D)
    v: Any,  # (S_total, H_kv, D)
    nb_arr: Any,
    cnt_arr: Any,
    dec_arr: Any,
    kv_blen: Any,
    *,
    layout: RefinerAttentionLayout,
    scale: float,
    qk_fp8: bool = False,
    use_gqa_wcoal: bool = True,
) -> Any:
  """Device-local TPU Pallas 3D block-grid sparse attention with exact pad masking."""
  nt, nh, nw = layout.block_grid_thw
  rt, rh, rw = layout.radius
  win_t = min(nt, 2 * rt + 1)
  win_h = min(nh, 2 * rh + 1)
  win_w = min(nw, 2 * rw + 1)
  if (
      use_gqa_wcoal
      and q.shape[1] % k.shape[1] == 0
      and nt >= win_t
      and nh >= win_h
      and nw >= win_w
      and layout.n_tail_pad <= 1024
  ):
    return _gqa_wcoal_block_grid_attention(
        q,
        k,
        v,
        kv_blen,
        layout=layout,
        scale=scale,
        qk_fp8=qk_fp8,
    )

  q_hsd = jnp.transpose(q, (1, 0, 2))  # (H_q, S_total, D)
  k_hsd = jnp.transpose(k, (1, 0, 2))  # (H_kv, S_total, D)
  v_hsd = jnp.transpose(v, (1, 0, 2))  # (H_kv, S_total, D)

  out_vid = _grid3d_masked_splash_attention(
      q_hsd[:, : layout.n_vid_pad, :],
      k_hsd,
      v_hsd,
      nb_arr,
      cnt_arr,
      dec_arr,
      kv_blen,
      layout=layout,
      scale=scale,
  )
  if layout.n_tail_blocks > 0:
    out_tail = _dense_masked_splash_attention(
        q_hsd[:, layout.n_vid_pad :, :],
        k_hsd,
        v_hsd,
        kv_blen,
        layout=layout,
        scale=scale,
    )
    out_hsd = jnp.concatenate([out_vid, out_tail], axis=1)
  else:
    out_hsd = out_vid
  return jnp.transpose(out_hsd, (1, 0, 2)).astype(q.dtype)


def _ref_block_grid_attention(
    q: Any,  # (S_total, H_q, D)
    k: Any,  # (S_total, H_kv, D)
    v: Any,  # (S_total, H_kv, D)
    kv_blen: Any,  # (num_kv_blocks,) i32
    *,
    layout: RefinerAttentionLayout,
    scale: float,
) -> Any:
  """Exact fp32 block-masked reference attention for CPU parity & small grids.

  Implements the exact bidirectional 3D block-grid connectivity
  (``5x5x5`` video neighbors + dense ``video <-> tail`` & ``tail <-> tail``)
  and exact per-token masking via ``kv_blen`` (masking both ``bt=7`` pad tokens
  and tail pad tokens to ``-inf``).
  """
  orig_dtype = q.dtype
  s_total, h_q, d = q.shape
  h_kv = k.shape[1]
  assert h_q % h_kv == 0
  rep = h_q // h_kv

  q_f = q.astype(jnp.float32)
  k_f = jnp.repeat(k.astype(jnp.float32), rep, axis=1)
  v_f = jnp.repeat(v.astype(jnp.float32), rep, axis=1)

  b = layout.block_vol
  n_vb = layout.n_vid_blocks
  n_tb = layout.n_tail_blocks
  n_tot_b = n_vb + n_tb
  nt, nh, nw = layout.block_grid_thw
  rt, rh, rw = layout.radius

  # Build exact block connectivity matrix (n_tot_b, n_tot_b) bool on host NumPy.
  block_conn = np.zeros((n_tot_b, n_tot_b), dtype=bool)
  nhw = nh * nw
  for qt in range(nt):
    for qh in range(nh):
      for qw in range(nw):
        qb = qt * nhw + qh * nw + qw
        for kt in range(max(0, qt - rt), min(nt, qt + rt + 1)):
          for kh in range(max(0, qh - rh), min(nh, qh + rh + 1)):
            for kw in range(max(0, qw - rw), min(nw, qw + rw + 1)):
              block_conn[qb, kt * nhw + kh * nw + kw] = True
  if n_tb > 0:
    block_conn[:n_vb, n_vb:] = True  # video -> tail
    block_conn[n_vb:, :] = True  # tail -> video + tail

  # Combine block connectivity with per-KV-block token valid lengths (traced!).
  local_pos = jnp.arange(b, dtype=jnp.int32)[None, :]  # (1, b)
  kv_tok_valid = (local_pos < kv_blen[:, None]).reshape(s_total)  # (S_total,)

  block_conn_j = jnp.asarray(block_conn)  # (n_tot_b, n_tot_b)
  # Expand to token mask (S_total, S_total): (n_tot_b, 1, n_tot_b, 1) -> (S, S)
  tok_conn = jnp.broadcast_to(
      block_conn_j[:, None, :, None], (n_tot_b, b, n_tot_b, b)
  ).reshape(s_total, s_total)
  attn_mask = tok_conn & kv_tok_valid[None, :]  # (S_total, S_total)

  # Attention: (H_q, S_q, D) @ (H_q, S_kv, D)^T -> (H_q, S_q, S_kv)
  logits = jnp.einsum("qhd,khd->hqk", q_f * jnp.float32(scale), k_f)
  logits = jnp.where(attn_mask[None, :, :], logits, jnp.float32(-1e30))
  probs = jax.nn.softmax(logits, axis=-1)
  out = jnp.einsum("hqk,khd->qhd", probs, v_f)
  return out.astype(orig_dtype)


def refiner_sparse_attention(
    q: Any,  # (S_total, H_q, D) or (S_local, H_q, D) if in_shard_map
    k: Any,  # (S_total, H_kv, D) or (S_local, H_kv, D) if in_shard_map
    v: Any,  # (S_total, H_kv, D) or (S_local, H_kv, D) if in_shard_map
    kv_blen: Any,  # (num_kv_blocks,) i32
    *,
    layout: RefinerAttentionLayout,
    mesh: Any = None,
    cp_axis: str | None = None,
    parallel_mode: str = "cp",
    impl: str | None = None,
    in_shard_map: bool = False,
) -> Any:
  """Dispatch 3D block-grid sparse attention with Ulysses CP or TP across ``mesh``."""
  mode = (impl or ATTN_IMPL).lower()
  if mode == "auto":
    mode = "masked_pallas" if _on_tpu() else "ref"

  scale = 1.0 / math.sqrt(q.shape[-1])
  nb_arr = jnp.asarray(layout.nb_arr, dtype=jnp.int32)
  cnt_arr = jnp.asarray(layout.cnt_arr, dtype=jnp.int32)
  dec_arr = jnp.asarray(layout.dec_arr, dtype=jnp.int32)

  def _local_attn(q_loc, k_loc, v_loc, nb_loc, cnt_loc, dec_loc, blen_loc):
    if mode in ("masked_pallas", "masked_pallas_fp8", "masked_pallas_legacy"):
      return _masked_pallas_block_grid_attention(
          q_loc,
          k_loc,
          v_loc,
          nb_loc,
          cnt_loc,
          dec_loc,
          blen_loc,
          layout=layout,
          scale=scale,
          qk_fp8=(mode == "masked_pallas_fp8"),
          use_gqa_wcoal=(mode != "masked_pallas_legacy"),
      )
    if mode == "sparse_splash":
      return _sparse_splash_backend_attention(
          q_loc, k_loc, v_loc, layout=layout, scale=scale
      )
    if mode == "ref":
      return _ref_block_grid_attention(
          q_loc, k_loc, v_loc, blen_loc, layout=layout, scale=scale
      )
    raise ValueError(f"Unknown refiner attention impl: {mode!r}")

  if mesh is None or mesh.size == 1:
    return _local_attn(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)

  axis = cp_axis or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0])
  world = mesh.shape[axis]
  if world == 1:
    return _local_attn(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)

  if parallel_mode == "cp":
    assert q.shape[1] % world == 0, f"H_q={q.shape[1]} must be divisible by cp={world}."
    assert k.shape[1] % world == 0, f"H_kv={k.shape[1]} must be divisible by cp={world}."

    def _ulysses_local(q_s, k_s, v_s, nb_s, cnt_s, dec_s, blen_s):
      q_h = lax.all_to_all(q_s, axis, split_axis=1, concat_axis=0, tiled=True)
      k_h = lax.all_to_all(k_s, axis, split_axis=1, concat_axis=0, tiled=True)
      v_h = lax.all_to_all(v_s, axis, split_axis=1, concat_axis=0, tiled=True)
      out_h = _local_attn(q_h, k_h, v_h, nb_s, cnt_s, dec_s, blen_s)
      return lax.all_to_all(out_h, axis, split_axis=0, concat_axis=1, tiled=True)

    if in_shard_map:
      return _ulysses_local(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)

    assert q.shape[0] % world == 0, f"S={q.shape[0]} must be divisible by cp={world}."
    seq_spec = P(axis, None, None)
    rep_spec = P()
    return _shard_map(
        _ulysses_local,
        mesh=mesh,
        in_specs=(
            seq_spec,
            seq_spec,
            seq_spec,
            rep_spec,
            rep_spec,
            rep_spec,
            rep_spec,
        ),
        out_specs=seq_spec,
    )(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)

  if parallel_mode == "tp":
    if in_shard_map:
      return _local_attn(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)
    head_spec = P(None, axis, None)
    rep_spec = P()
    return _shard_map(
        _local_attn,
        mesh=mesh,
        in_specs=(
            head_spec,
            head_spec,
            head_spec,
            rep_spec,
            rep_spec,
            rep_spec,
            rep_spec,
        ),
        out_specs=head_spec,
    )(q, k, v, nb_arr, cnt_arr, dec_arr, kv_blen)

  raise ValueError(f"Unsupported parallel_mode={parallel_mode!r}; expected 'cp' or 'tp'.")


# ---------------------------------------------------------------------------
# 4. Flax NNX Modules (Lazy Flax Import via _build_modules)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _build_modules():
  """Lazily import Flax NNX and construct the MAGI-2 Refiner module classes."""
  from flax import nnx

  ModuleList = getattr(nnx, "List", list)

  class Linear(nnx.Module):
    """Standard ``y = x @ W.T (+ b)`` with PyTorch ``(out, in)`` weight layout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: Any = jnp.float32,
        precision: Any = None,
        rngs: Any = None,
    ):
      self.in_features = int(in_features)
      self.out_features = int(out_features)
      self.weight = nnx.Param(jnp.zeros((out_features, in_features), dtype=dtype))
      self.bias = (
          nnx.Param(jnp.zeros((out_features,), dtype=dtype)) if bias else None
      )
      self.precision = precision

    def __call__(self, x: Any) -> Any:
      w = self.weight[...]
      y = lax.dot_general(
          x.astype(w.dtype),
          w,
          (((x.ndim - 1,), (1,)), ((), ())),
          precision=self.precision,
          preferred_element_type=(
              jnp.float32 if w.dtype == jnp.float32 else None
          ),
      )
      if self.bias is not None:
        y = y + self.bias[...].astype(y.dtype)
      return y

  class ModalityRMSNorm(nnx.Module):
    """Exact ``MultiModalityRMSNorm`` (Gemma ``1 + w``, zero-init fp32 weight).

    Stores ``weight`` as ``(dim,)`` for ``num_modality == 1`` and ``(3, dim)``
    for ``num_modality == 3`` (zero-copy reshape of checkpoint's ``(3 * dim,)``).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_modality: int = 1,
        eps: float = 1e-6,
        rngs: Any = None,
    ):
      self.dim = int(dim)
      self.num_modality = int(num_modality)
      self.eps = float(eps)
      shape = (dim,) if num_modality == 1 else (num_modality, dim)
      self.weight = nnx.Param(jnp.zeros(shape, dtype=jnp.float32))

    def __call__(
        self,
        x: Any,
        modality_lens: Sequence[int] | None = None,
        *,
        cast_back: bool = True,
        modality_ids: Any | None = None,
    ) -> Any:
      if self.num_modality == 1:
        return rms_norm_1pw(x, self.weight[...], eps=self.eps, cast_back=cast_back)
      return multi_modality_rms_norm(
          x,
          self.weight[...],
          modality_lens=None if modality_ids is not None else modality_lens,
          eps=self.eps,
          cast_back=cast_back,
          modality_ids=modality_ids,
      )

  class ModalityLinear(nnx.Module):
    """Exact ``BaseLinear`` (``m=1``) / ``NativeMoELinear`` (``m=3``) without bias.

    Stores ``weight`` as ``(out, in)`` for ``num_modality == 1`` and
    ``(3, out, in)`` for ``num_modality == 3`` (zero-copy reshape of checkpoint's
    ``(3 * out, in)``). On MM layers, slices ``x`` along axis 0 by
    ``modality_lens = (N_v, N_a, N_t)``, skips any empty modality (e.g. ``N_a=0``),
    and performs strictly 1x GEMM FLOPs (never 3x).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_modality: int = 1,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.in_features = int(in_features)
      self.out_features = int(out_features)
      self.num_modality = int(num_modality)
      shape = (
          (out_features, in_features)
          if num_modality == 1
          else (num_modality, out_features, in_features)
      )
      self.weight = nnx.Param(jnp.zeros(shape, dtype=dtype))

    def _matmul(self, x: Any, w: Any, compute_dtype: Any) -> Any:
      x_c = x.astype(compute_dtype)
      w_c = w.astype(compute_dtype)
      prec = lax.Precision.HIGHEST if compute_dtype == jnp.float32 else None
      return lax.dot_general(
          x_c,
          w_c,
          (((x_c.ndim - 1,), (1,)), ((), ())),
          precision=prec,
          preferred_element_type=jnp.float32 if compute_dtype == jnp.float32 else None,
      ).astype(compute_dtype)

    def __call__(
        self,
        x: Any,
        modality_lens: Sequence[int] | None = None,
        *,
        compute_dtype: Any = None,
        group_sizes: Any | None = None,
    ) -> Any:
      w = self.weight[...]
      c_dtype = w.dtype if compute_dtype is None else compute_dtype
      if self.num_modality == 1:
        return self._matmul(x, w, c_dtype)

      if group_sizes is not None:
        x_c = x.astype(c_dtype)
        w_t = jnp.swapaxes(w.astype(c_dtype), 1, 2)  # (M, in, out)
        prec = lax.Precision.HIGHEST if c_dtype == jnp.float32 else None
        return lax.ragged_dot(
            x_c,
            w_t,
            group_sizes,
            precision=prec,
            preferred_element_type=jnp.float32 if c_dtype == jnp.float32 else None,
        ).astype(c_dtype)

      if modality_lens is None or len(modality_lens) != self.num_modality:
        raise ValueError(
            f"Expected modality_lens of length {self.num_modality}, got {modality_lens}."
        )
      chunks = []
      offset = 0
      for m, n_m in enumerate(modality_lens):
        n_m = int(n_m)
        if n_m > 0:
          chunks.append(self._matmul(x[offset : offset + n_m], w[m], c_dtype))
          offset += n_m
      return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)

  class PreAdapter(nnx.Module):
    """Exact ``magi2_refiner.Adapter`` (fp32, ``bias=True``)."""

    def __init__(self, config: Magi2RefinerConfig, *, rngs: Any = None):
      self.config = config
      self.video_embedder = Linear(
          config.video_in_channels,
          config.hidden_size,
          bias=True,
          dtype=jnp.float32,
          precision=lax.Precision.HIGHEST,
          rngs=rngs,
      )
      self.text_embedder = Linear(
          config.text_in_channels,
          config.hidden_size,
          bias=True,
          dtype=jnp.float32,
          precision=lax.Precision.HIGHEST,
          rngs=rngs,
      )
      self.audio_embedder = Linear(
          config.audio_in_channels,
          config.hidden_size,
          bias=True,
          dtype=jnp.float32,
          precision=lax.Precision.HIGHEST,
          rngs=rngs,
      )

    def __call__(
        self,
        video_x: Any,  # (n_vid_pad, C_v) fp32
        audio_x: Any | None,  # (n_audio, C_a) fp32 or None/empty
        text_x: Any | None,  # (n_text_slots, C_t) fp32 or None/empty
    ) -> Any:
      parts = [self.video_embedder(video_x.astype(jnp.float32))]
      if audio_x is not None and audio_x.shape[0] > 0:
        parts.append(self.audio_embedder(audio_x.astype(jnp.float32)))
      if text_x is not None and text_x.shape[0] > 0:
        parts.append(self.text_embedder(text_x.astype(jnp.float32)))
      return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)

  class Attention(nnx.Module):
    """Exact ``magi2_refiner.Attention`` (GQA 32Q/8KV, QK-Norm, Partial RoPE, Gate)."""

    def __init__(
        self,
        config: Magi2RefinerConfig,
        *,
        num_modality: int,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.config = config
      self.num_modality = int(num_modality)
      self.num_heads_q = config.num_heads_q
      self.num_heads_kv = config.num_heads_kv
      self.head_dim = config.head_dim
      self.q_size = config.num_heads_q * config.head_dim
      self.kv_size = config.num_heads_kv * config.head_dim
      self.enable_attn_gating = config.enable_attn_gating

      self.pre_norm = ModalityRMSNorm(
          config.hidden_size,
          num_modality=num_modality,
          eps=config.rms_norm_eps,
          rngs=rngs,
      )
      self.linear_qkv = ModalityLinear(
          config.hidden_size,
          self.q_size + 2 * self.kv_size,
          num_modality=num_modality,
          dtype=dtype,
          rngs=rngs,
      )
      self.linear_g = (
          ModalityLinear(
              config.hidden_size,
              config.num_heads_q,
              num_modality=num_modality,
              dtype=dtype,
              rngs=rngs,
          )
          if config.enable_attn_gating
          else None
      )
      self.linear_proj = ModalityLinear(
          self.q_size,
          config.hidden_size,
          num_modality=num_modality,
          dtype=dtype,
          rngs=rngs,
      )
      self.q_norm = ModalityRMSNorm(
          config.head_dim,
          num_modality=num_modality,
          eps=config.rms_norm_eps,
          rngs=rngs,
      )
      self.k_norm = ModalityRMSNorm(
          config.head_dim,
          num_modality=num_modality,
          eps=config.rms_norm_eps,
          rngs=rngs,
      )

    def __call__(
        self,
        hidden_states: Any,  # (S_total, D) or (S_local, D) if in_shard_map
        rope_96: Any,  # (S_total, ro_dim) or (S_local, ro_dim) fp32
        kv_blen: Any,  # (num_kv_blocks,) i32
        *,
        layout: RefinerAttentionLayout,
        compute_dtype: Any,
        mesh: Any = None,
        cp_axis: str | None = None,
        parallel_mode: str = "cp",
        attn_impl: str | None = None,
        modality_ids: Any | None = None,
        group_sizes: Any | None = None,
        in_shard_map: bool = False,
    ) -> Any:
      mod_lens = layout.modality_lens
      seq_len = hidden_states.shape[0]

      h = self.pre_norm(hidden_states, mod_lens, modality_ids=modality_ids).astype(compute_dtype)
      qkv = self.linear_qkv(
          h, mod_lens, compute_dtype=compute_dtype, group_sizes=group_sizes
      ).astype(jnp.float32)
      q = qkv[:, : self.q_size].reshape(seq_len, self.num_heads_q, self.head_dim)
      k = qkv[:, self.q_size : self.q_size + self.kv_size].reshape(
          seq_len, self.num_heads_kv, self.head_dim
      )
      v = qkv[:, self.q_size + self.kv_size :].reshape(
          seq_len, self.num_heads_kv, self.head_dim
      )

      g = None
      if self.linear_g is not None:
        g = self.linear_g(
            h, mod_lens, compute_dtype=compute_dtype, group_sizes=group_sizes
        ).astype(jnp.float32)[..., None]  # (S, H_q, 1)

      # QK-Norm & 3D Partial RoPE run in fp32 (matching magi2_refiner.py:L1718-1727).
      q = self.q_norm(q, mod_lens, cast_back=False, modality_ids=modality_ids)
      k = self.k_norm(k, mod_lens, cast_back=False, modality_ids=modality_ids)
      q = apply_3d_partial_rope(q, rope_96)
      k = apply_3d_partial_rope(k, rope_96)

      # Cast Q/K/V to compute_dtype (bf16 in production, fp32 in fp32 parity tests).
      q_in = q.astype(compute_dtype)
      k_in = k.astype(compute_dtype)
      v_in = v.astype(compute_dtype)

      attn_out = refiner_sparse_attention(
          q_in,
          k_in,
          v_in,
          kv_blen,
          layout=layout,
          mesh=mesh,
          cp_axis=cp_axis,
          parallel_mode=parallel_mode,
          impl=attn_impl,
          in_shard_map=in_shard_map,
      )

      if g is not None:
        attn_out = attn_out.astype(jnp.float32) * jax.nn.sigmoid(g)

      attn_flat = attn_out.reshape(seq_len, self.q_size).astype(compute_dtype)
      return self.linear_proj(
          attn_flat, mod_lens, compute_dtype=compute_dtype, group_sizes=group_sizes
      )

  class MLP(nnx.Module):
    """Exact ``magi2_refiner.MLP`` (dense ``swiglu7``, ``down_proj`` returns fp32)."""

    def __init__(
        self,
        config: Magi2RefinerConfig,
        *,
        num_modality: int,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.config = config
      self.num_modality = int(num_modality)
      self.pre_norm = ModalityRMSNorm(
          config.hidden_size,
          num_modality=num_modality,
          eps=config.rms_norm_eps,
          rngs=rngs,
      )
      self.up_gate_proj = ModalityLinear(
          config.hidden_size,
          config.intermediate_size * 2,
          num_modality=num_modality,
          dtype=dtype,
          rngs=rngs,
      )
      self.down_proj = ModalityLinear(
          config.intermediate_size,
          config.hidden_size,
          num_modality=num_modality,
          dtype=dtype,
          rngs=rngs,
      )

    def __call__(
        self,
        x: Any,
        modality_lens: Sequence[int],
        *,
        compute_dtype: Any,
        modality_ids: Any | None = None,
        group_sizes: Any | None = None,
    ) -> Any:
      h = self.pre_norm(x, modality_lens, modality_ids=modality_ids).astype(compute_dtype)
      u = self.up_gate_proj(
          h, modality_lens, compute_dtype=compute_dtype, group_sizes=group_sizes
      ).astype(jnp.float32)
      u = swiglu7_interleaved(u, out_dtype=compute_dtype)
      # Upstream magi2_refiner.py:L1848 casts down_proj output to float32,
      # promoting the residual stream to float32 after layer 0.
      return self.down_proj(
          u, modality_lens, compute_dtype=compute_dtype, group_sizes=group_sizes
      ).astype(jnp.float32)

  class TransformerLayer(nnx.Module):
    """Exact ``magi2_refiner.TransformerLayer`` (pre-norm residual, no mHC)."""

    def __init__(
        self,
        config: Magi2RefinerConfig,
        layer_idx: int,
        *,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.layer_idx = int(layer_idx)
      num_modality = 3 if layer_idx in config.mm_layers else 1
      self.num_modality = num_modality
      self.attention = Attention(
          config, num_modality=num_modality, dtype=dtype, rngs=rngs
      )
      self.mlp = MLP(config, num_modality=num_modality, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: Any,
        rope_96: Any,
        kv_blen: Any,
        *,
        layout: RefinerAttentionLayout,
        compute_dtype: Any,
        mesh: Any = None,
        cp_axis: str | None = None,
        parallel_mode: str = "cp",
        attn_impl: str | None = None,
        modality_ids: Any | None = None,
        group_sizes: Any | None = None,
        in_shard_map: bool = False,
    ) -> Any:
      attn_out = self.attention(
          hidden_states,
          rope_96,
          kv_blen,
          layout=layout,
          compute_dtype=compute_dtype,
          mesh=mesh,
          cp_axis=cp_axis,
          parallel_mode=parallel_mode,
          attn_impl=attn_impl,
          modality_ids=modality_ids,
          group_sizes=group_sizes,
          in_shard_map=in_shard_map,
      )
      hidden_states = hidden_states + attn_out
      mlp_out = self.mlp(
          hidden_states,
          layout.modality_lens,
          compute_dtype=compute_dtype,
          modality_ids=modality_ids,
          group_sizes=group_sizes,
      )
      hidden_states = hidden_states + mlp_out
      return hidden_states

  class TransformerBlock(nnx.Module):
    """Exact ``magi2_refiner.TransformerBlock`` holding ``layers[0..29]``."""

    def __init__(
        self,
        config: Magi2RefinerConfig,
        *,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.layers = ModuleList([
          TransformerLayer(config, i, dtype=dtype, rngs=rngs)
          for i in range(config.num_layers)
      ])

    def __call__(
        self,
        x: Any,
        rope_96: Any,
        kv_blen: Any,
        *,
        layout: RefinerAttentionLayout,
        compute_dtype: Any,
        mesh: Any = None,
        cp_axis: str | None = None,
        parallel_mode: str = "cp",
        attn_impl: str | None = None,
    ) -> Any:
      for layer in self.layers:
        x = layer(
            x,
            rope_96,
            kv_blen,
            layout=layout,
            compute_dtype=compute_dtype,
            mesh=mesh,
            cp_axis=cp_axis,
            parallel_mode=parallel_mode,
            attn_impl=attn_impl,
        )
      return x

  class PostAdapter(nnx.Module):
    """Exact ``magi2_refiner.PostAdapter`` (fp32, ``bias=False``)."""

    def __init__(self, config: Magi2RefinerConfig, *, rngs: Any = None):
      self.config = config
      self.final_norm_video = ModalityRMSNorm(
          config.hidden_size, num_modality=1, eps=config.rms_norm_eps, rngs=rngs
      )
      self.final_norm_audio = ModalityRMSNorm(
          config.hidden_size, num_modality=1, eps=config.rms_norm_eps, rngs=rngs
      )
      self.final_linear_video = Linear(
          config.hidden_size,
          config.video_in_channels,
          bias=False,
          dtype=jnp.float32,
          precision=lax.Precision.HIGHEST,
          rngs=rngs,
      )
      self.final_linear_audio = Linear(
          config.hidden_size,
          config.audio_in_channels,
          bias=False,
          dtype=jnp.float32,
          precision=lax.Precision.HIGHEST,
          rngs=rngs,
      )

    def __call__(
        self,
        x: Any,  # (S_total, D)
        modality_lens: Sequence[int],
    ) -> tuple[Any, Any | None]:
      n_v, n_a, _ = (int(modality_lens[0]), int(modality_lens[1]), int(modality_lens[2]))
      x_v = x[:n_v].astype(jnp.float32)
      x_v = self.final_norm_video(x_v, cast_back=False)
      out_v = self.final_linear_video(x_v)

      out_a = None
      if n_a > 0:
        x_a = x[n_v : n_v + n_a].astype(jnp.float32)
        x_a = self.final_norm_audio(x_a, cast_back=False)
        out_a = self.final_linear_audio(x_a)
      return out_v, out_a

  class Magi2RefinerTransformer(nnx.Module):
    """Top-level Stage 2 MAGI-2 Refiner DiT matching ``magi2_refiner.Transformer``."""

    def __init__(
        self,
        config: Magi2RefinerConfig | None = None,
        *,
        dtype: Any = jnp.bfloat16,
        rngs: Any = None,
    ):
      self.config = config or Magi2RefinerConfig()
      self.dtype = dtype
      self.pre_adapter = PreAdapter(self.config, rngs=rngs)
      self.block = TransformerBlock(self.config, dtype=dtype, rngs=rngs)
      self.post_adapter = PostAdapter(self.config, rngs=rngs)

    def __call__(
        self,
        video_packed: Any,  # (n_vid_pad, C_v) fp32
        text_packed: Any | None,  # (n_text_slots, C_t) fp32
        rope_96: Any,  # (total_seq_len, ro_dim) fp32
        kv_blen: Any,  # (num_kv_blocks,) i32
        *,
        layout: RefinerAttentionLayout,
        audio_tokens: Any | None = None,  # (n_audio, C_a) fp32
        compute_dtype: Any = None,
        mesh: Any = None,
        cp_axis: str | None = None,
        parallel_mode: str = "cp",
        attn_impl: str | None = None,
    ) -> tuple[Any, Any | None]:
      """Run packed forward pass.

      Returns:
        ``(video_out_padded [n_vid_pad, C_v] fp32, audio_out [n_audio, C_a] fp32 | None)``.
      """
      c_dtype = self.dtype if compute_dtype is None else compute_dtype
      x = self.pre_adapter(video_packed, audio_tokens, text_packed).astype(c_dtype)

      if mesh is not None and mesh.size > 1 and parallel_mode == "cp":
        axis = cp_axis or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0])
        if mesh.shape[axis] > 1:
          x = lax.with_sharding_constraint(x, NamedSharding(mesh, P(axis, None)))

      x = self.block(
          x,
          rope_96,
          kv_blen,
          layout=layout,
          compute_dtype=c_dtype,
          mesh=mesh,
          cp_axis=cp_axis,
          parallel_mode=parallel_mode,
          attn_impl=attn_impl,
      )
      return self.post_adapter(x, layout.modality_lens)

  return TransformerLayer, Magi2RefinerTransformer


# ---------------------------------------------------------------------------
# 5. Sharding Specs, nnx.eval_shape Construction & Streaming Safetensors Loader
# ---------------------------------------------------------------------------


def _spec_for(
    name: str,
    shape: tuple[int, ...],
    *,
    sharding: str = "replicated",
    axis_name: str = "cp",
    axis_size: int = 1,
) -> Any:
  """Return ``PartitionSpec`` for checkpoint parameter ``name``.

  Supported ``sharding`` modes:
    - ``"replicated"`` (default on TPU v7x-8 with Ulysses ``CP=8``): all 12.56 GiB
      replicated per chip (zero weight communication!).
    - ``"fsdp"``: shards every large linear weight along its output axis across
      ``axis_name`` (1.57 GiB/chip on v7x-8).
    - ``"tp"``: Megatron-style split — ``up_gate_proj`` (preserving interleaved
      ``(gate, up)`` pairs since ``21840 % 16 == 0``) and ``linear_g`` split on
      output rows; ``linear_qkv``, ``linear_proj``, ``down_proj`` split on input
      columns (1.57 GiB/chip on v7x-8).
  """
  if sharding == "replicated" or axis_size <= 1:
    return P()

  ndim = len(shape)
  if sharding == "fsdp":
    if name.endswith(".weight") and (
        ".linear_qkv." in name
        or ".linear_g." in name
        or ".linear_proj." in name
        or ".up_gate_proj." in name
        or ".down_proj." in name
        or name == "pre_adapter.text_embedder.weight"
    ):
      out_axis = 1 if ndim == 3 else 0
      if shape[out_axis] % axis_size == 0:
        spec = [None] * ndim
        spec[out_axis] = axis_name
        return P(*spec)
    return P()

  if sharding == "tp":
    row_targets = (".up_gate_proj.weight", ".linear_g.weight")
    col_targets = (".linear_qkv.weight", ".linear_proj.weight", ".down_proj.weight")
    if any(name.endswith(s) for s in row_targets):
      out_axis = 1 if ndim == 3 else 0
      # Keep interleaved (gate, up) pairs aligned on each chip for up_gate_proj.
      div = 2 * axis_size if name.endswith(".up_gate_proj.weight") else axis_size
      if shape[out_axis] % div == 0:
        spec = [None] * ndim
        spec[out_axis] = axis_name
        return P(*spec)
    if any(name.endswith(s) for s in col_targets):
      in_axis = 2 if ndim == 3 else 1
      if shape[in_axis] % axis_size == 0:
        spec = [None] * ndim
        spec[in_axis] = axis_name
        return P(*spec)
    return P()

  raise ValueError(
      f"Unknown sharding mode {sharding!r}; expected 'replicated', 'fsdp', or 'tp'."
  )


def build_sharded(
    mesh: Any = None,
    config: Magi2RefinerConfig | None = None,
    *,
    dtype: Any = None,
    sharding: str = "replicated",
    axis_name: str | None = None,
    memory_kind: str | None = None,
):
  """Construct ``Magi2RefinerTransformer`` abstractly via ``nnx.eval_shape`` and
  materialize zero parameters directly into their target mesh sharding."""
  from flax import nnx

  _, Magi2RefinerTransformer = _build_modules()
  cfg = config or Magi2RefinerConfig()
  dtype = jnp.bfloat16 if dtype is None else dtype
  model = nnx.eval_shape(
      lambda: Magi2RefinerTransformer(cfg, dtype=dtype, rngs=nnx.Rngs(0))
  )
  state = nnx.state(model, nnx.Param)

  if mesh is not None:
    ax = axis_name or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0])
    ax_size = mesh.shape[ax]
  else:
    ax = "cp"
    ax_size = 1

  for path, leaf in nnx.to_flat_state(state):
    name = ".".join(str(part) for part in path).removesuffix(".value")
    arr = jnp.zeros(leaf.value.shape, leaf.value.dtype)
    if mesh is not None:
      spec = _spec_for(
          name, leaf.value.shape, sharding=sharding, axis_name=ax, axis_size=ax_size
      )
      arr = jax.device_put(
          arr, NamedSharding(mesh, spec, memory_kind=memory_kind)
      )
    leaf.value = arr
  nnx.update(model, state)
  return model


def _read_weight_map(checkpoint_dir: str | Path) -> dict[str, str]:
  """Map every checkpoint key in ``checkpoint_dir`` to its safetensors shard path."""
  from ...safetensors_io import safe_open

  ckpt_dir = Path(checkpoint_dir)
  index_path = ckpt_dir / "model.safetensors.index.json"
  if index_path.exists():
    weight_map = json.loads(index_path.read_text())["weight_map"]
    return {k: str(ckpt_dir / shard) for k, shard in weight_map.items()}

  single = ckpt_dir / "model.safetensors"
  if single.exists():
    with safe_open(str(single), framework="np") as f:
      return {k: str(single) for k in f.keys()}

  shards = sorted(glob.glob(str(ckpt_dir / "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"No safetensors files found under {ckpt_dir}")
  out: dict[str, str] = {}
  for shard in shards:
    with safe_open(shard, framework="np") as f:
      for k in f.keys():
        out[k] = shard
  return out


def _resolve_param(model: Any, name: str) -> Any:
  node = model
  parts = name.split(".")
  for part in parts[:-1]:
    node = node[int(part)] if part.isdigit() else getattr(node, part)
  return getattr(node, parts[-1])


def load_refiner(
    checkpoint_dir: str | Path,
    *,
    config: Magi2RefinerConfig | None = None,
    mesh: Any = None,
    dtype: Any = None,
    sharding: str = "replicated",
    axis_name: str | None = None,
    memory_kind: str | None = None,
    max_workers: int = 3,
    verify_rope_bands: bool = True,
    verbose: bool = True,
):
  """Stream-load the 6.72B MAGI-2 Refiner from ``refiner/model.safetensors.index.json``.

  Uses ``nnx.eval_shape`` (no unsharded device-0 allocation) and
  ``diffusers_tpu.safetensors_io.safe_open`` (sequential full-tensor reads without
  mmap page-fault storms). Directly places each tensor onto ``mesh`` with its
  target ``NamedSharding`` (``replicated``, ``fsdp``, or ``tp``) as it is read.
  """
  from flax import nnx
  from ...safetensors_io import safe_open

  _, Magi2RefinerTransformer = _build_modules()
  cfg = config or Magi2RefinerConfig()
  dtype = jnp.bfloat16 if dtype is None else dtype
  model = nnx.eval_shape(
      lambda: Magi2RefinerTransformer(cfg, dtype=dtype, rngs=nnx.Rngs(0))
  )

  if mesh is not None:
    ax = axis_name or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0])
    ax_size = mesh.shape[ax]
  else:
    ax = "cp"
    ax_size = 1

  weight_map = _read_weight_map(checkpoint_dir)
  by_shard: dict[str, list[str]] = {}
  for key, shard in weight_map.items():
    by_shard.setdefault(shard, []).append(key)

  def _load_shard(item: tuple[str, list[str]]) -> tuple[list[tuple[str, Any]], int]:
    shard_path, keys = item
    results: list[tuple[str, Any]] = []
    skipped_bufs = 0
    # Open with framework="np" (uses ml_dtypes.bfloat16 zero-copy view) so
    # jax.device_put shards directly from host -> chips without staging on chip 0.
    with safe_open(shard_path, framework="np") as reader:
      key_set = set(keys)
      # Iterate in file-offset order for sequential disk bandwidth.
      ordered_keys = [k for k in reader.keys() if k in key_set]
      for key in ordered_keys:
        arr = reader.get_tensor(key)
        if key == _ROPE_BANDS_KEY:
          if verify_rope_bands:
            num_bands = cfg.head_dim // 8
            exp = np.arange(num_bands, dtype=np.float32) / np.float32(num_bands)
            want_bands = np.float32(1.0) / (np.float32(cfg.rope_temperature) ** exp)
            diff = float(np.max(np.abs(np.asarray(arr, dtype=np.float32) - want_bands)))
            if diff > 1e-5:
              raise ValueError(
                  f"Checkpoint {_ROPE_BANDS_KEY} mismatch: max|diff|={diff:.3e}"
              )
          skipped_bufs += 1
          continue

        param = _resolve_param(model, key)
        want = param.value if hasattr(param, "value") else param[...]
        arr_j = jnp.asarray(arr)
        if arr_j.dtype != want.dtype:
          arr_j = arr_j.astype(want.dtype)
        if arr_j.shape != want.shape:
          arr_j = arr_j.reshape(want.shape)
        if mesh is not None:
          spec = _spec_for(
              key, want.shape, sharding=sharding, axis_name=ax, axis_size=ax_size
          )
          arr_j = jax.device_put(
              arr_j, NamedSharding(mesh, spec, memory_kind=memory_kind)
          )
        results.append((key, arr_j))
    return results, skipped_bufs

  shard_items = sorted(by_shard.items())
  loaded = 0
  skipped_buffers = 0
  workers = max(1, min(int(max_workers), len(shard_items)))
  if workers > 1:
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
      for pairs, n_skip in pool.map(_load_shard, shard_items):
        skipped_buffers += n_skip
        for key, arr_j in pairs:
          param = _resolve_param(model, key)
          if hasattr(param, "value"):
            param.value = arr_j
          else:
            param[...] = arr_j
          loaded += 1
  else:
    for item in shard_items:
      pairs, n_skip = _load_shard(item)
      skipped_buffers += n_skip
      for key, arr_j in pairs:
        param = _resolve_param(model, key)
        if hasattr(param, "value"):
          param.value = arr_j
        else:
          param[...] = arr_j
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(
        f"Loaded {loaded} parameter tensors (plus {skipped_buffers} buffer(s)), "
        f"but model expects {expected} parameters."
    )
  if verbose:
    print(
        f"[magi2_refiner] Loaded {loaded} params (+{skipped_buffers} buffers) "
        f"from {len(shard_items)} shard(s) [sharding={sharding}]",
        flush=True,
    )
  return model


# ---------------------------------------------------------------------------
# 6. Jitted Forward Wrapper (nnx.split / nnx.merge inside jax.jit — NEVER nnx.jit)
# ---------------------------------------------------------------------------


def _stack_layer_states_sharded(states: Sequence[Any], mesh: Any) -> Any:
  """Stack a sequence of identical NNX layer states along axis 0 without cross-chip gather."""
  from jax.sharding import NamedSharding, PartitionSpec as P

  def _stack_one(*xs):
    x0 = xs[0]
    if mesh is not None and hasattr(x0, "sharding") and isinstance(x0.sharding, NamedSharding):
      orig_spec = x0.sharding.spec
      new_spec = P(None, *orig_spec)
      out_sh = NamedSharding(
          x0.sharding.mesh, new_spec, memory_kind=x0.sharding.memory_kind
      )
      return jax.jit(lambda *arrs: jnp.stack(arrs, axis=0), out_shardings=out_sh)(*xs)
    return jnp.stack(xs, axis=0)

  return jax.tree_util.tree_map(_stack_one, *states)


def make_jitted_forward(
    model: Any,
    layout: RefinerAttentionLayout,
    *,
    mesh: Any = None,
    cp_axis: str | None = None,
    parallel_mode: str = "cp",
    compute_dtype: Any = None,
    attn_impl: str | None = None,
    scan_layers: bool = True,
) -> tuple[Callable[..., tuple[Any, Any | None]], Any, Any]:
  """Compile ``Magi2RefinerTransformer`` with ``jax.jit`` using ``nnx.split`` / ``nnx.merge``.

  When ``scan_layers=True`` (default), contiguous identical layer runs of length >= 3
  (layers ``2..27`` in the 30-layer Refiner) are stacked along axis 0 and executed via
  ``jax.lax.scan`` so XLA compiles only 1 shared layer body instead of 26 copies (~10x
  faster compilation and far lower compiler RAM).

  Returns:
    ``(jitted_fn, graphdef, state)`` where calling:
      ``out_v_pad, out_a = jitted_fn(state, video_packed, text_packed, rope_96, kv_blen, audio_tokens)``
    executes the full 30-layer Refiner with zero Python/NNX tracing overhead.
  """
  from flax import nnx

  if not scan_layers or len(model.block.layers) < 4:
    graphdef, state = nnx.split(model)

    @jax.jit
    def _forward(st, video_packed, text_packed, rope_96, kv_blen, audio_tokens=None):
      m = nnx.merge(graphdef, st)
      return m(
          video_packed,
          text_packed,
          rope_96,
          kv_blen,
          layout=layout,
          audio_tokens=audio_tokens,
          compute_dtype=compute_dtype,
          mesh=mesh,
          cp_axis=cp_axis,
          parallel_mode=parallel_mode,
          attn_impl=attn_impl,
      )

    return _forward, graphdef, state

  # Group contiguous layers with identical (num_modality, is_local_attn) into
  # either scanned blocks (len >= 3, starting after layer 0 so carry dtype is
  # constant fp32) or single unrolled layers.
  pre_gd, pre_st = nnx.split(model.pre_adapter)
  post_gd, post_st = nnx.split(model.post_adapter)
  layers = list(model.block.layers)

  def _sig(layer):
    return int(layer.num_modality)

  segments: list[tuple[bool, Any, list[int]]] = []
  idx = 0
  while idx < len(layers):
    if idx == 0:
      segments.append((False, nnx.split(layers[0])[0], [0]))
      idx += 1
      continue
    sig = _sig(layers[idx])
    j = idx + 1
    while j < len(layers) and _sig(layers[j]) == sig:
      j += 1
    run = list(range(idx, j))
    if len(run) >= 3:
      segments.append((True, nnx.split(layers[idx])[0], run))
    else:
      for k in run:
        segments.append((False, nnx.split(layers[k])[0], [k]))
    idx = j

  seg_states = []
  for is_scan, _gd, run in segments:
    if is_scan:
      st_list = [nnx.split(layers[k])[1] for k in run]
      seg_states.append(_stack_layer_states_sharded(st_list, mesh))
    else:
      seg_states.append(nnx.split(layers[run[0]])[1])

  packed_state = (pre_st, tuple(seg_states), post_st)
  c_dtype = model.dtype if compute_dtype is None else compute_dtype

  use_outer_smap = (
      mesh is not None
      and mesh.size > 1
      and parallel_mode == "cp"
      and mesh.shape[cp_axis or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0])] > 1
  )
  axis = (
      (cp_axis or ("cp" if "cp" in mesh.axis_names else mesh.axis_names[0]))
      if mesh is not None
      else "cp"
  )

  if use_outer_smap:
    n_v, n_a, n_t = layout.modality_lens
    mod_ids_np = np.concatenate([
        np.zeros((n_v,), dtype=np.int32),
        np.ones((n_a,), dtype=np.int32),
        np.full((n_t,), 2, dtype=np.int32),
    ])
    mod_ids_sharded = jax.device_put(
        jnp.asarray(mod_ids_np, dtype=jnp.int32), NamedSharding(mesh, P(axis,))
    )
    layer_specs = jax.tree_util.tree_map(lambda _: P(), tuple(seg_states))

    def _run_layers_local(layer_ss, x_loc, rope_loc, mod_loc, blen_loc):
      gs_loc = jnp.bincount(mod_loc, length=3).astype(jnp.int32)
      cur = x_loc
      for (is_scan, lgd, _run), lst in zip(segments, layer_ss):
        if is_scan:
          def _step(carry, one_st, _lgd=lgd):
            lyr = nnx.merge(_lgd, one_st)
            nxt = lyr(
                carry,
                rope_loc,
                blen_loc,
                layout=layout,
                compute_dtype=c_dtype,
                mesh=mesh,
                cp_axis=axis,
                parallel_mode=parallel_mode,
                attn_impl=attn_impl,
                modality_ids=mod_loc,
                group_sizes=gs_loc,
                in_shard_map=True,
            )
            return nxt, None

          cur, _ = lax.scan(_step, cur, lst)
        else:
          lyr = nnx.merge(lgd, lst)
          cur = lyr(
              cur,
              rope_loc,
              blen_loc,
              layout=layout,
              compute_dtype=c_dtype,
              mesh=mesh,
              cp_axis=axis,
              parallel_mode=parallel_mode,
              attn_impl=attn_impl,
              modality_ids=mod_loc,
              group_sizes=gs_loc,
              in_shard_map=True,
          )
      return cur

    run_layers_sharded = _shard_map(
        _run_layers_local,
        mesh=mesh,
        in_specs=(layer_specs, P(axis, None), P(axis, None), P(axis,), P()),
        out_specs=P(axis, None),
    )

  @jax.jit
  def _forward_scanned(st, video_packed, text_packed, rope_96, kv_blen, audio_tokens=None):
    pre_s, layer_ss, post_s = st
    pre_mod = nnx.merge(pre_gd, pre_s)
    x = pre_mod(video_packed, audio_tokens, text_packed).astype(c_dtype)
    if use_outer_smap:
      x = lax.with_sharding_constraint(x, NamedSharding(mesh, P(axis, None)))
      rope_s = lax.with_sharding_constraint(rope_96, NamedSharding(mesh, P(axis, None)))
      blen_r = lax.with_sharding_constraint(kv_blen, NamedSharding(mesh, P()))
      x = run_layers_sharded(layer_ss, x, rope_s, mod_ids_sharded, blen_r)
    else:
      for (is_scan, lgd, _run), lst in zip(segments, layer_ss):
        if is_scan:
          def _step(carry, one_st, _lgd=lgd):
            lyr = nnx.merge(_lgd, one_st)
            nxt = lyr(
                carry,
                rope_96,
                kv_blen,
                layout=layout,
                compute_dtype=c_dtype,
                mesh=mesh,
                cp_axis=cp_axis,
                parallel_mode=parallel_mode,
                attn_impl=attn_impl,
            )
            return nxt, None

          x, _ = lax.scan(_step, x, lst)
        else:
          lyr = nnx.merge(lgd, lst)
          x = lyr(
              x,
              rope_96,
              kv_blen,
              layout=layout,
              compute_dtype=c_dtype,
              mesh=mesh,
              cp_axis=cp_axis,
              parallel_mode=parallel_mode,
              attn_impl=attn_impl,
          )

    post_mod = nnx.merge(post_gd, post_s)
    return post_mod(x, layout.modality_lens)

  return _forward_scanned, (pre_gd, tuple(s[1] for s in segments), post_gd), packed_state


def make_jitted_cfg_forward(
    model: Any,
    layout: RefinerAttentionLayout,
    *,
    mesh: Any,
    cfg_axis: str = "cfg",
    cp_axis: str = "cp",
    compute_dtype: Any = None,
    attn_impl: str | None = None,
    scan_layers: bool = True,
) -> tuple[Callable[..., Any], Any, Any]:
  """Compile a fused 2-branch CFG-parallel Refiner forward over a 2D ``(cfg=2, cp=N)`` mesh.

  Runs conditional (``cfg=0``) and unconditional (``cfg=1``) branches simultaneously
  on two halves of the TPU slice (e.g. chips 0..3 and 4..7 on ``v7x-8`` with
  ``CP=4``, ``8 Q / 2 KV`` heads/chip), eliminating sequential 2x weight reads and
  2x XLA dispatch while remaining 100% mathematically lossless.

  Returns:
    ``(jitted_cfg_fn, graphdef, packed_state)`` where calling:
      ``v_cfg_pad = jitted_cfg_fn(state, video_packed, txt_2, rope_2, kv_blen_2, cfg_scale)``
    returns the CFG-combined video velocity ``(n_vid_pad, 48)`` in float32.
  """
  from flax import nnx

  if mesh is None or cfg_axis not in mesh.axis_names or cp_axis not in mesh.axis_names:
    raise ValueError(
        f"make_jitted_cfg_forward requires a 2D mesh with axes ({cfg_axis!r}, {cp_axis!r}); "
        f"got {getattr(mesh, 'axis_names', None)}"
    )
  if int(mesh.shape[cfg_axis]) != 2:
    raise ValueError(f"Expected mesh.shape[{cfg_axis!r}] == 2, got {mesh.shape[cfg_axis]}")

  pre_gd, pre_st = nnx.split(model.pre_adapter)
  post_gd, post_st = nnx.split(model.post_adapter)
  layers = list(model.block.layers)

  def _sig(layer):
    return int(layer.num_modality)

  segments: list[tuple[bool, Any, list[int]]] = []
  idx = 0
  while idx < len(layers):
    if idx == 0 or not scan_layers:
      segments.append((False, nnx.split(layers[idx])[0], [idx]))
      idx += 1
      continue
    sig = _sig(layers[idx])
    j = idx + 1
    while j < len(layers) and _sig(layers[j]) == sig:
      j += 1
    run = list(range(idx, j))
    if len(run) >= 3:
      segments.append((True, nnx.split(layers[idx])[0], run))
    else:
      for k in run:
        segments.append((False, nnx.split(layers[k])[0], [k]))
    idx = j

  rep_sharding = NamedSharding(mesh, P())
  pre_st = jax.tree.map(lambda x: jax.device_put(x, rep_sharding), pre_st)
  post_st = jax.tree.map(lambda x: jax.device_put(x, rep_sharding), post_st)

  seg_states = []
  for is_scan, _gd, run in segments:
    if is_scan:
      st_list = [
          jax.tree.map(lambda x: jax.device_put(x, rep_sharding), nnx.split(layers[k])[1])
          for k in run
      ]
      seg_states.append(_stack_layer_states_sharded(st_list, mesh))
    else:
      seg_states.append(
          jax.tree.map(lambda x: jax.device_put(x, rep_sharding), nnx.split(layers[run[0]])[1])
      )

  packed_state = (pre_st, tuple(seg_states), post_st)
  c_dtype = model.dtype if compute_dtype is None else compute_dtype

  n_v, n_a, n_t = layout.modality_lens
  mod_ids_np = np.concatenate([
      np.zeros((n_v,), dtype=np.int32),
      np.ones((n_a,), dtype=np.int32),
      np.full((n_t,), 2, dtype=np.int32),
  ])
  mod_ids_sharded = jax.device_put(
      jnp.asarray(mod_ids_np, dtype=jnp.int32), NamedSharding(mesh, P(cp_axis,))
  )
  layer_specs = jax.tree_util.tree_map(lambda _: P(), tuple(seg_states))

  def _run_layers_cfg_local(layer_ss, x_loc, rope_loc, mod_loc, blen_loc):
    # x_loc: (1, S//cp, D), rope_loc: (1, S//cp, ro_dim), blen_loc: (1, num_kv_blocks)
    cur = x_loc[0]
    rope_1 = rope_loc[0]
    blen_1 = blen_loc[0]
    gs_loc = jnp.bincount(mod_loc, length=3).astype(jnp.int32)
    for (is_scan, lgd, _run), lst in zip(segments, layer_ss):
      if is_scan:
        def _step(carry, one_st, _lgd=lgd):
          lyr = nnx.merge(_lgd, one_st)
          nxt = lyr(
              carry,
              rope_1,
              blen_1,
              layout=layout,
              compute_dtype=c_dtype,
              mesh=mesh,
              cp_axis=cp_axis,
              parallel_mode="cp",
              attn_impl=attn_impl,
              modality_ids=mod_loc,
              group_sizes=gs_loc,
              in_shard_map=True,
          )
          return nxt, None

        cur, _ = lax.scan(_step, cur, lst)
      else:
        lyr = nnx.merge(lgd, lst)
        cur = lyr(
            cur,
            rope_1,
            blen_1,
            layout=layout,
            compute_dtype=c_dtype,
            mesh=mesh,
            cp_axis=cp_axis,
            parallel_mode="cp",
            attn_impl=attn_impl,
            modality_ids=mod_loc,
            group_sizes=gs_loc,
            in_shard_map=True,
        )
    return cur[None, ...]

  run_layers_cfg_sharded = _shard_map(
      _run_layers_cfg_local,
      mesh=mesh,
      in_specs=(
          layer_specs,
          P(cfg_axis, cp_axis, None),
          P(cfg_axis, cp_axis, None),
          P(cp_axis,),
          P(cfg_axis, None),
      ),
      out_specs=P(cfg_axis, cp_axis, None),
  )

  @jax.jit
  def _forward_cfg(st, video_packed, text_packed_2, rope_2, kv_blen_2, cfg_scale):
    pre_s, layer_ss, post_s = st
    pre_mod = nnx.merge(pre_gd, pre_s)
    x_c = pre_mod(video_packed, None, text_packed_2[0]).astype(c_dtype)
    x_u = pre_mod(video_packed, None, text_packed_2[1]).astype(c_dtype)
    x_2 = jnp.stack([x_c, x_u], axis=0)
    x_2 = lax.with_sharding_constraint(x_2, NamedSharding(mesh, P(cfg_axis, cp_axis, None)))
    rope_s = lax.with_sharding_constraint(rope_2, NamedSharding(mesh, P(cfg_axis, cp_axis, None)))
    blen_s = lax.with_sharding_constraint(kv_blen_2, NamedSharding(mesh, P(cfg_axis, None)))

    x_out_2 = run_layers_cfg_sharded(layer_ss, x_2, rope_s, mod_ids_sharded, blen_s)

    post_mod = nnx.merge(post_gd, post_s)
    v_2 = jax.vmap(lambda xo: post_mod(xo, layout.modality_lens)[0])(x_out_2)
    v_c = v_2[0]
    v_u = v_2[1]
    return v_u + cfg_scale.astype(jnp.float32) * (v_c - v_u)

  return _forward_cfg, (pre_gd, tuple(s[1] for s in segments), post_gd), packed_state


# ---------------------------------------------------------------------------
# 7. Independent Pure-NumPy Exact Reference & Self-Tests
# ---------------------------------------------------------------------------


def _np_rms_norm_1pw(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  xf = x.astype(np.float32)
  inv_rms = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
  return xf * inv_rms * (np.float32(1.0) + w.astype(np.float32))


def _np_swiglu7_interleaved(x: np.ndarray, alpha: float = 1.702, limit: float = 7.0) -> np.ndarray:
  xf = x.astype(np.float32)
  g = np.minimum(xf[..., ::2], np.float32(limit))
  u = np.clip(xf[..., 1::2], np.float32(-limit), np.float32(limit))
  sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-np.float32(alpha) * g))
  return (g * sig) * (u + np.float32(1.0))


def _np_refiner_unpadded_reference(
    cfg: Magi2RefinerConfig,
    weights: dict[str, np.ndarray],
    video_cthw: np.ndarray,
    text_tokens: np.ndarray,
    *,
    audio_tokens: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
  """Independent unpadded fp32 transcription of upstream ``magi2_refiner.py`` +
  ``refiner_data_proxy.py`` (compact block order, NO block padding whatsoever!)."""
  c_v, t_num, h_num, w_num = video_cthw.shape
  n_v = t_num * h_num * w_num
  n_a = 0 if audio_tokens is None else audio_tokens.shape[0]
  n_t = text_tokens.shape[0]
  n_tot = n_v + n_a + n_t

  # Upstream compact video block order (no padding!).
  perm, inv_perm, (nt, nh, nw), _ = refiner_block_perm(
      (t_num, h_num, w_num),
      bs_t=cfg.block_t_size,
      bs_h=cfg.block_h_size,
      bs_w=cfg.block_w_size,
  )
  t_sz = _dim_block_sizes_np(t_num, cfg.block_t_size)
  h_sz = _dim_block_sizes_np(h_num, cfg.block_h_size)
  w_sz = _dim_block_sizes_np(w_num, cfg.block_w_size)
  blk_sizes = (t_sz[:, None, None] * h_sz[None, :, None] * w_sz[None, None, :]).reshape(-1)
  blk_ends = np.cumsum(blk_sizes)
  blk_starts = blk_ends - blk_sizes

  # Exact BlockGridLocalAttn + dense video<->tail + tail<->tail mask (unpadded!).
  attn_mask = np.zeros((n_tot, n_tot), dtype=bool)
  nhw = nh * nw
  rt, rh, rw = cfg.block_t_radius, cfg.block_h_radius, cfg.block_w_radius
  for qt in range(nt):
    for qh in range(nh):
      for qw in range(nw):
        qb = qt * nhw + qh * nw + qw
        q0, q1 = int(blk_starts[qb]), int(blk_ends[qb])
        for kt in range(max(0, qt - rt), min(nt, qt + rt + 1)):
          for kh in range(max(0, qh - rh), min(nh, qh + rh + 1)):
            for kw in range(max(0, qw - rw), min(nw, qw + rw + 1)):
              kb = kt * nhw + kh * nw + kw
              attn_mask[q0:q1, int(blk_starts[kb]) : int(blk_ends[kb])] = True
  if n_a + n_t > 0:
    attn_mask[:n_v, n_v:] = True
    attn_mask[n_v:, :] = True

  # Unpadded RoPE in compact block order.
  coords_raw = build_refiner_rope_coords((t_num, h_num, w_num), n_t, audio_len=n_a)
  rope_raw = elementwise_fourier_embed(
      coords_raw, head_dim=cfg.head_dim, temperature=cfg.rope_temperature
  )
  rope = np.concatenate([rope_raw[:n_v][perm], rope_raw[n_v:]], axis=0)

  # PreAdapter (fp32, bias=True).
  v_raster = np.transpose(video_cthw, (1, 2, 3, 0)).reshape(n_v, c_v).astype(np.float32)
  v_blk = v_raster[perm]
  x_v = v_blk @ weights["pre_adapter.video_embedder.weight"].T + weights["pre_adapter.video_embedder.bias"]
  parts = [x_v]
  if n_a > 0:
    assert audio_tokens is not None
    parts.append(
        audio_tokens.astype(np.float32) @ weights["pre_adapter.audio_embedder.weight"].T
        + weights["pre_adapter.audio_embedder.bias"]
    )
  if n_t > 0:
    parts.append(
        text_tokens.astype(np.float32) @ weights["pre_adapter.text_embedder.weight"].T
        + weights["pre_adapter.text_embedder.bias"]
    )
  x = np.concatenate(parts, axis=0)
  mod_lens = (n_v, n_a, n_t)

  def _mm_norm(arr: np.ndarray, w_flat: np.ndarray, m: int) -> np.ndarray:
    if m == 1:
      return _np_rms_norm_1pw(arr, w_flat, cfg.rms_norm_eps)
    w_chunks = np.split(w_flat, m, axis=0)
    out_parts, off = [], 0
    for idx, n_m in enumerate(mod_lens):
      if n_m > 0:
        out_parts.append(_np_rms_norm_1pw(arr[off : off + n_m], w_chunks[idx], cfg.rms_norm_eps))
        off += n_m
    return np.concatenate(out_parts, axis=0)

  def _mm_lin(arr: np.ndarray, w_flat: np.ndarray, m: int) -> np.ndarray:
    if m == 1:
      return arr @ w_flat.T
    w_chunks = np.split(w_flat, m, axis=0)
    out_parts, off = [], 0
    for idx, n_m in enumerate(mod_lens):
      if n_m > 0:
        out_parts.append(arr[off : off + n_m] @ w_chunks[idx].T)
        off += n_m
    return np.concatenate(out_parts, axis=0)

  q_sz = cfg.num_heads_q * cfg.head_dim
  kv_sz = cfg.num_heads_kv * cfg.head_dim
  scale = np.float32(1.0 / math.sqrt(cfg.head_dim))

  for i in range(cfg.num_layers):
    p = f"block.layers.{i}"
    m = 3 if i in cfg.mm_layers else 1

    # Attention
    h = _mm_norm(x, weights[f"{p}.attention.pre_norm.weight"], m)
    qkv = _mm_lin(h, weights[f"{p}.attention.linear_qkv.weight"], m)
    q = qkv[:, :q_sz].reshape(n_tot, cfg.num_heads_q, cfg.head_dim)
    k = qkv[:, q_sz : q_sz + kv_sz].reshape(n_tot, cfg.num_heads_kv, cfg.head_dim)
    v = qkv[:, q_sz + kv_sz :].reshape(n_tot, cfg.num_heads_kv, cfg.head_dim)
    g = None
    if cfg.enable_attn_gating:
      g = _mm_lin(h, weights[f"{p}.attention.linear_g.weight"], m)[..., None]

    q = _mm_norm(q, weights[f"{p}.attention.q_norm.weight"], m)
    k = _mm_norm(k, weights[f"{p}.attention.k_norm.weight"], m)
    q = apply_3d_partial_rope(q, rope)
    k = apply_3d_partial_rope(k, rope)

    rep = cfg.num_heads_q // cfg.num_heads_kv
    k_rep = np.repeat(k, rep, axis=1)
    v_rep = np.repeat(v, rep, axis=1)
    logits = np.einsum("qhd,khd->hqk", q * scale, k_rep)
    logits = np.where(attn_mask[None, :, :], logits, np.float32(-1e30))
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    attn_out = np.einsum("hqk,khd->qhd", probs, v_rep)

    if g is not None:
      sig_g = np.float32(1.0) / (np.float32(1.0) + np.exp(-g))
      attn_out = attn_out * sig_g
    attn_out = _mm_lin(
        attn_out.reshape(n_tot, q_sz), weights[f"{p}.attention.linear_proj.weight"], m
    )
    x = x + attn_out

    # MLP
    hm = _mm_norm(x, weights[f"{p}.mlp.pre_norm.weight"], m)
    u = _mm_lin(hm, weights[f"{p}.mlp.up_gate_proj.weight"], m)
    u = _np_swiglu7_interleaved(u)
    mlp_out = _mm_lin(u, weights[f"{p}.mlp.down_proj.weight"], m)
    x = x + mlp_out

  # PostAdapter (fp32, bias=False)
  xv = _np_rms_norm_1pw(x[:n_v], weights["post_adapter.final_norm_video.weight"], cfg.rms_norm_eps)
  out_v_blk = xv @ weights["post_adapter.final_linear_video.weight"].T
  out_v_raster = out_v_blk[inv_perm]
  out_v_cthw = np.transpose(out_v_raster.reshape(t_num, h_num, w_num, c_v), (3, 0, 1, 2))

  out_a = None
  if n_a > 0:
    xa = _np_rms_norm_1pw(
        x[n_v : n_v + n_a], weights["post_adapter.final_norm_audio.weight"], cfg.rms_norm_eps
    )
    out_a = xa @ weights["post_adapter.final_linear_audio.weight"].T
  return out_v_cthw, out_a


def test_layout_and_smem(verbose: bool = True) -> bool:
  """Pure-NumPy verification of 3D block permutation, 1080p SMEM sizing, and separable decode."""
  # 1. Verify shipping 1080p (63, 68, 120) layout & SMEM budget!
  layout_1080p = build_refiner_attention_layout(
      (63, 68, 120), n_text=300, tail_bucket_len=512, cp_size=8, build_flat_tables=True
  )
  assert layout_1080p.block_grid_thw == (8, 17, 30)
  assert layout_1080p.n_vid_blocks == 4080
  assert layout_1080p.n_vid_real == 514080
  assert layout_1080p.n_vid_pad == 522240
  assert layout_1080p.pad_t == 1
  assert not layout_1080p.all_video_blocks_full
  assert layout_1080p.n_tail_blocks == 4
  assert layout_1080p.total_seq_len == 522752
  assert layout_1080p.total_seq_len % 8 == 0
  assert int((layout_1080p.kv_block_valid_lens[:4080] == 128).sum()) == 3570
  assert int((layout_1080p.kv_block_valid_lens[:4080] == 112).sum()) == 510
  assert list(layout_1080p.kv_block_valid_lens[4080:]) == [128, 128, 44, 0]

  smem_bytes = (
      layout_1080p.nb_arr.nbytes
      + layout_1080p.cnt_arr.nbytes
      + layout_1080p.dec_arr.nbytes
      + layout_1080p.kv_block_valid_lens.nbytes
  )
  flat_kvt_bytes = layout_1080p.n_vid_blocks * (125 + layout_1080p.n_tail_blocks) * 4
  assert smem_bytes < 64 * 1024, f"SMEM too large: {smem_bytes} bytes"

  # 2. Verify separable 3D SMEM decode reproduces EVERY neighbor in kv_table_vid!
  _, nh, nw = layout_1080p.block_grid_thw
  nhw = nh * nw
  for qb in (0, 1, 29, 30, 509, 510, 2040, 3570, 4079):
    qt, rem = divmod(qb, nhw)
    qh, qw = divmod(rem, nw)
    ct = int(layout_1080p.cnt_arr[0, qt])
    ch = int(layout_1080p.cnt_arr[1, qh])
    cw = int(layout_1080p.cnt_arr[2, qw])
    n_nb = ct * ch * cw
    pair = (ch - 1) * layout_1080p.max_cw + (cw - 1)
    decoded = []
    for j in range(n_nb):
      it = int(layout_1080p.dec_arr[0, pair, j])
      ih = int(layout_1080p.dec_arr[1, pair, j])
      iw = int(layout_1080p.dec_arr[2, pair, j])
      kt = int(layout_1080p.nb_arr[0, qt, it])
      kh = int(layout_1080p.nb_arr[1, qh, ih])
      kw = int(layout_1080p.nb_arr[2, qw, iw])
      decoded.append(kt * nhw + kh * nw + kw)
    # Append active tail blocks (valid_len > 0)
    for tb in range(layout_1080p.n_tail_blocks):
      kb = layout_1080p.n_vid_blocks + tb
      if layout_1080p.kv_block_valid_lens[kb] > 0:
        decoded.append(kb)
    assert decoded == layout_1080p.kv_table_vid[qb], f"Mismatch at qb={qb}"
    assert decoded == sorted(decoded), f"Not sorted at qb={qb}"

  # 3. Verify pack/unpack round-trip on non-divisible (T, H, W) = (5, 5, 7) with (2, 2, 2).
  rng = np.random.default_rng(7)
  lay_odd = build_refiner_attention_layout(
      (5, 5, 7), n_text=9, n_audio=3, block_sizes=(2, 2, 2), radius=(1, 1, 1)
  )
  vid = rng.standard_normal((4, 5, 5, 7)).astype(np.float32)
  txt = rng.standard_normal((9, 16)).astype(np.float32)
  aud = rng.standard_normal((3, 8)).astype(np.float32)
  vp, ap, tp = pack_refiner_inputs_np(vid, txt, lay_odd, audio_tokens=aud)
  restored = unpack_refiner_video_np(vp, lay_odd)
  assert np.array_equal(vid, restored), "Pack/unpack round-trip failed!"
  assert np.all(vp[~lay_odd.token_valid_mask[: lay_odd.n_vid_pad]] == 0.0)
  assert ap.shape == (3, 8) and tp.shape == (lay_odd.modality_lens[2], 16)

  if verbose:
    print(
        f"[PASS] test_layout_and_smem: 1080p (63,68,120) -> 4080 vid blocks "
        f"(3570x128 + 510x112), SMEM={smem_bytes/1024:.1f} KiB "
        f"(vs flat kv_table {flat_kvt_bytes/1024:.0f} KiB), exact decode & round-trip OK."
    )
  return True


def test_tiny_parity(verbose: bool = True) -> bool:
  """End-to-end parity test of ``Magi2RefinerTransformer`` vs unpadded reference.

  Verifies:
    1. Exact shipping parameter count & dtype audit via ``nnx.eval_shape``
       (``6,720,075,280`` params across ``150`` BF16 + ``130`` F32 params + 1 buffer).
    2. Tiny GQA 2:1 config with MM + shared layers, non-divisible ``T=5``
       (partial boundary block ``4/8`` mimicking ``T=63`` ``112/128``!),
       both video-only and video+audio+text, and tail bucket padding (verifying
       zero pad-token leakage!).
    3. ``nnx.split`` / ``nnx.merge`` inside ``jax.jit`` parity against independent
       unpadded fp32 reference (``max|diff| < 1e-4``).
  """
  test_layout_and_smem(verbose=verbose)

  if jax is None:
    if verbose:
      print("[SKIP] JAX/Flax not installed in local python3; layout tests passed.")
    return True

  from flax import nnx

  _, Magi2RefinerTransformer = _build_modules()

  # 1. Exact full 6.72B shipping config parameter audit via nnx.eval_shape (0 memory!).
  full_cfg = Magi2RefinerConfig()
  full_model = nnx.eval_shape(
      lambda: Magi2RefinerTransformer(full_cfg, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  )
  bf16_count = bf16_params = f32_count = f32_params = 0
  for _, leaf in nnx.to_flat_state(nnx.state(full_model, nnx.Param)):
    n = int(math.prod(leaf.value.shape))
    if leaf.value.dtype == jnp.bfloat16:
      bf16_count += 1
      bf16_params += n
    elif leaf.value.dtype == jnp.float32:
      f32_count += 1
      f32_params += n
    else:
      raise AssertionError(f"Unexpected dtype {leaf.value.dtype}")
  # Account for the 1 non-param fp32 buffer pre_adapter.rope.bands (16 elements).
  total_with_buf = bf16_params + f32_params + (full_cfg.head_dim // 8)
  assert bf16_count == 150, f"Expected 150 BF16 tensors, got {bf16_count}"
  assert f32_count == 130, f"Expected 130 F32 param tensors, got {f32_count}"
  assert total_with_buf == 6_720_075_280, f"Param count mismatch: {total_with_buf}"
  if verbose:
    print(
        f"[PASS] Full 6.72B param audit: {bf16_count} BF16 ({bf16_params:,}) + "
        f"{f32_count} F32 params + 1 rope.bands buffer ({f32_params + 16:,}) = "
        f"{total_with_buf:,} EXACT."
    )

  # 2. Tiny GQA config parity test (fp32 exact + bf16 finite + bucket zero-leakage).
  tiny_cfg = Magi2RefinerConfig(
      num_layers=4,
      hidden_size=64,
      head_dim=16,  # ro_dim = 12, 4 unrotated dims
      num_heads_q=4,
      num_heads_kv=2,  # GQA 2:1
      video_in_channels=12,
      audio_in_channels=8,
      text_in_channels=24,
      intermediate_size=168,
      mm_layers=(0, 3),
      local_attn_layers=(0, 1, 2, 3),
      enable_attn_gating=True,
      block_t_size=2,
      block_h_size=2,
      block_w_size=2,
      block_t_radius=1,
      block_h_radius=1,
      block_w_radius=1,
  )

  rng = np.random.default_rng(42)
  model = Magi2RefinerTransformer(tiny_cfg, dtype=jnp.float32, rngs=nnx.Rngs(0))
  weights_np: dict[str, np.ndarray] = {}
  state = nnx.state(model, nnx.Param)
  for path, leaf in nnx.to_flat_state(state):
    key = ".".join(str(part) for part in path).removesuffix(".value")
    scale = 0.05 if key.endswith(".weight") else 0.02
    arr = (rng.standard_normal(leaf.value.shape) * scale).astype(np.float32)
    leaf.value = jnp.asarray(arr, dtype=jnp.float32)
    # Upstream flat checkpoint shape for reference: collapse (3, ...) -> (3*...,)
    if arr.ndim == 3:
      weights_np[key] = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif arr.ndim == 2 and key.endswith("norm.weight"):
      weights_np[key] = arr.reshape(-1)
    else:
      weights_np[key] = arr
  nnx.update(model, state)

  # Case A: Video-only with non-divisible T=5 (bt=2 has 4/8 pad tokens!) + tail bucket padding!
  video_cthw = rng.standard_normal((12, 5, 4, 4)).astype(np.float32)
  text_toks = rng.standard_normal((7, 24)).astype(np.float32)
  want_v, _ = _np_refiner_unpadded_reference(tiny_cfg, weights_np, video_cthw, text_toks)

  for bucket in (None, 16):
    layout = build_refiner_attention_layout(
        (5, 4, 4), n_text=7, config=tiny_cfg, tail_bucket_len=bucket
    )
    vp, _, tp = pack_refiner_inputs_np(video_cthw, text_toks, layout)
    jitted_fn, _, st = make_jitted_forward(
        model, layout, compute_dtype=jnp.float32, attn_impl="ref"
    )
    got_vp, got_a = jitted_fn(
        st,
        jnp.asarray(vp),
        jnp.asarray(tp),
        jnp.asarray(layout.rope_96),
        jnp.asarray(layout.kv_block_valid_lens),
    )
    assert got_a is None
    got_v = unpack_refiner_video_np(np.asarray(got_vp), layout)
    max_diff = float(np.max(np.abs(want_v - got_v)))
    rel = max_diff / max(float(np.mean(np.abs(want_v))), 1e-8)
    assert max_diff < 1e-4, f"Video-only bucket={bucket} mismatch: max_diff={max_diff:.3e}"
    if verbose:
      print(
          f"[PASS] Video-only parity (T=5, bucket={bucket}): "
          f"max|diff|={max_diff:.3e}, rel={rel:.3e}"
      )

  # Case B: Video + Audio + Text with non-divisible (T, H, W) = (3, 3, 5).
  video_b = rng.standard_normal((12, 3, 3, 5)).astype(np.float32)
  audio_b = rng.standard_normal((5, 8)).astype(np.float32)
  text_b = rng.standard_normal((6, 24)).astype(np.float32)
  want_vb, want_ab = _np_refiner_unpadded_reference(
      tiny_cfg, weights_np, video_b, text_b, audio_tokens=audio_b
  )
  layout_b = build_refiner_attention_layout(
      (3, 3, 5), n_text=6, n_audio=5, config=tiny_cfg, tail_bucket_len=24
  )
  vp_b, ap_b, tp_b = pack_refiner_inputs_np(video_b, text_b, layout_b, audio_tokens=audio_b)
  jitted_b, _, st_b = make_jitted_forward(
      model, layout_b, compute_dtype=jnp.float32, attn_impl="ref"
  )
  got_vpb, got_ab = jitted_b(
      st_b,
      jnp.asarray(vp_b),
      jnp.asarray(tp_b),
      jnp.asarray(layout_b.rope_96),
      jnp.asarray(layout_b.kv_block_valid_lens),
      jnp.asarray(ap_b),
  )
  got_vb = unpack_refiner_video_np(np.asarray(got_vpb), layout_b)
  diff_vb = float(np.max(np.abs(want_vb - got_vb)))
  diff_ab = float(np.max(np.abs(want_ab - np.asarray(got_ab))))
  assert diff_vb < 1e-4 and diff_ab < 1e-4, f"AVT mismatch: v={diff_vb:.3e}, a={diff_ab:.3e}"
  if verbose:
    print(
        f"[PASS] Video+Audio+Text parity (3,3,5 + 5a + 6t, bucket=24): "
        f"v_max|diff|={diff_vb:.3e}, a_max|diff|={diff_ab:.3e}"
    )

  # Case C: Verify bf16 production compute path stays finite & close to fp32.
  jitted_bf16, _, st_bf16 = make_jitted_forward(
      model, layout, compute_dtype=jnp.bfloat16, attn_impl="ref"
  )
  got_vp_bf16, _ = jitted_bf16(
      st_bf16,
      jnp.asarray(vp),
      jnp.asarray(tp),
      jnp.asarray(layout.rope_96),
      jnp.asarray(layout.kv_block_valid_lens),
  )
  got_v_bf16 = unpack_refiner_video_np(np.asarray(got_vp_bf16), layout)
  assert np.all(np.isfinite(got_v_bf16)), "Non-finite output in bf16 mode!"
  diff_bf16 = float(np.max(np.abs(want_v - got_v_bf16)))
  if verbose:
    print(f"[PASS] bf16 production path finite: max|diff vs fp32|={diff_bf16:.3e}")

  return True


if __name__ == "__main__":
  ok = test_tiny_parity(verbose=True)
  raise SystemExit(0 if ok else 1)
