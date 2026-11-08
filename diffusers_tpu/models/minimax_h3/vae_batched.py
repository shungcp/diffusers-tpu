#!/usr/bin/env python3
"""The VAE's spatial tiles as one batched program across every chip.

`CompiledTileDecoder` compiles the per-tile decode, but the VAE still walks
its tiles one at a time, so one chip works while the others idle and each
tile's conversion back to torch blocks until it is done.

Every interior tile has the same shape, so they stack into one batch.
This replaces `_decode_clip` with a version that collects the tiles,
runs them as a batch sharded over the mesh, and hands the results to the
VAE's own `_stitch_tiles`. The tiling geometry and the blending are the
model's own code, untouched.

In addition, `FastVideoVAEDecoder` / `install_fast_video_vae` batches multiple
temporal clips together (`clips_per_batch=4` -> `112 = 8 * 14` tiles/batch,
eliminating the `28 -> 32` tile padding waste), runs the 36-layer ViT decoder
in `bfloat16` (keeping `rope` and `norm` in `float32`, matching CUDA's
autocast), and executes spatial tile stitching, temporal overlap blending,
ImageNet denormalization, and `uint8` conversion directly on the TPU in a
single compiled JAX kernel.

  python vae_batched.py --num-frames 345
"""

from __future__ import annotations

import argparse
import functools
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import torch

REPO = "MiniMaxAI/MiniMax-H3"
VAE_SPATIAL, VAE_TEMPORAL = 16, 4


def _to_jax(tensor: torch.Tensor) -> jax.Array:
  if isinstance(tensor, jax.Array):
    return tensor      # already uploaded; slicing it is device-local
  t = tensor.detach().contiguous()
  if t.dtype == torch.bfloat16:
    return jnp.asarray(t.view(torch.uint16).numpy().view(jnp.bfloat16))
  if t.is_floating_point():
    return jnp.asarray(t.float().numpy())
  return jnp.asarray(t.numpy())


_JAX_DTYPE = {
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    torch.float16: jnp.float16,
}


def _to_jax_dtype(tensor: torch.Tensor, target_dtype) -> jax.Array:
  t = tensor.detach().contiguous()
  if target_dtype == jnp.bfloat16:
    if t.dtype == torch.bfloat16:
      return jnp.asarray(t.view(torch.uint16).numpy().view(jnp.bfloat16))
    return jnp.asarray(t.float().numpy(), dtype=jnp.bfloat16)
  if t.dtype == torch.bfloat16:
    return jnp.asarray(t.float().numpy(), dtype=target_dtype)
  return jnp.asarray(t.numpy(), dtype=target_dtype)


def _blend_jax(a: jax.Array, b: jax.Array, blend_extent: int, axis: int) -> jax.Array:
  blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)
  positions = jnp.arange(blend_extent, dtype=jnp.float32)
  shape = [1] * a.ndim
  shape[axis] = blend_extent
  weight_a = (1.0 - positions / blend_extent).reshape(shape)
  weight_b = (positions / blend_extent).reshape(shape)

  slice_a = [slice(None)] * a.ndim
  slice_a[axis] = slice(-blend_extent, None)
  slice_b = [slice(None)] * b.ndim
  slice_b[axis] = slice(0, blend_extent)
  blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

  if blend_extent == b.shape[axis]:
    return blended
  slice_rest = [slice(None)] * b.ndim
  slice_rest[axis] = slice(blend_extent, None)
  return jnp.concatenate([blended, b[tuple(slice_rest)]], axis=axis)


def _stitch_clip_jax(
    clip_tiles: jax.Array,
    n_rows: int,
    n_cols: int,
    y_overlaps: tuple[int, ...],
    x_overlaps: tuple[int, ...],
) -> jax.Array:
  clip_f32 = clip_tiles.astype(jnp.float32)
  rows = [
      [clip_f32[i * n_cols + j : i * n_cols + j + 1] for j in range(n_cols)]
      for i in range(n_rows)
  ]
  result_rows = []
  for i, row in enumerate(rows):
    result_row = []
    for j, tile in enumerate(row):
      if i > 0:
        tile = _blend_jax(rows[i - 1][j], tile, y_overlaps[i - 1], axis=-2)
      if j > 0:
        tile = _blend_jax(row[j - 1], tile, x_overlaps[j - 1], axis=-1)
      if i < n_rows - 1:
        tile = tile[..., : -y_overlaps[i], :]
      if j < n_cols - 1:
        tile = tile[..., :, : -x_overlaps[j]]
      result_row.append(tile)
    result_rows.append(jnp.concatenate(result_row, axis=-1))
  return jnp.concatenate(result_rows, axis=-2)


def _stitch_and_postprocess_on_tpu(
    all_tiles: jax.Array,
    *,
    num_chunks: int,
    n_rows: int,
    n_cols: int,
    y_overlaps: tuple[int, ...],
    x_overlaps: tuple[int, ...],
    chunk_num_frames: int,
    frame_pre_padding: int,
    frame_overlap: int,
    has_token_drop: bool,
    pad_frames: int,
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
) -> jax.Array:
  tiles_per_clip = n_rows * n_cols
  decoded_chunks = []
  overlap = None
  sub_chunks = int(has_token_drop) + 1
  for i in range(num_chunks):
    clip_tiles = all_tiles[i * tiles_per_clip : (i + 1) * tiles_per_clip]
    clip = _stitch_clip_jax(clip_tiles, n_rows, n_cols, y_overlaps, x_overlaps)
    for j in range(sub_chunks):
      frame_start = j * chunk_num_frames
      chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
      chunk = chunk[:, :, frame_pre_padding:]
      if j == 0:
        if overlap is not None:
          chunk = _blend_jax(overlap, chunk, frame_overlap, axis=-3)
        decoded_chunks.append(chunk)
      else:
        overlap = chunk
  if overlap is not None:
    decoded_chunks.append(overlap)

  video = jnp.concatenate(decoded_chunks, axis=2)
  if pad_frames > 0:
    video = video[:, :, :-pad_frames]
  mean = jnp.asarray(pixel_mean, dtype=jnp.float32).reshape(1, 3, 1, 1, 1)
  std = jnp.asarray(pixel_std, dtype=jnp.float32).reshape(1, 3, 1, 1, 1)
  video = jnp.clip(video * std + mean, 0.0, 1.0)
  video_hwc = jnp.transpose(video[0], (1, 2, 3, 0))
  return jnp.round(video_hwc * 255.0).astype(jnp.uint8)


class FastVideoVAEDecoder:
  """Multi-clip batched TPU Video VAE decoder + on-TPU stitch/blend/postprocess."""

  def __init__(
      self,
      vae,
      mesh: jax.sharding.Mesh,
      clips_per_batch: int = 4,
      use_bf16: bool = True,
      pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
      pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
  ):
    import torchax

    self._vae = vae
    self._mesh = mesh
    self._env = torchax.default_env()
    self._clips_per_batch = clips_per_batch
    self._compute_dtype = jnp.bfloat16 if use_bf16 else jnp.float32
    self._pixel_mean = tuple(float(x) for x in pixel_mean)
    self._pixel_std = tuple(float(x) for x in pixel_std)
    self._inner = torch.nn.Sequential(vae.post_quant_conv, vae.decoder).eval()
    self._tp_sharding = NamedSharding(mesh, P("tp"))
    self._rep_sharding = NamedSharding(mesh, P())
    self._params = {
        name: jax.device_put(
            _to_jax_dtype(
                tensor,
                jnp.float32 if ("rope" in name or "norm" in name) else self._compute_dtype,
            ),
            self._rep_sharding,
        )
        for name, tensor in list(self._inner.named_parameters())
        + list(self._inner.named_buffers())
    }
    self._decode_cache: dict = {}
    self._stitch_cache: dict = {}
    self.last_u8_frames: torch.Tensor | None = None
    self.compute = 0.0
    self.transfer = 0.0
    self.stitch = 0.0

  def _get_decode_fn(self, shape: tuple[int, ...]):
    fn = self._decode_cache.get(shape)
    if fn is None:
      def call(params, tiles):
        with self._env:
          out = torch.func.functional_call(
              self._inner,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              args=(self._env.j2t_iso(tiles),),
          )
        return self._env.t2j_iso(out)

      fn = jax.jit(
          call,
          in_shardings=(self._rep_sharding, self._tp_sharding),
          out_shardings=self._rep_sharding,
      )
      self._decode_cache[shape] = fn
    return fn

  def _get_stitch_fn(self, key: tuple):
    fn = self._stitch_cache.get(key)
    if fn is None:
      (
          num_chunks,
          n_rows,
          n_cols,
          y_ov,
          x_ov,
          chunk_num_frames,
          frame_pre_padding,
          frame_overlap,
          has_token_drop,
          pad_frames,
      ) = key
      fn = jax.jit(
          functools.partial(
              _stitch_and_postprocess_on_tpu,
              num_chunks=num_chunks,
              n_rows=n_rows,
              n_cols=n_cols,
              y_overlaps=y_ov,
              x_overlaps=x_ov,
              chunk_num_frames=chunk_num_frames,
              frame_pre_padding=frame_pre_padding,
              frame_overlap=frame_overlap,
              has_token_drop=has_token_drop,
              pad_frames=pad_frames,
              pixel_mean=self._pixel_mean,
              pixel_std=self._pixel_std,
          ),
          in_shardings=self._rep_sharding,
          out_shardings=self._rep_sharding,
      )
      self._stitch_cache[key] = fn
    return fn

  def decode(self, z: torch.Tensor, return_dict: bool = True):
    t0 = time.perf_counter()
    vae = self._vae
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
      z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    pad_frames = 0
    if pad_tokens > 0:
      intra_tail = vae.config.clip_length % temporal_ratio
      num_tokens_before_pad = z.shape[2] - pad_tokens
      pad_frames = sum(
          intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
          for k in range(pad_tokens)
      )

    height = z.shape[-2] * vae.spatial_compression_ratio
    width = z.shape[-1] * vae.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    ratio = vae.spatial_compression_ratio
    n_rows, n_cols = len(y_indices), len(x_indices)
    tiles_per_clip = n_rows * n_cols

    z_np = z.detach().float().numpy()
    if self._compute_dtype == jnp.bfloat16:
      z_np = z_np.astype(jnp.bfloat16)

    all_input_tiles = []
    for c_idx in range(num_chunks):
      start = c_idx * tokens_chunk_size
      z_clip = z_np[:, :, start : start + tokens_chunk_size + vae.token_overlap]
      for i_pos, i_len in zip(y_indices, y_lengths):
        for j_pos, j_len in zip(x_indices, x_lengths):
          all_input_tiles.append(
              z_clip[
                  0,
                  :,
                  :,
                  i_pos // ratio : i_pos // ratio + i_len // ratio,
                  j_pos // ratio : j_pos // ratio + j_len // ratio,
              ]
          )
    stacked_in = np.stack(all_input_tiles, axis=0)
    total_tiles = stacked_in.shape[0]
    shards = self._mesh.devices.size
    batch_tiles = self._clips_per_batch * tiles_per_clip

    out_batches = []
    with jax.set_mesh(self._mesh):
      cursor = 0
      while cursor < total_tiles:
        b_slice = stacked_in[cursor : min(cursor + batch_tiles, total_tiles)]
        actual = b_slice.shape[0]
        pad = (-actual) % shards
        if pad:
          b_slice = np.concatenate([b_slice, np.repeat(b_slice[-1:], pad, axis=0)], axis=0)
        b_dev = jax.device_put(b_slice, self._tp_sharding)
        decode_fn = self._get_decode_fn(tuple(b_slice.shape))
        b_out = decode_fn(self._params, b_dev)
        if pad:
          b_out = b_out[:actual]
        out_batches.append(b_out)
        cursor += actual

      all_out_tiles = jnp.concatenate(out_batches, axis=0)
      stitch_key = (
          num_chunks,
          n_rows,
          n_cols,
          tuple(y_overlaps),
          tuple(x_overlaps),
          chunk_num_frames,
          vae.frame_pre_padding,
          vae.frame_overlap,
          bool(token_drop > 0),
          pad_frames,
      )
      stitch_fn = self._get_stitch_fn(stitch_key)
      u8_tpu = stitch_fn(all_out_tiles)

    t_trans0 = time.perf_counter()
    u8_np = np.asarray(jax.device_get(u8_tpu))
    t_trans1 = time.perf_counter()
    self.transfer += t_trans1 - t_trans0
    self.compute += (t_trans0 - t0)
    self.last_u8_frames = torch.from_numpy(u8_np)

    # Return a 1x3x1x1x1 dummy tensor so MiniMaxH3VideoDecodeStep's CPU
    # `(video.float() * pixel_std + pixel_mean).clamp(0, 1)` costs 0 ms;
    # `postprocess_video` then returns `self.last_u8_frames` directly.
    dummy = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    if not return_dict:
      return (dummy,)
    from diffusers.models.autoencoders.vae import DecoderOutput

    return DecoderOutput(sample=dummy)

  def postprocess_video(self, video: torch.Tensor, output_type: str = "pil", **kwargs):
    if self.last_u8_frames is not None:
      u8_frames = self.last_u8_frames
      self.last_u8_frames = None
      return [u8_frames]
    return self._orig_postprocess_video(video, output_type=output_type, **kwargs)


def install_fast_video_vae(
    pipe,
    mesh: jax.sharding.Mesh,
    clips_per_batch: int = 4,
    use_bf16: bool = True,
) -> FastVideoVAEDecoder:
  """Installs FastVideoVAEDecoder on `pipe.vae` and `pipe.video_processor`."""
  pixel_mean = getattr(pipe, "pixel_mean", (0.485, 0.456, 0.406))
  pixel_std = getattr(pipe, "pixel_std", (0.229, 0.224, 0.225))
  fast_vae = FastVideoVAEDecoder(
      pipe.vae,
      mesh,
      clips_per_batch=clips_per_batch,
      use_bf16=use_bf16,
      pixel_mean=pixel_mean,
      pixel_std=pixel_std,
  )
  fast_vae._orig_postprocess_video = pipe.video_processor.postprocess_video
  pipe.vae._decode_clip = fast_vae
  pipe.vae.decode = fast_vae.decode
  pipe.video_processor.postprocess_video = fast_vae.postprocess_video
  return fast_vae


class BatchedTileDecoder:
  """Decodes a clip's spatial tiles as one sharded batch.

  Installed over `vae._decode_clip`, so the temporal chunking above it and
  the blending below it stay the model's own.
  """

  def __init__(self, vae, mesh: jax.sharding.Mesh):
    import torchax

    self._vae = vae
    self._mesh = mesh
    self._env = torchax.default_env()
    self._inner = torch.nn.Sequential(vae.post_quant_conv, vae.decoder).eval()
    self._cache: dict = {}
    self._params = {
        name: jax.device_put(_to_jax(tensor), NamedSharding(mesh, P()))
        for name, tensor in list(self._inner.named_parameters())
        + list(self._inner.named_buffers())
    }
    self.batches: dict[tuple, int] = {}
    self.compute = 0.0
    self.transfer = 0.0
    self.stitch = 0.0
    self.stitch_on_device = False
    self.upload_whole_clip = False
    self.fused_clip = False
    self._fused_cache: dict = {}
    self.to_uint8 = False
    self._pixel_mean = (0.485, 0.456, 0.406)
    self._pixel_std = (0.229, 0.224, 0.225)
    self._stitch_cache: dict = {}

  def _program(self, shape: tuple[int, ...]):
    fn = self._cache.get(shape)
    if fn is None:
      def call(params, tiles):
        # The decoder already takes `(B, C, T, H, W)`, so the tiles go in
        # as the batch rather than through a vmap; `torch.vmap` does not
        # compose with torchax anyway.
        with self._env:
          out = torch.func.functional_call(
              self._inner,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              args=(self._env.j2t_iso(tiles),),
          )
        return self._env.t2j_iso(out)

      # The batch axis is the only one cut: tiles are independent, so this
      # is plain data parallelism with the weights replicated.
      fn = jax.jit(
          call,
          in_shardings=(None, NamedSharding(self._mesh, P("tp"))),
          out_shardings=NamedSharding(self._mesh, P("tp")),
      )
      self._cache[shape] = fn
    return fn


  def _tile_geometry(self, latent_shape):
    """The tile cuts in latent coordinates, plus the stitch's overlaps.

    `_tiles_of` and the stitch each recompute this from the vae; pulling
    it out lets one jitted program hold the whole clip -- cut, decode,
    blend -- as static structure.
    """
    vae = self._vae
    ratio = vae.spatial_compression_ratio
    height = latent_shape[-2] * ratio
    width = latent_shape[-1] * ratio
    y_pos, y_len, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_pos, x_len, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    cuts = tuple(
        (i // ratio, (i + li) // ratio, j // ratio, (j + lj) // ratio)
        for i, li in zip(y_pos, y_len)
        for j, lj in zip(x_pos, x_len)
    )
    return cuts, len(y_pos), len(x_pos), tuple(y_overlaps), tuple(x_overlaps)

  def _fused_program(self, latent_shape, clips: int):
    """Cut, decode and stitch a group of clips in one program.

    Removes the ~540 small eager dispatches (~0.7 s) per decode by fusing
    the whole clip group into one program.

    Off by default: bit-identical and ~6% faster when warm, but no faster
    end to end, and the program is keyed by the whole clip geometry rather
    than tile shape. Each compile takes ~15 s, so a request whose tail clip
    has a different geometry pays it twice.
    """
    key = (tuple(latent_shape), clips)
    fn = self._fused_cache.get(key)
    if fn is not None:
      return fn
    # Log each new geometry, since every one costs a full compile.
    print(f"  [vae fused] compiling for latent {tuple(latent_shape)} "
          f"x{clips} clips", flush=True)
    cuts, rows_n, cols_n, y_overlaps, x_overlaps = self._tile_geometry(
        latent_shape
    )
    shards = self._mesh.devices.size

    def call(params, latents):
      # `latents` is (clips, C, T, H, W) -- the group, stacked.
      tiles = [
          latents[c : c + 1, :, :, y0:y1, x0:x1]
          for c in range(clips)
          for (y0, y1, x0, x1) in cuts
      ]
      groups: dict[tuple, list[int]] = {}
      for index, tile in enumerate(tiles):
        groups.setdefault(tuple(tile.shape), []).append(index)
      decoded: list = [None] * len(tiles)
      for shape, members in groups.items():
        pad = (-len(members)) % shards
        batch = jnp.concatenate(
            [tiles[i] for i in members] + [tiles[members[-1]]] * pad
        )
        out = self._program(shape)(params, batch)
        for slot, index in enumerate(members):
          decoded[index] = out[slot : slot + 1]
      stitched = []
      for c in range(clips):
        here = decoded[c * len(cuts) : (c + 1) * len(cuts)]
        rows = [
            here[r * cols_n : (r + 1) * cols_n] for r in range(rows_n)
        ]
        stitched.append(
            self._stitch_jax(
                rows, y_overlaps=y_overlaps, x_overlaps=x_overlaps
            )
        )
      return jnp.concatenate(stitched)

    fn = jax.jit(call)
    self._fused_cache[key] = fn
    return fn

  def _tiles_of(self, z):
    """One clip's spatial tiles, in the order `_stitch_tiles` expects.

    `z` may be a torch tensor or a jax array. Given the latter the
    slicing runs on the chip, avoiding one host upload per tile.
    """
    vae = self._vae
    ratio = vae.spatial_compression_ratio
    height = z.shape[-2] * ratio
    width = z.shape[-1] * ratio
    y_pos, y_len, _ = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_pos, x_len, _ = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    return [
        z[
            ...,
            i_pos // ratio : i_pos // ratio + i_len // ratio,
            j_pos // ratio : j_pos // ratio + j_len // ratio,
        ]
        for i_pos, i_len in zip(y_pos, y_len)
        for j_pos, j_len in zip(x_pos, x_len)
    ]

  def install_lookahead(self, clips_per_batch: int = 2) -> None:
    """Decode `clips_per_batch` temporal clips in one batch.

    One clip is 28 tiles at 768p and 12 at 480p, and neither divides the
    eight shards -- the batch is padded up by repeating a tile, so 14%
    and 33% of the decodes respectively are thrown away. Two clips give
    56 = 8 x 7 and 24 = 8 x 3, both exact.

    The caller hands over one clip at a time, so batching across clips
    needs look-ahead: this recomputes `vae.decode`'s temporal padding,
    keeps the padded latents, and slices the next clips out of it. If a
    computed slice ever disagrees with the one actually requested, the
    look-ahead disables itself and falls back to per-clip decoding.
    """
    vae = self._vae
    self._clips_per_batch = clips_per_batch
    self._ready: dict[int, torch.Tensor] = {}
    self._index = 0
    self._padded = None
    self._lookahead = True
    inner = vae.decode

    def decode(z, *a, **kw):
      tokens = z.shape[2] + vae.config.token_drop
      pad = (-tokens) % vae.tokens_chunk_size
      self._padded = (
          torch.cat([z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
          if pad else z
      )
      self._ready.clear()
      self._index = 0
      return inner(z, *a, **kw)

    vae.decode = decode

  def _slice_for(self, index: int):
    vae = self._vae
    start = index * vae.tokens_chunk_size
    stop = start + vae.tokens_chunk_size + vae.token_overlap
    if self._padded is None or start >= self._padded.shape[2]:
      return None
    return self._padded[:, :, start:stop]

  def _fill_from(self, index: int, asked: torch.Tensor) -> bool:
    """Decodes a group of clips starting at `index`; False if we cannot."""
    mine = self._slice_for(index)
    if mine is None or mine.shape != asked.shape or not torch.equal(
        mine.to(asked.dtype), asked
    ):
      self._lookahead = False  # chunking is not what we modelled
      return False
    if self.fused_clip:
      return self._fill_fused(index, asked)
    group, flat, counts = [], [], []
    for offset in range(self._clips_per_batch):
      piece = self._slice_for(index + offset)
      if piece is None:
        break
      # The VAE upcasts latents between `decode` and `_decode_clip`, so cast
      # the kept slice to the requested dtype to match the weights.
      if self.upload_whole_clip:
        # One upload for the clip, then slice on the chip.
        tiles = self._tiles_of(
            _to_jax_dtype(piece, _JAX_DTYPE[asked.dtype])
        )
      else:
        tiles = self._tiles_of(piece.to(asked.dtype))
      group.append(index + offset)
      counts.append(len(tiles))
      flat.extend(tiles)
    decoded = self._run(flat)
    cursor = 0
    for clip, count in zip(group, counts):
      self._ready[clip] = decoded[cursor : cursor + count]
      cursor += count
    return True


  def _fill_fused(self, index: int, asked: torch.Tensor) -> bool:
    """One program for the whole clip group: cut, decode, stitch."""
    pieces, group = [], []
    for offset in range(self._clips_per_batch):
      piece = self._slice_for(index + offset)
      if piece is None:
        break
      pieces.append(_to_jax_dtype(piece, _JAX_DTYPE[asked.dtype]))
      group.append(index + offset)
    if not pieces or any(p.shape != pieces[0].shape for p in pieces):
      # A short tail clip has its own geometry; fall back for it.
      self.fused_clip = False
      return False
    begin = time.perf_counter()
    fn = self._fused_program(pieces[0].shape[1:], len(pieces))
    with jax.set_mesh(self._mesh):
      stitched = fn(self._params, jnp.concatenate(pieces))
    # One transfer for the group, not one per clip.
    brought = np.asarray(jax.device_get(stitched))
    self.stitch += time.perf_counter() - begin
    for slot, clip in enumerate(group):
      self._ready[clip] = torch.from_numpy(
          brought[slot : slot + 1].astype("float32")
      )
    return True

  def __call__(self, z: torch.Tensor) -> torch.Tensor:
    vae = self._vae
    if not vae.use_tiling:
      return self._run([z])[0]

    height = z.shape[-2] * vae.spatial_compression_ratio
    width = z.shape[-1] * vae.spatial_compression_ratio
    y_pos, y_len, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_pos, x_len, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    ratio = vae.spatial_compression_ratio

    # With the look-ahead on, this clip's tiles were decoded alongside
    # the next one's; otherwise decode just this clip.
    decoded = None
    if getattr(self, "_lookahead", False):
      if self._index not in self._ready:
        self._fill_from(self._index, z)
      decoded = self._ready.pop(self._index, None)
      self._index += 1
      if isinstance(decoded, torch.Tensor):
        return decoded  # the fused program already stitched it
    if decoded is None:
      decoded = self._run(self._tiles_of(z))
    rows, index = [], 0
    for _ in y_pos:
      row = []
      for _ in x_pos:
        row.append(decoded[index])
        index += 1
      rows.append(row)
    # Times the stitch and spatial blend. With `stitch_on_device` this is
    # dominated by the `device_get` of the stitched clip; `to_uint8`
    # shrinks that transfer 4x.
    begin = time.perf_counter()
    if self.stitch_on_device:
      # Compiled rather than eager: op-by-op the blends are hundreds of
      # separate dispatches. One program per tile geometry, cached.
      key = (
          tuple(tuple(t.shape for t in row) for row in rows),
          tuple(y_overlaps),
          tuple(x_overlaps),
      )
      fn = self._stitch_cache.get(key)
      if fn is None:
        fn = jax.jit(
            functools.partial(
                self._stitch_jax,
                y_overlaps=tuple(y_overlaps),
                x_overlaps=tuple(x_overlaps),
            )
        )
        self._stitch_cache[key] = fn
      with jax.set_mesh(self._mesh):
        stitched = fn(rows)
        if self.to_uint8:
          # Denormalise and cast on device, as the pipeline would on the
          # host, so uint8 rather than float32 crosses back.
          mean = jnp.asarray(self._pixel_mean, jnp.float32).reshape(1, -1, 1, 1, 1)
          std = jnp.asarray(self._pixel_std, jnp.float32).reshape(1, -1, 1, 1, 1)
          scaled = jnp.clip(stitched.astype(jnp.float32) * std + mean, 0.0, 1.0)
          stitched = jnp.round(scaled * 255.0).astype(jnp.uint8)
      brought = np.asarray(jax.device_get(stitched))
      out = torch.from_numpy(
          brought if brought.dtype == np.uint8 else brought.astype("float32")
      )
    else:
      out = vae._stitch_tiles(rows, y_overlaps, x_overlaps)
    self.stitch += time.perf_counter() - begin
    return out

  def _run(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
    """One program per distinct tile shape, batched over the mesh."""
    out: list[torch.Tensor | None] = [None] * len(tiles)
    groups: dict[tuple, list[int]] = {}
    for index, tile in enumerate(tiles):
      groups.setdefault(tuple(tile.shape), []).append(index)

    shards = self._mesh.devices.size
    max_batch = (
        int(os.environ.get("DT_VAE_1CHIP_TILE_BATCH", "4"))
        if shards == 1
        else len(tiles)
    )
    for shape, members in groups.items():
      step = max(shards, min(len(members), max_batch))
      step = -(-step // shards) * shards
      for start in range(0, len(members), step):
        sub = members[start : start + step]
        # The batch is padded up to a whole number of shards (or fixed step size)
        # by repeating the last tile; the copies are dropped on the way out.
        pad = (step - len(sub)) if len(members) > step else ((-len(sub)) % shards)
        # Each tile is already `(1, C, T, H, W)`; concatenating on that
        # axis makes the batch the decoder reads.
        stack = jnp.concatenate(
            [_to_jax(tiles[i]) for i in sub]
            + [_to_jax(tiles[sub[-1]])] * pad
        )
        self.batches[shape] = self.batches.get(shape, 0) + 1
        # No `block_until_ready` here: it would serialise transfer and
        # compute that the runtime otherwise overlaps (~3x slower).
        with jax.set_mesh(self._mesh):
          result = self._program(shape)(self._params, stack)
        if self.stitch_on_device:
          # Keep tiles on device; only the (smaller, de-overlapped)
          # stitched clip crosses back.
          for slot, index in enumerate(sub):
            out[index] = result[slot : slot + 1]
          continue
        result = jax.device_get(result)
        for slot, index in enumerate(sub):
          out[index] = torch.from_numpy(
              result[slot : slot + 1].astype("float32")
          )
    return out  # type: ignore[return-value]

  def _stitch_jax(self, rows, *, y_overlaps, x_overlaps):
    """`_stitch_tiles` in jax, on the decoded tiles where they already are.

    Same arithmetic as the model's own: blend each tile with its upper
    and left neighbour over their overlap, trim the overlap off every
    tile that has a successor, then concatenate.
    """
    def blend(a, b, extent, axis):
      extent = min(a.shape[axis], b.shape[axis], extent)
      if extent <= 0:
        return b
      weight = jnp.arange(extent, dtype=jnp.float32) / extent
      shape = [1] * a.ndim
      shape[axis] = extent
      weight = weight.reshape(shape).astype(b.dtype)
      head = jax.lax.slice_in_dim(a, a.shape[axis] - extent, a.shape[axis], axis=axis)
      tail = jax.lax.slice_in_dim(b, 0, extent, axis=axis)
      mixed = head * (1.0 - weight) + tail * weight
      rest = jax.lax.slice_in_dim(b, extent, b.shape[axis], axis=axis)
      return jnp.concatenate([mixed, rest], axis=axis)

    out_rows = []
    for i, row in enumerate(rows):
      pieces = []
      for j, tile in enumerate(row):
        if i > 0:
          tile = blend(rows[i - 1][j], tile, y_overlaps[i - 1], -2)
        if j > 0:
          tile = blend(row[j - 1], tile, x_overlaps[j - 1], -1)
        if i < len(rows) - 1:
          tile = jax.lax.slice_in_dim(
              tile, 0, tile.shape[-2] - y_overlaps[i], axis=-2)
        if j < len(row) - 1:
          tile = jax.lax.slice_in_dim(
              tile, 0, tile.shape[-1] - x_overlaps[j], axis=-1)
        pieces.append(tile)
      out_rows.append(jnp.concatenate(pieces, axis=-1))
    return jnp.concatenate(out_rows, axis=-2)


class CompiledAudioDecoder(torch.nn.Module):
  """The audio VAE's decode as one compiled program.

  `decode` is a single call -- `decoder(dec_in_proj(latents))` on a
  `[2, 32, frames]` input, the two stereo channels as the batch -- so
  there is no tiling to unpick. Installed the same way as the video one:
  `dec_in_proj` becomes an identity and this takes `decoder`'s place.
  """

  def __init__(self, dec_in_proj, decoder, mesh: jax.sharding.Mesh):
    super().__init__()
    import torchax

    self._inner = torch.nn.Sequential(dec_in_proj, decoder).eval()
    self._mesh = mesh
    self._env = torchax.default_env()
    self._cache: dict = {}
    self._params = {
        name: jax.device_put(_to_jax(tensor), NamedSharding(mesh, P()))
        for name, tensor in list(self._inner.named_parameters())
        + list(self._inner.named_buffers())
    }

  def forward(self, latents: torch.Tensor) -> torch.Tensor:
    key = tuple(latents.shape)
    fn = self._cache.get(key)
    if fn is None:
      def call(params, x):
        with self._env:
          out = torch.func.functional_call(
              self._inner,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              args=(self._env.j2t_iso(x),),
          )
        return self._env.t2j_iso(out)

      fn = jax.jit(call)
      self._cache[key] = fn
    with jax.set_mesh(self._mesh):
      out = fn(self._params, _to_jax(latents))
    return torch.from_numpy(jax.device_get(out).astype("float32")).to(
        latents.dtype
    )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-frames", type=int, default=345)
  ap.add_argument("--height", type=int, default=768)
  ap.add_argument("--width", type=int, default=1344)
  args = ap.parse_args()

  from diffusers import AutoencoderKLMiniMaxH3

  from vae_decode import CompiledTileDecoder

  mesh = jax.make_mesh(
      (len(jax.devices()),), ("tp",), axis_types=(jax.sharding.AxisType.Auto,)
  )
  print(f"{jax.devices()[0].device_kind} x {len(jax.devices())}")

  torch.manual_seed(0)
  latents = None

  def fresh():
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.bfloat16
    ).eval()
    torch.manual_seed(0)
    z = torch.randn(
        1,
        vae.config.latent_channels,
        args.num_frames // VAE_TEMPORAL,
        args.height // VAE_SPATIAL,
        args.width // VAE_SPATIAL,
        dtype=torch.bfloat16,
    )
    return vae, z

  results = {}
  vae, latents = fresh()
  tile = CompiledTileDecoder(vae.post_quant_conv, vae.decoder)
  vae.post_quant_conv = torch.nn.Identity()
  vae.decoder = tile
  with torch.no_grad():
    start = time.perf_counter()
    out = vae.decode(latents, return_dict=False)[0]
  results["one chip, tile at a time"] = (time.perf_counter() - start, out)

  vae, latents = fresh()
  batched = BatchedTileDecoder(vae, mesh)
  vae._decode_clip = batched
  with torch.no_grad():
    start = time.perf_counter()
    out = vae.decode(latents, return_dict=False)[0]
  results["mesh, batched"] = (time.perf_counter() - start, out)

  for label, (seconds, tensor) in results.items():
    print(f"  {label:<26} {seconds:7.1f}s  out {tuple(tensor.shape)}")
  (a, b) = (v[1].float().numpy() for v in results.values())
  base = next(iter(results.values()))[0]
  print(f"\n  speedup {base/list(results.values())[1][0]:.1f}x")
  mse = float(((a - b) ** 2).mean())
  rng = float(np.abs(a).max())
  print(
      "  agreement: "
      + ("bit identical" if mse == 0 else f"PSNR {10*np.log10(rng**2/mse):.1f} dB")
  )


if __name__ == "__main__":
  main()


class BatchedTileEncoder:
  """`_encode_clip` on the mesh, the mirror of `BatchedTileDecoder`.

  `ref2va` conditions on reference images; without this, encoding them on
  the host dominates the request.

  The structure is the decoder's reflected: the same spatial tiling,
  `quant_conv(encoder(tile))` in place of `decoder(post_quant_conv(tile))`,
  and the model's own `_stitch_tiles` over the latent-space overlaps.
  """

  def __init__(self, vae, mesh: jax.sharding.Mesh, on_demand: bool = False):
    """`on_demand`: keep the weights on the host and place them on the mesh
    only for an encode, for when the chips are needed for something else
    between encodes (the JAX video decoder beside the reference DiT)."""
    import torchax

    self._vae = vae
    self._mesh = mesh
    self._env = torchax.default_env()
    self._inner = torch.nn.Sequential(vae.encoder, vae.quant_conv).eval()
    self._cache: dict = {}
    tensors = (list(self._inner.named_parameters())
               + list(self._inner.named_buffers()))
    if on_demand:
      self._host = {name: np.asarray(jax.device_get(_to_jax(tensor)))
                    for name, tensor in tensors}
      self._params = None
    else:
      self._host = None
      self._params = {
          name: jax.device_put(_to_jax(tensor), NamedSharding(mesh, P()))
          for name, tensor in tensors
      }
    self.calls = 0

  def _program(self, shape: tuple[int, ...]):
    fn = self._cache.get(shape)
    if fn is None:
      def call(params, tiles):
        with self._env:
          out = torch.func.functional_call(
              self._inner,
              {k: self._env.j2t_iso(v) for k, v in params.items()},
              args=(self._env.j2t_iso(tiles),),
          )
        return self._env.t2j_iso(out)

      fn = jax.jit(
          call,
          in_shardings=(None, NamedSharding(self._mesh, P("tp"))),
          out_shardings=NamedSharding(self._mesh, P("tp")),
      )
      self._cache[shape] = fn
    return fn

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    if self._host is None:
      return self._encode(x)
    self._params = jax.device_put(self._host, NamedSharding(self._mesh, P()))
    try:
      return self._encode(x)
    finally:
      self._params = None

  def _encode(self, x: torch.Tensor) -> torch.Tensor:
    vae = self._vae
    self.calls += 1
    if not vae.use_tiling:
      return self._run([x])[0]

    height, width = x.shape[-2], x.shape[-1]
    y_pos, y_len, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_pos, x_len, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    tiles = [
        x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
        for i_pos, i_len in zip(y_pos, y_len)
        for j_pos, j_len in zip(x_pos, x_len)
    ]
    encoded = self._run(tiles)
    rows, index = [], 0
    for _ in y_pos:
      row = []
      for _ in x_pos:
        row.append(encoded[index])
        index += 1
      rows.append(row)
    ratio = vae.spatial_compression_ratio
    return vae._stitch_tiles(
        rows,
        [o // ratio for o in y_overlaps],
        [o // ratio for o in x_overlaps],
    )

  def _run(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
    """One program per tile shape, the batch cut over the mesh."""
    out: list[torch.Tensor | None] = [None] * len(tiles)
    groups: dict[tuple, list[int]] = {}
    for index, tile in enumerate(tiles):
      groups.setdefault(tuple(tile.shape), []).append(index)
    shards = self._mesh.devices.size
    max_batch = (
        int(os.environ.get("DT_VAE_1CHIP_TILE_BATCH", "4"))
        if shards == 1
        else len(tiles)
    )
    for shape, members in groups.items():
      step = max(shards, min(len(members), max_batch))
      step = -(-step // shards) * shards
      for start in range(0, len(members), step):
        sub = members[start : start + step]
        pad = (step - len(sub)) if len(members) > step else ((-len(sub)) % shards)
        stack = jnp.concatenate(
            [_to_jax(tiles[i]) for i in sub]
            + [_to_jax(tiles[sub[-1]])] * pad
        )
        with jax.set_mesh(self._mesh):
          result = self._program(tuple(stack.shape))(self._params, stack)
        result = jax.device_get(result)
        for slot, index in enumerate(sub):
          out[index] = torch.from_numpy(result[slot : slot + 1].astype("float32"))
    return out  # type: ignore[return-value]


def install_batched_encoder(vae, mesh: jax.sharding.Mesh,
                            on_demand: bool = False) -> BatchedTileEncoder:
  """Puts `_encode_clip` on the mesh. Returns it so callers can read `calls`."""
  encoder = BatchedTileEncoder(vae, mesh, on_demand=on_demand)
  vae._encode_clip = encoder
  return encoder
