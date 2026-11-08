"""The JAX backend's video decode, usable from other backends.

`jax_server` and `jax_pipeline` decode with the tensor-parallel JAX ViT
decoder, temporal chunks K at a time, tiles batched over the mesh, the
crossfade and the uint8 conversion on the chips. The torchax backend ran
diffusers' VAE instead (tiles stitched and clips blended on the host,
float32 postprocessing): 9.5-11 s against 5.0 s at 768p/345 frames. This
class is the server's decoder lifted out so torchax can call it on the
latents its denoise loop leaves.
"""

import functools
import glob
import json
import os

import numpy as np


class VideoDecoder:
  """Normalized video latents `(1, C, F, h, w)` -> uint8 frames `(F, H, W, 3)`."""

  def __init__(self, snapshot, mesh, overlap_px=32, chunk_cap=0,
               block_q=2048, block_kv=4096):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from ...kernels import mesh_attention as ma
    from ...kernels import splash as csa
    from ...safetensors_io import safe_open
    from . import jax_vae as jvae

    self.mesh, self.overlap_px, self.chunk_cap = mesh, overlap_px, chunk_cap
    self.jvae = jvae
    dtype = jnp.bfloat16
    with open(os.path.join(snapshot, "vae", "config.json")) as f:
      cfg = json.load(f)
    self.lat_mean = jnp.asarray(cfg["latents_mean"], jnp.float32).reshape(1, -1, 1, 1, 1)
    self.lat_std = jnp.asarray(cfg["latents_std"], jnp.float32).reshape(1, -1, 1, 1, 1)

    local = functools.partial(csa.attention, scale=None, block_q=block_q,
                              block_kv=block_kv, block_kv_compute=block_kv,
                              use_fixed_m=True)

    def sdpa(q, k, v, kv_len=None):
      k = k - jnp.mean(k, axis=2, keepdims=True)
      with jax.default_matmul_precision("default"):
        return ma._on_mesh(local, mesh, q, k, v, True)

    decoder = nnx.eval_shape(
        lambda: jvae.ViTDecoder(**jvae.CONFIG, dtype=dtype, rngs=nnx.Rngs(0)))
    post_q = jvae.PostQuantConv(jvae.CONFIG["in_channels"], dtype=dtype,
                                rngs=nnx.Rngs(0))
    params = {}
    for shard in sorted(glob.glob(os.path.join(snapshot, "vae", "*.safetensors"))):
      with safe_open(shard, framework="flax") as reader:
        for k in reader.keys():
          if k.startswith("decoder."):
            params[k] = reader.get_tensor(k)
          elif k.startswith("post_quant_conv."):
            leaf = getattr(post_q, k.split(".")[-1])
            leaf.value = jnp.asarray(reader.get_tensor(k)).astype(
                leaf.value.dtype).reshape(leaf.value.shape)
    jvae.load_decoder(decoder, params, mesh=mesh)
    dec_graph, self.dec_state = nnx.split(decoder)
    pq_graph, pq_state = nnx.split(post_q)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    @functools.partial(jax.jit, out_shardings=replicated)
    def decode(state, z, pq=pq_state):
      with jax.default_matmul_precision("highest"):
        z = nnx.merge(pq_graph, pq)(z)
        return nnx.merge(dec_graph, state)(z, sdpa=sdpa)
    self._decode = decode

    @functools.partial(jax.jit, static_argnums=(2,))
    def blend(x, y, extent):
      extent = min(x.shape[2], y.shape[2], extent)
      if extent <= 0:
        return y
      w = jnp.arange(extent, dtype=y.dtype).reshape(1, 1, extent, 1, 1) / extent
      head = x[:, :, -extent:] * (1.0 - w) + y[:, :, :extent] * w
      return head if extent == y.shape[2] else jnp.concatenate([head, y[:, :, extent:]], axis=2)
    self._blend = blend

    from .jax_pipeline import uint8_frames
    self._to_uint8 = jax.jit(uint8_frames, out_shardings=replicated)

  def decode(self, latents):
    """`latents` as the denoise loop leaves them (normalized), any float dtype."""
    import jax
    # The decoder's tiling places arrays by PartitionSpec, which needs the
    # mesh in context; callers such as torchax do not set one.
    with jax.set_mesh(self.mesh):
      return self._decode_in_mesh(latents)

  def _decode_in_mesh(self, latents):
    import jax.numpy as jnp
    from .jax_pipeline import decode_chunk_batch

    clip_length, token_drop, ratio = 17, 3, 4
    tokens_chunk = -(-clip_length // ratio)
    pre_pad = (-clip_length) % ratio
    tok_overlap = (-token_drop) % tokens_chunk
    frame_overlap = max(tok_overlap * ratio - pre_pad, 0)
    chunk_frames = tokens_chunk * ratio
    jvae, dec_state, decode, n_dev = self.jvae, self.dec_state, self._decode, self.mesh.devices.size

    def decode_tiled(z):
      return jvae.tiled_decode(lambda t_: decode(dec_state, t_), z, tile_px=256,
                               overlap_px=self.overlap_px,
                               batch=28 if n_dev > 1 else 4)

    latents = latents.astype(jnp.float32) * self.lat_std + self.lat_mean
    frames_lat = latents.shape[2]
    num_tokens = frames_lat + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk - int(token_drop > 0)
    z = latents.astype(jnp.bfloat16)
    if pad_tokens:
      z = jnp.concatenate([z, jnp.repeat(z[:, :, -1:], pad_tokens, axis=2)], axis=2)
    base_cap = 8 if latents.shape[3] * latents.shape[4] <= 30 * 52 else 4
    K = decode_chunk_batch(frames_lat, 0, 1 if n_dev == 1 else
                           self.chunk_cap or max(1, (base_cap * n_dev) // 8))
    decoded, pieces, carry = {}, [], None
    for i in range(num_chunks):
      if i not in decoded:
        group = list(range(i, min(i + K, num_chunks)))
        zs = [z[:, :, g * tokens_chunk:g * tokens_chunk + tokens_chunk + tok_overlap]
              for g in group]
        zs += [zs[-1]] * (K - len(zs))
        out_g = decode_tiled(jnp.concatenate(zs, axis=0))
        for k, g in enumerate(group):
          decoded[g] = out_g[k:k + 1]
      out = decoded.pop(i)
      for j in range(int(token_drop > 0) + 1):
        f = j * chunk_frames
        piece = out[:, :, f:f + chunk_frames][:, :, pre_pad:]
        if j == 0:
          pieces.append(piece if carry is None else self._blend(carry, piece, frame_overlap))
        else:
          carry = piece
    if carry is not None:
      pieces.append(carry)
    frames = jnp.concatenate(pieces, axis=2)
    if pad_tokens:
      intra_tail = clip_length % ratio
      before = z.shape[2] - pad_tokens
      pad_frames = sum(intra_tail if intra_tail and (before + k) % tokens_chunk == 0
                       else ratio for k in range(pad_tokens))
      frames = frames[:, :, :-pad_frames]
    shape = (frames.shape[2], frames.shape[3], frames.shape[4], frames.shape[1])
    return np.asarray(self._to_uint8(frames)).reshape(shape)
