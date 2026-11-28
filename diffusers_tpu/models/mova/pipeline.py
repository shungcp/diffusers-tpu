"""MOVA image + text -> video + audio (TI2VA) with native JAX models.

    first frame -> Wan VAE encode (frame 0, then zeros) -> condition (4 mask + 16 latent)
    prompt / negative -> UMT5-XXL (512 tokens, padded rows zeroed)
    50 steps: [positive, negative] through video expert (high noise, then low
              noise below t = 900) + audio tower + bridge, CFG, Euler
    -> Wan VAE decode (video) and DAC decode (audio, 48 kHz)

The schedule is the official ``FlowMatchPairScheduler``'s: timesteps from a
shifted linear grid, and each Euler step's sigma is the entry of the
1000-step *training* table nearest to the timestep (``timestep_to_sigma``).

    PYTHONPATH=. python -m diffusers_tpu.models.mova.pipeline --image ref.png --prompt "..."
"""
import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from diffusers_tpu.models.dac import jax_dac
from diffusers_tpu.models.mova import jax_dit
from diffusers_tpu.models.wan_vae import jax_wan_vae

REPO = "OpenMOSS-Team/MOVA-360p"
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指")


def schedule(steps, shift=5.0, num_train=1000, boundary_ratio=0.9):
  """(timesteps [steps], sigma_from [steps], sigma_to [steps], expert [steps]).

  ``expert`` is 0 for ``video_dit`` and 1 for ``video_dit_2`` (from the first
  timestep below ``boundary_ratio * num_train``)."""
  def table(n):
    s = np.linspace(1.0, 0.0, n + 1, dtype=np.float32)[:-1]  # extra_one_step, sigma_min 0
    s = (shift * s / (1 + (shift - 1) * s)).astype(np.float32)
    return s * np.float32(num_train), s

  train_t, train_s = table(num_train)
  t, _ = table(steps)
  snap = lambda x: train_s[int(np.argmin(np.abs(train_t - np.float32(x))))]
  s_from = np.array([snap(x) for x in t], np.float32)
  s_to = np.array([snap(x) for x in t[1:]] + [0.0], np.float32)
  expert = (t < boundary_ratio * num_train).astype(np.int32)
  expert = np.maximum.accumulate(expert)
  return t, s_from, s_to, expert


def first_frame_condition(encode, vae_params, vae_config, image, frames):
  """image [H, W, 3] uint8 -> condition [1, 20, T, H/8, W/8] (mask + normalized latent)."""
  x = (image.astype(np.float32) / 255.0 * 2 - 1).transpose(2, 0, 1)[None, :, None, :, :]  # [1, 3, 1, H, W]
  hh, ww = image.shape[:2]
  first_j = jnp.asarray(x, jnp.bfloat16)
  if frames > 1:
    video = jnp.concatenate([first_j, jnp.zeros((1, 3, frames - 1, hh, ww), jnp.bfloat16)], axis=2)
  else:
    video = first_j
  latent = jax_wan_vae.normalize(vae_config, encode(vae_params, video))
  t = latent.shape[2]
  mask = np.zeros((1, 4, t, hh // 8, ww // 8), np.float32)
  mask[:, :, 0] = 1.0  # latent frame 0 holds the given image (4 pixel frames' worth of ones)
  return jnp.concatenate([jnp.asarray(mask), latent], 1)


def clean_prompt(text):
  import html
  import re

  from diffusers_tpu.models.moss_soundeffect.text import fix_text

  text = html.unescape(html.unescape(fix_text(text))).strip()
  return re.sub(r"\s+", " ", text).strip()


class TextEncoder:
  """UMT5-XXL (transformers' ``UMT5EncoderModel``): [1, 512, 4096], padded rows zero."""

  def __init__(self, snapshot, dtype=None):
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snapshot, "tokenizer"))
    self.model = UMT5EncoderModel.from_pretrained(os.path.join(snapshot, "text_encoder"),
                                                  dtype=dtype or torch.bfloat16).eval()
    self.cache = {}

  def __call__(self, prompt):
    import torch

    if prompt not in self.cache:
      enc = self.tokenizer([clean_prompt(prompt)], padding="max_length", max_length=512,
                           truncation=True, add_special_tokens=True, return_tensors="pt")
      with torch.no_grad():
        h = self.model(enc.input_ids, enc.attention_mask).last_hidden_state.float()
      n = int(enc.attention_mask.sum())
      h[:, n:] = 0
      self.cache[prompt] = h.numpy()
    return self.cache[prompt]


def make_step(vconf, aconf, bconf, video_fps, attention=jax_dit.dense_attention, video_layer=None,
              blocks_fn=None, ropes=None, return_vel=False, return_delta=False):
  """jitted ``(video, audio, bridge, latents_v, latents_a, condition, context, t,
  s_from, s_to, guidance) -> (latents_v, latents_a[, v_pred, a_pred[, dv, da]])``."""

  def step(video, audio, bridge, lat_v, lat_a, condition, context, t, s_from, s_to, guidance):
    batch = context.shape[0]
    x = jnp.concatenate([lat_v, condition], 1)
    x = jnp.broadcast_to(x, (batch,) + x.shape[1:])
    a = jnp.broadcast_to(lat_a, (batch,) + lat_a.shape[1:])
    v_pred, a_pred = jax_dit.dual_step(video, vconf, audio, aconf, bridge, bconf, x, a, context,
                                       t, t, video_fps, attention=attention,
                                       video_layer=video_layer, blocks_fn=blocks_fn, ropes=ropes)
    v_pred, a_pred = v_pred.astype(jnp.float32), a_pred.astype(jnp.float32)
    dv = da = None
    if batch > 1:
      dv = v_pred[0:1] - v_pred[1:2]
      da = a_pred[0:1] - a_pred[1:2]
      v_pred = v_pred[0:1] + (guidance - 1.0) * dv
      a_pred = a_pred[0:1] + (guidance - 1.0) * da
    dt = s_to - s_from
    out_v, out_a = lat_v + dt * v_pred, lat_a + dt * a_pred
    if return_delta:
      return out_v, out_a, v_pred, a_pred, dv, da
    if return_vel:
      return out_v, out_a, v_pred, a_pred
    return out_v, out_a

  return jax.jit(step)


def make_cond_step(vconf, aconf, bconf, video_fps, attention=jax_dit.dense_attention,
                   blocks_fn=None, ropes=None):
  """jitted ``cond-only`` (``batch=1`` on ``1x8`` tensor-parallel mesh) step with cached CFG delta."""

  def step(video, audio, bridge, lat_v, lat_a, condition, pos_context, t, s_from, s_to,
           guidance, delta_scale, cached_dv, cached_da):
    x = jnp.concatenate([lat_v, condition], 1)
    v_pos, a_pos = jax_dit.dual_step(video, vconf, audio, aconf, bridge, bconf, x, lat_a, pos_context,
                                     t, t, video_fps, attention=attention,
                                     blocks_fn=blocks_fn, ropes=ropes)
    v_pos, a_pos = v_pos.astype(jnp.float32), a_pos.astype(jnp.float32)
    scale = (guidance - 1.0) * delta_scale
    v_pred = v_pos + scale * cached_dv
    a_pred = a_pos + scale * cached_da
    dt = s_to - s_from
    return lat_v + dt * v_pred, lat_a + dt * a_pred, v_pred, a_pred

  return jax.jit(step)


def audio_frames(num_frames, video_fps, sample_rate=48000, hop=960):
  samples = int(sample_rate * num_frames / video_fps)
  return (samples - 1) // hop + 1, samples


def crop_and_resize(image, height, width):
  """The official reference-image preparation: center crop to the aspect, resize."""
  from PIL import Image

  image = np.array(image.convert("RGB"))
  ih, iw, _ = image.shape
  if ih / iw < height / width:
    cw = int(ih / height * width)
    left = (iw - cw) // 2
    image = image[:, left:left + cw]
  else:
    ch = int(iw / width * height)
    top = (ih - ch) // 2
    image = image[top:top + ch, :]
  return np.array(Image.fromarray(image).resize((width, height)))


# ---------------------------------------------------------------------------
# TPU generator
# ---------------------------------------------------------------------------

_ALL = ("cfg", "tensor")
TEXT_SHARDINGS = {
    r"block\.\d+\.layer\.0\.SelfAttention\.(q|k|v)\.weight": (_ALL, None),
    r"block\.\d+\.layer\.0\.SelfAttention\.o\.weight": (None, _ALL),
    r"block\.\d+\.layer\.1\.DenseReluDense\.(wi_0|wi_1)\.weight": (_ALL, None),
    r"block\.\d+\.layer\.1\.DenseReluDense\.wo\.weight": (None, _ALL),
}


def _text_core_class():
  import torch

  class TextCore(torch.nn.Module):
    """UMT5's encoder blocks and final norm, an additive key mask [1, 1, 1, L]."""

    def __init__(self, encoder):
      super().__init__()
      self.block = encoder.block
      self.final_layer_norm = encoder.final_layer_norm

    def forward(self, embeds, mask):
      h = embeds
      for layer in self.block:
        h = layer(h, mask)
      return self.final_layer_norm(h)

  return TextCore


def _is_high_hbm():
  try:
    return jax.devices()[0].memory_stats().get("bytes_limit", 0) >= 64 * 2**30
  except Exception:
    return False


class Generator:
  """Every model resident on ``mesh`` (``('cfg', 'tensor')``): both video experts,
  the audio tower and bridge (whole-resident on v7x ``>=64 GiB`` or sharded over
  ``tensor`` on v6e), UMT5 through torchax, and Wan2.1 VAE + DAC."""

  def __init__(self, repo=REPO, mesh=None, env=None, height=352, width=640, frames=193,
               video_fps=24.0, vae_chunk=None, seq_parallel=True, resident=None,
               fp8=False, qk_fp8=False, block_kv=None, sparse_window=0,
               sparse_dense_steps=8, step_cache_thresh=0.0, step_cache_max_skip=2,
               cfg_skip_after=0.0, cfg_skip_refresh=0, dit_backend="jax",
               load_dit=True, load_text=True):
    import contextlib
    import json

    import torch
    import torchax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from transformers import AutoTokenizer, UMT5EncoderModel

    from diffusers_tpu.hub import local_snapshot
    from diffusers_tpu.models.moss_image.torchax_pipeline import _move_module, _shard

    if height % 16 or width % 16 or (frames - 1) % 4:
      raise ValueError("height and width must be multiples of 16, frames 1 + 4k")
    self.mesh, self.env = mesh, env or torchax.default_env()
    self.height, self.width, self.frames, self.video_fps = height, width, frames, video_fps
    high_hbm = _is_high_hbm()
    self.resident = high_hbm if resident is None else bool(resident)
    self.fp8 = bool(fp8)
    self.qk_fp8 = bool(qk_fp8)
    self.block_kv = block_kv
    self.sparse_window = max(0, int(sparse_window))
    self.sparse_dense_steps = max(0, int(sparse_dense_steps))
    self.step_cache_thresh = max(0.0, float(step_cache_thresh))
    self.step_cache_max_skip = max(0, int(step_cache_max_skip))
    self.cfg_skip_after = float(cfg_skip_after)
    self.cfg_skip_refresh = max(0, int(cfg_skip_refresh))
    self.dit_backend = str(dit_backend)
    if vae_chunk is None:
      vae_chunk = 8 if high_hbm else 4
    _repo_path = os.path.expanduser(repo)
    snap = _repo_path if os.path.isdir(_repo_path) else local_snapshot(repo)
    self.snap = snap
    self.times = {}

    # Text: embedding lookup on the host, the blocks through torchax.
    if load_text:
      self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
      with contextlib.suppress(RuntimeError):
        torchax.disable_globally()
      t5 = UMT5EncoderModel.from_pretrained(os.path.join(snap, "text_encoder"), dtype=torch.bfloat16,
                                            attn_implementation="eager").eval()
      self._embed = t5.encoder.embed_tokens
      for blk in t5.encoder.block:
        blk.layer[1].DenseReluDense.wo.float()
      torchax.enable_globally()
      self.env._mesh = mesh
      self.env.default_device_or_sharding = NamedSharding(mesh, P())
      core = _text_core_class()(t5.encoder)
      _move_module(core, self.env)
      core = torchax.compile(core)
      core.params = _shard(core.params, TEXT_SHARDINGS, mesh)
      core.buffers = _shard(core.buffers, TEXT_SHARDINGS, mesh)
      self._text = core
      self._prompts = {}
      torchax.disable_globally()
    else:
      self.tokenizer = None
      self._embed = None
      self._text = None
      self._prompts = {}

    self.audio_len, self.audio_samples = audio_frames(frames, video_fps)
    self._cond_step = None
    self._vel_rel = None
    self.sparse_density = 1.0

    # DiT: both experts + audio + bridge (whole-resident on v7x, optional W8A8 FP8).
    if load_dit:
      self.audio, self.aconf = jax_dit.load_tower(os.path.join(snap, "audio_dit"))
      self.bridge, self.bconf = jax_dit.load_bridge(os.path.join(snap, "dual_tower_bridge"))
      self.audio = jax_dit.place(self.audio, mesh, resident=self.resident, fp8=self.fp8)
      self.bridge = jax_dit.place(self.bridge, mesh, resident=self.resident, fp8=self.fp8)
      self.experts = []
      for d in ("video_dit", "video_dit_2"):
        v, self.vconf = jax_dit.load_tower(os.path.join(snap, d))
        self.experts.append(jax_dit.place(v, mesh, resident=self.resident, fp8=self.fp8))
        del v

      patch = tuple(self.vconf["patch_size"])
      grid = (((frames - 1) // 4 + 1) // patch[0], (height // 8) // patch[1], (width // 8) // patch[2])
      tokens = grid[0] * grid[1] * grid[2]
      parts = mesh.shape["tensor"]

      # Precompute sharded RoPE tables once on the mesh (avoids baking 271 MB into HLO).
      vhd = self.vconf["dim"] // self.vconf["num_heads"]
      ahd = self.aconf["dim"] // self.aconf["num_heads"]
      vcos_np, vsin_np = jax_dit.video_rope(*grid, vhd)
      acos_np, asin_np = jax_dit.audio_rope(self.audio_len, ahd)
      (bvc_np, bvs_np), (bac_np, bas_np) = jax_dit.bridge_rope(
          grid, self.audio_len, video_fps, self.bconf["audio_fps"], self.bconf["head_dim"]
      )
      use_sp = seq_parallel and tokens % parts == 0 and self.vconf["num_heads"] % parts == 0
      v_rope_spec = P("tensor", None) if use_sp else P()
      self._ropes = (
          self._put(vcos_np, v_rope_spec),
          self._put(vsin_np, v_rope_spec),
          self._put(acos_np),
          self._put(asin_np),
          (self._put(bvc_np, v_rope_spec), self._put(bvs_np, v_rope_spec)),
          (self._put(bac_np), self._put(bas_np)),
      )

      if self.dit_backend == "torchax":
        from . import torchax_dit
        blocks_builder = torchax_dit.make_torchax_dual_blocks_fn
      elif self.dit_backend == "jax":
        blocks_builder = jax_dit.make_dual_blocks_fn
      else:
        raise ValueError(f"Unsupported dit_backend={self.dit_backend!r} (expected 'jax' or 'torchax')")

      use_cfg_skip = (
          0.0 < self.cfg_skip_after < 1.0
          and use_sp
          and self.resident
          and mesh.shape["cfg"] > 1
          and tokens % mesh.size == 0
          and self.vconf["num_heads"] % mesh.size == 0
      )
      need_vel = self.step_cache_thresh > 0.0
      attn = jax_dit.make_attention(mesh, qk_fp8=self.qk_fp8, block_kv=self.block_kv)
      sp_table = None
      if use_sp:
        blocks_dense = blocks_builder(
            mesh, self.vconf, self.aconf, self.bconf,
            self.experts[0], self.audio, self.bridge,
            resident=self.resident, qk_fp8=self.qk_fp8, block_kv=self.block_kv,
        )
        self._step = make_step(self.vconf, self.aconf, self.bconf, video_fps,
                               attention=attn, blocks_fn=blocks_dense, ropes=self._ropes,
                               return_vel=need_vel, return_delta=use_cfg_skip)
        if self.sparse_window > 0:
          bq_sp, bkv_sp, _ = jax_dit.resolve_mova_sparse_blocks(
              tokens, self.vconf["dim"] // self.vconf["num_heads"], block_kv=self.block_kv
          )
          sp_table = jax_dit.build_mova_sparse_kv_table(
              grid, self.sparse_window, block_q=bq_sp, block_kv=bkv_sp
          )
          kept = sum(len(r) for r in sp_table)
          tot = len(sp_table) * (-(-tokens // min(bkv_sp, tokens)))
          self.sparse_density = kept / max(1, tot)
          blocks_sp = blocks_builder(
              mesh, self.vconf, self.aconf, self.bconf,
              self.experts[0], self.audio, self.bridge,
              resident=self.resident, qk_fp8=self.qk_fp8, block_kv=self.block_kv,
              sparse_kv_table=sp_table,
          )
          self._sparse_step = make_step(self.vconf, self.aconf, self.bconf, video_fps,
                                        attention=attn, blocks_fn=blocks_sp, ropes=self._ropes,
                                        return_vel=need_vel, return_delta=use_cfg_skip)
        else:
          self._sparse_step = None

        if use_cfg_skip:
          # Second 1x8 mesh over the exact same devices: shares resident=True replicated weights (0 extra HBM!),
          # halves per-chip video tokens & heads (10 -> 5 heads/chip) for ~2x faster late cond-only steps.
          mesh_1x8 = Mesh(mesh.devices.reshape(1, mesh.size), ("cfg", "tensor"))
          put_1x8 = lambda a, sp=None: jax.device_put(a, NamedSharding(mesh_1x8, sp or P()))
          ropes_1x8 = (
              put_1x8(vcos_np, P("tensor", None)),
              put_1x8(vsin_np, P("tensor", None)),
              put_1x8(acos_np),
              put_1x8(asin_np),
              (put_1x8(bvc_np, P("tensor", None)), put_1x8(bvs_np, P("tensor", None))),
              (put_1x8(bac_np), put_1x8(bas_np)),
          )
          blocks_cond = blocks_builder(
              mesh_1x8, self.vconf, self.aconf, self.bconf,
              self.experts[0], self.audio, self.bridge,
              resident=True, qk_fp8=self.qk_fp8, block_kv=self.block_kv,
              sparse_kv_table=sp_table,
          )
          self._cond_step = make_cond_step(
              self.vconf, self.aconf, self.bconf, video_fps,
              attention=attn, blocks_fn=blocks_cond, ropes=ropes_1x8,
          )
      else:
        self._step = make_step(self.vconf, self.aconf, self.bconf, video_fps,
                               attention=attn, ropes=self._ropes, return_vel=need_vel)
        self._sparse_step = None

      if need_vel:
        @jax.jit
        def _vel_rel(vc, vp, ac, ap):
          rv = jnp.sum(jnp.abs(vc - vp)) / jnp.maximum(jnp.sum(jnp.abs(vc)), jnp.float32(1e-6))
          ra = jnp.sum(jnp.abs(ac - ap)) / jnp.maximum(jnp.sum(jnp.abs(ac)), jnp.float32(1e-6))
          return rv, ra
        self._vel_rel = _vel_rel
    else:
      with open(os.path.join(snap, "video_dit", "config.json")) as f:
        self.vconf = json.load(f)
      self.experts = []
      self.audio = self.bridge = None
      self._step = self._sparse_step = None

    # VAEs: shard on height if (H/8) divides mesh.size, else on width if (W/8) divides mesh.size
    # (fixes 720p where 90 % 8 != 0 while 160 % 8 == 0!), with on-TPU uint8 NDHWC decode (4x smaller D2H).
    if (height // 8) % mesh.size == 0:
      vae_axes, vae_spatial = ("cfg", "tensor"), 2
    elif (width // 8) % mesh.size == 0:
      vae_axes, vae_spatial = ("cfg", "tensor"), 3
    else:
      vae_axes, vae_spatial = ("tensor",), 2
    self.vae, self.vae_config, self._encode, decode = jax_wan_vae.make(
        os.path.join(snap, "video_vae"), mesh, dtype=jnp.bfloat16, chunk=vae_chunk,
        rows=vae_axes, spatial_axis=vae_spatial, to_uint8=True)
    self._decode_video = lambda p, z: decode(p, jax_wan_vae.denormalize(self.vae_config, z))
    self.dac, self._decode_audio, _ = jax_dac.make_decoder(os.path.join(snap, "audio_vae"), mesh=mesh)

  def _put(self, array, spec=None):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return jax.device_put(array, NamedSharding(self.mesh, spec or P()))

  def encode_text(self, prompt):
    """[1, 512, 4096] bf16 on the mesh, padded rows zeroed; cached per prompt."""
    import torch
    import torchax

    if prompt not in self._prompts:
      enc = self.tokenizer([clean_prompt(prompt)], padding="max_length", max_length=512,
                           truncation=True, add_special_tokens=True, return_tensors="np")
      ids, keep = enc["input_ids"], enc["attention_mask"]
      with torch.no_grad():
        embeds = self._embed(torch.from_numpy(ids).long())
      bias = np.where(keep[:, None, None, :] > 0, 0.0, np.finfo(np.float32).min)
      torchax.enable_globally()
      try:
        e = self.env.j2t_iso(jax.lax.bitcast_convert_type(
            self._put(embeds.view(torch.int16).numpy()), jnp.bfloat16))
        m = self.env.j2t_iso(self._put(bias.astype(jnp.bfloat16)))
        h = self.env.t2j_iso(self._text(e, m))
      finally:
        torchax.disable_globally()
      self._prompts[prompt] = jnp.where(self._put(keep[..., None] > 0), h, 0).astype(jnp.bfloat16)
    return self._prompts[prompt]

  def condition(self, image):
    """image [H, W, 3] uint8 -> [1, 20, T, h, w] on the mesh."""
    return first_frame_condition(lambda p, v: self._encode(p, self._put(v)), self.vae,
                                 self.vae_config, image, self.frames)

  def noise(self, seed):
    """The official order on the global torch CPU generator: video, then audio."""
    import torch

    torch.manual_seed(int(seed))
    t = (self.frames - 1) // 4 + 1
    lat_v = torch.randn((1, 16, t, self.height // 8, self.width // 8), dtype=torch.float32)
    lat_a = torch.randn((1, 128, self.audio_len), dtype=torch.float32)
    return self._put(lat_v.numpy()), self._put(lat_a.numpy())

  def denoise(self, condition, context, lat_v, lat_a, steps=50, cfg_scale=5.0, shift=5.0):
    t, s_from, s_to, expert = schedule(steps, shift)
    sinusoids = jax_dit.time_sinusoid(t)
    guidance = self._put(np.float32(cfg_scale))
    use_cache = self.step_cache_thresh > 0.0 and self._vel_rel is not None
    use_cfg_skip = self._cond_step is not None
    cfg_skip_start = int(round(steps * self.cfg_skip_after)) if use_cfg_skip else steps
    cache_start = max(6, int(round(steps * 0.14)))
    cache_end = steps - max(4, int(round(steps * 0.10)))
    prev_v = prev_a = None
    cached_dv = cached_da = None
    prev_expert = -1
    accum_err = 0.0
    last_rate = 1.0
    consec = 0
    n_cached = 0
    n_cond_only = 0
    cond_since_refresh = 0
    pos_context = context[0:1] if use_cfg_skip else None

    for i in range(steps):
      ex = int(expert[i])
      # Never reuse cached velocity or CFG delta across the high-noise -> low-noise expert boundary!
      same_expert = (ex == prev_expert)
      if not same_expert:
        cached_dv = cached_da = None
        cond_since_refresh = 0
      can_cache = (
          use_cache
          and prev_v is not None
          and same_expert
          and cache_start <= i < cache_end
          and consec < self.step_cache_max_skip
          and (accum_err + last_rate) < self.step_cache_thresh
      )
      if can_cache:
        dt = self._put(np.float32(s_to[i] - s_from[i]))
        lat_v = lat_v + dt * prev_v
        lat_a = lat_a + dt * prev_a
        accum_err += last_rate
        consec += 1
        n_cached += 1
        continue

      want_cond_only = (
          use_cfg_skip
          and i >= cfg_skip_start
          and same_expert
          and cached_dv is not None
          and (self.cfg_skip_refresh == 0 or cond_since_refresh < self.cfg_skip_refresh)
      )
      if want_cond_only:
        # Pure conditional velocity in late low-noise tail avoids integrating stale CFG delta into audio/video latents.
        delta_scale = self._put(np.float32(0.0))
        lat_v, lat_a, v_cur, a_cur = self._cond_step(
            self.experts[ex], self.audio, self.bridge, lat_v, lat_a,
            condition, pos_context, self._put(sinusoids[i:i + 1]),
            self._put(s_from[i]), self._put(s_to[i]), guidance,
            delta_scale, cached_dv, cached_da,
        )
        n_cond_only += 1
        cond_since_refresh += 1
        if use_cache:
          if prev_v is not None and same_expert:
            rv_j, ra_j = self._vel_rel(v_cur, prev_v, a_cur, prev_a)
            last_rate = max(float(rv_j), 0.7 * float(ra_j)) / float(consec + 1)
          else:
            last_rate = 1.0
          accum_err = 0.0
          consec = 0
          prev_v, prev_a = v_cur, a_cur
        prev_expert = ex
        continue

      step_fn = (
          self._sparse_step
          if (self._sparse_step is not None and i >= self.sparse_dense_steps)
          else self._step
      )
      if use_cfg_skip:
        lat_v, lat_a, v_cur, a_cur, cached_dv, cached_da = step_fn(
            self.experts[ex], self.audio, self.bridge, lat_v, lat_a,
            condition, context, self._put(sinusoids[i:i + 1]),
            self._put(s_from[i]), self._put(s_to[i]), guidance,
        )
        cond_since_refresh = 0
        if use_cache:
          if prev_v is not None and same_expert:
            rv_j, ra_j = self._vel_rel(v_cur, prev_v, a_cur, prev_a)
            last_rate = max(float(rv_j), 0.7 * float(ra_j)) / float(consec + 1)
          else:
            last_rate = 1.0
          accum_err = 0.0
          consec = 0
          prev_v, prev_a = v_cur, a_cur
        prev_expert = ex
      elif use_cache:
        lat_v, lat_a, v_cur, a_cur = step_fn(
            self.experts[ex], self.audio, self.bridge, lat_v, lat_a,
            condition, context, self._put(sinusoids[i:i + 1]),
            self._put(s_from[i]), self._put(s_to[i]), guidance,
        )
        if prev_v is not None and same_expert:
          rv_j, ra_j = self._vel_rel(v_cur, prev_v, a_cur, prev_a)
          last_rate = max(float(rv_j), 0.7 * float(ra_j)) / float(consec + 1)
        else:
          last_rate = 1.0
        accum_err = 0.0
        consec = 0
        prev_v, prev_a, prev_expert = v_cur, a_cur, ex
      else:
        lat_v, lat_a = step_fn(
            self.experts[ex], self.audio, self.bridge, lat_v, lat_a,
            condition, context, self._put(sinusoids[i:i + 1]),
            self._put(s_from[i]), self._put(s_to[i]), guidance,
        )
        prev_expert = ex
    if use_cache or use_cfg_skip:
      n_full = steps - n_cached - n_cond_only
      print(f"  [mova/sched] steps={steps} full_cfg={n_full} cond_only_1x8={n_cond_only} cached={n_cached}",
            flush=True)
    return lat_v, lat_a

  def decode(self, lat_v, lat_a):
    """(frames [F, H, W, 3] uint8, audio [samples] float32 at 48 kHz)."""
    video_u8 = self._decode_video(self.vae, self._put(lat_v))
    audio = self._decode_audio(self.dac, self._put(lat_a))
    frames = np.asarray(video_u8)[0]  # already [F, H, W, 3] uint8 from TPU!
    return frames, np.asarray(audio)[0, 0][:self.audio_samples]

  def __call__(self, image, prompt, negative_prompt=NEGATIVE_PROMPT, steps=50, cfg_scale=5.0,
               shift=5.0, seed=42):
    import time

    from jax.sharding import PartitionSpec as P

    t0 = time.perf_counter()
    cond = self.condition(image)
    context = jnp.concatenate([self.encode_text(prompt), self.encode_text(negative_prompt)])
    context = self._put(context, P("cfg"))  # [positive, negative]: split over cfg when 2
    lat_v, lat_a = self.noise(seed)
    jax.block_until_ready((cond, context))
    t1 = time.perf_counter()
    lat_v, lat_a = jax.block_until_ready(
        self.denoise(cond, context, lat_v, lat_a, steps, cfg_scale, shift))
    t2 = time.perf_counter()
    out = self.decode(lat_v, lat_a)
    t3 = time.perf_counter()
    self.times = {"encode": t1 - t0, "denoise": t2 - t1, "decode": t3 - t2}
    return out


def save_mp4(path, frames, audio, fps=24.0, rate=48000):
  import sys

  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
  from frames_to_mp4 import write

  write(frames, path, fps=int(round(fps)), wave=np.stack([audio, audio]), rate=rate)


def main():
  import argparse
  import time

  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from PIL import Image

  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--repo", default=REPO)
  p.add_argument("--image", default="")
  p.add_argument("--prompt", default="")
  p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
  p.add_argument("--height", type=int, default=352)
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--frames", type=int, default=193)
  p.add_argument("--fps", type=float, default=24.0)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--cfg-scale", type=float, default=5.0)
  p.add_argument("--shift", type=float, default=5.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--runs", type=int, default=1)
  p.add_argument("--tensor-parallel", action="store_true",
                 help="video layers tensor parallel (Megatron) instead of sequence parallel")
  p.add_argument("--cfg", type=int, default=0, choices=(0, 1, 2),
                 help="CFG-parallel mesh axis: 0 = 2 with 8+ chips, else 1; 1 above 480x832 on chips under 64 GB")
  p.add_argument("--resident", action=argparse.BooleanOptionalAction, default=None,
                 help="Keep whole DiT weights resident on every device (default: auto = True on >=64 GiB v7x, False on v6e)")
  p.add_argument("--fp8", action="store_true", help="On-TPU W8A8 E4M3 FP8 DiT block linears (halves expert HBM)")
  p.add_argument("--qk-fp8", action="store_true", help="E4M3 FP8 QK^T inside video self-attention Pallas kernel")
  p.add_argument("--block-kv", type=int, default=None, help="Override SplashAttention block_kv (e.g. 4096 on v7x)")
  p.add_argument("--sparse-window", type=int, default=0,
                 help="Temporal-window block-sparse video self-attention radius in latent frames (0 disables; e.g. 12)")
  p.add_argument("--sparse-dense-steps", type=int, default=8,
                 help="Initial dense steps before switching to --sparse-window (default: 8)")
  p.add_argument("--step-cache-thresh", type=float, default=0.0,
                 help="Audio+video Velocity TeaCache threshold (0 disables; --lossy uses 0.25)")
  p.add_argument("--step-cache-max-skip", type=int, default=2)
  p.add_argument("--cfg-skip-after", type=float, default=0.0,
                 help="Fraction of steps after which to run 1x8 cond-only with smoothly decayed cached CFG delta (0 disables; e.g. 0.60)")
  p.add_argument("--cfg-skip-refresh", type=int, default=0,
                 help="Optional full-CFG refresh cadence during --cfg-skip-after (0 = pure smooth decay)")
  p.add_argument("--dit-backend", choices=("jax", "torchax"), default="jax",
                 help="DiT execution backend: native JAX (jax) or PyTorch via torchax (torchax)")
  p.add_argument("--lossy", action="store_true",
                 help="Enable MOVA fast preset (--qk-fp8 --sparse-window 12 --sparse-dense-steps 8 --step-cache-thresh 0.25)")
  p.add_argument("--encode-only", default="", help="Stage 0 only: save condition + text context + noise to .npz and exit (skips 31B DiT)")
  p.add_argument("--decode-latents", default="", help="Stage 3 only: decode denoised latents .npz to --output mp4 (skips text & DiT)")
  p.add_argument("--vae-chunk", type=int, default=None, help="Wan VAE chunk size in latent frames (default: 8 on v7x, 4 on v6e)")
  p.add_argument("--output", default="mova.mp4")
  p.add_argument("--compile-cache-dir", default="~/.cache/mova_jax")
  args = p.parse_args()
  if not args.decode_latents and (not args.image or not args.prompt):
    p.error("--image and --prompt are required unless --decode-latents is set")
  if args.lossy:
    args.qk_fp8 = True
    if args.sparse_window == 0:
      args.sparse_window = 12
    if args.step_cache_thresh == 0.0:
      args.step_cache_thresh = 0.25
  print(args, flush=True)
  if args.compile_cache_dir:
    cache = os.path.expanduser(args.compile_cache_dir)
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
  n = len(jax.devices())
  if not args.cfg:
    hbm = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0)
    big = args.height * args.width > 480 * 832 and hbm < 64 * 2**30
    args.cfg = 2 if n >= 8 and not big else 1
  cfg = args.cfg
  if cfg not in (1, 2) or n % cfg:
    raise SystemExit(f"--cfg must be 1 or 2 and divide the {n} chips (got {cfg})")
  mesh = Mesh(mesh_utils.create_device_mesh((cfg, n // cfg)), ("cfg", "tensor"))
  print(f"mesh cfg={cfg} x tensor={n // cfg}, {jax.devices()[0].device_kind}", flush=True)

  load_dit = not (args.encode_only or args.decode_latents)
  load_text = not bool(args.decode_latents)
  t0 = time.perf_counter()
  gen = Generator(
      args.repo, mesh, height=args.height, width=args.width, frames=args.frames,
      video_fps=args.fps, vae_chunk=args.vae_chunk, seq_parallel=not args.tensor_parallel,
      resident=args.resident, fp8=args.fp8, qk_fp8=args.qk_fp8, block_kv=args.block_kv,
      sparse_window=args.sparse_window, sparse_dense_steps=args.sparse_dense_steps,
      step_cache_thresh=args.step_cache_thresh, step_cache_max_skip=args.step_cache_max_skip,
      cfg_skip_after=args.cfg_skip_after, cfg_skip_refresh=args.cfg_skip_refresh,
      dit_backend=args.dit_backend, load_dit=load_dit, load_text=load_text,
  )
  print(
      f"load: {time.perf_counter() - t0:.1f}s (backend={gen.dit_backend}, resident={gen.resident}, "
      f"fp8={gen.fp8}, qk_fp8={gen.qk_fp8}, sparse_w={gen.sparse_window} [{gen.sparse_density:.1%}], "
      f"cache={gen.step_cache_thresh}, cfg_skip={gen.cfg_skip_after})",
      flush=True,
  )

  if args.encode_only:
    t0 = time.perf_counter()
    image = crop_and_resize(Image.open(args.image), args.height, args.width)
    pos = gen.encode_text(args.prompt)
    neg = gen.encode_text(args.negative_prompt)
    context = jnp.concatenate([pos, neg], 0)
    condition = gen.condition(image)
    lat_v, lat_a = gen.noise(args.seed)
    t_sched, s_from, s_to, expert = schedule(args.steps, args.shift)
    sinusoids = jax_dit.time_sinusoid(t_sched)
    os.makedirs(os.path.dirname(os.path.abspath(args.encode_only)), exist_ok=True)
    np.savez(
        args.encode_only,
        context=np.asarray(context, dtype=np.float32),
        condition=np.asarray(condition, dtype=np.float32),
        lat_v=np.asarray(lat_v, dtype=np.float32),
        lat_a=np.asarray(lat_a, dtype=np.float32),
        t=t_sched.astype(np.float32),
        s_from=s_from.astype(np.float32),
        s_to=s_to.astype(np.float32),
        expert=expert.astype(np.int32),
        sinusoids=sinusoids.astype(np.float32),
        height=np.int32(args.height),
        width=np.int32(args.width),
        frames=np.int32(args.frames),
        fps=np.float32(args.fps),
    )
    print(f"wrote Stage 0 conditioning to {args.encode_only} in {time.perf_counter() - t0:.2f}s", flush=True)
    return

  if args.decode_latents:
    t0 = time.perf_counter()
    lat = np.load(args.decode_latents, allow_pickle=False)
    frames, audio = gen.decode(jnp.asarray(lat["lat_v"]), jnp.asarray(lat["lat_a"]))
    save_mp4(args.output, frames, audio, args.fps)
    print(f"decoded {args.decode_latents} -> {args.output} in {time.perf_counter() - t0:.2f}s "
          f"({frames.shape}, {len(audio) / 48000:.2f} s audio)", flush=True)
    return

  image = crop_and_resize(Image.open(args.image), args.height, args.width)
  for i in range(args.runs):
    t0 = time.perf_counter()
    frames, audio = gen(image, args.prompt, args.negative_prompt, args.steps, args.cfg_scale,
                        args.shift, args.seed)
    split = ", ".join(f"{k} {v:.1f}" for k, v in gen.times.items())
    print(f"run {i}: {time.perf_counter() - t0:.1f}s ({split})"
          + (" (includes compilation)" if i == 0 else ""), flush=True)
  for d in jax.devices():
    st = d.memory_stats()
    print(f"  {d.id}: peak {st.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB", flush=True)
  save_mp4(args.output, frames, audio, args.fps)
  print(f"wrote {args.output} ({frames.shape}, {len(audio) / 48000:.2f} s audio)")


if __name__ == "__main__":
  main()
