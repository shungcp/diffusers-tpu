"""MiniMax-H3's vision tower in native JAX, by reuse.

`ref2va` conditions on a reference image, which goes through
Qwen3-VL's vision tower. H3's is the same tower Qwen-Image-2.1 uses,
at the same size -- every field of `vision_config` matches that port's
defaults except one:

| | Qwen-Image-2.1 | MiniMax-H3 |
|---|---|---|
| depth | 27 | 27 |
| hidden_size | 1152 | 1152 |
| num_heads | 16 | 16 |
| intermediate_size | 4304 | 4304 |
| patch / temporal / merge | 16 / 2 / 2 | 16 / 2 / 2 |
| num_position_embeddings | 2304 | 2304 |
| deepstack indexes | (8, 16, 24) | (8, 16, 24) |
| **out_hidden_size** | **4096** | **5120** |

So this module is a config and a loader that reuses the Qwen-Image-2.1
implementation (as `jax_text.py` does for the language tower).

The checkpoint keeps the tower under `model.visual.` inside
`text_encoder/`; all 351 of its tensors sit in the last of fourteen
shards, so the vision tower can be fetched without the other 62 GiB.
"""

import glob
import json
import os

import jax
import jax.numpy as jnp
from flax import nnx

from ..qwen3_vl import jax_vision as qwen_vision

PREFIX = "model.visual."

# The shard that holds every `model.visual.*` tensor (and none of the
# language tower's), for use with `allow_patterns` when fetching weights.
SHARD = "model-00014-of-00014.safetensors"


def config_from(text_encoder_dir):
  """Read the tower's shape off the checkpoint rather than assuming it."""
  with open(os.path.join(text_encoder_dir, "config.json")) as f:
    cfg = json.load(f)["vision_config"]
  out = dict(qwen_vision.DEFAULTS)
  out.update(
      depth=cfg["depth"],
      hidden_size=cfg["hidden_size"],
      num_heads=cfg["num_heads"],
      intermediate_size=cfg["intermediate_size"],
      patch_size=cfg["patch_size"],
      temporal_patch_size=cfg["temporal_patch_size"],
      spatial_merge_size=cfg["spatial_merge_size"],
      in_channels=cfg["in_channels"],
      out_hidden_size=cfg["out_hidden_size"],
      num_position_embeddings=cfg["num_position_embeddings"],
      deepstack_indexes=tuple(cfg["deepstack_visual_indexes"]),
  )
  return out


def build(text_encoder_dir, *, dtype=jnp.bfloat16, rngs=None, abstract=False):
  """An uninitialised H3 vision tower shaped by its own config."""
  Vision = qwen_vision._build_modules()
  cfg = config_from(text_encoder_dir)
  make = lambda: Vision(**cfg, dtype=dtype, rngs=rngs or nnx.Rngs(0))
  return (nnx.eval_shape(make) if abstract else make()), cfg


def load(model, text_encoder_dir, mesh=None, spec_fn=None, place=None,
         memory_kind=None):
  """Read the vision half of H3's checkpoint into the nnx module.

  Every parameter must be filled. A name that does not resolve is an
  error rather than a warning, to catch silently skipped weights.
  """
  from jax.sharding import NamedSharding
  from ...safetensors_io import safe_open

  shards = sorted(glob.glob(os.path.join(text_encoder_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {text_encoder_dir}")

  def resolve(name):
    # Two names do not map straight onto the module tree: the position
    # embedding is a bare Param rather than a submodule, and the patch
    # embedding is a Conv3d in the checkpoint (`patch_embed.proj.*`,
    # shape `(embed, channels, t, p, p)`) but a Linear here.
    if name.startswith("pos_embed."):
      return model.pos_embed
    if name.startswith("patch_embed.proj."):
      return getattr(model.patch_embed,
                     "bias" if name.endswith("bias") else "weight")
    return qwen_vision._resolve(model, name.split("."))

  if (os.environ.get("DT_FAST_LOAD", "1") == "1" and place is None
      and mesh is not None and spec_fn is not None):
    return _load_parallel(model, shards, mesh, spec_fn, memory_kind, resolve)

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if not key.startswith(PREFIX):
          continue
        name = key[len(PREFIX):]
        param = resolve(name)
        want = param.value if hasattr(param, "value") else param[...]
        value = reader.get_tensor(key).astype(want.dtype).reshape(want.shape)
        if place is not None:
          value = jax.device_put(value, place)
        elif mesh is not None and spec_fn is not None:
          value = jax.device_put(
              value,
              NamedSharding(mesh, spec_fn(name.split("."), value.ndim),
                            memory_kind=memory_kind))
        if hasattr(param, "value"):
          param.value = value
        else:
          param[...] = value
        loaded += 1

  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return loaded


def _load_parallel(model, shards, mesh, spec_fn, memory_kind, resolve):
  """Parallel version of `load`.

  All 351 tensors live in one shard, so work is split per tensor across 16
  threads (each with its own reader). The serial path takes ~35 s on a cold read.
  """
  import concurrent.futures as cf

  import ml_dtypes
  import numpy as np
  import torch
  from jax.sharding import NamedSharding
  from ...safetensors_io import safe_open

  keys = []
  for shard in shards:
    with safe_open(shard, framework="pt") as reader:
      keys += [(shard, k) for k in reader.keys() if k.startswith(PREFIX)]

  def want_of(param):
    return param.value if hasattr(param, "value") else param[...]

  def work(chunk):
    out = []
    readers = {}
    try:
      for shard, key in chunk:
        if shard not in readers:
          readers[shard] = safe_open(shard, framework="pt").__enter__()
        name = key[len(PREFIX):]
        want = want_of(resolve(name))
        t = readers[shard].get_tensor(key)
        arr = (t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
               if t.dtype == torch.bfloat16 else t.numpy())
        arr = arr.astype(np.dtype(want.dtype), copy=False).reshape(want.shape)
        out.append((name, jax.device_put(arr, NamedSharding(
            mesh, spec_fn(name.split("."), arr.ndim), memory_kind=memory_kind))))
    finally:
      for r in readers.values():
        r.__exit__(None, None, None)
    return out

  n = 16
  chunks = [keys[i::n] for i in range(n)]
  with cf.ThreadPoolExecutor(max_workers=n) as ex:
    results = list(ex.map(work, chunks))
  loaded = 0
  for out in results:
    for name, value in out:
      param = resolve(name)
      if hasattr(param, "value"):
        param.value = value
      else:
        param[...] = value
      loaded += 1
  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return loaded
