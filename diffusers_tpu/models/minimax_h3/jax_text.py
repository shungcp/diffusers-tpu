"""MiniMax-H3's text encoder in native JAX, by reuse.

H3 conditions on `Qwen3VLForConditionalGeneration`, and so does
Qwen-Image-2.1. They are the same architecture at different sizes:

| | Qwen-Image-2.1 | MiniMax-H3 |
|---|---|---|
| layers | 36 | **64** |
| hidden | 4096 | **5120** |
| query heads | 32 | **64** |
| kv heads | 8 | 8 |
| intermediate | 12288 | **25600** |
| head dim | 128 | 128 |
| rope theta | 5e6 | 5e6 |
| mrope section | (24, 20, 20) | (24, 20, 20) |
| mrope interleaved | yes | yes |

The tricky parts -- per-head RMSNorm on q and k before RoPE, and the
interleaved mRoPE, where frequencies alternate between the three axes --
are identical and implemented in `models/qwen3_vl/jax_text.py`,
which is config-driven. So this module is a config and a loader, not a
second implementation.

The checkpoint names differ only in the prefix: H3 stores the language
half under `model.language_model.` where Qwen-Image-2.1 uses
`model.`.
"""

import glob
import json
import os

import jax
import jax.numpy as jnp
from flax import nnx

from ..qwen3_vl import jax_text as qwen_text

# H3 keeps the language tower under `model.language_model.`; `lm_head`
# and `model.visual.*` are not read -- nothing here generates tokens,
# and the vision tower is its own module and only `ref2va` needs it.
PREFIX = "model.language_model."


def config_from(text_encoder_dir):
  """Read the shape of the language tower off the checkpoint."""
  with open(os.path.join(text_encoder_dir, "config.json")) as f:
    cfg = json.load(f)["text_config"]
  scaling = cfg.get("rope_scaling") or {}
  return dict(
      num_layers=cfg["num_hidden_layers"],
      hidden_size=cfg["hidden_size"],
      head_dim=cfg["head_dim"],
      num_heads=cfg["num_attention_heads"],
      num_kv_heads=cfg["num_key_value_heads"],
      intermediate_size=cfg["intermediate_size"],
      vocab_size=cfg["vocab_size"],
      rms_norm_eps=cfg["rms_norm_eps"],
      rope_theta=float(cfg["rope_theta"]),
      mrope_section=tuple(scaling.get("mrope_section", (24, 20, 20))),
  )


def build(text_encoder_dir, *, dtype=jnp.bfloat16, rngs=None,
          abstract=False):
  """An uninitialised H3 language tower shaped by its own config."""
  Model = qwen_text._build_modules()
  cfg = config_from(text_encoder_dir)
  make = lambda: Model(**cfg, dtype=dtype, rngs=rngs or nnx.Rngs(0))
  return (nnx.eval_shape(make) if abstract else make()), cfg


def param_spec(keys, ndim):
  """Qwen's layout with this repository's axis name.

  The shared implementation names its axis `tensor` and the
  DiT here names it `tp`; a spec that mentions an axis the
  mesh does not have is not a sharding error, it is a crash
  at `device_put`.
  """
  from jax.sharding import PartitionSpec as P

  spec = qwen_text.param_spec(keys, ndim)
  return P(*[None if a is None else "tp" for a in spec])


def load(model, text_encoder_dir, mesh=None, spec_fn=param_spec, place=None,
         memory_kind=None, int8=False, keep_layers=None):
  """Read the language half of H3's checkpoint into the nnx module.

  64 layers at 5120 wide is 24 GiB in bfloat16, which does not fit a
  v5e chip beside anything else, so `mesh`/`spec_fn` are how a caller
  shards or parks it. Every tensor must land: a name that does not
  resolve is an error rather than a warning, because the failure this
  guards against is a silently skipped weight.
  """
  from safetensors import safe_open
  from jax.sharding import NamedSharding

  shards = sorted(glob.glob(os.path.join(text_encoder_dir, "*.safetensors")))
  if not shards:
    raise FileNotFoundError(f"no safetensors in {text_encoder_dir}")

  if (os.environ.get("DT_FAST_LOAD", "1") == "1" and mesh is not None
      and spec_fn is not None and place is None):
    return _load_parallel(model, shards, mesh, spec_fn, memory_kind,
                          int8=int8, keep_layers=keep_layers)
  if int8 or keep_layers:
    raise NotImplementedError("int8 / keep_layers need the parallel loader")

  loaded = 0
  for shard in shards:
    with safe_open(shard, framework="flax") as reader:
      for key in reader.keys():
        if not key.startswith(PREFIX):
          continue
        name = key[len(PREFIX):]
        if name == "embed_tokens.weight":
          param = model.embed_tokens
        else:
          param = qwen_text._resolve(model, name.split("."))
        want = param.value if hasattr(param, "value") else param[...]
        value = reader.get_tensor(key).astype(want.dtype)
        if value.shape != want.shape:
          raise ValueError(
              f"{key}: checkpoint {value.shape} vs model {want.shape}")
        if place is not None:
          value = jax.device_put(value, place)
        elif mesh is not None and spec_fn is not None:
          # 24 GiB unsharded would OOM chip 0, so the tower is split the
          # way the DiT is. `memory_kind` can park the split copy in host
          # memory rather than HBM.
          value = jax.device_put(
              value, NamedSharding(mesh, spec_fn(name.split("."), value.ndim),
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


def _load_parallel(model, shards, mesh, spec_fn, memory_kind, int8=False,
                   keep_layers=None):
  """Parallel version of `load`: one thread per shard, each tensor placed
  directly onto its sharding after the host read.

  Same approach as `jax_dit._load_transformer_parallel`; on a cold page
  cache the serial text-tower load takes ~269 s.
  """
  import concurrent.futures as cf

  import ml_dtypes
  import numpy as np
  import torch
  from jax.sharding import NamedSharding
  from safetensors import safe_open

  def target(name):
    if name == "embed_tokens.weight":
      param = model.embed_tokens
    else:
      param = qwen_text._resolve(model, name.split("."))
    want = param.value if hasattr(param, "value") else param[...]
    return param, want

  if keep_layers is not None:
    # H3 conditions on `hidden_states[50]`: layers 50..63 never run, and
    # dropping them saves 14 of the tower's 64 layers of weight.
    model.layers = nnx.List(list(model.layers)[:keep_layers])

  def one_shard(shard):
    out = []
    with safe_open(shard, framework="pt") as reader:
      for key in reader.keys():
        if not key.startswith(PREFIX):
          continue
        name = key[len(PREFIX):]
        parts = name.split(".")
        if (keep_layers is not None and parts[0] == "layers"
            and int(parts[1]) >= keep_layers):
          continue
        _, want = target(name)
        t = reader.get_tensor(key)
        if int8 and parts[0] == "layers" and name.endswith("_proj.weight"):
          from .jax_dit import _quantize_int8
          q, sc = _quantize_int8(t)
          from .jax_dit import cache_put
          cache_put(name, q.numpy())
          cache_put(name[:-len("weight")] + "weight_scale", sc.numpy())
          sh = NamedSharding(mesh, spec_fn(parts, 2), memory_kind=memory_kind)
          sc_sh = NamedSharding(mesh, jax.sharding.PartitionSpec(),
                                memory_kind=memory_kind)
          out.append((name, (jax.device_put(q.numpy(), sh),
                             jax.device_put(sc.numpy(), sc_sh))))
          continue
        arr = (t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
               if t.dtype == torch.bfloat16 else t.numpy())
        arr = arr.astype(np.dtype(want.dtype), copy=False)
        if arr.shape != want.shape:
          raise ValueError(f"{key}: checkpoint {arr.shape} vs model {want.shape}")
        from .jax_dit import cache_put
        # The module holds the embedding as a bare Param named `embed_tokens`.
        cache_put("embed_tokens" if name == "embed_tokens.weight" else name, arr)
        out.append((name, jax.device_put(arr, NamedSharding(
            mesh, spec_fn(name.split("."), arr.ndim), memory_kind=memory_kind))))
    return out

  with cf.ThreadPoolExecutor(max_workers=len(shards)) as ex:
    results = list(ex.map(one_shard, shards))
  loaded = 0
  n_scales = 0
  for out in results:
    for name, value in out:
      param, _ = target(name)
      if isinstance(value, tuple):
        node = model
        for part in name.split(".")[:-1]:
          node = node[int(part)] if part.isdigit() else getattr(node, part)
        node.weight = nnx.Param(value[0])
        node.weight_scale = nnx.data(nnx.Param(value[1]))
        loaded += 1
        n_scales += 1
        continue
      if hasattr(param, "value"):
        param.value = value
      else:
        param[...] = value
      loaded += 1
  expected = len(jax.tree.leaves(nnx.state(model, nnx.Param)))
  expected -= n_scales
  if loaded != expected:
    raise ValueError(f"loaded {loaded} tensors into {expected} parameters")
  return loaded
