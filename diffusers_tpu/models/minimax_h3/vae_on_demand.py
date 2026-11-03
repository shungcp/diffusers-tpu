"""Keep the VAE in HBM only while it is in use (torchax).

The VAE holds about 9.7 GiB of HBM but is idle for the whole text-encoding
and denoising phases. Parking it in pinned host memory frees that space,
e.g. enough for `--text-encoder-resident` instead of streaming the text
encoder layer by layer.

Uses the same mechanism as the other offloads: `jax.device_put` to
`memory_kind="pinned_host"`, and back to device on use. The sharding spec
is unchanged, so the arithmetic is identical.
"""

from __future__ import annotations

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import torch
import torchax


def install(vae, mesh, verbose=print) -> int:
  """Park the VAE in pinned host memory; move it to HBM only during encode/decode.

  Returns the number of tensors moved. Both entry points are wrapped:
  reference-to-video calls `vae.encode` on the reference image before
  denoising, not just `decode` afterwards.
  """
  env = torchax.default_env()
  host = lambda spec: NamedSharding(mesh, spec, memory_kind="pinned_host")  # noqa: E731
  dev = lambda spec: NamedSharding(mesh, spec)  # noqa: E731

  parked: list = []

  def move(to_host: bool) -> int:
    n = 0
    spec = P()          # VAE is replicated, not sharded
    tgt = host(spec) if to_host else dev(spec)
    for holder_attr in ("_encode_clip", "_decode_clip"):
      holder = getattr(vae, holder_attr, None)
      params_dict = getattr(holder, "_params", None)
      if isinstance(params_dict, dict):
        for k, arr in list(params_dict.items()):
          if isinstance(arr, jax.Array):
            params_dict[k] = jax.device_put(arr, tgt)
            n += 1
    for mod in vae.modules():
      for store, wrap in ((mod._parameters, True), (mod._buffers, False)):
        for name, t in list(store.items()):
          if t is None:
            continue
          raw = t.data if isinstance(t, torch.nn.Parameter) else t
          if not isinstance(raw, (torchax.tensor.Tensor, torchax.tensor.View)):
            continue
          arr = env.t2j_iso(raw)
          del wrap
          store[name] = env.j2t_iso(jax.device_put(arr, tgt))
          n += 1
    return n

  count = move(to_host=True)
  verbose(f"  [vae-on-demand] moved {count} VAE tensors to pinned host",
          flush=True)

  said = {"n": 0}

  def wrap(fn_name):
    inner = getattr(vae, fn_name)

    def staged(*a, **k):
      up = move(to_host=False)
      if said["n"] < 2:
        said["n"] += 1
        verbose(f"  [vae-on-demand] before {fn_name}: restored {up} tensors", flush=True)
      try:
        return inner(*a, **k)
      finally:
        move(to_host=True)

    setattr(vae, fn_name, staged)

  for name in ("encode", "decode"):
    if hasattr(vae, name):
      wrap(name)
  return count
