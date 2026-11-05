"""`safe_open` that reads each tensor with one large sequential read.

safetensors' own `safe_open` memory-maps the file, so a tensor arrives as
page faults: 4 KB at a time plus whatever the kernel reads ahead. On a disk
limited in operations per second rather than bytes, that is all the disk
does -- the Colab TPU runtime's disk (a loop device, read-ahead fixed at
128 KB and not settable from inside the container) read a checkpoint at
0.5 MB/s that way. Here every tensor is one `readinto` of its whole byte
range, and `keys()` lists tensors in file order, so a shard is read front to
back in a few large requests.

Same interface and same bytes as `safetensors.safe_open` for the parts this
repository uses (`keys`, `get_tensor`, and the context manager), for the
`np`, `pt` and `flax` frameworks. DT_SAFETENSORS_MMAP=1 restores the
memory-mapped reader.
"""

import json
import os
import struct

import numpy as np

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _np_dtype(name):
  if name == "BF16":
    import ml_dtypes
    return np.dtype(ml_dtypes.bfloat16)
  if name in ("F8_E4M3", "F8_E5M2"):
    import ml_dtypes
    return np.dtype(ml_dtypes.float8_e4m3fn if name == "F8_E4M3"
                    else ml_dtypes.float8_e5m2)
  return np.dtype(_DTYPES[name])


class _Reader:
  def __init__(self, path, framework):
    if framework not in ("np", "numpy", "pt", "flax", "jax"):
      raise ValueError(f"framework {framework!r} not supported")
    self.framework = framework
    self._f = open(path, "rb", buffering=0)
    (n,) = struct.unpack("<Q", self._f.read(8))
    header = json.loads(self._f.read(n))
    header.pop("__metadata__", None)
    self._base = 8 + n
    self._meta = header
    self._order = sorted(header, key=lambda k: header[k]["data_offsets"][0])

  def keys(self):
    return list(self._order)

  def get_numpy(self, key):
    m = self._meta[key]
    start, stop = m["data_offsets"]
    arr = np.empty(stop - start, np.uint8)
    self._f.seek(self._base + start)
    view, got = memoryview(arr), 0
    while got < arr.size:
      k = self._f.readinto(view[got:])
      if not k:
        raise IOError(f"short read for {key}: {got} of {arr.size} bytes")
      got += k
    return arr.view(_np_dtype(m["dtype"])).reshape(m["shape"])

  def get_tensor(self, key):
    arr = self.get_numpy(key)
    if self.framework == "pt":
      import torch
      if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
      return torch.from_numpy(arr)
    if self.framework in ("flax", "jax"):
      import jax.numpy as jnp
      return jnp.asarray(arr)
    return arr

  def close(self):
    self._f.close()

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
    return False


def safe_open(path, framework="np", device="cpu"):
  if os.environ.get("DT_SAFETENSORS_MMAP") == "1":
    from safetensors import safe_open as _mmap_open
    return _mmap_open(path, framework=framework, device=device)
  return _Reader(path, framework)
