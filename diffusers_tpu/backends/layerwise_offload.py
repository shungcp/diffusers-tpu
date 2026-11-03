# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from SGLang (https://github.com/sgl-project/sglang),
# licensed under the Apache License, Version 2.0.

"""Layerwise host-resident parameter offload for DiT models on torch_tpu.

A 14B DiT does not fit in one v6e chip's 31 GiB of HBM, so a single-chip run
has to keep the weights on the host and stream each block in just before it
runs. `.to('tpu')` is the obvious way to do that, but it is the slow path: it
converts host row-major data into the TPU's tiled layout on the CPU, and for
bfloat16 that relayout -- not the PCIe link -- is the bottleneck (7.2 GB/s on
v6e against 26.3 for a raw DMA).

So the weights are converted once, at load time, and kept on the host in
device layout: copy the tensor up with `.to('tpu')`, pull the tiled bytes back
with `batch_transfer_d2h`, and hold them in pinned memory. Every later upload
is then a raw DMA that skips the relayout entirely, at a bandwidth that does
not depend on the dtype.

The transfer does not need to overlap with compute to be worth it. One chip
holds ~27 GiB of Wan weights; at 26.3 GB/s that is ~1.0 s per step against
~78 s of single-chip compute, so a synchronous stream costs a little over one
percent.

`enabled=False` restores the previous behaviour (weights resident on device),
which is what the 8-chip configs want.
"""

import logging
import math
import os
import time

import torch


def _stage_h2d(tensor: "torch.Tensor", device) -> "torch.Tensor":
  """Pushes one host tensor up, pinning and bitcasting to int32 first.

  On v6e `.to('tpu')` only reaches full H2D bandwidth for 4-byte dtypes, so
  2-byte weights are sent as int32 and bitcast back. With the premapped
  buffer (`TPU_PREMAPPED_BUFFER_SIZE`) sized to the working set this cuts
  cold-start H2D from ~43 s to ~31 s; with a too-small premapped region the
  slower D2H cancels the gain. Output is bitwise identical.
  `DT_OFFLOAD_PLAIN_H2D=1` uses the plain `.to()` path.
  """
  if os.environ.get("DT_OFFLOAD_PLAIN_H2D") == "1":
    return tensor.to(device)
  if not tensor.is_pinned():
    tensor = tensor.contiguous().pin_memory()
  flat = tensor.reshape(-1)
  if not flat.is_contiguous() or (flat.numel() * flat.element_size()) % 4:
    return tensor.to(device)
  dtype, shape = flat.dtype, tensor.shape
  return flat.view(torch.int32).to(device).view(dtype).reshape(shape)

logger = logging.getLogger(__name__)

try:
  from torch_tpu._internal.batch_transfer import batch_transfer_d2h_sync
  from torch_tpu._internal.batch_transfer import await_all
  from torch_tpu._internal.batch_transfer import batch_transfer_h2d
  from torch_tpu._internal.batch_transfer import batch_transfer_h2d_sync

  _RAW_TRANSFER_AVAILABLE = True
except Exception:  # pragma: no cover - torch_tpu without the experimental API
  _RAW_TRANSFER_AVAILABLE = False


# Second-minor tile height by element width: 4- and 2-byte types tile as
# (8,128), 1-byte types as (32,128).
_SUBTILE_ROWS = {4: 8, 2: 8, 1: 32}
_TILE_COLS = 128


def tiled_numel(shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> int:
  """Element count a TPU-layout buffer needs for `shape`, padding included.

  A raw copy moves the padded physical buffer, so a host buffer sized to the
  logical shape is rejected ("Destination buffer too small for raw copy") for
  anything that is not already tile-aligned.
  """
  dims = list(shape) or [1]
  if len(dims) == 1:
    dims = [1] + dims
  rows = _SUBTILE_ROWS[dtype.itemsize]
  dims[-1] = math.ceil(dims[-1] / _TILE_COLS) * _TILE_COLS
  dims[-2] = math.ceil(dims[-2] / rows) * rows
  count = 1
  for dim in dims:
    count *= dim
  return count


class LayerwiseOffloadManager:
  """Keeps one DiT block's parameters on the device at a time.

  Parameters live in pinned host buffers holding the device's own tiled
  layout; `prefetch_layer` DMAs one block up and `release_layer` drops it.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      *,
      layers_attr_str: str,
      num_layers: int,
      enabled: bool,
      pin_cpu_memory: bool = True,
      prefetch_size: int = 1,
  ) -> None:
    self.model = model
    self.layers_attr_str = layers_attr_str
    self.num_layers = num_layers
    self.pin_cpu_memory = pin_cpu_memory
    self.prefetch_size = min(max(1, prefetch_size), max(1, num_layers))
    self.enabled = bool(enabled) and _RAW_TRANSFER_AVAILABLE
    if enabled and not _RAW_TRANSFER_AVAILABLE:
      raise RuntimeError(
          "Layerwise offload needs torch_tpu._internal.batch_transfer, which "
          "this build does not provide."
      )
    self._forward_hooks: list[object] = []
    # layer index -> in-flight `TransferFuture`s, when
    # `DT_PREFETCH_ASYNC=1`. Empty on the sync path.
    self._pending: dict[int, list] = {}
    # layer index -> list of (parameter, host buffer in device layout, shape).
    # The shape is kept here because a released parameter's `.data` is a
    # zero-element placeholder, so `tensor.shape` no longer describes it.
    self._host_copies: dict[
        int, list[tuple[torch.Tensor, torch.Tensor, torch.Size]]
    ] = {}
    # layer index -> device tensors, held so they outlive the block's forward
    self._resident: dict[int, list[torch.Tensor]] = {}
    # Blocks [0, pinned) stay on the chip for the whole request.
    self.pinned: int = 0
    self._device: torch.device | None = None

  # -- layer access ---------------------------------------------------------

  def _layers(self) -> torch.nn.ModuleList:
    layers = self.model
    for part in self.layers_attr_str.split("."):
      layers = getattr(layers, part)
    return layers

  @staticmethod
  def _tensors(layer: torch.nn.Module):
    yield from layer.named_parameters(recurse=True)
    yield from layer.named_buffers(recurse=True)

  @staticmethod
  def _slots(layer: torch.nn.Module):
    """Yield `(owner module, attribute, tensor, is_param)` for a block.

    Tensors are swapped by replacing the entry on the owning module rather
    than assigning `tensor.data`: when staging from the host the CPU and
    device tensors have different types, and torch_tpu rejects `.data`
    assignment across types.
    """
    for module in layer.modules():
      for attr, param in list(module.named_parameters(recurse=False)):
        yield module, attr, param, True
      for attr, buf in list(module.named_buffers(recurse=False)):
        if buf is not None:
          yield module, attr, buf, False

  @staticmethod
  def _put(module, attr, tensor, is_param):
    if is_param:
      module._parameters[attr] = torch.nn.Parameter(
          tensor, requires_grad=False)
    else:
      module._buffers[attr] = tensor

  # -- setup ----------------------------------------------------------------

  def offload_all_layers(self, stage_from_host: bool = False) -> None:
    """Move every block's tensors to pinned host memory in device layout.

    Call this once the weights are loaded and still on the device. Each tensor
    is pulled down and freed one at a time, so peak HBM never holds more than
    the model already did.

    `stage_from_host` is for models that never fit the device even once:
    each host tensor is pushed up, converted, and freed, so the peak is one
    tensor rather than one model. It needs `device` to say where.
    """
    if not self.enabled:
      return
    if stage_from_host and self._device is None:
      self._device = torch.device("tpu")
    layers = self._layers()
    total_bytes = 0
    # DT_OFFLOAD_TIMING=1 times each phase (H2D, alloc, D2H, stub) separately.
    timing = os.environ.get("DT_OFFLOAD_TIMING") == "1"
    t_h2d = t_alloc = t_d2h = t_put = 0.0
    # H2D time bucketed by tensor size separates per-tensor fixed overhead
    # (small buckets slow) from bandwidth limits (all buckets slow).
    buckets = [(1 << 16, "<64KiB"), (1 << 20, "64KiB-1MiB"),
               (1 << 24, "1-16MiB"), (1 << 62, ">16MiB")]
    hist = {name: [0, 0, 0.0] for _, name in buckets}  # count, bytes, seconds
    for idx, layer in enumerate(layers):
      copies = []
      for module, attr, tensor, is_param in self._slots(layer):
        t0 = time.perf_counter() if timing else 0.0
        if tensor.device.type == "cpu":
          # Models too large for the device arrive with blocks still on the
          # host; staging one tensor at a time keeps the HBM peak at a
          # single tensor.
          if not stage_from_host:
            continue
          staged = _stage_h2d(tensor.detach(), self._device)
        else:
          staged = tensor.detach()
        if timing:
          t1 = time.perf_counter(); t_h2d += t1 - t0
          nbytes = tensor.numel() * tensor.element_size()
          for limit, name in buckets:
            if nbytes < limit:
              b = hist[name]
              b[0] += 1; b[1] += nbytes; b[2] += t1 - t0
              break
        shape = tuple(staged.shape)
        host = torch.empty(
            tiled_numel(shape, staged.dtype),
            dtype=staged.dtype,
            pin_memory=self.pin_cpu_memory,
        )
        if timing:
          t2 = time.perf_counter(); t_alloc += t2 - t1
        batch_transfer_d2h_sync([staged], [host])
        if timing:
          t3 = time.perf_counter(); t_d2h += t3 - t2
        copies.append((module, attr, host, torch.Size(shape), is_param))
        total_bytes += host.numel() * host.element_size()
        del staged
        self._put(module, attr,
                  torch.empty(0, dtype=host.dtype, device=self._device),
                  is_param)
        if timing:
          t_put += time.perf_counter() - t3
      self._host_copies[idx] = copies
    if timing:
      print(f"  [offload timing] H2D+detach {t_h2d:.2f}s  "
            f"pinned alloc {t_alloc:.2f}s  D2H {t_d2h:.2f}s  "
            f"put-stub {t_put:.2f}s", flush=True)
      for _, name in buckets:
        n, nb, t = hist[name]
        if not n:
          continue
        print(f"  [offload h2d] {name:>11}  {n:6d} tensors  "
              f"{nb / 2**30:7.2f} GiB  {t:7.2f}s  "
              f"{nb / t / 1e9:5.2f} GB/s  {t / n * 1e3:6.2f} ms each",
              flush=True)
    logger.info(
        "Layerwise offload: %d blocks, %.2f GiB parked on the host",
        len(layers),
        total_bytes / 2**30,
    )
    print(f"  offload: {len(layers)} blocks, "
          f"{total_bytes / 2**30:.1f} GiB on the host", flush=True)
    if os.environ.get("DT_PREFETCH_PROBE") == "1":
      self._idle_prefetch_probe(len(layers))

  def _idle_prefetch_probe(self, n_layers: int) -> None:
    """Runs the normal prefetch path with the device idle (no DiT compute).

    Comparing this bandwidth with prefetch during denoising isolates
    DMA/compute contention from per-call allocation overhead.
    """
    import time as _t
    nb = t = 0.0
    for i in range(n_layers):
      copies = self._host_copies.get(i)
      if not copies:
        continue
      t0 = _t.perf_counter()
      self.prefetch_layer(i)
      t += _t.perf_counter() - t0
      nb += sum(h.numel() * h.element_size() for _, _, h, _, _ in copies)
      self.release_layer(i)
    if t:
      print(f"  [prefetch probe idle] {n_layers} layers  {nb / 2**30:.2f} GiB  "
            f"{t:.2f}s  {nb / t / 1e9:.2f} GB/s", flush=True)

  # -- streaming ------------------------------------------------------------

  def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
    del non_blocking  # the raw path is synchronous
    if not self.enabled or layer_idx in self._resident:
      return
    copies = self._host_copies.get(layer_idx)
    if not copies:
      return
    device_tensors: list[torch.Tensor] = []
    # A batch_transfer call requires every tensor in it to share one shape --
    # the API was written for KV cache blocks -- so group by shape and dtype.
    batches: dict[tuple, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for _, _, host, shape, _ in copies:
      staged = torch.empty(shape, dtype=host.dtype, device=self._device)
      device_tensors.append(staged)
      key = (tuple(shape), host.dtype)
      srcs, dsts = batches.setdefault(key, ([], []))
      srcs.append(host)
      dsts.append(staged)
    timing = os.environ.get("DT_OFFLOAD_TIMING") == "1"
    t0 = time.perf_counter() if timing else 0.0
    # `DT_PREFETCH_ASYNC=1` issues the copies without waiting; `await_layer`
    # blocks on them later. The sync path is ~10x slower during denoising
    # than on an idle device, either from HBM contention or from queueing
    # behind pending compute; async issue lets the host continue meanwhile.
    if os.environ.get("DT_PREFETCH_ASYNC") == "1":
      futures = [batch_transfer_h2d(srcs, dsts)
                 for srcs, dsts in batches.values()]
      self._pending.setdefault(layer_idx, []).extend(futures)
    else:
      for srcs, dsts in batches.values():
        batch_transfer_h2d_sync(srcs, dsts)
    if timing:
      # Accumulate prefetch bandwidth and batching stats; print every 50 calls.
      nb = sum(h.numel() * h.element_size() for _, _, h, _, _ in copies)
      self._pf_bytes = getattr(self, "_pf_bytes", 0) + nb
      self._pf_time = getattr(self, "_pf_time", 0.0) + time.perf_counter() - t0
      self._pf_calls = getattr(self, "_pf_calls", 0) + 1
      self._pf_batches = getattr(self, "_pf_batches", 0) + len(batches)
      self._pf_tensors = getattr(self, "_pf_tensors", 0) + len(copies)
      if self._pf_calls % 50 == 0:
        print(f"  [prefetch h2d] {self._pf_calls} calls  "
              f"{self._pf_bytes / 2**30:.2f} GiB  {self._pf_time:.2f}s  "
              f"{self._pf_bytes / self._pf_time / 1e9:.2f} GB/s  "
              f"| per layer {self._pf_batches / self._pf_calls:.1f} batches / "
              f"{self._pf_tensors / self._pf_calls:.1f} tensors, "
              f"avg {self._pf_bytes / self._pf_batches / 2**20:.1f} MiB/batch",
              flush=True)
    for (module, attr, host, _, is_param), staged in zip(copies, device_tensors):
      self._put(module, attr, staged, is_param)
    self._resident[layer_idx] = device_tensors

  def await_layer(self, layer_idx: int) -> None:
    """Blocks until `layer_idx`'s async copies have landed."""
    pending = getattr(self, "_pending", {}).pop(layer_idx, None)
    if pending:
      await_all(pending)

  def release_layer(self, layer_idx: int) -> None:
    # Pinned layers stay resident. When full residency does not fit, pinning
    # as many blocks as HBM allows removes their host traffic and leaves the
    # rest streaming.
    if layer_idx < self.pinned:
      return
    if not self.enabled or layer_idx not in self._resident:
      return
    for module, attr, host, _, is_param in self._host_copies.get(layer_idx, ()):
      self._put(module, attr,
                torch.empty(0, dtype=host.dtype, device=self._device),
                is_param)
    del self._resident[layer_idx]

  def release_all(self) -> None:
    for layer_idx in list(self._resident):
      self.release_layer(layer_idx)

  def load_all_layers(self) -> None:
    """Put every block back on the device (undoes the offload)."""
    if not self.enabled:
      return
    for layer_idx in self._host_copies:
      self.prefetch_layer(layer_idx)

  def prepare_for_next_req(self, non_blocking: bool = True) -> None:
    if not self.enabled:
      return
    self.release_all()
    self.prefetch_layer(0, non_blocking=non_blocking)

  # -- hooks ----------------------------------------------------------------

  def register_forward_hooks(self) -> None:
    if not self.enabled or self._forward_hooks:
      return

    def pre_hook(idx):
      def _hook(module, args):
        del module, args
        self.prefetch_layer(idx)
        for ahead in range(1, self.prefetch_size + 1):
          if idx + ahead < self.num_layers:
            self.prefetch_layer(idx + ahead)
        # Wait only for `idx`; lookahead copies land under this block's
        # compute.
        self.await_layer(idx)
        return None

      return _hook

    def post_hook(idx):
      def _hook(module, args, output):
        del module, args
        # Keep `prefetch_size` blocks in flight: drop the one furthest behind.
        stale = idx - self.prefetch_size
        if stale >= 0:
          self.release_layer(stale)
        return output

      return _hook

    for idx, layer in enumerate(self._layers()):
      self._forward_hooks.append(
          layer.register_forward_pre_hook(pre_hook(idx))
      )
      self._forward_hooks.append(layer.register_forward_hook(post_hook(idx)))

  def remove_forward_hooks(self) -> None:
    for hook_handle in self._forward_hooks:
      hook_handle.remove()
    self._forward_hooks.clear()


class OffloadableDiTMixin:
  """Gives a DiT the hooks that stream its blocks in from host memory."""

  layer_names: list[str] = []
  layerwise_offload_managers: list[LayerwiseOffloadManager] = []

  def configure_layerwise_offload(self, **kwargs: object) -> None:
    """Park the blocks on the host and hook the forward pass.

    `layer_names` names the `nn.ModuleList`s to stream ("blocks" for Wan).
    With `enabled=False` this stays a no-op and the weights keep sitting on
    the device, which is what the multi-chip configs want.
    """
    enabled = bool(kwargs.get("enabled", False))
    prefetch_size = int(kwargs.get("prefetch_size", 1))
    pin_cpu_memory = bool(kwargs.get("pin_cpu_memory", True))
    layer_names = kwargs.get("layer_names") or self.layer_names or ["blocks"]

    self.layerwise_offload_managers = []
    if not enabled:
      logger.info("Layerwise offload disabled; weights stay on device.")
      return

    for name in layer_names:
      layers = self
      for part in str(name).split("."):
        layers = getattr(layers, part, None)
        if layers is None:
          break
      if layers is None:
        logger.warning("Layerwise offload: no module list named %s", name)
        continue
      manager = LayerwiseOffloadManager(
          self,
          layers_attr_str=str(name),
          num_layers=len(layers),
          enabled=True,
          pin_cpu_memory=pin_cpu_memory,
          prefetch_size=prefetch_size,
      )
      manager.offload_all_layers()
      manager.register_forward_hooks()
      self.layerwise_offload_managers.append(manager)

  def prepare_for_next_req(self) -> None:
    if not self.layerwise_offload_managers:
      return
    for manager in self.layerwise_offload_managers:
      manager.prepare_for_next_req(non_blocking=True)

  def disable_offload(self) -> None:
    if not self.layerwise_offload_managers:
      return
    for manager in self.layerwise_offload_managers:
      if manager.enabled:
        manager.remove_forward_hooks()
        manager.load_all_layers()

  def enable_offload(self) -> None:
    if not self.layerwise_offload_managers:
      return
    for manager in self.layerwise_offload_managers:
      if manager.enabled:
        manager.register_forward_hooks()
