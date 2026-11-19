"""Compute Qwen3-VL index metadata on the host instead of the TPU.

Rope indices, vision interpolation indices / position ids / attention seqlens
and the deepstack scatter positions depend only on token ids and the image
grid. Upstream computes them from device tensors with `.tolist()`, boolean
masks and Python loops, so every new prompt length or reference size compiles
and syncs dozens of tiny data-dependent programs on the TPU (c725: ~15 of the
24 s text stage). Here they run on CPU copies and only the results are moved.
The default parts are integer math, so results are identical (c727).
`DT_VL_HOST_META=0` disables; a comma list of vis,visinterp,rope,deep picks
parts (default vis,rope,deep; visinterp changes float weights, see below).
"""
import os
import torch


def _cpu(x):
  return x.cpu() if isinstance(x, torch.Tensor) and x.device.type != "cpu" else x


def _to(x, dev):
  if isinstance(x, torch.Tensor):
    return x.to(dev)
  if isinstance(x, (tuple, list)):
    return type(x)(_to(v, dev) for v in x)
  return x


def _dev_of(*xs):
  for x in xs:
    if isinstance(x, torch.Tensor) and x.device.type != "cpu":
      return x.device
  return None


def install():
  v = os.environ.get("DT_VL_HOST_META", "1")
  if v == "0":
    return False
  parts = {"vis", "rope", "deep"} if v == "1" else set(v.split(","))
  from transformers.models.qwen3_vl import modeling_qwen3_vl as m

  def host(fn):
    def wrapped(grid_thw, *a, **k):
      dev = _dev_of(grid_thw)
      out = fn(_cpu(grid_thw), *a, **k)
      return _to(out, dev) if dev is not None else out
    return wrapped

  # Interpolation weights are float: CPU and TPU float32 differ in the last
  # bits and the output changes (c727), so only `visinterp` moves them.
  names = []
  if "vis" in parts:
    names += ["get_vision_position_ids", "get_vision_attention_seqlens"]
  if "visinterp" in parts:
    names.append("get_vision_interpolation_indices_and_weights")
  for name in names:
    setattr(m, name, host(getattr(m, name)))

  if "rope" in parts:
    _install_rope(m)
  if "deep" in parts:
    _install_deep(m)
  return True


def _install_rope(m):
  orig_rope = m.Qwen3VLModel.get_rope_index

  def get_rope_index(self, input_ids=None, image_grid_thw=None,
                     video_grid_thw=None, attention_mask=None, **k):
    dev = _dev_of(input_ids, attention_mask)
    k = {n: _cpu(v) for n, v in k.items()}
    pos, deltas = orig_rope(self, _cpu(input_ids), image_grid_thw=_cpu(image_grid_thw),
                            video_grid_thw=_cpu(video_grid_thw),
                            attention_mask=_cpu(attention_mask), **k)
    return (_to(pos, dev), _to(deltas, dev)) if dev is not None else (pos, deltas)
  m.Qwen3VLModel.get_rope_index = get_rope_index


def _install_deep(m):
  cache = {}

  def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    # Same as upstream (clone, then add visual_embeds at the masked rows), with
    # the row indices taken from a host copy of the mask once per forward.
    key = id(visual_pos_masks)
    if cache.get("key") != key:
      idx = _cpu(visual_pos_masks).reshape(-1).nonzero().squeeze(1)
      cache.update(key=key, idx=idx.to(hidden_states.device), mask=visual_pos_masks)
    idx = cache["idx"]
    shp = hidden_states.shape
    flat = hidden_states.reshape(-1, shp[-1]).clone()
    rows = flat.index_select(0, idx) + visual_embeds.to(flat.device, flat.dtype)
    flat.index_copy_(0, idx, rows)
    return flat.reshape(shp)
  m.Qwen3VLTextModel._deepstack_process = _deepstack_process
