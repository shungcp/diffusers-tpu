"""The v1 functional namespace: the few names transformers uses.

`InterpolationMode` is re-exported here as the real enum because
transformers uses `tvF.InterpolationMode` in a class-level annotation
(`PILImageResampling | tvF.InterpolationMode | int | None`). Falling
through to `__getattr__` would return a function, `EnumType | function`
raises `TypeError`, and the processor silently comes back as None.
"""
import torchvision
from torchvision.transforms import InterpolationMode  # noqa: F401


def pil_to_tensor(pic):
  """Converts a PIL image (HWC uint8) to a CHW uint8 tensor."""
  import numpy as np
  import torch

  array = np.asarray(pic)
  if array.ndim == 2:
    array = array[:, :, None]
  return torch.from_numpy(np.ascontiguousarray(
      array.transpose((2, 0, 1))))


def resize(img, size, interpolation=InterpolationMode.BILINEAR,
           max_size=None, antialias=True):
  """Resizes with `F.interpolate` (antialiased), like torchvision's tensor path.

  Against torchvision 0.25 on uint8 inputs, bilinear differs by at most 1
  (rounding). Bicubic upsampling differs by up to 18 because torchvision
  clamps overshoot differently; do not rely on it. The MiniMax-H3 image
  processor uses bilinear (the transformers default).
  """
  import torch
  import torch.nn.functional as F

  mode = {
      InterpolationMode.NEAREST: "nearest",
      InterpolationMode.NEAREST_EXACT: "nearest-exact",
      InterpolationMode.BILINEAR: "bilinear",
      InterpolationMode.BICUBIC: "bicubic",
  }.get(interpolation, "bilinear")

  if isinstance(size, int):
    size = [size]
  h, w = img.shape[-2], img.shape[-1]
  if len(size) == 1:
    short, long = (size[0], int(size[0] * max(h, w) / min(h, w)))
    new_h, new_w = (short, long) if h < w else (long, short)
    if max_size is not None and max(new_h, new_w) > max_size:
      scale = max_size / max(new_h, new_w)
      new_h, new_w = int(new_h * scale), int(new_w * scale)
  else:
    new_h, new_w = int(size[0]), int(size[1])

  was_uint8 = img.dtype == torch.uint8
  x = img.unsqueeze(0) if img.dim() == 3 else img
  x = x.float() if was_uint8 else x
  # `antialias` is meaningless for nearest and torch rejects it there.
  kw = {} if mode.startswith("nearest") else {
      "antialias": bool(antialias), "align_corners": False}
  out = F.interpolate(x, size=[new_h, new_w], mode=mode, **kw)
  if was_uint8:
    out = out.round().clamp(0, 255).to(torch.uint8)
  return out.squeeze(0) if img.dim() == 3 else out


def normalize(tensor, mean, std, inplace=False):
  """(x - mean) / std over the channel axis. No torchvision internals."""
  import torch

  if not inplace:
    tensor = tensor.clone()
  dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
  m = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
  s = torch.as_tensor(std, dtype=dtype, device=tensor.device)
  if (s == 0).any():
    raise ValueError(f"std contains a zero: {std}")
  shape = [-1] + [1] * (tensor.dim() - 1 - (tensor.dim() > 3))
  if tensor.dim() == 4:
    m, s = m.view(1, -1, 1, 1), s.view(1, -1, 1, 1)
  else:
    m, s = m.view(*shape), s.view(*shape)
  out = tensor.to(dtype)
  return out.sub_(m).div_(s)


def crop(img, top, left, height, width):
  """Plain slicing, with zero padding when the box runs off the edge,
  which is what torchvision does rather than clipping the box."""
  import torch.nn.functional as F

  h, w = img.shape[-2], img.shape[-1]
  right, bottom = left + width, top + height
  pad = [max(-left, 0), max(right - w, 0), max(-top, 0), max(bottom - h, 0)]
  if any(pad):
    img = F.pad(img, pad)
    top, left = top + pad[2], left + pad[0]
  return img[..., top:top + height, left:left + width]


def pad(img, padding, fill=0, padding_mode="constant"):
  """`padding` is torchvision's 1/2/4-element form, not torch's."""
  import torch.nn.functional as F

  if isinstance(padding, int):
    p = [padding] * 4
  elif len(padding) == 1:
    p = [padding[0]] * 4
  elif len(padding) == 2:
    p = [padding[0], padding[0], padding[1], padding[1]]
  else:
    p = list(padding)
  # torchvision's four-element form is (left, top, right, bottom);
  # torch's `F.pad` wants (left, right, top, bottom).
  torch_pad = [p[0], p[2], p[1], p[3]]
  mode = {"constant": "constant", "edge": "replicate",
          "reflect": "reflect", "symmetric": "reflect"}[padding_mode]
  kw = {"value": fill} if mode == "constant" else {}
  squeeze = img.dim() == 3
  x = img.unsqueeze(0) if squeeze else img
  out = F.pad(x, torch_pad, mode=mode, **kw)
  return out.squeeze(0) if squeeze else out


def to_pil_image(pic, mode=None):
  """Converts a CHW tensor or array to a PIL image."""
  import numpy as np
  from PIL import Image
  import torch

  if isinstance(pic, torch.Tensor):
    pic = pic.detach().cpu().numpy()
  array = np.asarray(pic)
  if array.ndim == 3 and array.shape[0] in (1, 3, 4):
    array = array.transpose((1, 2, 0))
  if array.ndim == 3 and array.shape[2] == 1:
    array = array[:, :, 0]
  if array.dtype != np.uint8:
    array = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
  return Image.fromarray(array, mode=mode)


def __getattr__(name):
  return torchvision._refuse
