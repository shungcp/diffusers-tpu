"""Names transformers and diffusers reach for at import time."""
import enum

import torchvision


class InterpolationMode(str, enum.Enum):
  NEAREST = "nearest"
  NEAREST_EXACT = "nearest-exact"
  BILINEAR = "bilinear"
  BICUBIC = "bicubic"
  BOX = "box"
  HAMMING = "hamming"
  LANCZOS = "lanczos"


def __getattr__(name):
  return torchvision._refuse
