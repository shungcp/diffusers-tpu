"""Every name resolves; calling an unimplemented one raises.

`InterpolationMode` is re-exported as the real enum because transformers
uses it in a type annotation, where the function `__getattr__` returns
would fail.
"""
import torchvision
from torchvision.transforms import InterpolationMode  # noqa: F401


# transformers imports this namespace (`from torchvision.transforms.v2
# import functional as tvF`), so keep this list in sync with what v1
# implements.
from torchvision.transforms.functional import (  # noqa: F401
    crop,
    normalize,
    pad,
    pil_to_tensor,
    resize,
    to_pil_image,
)


def __getattr__(name):
  return torchvision._refuse
