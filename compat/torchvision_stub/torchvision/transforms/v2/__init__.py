from ... import _refuse
from . import functional  # noqa: F401


class Compose:
  def __init__(self, *a, **k):
    pass

  __call__ = _refuse
