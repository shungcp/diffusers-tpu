from .. import _refuse


def __getattr__(name):
  return _refuse
