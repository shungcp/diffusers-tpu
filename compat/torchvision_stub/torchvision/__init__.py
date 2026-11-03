"""A minimal torchvision that exists only to satisfy imports.

The torch_tpu environment uses torch 2.13, and no torchvision release
matches it: the real package imports but its operators fail to register
(`operator torchvision::nms does not exist`).

transformers requires torchvision to construct `Qwen3VLProcessor`, because
that processor includes a video processor, which MiniMax-H3 does not use.
Unimplemented names raise only when called.

Used on `PYTHONPATH` for the torch_tpu backend only.
"""

__version__ = "0.0.0+stub"


def _refuse(*_args, **_kwargs):
  raise RuntimeError(
      "diffusers-tpu's torchvision stub was called. It exists so that "
      "transformers can build a processor it does not use; nothing here "
      "can actually transform an image or a video. Install a torchvision "
      "matching this environment's torch if you need one."
  )
