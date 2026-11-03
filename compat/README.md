# compat

Workarounds for environments upstream does not support, kept out of the
main package.

## `torchvision_stub`

`transformers` requires torchvision to build the Qwen3-VL processor used by
MiniMax-H3, because that processor includes a video processor. The model
does not use it.

The torch_tpu environment uses torch 2.13, and no torchvision release
matches it: the PyPI build imports but its operators fail to register
(`operator torchvision::nms does not exist`), and the matching nightly
pulls in a newer torch that torch_tpu is not built for.

The stub provides the names imported at load time (`InterpolationMode`,
`transforms.functional`, `transforms.v2.functional`, `io`). A few image
functions the processor actually calls (`pil_to_tensor`, `resize`,
`normalize`, `crop`, `pad`, `to_pil_image`) are implemented with plain
torch; everything else raises if called. Put it first on `PYTHONPATH` for
the torch_tpu backend only; the other backends use the real torchvision.

Remove it once a matching torchvision is available.
