"""One module per execution backend.

`torchax_qwen21` traces the PyTorch module; `torchtpu_qwen21` runs it
eagerly. The native JAX path is a reimplementation and lives under
`models/`.
"""
