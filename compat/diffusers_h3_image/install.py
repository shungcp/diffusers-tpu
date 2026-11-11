"""Register `MiniMax-H3-Image` modules into an older `diffusers` (< 875373ac).

When the installed `diffusers` already includes `MiniMaxH3ImageTransformer2DModel`
(`>= 875373ac`), `install()` is a no-op. Otherwise it loads the upstream
`875373ac` files from this folder directly under the `diffusers.*` namespace
so relative imports resolve against the installed `diffusers` package.
"""

import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_into(full_name: str, filepath: str, is_pkg: bool = False):
  if full_name in sys.modules:
    return sys.modules[full_name]
  if is_pkg:
    spec = importlib.util.spec_from_file_location(
        full_name, filepath, submodule_search_locations=[os.path.dirname(filepath)]
    )
  else:
    spec = importlib.util.spec_from_file_location(full_name, filepath)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[full_name] = mod
  parent_name, _, attr = full_name.rpartition(".")
  if parent_name and parent_name in sys.modules:
    setattr(sys.modules[parent_name], attr, mod)
  spec.loader.exec_module(mod)
  return mod


def install() -> bool:
  import diffusers
  import diffusers.models
  import diffusers.models.autoencoders
  import diffusers.models.transformers
  import diffusers.modular_pipelines
  import diffusers.schedulers

  if hasattr(diffusers, "MiniMaxH3ImageTransformer2DModel"):
    return False

  # 1. Updated scheduler (`euler_update_form="velocity"`)
  sched_name = "diffusers.schedulers.scheduling_minimax_h3"
  sys.modules.pop(sched_name, None)
  sched_mod = _load_into(sched_name, os.path.join(_HERE, "scheduling_minimax_h3.py"))
  diffusers.MiniMaxH3Scheduler = sched_mod.MiniMaxH3Scheduler
  diffusers.schedulers.MiniMaxH3Scheduler = sched_mod.MiniMaxH3Scheduler

  # 2. Transformer2DModel
  tr_name = "diffusers.models.transformers.transformer_minimax_h3_image"
  tr_mod = _load_into(tr_name, os.path.join(_HERE, "transformer_minimax_h3_image.py"))
  for cls_name in ("MiniMaxH3ImageTransformer2DModel",):
    cls = getattr(tr_mod, cls_name)
    setattr(diffusers, cls_name, cls)
    setattr(diffusers.models, cls_name, cls)
    setattr(diffusers.models.transformers, cls_name, cls)

  # 3. AutoencoderKLMiniMaxH3Image
  vae_name = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image"
  vae_mod = _load_into(vae_name, os.path.join(_HERE, "autoencoder_kl_minimax_h3_image.py"))
  for cls_name in ("AutoencoderKLMiniMaxH3Image",):
    cls = getattr(vae_mod, cls_name)
    setattr(diffusers, cls_name, cls)
    setattr(diffusers.models, cls_name, cls)
    setattr(diffusers.models.autoencoders, cls_name, cls)

  # 4. Modular pipeline package `diffusers.modular_pipelines.minimax_h3_image`
  pkg_dir = os.path.join(_HERE, "modular_pipelines", "minimax_h3_image")
  pkg_name = "diffusers.modular_pipelines.minimax_h3_image"
  for sub in (
      "modular_pipeline",
      "before_encoder",
      "encoders",
      "before_denoise",
      "denoise",
      "decoders",
      "modular_blocks_minimax_h3_image",
  ):
    # Ensure package stub exists before loading submodules
    if pkg_name not in sys.modules:
      pkg = types.ModuleType(pkg_name)
      pkg.__path__ = [pkg_dir]
      pkg.__package__ = pkg_name
      sys.modules[pkg_name] = pkg
      setattr(diffusers.modular_pipelines, "minimax_h3_image", pkg)
    _load_into(f"{pkg_name}.{sub}", os.path.join(pkg_dir, f"{sub}.py"))
  sys.modules.pop(pkg_name, None)
  mp_mod = _load_into(pkg_name, os.path.join(pkg_dir, "__init__.py"), is_pkg=True)
  for cls_name in ("MiniMaxH3ImageModularPipeline", "MiniMaxH3ImageBlocks"):
    if hasattr(mp_mod, cls_name):
      cls = getattr(mp_mod, cls_name)
      setattr(diffusers, cls_name, cls)
      setattr(diffusers.modular_pipelines, cls_name, cls)

  try:
    from diffusers.modular_pipelines import components_manager
    components_manager.SUPPORTED_MODULAR_TYPES["MiniMaxH3ImageModularPipeline"] = (
        "minimax_h3_image",
        "MiniMaxH3ImageBlocks",
    )
  except Exception:
    pass
  return True
