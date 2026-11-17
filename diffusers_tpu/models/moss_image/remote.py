"""The upstream MOSS-Image modeling code, imported from the cached snapshot.

MOSS-Image is not in diffusers: its transformer, scheduler and pipeline live
in one file, ``pipeline_moss_image.py``, shipped in the Hub repository and
loaded with ``trust_remote_code``. Importing that file by path gives the same
classes without going through ``DiffusionPipeline.from_pretrained``'s
custom-pipeline machinery, which would look the repo up on the network.
"""
import importlib.util
import os
import sys

from diffusers_tpu.hub import local_snapshot

DEFAULT_REPO = "OpenMOSS-Team/MOSS-Image-Base"


def snapshot(repo=DEFAULT_REPO):
  return local_snapshot(repo)


def component(repo, name):
  """``<snapshot>/<name>``; components the repo does not ship (MOSS-Image-SFT
  has only ``transformer/`` and ``scheduler/``) come from MOSS-Image-Base,
  whose text encoder, tokenizer and VAE are the same files."""
  path = os.path.join(snapshot(repo), name)
  if os.path.isdir(path):
    return path
  return os.path.join(snapshot(DEFAULT_REPO), name)


def load_module(repo=DEFAULT_REPO):
  """Imports ``pipeline_moss_image.py`` from the snapshot (once per process)."""
  name = "moss_image_remote"
  if name in sys.modules:
    return sys.modules[name]
  path = os.path.join(snapshot(repo), "pipeline_moss_image.py")
  if not os.path.isfile(path):  # a partial download; Base ships the same file
    path = os.path.join(snapshot(DEFAULT_REPO), "pipeline_moss_image.py")
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module
