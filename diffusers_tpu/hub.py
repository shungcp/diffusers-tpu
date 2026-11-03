"""Locating the model in the local Hugging Face cache, without network access."""
import os


def local_snapshot(repo="MiniMaxAI/MiniMax-H3"):
  """Directory of the cached snapshot of `repo` (honours HF_HOME / HF_HUB_CACHE).

  Never touches the network: with several processes starting at once, a hub
  lookup per process gets rate limited. A partial download (only the
  components a workflow needs) is accepted; missing files fail later, when
  they are loaded.
  """
  from huggingface_hub import constants

  root = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
  ref = os.path.join(root, "refs", "main")
  snaps = os.path.join(root, "snapshots")
  if os.path.isfile(ref):
    with open(ref) as fh:
      path = os.path.join(snaps, fh.read().strip())
    if os.path.isdir(path):
      return path
  if os.path.isdir(snaps) and os.listdir(snaps):
    return os.path.join(snaps, sorted(os.listdir(snaps))[-1])
  raise FileNotFoundError(
      f"no local snapshot of {repo} under {constants.HF_HUB_CACHE}; run "
      f"`huggingface-cli download {repo}` first (HF_HOME chooses the cache location)")
