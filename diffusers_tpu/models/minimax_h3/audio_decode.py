"""MiniMax-H3's audio VAE decoder on the chips, for the JAX backend.

The JAX pipeline denoises the audio rows with the video but had no audio
decoder, so its clips were silent. This runs diffusers'
`AutoencoderKLMiniMaxH3Audio.decode` through torchax as one jitted
program: 0.39 s for 15 s of stereo on two v6e chips, against 5.4 s on the
host CPU.
"""

import os


class AudioDecoder:
  """Rows `(2 * T, 32)` (channel-major, as packed) -> waveform `(2, samples)`."""

  def __init__(self, snapshot, precision="high", on_cpu=False):
    """`on_cpu`: decode with PyTorch on the host, in a background thread.
    For one 16 GB chip, which cannot hold the video decoder and this
    program's 1.8 GB of scratch at once; the host's ~2 s for a 6 s clip
    hides behind the chip's video decode."""
    import jax
    import numpy as np
    import torch
    import torchax
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from torch.func import functional_call

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        os.path.join(snapshot, "audio_vae"), torch_dtype=torch.float32).eval()
    self.sampling_rate = int(vae.config.sampling_rate)
    self.mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1)
    self.std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
    self.channels = int(vae.config.latent_channels)

    class _Decode(torch.nn.Module):
      def __init__(self, v):
        super().__init__()
        self.v = v

      def forward(self, z):
        return self.v.decode(z, return_dict=False)[0]

    module = _Decode(vae)
    self._cpu = None
    if on_cpu:
      import concurrent.futures
      self._cpu = (module, concurrent.futures.ThreadPoolExecutor(1))
      return
    env = torchax.default_env()
    # The two stereo channels are independent batch items: one per chip
    # when there are two (0.53 -> 0.37 s on v6e).
    devs = jax.devices()[:2]
    self._mesh = mesh = jax.sharding.Mesh(np.array(devs), ("b",))
    self._z_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(
        "b" if len(devs) == 2 else None))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self._params = {k: jax.device_put(v.detach().numpy(), rep)
                    for k, v in list(module.named_parameters())
                    + list(module.named_buffers())}

    def run(params, z):
      with env, jax.default_matmul_precision(precision):
        out = functional_call(module, {k: env.j2t_iso(v) for k, v in params.items()},
                              args=(env.j2t_iso(z),), strict=False)
        return env.t2j_iso(out)

    self._run = jax.jit(run)

  def start(self, rows):
    """Dispatch the decode of `rows` (numpy `(2 * T, channels)`) without
    waiting, so it can queue behind or beside other device work."""
    import jax
    import numpy as np
    import torch

    t = rows.shape[0] // 2
    z = torch.from_numpy(np.array(rows, np.float32).reshape(2, t, self.channels))
    z = z.permute(0, 2, 1) * self.std + self.mean
    if self._cpu is not None:
      module, pool = self._cpu

      def run_cpu():
        with torch.no_grad():
          return module(z).numpy()
      return pool.submit(run_cpu)
    # Its own mesh: the pipeline's context mesh spans every chip.
    with jax.set_mesh(self._mesh):
      return self._run(self._params, jax.device_put(z.numpy(), self._z_sharding))

  def finish(self, out):
    """Float32 numpy `(2, samples)` from what `start` returned."""
    import jax
    import numpy as np
    if self._cpu is not None:
      out = out.result()
    return np.asarray(jax.device_get(out), np.float32)[:, 0]

  def decode(self, rows):
    """`rows`: numpy `(2 * T, channels)`; returns float32 numpy `(2, samples)`."""
    return self.finish(self.start(rows))
