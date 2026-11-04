"""Convert saved frame arrays (uint8 or [0, 1] float, shape (F, H, W, 3), `.npy`) to H.264 mp4.

A `<name>_audio.npy` next to `<name>_frames.npy` (float32 (2, samples), 32 kHz)
is muxed in as an AAC track.

  python scripts/frames_to_mp4.py OUT_DIR out/foo_frames.npy [more.npy ...]
"""
import os
import sys

import av
import numpy as np


def write(frames, path, fps=24, wave=None, rate=32000):
  with av.open(path, "w") as out:
    stream = out.add_stream("libx264", rate=fps)
    stream.height, stream.width = frames.shape[1], frames.shape[2]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    if wave is not None:
      astream = out.add_stream("aac", rate=rate, layout="stereo")
    for f in frames:
      for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
        out.mux(packet)
    for packet in stream.encode():
      out.mux(packet)
    if wave is not None:
      wave = np.ascontiguousarray(np.clip(wave, -1, 1), np.float32)
      for i in range(0, wave.shape[1], 1024):
        af = av.AudioFrame.from_ndarray(wave[:, i:i + 1024], format="fltp", layout="stereo")
        af.sample_rate = rate
        for packet in astream.encode(af):
          out.mux(packet)
      for packet in astream.encode():
        out.mux(packet)


def main():
  dst = sys.argv[1]
  os.makedirs(dst, exist_ok=True)
  for src in sys.argv[2:]:
    name = os.path.basename(src).replace("_frames.npy", "").replace(".npy", "")
    frames = np.load(src)
    if frames.dtype != np.uint8:
      frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    path = os.path.join(dst, name + ".mp4")
    apath = src.replace("_frames.npy", "_audio.npy")
    wave = np.load(apath) if apath != src and os.path.exists(apath) else None
    write(frames, path, wave=wave)
    print(f"{path}  {frames.shape}  audio={'yes' if wave is not None else 'no'}  {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
  main()
