"""Convert saved frame arrays (uint8 or [0, 1] float, shape (F, H, W, 3), `.npy`) to H.264 mp4.

  python scripts/frames_to_mp4.py OUT_DIR out/foo_frames.npy [more.npy ...]
"""
import os
import sys

import av
import numpy as np


def write(frames, path, fps=24):
  with av.open(path, "w") as out:
    stream = out.add_stream("libx264", rate=fps)
    stream.height, stream.width = frames.shape[1], frames.shape[2]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    for f in frames:
      for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
        out.mux(packet)
    for packet in stream.encode():
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
    write(frames, path)
    print(f"{path}  {frames.shape}  {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
  main()
