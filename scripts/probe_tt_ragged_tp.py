"""Tensor-parallel all-reduce rebuilt from torch_tpu's async ragged_all_to_all.

`ragged_all_to_all` compiles to an async start/done pair that overlaps a
matmul (probe_tt_permute.py); `all_reduce` is synchronous. This times one
row-parallel projection of MiniMax-H3 -- (N, F/W) @ (F/W, D), then a sum over
ranks -- three ways, at the shapes of one rank at 768x1344/345 frames:

  ar        matmul, then all_reduce (what the model does today)
  a2a       matmul, then reduce-scatter and all-gather, both as one
            ragged_all_to_all each (+ a local sum)
  a2a xC    the same in C sequence chunks in one graph, so chunk c's
            traffic can run under chunk c+1's matmul

  run_probe_tt_overlap.sh with PROBE=ragged_tp; PROBE_CHUNKS="2 4 8".
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
  from diffusers_tpu.backends.torchtpu_h3 import _auto_env
  _auto_env()
  import torch
  import torch.distributed as dist
  import torch_tpu  # noqa: F401

  world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
  dist.init_process_group(backend="tpu_dist", init_method="env://",
                          world_size=world, rank=rank)
  pg = dist.new_group(list(range(world)), backend="tpu_dist")
  group = pg.group_name
  dev = torch.device("tpu", rank)
  chunks = [int(c) for c in os.environ.get("PROBE_CHUNKS", "2 4 8").split()]
  N = int(os.environ.get("PROBE_N", 104040))
  N -= N % (world * max(chunks))
  D, F = 5376, 14336 // world
  bf = torch.bfloat16
  x = torch.randn(N, F, dtype=bf, device=dev)
  w = torch.randn(F, D, dtype=bf, device=dev) * 0.01
  c10d = torch.ops._c10d_functional
  safe = torch.tpu.distributed.spmd_safe
  ops = torch.tpu.ops
  i32 = dict(dtype=torch.int32, device=dev)
  ar_ = torch.arange(world, **i32)

  def sizes(rows):
    """Offsets for a reduce-scatter and an all-gather of `rows * world` rows."""
    full = torch.full((world,), rows, **i32)
    zero = torch.zeros(world, **i32)
    mine = torch.full((world,), rank * rows, **i32)
    return dict(rs=(ar_ * rows, full, mine, full), ag=(zero, full, mine, full))

  plans = {n // world: sizes(n // world) for n in [N] + [N // c for c in chunks]}

  def rs_ag(y):
    rows = y.shape[0] // world
    p = plans[rows]
    got = ops.ragged_all_to_all(y, torch.empty_like(y), *p["rs"], group)
    part = got.view(world, rows, D).sum(0)            # this rank's rows, summed
    return ops.ragged_all_to_all(part, torch.empty_like(y), *p["ag"], group)

  def f_mm(x, w):
    return x @ w

  def f_ar(x, w):
    return c10d.wait_tensor(c10d.all_reduce(x @ w, "sum", group))

  def f_a2a(x, w):
    return rs_ag(x @ w)

  def f_chunked(c):
    def f(x, w):
      return torch.cat([rs_ag(xc @ w) for xc in x.chunk(c)], 0)
    return f

  reps = int(os.environ.get("PROBE_REPS", 10))
  ref = None

  def timed(name, fn):
    nonlocal ref
    f = torch.compile(safe(fn), backend="tpu", fullgraph=True)
    try:
      with torch.no_grad():
        for _ in range(2):
          out = f(x, w)
        torch.accelerator.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
          out = f(x, w)
        torch.accelerator.synchronize()
        ms = (time.perf_counter() - t) / reps * 1e3
        if ref is None and name.startswith("ar"):
          ref = out.float()
        err = "" if ref is None or name.startswith(("mm", "ar")) else (
            f"  max|diff| vs ar {float((out.float() - ref).abs().max()):.3g}")
      msg = f"{ms:8.2f} ms{err}"
    except Exception as e:
      lines = [l for l in str(e).splitlines() if l.strip()]
      msg = "FAILED " + " | ".join(lines[:4])[:500]
    if rank == 0:
      print(f"  {name:<28} {msg}", flush=True)

  if rank == 0:
    print(f"world {world}, x ({N}, {F}) @ w ({F}, {D})", flush=True)
  timed("mm alone", f_mm)
  timed("ar: mm + all_reduce", f_ar)
  timed("a2a: mm + RS + AG", f_a2a)
  for c in chunks:
    timed(f"a2a x{c} chunks", f_chunked(c))
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
