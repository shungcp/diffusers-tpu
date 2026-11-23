"""Can torch_tpu move data to a neighbour while a matmul runs?

On TPU only collective-permute overlaps a matmul -- all-reduce, all-to-all
and all-gather run serially even in JAX (see probe_tt_overlap.py). JAX hides
tensor-parallel traffic by building collective matmuls out of
collective-permutes. This checks the two ways torch_tpu offers to send rows to
rank + 1: `torch.tpu.ops.ragged_all_to_all` inside a compiled graph, and the
eager `torch.tpu.ops.send` / `recv` beside a compiled matmul.

  torchrun --nproc_per_node 8 scripts/probe_tt_permute.py   (or run_probe_tt_overlap.sh with PROBE=permute)
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
  dev = torch.device("tpu", rank)
  N = int(os.environ.get("PROBE_N", 104040)); N -= N % world
  D, F, R = 5376, 14336 // world, N // world
  bf = torch.bfloat16
  x = torch.randn(1, N, D, dtype=bf, device=dev)
  w1 = torch.randn(D, F, dtype=bf, device=dev) * 0.01
  w2 = torch.randn(F, D, dtype=bf, device=dev) * 0.01
  chunk = torch.randn(R, D, dtype=bf, device=dev)
  recv_buf = torch.zeros(R, D, dtype=bf, device=dev)
  nxt, prv = (rank + 1) % world, (rank - 1) % world
  i64 = dict(dtype=torch.int32, device=dev)
  in_off = torch.zeros(world, **i64)
  send_sz = torch.zeros(world, **i64); send_sz[nxt] = R
  out_off = torch.zeros(world, **i64)
  recv_sz = torch.zeros(world, **i64); recv_sz[prv] = R
  safe = torch.tpu.distributed.spmd_safe
  ops = torch.tpu.ops
  reps = int(os.environ.get("PROBE_REPS", 10))
  r = {}

  def timed(name, fn):
    try:
      with torch.no_grad():
        for _ in range(2):
          out = fn()
        torch.accelerator.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
          out = fn()
        torch.accelerator.synchronize()
      del out
      r[name] = (time.perf_counter() - t) / reps * 1e3
      msg = f"{r[name]:8.2f} ms"
    except Exception as e:  # report and go on: which paths exist is the question
      lines = [l for l in str(e).splitlines() if l.strip()]
      msg = f"FAILED {type(e).__name__}: " + " | ".join(lines[:6])[:600]
    if rank == 0:
      print(f"  {name:<36} {msg}", flush=True)

  def compiled(fn):
    f = torch.compile(safe(fn), backend="tpu", fullgraph=True)
    return lambda: f(x, w1, w2, chunk)

  def mm(x, w1, w2, c):
    return (x @ w1) @ w2

  def ragged(x, w1, w2, c):
    return ops.ragged_all_to_all(c, recv_buf, in_off, send_sz, out_off, recv_sz, pg.group_name)

  def both(x, w1, w2, c):
    return mm(x, w1, w2, c), ragged(x, w1, w2, c)

  if rank == 0:
    print(f"world {world}, N {N}, rows sent to rank+1: {R} x {D}", flush=True)
  timed("compute: 2 matmuls (compiled)", compiled(mm))
  timed("ragged_all_to_all to rank+1 (compiled)", compiled(ragged))
  timed("both in one graph", compiled(both))
  if rank == 0 and "ragged_all_to_all to rank+1 (compiled)" in r and "both in one graph" in r:
    a, b, c = r["compute: 2 matmuls (compiled)"], r["ragged_all_to_all to rank+1 (compiled)"], r["both in one graph"]
    print(f"  -> ragged: {c:.2f} vs sum {a + b:.2f} / max {max(a, b):.2f}: "
          f"{'overlapped' if c < a + b - 0.3 * min(a, b) else 'serial'}", flush=True)

  if os.environ.get("PROBE_SKIP_P2P") == "1":
    dist.destroy_process_group()
    return
  m = compiled(mm)

  def p2p():
    works = ([ops.send([chunk], nxt, 7), ops.recv([recv_buf], prv, 7)] if rank % 2 == 0
             else [ops.recv([recv_buf], prv, 7), ops.send([chunk], nxt, 7)])
    for w in works:
      w.wait()
    return recv_buf

  def p2p_beside_mm():
    works = ([ops.send([chunk], nxt, 7), ops.recv([recv_buf], prv, 7)] if rank % 2 == 0
             else [ops.recv([recv_buf], prv, 7), ops.send([chunk], nxt, 7)])
    out = m()
    for w in works:
      w.wait()
    return out

  timed("eager send/recv to rank+1", p2p)
  timed("eager send/recv beside compiled mm", p2p_beside_mm)
  if rank == 0 and "eager send/recv to rank+1" in r and "eager send/recv beside compiled mm" in r:
    a, b, c = r["compute: 2 matmuls (compiled)"], r["eager send/recv to rank+1"], r["eager send/recv beside compiled mm"]
    print(f"  -> send/recv: {c:.2f} vs sum {a + b:.2f} / max {max(a, b):.2f}: "
          f"{'overlapped' if c < a + b - 0.3 * min(a, b) else 'serial'}", flush=True)
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
