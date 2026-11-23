"""Can torch_tpu overlap a collective with compute? A model-free probe.

Shapes are one rank's share of MiniMax-H3 at 768x1344/345 frames under
8-way tensor parallelism: activations (1, N, 5376) bf16, N = 104,040.
Each case is timed alone and the overlap cases against the sum of their
parts: if `together` is close to `max(compute, comm)` the two overlapped,
if close to `compute + comm` they ran one after the other.

  compiled  one graph (`torch.compile`, `spmd_safe`, so the collective is
            fused into the graph instead of splitting it):
              indep   matmuls + an all-reduce of an unrelated tensor
              dep     all_reduce(x @ W1 @ W2) @ W1 @ W2 (H3's shape)
  eager     `dist.all_reduce(async_op=True)`, a compiled matmul program
            dispatched while it is in flight, then `work.wait()`
  comm      per-block traffic of tensor parallelism (two all-reduces of
            (1, N, D)) against sequence parallelism (four all-to-alls of
            (N/W, D)), for the Ulysses option of the plan

  torchrun --nproc_per_node 8 scripts/probe_tt_overlap.py
  PROBE_N, PROBE_REPS override the sequence length and repetitions.
  For HLO: LIBTPU_INIT_ARGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text".
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

  world = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["RANK"])
  dist.init_process_group(backend="tpu_dist", init_method="env://",
                          world_size=world, rank=rank)
  pg = dist.new_group(list(range(world)), backend="tpu_dist")
  group = pg.group_name
  dev = torch.device("tpu", rank)

  N = int(os.environ.get("PROBE_N", 104040))
  N -= N % (world * world)
  D = 5376
  F = 14336 // world
  bf = torch.bfloat16
  x = torch.randn(1, N, D, dtype=bf, device=dev)
  w1 = torch.randn(D, F, dtype=bf, device=dev) * 0.01
  w2 = torch.randn(F, D, dtype=bf, device=dev) * 0.01
  other = torch.randn(1, N, D, dtype=bf, device=dev)
  shard = torch.randn(N // world, D, dtype=bf, device=dev)

  c10d = torch.ops._c10d_functional
  wait = c10d.wait_tensor
  safe = torch.tpu.distributed.spmd_safe

  def ar(t):
    return wait(c10d.all_reduce(t, "sum", group))

  splits = [N // world // world] * world

  def a2a(t):
    return wait(c10d.all_to_all_single(t, splits, splits, group))

  def only_mm(x, w1, w2, other):
    return (x @ w1) @ w2, other

  def only_ar(x, w1, w2, other):
    return x, ar(other)

  def indep(x, w1, w2, other):
    return (x @ w1) @ w2, ar(other)

  def mm4(x, w1, w2, other):
    return (((x @ w1) @ w2) @ w1) @ w2, other

  def dep(x, w1, w2, other):
    return (ar((x @ w1) @ w2) @ w1) @ w2, other

  def tp_block_comm(x, w1, w2, other):
    return ar(x), ar(other)

  def sp_block_comm(x, w1, w2, other):
    s = shard
    return a2a(s), a2a(s + 1), a2a(s + 2), a2a(s + 3)

  reps = int(os.environ.get("PROBE_REPS", 10))
  args = (x, w1, w2, other)
  results = {}

  def timed(name, fn):
    with torch.no_grad():
      for _ in range(2):
        out = fn()
      torch.accelerator.synchronize()
      t = time.perf_counter()
      for _ in range(reps):
        out = fn()
      torch.accelerator.synchronize()
    del out
    results[name] = (time.perf_counter() - t) / reps * 1e3
    if rank == 0:
      print(f"  {name:<34} {results[name]:8.2f} ms", flush=True)

  def compiled(fn):
    f = torch.compile(safe(fn), backend="tpu", fullgraph=True)
    return lambda: f(*args)

  if rank == 0:
    print(f"world {world}, N {N}, D {D}, F {F}, "
          f"topology {os.environ.get('TORCH_TPU_TOPOLOGY')}", flush=True)
    print("compiled (one graph):", flush=True)
  for name, fn in [("compute: 2 matmuls", only_mm), ("comm: all-reduce", only_ar),
                   ("indep: together", indep), ("compute: 4 matmuls", mm4),
                   ("dep: together", dep)]:
    timed(name, compiled(fn))

  if rank == 0:
    print("eager async all-reduce beside a compiled matmul program:", flush=True)
  mm = compiled(only_mm)

  def eager_ar():
    dist.all_reduce(other, group=pg)

  def eager_overlap():
    work = dist.all_reduce(other, group=pg, async_op=True)
    out = mm()
    work.wait()
    return out

  timed("comm: eager all-reduce", eager_ar)
  timed("eager: together", eager_overlap)

  if rank == 0:
    print("per-block traffic, tensor vs sequence parallel:", flush=True)
  timed("TP: 2 all-reduce (1,N,D)", compiled(tp_block_comm))
  timed("SP: 4 all-to-all (N/W,D)", compiled(sp_block_comm))

  if rank == 0:
    r = results
    def verdict(together, a, b):
      s, m = r[a] + r[b], max(r[a], r[b])
      return f"{r[together]:.2f} ms vs sum {s:.2f} / max {m:.2f}: " + (
          "overlapped" if r[together] < s - 0.3 * (s - m) else "serial")
    print("verdict:", flush=True)
    print("  indep  " + verdict("indep: together", "compute: 2 matmuls",
                                "comm: all-reduce"), flush=True)
    print("  eager  " + verdict("eager: together", "compute: 2 matmuls",
                                "comm: eager all-reduce"), flush=True)
    print(f"  SP/TP traffic time {r['SP: 4 all-to-all (N/W,D)'] / r['TP: 2 all-reduce (1,N,D)']:.2f}",
          flush=True)
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
