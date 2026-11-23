"""How long does a chain of eager all-reduces wait when ranks arrive skewed?

Mirrors the edit request's text tower under torch_tpu: 51 layers, each a
compiled matmul piece followed by an eager `dist.all_reduce`, two per layer,
on one rank's (1, 256, 5120) activations. Before the chain each rank is
delayed by `rank * PROBE_SKEW_MS`, either on the host (sleep) or on the
device (a queue of small eager ops, like the vision tower before the text
tower). With ordinary all-reduce semantics every rank finishes at about
max(delay) + chain time. Prints per-rank finish times, absolute clock,
so the ranks can be lined up.

  PROBE=arskew bash scripts/run_probe_tt_overlap.sh
  PROBE_SKEW_MS (default 40), PROBE_REPS (default 5), PROBE_LAYERS (51).
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
  group = dist.new_group(list(range(world)), backend="tpu_dist")
  dev = torch.device("tpu", rank)
  skew = float(os.environ.get("PROBE_SKEW_MS", 40)) / 1e3
  reps = int(os.environ.get("PROBE_REPS", 5))
  layers = int(os.environ.get("PROBE_LAYERS", 51))
  bf = torch.bfloat16
  D, F = 5120, 25600 // world
  x = torch.randn(1, 256, D, dtype=bf, device=dev)
  wa = torch.randn(D, D // world, dtype=bf, device=dev) * 0.02
  wb = torch.randn(D // world, D, dtype=bf, device=dev) * 0.02
  wc = torch.randn(D, F, dtype=bf, device=dev) * 0.02
  wd = torch.randn(F, D, dtype=bf, device=dev) * 0.02
  attn = torch.compile(lambda h: (h @ wa) @ wb, backend="tpu", dynamic=False)
  mlp = torch.compile(lambda h: torch.nn.functional.silu(h @ wc) @ wd, backend="tpu", dynamic=False)
  small = torch.randn(1024, 128, dtype=bf, device=dev)

  def chain(h):
    for _ in range(layers):
      a = attn(h)
      dist.all_reduce(a, group=group)
      h = h + a
      m = mlp(h)
      dist.all_reduce(m, group=group)
      h = h + m
    return h

  def device_delay(sec):
    # ~0.1 ms per tiny eager op on v6e; queue enough of them to cover `sec`.
    t = small
    for _ in range(int(sec / 1e-4)):
      t = t * 1.0001
    return t

  def one(mode):
    dist.barrier()
    torch.accelerator.synchronize()
    dist.barrier()
    t0 = time.time()
    if mode == "host":
      time.sleep(rank * skew)
    elif mode == "device":
      device_delay(rank * skew)
    t_launch = time.time()
    chain(x)
    t_issued = time.time()
    torch.accelerator.synchronize()
    t_done = time.time()
    return t0, t_launch, t_issued, t_done

  for mode in ("warm", "none", "host", "device"):
    for r in range(reps if mode != "warm" else 2):
      t0, tl, ti, td = one("none" if mode == "warm" else mode)
      if mode == "warm":
        continue
      rows = [None] * world
      dist.all_gather_object(rows, (rank, t0, tl, ti, td))
      if rank == 0:
        base = min(row[1] for row in rows)
        last_launch = max(row[2] for row in rows) - base
        fin = " ".join(f"r{row[0]}:{(row[4] - base) * 1e3:6.1f}" for row in rows)
        print(f"[arskew] {mode:<6} rep {r}: last launch {last_launch * 1e3:6.1f} ms | done {fin}", flush=True)
  dist.barrier()


if __name__ == "__main__":
  main()
