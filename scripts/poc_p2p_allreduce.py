#!/usr/bin/env python
"""POC: RCCL P2P allreduce correctness + bandwidth at 27B gradient sizes.

Why this exists: the known gfx1100 IPC failure mode (ROCm#6565) is *silent*
data corruption in the P2P transport. A 27B QLoRA run allreduces ~233 MB of
bf16 gradients per step (116,727,808 trainable params, in 25 MB DDP buckets),
10x larger than the toy model's payload. This script stresses exactly those
sizes and verifies every result against an **exact integer reference**: all
values are small integers, exactly representable in bf16, so the allreduce sum
must match bit-for-bit — any corruption shows up immediately as a mismatch,
not as a subtle numerical difference.

Usage (two ranks shown, P2P enabled exactly like the launcher does it):

    HIP_VISIBLE_DEVICES=0,1 HSA_ENABLE_IPC_MODE_LEGACY=0 \
        .venv/bin/torchrun --standalone --nproc_per_node=2 \
        scripts/poc_p2p_allreduce.py --iters 30

Exit code 0 = every payload matched on every iteration. Set
NCCL_P2P_DISABLE=1 to run the same check over the shared-memory fallback and
compare bandwidth.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

# Sizes in *elements* of bf16:
#  * 116,727,808 = the 27B run's trainable params (233 MB per allreduce)
#  * 13,107,200  = one 25 MB DDP bucket
#  * 10,822,656  = the toy model's payload (21.6 MB)
#  * 524,288     = 1 MB
SIZES = {
    "27B-grads (233MB)": 116_727_808,
    "DDP-bucket (25MB)": 13_107_200,
    "toy-grads (21.6MB)": 10_822_656,
    "small (1MB)": 524_288,
}


def check(rank: int, world: int, dev: torch.device, n: int, iters: int) -> dict:
    """Allreduce `n` bf16 elements, verify exactness, measure bus bandwidth."""
    # base: integers in [-48, 48] — exact in bf16.
    base = torch.arange(n, dtype=torch.int32, device=dev) % 97 - 48
    # rank r contributes (r+1)*base; the sum over ranks is (1+2+...+world)*base,
    # and every partial sum stays a small even integer -> exact in bf16.
    total_mult = world * (world + 1) // 2
    expected = (base * total_mult).to(torch.bfloat16)
    x = (base * (rank + 1)).to(torch.bfloat16)

    mismatches = 0
    worst = 0.0
    for _ in range(iters):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        if not torch.equal(y, expected):
            mismatches += 1
            worst = max(worst, (y.float() - expected.float()).abs().max().item())

    # bandwidth (ring allreduce bus bandwidth): 2*(n-1)/n * bytes / time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    nbytes = n * 2
    busbw = 2 * (world - 1) / world * nbytes * iters / dt / 1e9
    return {"n": n, "mismatches": mismatches, "worst_diff": worst, "busbw_gbs": busbw, "seconds": dt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)

    if rank == 0:
        print(
            f"world={world}  P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '<unset>')}  "
            f"HSA_ENABLE_IPC_MODE_LEGACY={os.environ.get('HSA_ENABLE_IPC_MODE_LEGACY', '<unset>')}"
        )
        print(f"iters per size: {args.iters}")

    failures = 0
    for label, n in SIZES.items():
        r = check(rank, world, dev, n, args.iters)
        ok = r["mismatches"] == 0
        failures += 0 if ok else 1
        if rank == 0:
            mark = "OK  " if ok else "FAIL"
            print(
                f"  [{mark}] {label:20s} {args.iters} iters, mismatches={r['mismatches']}, "
                f"worst|diff|={r['worst_diff']:.3g}, {r['busbw_gbs']:.1f} GB/s busbw"
            )
        dist.barrier()

    if rank == 0:
        print("P2P ALLREDUCE CORRECTNESS:", "PASS" if failures == 0 else f"FAIL ({failures} sizes)")
    dist.destroy_process_group()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
