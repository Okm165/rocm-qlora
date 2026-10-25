"""In-process GPU diagnostics: the bitsandbytes 4-bit smoke and a matmul benchmark.

Both run inside the calling process on the freest visible GPU and only
*report* — they never configure a run. Attention, AOTriton and transport are
static config (`model.attn`, `model.sdpa_disable`, `env.aotriton`,
`env.extra`); docs/02 explains each one and what to check on your machine.
"""

from __future__ import annotations

import time
from typing import TypedDict

from .hardware import freest_gpu_index


class MatmulBench(TypedDict):
    """`matmul_benchmark`'s result."""

    dtype: str
    size: int
    seconds: float
    tflops: float


def matmul_benchmark(dtype_name: str = "bfloat16", size: int = 8192, iters: int = 20, warmup: int = 5) -> MatmulBench:
    """bf16/fp16 matmul TFLOPS on the freest GPU — a quick sanity check of the stack."""
    import torch

    index = freest_gpu_index()
    if index is None:
        raise RuntimeError("no GPU visible to torch — cannot run the matmul benchmark")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    device = f"cuda:{index}"
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(warmup):
        a @ b  # pyright: ignore[reportUnusedExpression]  # exercising the matmul
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b  # pyright: ignore[reportUnusedExpression]  # exercising the matmul
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = 2 * size**3 / dt / 1e12
    return {"dtype": dtype_name, "size": size, "seconds": dt, "tflops": tflops}


class BnbSmoke(TypedDict, total=False):
    """`probe_bnb_smoke`'s result — doctor's bitsandbytes check."""

    ok: bool
    error: str
    decode_nan: bool
    max_err_seq1: float  # numeric diagnostics for humans reading the report
    max_err_seq4: float


def probe_bnb_smoke() -> BnbSmoke:
    """Quantize a Linear on the freest GPU; check train-shape and decode-shape numerics."""
    import torch

    try:
        import bitsandbytes as bnb

        index = freest_gpu_index()
        if index is None:
            raise RuntimeError("no GPU visible to torch — cannot run the bitsandbytes smoke")
        device = f"cuda:{index}"
        torch.manual_seed(0)
        lin = torch.nn.Linear(512, 256, bias=False).to(device, dtype=torch.float16)
        q = bnb.nn.Linear4bit(
            512, 256, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4"
        )
        q.weight = bnb.nn.Params4bit(
            lin.weight.data,
            requires_grad=False,
            quant_type="nf4",  # pyright: ignore[reportCallIssue]  # bnb stubs omit the 4-bit params
            compress_statistics=True,  # pyright: ignore[reportCallIssue]
        )
        q = q.to(device)
        result: BnbSmoke = {"ok": True}
        errors: dict[int, float] = {}
        for seq in (1, 4):
            x = torch.randn(1, seq, 512, device=device, dtype=torch.bfloat16)
            y = q(x)
            ref = lin(x.to(torch.float16)).float()
            err = (y.float() - ref).abs().max().item()
            errors[seq] = err
            if seq == 1 and (torch.isnan(y).any() or err > 1.0):
                result["decode_nan"] = True
                result["ok"] = False
        result["max_err_seq1"] = errors.get(1, float("nan"))
        result["max_err_seq4"] = errors.get(4, float("nan"))
        return result
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
