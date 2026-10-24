"""Empirical GPU capability probes (doctor).

Each probe runs in a fresh subprocess: a failed kernel launch on RDNA3 can
surface as a *deferred* error and contaminate the process that tried it.
Torch is only imported inside the probe scripts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# SDPA backend probe (fresh subprocess per backend)
# ---------------------------------------------------------------------------

_SDPA_PROBE = r"""
import json, os, sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

name = sys.argv[1]
torch.manual_seed(0)
q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
result = {"backend": name, "env_aotriton": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")}
try:
    with sdpa_kernel([BACKENDS[name]]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        val = float(out.float().abs().sum().item())
    # Trailing *checked* op: RDNA3 kernel-launch failures are deferred and
    # only surface at the next synchronized operation.
    sanity = float((torch.ones(16, device="cuda") + 1).sum().item())
    result.update(ok=True, out_sum=val, sanity=sanity)
except Exception as exc:
    result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
print(json.dumps(result))
"""


def _run_probe(
    script: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
    *,
    python_exe: str | None = None,
) -> Any:
    """Run one probe script in a fresh process; returns its parsed JSON line or an error dict.

    The JSON comes from the embedded probe scripts, so it is untyped at this
    boundary; the public wrappers (``probe_sdpa``, ``probe_hip_ipc``) re-type it.
    """
    try:
        proc = subprocess.run(
            [python_exe or sys.executable, "-c", script, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"ok": False, "error": (proc.stderr.strip().splitlines() or ["no output"])[-1]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _sdpa_probe_env(env_mode: str) -> dict[str, str]:
    """Probe environment for one AOTriton mode ("on"/"off")."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8")
    if env_mode == "on":
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    else:
        env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
    return env


def _run_sdpa_probe(python_exe: str | None, backend: str, env: dict[str, str], timeout: float) -> SdpaBackendResult:
    """Run one backend in a fresh process; returns its JSON result or an error."""
    return _run_probe(_SDPA_PROBE, [backend], env, timeout, python_exe=python_exe)


def _sdpa_verdict(backends: dict[str, Any]) -> tuple[str, str]:
    """(verdict, attn_implementation) from the per-backend probe results."""
    flash_on = backends.get("flash_on", {}).get("ok", False)
    flash_off = backends.get("flash_off", {}).get("ok", False)
    eff_on = backends.get("efficient_on", {}).get("ok", False)
    math_ok = backends.get("math_on", {}).get("ok", False) or backends.get("math_off", {}).get("ok", False)

    if flash_on or eff_on:
        return "aotriton_on", "sdpa"
    if flash_off:
        return "aotriton_off", "sdpa"
    if math_ok:
        return "math_only", "eager"
    return "all_failed", "eager"


class SdpaBackendResult(TypedDict, total=False):
    """One backend's probe outcome — the complete subprocess JSON payload."""

    backend: str
    env_aotriton: str
    ok: bool
    out_sum: float
    sanity: float
    error: str


class SdpaProbe(TypedDict):
    """`probe_sdpa`'s result — the ``sdpa`` section of doctor-latest.json."""

    verdict: str
    attn_implementation: str
    backends: dict[str, SdpaBackendResult]


class HipIpcProbe(TypedDict, total=False):
    """`probe_hip_ipc`'s result — the complete subprocess JSON payload."""

    alloc_rc: int
    ipc_rc: int
    ok: bool
    ipc_mode_legacy: str
    error: str


class MatmulBench(TypedDict):
    """`matmul_benchmark`'s result — the ``bench`` section of doctor-latest.json."""

    dtype: str
    size: int
    seconds: float
    tflops: float


def probe_sdpa(python_exe: str | None = None, timeout: float = 300.0) -> SdpaProbe:
    """Empirically test SDPA backends on THIS machine.

    Runs each backend in a fresh process (deferred launch errors on RDNA3
    contaminate the process state — see pytorch issue #194498), with the
    AOTriton experimental flag ON (needed to enable fused attention on some
    gfx1100 wheels) and OFF, and reports what actually launches.
    """
    results: SdpaProbe = {"backends": {}, "verdict": "unknown", "attn_implementation": "sdpa"}

    for env_mode in ("on", "off"):
        env = _sdpa_probe_env(env_mode)
        for backend in ("flash", "efficient", "math"):
            results["backends"][f"{backend}_{env_mode}"] = _run_sdpa_probe(python_exe, backend, env, timeout)

    results["verdict"], results["attn_implementation"] = _sdpa_verdict(results["backends"])
    return results


_HIP_IPC_PROBE = r"""
import ctypes, json, os, sys
import torch

lib_path = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
lib = ctypes.CDLL(lib_path)
lib.hipGetErrorName.restype = ctypes.c_char_p
lib.hipGetErrorName.argtypes = [ctypes.c_int]
lib.hipInit(ctypes.c_uint(0))

class Handle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * 64)]

ptr = ctypes.c_void_p()
rc = lib.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(1 << 20))
handle = Handle()
rc2 = lib.hipIpcGetMemHandle(ctypes.byref(handle), ptr) if rc == 0 else -1
err = lib.hipGetErrorName(rc2).decode() if rc2 else None
print(json.dumps({
    "alloc_rc": rc,
    "ipc_rc": rc2,
    "ok": rc2 == 0,
    "error": err,
    "ipc_mode_legacy": os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY", "<unset>"),
}))
"""


def probe_hip_ipc(python_exe: str | None = None, timeout: float = 120.0) -> HipIpcProbe:
    """Can HIP export IPC memory handles in this environment?

    RCCL's P2P transport needs `hipIpcGetMemHandle`. ROCm 7.1 defaults to
    *legacy* IPC mode, whose export ioctl exists only in the out-of-tree DKMS
    driver — on mainline kernels it fails unless `HSA_ENABLE_IPC_MODE_LEGACY=0`
    selects the dma-buf path. Fresh subprocess: the flag must be set before
    HIP initializes.
    """
    env = dict(os.environ)
    env["HSA_ENABLE_IPC_MODE_LEGACY"] = "0"
    return _run_probe(_HIP_IPC_PROBE, [], env, timeout, python_exe=python_exe)


def matmul_benchmark(dtype_name: str = "bfloat16", size: int = 8192, iters: int = 20, warmup: int = 5) -> MatmulBench:
    """bf16/fp16 matmul TFLOPS on GPU 0 — a quick sanity check of the stack."""
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    device = "cuda"
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
