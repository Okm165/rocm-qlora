"""Hardware discovery and memory math for AMD ROCm systems.

* ``gpu_snapshot`` / ``gpu_processes`` — rocm-smi output, parsed tolerantly.
* ``estimate_training_vram`` — a transparent, printed decomposition.
* ``probe_sdpa`` — which attention backends actually launch on this stack
  (a fresh subprocess per backend: a failed RDNA3 kernel launch can surface
  as a *deferred* error and contaminate the process).
* ``select_gpus`` — pick GPUs whose free VRAM fits the estimate.

Everything here is safe to import without torch installed; torch is imported
lazily where needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# rocm-smi parsing
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    index: int
    name: str = "AMD GPU"
    mem_total_b: int = 0
    mem_used_b: int = 0
    temp_edge_c: float | None = None
    temp_junction_c: float | None = None
    power_w: float | None = None
    util_pct: float | None = None

    @property
    def mem_free_b(self) -> int:
        return max(0, self.mem_total_b - self.mem_used_b)

    @property
    def mem_free_gb(self) -> float:
        return self.mem_free_b / 1e9

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used_b / 1e9

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total_b / 1e9


def _first_key(d: dict[str, Any], *needles: str) -> Any:
    for needle in needles:
        for key, value in d.items():
            if needle.lower() in key.lower():
                return value
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def gpu_snapshot(timeout: float = 10.0) -> list[GpuInfo]:
    """Current per-GPU state via `rocm-smi --json`. Returns [] on any failure."""
    flags = ["--showmeminfo", "vram", "--showtemp", "--showpower", "--showuse"]
    try:
        proc = subprocess.run(
            ["rocm-smi", "--json", *flags],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        raw = json.loads(proc.stdout)
    except Exception:
        return []

    gpus: list[GpuInfo] = []
    for key in sorted(raw.keys()):
        if not key.startswith("card"):
            continue
        idx = _to_int(key.replace("card", ""))
        if idx is None:
            continue
        d = raw[key]
        gpus.append(
            GpuInfo(
                index=idx,
                mem_total_b=_to_int(_first_key(d, "VRAM Total Memory")) or 0,
                mem_used_b=_to_int(_first_key(d, "VRAM Total Used Memory")) or 0,
                temp_edge_c=_to_float(_first_key(d, "Temperature (Sensor edge)", "Temperature (Sensor Edge)")),
                temp_junction_c=_to_float(
                    _first_key(d, "Temperature (Sensor junction)", "Temperature (Sensor Junction)")
                ),
                power_w=_to_float(_first_key(d, "Average Graphics Package Power", "Average Power", "Power (W)")),
                util_pct=_to_float(_first_key(d, "GPU use (%)", "GPU Use (%)")),
            )
        )
    return gpus


class GpuProcess(TypedDict):
    """One process holding GPU memory (parsed from ``rocm-smi --showpids``)."""

    pid: int
    name: str
    gpus: str  # comma-separated device ids, as rocm-smi prints them
    vram_b: int


def gpu_processes(timeout: float = 10.0) -> list[GpuProcess]:
    """Processes holding GPU memory (parses the `--showpids` text table).

    Used by preflight to explain *why* free VRAM is low, e.g. a running
    inference server occupying all the GPUs.
    """
    try:
        proc = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return []
    out: list[GpuProcess] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            out.append(
                {
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "gpus": parts[2],
                    "vram_b": _to_int(parts[3]) or 0,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Host + environment fingerprint
# ---------------------------------------------------------------------------


class HostMemory(TypedDict):
    """Total/available host RAM in GiB."""

    total_gib: float
    available_gib: float


def host_memory() -> HostMemory:
    """Total/available host RAM in GiB (psutil if available, else /proc)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_gib": vm.total / 2**30, "available_gib": vm.available / 2**30}
    except Exception:
        try:
            meminfo = Path("/proc/meminfo").read_text()
            vals = {
                k.strip(): int(v.split()[0])
                for k, v in (line.split(":") for line in meminfo.splitlines() if ":" in line)
            }
            total = vals.get("MemTotal", 0) / 2**20
            avail = vals.get("MemAvailable", 0) / 2**20
            return {"total_gib": total, "available_gib": avail}
        except Exception:
            return {"total_gib": float("nan"), "available_gib": float("nan")}


class CpuInfo(TypedDict, total=False):
    """`cpu_info`'s result — the ``cpu`` section of doctor-latest.json."""

    cores: int | None
    physical_cores: int | None
    freq_mhz: int
    model: str


def cpu_info() -> CpuInfo:
    """CPU model + thread count from /proc/cpuinfo."""
    info: CpuInfo = {"cores": os.cpu_count()}
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["freq_mhz"] = round(psutil.cpu_freq().max or 0)
    except Exception:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return info


def kernel_version() -> str:
    """Running kernel release string."""
    try:
        import platform

        return platform.release()
    except Exception:  # pragma: no cover
        return "unknown"


def rocm_driver_version() -> str:
    """amdgpu kernel driver version via sysfs ('' if unavailable)."""
    try:
        return Path("/sys/module/amdgpu/version").read_text().strip()
    except Exception:
        return ""


def package_versions() -> dict[str, str]:
    """Installed versions of the relevant packages (no heavy imports)."""
    names = [
        "torch",
        "triton-rocm",
        "pytorch-triton-rocm",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "datasets",
        "accelerate",
        "rich",
        "matplotlib",
        "numpy",
        "psutil",
        "safetensors",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return out


class TorchRuntime(TypedDict, total=False):
    """`torch_runtime_info`'s result — the ``torch_runtime`` section of the fingerprint."""

    torch: str
    hip: str | None
    cuda_available: bool
    device_count: int
    device_names: list[str]
    gcn_arch: list[str]
    bf16_supported: bool
    error: str


def torch_runtime_info() -> TorchRuntime:
    """Torch/HIP runtime facts. Imports torch — call only in GPU-aware commands."""
    info: TorchRuntime = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["hip"] = getattr(torch.version, "hip", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            names, archs = [], []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                names.append(props.name)
                archs.append(getattr(props, "gcnArchName", "?"))
            info["device_names"] = names
            info["gcn_arch"] = archs
            info["bf16_supported"] = all(
                torch.cuda.get_device_properties(i).major >= 8
                or "gfx11" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")
                for i in range(torch.cuda.device_count())
            )
    except Exception as exc:  # pragma: no cover - doctor reports instead of crashing
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------


@dataclass
class GpuPlan:
    gpus: list[int]
    free_gb: dict[int, float]
    needed_gb: float
    warnings: list[str] = field(default_factory=list)


def select_gpus(needed_gb: float, *, requested: str | list[int] = "auto", safety_factor: float = 1.10) -> GpuPlan:
    """Pick GPUs whose *free* VRAM can hold the estimated footprint."""
    snap = gpu_snapshot()
    free = {g.index: g.mem_free_gb for g in snap}
    warnings: list[str] = []
    need = needed_gb * safety_factor

    if isinstance(requested, list):
        chosen = list(requested)
        for gpu in chosen:
            if gpu not in free:
                warnings.append(f"GPU {gpu} not visible to rocm-smi")
            elif free[gpu] < need:
                warnings.append(
                    f"GPU {gpu}: {free[gpu]:.1f} GB free < {need:.1f} GB needed "
                    f"(safety ×{safety_factor:.2f}) — expect OOM"
                )
        return GpuPlan(gpus=chosen, free_gb=free, needed_gb=need, warnings=warnings)

    if not free:
        raise RuntimeError("No GPUs detected by rocm-smi")

    fitting = sorted([g for g, f in free.items() if f >= need])
    if fitting:
        # Prefer using all fitting GPUs — data-parallel throughput is free.
        return GpuPlan(gpus=fitting, free_gb=free, needed_gb=need, warnings=warnings)

    # Nothing fits: report every GPU and the processes holding it.
    holders = gpu_processes()
    for g in free:
        who = ", ".join(f"{p['name']}(pid {p['pid']})" for p in holders if str(g) in str(p["gpus"]))
        warnings.append(
            f"GPU {g}: {free[g]:.1f} GB free of {snap[g].mem_total_gb:.0f} GB" + (f" — held by: {who}" if who else "")
        )
    raise RuntimeError(
        f"No GPU has {need:.1f} GB free VRAM (need = estimate {needed_gb:.1f} GB × "
        f"safety {safety_factor:.2f}).\n" + "\n".join(warnings) + "\n"
        "Options: free VRAM (stop other GPU processes), lower data.max_length or "
        "optim.per_device_batch_size, or use fewer/other GPUs via dist.gpus."
    )


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
    ok_key: str = "ok",
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
        return {ok_key: False, "error": (proc.stderr.strip().splitlines() or ["no output"])[-1]}
    except subprocess.TimeoutExpired:
        return {ok_key: False, "error": "timeout"}
    except Exception as exc:
        return {ok_key: False, "error": str(exc)}


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
    """One backend's probe outcome (parsed from the subprocess JSON)."""

    ok: bool
    error: str


class SdpaProbe(TypedDict):
    """`probe_sdpa`'s result — the ``sdpa`` section of doctor-latest.json."""

    verdict: str
    attn_implementation: str
    backends: dict[str, SdpaBackendResult]


class HipIpcProbe(TypedDict, total=False):
    """`probe_hip_ipc`'s result — the ``hip_ipc`` section of doctor-latest.json."""

    ipc_ok: bool
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
    "ipc_ok": rc2 == 0,
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
    return _run_probe(_HIP_IPC_PROBE, [], env, timeout, python_exe=python_exe, ok_key="ipc_ok")


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


def disk_free_gb(path: str | Path) -> float:
    """Free space (GiB) on the filesystem holding `path`."""
    import shutil

    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:
        return float("nan")
