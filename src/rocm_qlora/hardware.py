"""Hardware discovery for AMD ROCm systems.

* ``gpu_snapshot`` / ``gpu_processes`` — rocm-smi output, parsed tolerantly.
* ``select_gpus`` — the GPUs for a run: the requested ones, or every visible
  one. Free VRAM is reported as a fact; the run either fits or OOMs, and
  docs/02 (measured budget) + docs/05 (knobs) are what to react with.
* ``probe_sdpa`` — which attention backends actually launch on this stack
  (a fresh subprocess per backend: a failed RDNA3 kernel launch can surface
  as a *deferred* error and contaminate the process).

Everything here is safe to import without torch installed; torch is imported
lazily where needed.
"""

from __future__ import annotations

import json
import os
import subprocess
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


class GpuMetric(TypedDict):
    """One GPU's sampled state inside a train record."""

    index: int
    mem_used_gb: float
    mem_total_gb: float
    temp_edge_c: float | None
    temp_junction_c: float | None
    power_w: float | None
    util_pct: float | None


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
    warnings: list[str] = field(default_factory=list)


def select_gpus(*, requested: str | list[int] = "auto") -> GpuPlan:
    """The GPUs for this run: the requested ones, or every visible one.

    No size is guessed. If the choice is too small for the model, the run
    OOMs and the user reacts with the measured budget (docs/02) and the
    knobs in docs/05.
    """
    snap = gpu_snapshot()
    free = {g.index: g.mem_free_gb for g in snap}

    if isinstance(requested, list):
        warnings = [f"GPU {gpu} not visible to rocm-smi" for gpu in requested if gpu not in free]
        return GpuPlan(gpus=list(requested), free_gb=free, warnings=warnings)

    if not free:
        raise RuntimeError("No GPUs detected by rocm-smi")
    # Every visible GPU — data-parallel throughput is free.
    return GpuPlan(gpus=sorted(free), free_gb=free)


def disk_free_gb(path: str | Path) -> float:
    """Free space (GiB) on the filesystem holding `path`."""
    import shutil

    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:
        return float("nan")
