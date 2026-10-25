"""`rocm-qlora doctor` — environment + hardware report.

Reports facts only: host, GPUs, installed packages, torch runtime, the
bitsandbytes 4-bit smoke and an optional matmul benchmark. It configures
nothing — attention, AOTriton and transport are static config (docs/02).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from . import __version__
from .hardware import (
    CpuInfo,
    GpuInfo,
    HostMemory,
    TorchRuntime,
    cpu_info,
    disk_free_gb,
    gpu_snapshot,
    host_memory,
    kernel_version,
    package_versions,
    torch_runtime_info,
)

if TYPE_CHECKING:
    from .probes import BnbSmoke, MatmulBench

console = Console()


def _doctor_host_table(mem: HostMemory, cpu: CpuInfo) -> None:
    """Host summary table (CPU/RAM/kernel/driver/disk)."""

    host = Table(title="host")
    host.add_column("property", style="dim")
    host.add_column("value")
    host.add_row("CPU", f"{cpu.get('model', '?')} ({cpu.get('cores')} threads)")
    host.add_row("RAM", f"{mem['total_gib']:.0f} GiB total · {mem['available_gib']:.0f} GiB available")
    host.add_row("kernel", kernel_version())
    host.add_row("disk free", f"{disk_free_gb('.'):.1f} GB")
    console.print(host)


def _doctor_gpu_table(gpus: list[GpuInfo]) -> None:
    """Per-GPU VRAM/temp/power/util table."""
    gpu_table = Table(title="GPUs (rocm-smi)")
    for col in ("#", "VRAM total", "VRAM free", "temp", "power", "util"):
        gpu_table.add_column(col, justify="right" if col != "#" else "left")
    for g in gpus:
        gpu_table.add_row(
            str(g.index),
            f"{g.mem_total_gb:.0f} GB",
            f"{g.mem_free_gb:.1f} GB",
            f"{g.temp_edge_c:.0f}°C" if g.temp_edge_c is not None else "?",
            f"{g.power_w:.0f} W" if g.power_w is not None else "?",
            f"{g.util_pct:.0f}%" if g.util_pct is not None else "?",
        )
    console.print(gpu_table)


def _doctor_software_tables(versions: dict[str, str], runtime: TorchRuntime) -> None:
    """Installed-package and torch-runtime tables."""
    vtable = Table(title="software")
    vtable.add_column("package", style="dim")
    vtable.add_column("version")
    for name, version in versions.items():
        vtable.add_row(name, version)
    console.print(vtable)

    rtable = Table(title="torch runtime")
    rtable.add_column("key", style="dim")
    rtable.add_column("value")
    for key, value in runtime.items():
        rtable.add_row(key, str(value))
    console.print(rtable)


def _probe_bnb() -> BnbSmoke:
    """bitsandbytes 4-bit smoke (train + decode shapes) + report."""
    from .probes import probe_bnb_smoke

    result = probe_bnb_smoke()
    status = "[green]✓[/green]" if result.get("ok") else f"[red]✗[/red] {result.get('error', '')}"
    console.print(f"bitsandbytes 4-bit smoke (train + decode shapes): {status}")
    if result.get("decode_nan"):
        console.print("[red]decode-shape NaN detected — upgrade bitsandbytes (ROCm GEMV fix) before using `chat`[/red]")
    return result


def _bench_matmul() -> MatmulBench:
    """bf16 matmul TFLOPS on GPU 0 + report."""
    from .probes import matmul_benchmark

    console.print("[dim]bf16 matmul benchmark (8192³)…[/dim]")
    bench = matmul_benchmark()
    console.print(f"[bold]{bench['tflops']:.1f} TFLOPS[/bold] bf16 (8192³, {bench['seconds'] * 1000:.0f} ms/matmul)")
    return bench


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run the environment doctor (a report; nothing is configured from it)."""
    console.rule(f"[bold cyan]rocm-qlora doctor[/bold cyan] · v{__version__}")

    mem = host_memory()
    cpu = cpu_info()
    _doctor_host_table(mem, cpu)

    gpus = gpu_snapshot()
    _doctor_gpu_table(gpus)

    versions = package_versions()
    runtime = torch_runtime_info()
    _doctor_software_tables(versions, runtime)

    _probe_bnb()
    if args.bench:
        _bench_matmul()
    return 0
