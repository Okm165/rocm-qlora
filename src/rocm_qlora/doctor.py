"""`rocm-qlora doctor` — environment + hardware report with GPU probes.

Writes ``.context/doctor-latest.json``: the artifact the train launcher
reads to pick attention settings and AOTriton flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .panels import kv_table
from .utils import human_bytes, timestamp_slug, write_json

if TYPE_CHECKING:
    from .hardware import CpuInfo, GpuInfo, HostMemory, TorchRuntime
    from .launcher import BnbSmoke, DoctorVerdict
    from .probes import HipIpcProbe, MatmulBench, SdpaProbe

console = Console()


def _doctor_host_panel(mem: HostMemory, cpu: CpuInfo) -> None:
    """Host summary panel (CPU/RAM/kernel/driver/disk)."""
    from .hardware import disk_free_gb, kernel_version, rocm_driver_version

    host = kv_table()
    host.add_row("CPU", f"{cpu.get('model', '?')} ({cpu.get('cores')} threads)")
    host.add_row("RAM", f"{mem['total_gib']:.0f} GiB total · {mem['available_gib']:.0f} GiB available")
    host.add_row("kernel", kernel_version())
    host.add_row("amdgpu driver", rocm_driver_version() or "n/a")
    host.add_row("disk free", human_bytes(disk_free_gb(".") * 1e9))
    console.print(Panel(host, title="host", border_style="blue"))


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


def _probe_sdpa() -> SdpaProbe:
    """SDPA backend probe + table; returns the ``sdpa`` verdict section."""
    from .probes import probe_sdpa

    console.print("[dim]probing SDPA backends (fresh subprocess per backend, ~30-60 s)…[/dim]")
    sdpa = probe_sdpa()
    table = Table(title="SDPA backend probe")
    table.add_column("backend", style="dim")
    table.add_column("AOTriton env on", justify="center")
    table.add_column("env off", justify="center")
    for backend in ("flash", "efficient", "math"):
        on = sdpa["backends"].get(f"{backend}_on", {})
        off = sdpa["backends"].get(f"{backend}_off", {})
        table.add_row(
            backend,
            "[green]✓[/green]" if on.get("ok") else f"[red]✗[/red] [dim]{str(on.get('error', ''))[:40]}[/dim]",
            "[green]✓[/green]" if off.get("ok") else "[red]✗[/red]",
        )
    console.print(table)
    console.print(f"[bold]verdict:[/bold] {sdpa['verdict']} → attn_implementation={sdpa['attn_implementation']}")
    return sdpa


def _probe_bnb() -> BnbSmoke:
    """bitsandbytes 4-bit smoke (train + decode shapes) + report."""
    result = _doctor_bnb_smoke()
    status = "[green]✓[/green]" if result.get("ok") else f"[red]✗[/red] {result.get('error', '')}"
    console.print(f"bitsandbytes 4-bit smoke (train + decode shapes): {status}")
    if result.get("decode_nan"):
        console.print("[red]decode-shape NaN detected — upgrade bitsandbytes (ROCm GEMV fix) before using `chat`[/red]")
    return result


def _probe_hip_ipc() -> HipIpcProbe:
    """RCCL IPC-handle probe + report."""
    from .probes import probe_hip_ipc

    ipc = probe_hip_ipc()
    if ipc.get("ok"):
        console.print("RCCL IPC handles (hipIpcGetMemHandle, dma-buf mode): [green]✓[/green] → P2P collectives enabled")
    else:
        console.print(
            f"RCCL IPC handles: [red]✗[/red] {ipc.get('error', '')} — "
            "training will fall back to shared-memory collectives; set NCCL_P2P_DISABLE=1 in [env.extra] to silence"
        )
    return ipc


def _bench_matmul() -> MatmulBench:
    """bf16 matmul TFLOPS on GPU 0 + report."""
    from .probes import matmul_benchmark

    console.print("[dim]bf16 matmul benchmark (8192³)…[/dim]")
    bench = matmul_benchmark()
    console.print(f"[bold]{bench['tflops']:.1f} TFLOPS[/bold] bf16 (8192³, {bench['seconds'] * 1000:.0f} ms/matmul)")
    return bench


def _doctor_probes(args: argparse.Namespace) -> DoctorVerdict:
    """SDPA / bitsandbytes / RCCL-IPC probes (+ optional matmul bench).

    Returns the verdict dict the launcher reads from .context/doctor-latest.json.
    """
    verdict: DoctorVerdict = {}
    if not args.fast:
        verdict["sdpa"] = _probe_sdpa()
        verdict["bnb_smoke"] = _probe_bnb()
        verdict["hip_ipc"] = _probe_hip_ipc()
    if args.bench:
        verdict["bench"] = _bench_matmul()
    return verdict


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run the environment doctor; writes .context/doctor-latest.json."""
    from .hardware import cpu_info, gpu_snapshot, host_memory, package_versions, torch_runtime_info

    context_dir = Path(".context")
    context_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]rocm-qlora doctor[/bold cyan] · v{__version__}")

    mem = host_memory()
    cpu = cpu_info()
    _doctor_host_panel(mem, cpu)

    gpus = gpu_snapshot()
    _doctor_gpu_table(gpus)

    versions = package_versions()
    runtime = torch_runtime_info()
    _doctor_software_tables(versions, runtime)

    verdict = _doctor_probes(args)

    # --- write artifacts ------------------------------------------------------
    verdict["created"] = timestamp_slug()
    verdict["host"] = {"ram": mem, "cpu": cpu}
    verdict["packages"] = versions
    write_json(context_dir / "doctor-latest.json", verdict)
    console.print(f"[dim]saved: {context_dir / 'doctor-latest.json'}[/dim]")
    console.print(
        "[dim]hint: the train launcher reads the SDPA verdict from doctor-latest.json to pick "
        "attention settings automatically.[/dim]"
    )
    return 0


def _doctor_bnb_smoke() -> BnbSmoke:
    """Quantize a Linear on GPU; check train-shape and decode-shape numerics."""
    import torch

    try:
        import bitsandbytes as bnb

        from .hardware import gpu_snapshot

        snap = gpu_snapshot()
        device = f"cuda:{max(snap, key=lambda g: g.mem_free_b).index}" if snap else "cuda:0"
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
