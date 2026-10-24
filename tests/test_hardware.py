"""Hardware parsing + GPU selection tests (no GPU needed)."""

from __future__ import annotations

import json
import subprocess

from rocm_qlora import hardware
from rocm_qlora.hardware import GpuInfo

ROCM_SMI_JSON = json.dumps(
    {
        "card0": {
            "Temperature (Sensor edge) (C)": "48.0",
            "Temperature (Sensor junction) (C)": "53.0",
            "Average Graphics Package Power (W)": "14.0",
            "GPU use (%)": "0",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "27959296",
        },
        "card1": {
            "Temperature (Sensor edge) (C)": "42.0",
            "Average Graphics Package Power (W)": "15.0",
            "GPU use (%)": "37",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "20000000000",
        },
    }
)


def test_gpu_snapshot_parses_rocm_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=ROCM_SMI_JSON, stderr="")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    gpus = hardware.gpu_snapshot()
    assert len(gpus) == 2
    assert gpus[0].mem_total_b == 25753026560
    assert abs(gpus[0].mem_free_gb - 25.72) < 0.1
    assert gpus[1].util_pct == 37.0
    assert gpus[1].mem_free_gb < 6.0


def test_gpu_processes_parsing(monkeypatch):
    text = (
        "============================ ROCm System Management Interface ============================\n"
        "PID    \tPROCESS NAME\tGPU(s)\tVRAM USED  \tSDMA USED\tCU OCCUPANCY\t\n"
        "1049858\tinference-server\t3    \t49975758848\t0        \tUNKNOWN      \t\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=text, stderr="")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    procs = hardware.gpu_processes()
    assert procs and procs[0]["name"] == "inference-server"
    assert procs[0]["vram_b"] == 49975758848


def test_select_gpus_auto_picks_fitting(monkeypatch):
    snap = [
        GpuInfo(index=0, mem_total_b=24_000_000_000, mem_used_b=3_000_000_000),  # 21 free
        GpuInfo(index=1, mem_total_b=24_000_000_000, mem_used_b=20_000_000_000),  # 4 free
        GpuInfo(index=2, mem_total_b=24_000_000_000, mem_used_b=2_000_000_000),  # 22 free
    ]
    monkeypatch.setattr(hardware, "gpu_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(hardware, "gpu_processes", lambda *a, **k: [])
    plan = hardware.select_gpus(15.0)
    assert plan.gpus == [0, 2]


def test_select_gpus_raises_with_holders(monkeypatch):
    snap = [GpuInfo(index=0, mem_total_b=24_000_000_000, mem_used_b=20_000_000_000)]
    monkeypatch.setattr(hardware, "gpu_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(
        hardware,
        "gpu_processes",
        lambda *a, **k: [{"pid": 1, "name": "inference-server", "gpus": "0", "vram_b": 20_000_000_000}],
    )
    try:
        hardware.select_gpus(15.0)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "inference-server" in str(exc)
