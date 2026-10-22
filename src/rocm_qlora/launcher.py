"""Launcher: GPU selection, environment assembly, worker orchestration.

The `train` command is a *parent* process: it prints preflight information,
picks GPUs, snapshots the config into the run directory, and then spawns the
actual training worker(s) — one process for single-GPU runs, `torchrun` for
data-parallel runs. Keeping the worker in a child process guarantees a clean
HIP context: environment variables that must exist *before* torch initializes
(AOTriton flags, allocator config) are set on the child only.

Doctor verdicts
---------------
``rocm-qlora doctor`` probes which SDPA backends actually launch on this
machine and records a verdict in ``.context/doctor-latest.json``. The
launcher consults it to decide:

* attention implementation: ``sdpa`` vs ``eager`` (math fallback),
* whether ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` should be exported,
* which SDPA backends to disable outright: a probe-failed backend stays off
  for the whole worker process — the mem-efficient *backward* is the observed
  intermittent-NaN source on gfx1100, and padded batches (4D mask) are what
  dispatch to it (see docs/05).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .hardware import gpu_processes, gpu_snapshot, select_gpus
from .utils import read_json, timestamp_slug

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTOR_VERDICT = REPO_ROOT / ".context" / "doctor-latest.json"


# ---------------------------------------------------------------------------
# Doctor verdict plumbing
# ---------------------------------------------------------------------------


def load_doctor_verdict() -> dict | None:
    try:
        return read_json(DOCTOR_VERDICT)
    except Exception:
        return None


def resolve_attn_impl(cfg: Config) -> str:
    """Attention implementation: config wins; else doctor verdict; else sdpa."""
    if cfg.model.attn != "auto":
        return cfg.model.attn
    verdict = load_doctor_verdict()
    if verdict and verdict.get("sdpa", {}).get("attn_implementation"):
        return verdict["sdpa"]["attn_implementation"]
    return "sdpa"


def resolve_aotriton(cfg: Config) -> bool:
    """Should the worker export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1?"""
    if cfg.env.aotriton == "on":
        return True
    if cfg.env.aotriton == "off":
        return False
    verdict = load_doctor_verdict()
    if verdict and verdict.get("sdpa", {}).get("verdict"):
        return verdict["sdpa"]["verdict"] == "aotriton_on"
    return True  # recommended default on gfx1100; harmless elsewhere


def apply_sdpa_verdict() -> list[str]:
    """Turn off SDPA backends the doctor probe flagged as broken.

    The probe covers *forward* kernels; on gfx1100 the mem-efficient
    **backward** is the observed intermittent-NaN source (docs/05), and an
    explicit attention mask — which padded batches always pass — is what
    dispatches to it. A failed probe therefore disables the backend for the
    whole worker process: masked batches fall back to math attention, while
    causal unmasked ones stay on the AOTriton flash path. No verdict on
    disk → no change.
    """
    verdict = load_doctor_verdict()
    if not verdict:
        return []
    backends = verdict.get("sdpa", {}).get("backends", {})
    mode = "on" if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1" else "off"
    import torch

    disabled: list[str] = []
    if backends.get(f"flash_{mode}", {}).get("ok") is False and torch.backends.cuda.flash_sdp_enabled():
        torch.backends.cuda.enable_flash_sdp(False)
        disabled.append("flash")
    if backends.get(f"efficient_{mode}", {}).get("ok") is False and torch.backends.cuda.mem_efficient_sdp_enabled():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        disabled.append("mem_efficient")
    return disabled


# ---------------------------------------------------------------------------
# Environment for workers
# ---------------------------------------------------------------------------


def build_worker_env(cfg: Config, gpu_ids: list[int]) -> dict[str, str]:
    env = dict(os.environ)
    visible = ",".join(str(g) for g in gpu_ids)
    # Both are honoured (HIP runtime reads HIP_VISIBLE_DEVICES; torch.cuda
    # reads CUDA_VISIBLE_DEVICES) — setting both keeps device indices aligned.
    env["HIP_VISIBLE_DEVICES"] = visible
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["OMP_NUM_THREADS"] = str(cfg.env.omp_num_threads)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # --- RCCL transport on consumer RDNA3 -----------------------------------
    # P2P *works* here once the runtime uses dma-buf IPC (the flag below):
    # `HSA_ENABLE_IPC_MODE_LEGACY=0` makes `hipIpcGetMemHandle` succeed and
    # RCCL runs collectives `via P2P/IPC` at ~26 GB/s vs ~5.2 GB/s for the
    # shared-memory fallback (measured; see scripts/poc_p2p_allreduce.py,
    # which verifies P2P allreduce is bit-exact at the 27B gradient sizes).
    #
    # The default is nevertheless the *shared-memory* transport — a
    # simplicity choice that keeps the IPC/IOMMU path out of the picture.
    # The intermittent NaN-gradient events observed on the 27B are
    # model-side, not transport-side: they occurred over both transports,
    # and plain allreduce is bit-exact on both at the 27B's gradient sizes
    # (100 iters x 4 sizes). For LoRA, SHM costs ~45 ms/step — noise next to
    # a ~10-25 s step. The NaN guards abort loudly if anything misbehaves.
    #
    # Opt back into P2P via the config:
    #   [env.extra]
    #   NCCL_P2P_DISABLE = "0"
    env.setdefault("HSA_ENABLE_IPC_MODE_LEGACY", "0")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")  # sensible default on machines without InfiniBand
    env.setdefault("NCCL_DEBUG", "WARN")
    if cfg.env.hip_alloc_conf:
        env["PYTORCH_HIP_ALLOC_CONF"] = cfg.env.hip_alloc_conf
    if resolve_aotriton(cfg):
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    else:
        env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
    env.update({str(k): str(v) for k, v in cfg.env.extra.items()})
    return env


# ---------------------------------------------------------------------------
# Run directory + launch
# ---------------------------------------------------------------------------


def make_run_dir(cfg: Config, *, dry_run: bool = False) -> Path:
    root = Path(cfg.run.output_root) / "runs"
    slug = timestamp_slug()
    name = f"{slug}-{cfg.run.name}" + ("-dryrun" if dry_run else "")
    run_dir = root / name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    return run_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class LaunchResult:
    returncode: int
    run_dir: Path
    gpu_ids: list[int]


def launch_worker(
    cfg: Config, run_dir: Path, gpu_ids: list[int], *, extra_args: list[str] | None = None
) -> LaunchResult:
    """Spawn the training worker(s) with a fully-specified environment."""
    env = build_worker_env(cfg, gpu_ids)
    extra_args = extra_args or []
    if len(gpu_ids) == 1:
        cmd = [sys.executable, "-m", "rocm_qlora.train", "--run-dir", str(run_dir), *extra_args]
    else:
        worker = REPO_ROOT / "scripts" / "train_worker.py"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={len(gpu_ids)}",
            f"--master_port={_free_port()}",
            str(worker),
            "--run-dir",
            str(run_dir),
            *extra_args,
        ]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return LaunchResult(returncode=proc.returncode, run_dir=run_dir, gpu_ids=gpu_ids)


# ---------------------------------------------------------------------------
# Preflight (printed by `train` before launching)
# ---------------------------------------------------------------------------


@dataclass
class Preflight:
    gpu_ids: list[int]
    free_gb: dict[int, float]
    warnings: list[str]


def preflight_gpus(cfg: Config, needed_gb: float) -> Preflight:
    """Choose GPUs for this run and collect human-readable context."""
    requested = cfg.dist.gpus
    mode = cfg.dist.mode
    if mode == "single" and isinstance(requested, str):
        requested = [0]

    plan = select_gpus(needed_gb, requested=requested, safety_factor=cfg.dist.safety_factor)
    warnings = list(plan.warnings)

    # Note about other VRAM consumers (e.g. a running inference server).
    holders = gpu_processes()
    for g in plan.gpus:
        who = [f"{p['name']} (pid {p['pid']}, {p['vram_b'] / 1e9:.1f} GB)" for p in holders if str(g) in str(p["gpus"])]
        if who:
            warnings.append(f"GPU {g} is currently shared with: {', '.join(who)}")

    if mode == "fsdp" and len(plan.gpus) < 2:
        warnings.append("dist.mode='fsdp' with a single GPU behaves like single-device training")
    return Preflight(gpu_ids=plan.gpus, free_gb=plan.free_gb, warnings=warnings)


def write_env_fingerprint(run_dir: Path, cfg: Config, gpu_ids: list[int]) -> None:
    """Snapshot of machine + software state, stored with the run."""
    from .hardware import (
        cpu_info,
        host_memory,
        kernel_version,
        package_versions,
        rocm_driver_version,
        torch_runtime_info,
    )

    fingerprint = {
        "created": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpus_selected": gpu_ids,
        "gpu_snapshot": [
            {
                "index": g.index,
                "mem_total_gb": g.mem_total_b / 1e9,
                "mem_free_gb": g.mem_free_gb,
                "temp_edge_c": g.temp_edge_c,
                "power_w": g.power_w,
            }
            for g in gpu_snapshot()
        ],
        "host_memory": host_memory(),
        "cpu": cpu_info(),
        "kernel": kernel_version(),
        "amdgpu_driver": rocm_driver_version(),
        "packages": package_versions(),
        "torch_runtime": torch_runtime_info(),
        "env": {
            "PYTORCH_HIP_ALLOC_CONF": cfg.env.hip_alloc_conf,
            "aotriton": resolve_aotriton(cfg),
            "attn_impl": resolve_attn_impl(cfg),
            "rccl_transport": (
                "shared-memory (default; opt into P2P with [env.extra] NCCL_P2P_DISABLE=0)"
                if str(cfg.env.extra.get("NCCL_P2P_DISABLE", "1")) != "0"
                else "P2P/IPC (dma-buf IPC; verified bit-exact at 27B sizes — see docs/02)"
            ),
        },
    }
    (run_dir / "env_fingerprint.json").write_text(json.dumps(fingerprint, indent=2, default=str))
