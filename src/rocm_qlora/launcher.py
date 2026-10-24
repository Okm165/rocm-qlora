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

import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from .config import Config
from .hardware import (
    CpuInfo,
    GpuPlan,
    HostMemory,
    TorchRuntime,
    cpu_info,
    gpu_processes,
    gpu_snapshot,
    host_memory,
    kernel_version,
    package_versions,
    select_gpus,
    torch_runtime_info,
)
from .probes import BnbSmoke, HipIpcProbe, MatmulBench, SdpaProbe
from .utils import now_iso, read_json, timestamp_slug, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTOR_VERDICT = REPO_ROOT / ".context" / "doctor-latest.json"


# ---------------------------------------------------------------------------
# Doctor verdict plumbing
# ---------------------------------------------------------------------------


class DoctorHost(TypedDict):
    """The ``host`` section of doctor-latest.json."""

    ram: HostMemory
    cpu: CpuInfo


class DoctorVerdict(TypedDict, total=False):
    """`.context/doctor-latest.json`: what `doctor` writes, what the launcher reads.

    The launcher consumes the ``sdpa`` section (backend guard); the rest is the
    artifact for humans and issue reports. ``total=False``: ``doctor --fast``
    skips probes, so every section is optional.
    """

    created: str
    host: DoctorHost
    packages: dict[str, str]
    sdpa: SdpaProbe
    bnb_smoke: BnbSmoke
    hip_ipc: HipIpcProbe
    bench: MatmulBench


def load_doctor_verdict() -> DoctorVerdict | None:
    """Read .context/doctor-latest.json (None when missing/invalid)."""
    try:
        return read_json(DOCTOR_VERDICT)
    except Exception:
        return None


def resolve_attn_impl(cfg: Config) -> str:
    """Attention implementation: config wins; else doctor verdict; else sdpa."""
    if cfg.model.attn != "auto":
        return cfg.model.attn
    verdict = load_doctor_verdict()
    sdpa = verdict.get("sdpa") if verdict else None
    if sdpa and sdpa.get("attn_implementation"):
        return sdpa["attn_implementation"]
    return "sdpa"


def resolve_aotriton(cfg: Config) -> bool:
    """Should the worker export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1?"""
    if cfg.env.aotriton == "on":
        return True
    if cfg.env.aotriton == "off":
        return False
    verdict = load_doctor_verdict()
    sdpa = verdict.get("sdpa") if verdict else None
    if sdpa and sdpa.get("verdict"):
        return sdpa["verdict"] == "aotriton_on"
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
    sdpa = verdict.get("sdpa")
    backends = sdpa.get("backends", {}) if sdpa else {}
    mode = "on" if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1" else "off"
    import torch

    def probe_failed(backend: str) -> bool:
        # A hand-edited or truncated verdict must not crash the worker: only a
        # well-formed entry with ok=False disables a backend.
        entry = backends.get(f"{backend}_{mode}")
        return isinstance(entry, dict) and entry.get("ok") is False

    disabled: list[str] = []
    if probe_failed("flash") and torch.backends.cuda.flash_sdp_enabled():
        torch.backends.cuda.enable_flash_sdp(False)
        disabled.append("flash")
    if probe_failed("efficient") and torch.backends.cuda.mem_efficient_sdp_enabled():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        disabled.append("mem_efficient")
    return disabled


# ---------------------------------------------------------------------------
# Environment for workers
# ---------------------------------------------------------------------------


def build_worker_env(cfg: Config, gpu_ids: list[int]) -> dict[str, str]:
    """Full worker environment: device visibility, allocator, RCCL, seeds."""
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
    # P2P works via dma-buf IPC (HSA_ENABLE_IPC_MODE_LEGACY=0) at ~26 GB/s vs
    # ~5.2 GB/s shared-memory, and is bit-exact at 27B gradient sizes
    # (scripts/poc_p2p_allreduce.py). The default is still shared memory: a
    # simplicity choice — for LoRA its ~45 ms/step is noise next to a 10-25 s
    # step. Opt back in via [env.extra] NCCL_P2P_DISABLE = "0" (docs/02).
    env.setdefault("HSA_ENABLE_IPC_MODE_LEGACY", "0")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")  # sensible default on machines without InfiniBand
    env.setdefault("NCCL_DEBUG", "WARN")
    # The torch wheel's libdrm prints "(null): No such file or directory" per
    # GPU at HIP init unless the ASIC-ID table path points at the system one —
    # set it only when that table is actually installed.
    if Path("/usr/share/libdrm").is_dir():
        env.setdefault("AMDGPU_ASIC_ID_TABLE_PATHS", "/usr/share/libdrm")
    # Fix hash randomization before the interpreter starts (reproducible order).
    env.setdefault("PYTHONHASHSEED", str(cfg.run.seed))
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
    """Create <output_root>/runs/<timestamp>-<name>/ with logs + charts.

    The timestamp slug is minute-resolution, so two launches within the same
    minute would collide; a numeric suffix keeps every run's stream its own.
    """
    root = Path(cfg.run.output_root) / "runs"
    name = f"{timestamp_slug()}-{cfg.run.name}" + ("-dryrun" if dry_run else "")
    run_dir = root / name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{name}-{suffix}"
        suffix += 1
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    return run_dir


def launch_worker(cfg: Config, run_dir: Path, gpu_ids: list[int], *, extra_args: list[str] | None = None) -> int:
    """Spawn the training worker(s); returns the process exit code.

    Both paths run the same module entrypoint (``rocm_qlora.train``):
    directly for one GPU, under ``torchrun --module`` for data parallelism.
    """
    env = build_worker_env(cfg, gpu_ids)
    args = ["--run-dir", str(run_dir), *(extra_args or [])]
    if len(gpu_ids) == 1:
        cmd = [sys.executable, "-m", "rocm_qlora.train", *args]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",  # torchrun's standalone mode assigns its own free port
            "--nnodes=1",
            f"--nproc_per_node={len(gpu_ids)}",
            "--module",
            "rocm_qlora.train",
            *args,
        ]
    return subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode


# ---------------------------------------------------------------------------
# Preflight (printed by `train` before launching)
# ---------------------------------------------------------------------------


def preflight_gpus(cfg: Config) -> GpuPlan:
    """Choose GPUs for this run and collect human-readable context."""
    requested = cfg.dist.gpus
    if cfg.dist.mode == "single":
        # One GPU: the first requested one, or GPU 0.
        requested = [requested[0]] if isinstance(requested, list) else [0]

    plan = select_gpus(requested=requested)
    # Note about other VRAM consumers (e.g. a running inference server).
    holders = gpu_processes()
    for g in plan.gpus:
        who = [f"{p['name']} (pid {p['pid']}, {p['vram_b'] / 1e9:.1f} GB)" for p in holders if str(g) in str(p["gpus"])]
        if who:
            plan.warnings.append(f"GPU {g} is currently shared with: {', '.join(who)}")
    # The SDPA guard needs probe results; without them nothing is disabled.
    sdpa = (load_doctor_verdict() or {}).get("sdpa") or {}
    if not sdpa:
        plan.warnings.append(
            "no SDPA probe verdict in .context/doctor-latest.json — every attention backend stays enabled "
            "(the mem-efficient backward is the known intermittent-NaN source, docs/05); run `rocm-qlora doctor`"
        )
    elif not any(key.startswith(("flash_", "efficient_", "math_")) for key in (sdpa.get("backends") or {})):
        plan.warnings.append(
            "the SDPA verdict has no per-backend results (probe format changed?) — no backend will be disabled; "
            "run `rocm-qlora doctor`"
        )
    return plan


class GpuFingerprint(TypedDict):
    """One GPU as recorded in env_fingerprint.json."""

    index: int
    mem_total_gb: float
    mem_free_gb: float
    temp_edge_c: float | None
    power_w: float | None


class EnvFingerprint(TypedDict):
    """``env_fingerprint.json``: machine + software state stored with a run.

    A *frozen* record: it says what was actually used at run start (resolved
    GPU ids, the doctor verdict applied to the env), which is the point — the
    config snapshot can be edited afterwards, this cannot. It overlaps
    ``.context/doctor-latest.json`` on purpose: the doctor verdict is
    machine state *now*, this is the state that run saw.
    """

    created: str
    gpus_selected: list[int]
    gpu_snapshot: list[GpuFingerprint]
    host_memory: HostMemory
    cpu: CpuInfo
    kernel: str
    packages: dict[str, str]
    torch_runtime: TorchRuntime
    env: dict[str, str | bool]


def write_env_fingerprint(run_dir: Path, cfg: Config, gpu_ids: list[int]) -> None:
    """Snapshot of machine + software state, stored with the run."""

    fingerprint: EnvFingerprint = {
        "created": now_iso(),
        "gpus_selected": gpu_ids,
        "gpu_snapshot": [
            GpuFingerprint(
                index=g.index,
                mem_total_gb=g.mem_total_gb,
                mem_free_gb=g.mem_free_gb,
                temp_edge_c=g.temp_edge_c,
                power_w=g.power_w,
            )
            for g in gpu_snapshot()
        ],
        "host_memory": host_memory(),
        "cpu": cpu_info(),
        "kernel": kernel_version(),
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
    write_json(run_dir / "env_fingerprint.json", fingerprint)
