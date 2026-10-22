"""Hardware discovery and memory math for AMD ROCm systems.

This module is the project's "eyes on the machine". It answers:

* What GPUs exist, how much VRAM is free right now, who is using it?
  (``rocm-smi`` JSON, parsed tolerantly across rocm-smi versions.)
* How much VRAM will a given training config need? (``estimate_training_vram``
  — a transparent, printed decomposition rather than a magic number.)
* Which SDPA attention backends actually work on this ROCm/PyTorch stack?
  (``probe_sdpa`` — subprocess probes, because a failed kernel launch on
  RDNA3 can surface as a *deferred* error and contaminate the process.)
* Which GPUs should this run use? (``select_gpus``.)

Design rule: everything here is safe to import without torch installed
(tests, plotting, `data` command). Torch is imported lazily where needed.
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
from typing import Any

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


def gpu_processes(timeout: float = 10.0) -> list[dict[str, Any]]:
    """Processes holding GPU memory (parses the `--showpids` text table).

    Used by preflight to explain *why* free VRAM is low, e.g. a running
    inference server occupying all the GPUs.
    """
    try:
        proc = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
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


def host_memory() -> dict[str, float]:
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


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cores": os.cpu_count()}
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


def torch_runtime_info() -> dict[str, Any]:
    """Torch/HIP runtime facts. Imports torch — call only in GPU-aware commands."""
    info: dict[str, Any] = {}
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
# Model geometry + VRAM estimation
# ---------------------------------------------------------------------------


@dataclass
class ModelGeometry:
    """Parameter accounting for the *prepared* (quantized) checkpoint.

    Numbers come from ``finetune_meta.json`` when the checkpoint was created
    by ``llm-finetune prepare`` (exact), otherwise from ``config.json``
    (heuristic). Everything is in *parameters*, not bytes.
    """

    name: str = "unknown"
    total_params: float = 0.0
    embed_params: float = 0.0
    lm_head_params: float = 0.0  # 0 if tied
    quantized_params: float = 0.0
    nonquant_16bit_params: float = 0.0  # norms, convs, non-requantized extras
    vision_params: float = 0.0  # 0 if dropped in prepare
    # LoRA-relevant per-layer capacity (for trainable-parameter estimate):
    lora_units: float = 0.0  # sum over target linears of (in_features + out_features)
    hidden_size: int = 0
    n_layers: int = 0
    vocab_size: int = 0
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)

    @property
    def trainable_lora_params(self) -> float:
        # r * (in+out) per target linear, summed over all layers.
        return self.lora_units * 0.0  # placeholder, filled by caller with r


def geometry_from_finetune_meta(meta_path: Path, *, lora_r: int) -> ModelGeometry | None:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    geom = ModelGeometry(
        name=meta.get("name", "unknown"),
        total_params=float(meta.get("total_params", 0.0)),
        embed_params=float(meta.get("embed_params", 0.0)),
        lm_head_params=float(meta.get("lm_head_params", 0.0)),
        quantized_params=float(meta.get("quantized_params", 0.0)),
        nonquant_16bit_params=float(meta.get("nonquant_16bit_params", 0.0)),
        vision_params=float(meta.get("vision_params", 0.0)),
        lora_units=float(meta.get("lora_units", 0.0)),
        hidden_size=int(meta.get("hidden_size", 0) or 0),
        n_layers=int(meta.get("n_layers", 0) or 0),
        vocab_size=int(meta.get("vocab_size", 0) or 0),
        source="finetune_meta",
    )
    geom.notes.append(f"geometry from {meta_path}")
    return geom


def geometry_from_hf_config(config_path: Path, *, lora_r: int) -> ModelGeometry | None:
    """Heuristic geometry from a HF config.json (text config preferred)."""
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return None
    tc = cfg.get("text_config", cfg)
    hidden = tc.get("hidden_size")
    layers = tc.get("num_hidden_layers")
    vocab = tc.get("vocab_size")
    inter = tc.get("intermediate_size")
    if not all(isinstance(v, int) for v in (hidden, layers, vocab, inter)):
        return None
    tied = bool(tc.get("tie_word_embeddings", False))
    layer_types = tc.get("layer_types") or ["full_attention"] * layers
    n_linear = sum(1 for t in layer_types if "linear" in t)
    n_full = layers - n_linear

    # MLP per layer (gate+up+down)
    mlp_per_layer = 3 * hidden * inter
    # Full attention per layer: q (hidden->heads*head_dim*2 for gated Qwen3.5),
    # k/v (hidden->kv_heads*head_dim), o (heads*head_dim->hidden)
    heads = tc.get("num_attention_heads", 0)
    kv_heads = tc.get("num_key_value_heads", heads)
    head_dim = tc.get("head_dim", hidden // max(heads, 1))
    q_out = heads * head_dim * 2 if tc.get("attention_bias") is not None or True else heads * head_dim
    # Qwen3.5-style: q_proj outputs 2x (query + gate). Fall back gracefully:
    q_out = heads * head_dim * 2
    attn_per_layer = hidden * q_out + 2 * hidden * (kv_heads * head_dim) + (heads * head_dim) * hidden
    # Linear attention (Gated DeltaNet) per layer:
    # in_proj_qkv: hidden -> 2*k_dim + v_dim; in_proj_z: hidden -> v_dim;
    # in_proj_a/b: hidden -> num_v_heads; out_proj: v_dim -> hidden
    lk_heads = tc.get("linear_num_key_heads", 0)
    lv_heads = tc.get("linear_num_value_heads", 0)
    lk_dim = tc.get("linear_key_head_dim", 0)
    lv_dim = tc.get("linear_value_head_dim", 0)
    if lk_heads and lv_heads and lk_dim and lv_dim:
        k_dim, v_dim = lk_heads * lk_dim, lv_heads * lv_dim
        linear_per_layer = (
            hidden * (2 * k_dim + v_dim)  # in_proj_qkv
            + hidden * v_dim  # in_proj_z
            + hidden * lv_heads  # in_proj_a
            + hidden * lv_heads  # in_proj_b
            + v_dim * hidden  # out_proj
        )
        lora_units_linear = (hidden + 2 * k_dim + v_dim) + (hidden + v_dim) + (hidden + lv_heads) * 2 + (v_dim + hidden)
    else:
        linear_per_layer = 0
        lora_units_linear = 0
    lora_units_attn = (hidden + q_out) + 2 * (hidden + kv_heads * head_dim) + (heads * head_dim + hidden)
    lora_units_mlp = (hidden + inter) * 2 + (inter + hidden)

    vision = 0.0
    if "vision_config" in cfg and not cfg.get("language_model_only", False):
        vc = cfg["vision_config"]
        v_hidden = vc.get("hidden_size", 0)
        v_depth = vc.get("depth", 0)
        if v_hidden and v_depth:
            vision = v_depth * 4 * v_hidden * v_hidden * 4 / 1e9  # rough: attn+mlp 4x

    embed = vocab * hidden
    lm_head = 0.0 if tied else vocab * hidden
    core = n_full * (attn_per_layer + mlp_per_layer) + n_linear * (linear_per_layer + mlp_per_layer)
    total = embed + lm_head + core + vision * 1e9
    # Default skip list of typical upstream 4-bit checkpoints: in_proj_qkv.
    notes = ["geometry from config.json (heuristic)"]
    return ModelGeometry(
        name=str(cfg.get("_name_or_path", "unknown")),
        total_params=total,
        embed_params=embed,
        lm_head_params=lm_head,
        quantized_params=core * 0.9,  # most Linears quantized
        nonquant_16bit_params=core * 0.1 + n_linear * hidden * (2 * (lk_heads * lk_dim) + lv_heads * lv_dim) * 0.9,
        vision_params=vision * 1e9,
        lora_units=(n_full * (lora_units_attn + lora_units_mlp) + n_linear * (lora_units_linear + lora_units_mlp)),
        source="hf_config",
        notes=notes,
    )


@dataclass
class VramEstimate:
    weights_gb: float
    lora_states_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float
    total_gb: float
    trainable_params: float
    notes: list[str] = field(default_factory=list)

    def breakdown(self) -> dict[str, float]:
        return {
            "weights": self.weights_gb,
            "lora + optimizer": self.lora_states_gb,
            "activations": self.activations_gb,
            "logits spike": self.logits_gb,
            "CUDA/HIP context + misc": self.overhead_gb,
            "TOTAL": self.total_gb,
        }


def estimate_training_vram(
    geom: ModelGeometry,
    *,
    lora_r: int,
    max_length: int,
    batch_size: int,
    grad_checkpointing: bool = True,
    optimizer: str = "adamw_8bit",
    loss_type: str = "nll",
    compute_dtype_bytes: float = 2.0,
) -> VramEstimate:
    """Transparent VRAM estimate for one training process (one GPU).

    The formulas are deliberately simple and printed to the user; they are
    meant to be *directionally right and explainable*, while the dashboard
    reports the real numbers from the allocator during training.
    """
    gb = 1e9
    # --- weights -----------------------------------------------------------
    # NF4 + double quant ≈ 0.55 bytes/param; non-quantized 16-bit = 2 B/param.
    weights = (
        geom.quantized_params * 0.55
        + geom.nonquant_16bit_params * compute_dtype_bytes
        + geom.embed_params * compute_dtype_bytes
        + geom.lm_head_params * compute_dtype_bytes
        + geom.vision_params * compute_dtype_bytes
    ) / gb

    # --- LoRA + optimizer ----------------------------------------------------
    trainable = geom.lora_units * lora_r
    lora_weights = trainable * compute_dtype_bytes
    lora_grads = trainable * compute_dtype_bytes
    if optimizer == "adamw_8bit":
        opt_bytes = 2 * trainable * 1.0  # two 8-bit moments (+ blockwise scales ~)
        opt_bytes *= 1.1
    elif optimizer == "adafactor":
        opt_bytes = trainable * 4 * 0.5  # factored second moments
    else:
        opt_bytes = trainable * 4 * 2  # two fp32 moments
    lora_states = (lora_weights + lora_grads + opt_bytes) / gb

    # --- activations ---------------------------------------------------------
    # Stored activations scale with layers × batch × seq × hidden; gradient
    # checkpointing recomputes them, keeping only layer inputs (~25%).
    hidden = geom.hidden_size or 4096
    n_layers = geom.n_layers or max(1, int(geom.total_params / 4.3e8))
    per_layer_mb = batch_size * max_length * hidden * 2 / 1e6
    stored = n_layers * per_layer_mb * (0.25 if grad_checkpointing else 1.0)
    transient = per_layer_mb * 2.5  # recompute / MLP / attention peaks
    activations = (stored + transient) / 1024

    # --- logits --------------------------------------------------------------
    # TRL 1.16 computes every SFT loss through a fused, chunked LM head, so
    # full fp32 logits are never materialized — `nll` and `dft` alike.
    logits = 0.0

    overhead = 1.0  # HIP context, RCCL buffers, allocator fragmentation, fragmentation margin
    total = weights + lora_states + activations + logits + overhead
    return VramEstimate(
        weights_gb=weights,
        lora_states_gb=lora_states,
        activations_gb=activations,
        logits_gb=logits,
        overhead_gb=overhead,
        total_gb=total,
        trainable_params=trainable,
        notes=[
            f"trainable ≈ {trainable / 1e6:.1f}M params (r={lora_r}, all-linear)",
            f"gradient checkpointing: {'on' if grad_checkpointing else 'off'}",
            f"loss: {loss_type}",
        ],
    )


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

    # Nothing fits: report the best candidates and the holders.
    sorted(free, key=lambda g: -free[g])[:1]
    holders = gpu_processes()
    for g in free:
        who = ", ".join(f"{p['name']}(pid {p['pid']})" for p in holders if str(g) in str(p["gpus"]))
        warnings.append(
            f"GPU {g}: {free[g]:.1f} GB free of {snap[g].mem_total_b / 1e9:.0f} GB"
            + (f" — held by: {who}" if who else "")
        )
    raise RuntimeError(
        f"No GPU has {need:.1f} GB free VRAM (need = estimate {needed_gb:.1f} GB × "
        f"safety {safety_factor:.2f}).\n" + "\n".join(warnings) + "\n"
        "Options: free VRAM (stop other GPU processes), lower data.max_length or "
        "optim.per_device_batch_size, use fewer/other GPUs via dist.gpus, or use "
        "dist.mode = 'fsdp' to shard the model across GPUs."
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


def probe_sdpa(python_exe: str | None = None, timeout: float = 300.0) -> dict[str, Any]:
    """Empirically test SDPA backends on THIS machine.

    Runs each backend in a fresh process (deferred launch errors on RDNA3
    contaminate the process state — see pytorch issue #194498), with the
    AOTriton experimental flag ON (needed to enable fused attention on some
    gfx1100 wheels) and OFF, and reports what actually launches.
    """
    python_exe = python_exe or sys.executable
    results: dict[str, Any] = {"backends": {}, "verdict": "unknown", "attn_implementation": "sdpa"}

    for env_mode in ("on", "off"):
        env = dict(os.environ)
        env.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8")
        if env_mode == "on":
            env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        else:
            env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
        for backend in ("flash", "efficient", "math"):
            key = f"{backend}_{env_mode}"
            try:
                proc = subprocess.run(
                    [python_exe, "-c", _SDPA_PROBE, backend],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                results["backends"][key] = (
                    json.loads(line[-1])
                    if line
                    else {
                        "ok": False,
                        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1],
                    }
                )
            except subprocess.TimeoutExpired:
                results["backends"][key] = {"ok": False, "error": "timeout"}
            except Exception as exc:
                results["backends"][key] = {"ok": False, "error": str(exc)}

    flash_on = results["backends"].get("flash_on", {}).get("ok", False)
    flash_off = results["backends"].get("flash_off", {}).get("ok", False)
    eff_on = results["backends"].get("efficient_on", {}).get("ok", False)
    math_ok = results["backends"].get("math_on", {}).get("ok", False) or results["backends"].get("math_off", {}).get(
        "ok", False
    )

    if flash_on or eff_on:
        results["verdict"] = "aotriton_on"
        results["attn_implementation"] = "sdpa"
    elif flash_off:
        results["verdict"] = "aotriton_off"
        results["attn_implementation"] = "sdpa"
    elif math_ok:
        results["verdict"] = "math_only"
        results["attn_implementation"] = "eager"
    else:
        results["verdict"] = "all_failed"
        results["attn_implementation"] = "eager"
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


def probe_hip_ipc(python_exe: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    """Can HIP export IPC memory handles in this environment?

    RCCL's P2P transport needs `hipIpcGetMemHandle`. The ROCm 7.1 runtime
    defaults to *legacy* IPC mode, whose export ioctl exists only in the
    out-of-tree DKMS driver — on mainline kernels it fails unless
    `HSA_ENABLE_IPC_MODE_LEGACY=0` selects the dma-buf path. This probe runs
    in a fresh subprocess (the flag must be set before HIP initializes) with
    the flag applied, so `doctor` catches any future stack change that breaks
    the P2P transport again.
    """
    import sys as _sys

    env = dict(os.environ)
    env["HSA_ENABLE_IPC_MODE_LEGACY"] = "0"
    try:
        proc = subprocess.run(
            [python_exe or _sys.executable, "-c", _HIP_IPC_PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        return (
            json.loads(line[-1])
            if line
            else {
                "ipc_ok": False,
                "error": (proc.stderr.strip().splitlines() or ["no output"])[-1],
            }
        )
    except subprocess.TimeoutExpired:
        return {"ipc_ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ipc_ok": False, "error": str(exc)}


def matmul_benchmark(
    dtype_name: str = "bfloat16", size: int = 8192, iters: int = 20, warmup: int = 5
) -> dict[str, Any]:
    """bf16/fp16 matmul TFLOPS on GPU 0 — a quick sanity check of the stack."""
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    device = "cuda"
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = 2 * size**3 / dt / 1e12
    return {"dtype": dtype_name, "size": size, "seconds": dt, "tflops": tflops}


def disk_free_gb(path: str | Path) -> float:
    import shutil

    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:
        return float("nan")
