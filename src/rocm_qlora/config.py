"""Configuration: TOML -> validated dataclasses -> snapshots.

Design notes
------------
* **Strict parsing**: unknown TOML keys raise errors (typos in a 100-line
  config are otherwise silent and expensive). The error message lists the
  offending key and its location.
* **Single source of truth**: every knob the trainer uses is a field here.
  The resolved config is snapshotted into the run directory as both TOML
  (human-readable) and JSON (machine-readable), so a run is reproducible
  from its own output folder.
* **CLI overrides** are applied on top of the file (see ``with_overrides``),
  and *validation always runs last*.

The comments in ``configs/*.toml`` are the user-facing documentation; this
module is the contract.
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .utils import write_json, write_text

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    name: str = "qlora-run"
    seed: int = 42
    output_root: str = ".outputs"
    explainers: bool = True
    console_refresh_per_second: float = 4.0


@dataclass
class ModelConfig:
    #: Hugging Face id of the *base* model (used by `prepare` when a prepared
    #: checkpoint does not exist yet).
    name: str = "Qwen/Qwen3.8-27B"
    #: Local directory with the prepared (NF4) checkpoint. `train` loads from
    #: here; `prepare` writes here.
    prepared_dir: str = ".data/models/qwen3.8-27b-nf4"
    #: "auto" lets `doctor`'s probe decide; else "sdpa" | "eager".
    attn: str = "auto"
    dtype: str = "bfloat16"  # compute dtype for quantized matmuls
    trust_remote_code: bool = False


@dataclass
class QuantizationConfig:
    quant_type: str = "nf4"  # "nf4" | "fp4"
    double_quant: bool = True
    quant_storage: str = "uint8"  # required for FSDP/DDP shardability
    #: Re-quantize the (unused) vision tower's Linears during `prepare`.
    requantize_vision: bool = True
    #: Module name suffixes to (re-)quantize in `prepare` if the upstream
    #: checkpoint left them in 16-bit. [] disables surgery. NOTE: `lm_head`
    #: must stay 16-bit — TRL's chunked CE slices the raw weight tensor.
    requantize_modules: list[str] = field(default_factory=lambda: ["in_proj_qkv"])


@dataclass
class DataConfig:
    path: str = ".data/raw/dataset.json"
    messages_field: str = "messages"
    eval_fraction: float = 0.05
    #: Sequence cap. Set from the data profile (p95–p99), not the model window.
    max_length: int = 1024
    truncation_mode: str = "keep_start"
    #: See explainer "packing" — correctness hazard without document-aware
    #: attention; off by default.
    packing: bool = False
    packing_strategy: str = "bfd_split"
    #: Loss only on assistant turns (chat template must expose assistant spans;
    #: TRL patches known families automatically).
    assistant_only_loss: bool = True
    dataset_num_proc: int = 8


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: "all-linear" adapts every linear layer except the output head — robust
    #: across architectures (attention + MLP + linear-attention projections).
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"


@dataclass
class OptimConfig:
    lr: float = 1e-4
    epochs: float = 3.0
    per_device_batch_size: int = 1
    grad_accum: int = 4
    max_grad_norm: float = 0.3  # QLoRA paper value
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    gradient_checkpointing: bool = True
    #: "nll" (standard) | "dft" (dynamic fine-tuning, up-weights uncertain
    #: tokens). TRL computes both through a fused, chunked kernel that never
    #: materializes full logits.
    loss_type: str = "nll"


@dataclass
class EvalConfig:
    per_epoch: bool = True
    per_device_batch_size: int = 1
    eval_packing: bool = False


@dataclass
class LoggingConfig:
    #: Regenerate the live charts (loss + dashboard) every N optimizer steps.
    chart_snapshot_steps: int = 10
    #: rocm-smi sampling period for the GPU health panel (seconds).
    gpu_sample_seconds: float = 5.0
    #: Keep per-epoch adapter checkpoints in addition to the final one.
    keep_epoch_checkpoints: bool = True
    #: Write a JSONL record for every optimizer step (cheap; enables
    #: post-hoc analysis and chart regeneration).
    metrics_every_step: bool = True


@dataclass
class EnvConfig:
    #: HIP allocator settings. garbage_collection_threshold keeps fragmentation
    #: in check without expandable_segments (which has a contested track record
    #: on RDNA3 — see docs/02-amd-rocm-notes.md).
    hip_alloc_conf: str = "garbage_collection_threshold:0.8"
    #: "auto" -> use `doctor`'s SDPA verdict; "on"/"off" force the AOTriton env.
    aotriton: str = "auto"
    #: Threads per rank. 3 ranks × 8 threads ≪ 64 cores — avoids oversubscription.
    omp_num_threads: int = 8
    #: Extra environment variables for the training workers (string→string).
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class DistConfig:
    #: "auto" (all GPUs that fit) | "ddp" (same) | "single" (first GPU)
    mode: str = "auto"
    #: "auto" picks the GPUs with enough free VRAM; else e.g. [0, 2].
    gpus: str | list[int] = "auto"
    #: Required free VRAM = estimate × safety_factor.
    safety_factor: float = 1.10


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    dist: DistConfig = field(default_factory=DistConfig)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def snapshot(self, run_dir: str | Path) -> None:
        """Write resolved config into the run directory (TOML + JSON)."""
        run_dir = Path(run_dir)
        write_json(run_dir / "config.resolved.json", self.to_dict())
        write_text(run_dir / "config.resolved.toml", dumps_toml(self.to_dict()))

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        cfg = _build(cls, raw, prefix="")
        errors = cfg.validate()
        if errors:
            raise ConfigError("\n".join(errors))
        return cfg

    # -- overrides ----------------------------------------------------------

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """Apply CLI overrides given as {"optim.lr": 2e-4, "dist.gpus": [0,1]}.

        Overrides are merged *before* validation, so an invalid override fails
        with the same strictness as an invalid file.
        """
        merged = self.to_dict()
        for dotted, value in overrides.items():
            if value is None:
                continue
            node: dict[str, Any] = merged
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise ConfigError(f"Override target '{dotted}' does not exist")
                node = node[part]
            if parts[-1] not in node:
                raise ConfigError(f"Override target '{dotted}' does not exist")
            node[parts[-1]] = value
        return Config.from_dict(merged)

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        check(self.run.seed >= 0, "run.seed must be >= 0")
        check(self.run.console_refresh_per_second > 0, "run.console_refresh_per_second must be > 0")
        check(self.model.dtype in {"bfloat16", "float16"}, "model.dtype must be bfloat16 or float16")
        check(self.model.attn in {"auto", "sdpa", "eager"}, "model.attn must be auto|sdpa|eager")
        check(
            self.quantization.quant_type in {"nf4", "fp4"},
            "quantization.quant_type must be nf4 or fp4",
        )
        check(
            self.quantization.quant_storage in {"uint8", "bfloat16", "float16", "float32"},
            "quantization.quant_storage must be one of uint8|bfloat16|float16|float32",
        )
        check(0.0 <= self.data.eval_fraction < 0.5, "data.eval_fraction must be in [0, 0.5)")
        check(128 <= self.data.max_length <= 32768, "data.max_length must be in [128, 32768]")
        check(
            self.data.truncation_mode in {"keep_start", "keep_end"},
            "data.truncation_mode must be keep_start|keep_end",
        )
        check(
            self.data.packing_strategy in {"bfd", "bfd_split", "wrapped"},
            "data.packing_strategy must be bfd|bfd_split|wrapped",
        )
        check(1 <= self.lora.r <= 512, "lora.r must be in [1, 512]")
        check(self.lora.alpha > 0, "lora.alpha must be > 0")
        check(0.0 <= self.lora.dropout < 1.0, "lora.dropout must be in [0, 1)")
        check(self.lora.bias in {"none", "all", "lora_only"}, "lora.bias must be none|all|lora_only")
        if isinstance(self.lora.target_modules, list):
            check(len(self.lora.target_modules) > 0, "lora.target_modules list must not be empty")
        check(0.0 < self.optim.lr <= 1e-2, "optim.lr must be in (0, 1e-2]")
        check(self.optim.epochs > 0, "optim.epochs must be > 0")
        check(self.optim.per_device_batch_size >= 1, "optim.per_device_batch_size must be >= 1")
        check(self.optim.grad_accum >= 1, "optim.grad_accum must be >= 1")
        check(self.optim.max_grad_norm > 0, "optim.max_grad_norm must be > 0")
        check(
            self.optim.optimizer in {"adamw_torch", "adamw_8bit", "adafactor"},
            "optim.optimizer must be adamw_torch|adamw_8bit|adafactor "
            "(paged optimizers are CUDA-only and unsupported on ROCm)",
        )
        check(
            self.optim.scheduler in {"cosine", "linear", "constant", "cosine_with_restarts"},
            "optim.scheduler must be cosine|linear|constant|cosine_with_restarts",
        )
        check(0.0 <= self.optim.warmup_ratio < 0.5, "optim.warmup_ratio must be in [0, 0.5)")
        check(
            self.optim.loss_type in {"nll", "dft"},
            "optim.loss_type must be nll|dft",
        )
        if "lm_head" in self.quantization.requantize_modules:
            errors.append(
                "quantization.requantize_modules contains 'lm_head', which cannot be quantized: TRL 1.16 "
                "computes every SFT loss through a fused chunked head that slices the raw lm_head weight "
                "tensor (no full logits are materialized). The head must stay in 16-bit."
            )
        check(self.eval.per_device_batch_size >= 1, "eval.per_device_batch_size must be >= 1")
        check(self.logging.chart_snapshot_steps >= 1, "logging.chart_snapshot_steps must be >= 1")
        check(self.logging.gpu_sample_seconds >= 1.0, "logging.gpu_sample_seconds must be >= 1.0")
        check(self.env.aotriton in {"auto", "on", "off"}, "env.aotriton must be auto|on|off")
        check(self.env.omp_num_threads >= 1, "env.omp_num_threads must be >= 1")
        check(self.dist.mode in {"auto", "ddp", "single"}, "dist.mode must be auto|ddp|single")
        if isinstance(self.dist.gpus, list):
            check(len(self.dist.gpus) > 0, "dist.gpus list must not be empty")
            check(all(g >= 0 for g in self.dist.gpus), "dist.gpus ids must be >= 0")
        check(self.dist.safety_factor >= 1.0, "dist.safety_factor must be >= 1.0")
        return errors


class ConfigError(Exception):
    """Raised for malformed or invalid configuration (message is user-facing)."""


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build(cls: type, raw: Mapping[str, Any], prefix: str) -> Any:
    """Recursively construct dataclasses from dicts, rejecting unknown keys."""
    from typing import get_type_hints

    if not isinstance(raw, Mapping):
        raise ConfigError(f"Expected a table at '{prefix or '<root>'}', got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)  # resolves string annotations (future import)
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{prefix or '<root>'}': {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        child_prefix = f"{prefix}.{name}" if prefix else name
        field_type = hints.get(name)
        if field_type is not None and is_dataclass(field_type) and isinstance(value, Mapping):
            kwargs[name] = _build(field_type, value, child_prefix)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Minimal deterministic TOML writer for snapshots (flat-ish structures)."""
    lines: list[str] = []

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if v is None:
            return '""  # null'
        return repr(v)

    def emit_table(prefix: str, node: Mapping[str, Any]) -> None:
        scalars = {k: v for k, v in node.items() if not isinstance(v, Mapping)}
        tables = {k: v for k, v in node.items() if isinstance(v, Mapping)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {fmt(v)}")
        lines.append("")
        for k, v in tables.items():
            emit_table(f"{prefix}.{k}" if prefix else k, v)

    emit_table("", data)
    return "\n".join(lines).strip() + "\n"
