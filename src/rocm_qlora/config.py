"""Configuration: TOML -> validated pydantic models -> snapshots.

Validation is the library's job: unknown keys are rejected, types are
enforced, and ranges/literal sets live next to the fields that own them.
The resolved config is snapshotted into the run directory as
``config.resolved.toml``, so a run is reproducible from its own output
folder. CLI overrides are applied on top of the file, and validation always
runs last.

The comments in ``configs/*.toml`` are the user-facing documentation; this
module is the contract.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .utils import write_text


class ConfigError(Exception):
    """Raised for malformed or invalid configuration (message is user-facing)."""


class _Section(BaseModel):
    """Base for every config section: unknown keys are typos, not extensions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class RunConfig(_Section):
    #: Run identifier used in run-dir names and chart titles.
    name: str = "qlora-run"
    #: Global seed: data order, LoRA init and hash randomization.
    seed: Annotated[int, Field(ge=0)] = 42
    #: Root directory for run outputs (runs land in <root>/runs/<ts>-<name>).
    output_root: str = ".outputs"


class ModelConfig(_Section):
    #: Hugging Face id of the *base* model (used by `prepare` when a prepared
    #: checkpoint does not exist yet).
    name: str = "Qwen/Qwen3.8-27B"
    #: Local directory with the prepared (NF4) checkpoint. `train` loads from
    #: here; `prepare` writes here.
    prepared_dir: str = ".data/models/qwen3.8-27b-nf4"
    #: "auto" lets `doctor`'s probe decide; else "sdpa" | "eager".
    attn: Literal["auto", "sdpa", "eager"] = "auto"
    dtype: Literal["bfloat16", "float16"] = "bfloat16"  # compute dtype for quantized matmuls
    #: Allow custom modeling code from the hub (off by default).
    trust_remote_code: bool = False


class QuantizationConfig(_Section):
    quant_type: Literal["nf4", "fp4"] = "nf4"
    #: Quantize the quantization constants too (~1.1 GB saved at 27B: fp32
    #: absmax per 64-param block would cost 1.5 GB, 8-bit nested costs 0.4).
    double_quant: bool = True
    #: Storage dtype of the 4-bit weights (uint8 required for FSDP/DDP sharding).
    quant_storage: Literal["uint8", "bfloat16", "float16", "float32"] = "uint8"
    #: Re-quantize the (unused) vision tower's Linears during `prepare`.
    requantize_vision: bool = True
    #: Module name suffixes to (re-)quantize in `prepare` if the upstream
    #: checkpoint left them in 16-bit. [] disables surgery. NOTE: `lm_head`
    #: must stay 16-bit — TRL's chunked CE slices the raw weight tensor.
    requantize_modules: list[str] = Field(default_factory=lambda: ["in_proj_qkv"])

    @field_validator("requantize_modules")
    @classmethod
    def _reject_lm_head(cls, value: list[str]) -> list[str]:
        if "lm_head" in value:
            raise ValueError(
                "quantization.requantize_modules contains 'lm_head', which cannot be quantized: TRL "
                "computes every SFT loss through a fused chunked head that slices the raw lm_head weight "
                "tensor (no full logits are materialized). The head must stay in 16-bit."
            )
        return value


class DataConfig(_Section):
    #: ShareGPT-style JSON file (list of {messages: [...]}).
    path: str = ".data/raw/dataset.json"
    #: Key holding the conversation turns in each example.
    messages_field: str = "messages"
    #: Fraction held out for eval ([0, 0.5); tiny is fine).
    eval_fraction: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.05
    #: Sequence cap. Set from the data profile (p95–p99), not the model window.
    max_length: Annotated[int, Field(ge=128, le=32768)] = 1024
    #: Only keep_start is accepted (TRL 1.16 deprecated keep_end; removed in TRL 2).
    truncation_mode: Literal["keep_start"] = "keep_start"
    #: Correctness hazard without document-aware attention; off by default.
    packing: bool = False
    #: Sequence packing strategy when data.packing is on.
    packing_strategy: Literal["bfd", "bfd_split", "wrapped"] = "bfd_split"
    #: Loss only on assistant turns (chat template must expose assistant spans;
    #: TRL patches known families automatically).
    assistant_only_loss: bool = True
    #: Parallel dataset-preprocessing processes.
    dataset_num_proc: int = 8


class LoraConfig(_Section):
    #: LoRA rank (capacity vs VRAM).
    r: Annotated[int, Field(ge=1, le=512)] = 16
    #: LoRA scaling factor (alpha/r is the effective scale).
    alpha: Annotated[int, Field(gt=0)] = 32
    #: LoRA dropout.
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.05
    #: "all-linear" adapts every linear layer except the output head — robust
    #: across architectures (attention + MLP + linear-attention projections).
    target_modules: str | list[str] = "all-linear"
    #: Which biases to train (none | all | lora_only).
    bias: Literal["none", "all", "lora_only"] = "none"

    @field_validator("target_modules")
    @classmethod
    def _non_empty_list(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("lora.target_modules list must not be empty")
        return value


class OptimConfig(_Section):
    #: Peak learning rate (schedule-decayed).
    lr: Annotated[float, Field(gt=0, le=1e-2)] = 1e-4
    #: Training epochs.
    epochs: Annotated[float, Field(gt=0)] = 3.0
    #: Micro-batch per GPU (1 for 27B on 24 GB).
    per_device_batch_size: Annotated[int, Field(ge=1)] = 1
    #: Gradient accumulation (effective batch = bsz × accum × GPUs).
    grad_accum: Annotated[int, Field(ge=1)] = 4
    max_grad_norm: Annotated[float, Field(gt=0)] = 0.3  # QLoRA paper value
    #: AdamW weight decay.
    weight_decay: float = 0.01
    #: adamw_torch | adamw_8bit | adafactor (paged is CUDA-only).
    optimizer: str = "adamw_8bit"
    #: LR schedule (cosine | linear | constant | cosine_with_restarts).
    scheduler: Literal["cosine", "linear", "constant", "cosine_with_restarts"] = "cosine"
    #: Fraction of steps spent warming up.
    warmup_ratio: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.03
    #: Recompute activations to save VRAM (slower).
    gradient_checkpointing: bool = True
    #: Keep checkpointed activations in pinned host RAM (frees device memory,
    #: costs a device-to-host copy in the forward and back in the backward).
    activation_offloading: bool = False
    #: Checkpoint every Nth layer only (1 = all layers). >1 trades memory back
    #: for speed when full checkpointing frees more than the run needs.
    checkpoint_every_n_layers: Annotated[int, Field(ge=1, le=8)] = 1
    #: "nll" (standard) | "dft" (dynamic fine-tuning, up-weights uncertain
    #: tokens). TRL computes both through a fused, chunked kernel that never
    #: materializes full logits.
    loss_type: Literal["nll", "dft"] = "nll"

    @field_validator("optimizer")
    @classmethod
    def _known_optimizer(cls, value: str) -> str:
        if value not in {"adamw_torch", "adamw_8bit", "adafactor"}:
            raise ValueError(
                "optim.optimizer must be adamw_torch|adamw_8bit|adafactor "
                "(paged optimizers are CUDA-only and unsupported on ROCm)"
            )
        return value


class EvalConfig(_Section):
    #: Run eval at every epoch boundary.
    per_epoch: bool = True
    #: Eval micro-batch per GPU.
    per_device_batch_size: Annotated[int, Field(ge=1)] = 1
    #: Pack eval sequences too (faster eval).
    eval_packing: bool = False


class LoggingConfig(_Section):
    #: Regenerate the live charts (loss + overview) every N optimizer steps.
    chart_snapshot_steps: Annotated[int, Field(ge=1)] = 10
    #: rocm-smi sampling period for the GPU health records (seconds).
    gpu_sample_seconds: Annotated[float, Field(ge=1.0)] = 5.0
    #: Keep older checkpoints instead of only the latest (save_total_limit=None).
    keep_epoch_checkpoints: bool = True
    #: Switch checkpointing to step cadence: a full checkpoint (adapter +
    #: optimizer/scheduler state) every N optimizer steps — resilience for long
    #: runs on flaky hardware. 0 keeps the epoch-only cadence.
    checkpoint_every_n_steps: Annotated[int, Field(ge=0)] = 0


class EnvConfig(_Section):
    #: HIP allocator settings. Do NOT enable `expandable_segments` on this
    #: stack: tried 2026-10-23 and it corrupted numerics (step-1 loss shift +
    #: NaN grads on an identical batch — see docs/05). The GC threshold alone
    #: keeps fragmentation in check; headroom is bought with batch size.
    hip_alloc_conf: str = "garbage_collection_threshold:0.8"
    #: "auto" -> use `doctor`'s SDPA verdict; "on"/"off" force the AOTriton env.
    aotriton: Literal["auto", "on", "off"] = "auto"
    #: Threads per rank. A handful per rank avoids host oversubscription.
    omp_num_threads: Annotated[int, Field(ge=1)] = 8
    #: Extra environment variables for the training workers (string→string).
    extra: dict[str, str] = Field(default_factory=dict)


class DistConfig(_Section):
    #: "auto" (every visible GPU) | "ddp" (same) | "single" (first GPU)
    mode: Literal["auto", "ddp", "single"] = "auto"
    #: "auto" uses every visible GPU; else e.g. [0, 2].
    gpus: str | list[int] = "auto"

    @field_validator("gpus")
    @classmethod
    def _valid_gpu_list(cls, value: str | list[int]) -> str | list[int]:
        if isinstance(value, list):
            if not value:
                raise ValueError("dist.gpus list must not be empty")
            if any(gpu < 0 for gpu in value):
                raise ValueError("dist.gpus ids must be >= 0")
        return value


class Config(_Section):
    """Full resolved configuration: one model per TOML section."""

    run: RunConfig = Field(default_factory=lambda: RunConfig())
    model: ModelConfig = Field(default_factory=lambda: ModelConfig())
    quantization: QuantizationConfig = Field(default_factory=lambda: QuantizationConfig())
    data: DataConfig = Field(default_factory=lambda: DataConfig())
    lora: LoraConfig = Field(default_factory=lambda: LoraConfig())
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig())
    eval: EvalConfig = Field(default_factory=lambda: EvalConfig())
    logging: LoggingConfig = Field(default_factory=lambda: LoggingConfig())
    env: EnvConfig = Field(default_factory=lambda: EnvConfig())
    dist: DistConfig = Field(default_factory=lambda: DistConfig())

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Config as a nested plain dict (TOML-shaped)."""
        return self.model_dump()

    def snapshot(self, run_dir: str | Path) -> None:
        """Write the resolved config into the run directory (config.resolved.toml)."""
        run_dir = Path(run_dir)
        write_text(run_dir / "config.resolved.toml", tomli_w.dumps(self.to_dict()))

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and validate a TOML config file."""
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        """Build a Config from a nested dict (e.g. a run's snapshot)."""
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError("\n".join(_format_errors(exc))) from exc

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


def _format_errors(exc: ValidationError) -> list[str]:
    """One user-facing line per pydantic error (loc path + message)."""
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        if error["type"] == "extra_forbidden":
            section, _, key = loc.rpartition(".")
            lines.append(f"Unknown key in '{section or '<root>'}': '{key}'")
        else:
            lines.append(f"{loc or '<root>'}: {error['msg']}")
    return lines
