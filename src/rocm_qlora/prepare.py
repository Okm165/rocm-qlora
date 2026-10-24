"""Checkpoint preparation: NF4 surgery on a source model, once per base model.

`prepare` takes a community or plain checkpoint and makes it cheap to load:

    load the source into host RAM (fp16 — it is far larger than one card)
    re-quantize the Linears the upstream left in 16-bit, on the freest GPU
    write a self-describing quantization config (skip list from reality)
    save -> .data/models/<name>-nf4 + finetune_meta.json (exact accounting)

Why this step exists (docs/01 + docs/02): loading 4-bit from a bf16
checkpoint stages the whole bf16 model in host RAM first — ~55 GB for a 27B
model, times the number of data-parallel processes. Preparing once removes
that wall. The runtime half (4-bit load, LoRA, accounting) is `model.py`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from .config import Config
from .hardware import gpu_snapshot
from .model import compute_dtype, load_tokenizer, logical_numel, quant_storage_dtype, resolve_model_class
from .utils import now_iso, write_json

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Preparation (download + surgery + save)
# ---------------------------------------------------------------------------


class FinetuneMeta(TypedDict):
    """`finetune_meta.json` in a prepared checkpoint: what `prepare` wrote.

    ``total_params`` is exact (counted from the tensors); the rest is
    provenance. The quantization settings are not duplicated here — the
    checkpoint's ``config.json`` carries them. Older checkpoints may miss
    keys — readers stay tolerant.
    """

    total_params: float
    source: NotRequired[str]
    requantized: NotRequired[list[str]]
    created: NotRequired[str]


@dataclass
class PrepareResult:
    out_dir: Path
    requantized: list[str]
    seconds: float
    total_params: float
    disk_gb: float


def _requantize_linear(module: nn.Linear, cfg: Config):
    """Convert an fp16 nn.Linear into bnb Linear4bit (NF4), preserving math.

    The source module must already live on the target GPU: moving the new
    module triggers bitsandbytes' lazy quantization there.
    """
    import bitsandbytes as bnb
    import torch

    new = bnb.nn.Linear4bit(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,  # pyright: ignore[reportUnnecessaryComparison]  # biasless Linears
        compute_dtype=compute_dtype(cfg),
        compress_statistics=cfg.quantization.double_quant,
        quant_type=cfg.quantization.quant_type,
        quant_storage=quant_storage_dtype(cfg),
    )
    weight = module.weight.data
    new.weight = bnb.nn.Params4bit(
        weight,
        requires_grad=False,
        quant_type=cfg.quantization.quant_type,  # pyright: ignore[reportCallIssue]  # bnb stubs omit 4-bit params
        compress_statistics=cfg.quantization.double_quant,  # pyright: ignore[reportCallIssue]
        quant_storage=quant_storage_dtype(cfg),  # pyright: ignore[reportCallIssue]
    )
    if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]  # biasless Linears exist
        new.bias = torch.nn.Parameter(module.bias.data.clone(), requires_grad=False)
    return new.to(weight.device)


def parent_of(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _match_requantize(name: str, cfg: Config) -> bool:
    qc = cfg.quantization
    if qc.requantize_vision and ".visual." in name:
        return True
    return any(name.endswith(suffix) for suffix in qc.requantize_modules)


def prepare_checkpoint(cfg: Config, *, force: bool = False, log: Callable[[str], None] = print) -> PrepareResult:
    """Create the prepared NF4 checkpoint at ``cfg.model.prepared_dir``."""
    out_dir = Path(cfg.model.prepared_dir)
    meta_path = out_dir / "finetune_meta.json"
    cached = _cached_prepare_result(out_dir, meta_path, force=force)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    model = _load_source_model(cfg, log)
    # Two kinds of sources: already-quantized checkpoints (e.g. unsloth
    # bnb-4bit) need surgery only where the upstream left 16-bit modules;
    # plain bf16/fp16 checkpoints get every Linear quantized.
    source_quantized = getattr(model.config, "quantization_config", None) is not None
    requantized = _requantize_all(model, cfg, source_quantized=source_quantized, log=log)
    _write_quantization_config(model, cfg)

    log(f"Saving prepared checkpoint to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_prepared(model, out_dir)
    load_tokenizer(cfg).save_pretrained(out_dir)

    meta = _prepare_meta(model, cfg, requantized)
    write_json(meta_path, meta)
    seconds = time.perf_counter() - t0
    log(f"Prepared checkpoint ready ({_dir_size_gb(out_dir):.1f} GB on disk, {seconds / 60:.1f} min)")
    return PrepareResult(
        out_dir=out_dir,
        requantized=requantized,
        seconds=seconds,
        total_params=meta["total_params"],
        disk_gb=_dir_size_gb(out_dir),
    )


def _cached_prepare_result(out_dir: Path, meta_path: Path, *, force: bool) -> PrepareResult | None:
    """Return the cached result when a complete prepared dir exists (and not forced)."""
    if force or not out_dir.exists() or not (out_dir / "config.json").exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return PrepareResult(
        out_dir=out_dir,
        requantized=meta.get("requantized", []),
        seconds=0.0,
        total_params=float(meta.get("total_params", 0.0)),
        disk_gb=_dir_size_gb(out_dir),
    )


def _load_source_model(cfg: Config, log: Callable[[str], None]):
    """Load the source checkpoint into host RAM in fp16.

    Host staging, not a CPU execution mode: the checkpoint is far too large
    for one card, so it lives in host RAM and only each module's quantization
    hops onto a GPU (see `_requantize_all`).
    """
    import torch

    log(f"Loading source checkpoint {cfg.model.name!r} into host RAM (fp16)…")
    model_cls = resolve_model_class(cfg.model.name)
    model = model_cls.from_pretrained(
        cfg.model.name,
        dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.eval()
    return model


def _should_quantize(name: str, module: nn.Module, cfg: Config, *, source_quantized: bool) -> bool:
    """Whether one module needs NF4 surgery (lm_head always stays 16-bit)."""
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    import bitsandbytes as bnb

    if isinstance(module, bnb.nn.Linear4bit):
        return False
    if name.endswith("lm_head"):
        # The output head always stays 16-bit: TRL 1.16 computes the SFT
        # loss through a fused chunked head that slices the raw weight
        # tensor — a quantized head (flattened uint8 buffer) breaks it.
        return False
    if source_quantized:
        return _match_requantize(name, cfg)
    return True  # plain source: quantize all remaining Linears


def _requantize_all(model: nn.Module, cfg: Config, *, source_quantized: bool, log: Callable[[str], None]) -> list[str]:
    """Quantize every module that needs it; returns the requantized names."""
    requantized: list[str] = []
    if not (cfg.quantization.requantize_modules or cfg.quantization.requantize_vision or not source_quantized):
        return requantized
    # The quantization runs on the freest GPU (this is a GPU-only tool); the
    # model itself stays host-resident because it is far larger than one card.
    device = _surgery_device()
    for name, module in list(model.named_modules()):
        if not _should_quantize(name, module, cfg, source_quantized=source_quantized):
            continue
        parent, attr = parent_of(model, name)
        new = _requantize_linear(cast("nn.Linear", module.to(device)), cfg)
        setattr(parent, attr, new.to("cpu"))  # back to host RAM for the save
        requantized.append(name)
    log(f"Re-quantized {len(requantized)} module(s) to NF4")
    return requantized


def _write_quantization_config(model: PreTrainedModel, cfg: Config) -> None:
    """Attach the load-time-authoritative bnb config.

    The stored config is authoritative at load time (it lists exactly which
    modules are NOT quantized), so derive the skip list from *reality*:
    every plain Linear that survived surgery, by its module suffix. The dict
    comes from `BitsAndBytesConfig` itself, so a key this project misspells
    fails here instead of being silently ignored at load time.
    """
    import bitsandbytes as bnb
    import torch
    from transformers import BitsAndBytesConfig

    remaining = sorted(
        {
            name.split(".")[-1]
            for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and not isinstance(m, bnb.nn.Linear4bit)
        }
    )
    model.config.quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        bnb_4bit_compute_dtype=cfg.model.dtype,
        bnb_4bit_quant_storage=cfg.quantization.quant_storage,
        llm_int8_skip_modules=remaining,
    ).to_dict()


def _prepare_meta(model: PreTrainedModel, cfg: Config, requantized: list[str]) -> FinetuneMeta:
    """What ``prepare`` did to this checkpoint: exact total + provenance.

    ``named_parameters()`` deduplicates shared tensors, so a tied
    embed/lm_head weight is counted once. The quantization settings are not
    duplicated here — the checkpoint's own ``config.json`` carries them and is
    authoritative at load time.
    """
    return {
        "total_params": float(sum(logical_numel(p) for p in model.parameters())),
        "source": cfg.model.name,
        "requantized": requantized,
        "created": now_iso(),
    }


def _save_prepared(model: PreTrainedModel, out_dir: Path) -> None:
    """Save the prepared checkpoint.

    ``save_original_format=False``: the weights are already in their final
    format (NF4 + the 16-bit leftovers), so transformers must not try to
    revert the load-time weight conversions — some VLM checkpoints carry
    vision-side conversions without a reverse, which fails the plain save.
    """
    model.save_pretrained(out_dir, max_shard_size="4GB", save_original_format=False)


def _surgery_device() -> str:
    """The freest visible GPU for the quantization surgery.

    ``prepare`` is a GPU tool: quantizing on the host CPU is not supported, so
    a machine where rocm-smi reports no GPUs fails here loudly.
    """

    snap = gpu_snapshot()
    if not snap:
        raise RuntimeError(
            "prepare needs a visible GPU for the quantization surgery "
            "(rocm-smi reported none — run `rocm-qlora doctor` to see why)"
        )
    best = max(snap, key=lambda g: g.mem_free_b)
    return f"cuda:{best.index}"


def _dir_size_gb(path: Path) -> float:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
    except Exception:
        return float("nan")
