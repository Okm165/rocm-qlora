"""Model assembly: NF4 preparation, 4-bit loading, LoRA, k-bit prep.

The lifecycle of a fine-tune in this project:

    prepare   (once per base model)
        load source checkpoint into host RAM (fp16; it is far larger than
        one card — only each module's quantization runs on a GPU)
        optionally re-quantize modules the upstream checkpoint left in 16-bit
        normalize the stored quantization config (compute dtype, skip list)
        save -> .data/models/<name>-nf4  +  finetune_meta.json (exact numbers)
    train     (every run)
        load the prepared NF4 checkpoint directly (small RAM, fast)
        selective fp32 cast + freeze + gradient checkpointing
        attach LoRA (all-linear / explicit target list)
        -> SFTTrainer

Why prepare as a separate step (docs/01 + docs/02): loading 4-bit from a
bf16 checkpoint stages the *entire* bf16 model in host RAM first (~2
bytes/param — ~55 GB for a 27B model, multiplied by the number of
data-parallel processes). Preparing once removes that wall entirely.

Merging a trained adapter back to bf16 lives in ``merge.py``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

from .config import Config
from .utils import now_iso, write_json

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def compute_dtype(cfg: Config):
    """torch dtype for quantized matmuls (from model.dtype)."""
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.model.dtype]


def quant_storage_dtype(cfg: Config):
    """torch dtype for the 4-bit storage format."""
    import torch

    return {"uint8": torch.uint8, "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.quantization.quant_storage
    ]


def load_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """Tokenizer from the prepared checkpoint (or the hub source)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        # Qwen uses <|endoftext|> as both eos and pad; set explicitly so the
        # collator never guesses.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_model_class(model_name: str):
    """Pick the right Auto class for a checkpoint (VLM vs text-only)."""
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    archs = getattr(config, "architectures", []) or []
    # Multimodal checkpoints (e.g. Qwen3.8-27B) use the *ForConditionalGeneration
    # class; text-only checkpoints (e.g. Qwen3-8B) use AutoModelForCausalLM.
    is_vlm = any("ConditionalGeneration" in a for a in archs) or (
        hasattr(config, "vision_config") and getattr(config, "vision_config", None)
    )
    if is_vlm:
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is not None:
            return cls
        # Last resort: explicit class by architecture name
        for a in archs:
            cls = getattr(transformers, a, None)
            if cls is not None:
                return cls
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Preparation (download + surgery + save)
# ---------------------------------------------------------------------------


class QuantizationMeta(TypedDict):
    """The ``quantization`` section of finetune_meta.json."""

    quant_type: str
    double_quant: bool
    quant_storage: str
    compute_dtype: str


class FinetuneMeta(TypedDict):
    """`finetune_meta.json` in a prepared checkpoint: what `prepare` wrote.

    ``total_params`` is exact (counted from the tensors); the rest is
    provenance. Older checkpoints may miss keys — readers stay tolerant.
    """

    total_params: float
    source: NotRequired[str]
    requantized: NotRequired[list[str]]
    created: NotRequired[str]
    quantization: NotRequired[QuantizationMeta]


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
    embed/lm_head weight is counted once.
    """
    return {
        "total_params": float(sum(_logical_numel(p) for p in model.parameters())),
        "source": cfg.model.name,
        "requantized": requantized,
        "created": now_iso(),
        "quantization": {
            "quant_type": cfg.quantization.quant_type,
            "double_quant": cfg.quantization.double_quant,
            "quant_storage": cfg.quantization.quant_storage,
            "compute_dtype": cfg.model.dtype,
        },
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
    from .hardware import gpu_snapshot

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


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def _logical_numel(param: nn.Parameter) -> int:
    """True parameter count of a tensor.

    bitsandbytes packs two NF4 values per uint8, so ``numel()`` on a 4-bit
    weight reports half the logical parameter count — the difference between
    the ~15B a naive sum shows and the real ~27B of this checkpoint.
    """
    import torch

    if param.dtype == torch.uint8 and getattr(param, "quant_state", None) is not None:
        return param.numel() * 2
    return param.numel()


# ---------------------------------------------------------------------------
# Training-time assembly
# ---------------------------------------------------------------------------


def load_model_for_training(cfg: Config, *, local_rank: int = 0, attn: str | None = None):
    """Load the prepared NF4 checkpoint onto one GPU (one process per GPU).

    No ``quantization_config`` is passed: the prepared checkpoint carries its
    own (quant type, storage dtype, and the exact skip list), and that stored
    config is authoritative — passing a different one silently changes which
    modules are constructed as 4-bit.
    """
    import torch

    model_cls = resolve_model_class(cfg.model.prepared_dir if Path(cfg.model.prepared_dir).exists() else cfg.model.name)
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype(cfg),
        "device_map": {"": f"cuda:{local_rank}"},
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    attn_impl = attn or ("sdpa" if cfg.model.attn == "auto" else cfg.model.attn)
    kwargs["attn_implementation"] = attn_impl
    model = model_cls.from_pretrained(str(cfg.model.prepared_dir), **kwargs)
    model.config.use_cache = False

    # --- drop the (unused) vision tower -------------------------------------
    # Text-only training never touches it; dropping it frees the quantized
    # tower (~0.25 GiB) before DDP walks the parameters.
    if hasattr(model, "model") and getattr(model.model, "visual", None) is not None:
        model.model.visual = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return model


def prepare_for_qlora(model: PreTrainedModel, cfg: Config):
    """Freeze + selective fp32 cast + gradient checkpointing (QLoRA recipe).

    Differences from ``peft.prepare_model_for_kbit_training``:
    * the two big frozen lookup tables — the input embedding and the output
      head — stay in bf16 (2 bytes/param instead of 4; upcasting lm_head
      alone costs ~2.5 GB and buys nothing: it runs under bf16 autocast),
    * every other non-quantized 16-bit tensor (norms, conv1d, dt_bias, A_log,
      the small DeltaNet projections) is upcast to fp32, which *does* matter
      for numerics.
    """
    import bitsandbytes as bnb
    import torch

    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for _pname, param in module.named_parameters(recurse=False):
            if param.dtype in (torch.float16, torch.bfloat16):
                if "embed_tokens" in name or "lm_head" in name:
                    continue  # big frozen tables stay 16-bit (see docstring)
                param.data = param.data.to(torch.float32)

    if cfg.optim.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
            every_n_layers=cfg.optim.checkpoint_every_n_layers,
            offload=cfg.optim.activation_offloading,
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def attach_lora(model: PreTrainedModel, cfg: Config):
    """Attach LoRA adapters per config; returns the PeftModel."""
    from peft import LoraConfig, get_peft_model

    # The vision tower is dropped at load (`load_model_for_training`); only pass
    # the exclusion when it is actually present, or PEFT warns about an empty
    # exclusion set on text-only checkpoints.
    has_visual = any("visual" in name for name, _ in model.named_modules())
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
        exclude_modules=["visual"] if has_visual else None,  # keep the vision tower untouched when present
    )
    return get_peft_model(model, lora)


class ParameterReport(TypedDict):
    """`parameter_report`'s result: totals + per-module trainable breakdown."""

    total: int
    trainable: int
    pct: float
    by_module: dict[str, int]


def parameter_report(model: PreTrainedModel) -> ParameterReport:
    """Total vs trainable parameters + a per-module breakdown of the trainable ones."""
    from collections import Counter

    total = sum(_logical_numel(p) for p in model.parameters())  # logical count (NF4 packs 2/byte)
    trainable = 0
    by_module: Counter[str] = Counter()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable += param.numel()
        parts = name.split(".")
        # PEFT names the adapter tensors `...<module>.lora_A[.name].weight`
        lora_at = [i for i, p in enumerate(parts) if p in ("lora_A", "lora_B")]
        module = parts[lora_at[0] - 1] if lora_at else (parts[-2] if parts[-1] == "weight" else parts[-1])
        by_module[module] += param.numel()
    return {
        "total": total,
        "trainable": trainable,
        "pct": 100.0 * trainable / max(1, total),
        "by_module": dict(by_module.most_common(12)),
    }
