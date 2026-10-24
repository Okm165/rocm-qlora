"""Model assembly: NF4 preparation, 4-bit loading, LoRA, k-bit prep.

The lifecycle of a fine-tune in this project:

    prepare   (once per base model)
        load source checkpoint on CPU (fp16)
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
from typing import TYPE_CHECKING, Any, TypedDict, cast

from .config import Config
from .utils import now_iso, write_json

if TYPE_CHECKING:
    import torch
    from torch import nn
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from .vram import FinetuneMeta

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


@dataclass
class PrepareResult:
    out_dir: Path
    source: str
    requantized: list[str]
    seconds: float
    total_params: float
    disk_gb: float


def _requantize_linear(module: nn.Linear, cfg: Config, *, device: str | None = None):
    """Convert an fp16 nn.Linear into bnb Linear4bit (NF4), preserving math."""
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
    # Moving the module triggers lazy quantization on the target device.
    return new.to(device or weight.device)


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
    _save_prepared(model, out_dir, log=log)
    load_tokenizer(cfg).save_pretrained(out_dir)

    meta = _prepare_meta(model, cfg, requantized)
    write_json(meta_path, meta)
    seconds = time.perf_counter() - t0
    log(f"Prepared checkpoint ready ({_dir_size_gb(out_dir):.1f} GB on disk, {seconds / 60:.1f} min)")
    return PrepareResult(
        out_dir=out_dir,
        source=cfg.model.name,
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
        source=meta.get("source", "?"),
        requantized=meta.get("requantized", []),
        seconds=0.0,
        total_params=float(meta.get("total_params", 0.0)),
        disk_gb=_dir_size_gb(out_dir),
    )


def _load_source_model(cfg: Config, log: Callable[[str], None]):
    """Load the source checkpoint on CPU in fp16 (low-RAM mode)."""
    import torch

    log(f"Loading source checkpoint {cfg.model.name!r} on CPU (fp16, low RAM mode)…")
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
    # Quantizing on a GPU is much faster than CPU; use the freest GPU.
    target_device = _pick_free_gpu_for_surgery()
    for name, module in list(model.named_modules()):
        if not _should_quantize(name, module, cfg, source_quantized=source_quantized):
            continue
        parent, attr = parent_of(model, name)
        old = module
        if target_device is not None:
            old = old.to(target_device)
        new = _requantize_linear(cast("nn.Linear", old), cfg, device=target_device or "cpu")
        if target_device is not None:
            new = new.to("cpu")  # keep the model CPU-resident for save
        setattr(parent, attr, new)
        requantized.append(name)
    log(f"Re-quantized {len(requantized)} module(s) to NF4")
    return requantized


def _write_quantization_config(model: PreTrainedModel, cfg: Config) -> None:
    """Attach the load-time-authoritative bnb config.

    The stored config is authoritative at load time (it lists exactly which
    modules are NOT quantized), so derive the skip list from *reality*:
    every plain Linear that survived surgery, by its module suffix.
    """
    import bitsandbytes as bnb
    import torch

    remaining = sorted(
        {
            name.split(".")[-1]
            for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and not isinstance(m, bnb.nn.Linear4bit)
        }
    )
    model.config.quantization_config = {
        "load_in_4bit": True,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": cfg.quantization.quant_type,
        "bnb_4bit_use_double_quant": cfg.quantization.double_quant,
        "bnb_4bit_compute_dtype": cfg.model.dtype,
        "bnb_4bit_quant_storage": cfg.quantization.quant_storage,
        "llm_int8_skip_modules": remaining,
    }


def _prepare_meta(model: PreTrainedModel, cfg: Config, requantized: list[str]) -> FinetuneMeta:
    """Exact parameter accounting + provenance for the VRAM estimator."""
    meta = model_parameter_meta(model, cfg)
    meta.update(
        {
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
    )
    return meta


def _save_prepared(model: PreTrainedModel, out_dir: Path, *, log: Callable[[str], None]) -> None:
    """save_pretrained with a fallback for non-reversible weight conversions.

    transformers v5 replays checkpoint->model conversions in reverse when
    saving; some VLM checkpoints carry vision-side conversions without a
    reverse, failing the plain save. Vision tensors are unused for text-only
    training, so dropping the non-reversible conversions is safe (the dry-run
    reload verifies the text-side round-trip).
    """
    try:
        model.save_pretrained(out_dir, max_shard_size="4GB")
        return
    except (NotImplementedError, ValueError) as exc:
        log(f"save hit a non-reversible weight conversion ({type(exc).__name__}); retrying with a filtered set")

    conversions = list(getattr(model, "_weight_conversions", None) or [])
    kept = []
    for conversion in conversions:
        try:
            conversion.reverse_transform()
            kept.append(conversion)
        except (NotImplementedError, ValueError):
            log(
                f"  dropping non-reversible conversion: {type(conversion).__name__} "
                f"{getattr(conversion, 'source_patterns', '')}"
            )
    model._weight_conversions = kept or None  # pyright: ignore[reportArgumentType]  # dynamic attr
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")


def _pick_free_gpu_for_surgery() -> str | None:
    try:
        from .hardware import gpu_snapshot

        snap = gpu_snapshot()
        if not snap:
            return None
        best = max(snap, key=lambda g: g.mem_free_b)
        if best.mem_free_b < 2 * 10**9:
            return None
        return f"cuda:{best.index}"
    except Exception:
        return None


def _dir_size_gb(path: Path) -> float:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def _match_target(name: str, target_modules: str | list[str]) -> bool:
    if ".visual." in name:
        return False  # vision tower is never adapted for text training
    if name.endswith("lm_head") or name == "lm_head":
        return False  # never adapt the output head
    if target_modules == "all-linear":
        return True
    return any(name.endswith(suffix) for suffix in target_modules)


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


def _classify_param(
    param: torch.nn.Parameter,
    module: nn.Module,
    name: str,
    *,
    embed_module: nn.Module | None,
    head: nn.Module | None,
) -> str:
    """Which accounting bucket one parameter belongs to (checked in this order)."""
    if embed_module is not None and param is getattr(embed_module, "weight", None):
        return "embed"
    if head is not None and param is getattr(head, "weight", None):
        return "lm_head"
    if "visual" in name:
        return "vision"
    import bitsandbytes as bnb

    if isinstance(module, bnb.nn.Linear4bit):
        return "quantized"
    return "nonquant"


def model_parameter_meta(model: PreTrainedModel, cfg: Config) -> FinetuneMeta:
    """Exact parameter counts for the VRAM estimator + teaching output."""
    import bitsandbytes as bnb
    import torch

    total = 0
    lora_units = 0.0
    buckets = {"embed": 0, "lm_head": 0, "vision": 0, "quantized": 0, "nonquant": 0}
    embed_module = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            n = _logical_numel(param)
            total += n
            buckets[_classify_param(param, module, name, embed_module=embed_module, head=head)] += n
        if (
            "visual" not in name
            and isinstance(module, (bnb.nn.Linear4bit, torch.nn.Linear))
            and _match_target(name, cfg.lora.target_modules)
        ):
            lora_units += module.in_features + module.out_features
    text_cfg = getattr(model.config, "text_config", model.config)
    return {
        "name": cfg.model.name,
        "total_params": float(total),
        "embed_params": float(buckets["embed"]),
        "lm_head_params": float(buckets["lm_head"]),
        "quantized_params": float(buckets["quantized"]),
        "nonquant_16bit_params": float(buckets["nonquant"]),
        "vision_params": float(buckets["vision"]),  # dropped after load; counted for completeness
        "lora_units": float(lora_units),
        "hidden_size": int(getattr(text_cfg, "hidden_size", 0) or 0),
        "n_layers": int(getattr(text_cfg, "num_hidden_layers", 0) or 0),
        "vocab_size": int(getattr(text_cfg, "vocab_size", 0) or 0),
    }


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
    # Text-only training never touches it; dropping it frees ~0.7 GiB per GPU
    # and shrinks the initial DDP parameter broadcast.
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
    """`parameter_report`'s result: totals + per-suffix trainable breakdown."""

    total: int
    trainable: int
    pct: float
    by_suffix: dict[str, int]


def parameter_report(model: PreTrainedModel) -> ParameterReport:
    """Total vs trainable parameters + a per-suffix breakdown for teaching."""
    from collections import Counter

    total = sum(_logical_numel(p) for p in model.parameters())  # logical count (NF4 packs 2/byte)
    trainable = 0
    by_suffix: Counter[str] = Counter()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable += param.numel()
        suffix = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
        by_suffix[suffix] += param.numel()
    return {
        "total": total,
        "trainable": trainable,
        "pct": 100.0 * trainable / max(1, total),
        "by_suffix": dict(by_suffix.most_common(12)),
    }
