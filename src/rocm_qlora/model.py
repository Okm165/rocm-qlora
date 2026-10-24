"""Model assembly: the training-time half of the lifecycle.

    train     (every run)
        load the prepared NF4 checkpoint directly onto this rank's GPU
        selective fp32 cast + freeze + gradient checkpointing
        attach LoRA (all-linear / explicit target list)
        -> SFTTrainer

The one-time checkpoint surgery (`prepare`) lives in ``prepare.py``; merging a
trained adapter back to bf16 lives in ``merge.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .config import Config

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
    """Tokenizer from the prepared checkpoint when it exists, else the hub source.

    `prepare` saves the tokenizer into the prepared directory, so everything
    after it (train, merge, chat) works without touching the hub.
    """
    from transformers import AutoTokenizer

    prepared = Path(cfg.model.prepared_dir)
    source = str(prepared) if prepared.exists() else cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:  # generic guard; Qwen checkpoints ship both
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_model_class(model_name: str, *, trust_remote_code: bool = False):
    """Pick the right Auto class for a checkpoint (VLM vs text-only)."""
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
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
# Parameter accounting
# ---------------------------------------------------------------------------


def logical_numel(param: nn.Parameter) -> int:
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

    prepared = Path(cfg.model.prepared_dir)
    if not (prepared / "config.json").exists():
        raise RuntimeError(
            f"prepared checkpoint not found at {prepared} — run "
            f"`rocm-qlora prepare --config <config>` first (docs/01 explains why it exists)"
        )
    model_cls = resolve_model_class(str(prepared), trust_remote_code=cfg.model.trust_remote_code)
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
    model.enable_input_require_grads()  # PreTrainedModel API; grads must reach the frozen inputs
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

    total = sum(logical_numel(p) for p in model.parameters())  # logical count (NF4 packs 2/byte)
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
