"""Model assembly: the training-time half of the lifecycle.

    train     (every run)
        load the prepared NF4 checkpoint directly onto this rank's GPU
        selective fp32 cast + freeze
        attach LoRA (all-linear / explicit target list)
        -> SFTTrainer (which enables gradient checkpointing from the config)

The one-time checkpoint surgery (`prepare`) lives in ``prepare.py``; merging a
trained adapter back to bf16 lives in ``merge.py``.
"""

from __future__ import annotations

import math
import warnings
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


def load_tokenizer(cfg: Config, *, prefer_prepared: bool = True) -> PreTrainedTokenizerBase:
    """Tokenizer from the prepared checkpoint when it exists, else the hub source.

    `prepare` saves the tokenizer into the prepared directory, so everything
    after it (train, merge, chat) works without touching the hub. `prepare`
    itself passes ``prefer_prepared=False``: the checkpoint must carry the
    *source* tokenizer, not a copy of itself.
    """
    from transformers import AutoTokenizer

    prepared = Path(cfg.model.prepared_dir)
    source = str(prepared) if prefer_prepared and prepared_checkpoint_exists(cfg) else cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:  # generic guard; Qwen checkpoints ship both
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepared_checkpoint_exists(cfg: Config) -> bool:
    """True when `prepare` has written a complete checkpoint.

    `finetune_meta.json` is written after the weights and the tokenizer, so it
    marks a finished save; a directory with only `config.json` is a crashed or
    foreign one.
    """
    prepared = Path(cfg.model.prepared_dir)
    return (prepared / "config.json").exists() and (prepared / "finetune_meta.json").exists()


def resolve_model_class(model_name: str, *, trust_remote_code: bool = False):
    """Pick the right Auto class for a checkpoint (VLM vs text-only)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # The library's own routing table decides (qwen3_5 is in it). No fallback on
    # config.vision_config: transformers auto-fills non-optional sub-configs, so
    # its presence does not mean the checkpoint declared a vision tower.
    is_vlm = config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    return AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def logical_numel(param: nn.Parameter) -> int:
    """Logical element count of a parameter, packed 4-bit storage included.

    bitsandbytes stores quantized weights packed — two 4-bit values per byte of
    ``quant_storage`` — and keeps the unpacked shape on ``quant_state``, so the
    tensor's own ``numel()`` is the packed count, not the logical one (this is
    why a naive sum shows ~15B for a 27B checkpoint). Every other parameter's
    tensor shape is already logical.
    """
    quant_state = getattr(param, "quant_state", None)
    shape = getattr(quant_state, "shape", None)
    if shape is not None:
        return int(math.prod(shape))
    import bitsandbytes as bnb

    if isinstance(param, bnb.nn.Params4bit):
        raise RuntimeError(
            "a quantized parameter has no quant_state — bitsandbytes changed its layout; refusing to report "
            "the packed element count as if it were logical (docs/01's accounting must stay exact)"
        )
    return int(param.numel())


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

    if not prepared_checkpoint_exists(cfg):
        raise RuntimeError(
            f"prepared checkpoint missing or incomplete at {cfg.model.prepared_dir} — run "
            f"`rocm-qlora prepare --config <config>` first (docs/01 explains why it exists)"
        )
    model_cls = resolve_model_class(cfg.model.prepared_dir, trust_remote_code=cfg.model.trust_remote_code)
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
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        model.model.visual = None  # type: ignore[union-attr]
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    elif getattr(model.config, "vision_config", None) is not None:
        warnings.warn(
            "vision_config is set but no 'model.visual' module was found — the vision tower stays in the "
            "model, and LoRA's target modules will adapt it too; teach load_model_for_training this "
            "checkpoint's tower path",
            stacklevel=2,
        )
    return model


def prepare_for_qlora(model: PreTrainedModel):
    """Freeze + selective fp32 cast (QLoRA recipe).

    Gradient checkpointing is *not* enabled here: the trainer enables it from
    the SFTConfig (with the same `use_reentrant`/`every_n_layers`/`offload`
    kwargs, after PEFT wrapping — where it must be enabled).

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

    # The two big frozen lookup tables are found by module identity, not by
    # module name (names are architecture-specific).
    big_tables = {
        id(m.weight)
        for m in (model.get_input_embeddings(), model.get_output_embeddings())
        if m is not None and hasattr(m, "weight")
    }
    for module in model.modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for _pname, param in module.named_parameters(recurse=False):
            if param.dtype in (torch.float16, torch.bfloat16) and id(param) not in big_tables:
                param.data = param.data.to(torch.float32)

    model.enable_input_require_grads()  # PreTrainedModel API; grads must reach the frozen inputs
    return model


def attach_lora(model: PreTrainedModel, cfg: Config):
    """Attach LoRA adapters per config; returns the PeftModel.

    Call this *after* `load_model_for_training`: the vision tower is already
    dropped there, so there is nothing to exclude from the target modules.
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
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
