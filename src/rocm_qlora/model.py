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

Why prepare as a separate step? See the "prepared_checkpoint" explainer:
loading 4-bit from a bf16 checkpoint stages the *entire* bf16 model in host
RAM first (~2 bytes/param — ~55 GB for a 27B model, multiplied by the number
of data-parallel processes). Preparing once removes that wall entirely.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .utils import human_bytes, now_iso, write_json

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"bfloat16": "torch.bfloat16", "float16": "torch.float16"}


def compute_dtype(cfg: Config):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.model.dtype]


def quant_storage_dtype(cfg: Config):
    import torch

    return {"uint8": torch.uint8, "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.quantization.quant_storage
    ]


def bnb_config_from(cfg: Config):
    """BitsAndBytesConfig that mirrors what `prepare` wrote into the checkpoint."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype(cfg),
        bnb_4bit_quant_storage=quant_storage_dtype(cfg),
        llm_int8_skip_modules=None,  # pre-quantized checkpoint: stored config governs
    )


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        # Qwen uses <|endoftext|> as both eos and pad; set explicitly so the
        # collator never guesses.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_model_class(model_name: str, *, prefer_text_only: bool = False):
    """Pick the right Auto class for a checkpoint (VLM vs text-only)."""
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    archs = getattr(config, "architectures", []) or []
    getattr(config, "model_type", "")
    if prefer_text_only and "vision_config" not in getattr(config, "to_dict", dict)():
        pass
    # Multimodal checkpoints (e.g. Qwen3.8-27B) use the *ForConditionalGeneration
    # class; text-only checkpoints (e.g. Qwen3-8B) use AutoModelForCausalLM.
    is_vlm = any("ConditionalGeneration" in a for a in archs) or (
        hasattr(config, "vision_config") and getattr(config, "vision_config", None)
    )
    if is_vlm and not prefer_text_only:
        for attr in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            cls = getattr(transformers, attr, None)
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


def _requantize_linear(module, cfg: Config, *, device: str | None = None):
    """Convert an fp16 nn.Linear into bnb Linear4bit (NF4), preserving math."""
    import bitsandbytes as bnb
    import torch

    new = bnb.nn.Linear4bit(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        compute_dtype=compute_dtype(cfg),
        compress_statistics=cfg.quantization.double_quant,
        quant_type=cfg.quantization.quant_type,
        quant_storage=quant_storage_dtype(cfg),
    )
    weight = module.weight.data
    new.weight = bnb.nn.Params4bit(
        weight,
        requires_grad=False,
        quant_type=cfg.quantization.quant_type,
        compress_statistics=cfg.quantization.double_quant,
        quant_storage=quant_storage_dtype(cfg),
    )
    if module.bias is not None:
        new.bias = torch.nn.Parameter(module.bias.data.clone(), requires_grad=False)
    # Moving the module triggers lazy quantization on the target device.
    return new.to(device or weight.device)


def _parent_of(model, dotted_name: str):
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
    import torch

    out_dir = Path(cfg.model.prepared_dir)
    meta_path = out_dir / "finetune_meta.json"
    if out_dir.exists() and (out_dir / "config.json").exists() and meta_path.exists() and not force:
        import json

        meta = json.loads(meta_path.read_text())
        return PrepareResult(
            out_dir=out_dir,
            source=meta.get("source", "?"),
            requantized=meta.get("requantized", []),
            seconds=0.0,
            total_params=float(meta.get("total_params", 0.0)),
            disk_gb=_dir_size_gb(out_dir),
        )

    t0 = time.perf_counter()
    log(f"Loading source checkpoint {cfg.model.name!r} on CPU (fp16, low RAM mode)…")
    model_cls = resolve_model_class(cfg.model.name)
    model = model_cls.from_pretrained(
        cfg.model.name,
        dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.eval()

    # --- decide what to quantize -------------------------------------------
    # Two kinds of sources:
    #  * already-quantized checkpoints (e.g. unsloth bnb-4bit): only the
    #    modules the upstream left in 16-bit need surgery;
    #  * plain bf16/fp16 checkpoints: every Linear gets quantized.
    source_quantized = getattr(model.config, "quantization_config", None) is not None

    def should_quantize(name: str, module) -> bool:
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

    requantized: list[str] = []
    if cfg.quantization.requantize_modules or cfg.quantization.requantize_vision or not source_quantized:
        # Quantizing on a GPU is much faster than CPU; use the freest GPU.
        target_device = _pick_free_gpu_for_surgery()
        for name, module in list(model.named_modules()):
            if not should_quantize(name, module):
                continue
            parent, attr = _parent_of(model, name)
            old = module
            if target_device is not None:
                old = old.to(target_device)
            new = _requantize_linear(old, cfg, device=target_device or "cpu")
            if target_device is not None:
                new = new.to("cpu")  # keep the model CPU-resident for save
            setattr(parent, attr, new)
            requantized.append(name)
        log(f"Re-quantized {len(requantized)} module(s) to NF4")

    # --- write the quantization config the checkpoint will carry ------------
    # The stored config is authoritative at load time (it lists exactly which
    # modules are NOT quantized), so derive the skip list from *reality*:
    # every plain Linear that survived surgery, by its module suffix.
    import bitsandbytes as bnb

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

    # --- save ---------------------------------------------------------------
    log(f"Saving prepared checkpoint to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_prepared(model, out_dir, log=log)
    tokenizer = load_tokenizer(cfg)
    tokenizer.save_pretrained(out_dir)

    # --- exact parameter accounting for the VRAM estimator ------------------
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


def _save_prepared(model, out_dir: Path, *, log: Callable[[str], None]) -> None:
    """save_pretrained with a fallback for non-reversible weight conversions.

    transformers v5 replays the checkpoint->model weight conversions in
    reverse when saving. Some VLM checkpoints (Qwen3.8 included) carry
    vision-side conversions without a reverse implementation, which makes the
    plain save fail with NotImplementedError. Vision tensors are unused for
    text-only training, so dropping the non-reversible conversions is safe —
    and the training dry-run (which reloads the checkpoint) verifies the
    text-side round-trip end to end.
    """
    try:
        model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
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
    model._weight_conversions = kept or None
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


def _logical_numel(param) -> int:
    """True parameter count of a tensor.

    bitsandbytes packs two NF4 values per uint8, so ``numel()`` on a 4-bit
    weight reports half the logical parameter count — the difference between
    the ~15B a naive sum shows and the real ~27B of this checkpoint.
    """
    if param.dtype == torch_uint8() and getattr(param, "quant_state", None) is not None:
        return param.numel() * 2
    return param.numel()


def model_parameter_meta(model, cfg: Config) -> dict[str, Any]:
    """Exact parameter counts for the VRAM estimator + teaching output."""
    import bitsandbytes as bnb

    total = embed = lm_head = quantized = nonquant = vision = 0
    lora_units = 0.0
    embed_module = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            n = _logical_numel(param)
            total += n
            if embed_module is not None and param is getattr(embed_module, "weight", None):
                embed += n
            elif head is not None and param is getattr(head, "weight", None):
                lm_head += n
            elif "visual" in name:
                vision += n  # dropped right after load; counted for completeness
            elif isinstance(module, bnb.nn.Linear4bit):
                quantized += n
            else:
                nonquant += n
        if (
            "visual" not in name
            and isinstance(module, (bnb.nn.Linear4bit, torch_nn_linear()))
            and _match_target(name, cfg.lora.target_modules)
        ):
            lora_units += module.in_features + module.out_features
    text_cfg = getattr(model.config, "text_config", model.config)
    return {
        "name": cfg.model.name,
        "total_params": float(total),
        "embed_params": float(embed),
        "lm_head_params": float(lm_head),
        "quantized_params": float(quantized),
        "nonquant_16bit_params": float(nonquant),
        "vision_params": float(vision),
        "lora_units": float(lora_units),
        "hidden_size": int(getattr(text_cfg, "hidden_size", 0) or 0),
        "n_layers": int(getattr(text_cfg, "num_hidden_layers", 0) or 0),
        "vocab_size": int(getattr(text_cfg, "vocab_size", 0) or 0),
    }


def torch_nn_linear():
    import torch

    return torch.nn.Linear


def torch_fp16():
    import torch

    return torch.float16


def torch_bf16():
    import torch

    return torch.bfloat16


def torch_uint8():
    import torch

    return torch.uint8


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
        "low_cpu_mem_usage": True,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    attn_impl = attn or ("sdpa" if cfg.model.attn == "auto" else cfg.model.attn)
    kwargs["attn_implementation"] = attn_impl
    model = model_cls.from_pretrained(str(cfg.model.prepared_dir), **kwargs)
    model.config.use_cache = False

    # --- drop the (unused) vision tower -------------------------------------
    # Text-only training never touches `model.model.visual` (it is only
    # reachable via pixel_values). Dropping it frees ~0.7 GiB per GPU and,
    # critically at 27B scale, shrinks the initial DDP parameter broadcast —
    # which otherwise needs a ~4.9 GiB transient buffer (the fp16
    # embed_tokens + lm_head pair) on a 24 GB card.
    if hasattr(model, "model") and getattr(model.model, "visual", None) is not None:
        model.model.visual = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return model


def prepare_for_qlora(model, cfg: Config):
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def attach_lora(model, cfg: Config):
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


def parameter_report(model) -> dict[str, Any]:
    """Total vs trainable parameters + a per-suffix breakdown for teaching."""
    from collections import Counter

    total = sum(p.numel() for p in model.parameters())
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


# ---------------------------------------------------------------------------
# Merge: adapter -> bf16 model (ready for GGUF export)
# ---------------------------------------------------------------------------


def merge_run(run_dir: str | Path, *, out_dir: str | Path | None = None, console=None) -> Path:
    """Fold the adapter into the base, dequantize NF4 → bf16, save.

    A 27B model needs ~55 GB of bf16 memory: host RAM if it fits, otherwise
    the model is sharded across the available GPUs (``device_map="balanced"``).
    """
    import json as _json

    import torch
    from rich.console import Console
    from rich.panel import Panel

    from .hardware import host_memory

    console = console or Console()
    run_dir = Path(run_dir)
    cfg = Config.from_dict(_json.loads((run_dir / "config.resolved.json").read_text()))
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"No adapter found at {adapter_dir}")
    out = Path(out_dir) if out_dir else run_dir / "merged"
    out.mkdir(parents=True, exist_ok=True)

    model_bytes = _estimate_model_bytes(cfg)
    ram = host_memory()
    if ram.get("available_gib", 0) * 2**30 > model_bytes * 2.2:
        device_map: Any = {"": "cpu"}
        console.print(
            f"[dim]merging on CPU ({human_bytes(model_bytes)} model, {ram['available_gib']:.0f} GiB RAM free)[/dim]"
        )
    else:
        device_map = "balanced"
        console.print(
            f"[dim]merging sharded across GPUs (model ≈ {human_bytes(model_bytes)}, "
            f"host RAM free {ram['available_gib']:.0f} GiB)[/dim]"
        )

    from peft import PeftModel

    console.print("[dim]loading prepared 4-bit checkpoint…[/dim]")
    model_cls = resolve_model_class(cfg.model.prepared_dir)
    model = model_cls.from_pretrained(
        cfg.model.prepared_dir,
        dtype=compute_dtype(cfg),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    console.print("[dim]folding LoRA weights into the quantized model…[/dim]")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    console.print("[dim]dequantizing NF4 → bf16…[/dim]")
    model = _dequantize_linear4bit(model, dtype=torch.bfloat16)
    model.config.use_cache = True

    console.print(f"[dim]saving bf16 model to {out} …[/dim]")
    # `save_original_format=False`: transformers would otherwise try to revert
    # the bnb load conversion (re-packing bf16 weights into NF4), which is not
    # implemented — the weights are plain bf16 and are saved as such.
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB", save_original_format=False)
    load_tokenizer(cfg).save_pretrained(out)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    console.print(
        Panel(
            f"merged model: [bold]{out}[/bold] ({human_bytes(size)})\n\n"
            "Convert to GGUF for llama.cpp:\n"
            f"  python llama.cpp/convert_hf_to_gguf.py {out} --outfile model-f16.gguf --outtype f16\n"
            "  llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M",
            title="[bold green]merge complete[/bold green]",
            border_style="green",
        )
    )
    return out


def _estimate_model_bytes(cfg: Config) -> int:
    """Rough bf16 size of the merged model — drives the CPU-vs-GPU choice."""
    import json as _json

    base = Path(cfg.model.prepared_dir)
    try:
        total_params = _json.loads((base / "finetune_meta.json").read_text()).get("total_params")
        if total_params:
            return int(total_params) * 2  # bf16 bytes per parameter
    except Exception:
        pass
    try:
        index = base / "model.safetensors.index.json"
        if index.exists():
            return int(_json.loads(index.read_text())["metadata"].get("total_size", 0)) * 2
        nf4_bytes = sum(f.stat().st_size for f in base.glob("*.safetensors"))
        if nf4_bytes:
            return nf4_bytes * 2
    except Exception:
        pass
    return 60 * 10**9


def _dequantize_linear4bit(model, *, dtype):
    """Replace every Linear4bit module with a plain Linear holding bf16 weights."""
    import bitsandbytes as bnb
    import torch

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight = module.weight
        dequant = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(dtype)
        new = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        new.weight = torch.nn.Parameter(dequant, requires_grad=False)
        if module.bias is not None:
            new.bias = torch.nn.Parameter(module.bias.data.to(dtype), requires_grad=False)
        new = new.to(device=dequant.device)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new)
    # The weights are bf16 now: drop the quantization metadata (a config that
    # still claims 4-bit would re-quantize on reload and confuse GGUF
    # converters) and the consumed load-time conversion record (its reverse —
    # re-packing to NF4 — is not implemented, so keeping it breaks saving).
    model.config.__dict__.pop("quantization_config", None)
    with contextlib.suppress(Exception):
        model._weight_conversions = None
    import gc

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    return model
