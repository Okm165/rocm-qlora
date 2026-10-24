"""Merge: fold a LoRA adapter into a bf16 model (ready for GGUF export)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config
from .hardware import host_memory
from .model import compute_dtype, load_tokenizer, parent_of, resolve_model_class
from .utils import human_bytes

if TYPE_CHECKING:
    import torch
    from rich.console import Console
    from transformers import PreTrainedModel

    from .hardware import HostMemory


# ---------------------------------------------------------------------------
# Merge: adapter -> bf16 model (ready for GGUF export)
# ---------------------------------------------------------------------------


def _choose_merge_device(model_bytes: int, ram: HostMemory, console: Console) -> Any:
    """CPU when host RAM fits the model twice over, else shard across GPUs."""
    if ram.get("available_gib", 0) * 2**30 > model_bytes * 2.2:
        console.print(
            f"[dim]merging on CPU ({human_bytes(model_bytes)} model, {ram['available_gib']:.0f} GiB RAM free)[/dim]"
        )
        return {"": "cpu"}
    console.print(
        f"[dim]merging sharded across GPUs (model ≈ {human_bytes(model_bytes)}, "
        f"host RAM free {ram['available_gib']:.0f} GiB)[/dim]"
    )
    return "balanced"


def _fold_adapter(cfg: Config, adapter_dir: Path, device_map: Any, console: Console) -> Any:
    """Load the prepared 4-bit base, fold the LoRA adapter, dequantize to bf16."""
    import torch
    from peft import PeftModel

    console.print("[dim]loading prepared 4-bit checkpoint…[/dim]")
    model_cls = resolve_model_class(cfg.model.prepared_dir)
    model = model_cls.from_pretrained(
        cfg.model.prepared_dir,
        dtype=compute_dtype(cfg),
        device_map=device_map,
    )
    console.print("[dim]folding LoRA weights into the quantized model…[/dim]")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    console.print("[dim]dequantizing NF4 → bf16…[/dim]")
    model = _dequantize_linear4bit(model, dtype=torch.bfloat16)
    model.config.use_cache = True
    return model


def merge_run(run_dir: str | Path, *, out_dir: str | Path | None = None, console: Console | None = None) -> Path:
    """Fold the adapter into the base, dequantize NF4 → bf16, save.

    A 27B model needs ~55 GB of bf16 memory: host RAM if it fits, otherwise
    the model is sharded across the available GPUs (``device_map="balanced"``).
    """
    from rich.console import Console
    from rich.panel import Panel

    console = console or Console()
    run_dir = Path(run_dir)
    cfg = Config.from_toml(run_dir / "config.resolved.toml")
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"No adapter found at {adapter_dir}")
    out = Path(out_dir) if out_dir else run_dir / "merged"
    out.mkdir(parents=True, exist_ok=True)

    device_map = _choose_merge_device(_estimate_model_bytes(cfg), host_memory(), console)
    model = _fold_adapter(cfg, adapter_dir, device_map, console)

    console.print(f"[dim]saving bf16 model to {out} …[/dim]")
    # `save_original_format=False`: transformers would otherwise try to revert
    # the bnb load conversion (re-packing bf16 weights into NF4), which is not
    # implemented — the weights are plain bf16 and are saved as such.
    model.save_pretrained(out, max_shard_size="4GB", save_original_format=False)
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
    base = Path(cfg.model.prepared_dir)
    try:
        total_params = json.loads((base / "finetune_meta.json").read_text()).get("total_params")
        if total_params:
            return int(total_params) * 2  # bf16 bytes per parameter
    except Exception:
        pass
    try:
        index = base / "model.safetensors.index.json"
        if index.exists():
            return int(json.loads(index.read_text())["metadata"].get("total_size", 0)) * 2
        nf4_bytes = sum(f.stat().st_size for f in base.glob("*.safetensors"))
        if nf4_bytes:
            return nf4_bytes * 2
    except Exception:
        pass
    return 60 * 10**9


def _dequantize_linear4bit(model: PreTrainedModel, *, dtype: torch.dtype):
    """Replace every Linear4bit module with a plain Linear holding bf16 weights."""
    import bitsandbytes as bnb
    import torch

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight = module.weight
        dequant = bnb.functional.dequantize_4bit(  # pyright: ignore[reportAttributeAccessIssue]  # runtime API
            weight.data,
            weight.quant_state,  # pyright: ignore[reportAttributeAccessIssue]  # set by Params4bit
        ).to(dtype)
        new = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)  # pyright: ignore[reportUnnecessaryComparison]
        new.weight = torch.nn.Parameter(dequant, requires_grad=False)
        if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]  # biasless Linears
            new.bias = torch.nn.Parameter(module.bias.data.to(dtype), requires_grad=False)
        new = new.to(device=dequant.device)
        parent, attr = parent_of(model, name)
        setattr(parent, attr, new)
    # The weights are bf16 now: drop the quantization metadata (a config that
    # still claims 4-bit would re-quantize on reload and confuse GGUF
    # converters) and the consumed load-time conversion record (its reverse —
    # re-packing to NF4 — is not implemented, so keeping it breaks saving).
    with contextlib.suppress(AttributeError):
        delattr(model.config, "quantization_config")
    with contextlib.suppress(Exception):
        model._weight_conversions = None  # pyright: ignore[reportArgumentType]  # dynamic attr
    import gc

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    return model
