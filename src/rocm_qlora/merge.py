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
    """CPU when host RAM fits the model with ~2.2x headroom, else shard across GPUs."""
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

    device_map = _choose_merge_device(_bf16_model_bytes(Path(cfg.model.prepared_dir)), host_memory(), console)
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
            "[dim]GGUF export for llama.cpp: docs/06-27b-runbook.md (step 6).[/dim]",
            title="[bold green]merge complete[/bold green]",
            border_style="green",
        )
    )
    return out


def _bf16_model_bytes(prepared_dir: Path) -> int:
    """Exact bf16 size of a prepared checkpoint, from the safetensors headers.

    Disk bytes are not parameter counts (NF4 packs two values per uint8), and
    the headers carry the shapes — so the merged size is exact for any
    prepared directory, no meta file required.
    """
    import math
    import struct

    dtype_bytes = {"U8": 4, "F16": 2, "BF16": 2, "F32": 4}  # U8 = 2 NF4 values/byte, 2 bytes each in bf16
    aux = (".absmax", ".quant_map", ".nested_", ".quant_state")
    total = 0
    for shard in sorted(prepared_dir.glob("*.safetensors")):
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, info in header.items():
            if name == "__metadata__" or any(tag in name for tag in aux):
                continue
            total += math.prod(info["shape"]) * dtype_bytes.get(info["dtype"], 2)
    return total


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
