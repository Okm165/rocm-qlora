"""Merge: fold a LoRA adapter into a bf16 model (ready for GGUF export)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from .config import Config
from .model import compute_dtype, load_tokenizer, resolve_model_class
from .utils import human_bytes

if TYPE_CHECKING:
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Merge: adapter -> bf16 model (ready for GGUF export)
# ---------------------------------------------------------------------------


def _gpu_memory_budget() -> dict[int, int]:
    """Free VRAM per GPU, as the dispatch budget.

    ``device_map="balanced"`` on its own would let accelerate spill layers to
    the CPU (its default budget includes host RAM); passing only GPUs makes a
    too-small device set fail loudly instead of silently merging on CPU.
    """
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("merge needs visible GPUs — this tool has no CPU path")
    return {i: torch.cuda.mem_get_info(i)[0] for i in range(count)}


def _fold_adapter(cfg: Config, adapter_dir: Path, console: Console) -> PreTrainedModel:
    """Load the prepared 4-bit base on the GPUs, fold the LoRA adapter, dequantize."""
    import torch
    from peft import PeftModel

    console.print("[dim]loading prepared 4-bit checkpoint on the GPUs…[/dim]")
    model_cls = resolve_model_class(cfg.model.prepared_dir)
    model = model_cls.from_pretrained(
        cfg.model.prepared_dir,
        dtype=compute_dtype(cfg),
        device_map="balanced",
        max_memory=_gpu_memory_budget(),
    )
    console.print("[dim]folding LoRA weights into the quantized model…[/dim]")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    console.print("[dim]dequantizing NF4 → bf16 on the GPUs…[/dim]")
    # transformers' own dequantize path: replaces every Linear4bit with a
    # plain Linear (bf16), keeps the accelerate dispatch hooks, and drops the
    # quantization metadata so the saved config cannot claim 4-bit.
    model = model.dequantize(dtype=torch.bfloat16)
    model.config.use_cache = True
    return model


def merge_run(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    console: Console | None = None,
) -> Path:
    """Fold the adapter into the base, dequantize NF4 → bf16, save.

    A 27B model needs ~55 GB of bf16 memory, sharded across the GPUs
    (``device_map="balanced"``, GPUs only — the free-VRAM budget is printed
    by the run itself; if the device set is too small, it fails loudly).
    """
    console = console or Console()
    run_dir = Path(run_dir)
    cfg = Config.from_toml(run_dir / "config.resolved.toml")
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"No adapter found at {adapter_dir}")
    out = Path(out_dir) if out_dir else run_dir / "merged"
    out.mkdir(parents=True, exist_ok=True)

    model = _fold_adapter(cfg, adapter_dir, console)

    console.print(f"[dim]saving bf16 model to {out} …[/dim]")
    # `save_original_format=False`: the model is already in its final format
    # (plain bf16), so transformers must not try to revert the bnb load
    # conversion (re-packing bf16 weights into NF4 is not implemented).
    model.save_pretrained(out, max_shard_size="4GB", save_original_format=False)
    load_tokenizer(cfg).save_pretrained(out)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    console.print(
        f"[green]merge complete:[/green] {out} ({human_bytes(size)})\n"
        "  [dim]GGUF export for llama.cpp: docs/06-27b-runbook.md (step 6).[/dim]"
    )
    return out
