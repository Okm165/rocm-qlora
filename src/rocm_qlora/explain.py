"""Contextual explainers: short teaching panels shown when a concept first matters.

The book is deduplicated per run and disabled with ``--quiet``; each panel
points at the matching deep-dive in ``docs/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


@dataclass(frozen=True)
class Explainer:
    key: str
    title: str
    body: str
    links: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

EXPLAINERS: dict[str, Explainer] = {
    e.key: e
    for e in [
        Explainer(
            key="qlora",
            title="QLoRA in one panel",
            body=(
                "The base model is frozen in **4-bit NF4** (16 quantile-spaced bins, "
                "~0.5 bytes/param) and only a small **LoRA adapter** is trained: rank-`r` "
                "matrices added next to every linear layer, `ΔW = (α/r)·B·A`.\n\n"
                "**Expect**: ~4x less memory than bf16 training; quality close to full "
                "fine-tuning for style/format/domain adaptation, less so for injecting "
                "large amounts of new knowledge."
            ),
            links=(
                "QLoRA paper (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314",
                "docs/01-qlora-explained.md",
            ),
        ),
        Explainer(
            key="prepared_checkpoint",
            title="Why a pre-quantized checkpoint?",
            body=(
                "`prepare` quantizes the base model **once** into a local NF4 checkpoint. "
                "Loading 4-bit from a bf16 checkpoint would first stage the entire bf16 "
                "model in host RAM (~2 bytes/param, multiplied per data-parallel process); "
                "the prepared checkpoint loads small and fast, and its stored skip list is "
                "authoritative at load time."
            ),
            links=("docs/02-amd-rocm-notes.md",),
        ),
        Explainer(
            key="rocm_attention",
            title="Attention on RDNA3: SDPA + AOTriton",
            body=(
                "There is no `flash-attn` for RDNA3; the fast path is PyTorch **SDPA** "
                "dispatching to **AOTriton** kernels. `rocm-qlora doctor` probes each "
                "backend in a fresh process and records a verdict; the launcher applies it "
                "— disabling probe-failed backends and setting the AOTriton flag. If the "
                "fused backends fail, masked attention falls back to the math path "
                "(slower, same math)."
            ),
            links=(
                "docs/02-amd-rocm-notes.md",
                "https://rocm.docs.amd.com/en/latest/compatibility/pytorch-compatibility.html",
            ),
        ),
        Explainer(
            key="hybrid_attention",
            title="Hybrid attention: DeltaNet + full attention",
            body=(
                "Qwen3.8 mixes layer types: 3 of every 4 layers are **Gated DeltaNet** "
                "(linear attention, O(n) recurrent state) and every 4th is standard causal "
                "attention. The Triton kernels for the linear layers have spotty ROCm "
                "support, so this project uses transformers' pure-PyTorch fallbacks — "
                "correct and memory-frugal, somewhat slower."
            ),
            links=("docs/02-amd-rocm-notes.md",),
        ),
        Explainer(
            key="reading_loss",
            title="How to read the loss",
            body=(
                "Loss is the average negative log-likelihood per predicted token (nats): "
                "L≈1 means ~37% probability on each correct token.\n\n"
                "* per-step loss is noisy — watch the **EMA** line\n"
                "* direction matters more than the value; red flags are a flat line (LR "
                "too low, or template/data mismatch) and spikes that never recover (LR too "
                "high)\n"
                "* NaN/Inf loss or NaN grad_norm aborts the run automatically"
            ),
            links=("docs/04-reading-the-plots.md",),
        ),
        Explainer(
            key="eval",
            title="Train vs eval loss: the overfitting dial",
            body=(
                "A ~5% slice of the data is held out and evaluated each epoch — the "
                "generalization estimate.\n\n"
                "* both falling, small gap → healthy\n"
                "* train falling, eval flat/rising → overfitting: fewer epochs, more "
                "dropout, lower rank, more data\n"
                "* both high and flat → underfitting or a data/template problem\n\n"
                "With a few hundred examples the holdout is a few dozen samples — read the "
                "*shape*, not the third decimal."
            ),
            links=("docs/04-reading-the-plots.md",),
        ),
        Explainer(
            key="vram_budget",
            title="Where the VRAM goes",
            body=(
                "Per GPU, roughly: `NF4 weights + 16-bit parts (embeddings/head) + LoRA "
                "weights + optimizer states + activations`.\n\n"
                "* gradient checkpointing recomputes activations instead of storing them "
                "(slower, much smaller); `activation_offloading` parks the saved copies in "
                "pinned host RAM\n"
                "* TRL's fused chunked head never materializes full logits — which is also "
                "why `lm_head` stays 16-bit\n\n"
                "The preflight table prints this decomposition for the resolved config."
            ),
            links=("docs/04-reading-the-plots.md",),
        ),
        Explainer(
            key="next_steps",
            title="After training: chat, merge, GGUF",
            body=(
                "A run leaves a **LoRA adapter** next to the frozen base model. Three ways "
                "to use it:\n\n"
                "1. `rocm-qlora chat --run <dir>` — talk to it (fastest sanity check)\n"
                "2. `rocm-qlora merge --run <dir>` — fold into a bf16 model for serving\n"
                "3. convert the merged model to **GGUF** for `llama.cpp` (recipe in the run "
                "report)\n\n"
                "Keep the adapter: it weighs little, composes, and can be re-merged later."
            ),
            links=("README.md",),
        ),
    ]
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def kv_table() -> Table:
    """Two-column key/value table: dim right-aligned keys, plain values."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    return table


@final
class ExplainerBook:
    """Per-run registry of already-shown explainers (dedupe + quiet mode)."""

    def __init__(self, console: Console, enabled: bool = True) -> None:
        self.console = console
        self.enabled = enabled
        self.shown: set[str] = set()

    def show(self, key: str, *, force: bool = False) -> None:
        """Print the explainer with this key once per run (no-op when disabled)."""
        if not self.enabled or (key in self.shown and not force):
            return
        explainer = EXPLAINERS.get(key)
        if explainer is None:  # pragma: no cover - programming error
            raise KeyError(f"Unknown explainer: {key}")
        self.shown.add(key)
        self.console.print(render_explainer(explainer))

    def hint(self, text: str, *, icon: str = "💡") -> None:
        """One-line technical hint, visually distinct from explainers."""
        if not self.enabled:
            return
        self.console.print(f"[dim]{icon} {text}[/dim]")


def render_explainer(explainer: Explainer) -> Panel:
    """One explainer panel with its links."""
    body = explainer.body
    if explainer.links:
        links = "\n".join(f"- {link}" for link in explainer.links)
        body += f"\n\n**Read more**\n{links}"
    return Panel(
        Markdown(body),
        title=f"[bold]{explainer.title}[/bold]",
        border_style="cyan",
        padding=(0, 2),
    )
