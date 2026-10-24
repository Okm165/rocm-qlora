"""Terminal presentation: the shared key/value table + the rank-0 run panels.

Pure status output (dataset, model, wrap-up) — no teaching text, no
knowledge: that lives in docs/. The rank-0 parts run on rank 0 only; other
ranks log to their own files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from rich.table import Table

from .config import Config
from .dashboard import RichDashboard
from .monitor import GpuSampler, MetricsWriter
from .utils import fmt_int, human_duration

if TYPE_CHECKING:
    from datasets import Dataset
    from rich.console import Console
    from transformers import PreTrainedTokenizerBase

    from .data import ChatMessage, DatasetStats, TokenProfile
    from .model import ParameterReport
    from .monitor import RunSummary


def kv_table() -> Table:
    """Two-column key/value table: dim right-aligned keys, plain values."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    return table


def rank0_ui(cfg: Config, run_dir: Path, *, enabled: bool) -> Rank0UI | None:
    """Console, GPU sampler, metrics writer, live dashboard.

    Rank 0 only; non-zero ranks get `None` and log to their own rank file
    instead (`setup_logging`).
    """
    if not enabled:
        return None
    from rich.console import Console

    from .dashboard import RichDashboard
    from .monitor import GpuSampler, MetricsWriter

    console = Console()
    sampler = GpuSampler(cfg.logging.gpu_sample_seconds)
    writer = MetricsWriter(run_dir / "metrics.jsonl")
    dashboard = RichDashboard(console, refresh_per_second=cfg.run.console_refresh_per_second)
    dashboard.state.run_name = run_dir.name
    dashboard.update(phase="loading data")
    return Rank0UI(console, sampler, writer, dashboard)


class Rank0UI(NamedTuple):
    """The rank-0 terminal stack (``None`` on every other rank)."""

    console: Console
    sampler: GpuSampler
    writer: MetricsWriter
    dashboard: RichDashboard


def show_dataset_panel(
    ui: Rank0UI,
    cfg: Config,
    stats: DatasetStats,
    profile: TokenProfile,
    n_train: int,
    eval_ds: Dataset | None,
    examples: list[dict[str, list[ChatMessage]]],
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> None:
    """Rank-0 dataset summary panel and first-example preview."""
    from rich.panel import Panel

    from .data import render_preview

    console = ui.console
    pcts = profile.percentiles()
    table = kv_table()
    table.add_row(
        "examples", f"{stats.n_examples} (train {n_train}, eval {len(eval_ds) if eval_ds is not None else 0})"
    )
    table.add_row("turns", ", ".join(f"{k}: {v}" for k, v in stats.turns_per_example.items()))
    table.add_row(
        "tokens/example",
        f"p50={pcts.p50:.0f} p90={pcts.p90:.0f} p95={pcts.p95:.0f} p99={pcts.p99:.0f} max={pcts.p100:.0f}",
    )
    table.add_row(
        f"coverage @ max_length={cfg.data.max_length}",
        f"{profile.coverage_at(cfg.data.max_length) * 100:.1f}% "
        f"({profile.n_truncated} truncated, {profile.truncation_pct:.1f}%)",
    )
    table.add_row(
        "reasoning traces",
        f"{stats.think_blocks} assistant turns with  thinking ({stats.think_complete} complete)",
    )
    console.print(Panel(table, title="[bold]dataset[/bold]", border_style="cyan"))
    console.print(
        Panel(
            render_preview(tokenizer, examples[0], messages_field=cfg.data.messages_field),
            title="first example as the model sees it",
            border_style="blue",
        )
    )
    logger.info("dataset ready: %d train / %s eval", n_train, len(eval_ds) if eval_ds is not None else 0)


def show_model_panel(ui: Rank0UI, attn: str, disabled: list[str], report: ParameterReport, *, world_size: int) -> None:
    """Rank-0 model summary panel and model_loaded event."""
    from rich.panel import Panel

    console = ui.console
    table = kv_table()
    table.add_row("total params", fmt_int(report["total"]))
    table.add_row("trainable (LoRA)", f"{fmt_int(report['trainable'])}  ({report['pct']:.3f}%)")
    breakdown = " · ".join(f"{k}: {fmt_int(v)}" for k, v in list(report["by_module"].items())[:8])
    table.add_row("by module", breakdown)
    attn_note = f"{attn}  (doctor-verified where available)"
    if disabled:
        attn_note = f"{attn}  ({', '.join(disabled)} off per doctor verdict)"
    table.add_row("attention", attn_note)
    console.print(Panel(table, title="[bold]model[/bold]", border_style="cyan"))
    if ui.writer:
        from .monitor import EventRecord

        ui.writer.write(
            EventRecord(
                type="event",
                level="info",
                event="model_loaded",
                total_params=report["total"],
                trainable=report["trainable"],
                attn=attn,
                sdpa_disabled=disabled,
                world_size=world_size,
            )
        )


def _short_path(path: Path) -> str:
    """cwd-relative when possible — keeps wrap-up lines short and copy-pasteable."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def print_wrapup(console: Console, run_dir: Path, summary: RunSummary, *, dry_run: bool) -> None:
    """Run-complete panel: headline numbers + next-step hints."""
    from rich.panel import Panel

    table = kv_table()
    table.add_row("steps", fmt_int(summary.steps))
    nan = float("nan")
    table.add_row(
        "loss",
        f"{(summary.first_loss if summary.first_loss is not None else nan):.4f} → "
        f"{(summary.last_loss if summary.last_loss is not None else nan):.4f} "
        f"(best {(summary.best_loss if summary.best_loss is not None else nan):.4f} @ step {summary.best_step})",
    )
    if summary.last_eval_loss is not None:
        table.add_row(
            "eval loss",
            f"{(summary.first_eval_loss if summary.first_eval_loss is not None else nan):.4f} → "
            f"{summary.last_eval_loss:.4f} "
            f"(best {(summary.best_eval_loss if summary.best_eval_loss is not None else nan):.4f})",
        )
    table.add_row("tokens seen", fmt_int(summary.tokens_seen))
    table.add_row("wall time", human_duration(summary.total_seconds))
    table.add_row(
        "peak VRAM (max GPU)", f"{(summary.peak_vram_gb if summary.peak_vram_gb is not None else nan):.1f} GB"
    )
    table.add_row("adapter", _short_path(Path(summary.adapter_dir)))
    table.add_row("charts", _short_path(run_dir / "charts"))
    table.add_row("report", _short_path(run_dir / "report.md"))
    console.print(Panel(table, title="[bold green]run complete[/bold green]", border_style="green"))
    if dry_run:
        console.print("[yellow]This was a dry run (3 steps). Launch without --dry-run for the real thing.[/yellow]")
    short = _short_path(run_dir)
    console.print("[dim]next:[/dim]")
    console.print(f"  [bold]rocm-qlora chat   --run {short}[/bold]")
    console.print(f"  [bold]rocm-qlora merge  --run {short}[/bold]")
    console.print(f"  [bold]rocm-qlora report --run {short}[/bold]")
