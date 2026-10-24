"""Live terminal dashboard (rank 0 only).

Pure presentation: renders the callback's state with rich, refreshed by a
ticker thread so elapsed/ETA keep moving between optimizer steps. The
metrics pipeline in ``monitor.py`` works without this module.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any, final

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .hardware import GpuMetric
from .utils import fmt_eta, fmt_float, fmt_int, human_duration, safe_div

# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------


@dataclass
class DashboardState:
    run_name: str = ""
    model: str = ""
    world_size: int = 1
    step: int = 0
    max_steps: int = 0
    epoch: float = 0.0
    total_epochs: float = 0.0
    loss: float | None = None
    loss_ema: float | None = None
    best_loss: float | None = None
    loss_trend: str = "…"
    lr: float | None = None
    grad_norm: float | None = None
    tok_per_s: float | None = None
    samples_per_s: float | None = None
    tokens_seen: int = 0
    total_tokens_estimate: int = 0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    gpus: list[GpuMetric] = field(default_factory=list)
    health: list[str] = field(default_factory=list)
    phase: str = "starting"
    eval_loss: float | None = None


@final
class RichDashboard:
    """The live terminal display (rank 0 only)."""

    def __init__(self, console: Console, *, refresh_per_second: float = 4.0) -> None:
        self.console = console
        self.state = DashboardState()
        self._live = Live(self._render(), console=console, refresh_per_second=refresh_per_second, transient=False)
        self._stop = threading.Event()

    def start(self) -> None:
        self._live.start(refresh=True)
        threading.Thread(target=self._tick, daemon=True, name="dashboard-ticker").start()

    def _tick(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._live.update(self._render())
            self._stop.wait(0.5)

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._live.stop()

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self.state, key, value)
        with contextlib.suppress(Exception):
            self._live.update(self._render())

    def _progress(self) -> Progress:
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=34, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=False,
        )
        total = max(1, self.state.max_steps)
        progress.add_task(
            f"step {self.state.step}/{self.state.max_steps}", total=total, completed=min(self.state.step, total)
        )
        return progress

    def _metrics_table(self) -> Table:
        s = self.state
        metrics = Table.grid(padding=(0, 2))
        metrics.add_column(justify="right", style="dim")
        metrics.add_column()
        trend_color = {"↘ improving": "green", "→ flat": "yellow", "↗ rising": "red"}.get(s.loss_trend, "dim")
        metrics.add_row(
            "loss",
            f"[bold]{fmt_float(s.loss)}[/bold]  [dim]EMA[/dim] {fmt_float(s.loss_ema)}  "
            f"[{trend_color}]{s.loss_trend}[/{trend_color}]",
        )
        metrics.add_row("best", fmt_float(s.best_loss))
        if s.eval_loss is not None:
            metrics.add_row("eval loss", f"[orange1]{fmt_float(s.eval_loss)}[/orange1]")
        metrics.add_row("lr", f"{s.lr:.3e}" if s.lr is not None else "n/a")
        metrics.add_row("grad norm", fmt_float(s.grad_norm, 3))
        metrics.add_row("throughput", f"{fmt_int(s.tok_per_s)} tok/s · {fmt_float(s.samples_per_s, 2)} samp/s")
        metrics.add_row(
            "tokens seen",
            fmt_int(s.tokens_seen) + (f" / ~{fmt_int(s.total_tokens_estimate)}" if s.total_tokens_estimate else ""),
        )
        metrics.add_row("epoch", f"{s.epoch:.2f} / {s.total_epochs:.0f}")
        return metrics

    def _gpu_table(self) -> Table:
        gpu_table = Table.grid(padding=(0, 2))
        gpu_table.add_column(style="dim")
        gpu_table.add_column()
        for g in self.state.gpus:
            frac = safe_div(g["mem_used_gb"], g["mem_total_gb"], 0)
            color = "red" if frac > 0.92 else ("yellow" if frac > 0.8 else "green")
            temp = f"{g['temp_edge_c']:.0f}°C" if g.get("temp_edge_c") is not None else "?"
            power = f"{g['power_w']:.0f}W" if g.get("power_w") is not None else "?"
            util = f"{g['util_pct']:.0f}%" if g.get("util_pct") is not None else "?"
            gpu_table.add_row(
                f"GPU {g['index']}",
                f"[{color}]{g['mem_used_gb']:.1f}/{g['mem_total_gb']:.0f}GB[/{color}]  {temp}  {power}  util {util}",
            )
        return gpu_table

    def _render(self) -> Panel:
        s = self.state
        body = Group(
            self._progress(),
            Text(f"elapsed {human_duration(s.elapsed_s)} · ETA {fmt_eta(s.eta_s)}", style="dim"),
            self._metrics_table(),
            Panel(self._gpu_table(), title="GPUs", border_style="blue", padding=(0, 1)),
            Panel(
                Text.from_markup("\n".join(s.health or ["starting…"]), style="dim"),
                title="health",
                border_style="green",
                padding=(0, 1),
            ),
        )
        return Panel(body, title=f"[bold cyan]rocm-qlora[/bold cyan] · {s.run_name} · {s.phase}", border_style="cyan")
