"""Observability: live dashboard, metrics JSONL, health checks, summaries.

One module for "watch it learn" — it owns:

* ``MetricsWriter`` / ``read_metrics`` — the append-only ``metrics.jsonl``
  (the single source of truth every chart and report reads from),
* smoothing helpers (EMA, rolling median, trend arrow),
* ``GpuSampler`` — a background thread sampling rocm-smi (never blocks training),
* ``RichDashboard`` — the live terminal display (rank 0 only),
* ``MetricsCallback`` — the TrainerCallback wiring it all into training,
* ``summarize`` — the end-of-run numbers.

DDP note: UI + JSONL are rank-0 only. Other ranks write warnings to their own
log file (set up in ``train.py``).
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import Config
from .explain import ExplainerBook
from .hardware import GpuInfo, gpu_snapshot
from .utils import Stopwatch, fmt_eta, fmt_float, fmt_int, human_duration, linear_slope, now_iso, safe_div

METRICS_FILENAME = "metrics.jsonl"


# ---------------------------------------------------------------------------
# Metrics file
# ---------------------------------------------------------------------------


class MetricsWriter:
    """Append-only JSONL writer with immediate flush (crash-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 — long-lived append handle, closed in close()

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", now_iso())
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


def read_metrics(run_dir_or_file: str | Path) -> list[dict[str, Any]]:
    """Read all records; tolerate a truncated last line (killed process)."""
    path = Path(run_dir_or_file)
    if path.is_dir():
        path = path / METRICS_FILENAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def select(records: Sequence[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("type") == kind]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def ema(values: Sequence[float], span: int = 20) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    acc: float | None = None
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(acc if acc is not None else float("nan"))
            continue
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def rolling_median(values: Sequence[float], window: int = 15) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = sorted(v for v in values[lo:hi] if v is not None and not math.isnan(v))
        out.append(chunk[len(chunk) // 2] if chunk else float("nan"))
    return out


def trend_arrow(values: Sequence[float], window: int = 25, eps: float = 1e-3) -> str:
    tail = [v for v in values[-window:] if v is not None and not math.isnan(v)]
    slope = linear_slope(tail)
    if math.isnan(slope) or len(tail) < 5:
        return "…"
    if slope < -eps:
        return "↘ improving"
    if slope > eps:
        return "↗ rising"
    return "→ flat"


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    steps: int = 0
    epochs_done: float = 0.0
    first_loss: float | None = None
    last_loss: float | None = None
    best_loss: float | None = None
    best_step: int | None = None
    first_eval_loss: float | None = None
    last_eval_loss: float | None = None
    best_eval_loss: float | None = None
    loss_slope: float | None = None
    train_seconds: float = 0.0
    total_seconds: float = 0.0
    tokens_seen: int = 0
    avg_tok_per_s: float | None = None
    peak_vram_gb: float | None = None
    warnings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize(records: Sequence[dict[str, Any]]) -> RunSummary:
    s = RunSummary()
    train = select(records, "train")
    evals = select(records, "eval")
    epochs = select(records, "epoch_end")
    events = select(records, "event")

    if train:
        finite = [(r.get("step"), r.get("loss")) for r in train if r.get("loss") is not None]
        s.steps = int(train[-1].get("step") or len(train))
        if finite:
            s.first_loss, s.last_loss = finite[0][1], finite[-1][1]
            s.best_step, s.best_loss = min(finite, key=lambda t: t[1])
        s.loss_slope = linear_slope([loss for _, loss in finite[-50:]])
        s.tokens_seen = int(train[-1].get("tokens_seen") or 0)
        rates = [r.get("tok_per_s") for r in train if r.get("tok_per_s")]
        if rates:
            s.avg_tok_per_s = sum(rates) / len(rates)
        vrams = [r.get("vram_peak_gb") for r in train if r.get("vram_peak_gb")]
        if vrams:
            s.peak_vram_gb = max(vrams)
        times = [r.get("elapsed_s") for r in train if r.get("elapsed_s")]
        if times:
            s.train_seconds = times[-1]
    if evals:
        el = [r.get("eval_loss") for r in evals if r.get("eval_loss") is not None]
        if el:
            s.first_eval_loss, s.last_eval_loss, s.best_eval_loss = el[0], el[-1], min(el)
    if epochs:
        s.epochs_done = float(epochs[-1].get("epoch") or 0.0)
    if events:
        s.warnings = sum(1 for e in events if e.get("level") == "warn")
    return s


def health_violation(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(event, message)`` if a train record shows numerically broken
    training, else ``None``. Three failure modes are loud errors, never silent
    drift — a long run that quietly degrades into a garbage adapter is the
    worst possible outcome:

    * **non-finite loss** — the classic NaN/Inf loss spike.
    * **loss exactly 0.0** — TRL's fused loss is ``sum(-log_probs) / n`` where
      ignored positions contribute zeros, so a collapsed or all-masked batch
      reports exactly zero and the loss EMA would keep "improving" on zeros.
    * **finite loss with NaN grad_norm** — the forward was fine but the
      backward poisoned the gradients; the next optimizer step writes NaN
      into every weight. This slipped through a loss-only guard in a real
      27B run (step 6, loss 1.07, grad_norm NaN).
    """
    step = record.get("step")
    loss = record.get("loss")
    if loss is None:
        return None
    if loss != loss or loss in (float("inf"), float("-inf")):
        return "nan_loss", f"Loss became {loss} at step {step} — aborting. See docs/05-troubleshooting.md"
    if loss == 0.0:
        return (
            "zero_loss",
            (
                f"Loss is exactly 0.0 at step {step}: the batch had no trainable (assistant) tokens, or the loss "
                "collapsed. Aborting before the EMA masks the failure. See docs/05-troubleshooting.md"
            ),
        )
    grad = record.get("grad_norm")
    if grad is not None and not math.isfinite(grad):
        return (
            "nan_grad_norm",
            (
                f"grad_norm is {grad} at step {step} while the loss is finite ({loss:.4f}): the backward pass "
                "produced non-finite gradients and the next optimizer step would write NaN into the weights. "
                "Aborting. See docs/05-troubleshooting.md"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# GPU sampler (background thread)
# ---------------------------------------------------------------------------


class GpuSampler:
    def __init__(self, interval_s: float = 5.0) -> None:
        self.interval_s = interval_s
        self._gpus: list[GpuInfo] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-sampler")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = gpu_snapshot()
            if snap:
                with self._lock:
                    self._gpus = snap
            self._stop.wait(self.interval_s)

    def latest(self) -> list[GpuInfo]:
        with self._lock:
            return list(self._gpus)

    def stop(self) -> None:
        self._stop.set()


def _gpu_records(gpus: Sequence[GpuInfo]) -> list[dict[str, Any]]:
    return [
        {
            "index": g.index,
            "mem_used_gb": g.mem_used_b / 1e9,
            "mem_total_gb": g.mem_total_b / 1e9,
            "temp_edge_c": g.temp_edge_c,
            "temp_junction_c": g.temp_junction_c,
            "power_w": g.power_w,
            "util_pct": g.util_pct,
        }
        for g in gpus
    ]


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
    gpus: list[dict[str, Any]] = field(default_factory=list)
    health: list[str] = field(default_factory=list)
    phase: str = "starting"
    eval_loss: float | None = None


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

    def _render(self) -> Panel:
        s = self.state
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=34, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=False,
        )
        total = max(1, s.max_steps)
        progress.add_task(f"step {s.step}/{s.max_steps}", total=total, completed=min(s.step, total))

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

        gpu_table = Table.grid(padding=(0, 2))
        gpu_table.add_column(style="dim")
        gpu_table.add_column()
        for g in s.gpus:
            frac = safe_div(g["mem_used_gb"], g["mem_total_gb"], 0)
            color = "red" if frac > 0.92 else ("yellow" if frac > 0.8 else "green")
            temp = f"{g['temp_edge_c']:.0f}°C" if g.get("temp_edge_c") is not None else "?"
            power = f"{g['power_w']:.0f}W" if g.get("power_w") is not None else "?"
            util = f"{g['util_pct']:.0f}%" if g.get("util_pct") is not None else "?"
            gpu_table.add_row(
                f"GPU {g['index']}",
                f"[{color}]{g['mem_used_gb']:.1f}/{g['mem_total_gb']:.0f}GB[/{color}]  {temp}  {power}  util {util}",
            )

        body = Group(
            progress,
            Text(f"elapsed {human_duration(s.elapsed_s)} · ETA {fmt_eta(s.eta_s)}", style="dim"),
            metrics,
            Panel(gpu_table, title="GPUs", border_style="blue", padding=(0, 1)),
            Panel(
                Text.from_markup("\n".join(s.health or ["starting…"]), style="dim"),
                title="health",
                border_style="green",
                padding=(0, 1),
            ),
        )
        return Panel(body, title=f"[bold cyan]rocm-qlora[/bold cyan] · {s.run_name} · {s.phase}", border_style="cyan")


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


class MetricsCallback(TrainerCallback):
    """Writes metrics.jsonl, drives the dashboard, charts and health checks."""

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        *,
        console: Console,
        dashboard: RichDashboard | None,
        book: ExplainerBook,
        sampler: GpuSampler,
        writer: MetricsWriter,
        total_tokens_estimate: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.console = console
        self.dashboard = dashboard
        self.book = book
        self.sampler = sampler
        self.writer = writer
        self.total_tokens_estimate = total_tokens_estimate
        self.world_size = max(1, world_size)

        self.records: list[dict[str, Any]] = []
        self.stopwatch = Stopwatch()
        self._last_step_time: float | None = None
        self._step_times: list[float] = []
        self._last_tokens: int | None = None
        self._last_tokens_time: float | None = None
        self._epoch_losses: list[float] = []
        self._epoch_index = 0
        self._warned_grad_spike_at = -(10**9)
        self._loss_series: list[float] = []
        self._gpu_mem_series: list[float] = []
        self._snapshot_thread: threading.Thread | None = None
        self._last_snapshot_step = 0
        self.max_steps = 0
        self.warmup_end_step: int | None = None
        self._last_step: int | None = None

    # -- helpers -------------------------------------------------------------

    def _health_lines(self) -> list[str]:
        lines: list[str] = []
        trend = trend_arrow(self._loss_series)
        if trend == "↗ rising":
            lines.append("[red]✗ loss EMA is rising — check LR / data quality[/red]")
        elif self._loss_series:
            lines.append("[green]✓ loss trend healthy[/green]")
        gpus = self.sampler.latest()
        if gpus:
            hottest = max(gpus, key=lambda g: g.temp_edge_c or 0)
            if (hottest.temp_edge_c or 0) >= 90:
                lines.append(f"[red]✗ GPU {hottest.index} at {hottest.temp_edge_c:.0f}°C — throttling risk[/red]")
            else:
                lines.append(
                    f"[green]✓ temps OK[/green] [dim](hottest GPU {hottest.index}: {hottest.temp_edge_c:.0f}°C)[/dim]"
                )
            if max(safe_div(g.mem_used_gb, g.mem_total_gb, 0) for g in gpus) > 0.95:
                lines.append("[yellow]! VRAM >95% — OOM risk on a longer batch[/yellow]")
        if len(self._gpu_mem_series) > 20:
            growth = self._gpu_mem_series[-1] - self._gpu_mem_series[10]
            if growth > 1.5:
                lines.append(f"[yellow]! VRAM grew {growth:.1f} GB since step 10 — possible leak[/yellow]")
        return lines

    def _chart_meta(self) -> dict[str, Any]:
        return {
            "run_name": self.cfg.run.name,
            "model": self.cfg.model.name.split("/")[-1],
            "step": f"step {self._last_step or 0}",
            "max_grad_norm": self.cfg.optim.max_grad_norm,
            "warmup_end_step": self.warmup_end_step,
        }

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("world_size", self.world_size)
        self.records.append(record)
        self._last_step = record.get("step", self._last_step)
        self.writer.write(record)

    # -- Trainer hooks ---------------------------------------------------------

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.max_steps = state.max_steps
        self.stopwatch = Stopwatch()
        if self.cfg.optim.warmup_ratio and state.max_steps:
            self.warmup_end_step = int(state.max_steps * self.cfg.optim.warmup_ratio)
        self._write({"type": "event", "level": "info", "event": "train_start", "max_steps": state.max_steps})
        if self.dashboard:
            self.dashboard.update(phase="training", max_steps=state.max_steps, total_epochs=self.cfg.optim.epochs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        logs = logs or {}
        now = time.monotonic()

        if "eval_loss" in logs:
            record = {
                "type": "eval",
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0), 4),
                "eval_loss": float(logs["eval_loss"]),
            }
            self._write(record)
            if self.dashboard:
                self.dashboard.update(eval_loss=record["eval_loss"])
            return
        if "loss" not in logs:
            return

        step_dt = (now - self._last_step_time) if self._last_step_time else None
        if step_dt and step_dt > 0:
            self._step_times.append(step_dt)
        self._last_step_time = now
        step_time_ema = (sum(self._step_times[-20:]) / len(self._step_times[-20:])) if self._step_times else None

        tokens = logs.get("num_input_tokens_seen")
        if tokens is None:
            tokens = (
                state.global_step
                * self.cfg.optim.per_device_batch_size
                * self.cfg.optim.grad_accum
                * self.world_size
                * self.cfg.data.max_length
            )
        tok_per_s = None
        if self._last_tokens is not None and self._last_tokens_time is not None:
            dt = now - self._last_tokens_time
            if dt > 0:
                tok_per_s = (tokens - self._last_tokens) / dt
        self._last_tokens, self._last_tokens_time = tokens, now

        loss = float(logs["loss"])
        self._epoch_losses.append(loss)
        self._loss_series.append(loss)
        smoothed = ema(self._loss_series, span=20)
        loss_ema = smoothed[-1] if smoothed else None

        gpu_recs = _gpu_records(self.sampler.latest())
        vram_peak = max((g["mem_used_gb"] for g in gpu_recs), default=None)
        if vram_peak is not None:
            self._gpu_mem_series.append(vram_peak)

        remaining = max(0, self.max_steps - state.global_step)
        eta = step_time_ema * remaining if step_time_ema is not None else None

        record = {
            "type": "train",
            "step": state.global_step,
            "epoch": round(float(state.epoch or 0), 4),
            "loss": loss,
            "loss_ema": loss_ema,
            "lr": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "tok_per_s": tok_per_s,
            "samples_per_s": (tok_per_s / self.cfg.data.max_length) if tok_per_s else None,
            "tokens_seen": int(tokens),
            "elapsed_s": self.stopwatch.elapsed(),
            "eta_s": eta,
            "vram_peak_gb": vram_peak,
            "gpu": gpu_recs,
        }
        self._write(record)
        self._health_checks(record, control)

        if self.dashboard:
            best = min(
                (r["loss_ema"] for r in self.records if r.get("type") == "train" and r.get("loss_ema") is not None),
                default=None,
            )
            self.dashboard.update(
                step=state.global_step,
                epoch=float(state.epoch or 0),
                loss=loss,
                loss_ema=loss_ema,
                best_loss=best,
                loss_trend=trend_arrow(
                    [r["loss_ema"] for r in self.records if r.get("type") == "train" and r.get("loss_ema") is not None]
                ),
                lr=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
                tok_per_s=tok_per_s,
                samples_per_s=record["samples_per_s"],
                tokens_seen=int(tokens),
                total_tokens_estimate=self.total_tokens_estimate,
                elapsed_s=self.stopwatch.elapsed(),
                eta_s=eta,
                gpus=gpu_recs,
                health=self._health_lines(),
            )

        if state.global_step - self._last_snapshot_step >= self.cfg.logging.chart_snapshot_steps:
            self._last_snapshot_step = state.global_step
            self._schedule_snapshot()

    def _health_checks(self, record: dict[str, Any], control: TrainerControl) -> None:
        """Hard-stop on numerically broken training (see `health_violation`)."""
        violation = health_violation(record)
        if violation is not None:
            event, message = violation
            self.writer.write(
                {"type": "event", "level": "error", "event": event, "step": record["step"], "detail": message}
            )
            raise RuntimeError(message)
        grad = record.get("grad_norm")
        norms = [r["grad_norm"] for r in self.records if r.get("type") == "train" and r.get("grad_norm") is not None]
        if grad is not None and len(norms) >= 10:
            med = sorted(norms[-30:])[len(norms[-30:]) // 2]
            if med > 1e-3 and grad > 10 * med and record["step"] - self._warned_grad_spike_at > 50:
                self._warned_grad_spike_at = record["step"]
                self._write(
                    {
                        "type": "event",
                        "level": "warn",
                        "event": "grad_spike",
                        "step": record["step"],
                        "value": grad,
                        "median": med,
                    }
                )
                if self.console:
                    self.console.print(
                        f"[yellow]! gradient spike at step {record['step']}: {grad:.2f} vs median "
                        f"{med:.2f} — clipping contains it; watch the EMA loss.[/yellow]"
                    )

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._epoch_index += 1
        recent_eval = next((r for r in reversed(self.records) if r.get("type") == "eval"), None)
        record = {
            "type": "epoch_end",
            "epoch": self._epoch_index,
            "step": state.global_step,
            "train_loss_mean": (sum(self._epoch_losses) / len(self._epoch_losses)) if self._epoch_losses else None,
            "eval_loss": recent_eval.get("eval_loss") if recent_eval else None,
            "elapsed_s": self.stopwatch.elapsed(),
        }
        self._write(record)
        self._epoch_losses = []
        try:
            from .plots import render_charts

            render_charts(self.run_dir, records=self.records, meta=self._chart_meta(), live=False)
        except Exception:
            pass
        self._write(
            {
                "type": "event",
                "level": "info",
                "event": "epoch_end",
                "epoch": self._epoch_index,
                "step": state.global_step,
            }
        )

    def _schedule_snapshot(self) -> None:
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            return

        def job() -> None:
            try:
                from .plots import render_charts

                render_charts(self.run_dir, records=list(self.records), meta=self._chart_meta(), live=True)
            except Exception:
                pass

        self._snapshot_thread = threading.Thread(target=job, daemon=True, name="chart-snapshot")
        self._snapshot_thread.start()
