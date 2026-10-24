"""Observability: metrics JSONL, one-line progress, health checks, summaries.

One module for "watch it learn" — it owns:

* ``MetricsWriter`` / ``read_metrics`` — the append-only ``metrics.jsonl``
  (the single source of truth every chart and report reads from),
* smoothing helpers (EMA, rolling median, trend arrow),
* ``GpuSampler`` — a background thread sampling rocm-smi (never blocks training),
* ``MetricsCallback`` — the TrainerCallback wiring it all into training and
  printing one progress line per logged step (rank 0),
* ``summarize`` — the end-of-run numbers.

DDP note: console + JSONL are rank-0 only. Other ranks write warnings to their
own log file (set up in ``train.py``).
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, final, overload, override

from rich.console import Console
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import Config
from .hardware import GpuInfo, GpuMetric, gpu_snapshot
from .utils import as_float, fmt_eta, fmt_float, fmt_int, linear_slope, now_iso

METRICS_FILENAME = "metrics.jsonl"


# ---------------------------------------------------------------------------
# Metrics file
# ---------------------------------------------------------------------------


class TrainRecord(TypedDict):
    """A per-step training record (``type == "train"``)."""

    type: Literal["train"]
    step: int
    epoch: float
    loss: float
    loss_ema: float | None
    lr: float | None
    grad_norm: float | None
    tok_per_s: float | None
    tokens_seen: int | None
    eta_s: float | None
    vram_peak_gb: float | None
    gpu: list[GpuMetric]
    world_size: NotRequired[int]  # added by MetricsCallback._write
    ts: NotRequired[str]  # added by MetricsWriter.write


class EvalRecord(TypedDict):
    """One holdout evaluation (``type == "eval"``)."""

    type: Literal["eval"]
    step: int
    epoch: float
    eval_loss: float
    world_size: NotRequired[int]
    ts: NotRequired[str]


class EpochEndRecord(TypedDict):
    """Per-epoch aggregates (``type == "epoch_end"``)."""

    type: Literal["epoch_end"]
    epoch: int
    step: int
    train_loss_mean: float | None
    eval_loss: float | None
    elapsed_s: float
    world_size: NotRequired[int]
    ts: NotRequired[str]


class EventRecord(TypedDict):
    """Lifecycle + health events (``type == "event"``)."""

    type: Literal["event"]
    level: Literal["info", "warn", "error"]
    event: str
    step: NotRequired[int]
    epoch: NotRequired[int]
    detail: NotRequired[str]
    error: NotRequired[str]
    total_params: NotRequired[float]
    trainable: NotRequired[float]
    attn: NotRequired[str]
    sdpa_disabled: NotRequired[list[str]]
    value: NotRequired[float]
    median: NotRequired[float]
    max_steps: NotRequired[int]
    world_size: NotRequired[int]
    ts: NotRequired[str]


class RunEndRecord(TypedDict):
    """Final tally written once at the end of a run (``type == "run_end"``)."""

    type: Literal["run_end"]
    total_seconds: float
    steps: int
    best_loss: float | None
    best_eval_loss: float | None
    tokens_seen: int | None
    world_size: NotRequired[int]
    dry_run: NotRequired[bool]
    ts: NotRequired[str]


class ChartMeta(TypedDict, total=False):
    """Annotation context for chart rendering (see ``plots.render_charts``)."""

    run_name: str
    model: str
    step: str
    max_grad_norm: float
    warmup_end_step: int | None


#: One line of ``metrics.jsonl`` — the union of every record shape.
MetricsRecord = TrainRecord | EvalRecord | EpochEndRecord | EventRecord | RunEndRecord


@final
class MetricsWriter:
    """Append-only JSONL writer with immediate flush (crash-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 — long-lived append handle, closed in close()

    def write(self, record: MetricsRecord) -> None:
        if "ts" not in record:
            record["ts"] = now_iso()
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


def read_metrics(run_dir_or_file: str | Path) -> list[MetricsRecord]:
    """Read all records; tolerate a truncated last line (killed process)."""
    path = Path(run_dir_or_file)
    if path.is_dir():
        path = path / METRICS_FILENAME
    if not path.exists():
        return []
    records: list[MetricsRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@overload
def select(records: Sequence[MetricsRecord], kind: Literal["train"]) -> list[TrainRecord]: ...
@overload
def select(records: Sequence[MetricsRecord], kind: Literal["eval"]) -> list[EvalRecord]: ...
@overload
def select(records: Sequence[MetricsRecord], kind: Literal["epoch_end"]) -> list[EpochEndRecord]: ...
@overload
def select(records: Sequence[MetricsRecord], kind: Literal["event"]) -> list[EventRecord]: ...
@overload
def select(records: Sequence[MetricsRecord], kind: Literal["run_end"]) -> list[RunEndRecord]: ...
def select(records: Sequence[MetricsRecord], kind: str) -> Any:
    """Records of one type ('train'/'eval'/'epoch_end'/'event'/'run_end')."""
    return [r for r in records if r.get("type") == kind]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def ema(values: Sequence[float | None], span: int = 20) -> list[float]:
    """Exponential moving average of a series."""
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


def rolling_median(values: Sequence[float | None], window: int = 15) -> list[float]:
    """Rolling median over the last `window` values."""
    out: list[float] = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = sorted(v for v in values[lo:hi] if v is not None and not math.isnan(v))
        out.append(chunk[len(chunk) // 2] if chunk else float("nan"))
    return out


def trend_arrow(values: Sequence[float | None], window: int = 25, eps: float = 1e-3) -> str:
    """Human trend label ('↗ rising' etc.) from the recent slope."""
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
    """End-of-run numbers, derived from the metrics stream.

    ``adapter_dir`` is the one fact the records do not carry; the worker fills
    it live for the wrap-up lines. ``total_seconds`` comes from the run_end
    record — a live wrap-up passes it in instead (the record is written after).
    """

    steps: int = 0
    first_loss: float | None = None
    last_loss: float | None = None
    best_loss: float | None = None
    best_step: int | None = None
    first_eval_loss: float | None = None
    last_eval_loss: float | None = None
    best_eval_loss: float | None = None
    total_seconds: float = 0.0
    tokens_seen: int | None = None
    avg_tok_per_s: float | None = None
    peak_vram_gb: float | None = None
    warnings: int = 0
    world_size: int = 1
    dry_run: bool = False
    adapter_dir: str = ""


def summarize(records: Sequence[MetricsRecord]) -> RunSummary:
    """End-of-run numbers (best loss/step, tokens, VRAM) from the records."""
    s = RunSummary()
    train = select(records, "train")
    evals = select(records, "eval")
    events = select(records, "event")

    if train:
        finite = [(r["step"], r["loss"]) for r in train]
        s.steps = int(train[-1]["step"])
        if finite:
            s.first_loss, s.last_loss = finite[0][1], finite[-1][1]
            s.best_step, s.best_loss = min(finite, key=lambda t: t[1])
        seen = train[-1].get("tokens_seen")
        s.tokens_seen = int(seen) if seen is not None else None
        rates = [as_float(r["tok_per_s"]) for r in train if r.get("tok_per_s")]
        if rates:
            s.avg_tok_per_s = sum(rates) / len(rates)
        vrams = [as_float(r["vram_peak_gb"]) for r in train if r.get("vram_peak_gb")]
        if vrams:
            s.peak_vram_gb = max(vrams)
    if evals:
        el = [r["eval_loss"] for r in evals]
        if el:
            s.first_eval_loss, s.last_eval_loss, s.best_eval_loss = el[0], el[-1], min(el)
    if events:
        s.warnings = sum(1 for e in events if e.get("level") == "warn")
    ws = next((r.get("world_size") for r in reversed(records) if r.get("world_size")), None)
    if ws:
        s.world_size = int(ws)
    run_ends = select(records, "run_end")
    if run_ends:
        s.dry_run = bool(run_ends[-1].get("dry_run", False))
        s.total_seconds = float(run_ends[-1].get("total_seconds", 0.0))
    return s


def health_violation(record: TrainRecord) -> tuple[str, str] | None:
    """Return ``(event, message)`` if a train record shows numerically broken
    training, else ``None`` — three failure modes, each a loud abort:

    * **non-finite loss** — the classic NaN/Inf loss spike.
    * **loss exactly 0.0** — the fused loss divides masked positions in as
      zeros, so a collapsed or all-masked batch reports exactly zero while the
      EMA keeps "improving".
    * **finite loss with NaN grad_norm** — the forward was fine, the backward
      poisoned the gradients; the next optimizer step would write NaN into
      every weight (seen in a real 27B run at step 6: loss 1.07, grad NaN).
    """
    step = record["step"]
    loss = record["loss"]
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


@final
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


def _gpu_records(gpus: Sequence[GpuInfo]) -> list[GpuMetric]:
    return [
        GpuMetric(
            index=g.index,
            mem_used_gb=g.mem_used_gb,
            mem_total_gb=g.mem_total_gb,
            temp_edge_c=g.temp_edge_c,
            temp_junction_c=g.temp_junction_c,
            power_w=g.power_w,
            util_pct=g.util_pct,
        )
        for g in gpus
    ]


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


@final
class MetricsCallback(TrainerCallback):
    """Writes metrics.jsonl, prints progress lines, drives charts and health checks."""

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        *,
        console: Console,
        sampler: GpuSampler,
        writer: MetricsWriter,
        world_size: int = 1,
        chart_renderer: Callable[..., None],
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.console = console
        self.sampler = sampler
        self.writer = writer
        self.world_size = max(1, world_size)
        # Injected by train.py: keeps `monitor` out of the `plots` import cycle
        # while deferring matplotlib to the first render.
        self._chart_renderer = chart_renderer

        self.records: list[MetricsRecord] = []
        self._t0 = time.monotonic()
        self._last_step_time: float | None = None
        self._step_times: list[float] = []
        self._last_tokens: int | None = None
        self._last_tokens_time: float | None = None
        self._epoch_losses: list[float] = []
        self._epoch_index = 0
        self._warned_grad_spike_at = -(10**9)
        self._warned_health_at = -(10**9)
        self._loss_series: list[float] = []
        self._gpu_mem_series: list[tuple[int, float]] = []
        self._snapshot_thread: threading.Thread | None = None
        self._last_snapshot_step = 0
        self.max_steps = 0
        self.warmup_end_step: int | None = None
        self._last_step: int | None = None

    # -- helpers -------------------------------------------------------------

    def _print_health_warnings(self, record: TrainRecord) -> None:
        """Rare, loud warnings — each family at most once per 100 steps."""
        step = record["step"]
        if step - self._warned_health_at < 100:
            return
        warnings: list[str] = []
        if trend_arrow(self._loss_series) == "↗ rising":
            warnings.append("loss EMA is rising — check LR / data quality")
        gpus = self.sampler.latest()
        hottest = max(gpus, key=lambda g: g.temp_edge_c or 0, default=None)
        if hottest is not None and (hottest.temp_edge_c or 0) >= 90:
            warnings.append(f"GPU {hottest.index} at {hottest.temp_edge_c:.0f}°C — throttling risk")
        if len(self._gpu_mem_series) > 20:
            step0, mem0 = next((s for s in self._gpu_mem_series if s[0] >= 10), self._gpu_mem_series[0])
            growth = self._gpu_mem_series[-1][1] - mem0
            if growth > 1.5:
                warnings.append(f"VRAM grew {growth:.1f} GB since step {step0} — possible leak")
        if warnings:
            self._warned_health_at = step
            for warning in warnings:
                self.console.print(f"[yellow]! {warning}[/yellow]")

    def _print_progress(self, record: TrainRecord) -> None:
        """One compact status line per logged step (the live view, as text)."""
        parts = [
            f"step {record['step']}/{self.max_steps}",
            f"loss {fmt_float(record['loss'])}",
            f"EMA {fmt_float(record['loss_ema'])} {trend_arrow(self._loss_series)}",
        ]
        if record.get("lr") is not None:
            parts.append(f"lr {record['lr']:.2e}")
        if record.get("grad_norm") is not None:
            parts.append(f"grad {fmt_float(record['grad_norm'], 2)}")
        if record.get("tok_per_s"):
            parts.append(f"{fmt_int(record['tok_per_s'])} tok/s")
        if record.get("vram_peak_gb") is not None:
            parts.append(f"VRAM {record['vram_peak_gb']:.1f} GB")
        if record.get("eta_s") is not None:
            parts.append(f"ETA {fmt_eta(record['eta_s'])}")
        self.console.print("[dim]" + " · ".join(parts) + "[/dim]")
        self._print_health_warnings(record)

    def _chart_meta(self) -> ChartMeta:
        return {
            "run_name": self.cfg.run.name,
            "model": self.cfg.model.name.split("/")[-1],
            "step": f"step {self._last_step or 0}",
            "max_grad_norm": self.cfg.optim.max_grad_norm,
            "warmup_end_step": self.warmup_end_step,
        }

    def _write(self, record: MetricsRecord) -> None:
        if "world_size" not in record:
            record["world_size"] = self.world_size
        self.records.append(record)
        self._last_step = record.get("step", self._last_step)
        self.writer.write(record)

    # -- Trainer hooks ---------------------------------------------------------

    @override
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Reset the elapsed clock and record train_start."""
        self.max_steps = state.max_steps
        self._t0 = time.monotonic()
        if self.cfg.optim.warmup_ratio and state.max_steps:
            self.warmup_end_step = int(state.max_steps * self.cfg.optim.warmup_ratio)
        self._write(EventRecord(type="event", level="info", event="train_start", max_steps=state.max_steps))

    @override
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Per-step hot path: write the metrics record, print the progress line."""
        logs = logs or {}
        if "eval_loss" in logs:
            self._record_eval(state, logs["eval_loss"])
            return
        if "loss" not in logs:
            return

        record = self._track_step(time.monotonic(), logs, state)
        self._write(record)
        self._health_checks(record)
        self._print_progress(record)

        if state.global_step - self._last_snapshot_step >= self.cfg.logging.chart_snapshot_steps:
            self._last_snapshot_step = state.global_step
            self._schedule_snapshot()

    def _track_step(self, now: float, logs: dict[str, Any], state: TrainerState) -> TrainRecord:
        """Assemble one train record from the tracker helpers."""
        step_time_avg = self._track_step_timing(now)
        tokens, tok_per_s = self._track_tokens(now, logs)
        loss, loss_ema = self._track_loss(logs)
        gpu_recs, vram_peak = self._track_vram(state.global_step)
        remaining = max(0, self.max_steps - state.global_step)
        record: TrainRecord = {
            "type": "train",
            "step": state.global_step,
            "epoch": round(float(state.epoch or 0), 4),
            "loss": loss,
            "loss_ema": loss_ema,
            "lr": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "tok_per_s": tok_per_s,
            "tokens_seen": tokens,
            "eta_s": step_time_avg * remaining if step_time_avg is not None else None,
            "vram_peak_gb": vram_peak,
            "gpu": gpu_recs,
        }
        return record

    def _record_eval(self, state: TrainerState, eval_loss: float) -> None:
        record: EvalRecord = {
            "type": "eval",
            "step": state.global_step,
            "epoch": round(float(state.epoch or 0), 4),
            "eval_loss": float(eval_loss),
        }
        self._write(record)
        self.console.print(f"[dim]eval (step {record['step']}): loss {fmt_float(record['eval_loss'])}[/dim]")

    def _track_step_timing(self, now: float) -> float | None:
        """Mean of the last 20 step times (drives the ETA)."""
        step_dt = (now - self._last_step_time) if self._last_step_time is not None else None
        if step_dt and step_dt > 0:
            self._step_times.append(step_dt)
        self._last_step_time = now
        return (sum(self._step_times[-20:]) / len(self._step_times[-20:])) if self._step_times else None

    def _track_tokens(self, now: float, logs: dict[str, Any]) -> tuple[int | None, float | None]:
        """Tokens seen and tokens/s since the previous log.

        TRL reports ``num_input_tokens_seen`` (non-padding tokens) because
        ``sft._sft_kwargs`` sets ``include_num_input_tokens_seen``. If the key
        is ever absent the count stays *unknown* — no fabricated estimate.
        """
        seen = logs.get("num_input_tokens_seen")
        if seen is None:
            return None, None
        tokens = int(seen)
        tok_per_s = None
        if self._last_tokens is not None and self._last_tokens_time is not None:
            dt = now - self._last_tokens_time
            if dt > 0:
                tok_per_s = (tokens - self._last_tokens) / dt
        self._last_tokens, self._last_tokens_time = tokens, now
        return tokens, tok_per_s

    def _track_loss(self, logs: dict[str, Any]) -> tuple[float, float | None]:
        loss = float(logs["loss"])
        self._epoch_losses.append(loss)
        self._loss_series.append(loss)
        smoothed = ema(self._loss_series, span=20)
        return loss, (smoothed[-1] if smoothed else None)

    def _track_vram(self, step: int) -> tuple[list[GpuMetric], float | None]:
        gpu_recs = _gpu_records(self.sampler.latest())
        vram_peak = max((g["mem_used_gb"] for g in gpu_recs), default=None)
        if vram_peak is not None:
            self._gpu_mem_series.append((step, vram_peak))
        return gpu_recs, vram_peak

    def _health_checks(self, record: TrainRecord) -> None:
        """Hard-stop on numerically broken training (see `health_violation`)."""
        violation = health_violation(record)
        if violation is not None:
            event, message = violation
            self.writer.write(
                EventRecord(type="event", level="error", event=event, step=record["step"], detail=message)
            )
            raise RuntimeError(message)
        grad = record.get("grad_norm")
        norms = [as_float(r["grad_norm"]) for r in select(self.records, "train") if r.get("grad_norm") is not None]
        if grad is not None and len(norms) >= 10:
            med = sorted(norms[-30:])[len(norms[-30:]) // 2]
            if med > 1e-3 and grad > 10 * med and record["step"] - self._warned_grad_spike_at > 50:
                self._warned_grad_spike_at = record["step"]
                self._write(
                    EventRecord(
                        type="event",
                        level="warn",
                        event="grad_spike",
                        step=record["step"],
                        value=grad,
                        median=med,
                    )
                )
                if self.console:
                    self.console.print(
                        f"[yellow]! gradient spike at step {record['step']}: {grad:.2f} vs median "
                        f"{med:.2f} — clipping contains it; watch the EMA loss.[/yellow]"
                    )

    @override
    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Record epoch aggregates and refresh the charts."""
        self._epoch_index += 1
        recent_eval = next((r for r in reversed(self.records) if r.get("type") == "eval"), None)
        record: EpochEndRecord = {
            "type": "epoch_end",
            "epoch": self._epoch_index,
            "step": state.global_step,
            "train_loss_mean": (sum(self._epoch_losses) / len(self._epoch_losses)) if self._epoch_losses else None,
            "eval_loss": recent_eval.get("eval_loss") if recent_eval else None,
            "elapsed_s": time.monotonic() - self._t0,
        }
        self._write(record)
        self._epoch_losses = []
        with contextlib.suppress(Exception):
            self._chart_renderer(self.run_dir, records=self.records, meta=self._chart_meta(), live=False)
        self._write(
            EventRecord(
                type="event",
                level="info",
                event="epoch_end",
                epoch=self._epoch_index,
                step=state.global_step,
            )
        )

    def _schedule_snapshot(self) -> None:
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            return

        def job() -> None:
            with contextlib.suppress(Exception):
                self._chart_renderer(self.run_dir, records=list(self.records), meta=self._chart_meta(), live=True)

        self._snapshot_thread = threading.Thread(target=job, daemon=True, name="chart-snapshot")
        self._snapshot_thread.start()
