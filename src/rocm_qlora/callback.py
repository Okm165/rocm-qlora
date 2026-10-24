"""The training-time callback: progress lines, health checks, chart scheduling.

Wires the metrics stream (``metrics.py``) into the Trainer: one record per
logged step, the rank-0 progress line, the health guards, and chart renders
(``GpuSampler`` samples rocm-smi on a background thread so training never
blocks on it).

DDP note: console + JSONL are rank-0 only; other ranks write warnings to their
own log file (set up in ``train.py``).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, final, override

from rich.console import Console
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import Config
from .hardware import GpuInfo, GpuMetric, gpu_snapshot
from .metrics import (
    GPU_TEMP_WARN_C,
    ChartMeta,
    EpochEndRecord,
    EvalRecord,
    EventRecord,
    MetricsRecord,
    MetricsWriter,
    TrainRecord,
    ema,
    health_violation,
    select,
    trend_arrow,
)
from .utils import as_float, fmt_float, fmt_int


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
    """Project sampled GPU state onto the record fields the charts read."""
    return [
        GpuMetric(
            mem_used_gb=g.mem_used_gb,
            mem_total_gb=g.mem_total_gb,
            temp_edge_c=g.temp_edge_c,
            power_w=g.power_w,
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
        # Injected by train.py: keeps `callback` out of the `plots` import cycle
        # while deferring matplotlib to the first render.
        self._chart_renderer = chart_renderer

        self.records: list[MetricsRecord] = []
        self._t0 = time.monotonic()
        self._last_tokens: tuple[int, float] | None = None  # (tokens_seen, monotonic time)
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
        if hottest is not None and (hottest.temp_edge_c or 0) >= GPU_TEMP_WARN_C:
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

    def _print_progress(self, record: TrainRecord, *, loss_ema: float | None) -> None:
        """One compact status line per logged step (the live view, as text).

        ``loss_ema`` is live-only (the charts rebuild it from ``loss``); the
        VRAM peak is read straight off the record's GPU samples. No ETA: a
        linear extrapolation of step times is a prediction, and this tool
        reports what happened.
        """
        parts = [
            f"step {record['step']}/{self.max_steps}",
            f"loss {fmt_float(record['loss'])}",
            f"EMA {fmt_float(loss_ema)} {trend_arrow(self._loss_series)}",
        ]
        if record.get("lr") is not None:
            parts.append(f"lr {record['lr']:.2e}")
        if record.get("grad_norm") is not None:
            parts.append(f"grad {fmt_float(record['grad_norm'], 2)}")
        if record.get("tok_per_s"):
            parts.append(f"{fmt_int(record['tok_per_s'])} tok/s")
        if record["gpu"]:
            parts.append(f"VRAM {max(g['mem_used_gb'] for g in record['gpu']):.1f} GB")
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

    def _write(self, record: TrainRecord | EvalRecord | EpochEndRecord | EventRecord) -> None:
        """Append + write. Every record *this* method writes carries world_size
        (crash tolerance: a killed stream still says how many ranks ran);
        run_end is written directly by train._finalize_run and never passes here.
        """
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
            # The library's own rule (a ratio below 1 becomes ceil(steps * ratio));
            # chart_meta derives the same value for post-hoc renders.
            self.warmup_end_step = args.get_warmup_steps(state.max_steps)
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

        record, loss_ema = self._track_step(time.monotonic(), logs, state)
        self._write(record)
        self._health_checks(record)
        self._print_progress(record, loss_ema=loss_ema)

        if state.global_step - self._last_snapshot_step >= self.cfg.logging.chart_snapshot_steps:
            self._last_snapshot_step = state.global_step
            self._schedule_snapshot()

    def _track_step(self, now: float, logs: dict[str, Any], state: TrainerState) -> tuple[TrainRecord, float | None]:
        """Assemble one train record plus the progress line's live-only EMA.

        Returns ``(record, loss_ema)`` — the EMA is shown live but not
        persisted (see ``TrainRecord``).
        """
        tokens, tok_per_s = self._track_tokens(now, logs)
        loss, loss_ema = self._track_loss(logs)
        gpu_recs = self._track_vram(state.global_step)
        record: TrainRecord = {
            "type": "train",
            "step": state.global_step,
            "epoch": round(float(state.epoch or 0), 4),
            "loss": loss,
            "lr": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "tok_per_s": tok_per_s,
            "tokens_seen": tokens,
            "gpu": gpu_recs,
        }
        return record, loss_ema

    def _record_eval(self, state: TrainerState, eval_loss: float) -> None:
        record: EvalRecord = {
            "type": "eval",
            "step": state.global_step,
            "eval_loss": float(eval_loss),
        }
        self._write(record)
        self.console.print(f"[dim]eval (step {record['step']}): loss {fmt_float(record['eval_loss'])}[/dim]")

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
        if self._last_tokens is not None:
            prev_tokens, prev_time = self._last_tokens
            dt = now - prev_time
            if dt > 0:
                tok_per_s = (tokens - prev_tokens) / dt
        self._last_tokens = (tokens, now)
        return tokens, tok_per_s

    def _track_loss(self, logs: dict[str, Any]) -> tuple[float, float | None]:
        loss = float(logs["loss"])
        self._epoch_losses.append(loss)
        self._loss_series.append(loss)
        smoothed = ema(self._loss_series, span=20)
        return loss, (smoothed[-1] if smoothed else None)

    def _track_vram(self, step: int) -> list[GpuMetric]:
        """GPU samples for this step (busiest GPU also feeds the leak series)."""
        gpu_recs = _gpu_records(self.sampler.latest())
        peak = max((g["mem_used_gb"] for g in gpu_recs), default=None)
        if peak is not None:
            self._gpu_mem_series.append((step, peak))
        return gpu_recs

    def _health_checks(self, record: TrainRecord) -> None:
        """Hard-stop on numerically broken training (see `health_violation`)."""
        violation = health_violation(record)
        if violation is not None:
            event, message = violation
            self._write(EventRecord(type="event", level="error", event=event, step=record["step"], detail=message))
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
        record: EpochEndRecord = {
            "type": "epoch_end",
            "epoch": self._epoch_index,
            "step": state.global_step,
            "train_loss_mean": (sum(self._epoch_losses) / len(self._epoch_losses)) if self._epoch_losses else None,
            "elapsed_s": time.monotonic() - self._t0,
        }
        self._write(record)
        self._epoch_losses = []
        with contextlib.suppress(Exception):
            self._chart_renderer(self.run_dir, records=self.records, meta=self._chart_meta(), live=False)

    def _schedule_snapshot(self) -> None:
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            return

        def job() -> None:
            with contextlib.suppress(Exception):
                self._chart_renderer(self.run_dir, records=list(self.records), meta=self._chart_meta(), live=True)

        self._snapshot_thread = threading.Thread(target=job, daemon=True, name="chart-snapshot")
        self._snapshot_thread.start()
