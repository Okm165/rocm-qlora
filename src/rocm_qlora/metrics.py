"""Metrics: the JSONL stream, its records, smoothing, summaries, health.

The append-only ``metrics.jsonl`` is the single source of truth every chart
and report reads from. This module is the pure half of observability — no
Trainer, no console, no matplotlib:

* the record schemas (``TrainRecord`` / ``EvalRecord`` / ``EpochEndRecord`` /
  ``EventRecord`` / ``RunEndRecord``) and their union,
* ``MetricsWriter`` / ``read_metrics`` / ``select`` — I/O and filtering,
* smoothing helpers (EMA, rolling median, trend arrow),
* ``RunSummary`` / ``summarize`` — the end-of-run numbers,
* ``health_violation`` — the numerically-broken-training guard,
* ``ChartMeta`` / ``chart_meta`` — the annotation contract with ``plots``.

The training-time half (GPU sampler, progress lines, chart scheduling) lives
in ``callback.py``.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, final, overload

from .hardware import GpuMetric
from .utils import as_float, linear_slope, now_iso

METRICS_FILENAME = "metrics.jsonl"

#: EMA span for the live loss curve and the charts (docs/04's "EMA" line).
EMA_SPAN = 20
#: Edge-temperature warning threshold (°C): the run's ! warning and the
#: gpu_health chart's dashed line share it.
GPU_TEMP_WARN_C = 90


# ---------------------------------------------------------------------------
# Metrics file
# ---------------------------------------------------------------------------


class TrainRecord(TypedDict):
    """A per-step training record (``type == "train"``).

    Only facts that cannot be derived from another record in the stream:
    ``loss_ema``/``vram_peak_gb`` are computed live for the progress line
    (the charts rebuild the EMA from ``loss``; the VRAM peak is
    ``max(gpu[].mem_used_gb)``).
    """

    type: Literal["train"]
    step: int
    epoch: float
    loss: float
    lr: float | None
    grad_norm: float | None
    tok_per_s: float | None
    tokens_seen: int | None
    gpu: list[GpuMetric]
    world_size: NotRequired[int]  # added by MetricsCallback._write
    ts: NotRequired[str]  # human provenance: when the step happened


class EvalRecord(TypedDict):
    """One holdout evaluation (``type == "eval"``)."""

    type: Literal["eval"]
    step: int
    eval_loss: float
    world_size: NotRequired[int]
    ts: NotRequired[str]


class EpochEndRecord(TypedDict):
    """Per-epoch aggregates (``type == "epoch_end"``).

    ``step`` is the epoch's last optimizer step. The holdout eval for this
    epoch is the ``eval`` record with the *same* step — the Trainer evaluates
    right after ``on_epoch_end`` fires, so a copy taken here would be the
    previous epoch's eval.
    """

    type: Literal["epoch_end"]
    epoch: int
    step: int
    train_loss_mean: float | None
    elapsed_s: float
    world_size: NotRequired[int]
    ts: NotRequired[str]


class EventRecord(TypedDict):
    """Lifecycle + health events (``type == "event"``)."""

    type: Literal["event"]
    level: Literal["info", "warn", "error"]
    event: str
    step: NotRequired[int]
    detail: NotRequired[str]
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
    """Run end marker written once (``type == "run_end"``).

    Only the facts the per-step records cannot carry: wall time and whether
    this was a dry run. Steps, losses, tokens and world size are derived from
    the stream itself (``summarize``), so they are not duplicated here.
    """

    type: Literal["run_end"]
    total_seconds: float
    dry_run: bool
    ts: NotRequired[str]


class ChartMeta(TypedDict, total=False):
    """Annotation context for chart rendering (see ``plots.render_charts``)."""

    run_name: str
    model: str
    step: str
    max_grad_norm: float
    warmup_end_step: int | None


def chart_meta(
    records: Sequence[MetricsRecord], *, run_name: str, model: str, max_grad_norm: float, warmup_ratio: float = 0.0
) -> ChartMeta:
    """Chart annotations for a full render, reconstructed from the stream.

    The live callback annotates with the current step and the trainer's warmup
    end; a final or post-hoc render rebuilds both from the records (last step;
    ``max_steps`` from train_start with the ratio) — the trainer's own rule is
    ``ceil(max_steps * ratio)`` (``TrainingArguments.get_warmup_steps``), and
    flooring it would shade one step short.
    """
    meta: ChartMeta = {"run_name": run_name, "model": model, "max_grad_norm": max_grad_norm}
    train = select(records, "train")
    if train:
        meta["step"] = f"step {train[-1]['step']}"
    starts = [e for e in select(records, "event") if e.get("event") == "train_start"]
    max_steps = starts[-1].get("max_steps") if starts else None
    if warmup_ratio and max_steps:
        meta["warmup_end_step"] = math.ceil(max_steps * warmup_ratio)
    return meta


#: One line of ``metrics.jsonl`` — the union of every record shape.
MetricsRecord = TrainRecord | EvalRecord | EpochEndRecord | EventRecord | RunEndRecord


@final
class MetricsWriter:
    """JSONL writer with immediate flush (crash-safe). One stream per file.

    Opens exclusively: a second writer on the same path (two runs sharing a
    directory) fails loudly instead of silently interleaving two streams.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"metrics stream already exists: {self.path} (two runs in one directory?)")
        self._fh = open(self.path, "x", encoding="utf-8")  # noqa: SIM115 — long-lived handle, closed in close()

    def write(self, record: MetricsRecord) -> None:
        if self._fh.closed:  # a late failure event after close(): same stream, reopened
            self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 — same long-lived handle
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


def ema(values: Sequence[float | None], span: int = EMA_SPAN) -> list[float]:
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
    total_seconds: float | None = None  # None when the stream has no run_end record (crashed run)
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
        losses = [(r["step"], r["loss"]) for r in train]
        s.steps = int(train[-1]["step"])
        # first/last stay raw (a NaN at the end is the abort being visible);
        # "best" must not be decided by a non-finite value.
        s.first_loss, s.last_loss = losses[0][1], losses[-1][1]
        finite = [(step, loss) for step, loss in losses if math.isfinite(loss)]
        if finite:
            s.best_step, s.best_loss = min(finite, key=lambda t: t[1])
        seen = train[-1].get("tokens_seen")
        s.tokens_seen = int(seen) if seen is not None else None
        rates = [as_float(r["tok_per_s"]) for r in train if r.get("tok_per_s")]
        if rates:
            s.avg_tok_per_s = sum(rates) / len(rates)
        samples = [
            v for v in (as_float(g["mem_used_gb"]) for r in train for g in (r.get("gpu") or [])) if math.isfinite(v)
        ]
        if samples:
            s.peak_vram_gb = max(samples)
    if evals:
        el = [r["eval_loss"] for r in evals]
        s.first_eval_loss, s.last_eval_loss, s.best_eval_loss = el[0], el[-1], min(el)
    if events:
        s.warnings = sum(1 for e in events if e.get("level") == "warn")
    ws = next((r.get("world_size") for r in reversed(records) if r.get("world_size")), None)
    if ws:
        s.world_size = int(ws)
    run_ends = select(records, "run_end")
    if run_ends:
        s.dry_run = bool(run_ends[-1].get("dry_run", False))
        # total_seconds has been in every run_end record since the first
        # metrics version (checked against every stream in .outputs); only
        # dry_run is a later addition, hence the .get above.
        s.total_seconds = float(run_ends[-1]["total_seconds"])
    return s


def health_violation(record: TrainRecord) -> tuple[str, str] | None:
    """Return ``(event, message)`` if a train record shows numerically broken
    training, else ``None`` — three failure modes, each a loud abort:

    * **non-finite loss** — the classic NaN/Inf loss spike.
    * **loss exactly 0.0** — the fused loss divides masked positions in as
      zeros, so a collapsed or all-masked batch reports exactly zero while the
      EMA keeps "improving".
    * **finite loss with a non-finite grad_norm (NaN or ±Inf)** — the forward
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
