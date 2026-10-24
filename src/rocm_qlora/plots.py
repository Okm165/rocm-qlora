"""Charts (matplotlib) + the Markdown run report.

* ``render_charts`` — the chart set written to ``<run>/charts/``; ``live=True``
  renders the cheap subset used for periodic snapshots during training.
* ``write_report`` — a Markdown report built purely from the metrics stream
  (``metrics.jsonl`` + ``config.resolved.toml``), so it works even for a
  crashed run.

Charts are annotated where it helps (epoch boundaries, warmup shading, best
step, clip line), dark-themed, and headless-safe (Agg backend).
"""

from __future__ import annotations

import contextlib
import json
import math
import tomllib
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .monitor import (
    ChartMeta,
    EpochEndRecord,
    EvalRecord,
    EventRecord,
    MetricsRecord,
    RunSummary,
    TrainRecord,
    ema,
    read_metrics,
    rolling_median,
    select,
    summarize,
)
from .utils import as_float, fmt_float, fmt_int, human_duration, percentile

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

COLORS = {
    "bg": "#10141a",
    "panel": "#161c24",
    "fg": "#d7dee8",
    "dim": "#8b97a8",
    "grid": "#232b36",
    "loss_raw": "#3d5a80",
    "loss_ema": "#7fb2f0",
    "best": "#e0af68",
    "lr": "#e0af68",
    "grad": "#f7768e",
    "grad_med": "#f9a8b8",
    "tok": "#9ece6a",
    "eval": "#ff9e64",
    "warmup": "#2b3442",
    "gpus": ["#7aa2f7", "#bb9af7", "#7dcfff", "#ff9e64"],
}


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["fg"],
            "axes.titlecolor": COLORS["fg"],
            "text.color": COLORS["fg"],
            "xtick.color": COLORS["dim"],
            "ytick.color": COLORS["dim"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.labelcolor": COLORS["fg"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 110,
        }
    )


def _save(fig: Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _watermark(fig: Figure, meta: ChartMeta | None) -> None:
    if not meta:
        return
    text = " · ".join(str(v) for v in [meta.get("run_name"), meta.get("step"), meta.get("model")] if v)
    if text:
        fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color=COLORS["dim"])


def _no_data(ax: Axes, msg: str = "waiting for data…") -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color=COLORS["dim"], fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


@contextlib.contextmanager
def _chart(out_path: str | Path, *args: Any, **kwargs: Any) -> Generator[tuple[Figure, Any], None, None]:
    """Atomic chart block: theme → figure → guaranteed save on exit."""
    _apply_theme()
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        _save(fig, out_path)


def _warmup_span(ax: Axes, meta: ChartMeta | None, first_step: int) -> None:
    """Shade the warmup window when the run recorded it."""
    end = meta.get("warmup_end_step") if meta else None
    if not end:
        return
    ax.axvspan(first_step, end, color=COLORS["warmup"], alpha=0.55, lw=0)
    ax.annotate("warmup", xy=((first_step + end) / 2, ax.get_ylim()[0]), ha="center", fontsize=8, color=COLORS["dim"])


def _epoch_lines(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Dotted lines + labels at epoch boundaries."""
    prev_epoch = math.floor(train[0].get("epoch", 0)) if train else 0
    for rec in train:
        epoch = math.floor(rec.get("epoch", 0))
        if epoch > prev_epoch:
            ax.axvline(rec["step"], color=COLORS["dim"], lw=0.8, ls=":", alpha=0.8)
            ax.annotate(
                f"epoch {epoch}",
                xy=(rec["step"], ax.get_ylim()[1]),
                xytext=(3, -12),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["dim"],
            )
        prev_epoch = epoch


#: The GPU-metric keys the charts read from each train record.
GpuField = Literal["mem_used_gb", "temp_edge_c", "power_w"]


def _gpu_field(train: Sequence[TrainRecord], gi: int, field: GpuField) -> list[float]:
    """One GPU-metric series across steps (NaN where the GPU is absent)."""
    out: list[float] = []
    for rec in train:
        gpus = rec.get("gpu") or []
        value = gpus[gi].get(field) if gi < len(gpus) else None
        out.append(float(value) if value is not None else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------


def _loss_curves(ax: Axes, train: Sequence[TrainRecord], *, labels: bool) -> tuple[list[int], list[float]]:
    """Raw + EMA loss lines; returns (steps, EMA series) for further annotation."""
    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    smoothed = ema(losses, span=20)
    ax.plot(steps, losses, color=COLORS["loss_raw"], lw=1.0, alpha=0.6, label="per-step loss" if labels else None)
    ax.plot(steps, smoothed, color=COLORS["loss_ema"], lw=2.2, label="EMA (span 20)" if labels else None)
    return steps, smoothed


def _eval_overlay(ax: Axes, eval_records: Sequence[EvalRecord], *, label: str | None = None) -> None:
    """Dashed eval-loss overlay (shared by the loss chart and the dashboard chart)."""
    ev = [(r["step"], r["eval_loss"]) for r in eval_records]
    if ev:
        ax.plot([e[0] for e in ev], [e[1] for e in ev], "o--", color=COLORS["eval"], lw=1.6, ms=6, label=label)


def loss_curve(
    records: Sequence[MetricsRecord],
    out_path: str | Path,
    *,
    meta: ChartMeta | None = None,
    eval_records: Sequence[EvalRecord] = (),
    title: str = "Training loss",
) -> None:
    """Loss chart: raw + EMA (+ eval markers)."""
    train = select(records, "train")
    with _chart(out_path, figsize=(10, 5)) as (fig, ax):
        if not train:
            _no_data(ax)
            ax.set_title(title)
            return
        _warmup_span(ax, meta, train[0]["step"])
        steps, smoothed = _loss_curves(ax, train, labels=True)
        best_i = min(range(len(smoothed)), key=lambda i: smoothed[i])
        ax.scatter(
            [steps[best_i]],
            [smoothed[best_i]],
            color=COLORS["best"],
            zorder=5,
            s=42,
            label=f"best EMA {smoothed[best_i]:.3f}",
        )
        _epoch_lines(ax, train)
        _eval_overlay(ax, eval_records, label="eval loss (holdout)")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("cross-entropy loss (nats/token)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def _lr_series(ax: Axes, train: Sequence[TrainRecord], *, label: str | None = None) -> None:
    """Learning-rate line (sci-notation y-axis), shared by the lr and dashboard charts."""
    recs = [r for r in train if r.get("lr") is not None]
    ax.plot([r["step"] for r in recs], [as_float(r["lr"]) for r in recs], color=COLORS["lr"], lw=2.0, label=label)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def _tok_series(ax: Axes, train: Sequence[TrainRecord], *, label: str | None = None) -> None:
    """Tokens/s line, shared by the throughput and dashboard charts."""
    recs = [r for r in train if r.get("tok_per_s")]
    ax.plot(
        [r["step"] for r in recs], [as_float(r["tok_per_s"]) for r in recs], color=COLORS["tok"], lw=1.8, label=label
    )


def _vram_series(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Per-GPU VRAM lines, shared by the vram and dashboard charts."""
    for gi in range(len(train[-1]["gpu"])):
        total = (train[-1]["gpu"][gi] or {}).get("mem_total_gb")
        ax.plot(
            [r["step"] for r in train],
            _gpu_field(train, gi, "mem_used_gb"),
            color=COLORS["gpus"][gi % len(COLORS["gpus"])],
            lw=1.6,
            label=f"GPU {gi}" + (f" (of {total:.0f} GB)" if total else ""),
        )


def lr_schedule(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """Learning-rate schedule chart."""
    train = [r for r in select(records, "train") if r.get("lr") is not None]
    with _chart(out_path, figsize=(10, 3.2)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        _lr_series(ax, train)
        _warmup_span(ax, meta, train[0]["step"])
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning-rate schedule")
        _watermark(fig, meta)


def grad_norm(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """Gradient-norm chart with the clipping threshold."""
    train = [r for r in select(records, "train") if r.get("grad_norm") is not None]
    with _chart(out_path, figsize=(10, 3.4)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        steps = [r["step"] for r in train]
        norms = [r["grad_norm"] for r in train]
        ax.plot(steps, norms, color=COLORS["grad"], lw=1.2, alpha=0.75, label="grad norm (pre-clip)")
        ax.plot(steps, rolling_median(norms, window=15), color=COLORS["grad_med"], lw=2.0, label="rolling median")
        clip = meta.get("max_grad_norm") if meta else None
        if clip:
            ax.axhline(clip, color=COLORS["dim"], ls="--", lw=1.0)
            ax.annotate(
                f"clip at {clip}",
                xy=(steps[-1], clip),
                xytext=(-70, 6),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["dim"],
            )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("global grad norm")
        ax.set_title("Gradient norm (pre-clip)")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def throughput(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """Tokens/s chart."""
    train = [r for r in select(records, "train") if r.get("tok_per_s")]
    with _chart(out_path, figsize=(10, 3.4)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        steps = [r["step"] for r in train]
        _tok_series(ax, train, label="tokens/s (all GPUs)")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("tokens / second")
        if any(r.get("tokens_seen") for r in train):
            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.plot(
                steps,
                [r.get("tokens_seen", 0) / 1e6 for r in train],
                color=COLORS["dim"],
                lw=1.2,
                ls="--",
                label="cumulative (M tokens)",
            )
            ax2.set_ylabel("cumulative tokens (M)", color=COLORS["dim"])
        ax.set_title("Throughput")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def vram(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """VRAM-per-GPU chart."""
    train = [r for r in select(records, "train") if r.get("gpu")]
    with _chart(out_path, figsize=(10, 3.6)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        _vram_series(ax, train)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("VRAM used (GB)")
        ax.set_title("VRAM per GPU")
        ax.legend(loc="upper left", ncols=3)
        _watermark(fig, meta)


def gpu_health(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """GPU temperature/power chart."""
    train = [r for r in select(records, "train") if r.get("gpu")]
    with _chart(out_path, 2, 1, figsize=(10, 5), sharex=True) as (fig, (ax_t, ax_p)):
        if not train:
            _no_data(ax_t)
            _no_data(ax_p)
            return
        steps = [r["step"] for r in train]
        n_gpus = len(train[-1]["gpu"])
        for gi in range(n_gpus):
            c = COLORS["gpus"][gi % len(COLORS["gpus"])]
            ax_t.plot(steps, _gpu_field(train, gi, "temp_edge_c"), color=c, lw=1.5, label=f"GPU {gi} edge °C")
            ax_p.plot(steps, _gpu_field(train, gi, "power_w"), color=c, lw=1.5, label=f"GPU {gi} W")
        ax_t.axhline(85, color=COLORS["best"], ls="--", lw=1.0)
        ax_t.axhline(95, color=COLORS["grad"], ls="--", lw=1.0)
        ax_t.annotate(
            "85 / 95 °C thresholds",
            xy=(steps[0], 95),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["dim"],
        )
        ax_t.set_ylabel("temperature (°C)")
        ax_t.set_title("GPU health")
        ax_t.legend(loc="upper left", ncols=3)
        ax_p.set_xlabel("optimizer step")
        ax_p.set_ylabel("power (W)")
        ax_p.legend(loc="upper left", ncols=3)
        _watermark(fig, meta)


def epoch_bars(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: ChartMeta | None = None) -> None:
    """Per-epoch train/eval loss bars."""
    epochs = select(records, "epoch_end")
    with _chart(out_path, figsize=(8, 4)) as (fig, ax):
        if not epochs:
            _no_data(ax)
            return
        labels = [f"ep {int(e.get('epoch', i + 1))}" for i, e in enumerate(epochs)]
        train_means = [e.get("train_loss_mean") or float("nan") for e in epochs]
        eval_losses = [e.get("eval_loss") or float("nan") for e in epochs]
        x = list(range(len(labels)))
        ax.bar([i - 0.18 for i in x], train_means, width=0.36, color=COLORS["loss_ema"], label="train (epoch mean)")
        ax.bar([i + 0.18 for i in x], eval_losses, width=0.36, color=COLORS["eval"], label="eval (holdout)")
        ax.set_xticks(x, labels)
        ax.set_ylabel("loss (nats/token)")
        ax.set_title("Per-epoch losses")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def _panel_loss(ax: Axes, train: Sequence[TrainRecord], eval_records: Sequence[EvalRecord]) -> None:
    """Loss quadrant: raw + EMA + eval markers."""
    if not train:
        _no_data(ax)
        return
    _loss_curves(ax, train, labels=False)
    _eval_overlay(ax, eval_records)
    _epoch_lines(ax, train)
    ax.set_title("loss")
    ax.set_xlabel("step")


def _panel_lr(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Learning-rate quadrant."""
    lr_recs = [r for r in train if r.get("lr") is not None]
    if not lr_recs:
        _no_data(ax)
        return
    _lr_series(ax, lr_recs)
    ax.set_title("learning rate")


def _panel_throughput(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Tokens/s quadrant."""
    tput = [r for r in train if r.get("tok_per_s")]
    if not tput:
        _no_data(ax)
        return
    _tok_series(ax, tput)
    ax.set_title("tokens/s")
    ax.set_xlabel("step")


def _panel_vram(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Per-GPU VRAM quadrant."""
    gpu_recs = [r for r in train if r.get("gpu")]
    if not gpu_recs:
        _no_data(ax)
        return
    _vram_series(ax, gpu_recs)
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("VRAM used (GB)")
    ax.set_xlabel("step")


def dashboard(
    records: Sequence[MetricsRecord],
    out_path: str | Path,
    *,
    meta: ChartMeta | None = None,
    eval_records: Sequence[EvalRecord] = (),
) -> None:
    """2×2 live snapshot: loss, lr, throughput, VRAM."""
    with _chart(out_path, 2, 2, figsize=(13, 7)) as (fig, axes):
        fig.suptitle(f"Run dashboard — {meta.get('run_name', '') if meta else ''}", fontsize=12, fontweight="bold")
        train = select(records, "train")

        _panel_loss(axes[0][0], train, eval_records)
        _panel_lr(axes[0][1], train)
        _panel_throughput(axes[1][0], train)
        _panel_vram(axes[1][1], train)

        _watermark(fig, meta)


def _length_hist_panel(ax: Axes, srt: Sequence[int], max_length: int) -> None:
    """Histogram of token lengths with p50/p90/p99 markers and the cap line."""
    n = len(srt)
    ax.hist(srt, bins=min(60, max(10, n // 5)), color=COLORS["loss_ema"], alpha=0.85, edgecolor=COLORS["bg"])
    for q, c in ((50, COLORS["tok"]), (90, COLORS["best"]), (99, COLORS["grad"])):
        v = percentile(srt, q)
        ax.axvline(v, color=c, ls="--", lw=1.2)
        ax.annotate(
            f"p{q}={v:.0f}",
            xy=(v, ax.get_ylim()[1]),
            xytext=(3, -11),
            textcoords="offset points",
            fontsize=8,
            color=c,
        )
    ax.axvline(max_length, color=COLORS["fg"], lw=1.6)
    ax.annotate(
        f"max_length={max_length}",
        xy=(max_length, ax.get_ylim()[1] * 0.86),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["fg"],
    )
    ax.set_xlabel("tokens per example (chat-templated)")
    ax.set_ylabel("examples")
    ax.set_title("Token length distribution")


def _coverage_panel(ax: Axes, srt: Sequence[int], max_length: int) -> None:
    """Fraction of examples fitting under each candidate sequence cap."""
    n = len(srt)
    caps = list(range(128, int(max(srt[-1] * 1.1, 512)) + 128, max(64, int(srt[-1] / 60))))
    cov = [sum(1 for length in srt if length <= c) / n for c in caps]
    ax.plot(caps, cov, color=COLORS["tok"], lw=2.0)
    ax.axvline(max_length, color=COLORS["fg"], lw=1.6)
    covered = sum(1 for length in srt if length <= max_length) / n
    ax.annotate(
        f"{covered * 100:.1f}% covered at {max_length}",
        xy=(max_length, covered),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["fg"],
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("candidate max_length")
    ax.set_ylabel("fraction of examples fitting")
    ax.set_title("Coverage vs sequence cap")


def token_length_hist(
    lengths: Sequence[int], out_path: str | Path, *, max_length: int, meta: ChartMeta | None = None
) -> None:
    """Data-profile chart: token length distribution + coverage at the cap."""
    with _chart(out_path, 1, 2, figsize=(13, 4.2)) as (fig, (ax, ax2)):
        if not lengths:
            _no_data(ax)
            return
        srt = sorted(lengths)
        _length_hist_panel(ax, srt, max_length)
        _coverage_panel(ax2, srt, max_length)
        _watermark(fig, meta)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def render_charts(
    run_dir: str | Path, *, records: Sequence[MetricsRecord], meta: ChartMeta | None = None, live: bool = False
) -> list[Path]:
    """Render the chart set into ``<run>/charts`` (``live`` = cheap subset)."""
    run_dir = Path(run_dir)
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    if not live:
        # The full pass owns the chart set: drop stale PNGs (e.g. a chart
        # removed from the code) so the report only embeds current output.
        for stale in charts.glob("*.png"):
            stale.unlink()
    evals = select(records, "eval")
    if live:
        specs = [
            ("dashboard.png", lambda p: dashboard(records, p, meta=meta, eval_records=evals)),
            ("loss.png", lambda p: loss_curve(records, p, meta=meta, eval_records=evals)),
        ]
    else:
        specs = [
            ("loss.png", lambda p: loss_curve(records, p, meta=meta, eval_records=evals)),
            ("lr.png", lambda p: lr_schedule(records, p, meta=meta)),
            ("grad_norm.png", lambda p: grad_norm(records, p, meta=meta)),
            ("throughput.png", lambda p: throughput(records, p, meta=meta)),
            ("vram.png", lambda p: vram(records, p, meta=meta)),
            ("gpu_health.png", lambda p: gpu_health(records, p, meta=meta)),
            ("epochs.png", lambda p: epoch_bars(records, p, meta=meta)),
            ("dashboard.png", lambda p: dashboard(records, p, meta=meta, eval_records=evals)),
        ]
    written: list[Path] = []
    for fname, fn in specs:
        path = charts / fname
        try:
            fn(path)
            written.append(path)
        except Exception as exc:  # charts must never kill a run
            import sys

            print(f"[charts] {fname} failed: {exc}", file=sys.stderr)
    return written


def _read_config(run_dir: Path) -> dict[str, Any]:
    """Tolerant read of the run's config snapshot (old or partial runs)."""
    try:
        with open(run_dir / "config.resolved.toml", "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def replot(run_dir: str | Path) -> list[Path]:
    """Regenerate every chart from metrics.jsonl (post-hoc / crashed runs)."""
    run_dir = Path(run_dir)
    records = read_metrics(run_dir)
    cfg = _read_config(run_dir)
    meta: ChartMeta = {
        "run_name": run_dir.name,
        "model": str(cfg.get("model", {}).get("name", "")).split("/")[-1],
        "max_grad_norm": cfg.get("optim", {}).get("max_grad_norm", 1.0),
    }
    return render_charts(run_dir, records=records, meta=meta)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_report(run_dir: str | Path) -> Path:
    """Write ``report.md`` from the metrics stream (crash-tolerant)."""
    run_dir = Path(run_dir)
    records = read_metrics(run_dir)
    summary = summarize(records)
    cfg = _read_config(run_dir)

    lines = [
        f"# Run report — {run_dir.name}\n",
        (
            f"*Model:* `{cfg.get('model', {}).get('name') or '?'}` · "
            f"*dataset:* `{cfg.get('data', {}).get('path') or '?'}` · "
            f"*GPUs:* {summary.world_size} · *dry run:* {summary.dry_run}\n"
        ),
        *_headline_table(summary),
        *_epoch_table(select(records, "epoch_end")),
        *_charts_section(sorted((run_dir / "charts").glob("*.png"))),
        *_events_table(select(records, "event")),
        *_continue_section(run_dir),
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir / "report.md"


def _headline_table(summary: RunSummary) -> list[str]:
    """The headline-numbers markdown table."""
    rows: list[tuple[str, str]] = [
        ("optimizer steps", fmt_int(summary.steps)),
        ("loss (first → last)", f"{fmt_float(summary.first_loss)} → {fmt_float(summary.last_loss)}"),
        ("best train loss", f"{fmt_float(summary.best_loss)} @ step {summary.best_step}"),
    ]
    if summary.best_eval_loss is not None:
        rows.append(
            ("eval loss (first → last)", f"{fmt_float(summary.first_eval_loss)} → {fmt_float(summary.last_eval_loss)}")
        )
    rows += [("tokens seen", fmt_int(summary.tokens_seen)), ("wall time", human_duration(summary.total_seconds))]
    if summary.peak_vram_gb:
        rows.append(("peak VRAM (busiest GPU)", f"{summary.peak_vram_gb:.1f} GB"))
    if summary.avg_tok_per_s:
        rows.append(("throughput (mean)", f"{summary.avg_tok_per_s:,.0f} tokens/s"))
    if summary.warnings:
        rows.append(("warnings", str(summary.warnings)))
    lines = ["## Headline numbers\n", "| metric | value |", "|---|---|"]
    lines += [f"| {metric} | {value} |" for metric, value in rows]
    lines.append("")
    return lines


def _epoch_table(epochs: Sequence[EpochEndRecord]) -> list[str]:
    """Per-epoch markdown table (empty list when there are no epochs)."""
    if not epochs:
        return []
    lines = [
        "## Per-epoch summary\n",
        "| epoch | step | train loss (mean) | eval loss | elapsed |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {e.get('epoch')} | {e.get('step')} | {fmt_float(e.get('train_loss_mean'))} | "
        f"{fmt_float(e.get('eval_loss')) if e.get('eval_loss') is not None else '—'} | "
        f"{human_duration(e.get('elapsed_s'))} |"
        for e in epochs
    ]
    lines.append("")
    return lines


def _charts_section(charts: Sequence[Path]) -> list[str]:
    """Embedded chart images (empty list when no charts were rendered)."""
    if not charts:
        return []
    lines = ["## Charts\n"]
    lines += [f"### {chart.stem}\n\n![{chart.stem}](charts/{chart.name})\n" for chart in charts]
    return lines


def _events_table(events: Sequence[EventRecord]) -> list[str]:
    """The events table (first 60 records; empty list when there are none)."""
    if not events:
        return []
    lines = ["## Events\n", "| step | level | event | detail |", "|---|---|---|---|"]
    for e in events[:60]:
        detail = {k: v for k, v in e.items() if k not in {"type", "level", "event", "step", "ts"}}
        lines.append(
            f"| {e.get('step', '—')} | {e.get('level', 'info')} | {e.get('event')} | `{json.dumps(detail)[:120]}` |"
        )
    lines.append("")
    return lines


def _continue_section(run_dir: Path) -> list[str]:
    """Artifact paths + the follow-up commands (chat, merge)."""
    return [
        "## Where everything is\n",
        f"- metrics (JSONL): `{run_dir / 'metrics.jsonl'}`",
        f"- adapter: `{run_dir / 'adapter'}`",
        f"- charts: `{run_dir / 'charts'}`",
        f"- config snapshot: `{run_dir / 'config.resolved.toml'}`",
        f"- environment fingerprint: `{run_dir / 'env_fingerprint.json'}`",
        f"- per-rank logs: `{run_dir / 'logs'}`",
        "",
        "## Continue from here\n",
        "```bash",
        f"uv run rocm-qlora chat --run {run_dir}     # talk to the adapter",
        f"uv run rocm-qlora merge --run {run_dir}    # merge to bf16 (GGUF export: docs/06 step 6)",
        "```",
    ]
