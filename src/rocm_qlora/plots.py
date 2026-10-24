"""Charts (matplotlib) + the Markdown run report.

One module for everything visual:

* ``render_charts`` — the chart set written to ``<run>/charts/``; ``live=True``
  renders the cheap subset used for periodic snapshots during training.
* ``write_report`` — a Markdown report built purely from ``metrics.jsonl`` +
  ``summary.json``, so it works even for a crashed run.

Chart philosophy: annotated, not just plotted (epoch boundaries, warmup
shading, best-step markers, clip lines, coverage markers), dark theme,
degrade gracefully on partial data, headless-safe (Agg backend).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .monitor import (
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
from .utils import as_float, fmt_float, fmt_int, human_duration, percentile, read_json

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


def _watermark(fig: Figure, meta: dict[str, Any] | None) -> None:
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


def _epoch_lines(ax: Axes, train: Sequence[TrainRecord]) -> None:
    seen: set[int] = set()
    for rec in train:
        epi = math.floor(rec["epoch"])
        step = rec["step"]
        if epi >= 1 and epi not in seen:
            prev = [r for r in train if r.get("step", 0) < step]
            if prev and math.floor(prev[-1].get("epoch", 0)) == epi - 1:
                seen.add(epi)
                ax.axvline(step, color=COLORS["dim"], lw=0.8, ls=":", alpha=0.8)
                ax.annotate(
                    f"epoch {epi}",
                    xy=(step, ax.get_ylim()[1]),
                    xytext=(3, -12),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLORS["dim"],
                )


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------


def loss_curve(
    records: Sequence[MetricsRecord],
    out_path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
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
        steps = [r["step"] for r in train]
        losses = [r["loss"] for r in train]
        smoothed = ema(losses, span=20)
        warmup_end = meta.get("warmup_end_step") if meta else None
        if warmup_end:
            ax.axvspan(min(steps), warmup_end, color=COLORS["warmup"], alpha=0.55, lw=0)
            ax.annotate(
                "warmup",
                xy=((min(steps) + warmup_end) / 2, ax.get_ylim()[0]),
                ha="center",
                fontsize=8,
                color=COLORS["dim"],
            )
        ax.plot(steps, losses, color=COLORS["loss_raw"], lw=1.0, alpha=0.6, label="per-step loss")
        ax.plot(steps, smoothed, color=COLORS["loss_ema"], lw=2.2, label="EMA (span 20)")
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
        ev = [(r["step"], r["eval_loss"]) for r in eval_records]
        if ev:
            ax.plot(
                [e[0] for e in ev],
                [e[1] for e in ev],
                "o--",
                color=COLORS["eval"],
                lw=1.6,
                ms=6,
                label="eval loss (holdout)",
            )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("cross-entropy loss (nats/token)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.annotate(
            f"first {fmt_float(losses[0])}  →  last {fmt_float(losses[-1])}",
            xy=(0.01, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            color=COLORS["dim"],
        )
        _watermark(fig, meta)


def lr_schedule(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
    """Learning-rate schedule chart."""
    train = [r for r in select(records, "train") if r.get("lr") is not None]
    with _chart(out_path, figsize=(10, 3.2)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        ax.plot([r["step"] for r in train], [r["lr"] for r in train], color=COLORS["lr"], lw=2.0)
        warmup_end = meta.get("warmup_end_step") if meta else None
        if warmup_end:
            ax.axvspan(train[0]["step"], warmup_end, color=COLORS["warmup"], alpha=0.55, lw=0)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning-rate schedule (warmup → cosine decay)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        _watermark(fig, meta)


def grad_norm(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
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
        ax.set_title("Gradient norm — spikes are contained by clipping")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def throughput(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
    """Tokens/s chart."""
    train = [r for r in select(records, "train") if r.get("tok_per_s")]
    with _chart(out_path, figsize=(10, 3.4)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        steps = [r["step"] for r in train]
        ax.plot(steps, [r["tok_per_s"] for r in train], color=COLORS["tok"], lw=1.8, label="tokens/s (all GPUs)")
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


def vram(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
    """VRAM-per-GPU chart."""
    train = [r for r in select(records, "train") if r.get("gpu")]
    with _chart(out_path, figsize=(10, 3.6)) as (fig, ax):
        if not train:
            _no_data(ax)
            return
        steps = [r["step"] for r in train]
        n_gpus = len(train[-1]["gpu"])
        for gi in range(n_gpus):
            ys = [
                (r["gpu"][gi] or {}).get("mem_used_gb", float("nan")) if gi < len(r.get("gpu", [])) else float("nan")
                for r in train
            ]
            total = (train[-1]["gpu"][gi] or {}).get("mem_total_gb")
            ax.plot(
                steps,
                ys,
                color=COLORS["gpus"][gi % len(COLORS["gpus"])],
                lw=1.6,
                label=f"GPU {gi}" + (f" (of {total:.0f} GB)" if total else ""),
            )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("VRAM used (GB)")
        ax.set_title("VRAM per GPU — watch for a monotonic climb (leak)")
        ax.legend(loc="upper left", ncols=3)
        _watermark(fig, meta)


def gpu_health(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
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
            ax_t.plot(
                steps,
                [(r["gpu"][gi] or {}).get("temp_edge_c") if gi < len(r.get("gpu", [])) else None for r in train],
                color=c,
                lw=1.5,
                label=f"GPU {gi} edge °C",
            )
            ax_p.plot(
                steps,
                [(r["gpu"][gi] or {}).get("power_w") if gi < len(r.get("gpu", [])) else None for r in train],
                color=c,
                lw=1.5,
                label=f"GPU {gi} W",
            )
        ax_t.axhline(85, color=COLORS["best"], ls="--", lw=1.0)
        ax_t.axhline(95, color=COLORS["grad"], ls="--", lw=1.0)
        ax_t.annotate(
            "85 °C — warm but fine · 95 °C — throttling",
            xy=(steps[0], 95),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["dim"],
        )
        ax_t.set_ylabel("temperature (°C)")
        ax_t.set_title("GPU health — temperature and power")
        ax_t.legend(loc="upper left", ncols=3)
        ax_p.set_xlabel("optimizer step")
        ax_p.set_ylabel("power (W)")
        ax_p.legend(loc="upper left", ncols=3)
        _watermark(fig, meta)


def eval_curve(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
    """Eval-loss chart."""
    train = select(records, "train")
    evals = select(records, "eval")
    with _chart(out_path, figsize=(10, 4)) as (fig, ax):
        if not train and not evals:
            _no_data(ax)
            return
        if train:
            losses = [r["loss"] for r in train]
            steps = [r["step"] for r in train]
            ax.plot(steps, ema(losses, span=20), color=COLORS["loss_ema"], lw=2.0, label="train EMA")
        if evals:
            es = [r["step"] for r in evals]
            el = [r["eval_loss"] for r in evals]
            ax.plot(es, el, "o--", color=COLORS["eval"], lw=1.8, ms=7, label="eval (holdout)")
            for x, y in zip(es, el, strict=False):
                ax.annotate(
                    f"{y:.3f}", (x, y), xytext=(4, 6), textcoords="offset points", fontsize=8, color=COLORS["eval"]
                )
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss (nats/token)")
        ax.set_title("Train vs eval — the overfitting dial")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def epoch_bars(records: Sequence[MetricsRecord], out_path: str | Path, *, meta: dict[str, Any] | None = None) -> None:
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
        for i, (t, e) in enumerate(zip(train_means, eval_losses, strict=False)):
            if not math.isnan(t):
                ax.annotate(
                    f"{t:.3f}",
                    (i - 0.18, t),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=COLORS["fg"],
                )
            if not math.isnan(e):
                ax.annotate(
                    f"{e:.3f}",
                    (i + 0.18, e),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=COLORS["fg"],
                )
        ax.set_xticks(x, labels)
        ax.set_ylabel("loss (nats/token)")
        ax.set_title("Per-epoch losses — watch the train/eval gap")
        ax.legend(loc="upper right")
        _watermark(fig, meta)


def _panel_loss(ax: Axes, train: Sequence[TrainRecord], eval_records: Sequence[EvalRecord]) -> None:
    """Loss quadrant: raw + EMA + eval markers."""
    if not train:
        _no_data(ax)
        return
    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    ax.plot(steps, losses, color=COLORS["loss_raw"], lw=1.0, alpha=0.6)
    ax.plot(steps, ema(losses, span=20), color=COLORS["loss_ema"], lw=2.2)
    ev = [(r["step"], r["eval_loss"]) for r in eval_records]
    if ev:
        ax.plot([e[0] for e in ev], [e[1] for e in ev], "o--", color=COLORS["eval"], lw=1.4, ms=5)
    _epoch_lines(ax, train)
    ax.set_title("loss")
    ax.set_xlabel("step")


def _panel_lr(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Learning-rate quadrant."""
    lr_recs = [r for r in train if r.get("lr") is not None]
    if not lr_recs:
        _no_data(ax)
        return
    ax.plot([r["step"] for r in lr_recs], [as_float(r["lr"]) for r in lr_recs], color=COLORS["lr"], lw=2.0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_title("learning rate")


def _panel_throughput(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Tokens/s quadrant."""
    tput = [r for r in train if r.get("tok_per_s")]
    if not tput:
        _no_data(ax)
        return
    ax.plot([r["step"] for r in tput], [as_float(r["tok_per_s"]) for r in tput], color=COLORS["tok"], lw=1.8)
    ax.set_title("tokens/s")
    ax.set_xlabel("step")


def _panel_vram(ax: Axes, train: Sequence[TrainRecord]) -> None:
    """Per-GPU VRAM quadrant."""
    gpu_recs = [r for r in train if r.get("gpu")]
    if not gpu_recs:
        _no_data(ax)
        return
    for gi in range(len(gpu_recs[-1]["gpu"])):
        ys = [
            (r["gpu"][gi] or {}).get("mem_used_gb", float("nan")) if gi < len(r.get("gpu", [])) else float("nan")
            for r in gpu_recs
        ]
        ax.plot(
            [r["step"] for r in gpu_recs],
            ys,
            color=COLORS["gpus"][gi % len(COLORS["gpus"])],
            lw=1.5,
            label=f"GPU {gi}",
        )
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("VRAM used (GB)")
    ax.set_xlabel("step")


def dashboard(
    records: Sequence[MetricsRecord],
    out_path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
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


def token_length_hist(
    lengths: Sequence[int], out_path: str | Path, *, max_length: int, meta: dict[str, Any] | None = None
) -> None:
    """Data-profile chart: token length distribution + coverage at the cap."""
    with _chart(out_path, 1, 2, figsize=(13, 4.2)) as (fig, (ax, ax2)):
        if not lengths:
            _no_data(ax)
            return
        srt = sorted(lengths)
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

        caps = list(range(128, int(max(srt[-1] * 1.1, 512)) + 128, max(64, int(srt[-1] / 60))))
        cov = [sum(1 for length in srt if length <= c) / n for c in caps]
        ax2.plot(caps, cov, color=COLORS["tok"], lw=2.0)
        ax2.axvline(max_length, color=COLORS["fg"], lw=1.6)
        covered = sum(1 for length in srt if length <= max_length) / n
        ax2.annotate(
            f"{covered * 100:.1f}% covered at {max_length}",
            xy=(max_length, covered),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=9,
            color=COLORS["fg"],
        )
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("candidate max_length")
        ax2.set_ylabel("fraction of examples fitting")
        ax2.set_title("Coverage vs sequence cap")
        _watermark(fig, meta)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def render_charts(
    run_dir: str | Path, *, records: Sequence[MetricsRecord], meta: dict[str, Any] | None = None, live: bool = False
) -> list[Path]:
    """Render the chart set into ``<run>/charts`` (``live`` = cheap subset)."""
    run_dir = Path(run_dir)
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    evals = select(records, "eval")
    if live:
        specs = [
            ("dashboard.png", lambda p: dashboard(records, p, meta=meta, eval_records=evals)),
            ("loss.png", lambda p: loss_curve(records, p, meta=meta, eval_records=evals)),
        ]
    else:
        specs = [
            ("loss.png", lambda p: loss_curve(records, p, meta=meta, eval_records=evals)),
            ("eval.png", lambda p: eval_curve(records, p, meta=meta)),
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


def replot(run_dir: str | Path) -> list[Path]:
    """Regenerate every chart from metrics.jsonl (post-hoc / crashed runs)."""
    run_dir = Path(run_dir)
    records = read_metrics(run_dir)
    cfg = {}
    with contextlib.suppress(Exception):
        cfg = read_json(run_dir / "config.resolved.json")
    meta = {
        "run_name": run_dir.name,
        "model": str(cfg.get("model", {}).get("name", "")).split("/")[-1],
        "max_grad_norm": cfg.get("optim", {}).get("max_grad_norm", 1.0),
    }
    return render_charts(run_dir, records=records, meta=meta)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _interpretation_notes(records: Sequence[MetricsRecord], summary: RunSummary) -> list[str]:
    notes: list[str] = []
    first, last = summary.first_loss, summary.last_loss
    if first is not None and last is not None:
        drop = first - last
        if drop > 0.1:
            notes.append(
                f"**Loss fell {drop:.3f} nats** ({first:.3f} → {last:.3f}) — the adapter "
                f"learned something real on this dataset."
            )
        elif drop > 0:
            notes.append(
                f"Loss moved only {drop:.3f} nats — enough for format/style adaptation, "
                f"but consider more epochs or a higher LR."
            )
        else:
            notes.append(
                "Loss did **not** decrease — check docs/05-troubleshooting.md (LR, template, data formatting)."
            )
    ev_first, ev_last = summary.first_eval_loss, summary.last_eval_loss
    if ev_first is not None and ev_last is not None:
        gap = ev_last - ev_first
        if gap > 0.15:
            notes.append(
                f"**Eval loss rose {gap:+.3f}** — overfitting. Fewer epochs / more dropout / more data would help."
            )
        elif gap < -0.05:
            notes.append(
                f"Eval loss fell {gap:.3f} — the holdout improved, so the adapter generalizes rather than memorizes."
            )
        else:
            notes.append(
                "Eval loss is roughly flat — the holdout is tiny (~5%), so read the shape, not the third decimal."
            )
    slope = summary.loss_slope
    if slope is not None and slope > 1e-3:
        notes.append(
            "The last 50 steps show a *rising* loss trend — too-high LR, or the schedule ended while still exploring."
        )
    warns = [e for e in select(records, "event") if e.get("level") == "warn"]
    if warns:
        notes.append(
            f"{len(warns)} warning event(s) recorded — see the events table (gradient "
            f"spikes are usually benign if the EMA loss kept falling)."
        )
    if summary.avg_tok_per_s:
        notes.append(f"Throughput averaged ~{summary.avg_tok_per_s:,.0f} tokens/s across all GPUs.")
    return notes or ["Not enough data for interpretation — did the run finish?"]


def write_report(run_dir: str | Path) -> Path:
    """Write ``report.md`` from metrics.jsonl + summary.json (crash-tolerant)."""
    run_dir = Path(run_dir)
    records = read_metrics(run_dir)
    try:
        summary = RunSummary.from_dict(read_json(run_dir / "summary.json"))
    except Exception:
        summary = summarize(records)
    try:
        cfg = read_json(run_dir / "config.resolved.json")
    except Exception:
        cfg = {}

    charts = sorted((run_dir / "charts").glob("*.png"))
    events = select(records, "event")
    epochs = select(records, "epoch_end")

    lines: list[str] = []
    lines += _report_header(run_dir, cfg, summary)
    lines += _report_headline_table(summary)
    lines += _report_interpretation(records, summary)
    lines += _report_epoch_table(epochs)
    lines += _report_charts_section(charts)
    lines += _report_events_table(events)
    lines += _report_paths_section(run_dir)
    lines += _report_next_steps(run_dir)

    md = "\n".join(lines)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    return run_dir / "report.md"


def _report_header(run_dir: Path, cfg: dict[str, Any], summary: RunSummary) -> list[str]:
    """Title + model/dataset provenance line."""
    return [
        f"# Run report — {run_dir.name}\n",
        (
            f"*Model:* `{cfg.get('model', {}).get('name', summary.model or '?')}` · "
            f"*dataset:* `{cfg.get('data', {}).get('path', summary.dataset or '?')}` · "
            f"*GPUs:* {summary.world_size} · *dry run:* {summary.dry_run}\n"
        ),
    ]


def _report_headline_table(summary: RunSummary) -> list[str]:
    """The headline-numbers markdown table."""
    lines = [
        "## Headline numbers\n",
        "| metric | value |",
        "|---|---|",
        f"| optimizer steps | {fmt_int(summary.steps)} |",
        f"| loss (first → last) | {fmt_float(summary.first_loss)} → {fmt_float(summary.last_loss)} |",
        f"| best train loss | {fmt_float(summary.best_loss)} @ step {summary.best_step} |",
    ]
    if summary.best_eval_loss is not None:
        lines.append(
            f"| eval loss (first → last) | {fmt_float(summary.first_eval_loss)} → {fmt_float(summary.last_eval_loss)} |"
        )
    lines.append(f"| tokens seen | {fmt_int(summary.tokens_seen)} |")
    lines.append(f"| wall time | {human_duration(summary.total_seconds)} |")
    if summary.peak_vram_gb:
        lines.append(f"| peak VRAM (busiest GPU) | {summary.peak_vram_gb:.1f} GB |")
    lines.append("")
    return lines


def _report_interpretation(records: Sequence[MetricsRecord], summary: RunSummary) -> list[str]:
    """The "what the numbers mean" section."""
    lines = ["## What the numbers mean\n"]
    lines.extend(f"- {note}" for note in _interpretation_notes(records, summary))
    lines.append("")
    return lines


def _report_epoch_table(epochs: Sequence[EpochEndRecord]) -> list[str]:
    """Per-epoch markdown table (empty list when there are no epochs)."""
    if not epochs:
        return []
    lines = [
        "## Per-epoch summary\n",
        "| epoch | step | train loss (mean) | eval loss | elapsed |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {e.get('epoch')} | {e.get('step')} | {fmt_float(e.get('train_loss_mean'))} | "
        f"{fmt_float(e.get('eval_loss')) if e.get('eval_loss') is not None else '—'} | "
        f"{human_duration(e.get('elapsed_s'))} |"
        for e in epochs
    )
    lines.append("")
    return lines


def _report_charts_section(charts: Sequence[Path]) -> list[str]:
    """Embedded chart images (empty list when no charts were rendered)."""
    if not charts:
        return []
    lines = ["## Charts\n"]
    lines.extend(f"### {chart.stem}\n\n![{chart.stem}](charts/{chart.name})\n" for chart in charts)
    return lines


def _report_events_table(events: Sequence[EventRecord]) -> list[str]:
    """The events table (first 60 records; empty list when there are none)."""
    if not events:
        return []
    import json as _json

    lines = ["## Events\n", "| step | level | event | detail |", "|---|---|---|---|"]
    for e in events[:60]:
        detail = {k: v for k, v in e.items() if k not in {"type", "level", "event", "step", "ts"}}
        lines.append(
            f"| {e.get('step', '—')} | {e.get('level', 'info')} | {e.get('event')} | `{_json.dumps(detail)[:120]}` |"
        )
    lines.append("")
    return lines


def _report_paths_section(run_dir: Path) -> list[str]:
    """Where everything lives."""
    return [
        "## Where everything is\n",
        f"- metrics (JSONL): `{run_dir / 'metrics.jsonl'}`",
        f"- adapter: `{run_dir / 'adapter'}`",
        f"- charts: `{run_dir / 'charts'}`",
        f"- config snapshot: `{run_dir / 'config.resolved.toml'}`",
        f"- environment fingerprint: `{run_dir / 'env_fingerprint.json'}`",
        f"- per-rank logs: `{run_dir / 'logs'}`",
        "",
    ]


def _report_next_steps(run_dir: Path) -> list[str]:
    """Chat/merge commands + GGUF export recipe."""
    return [
        "## Continue from here\n",
        "```bash",
        f"uv run rocm-qlora chat --run {run_dir}     # talk to the adapter",
        f"uv run rocm-qlora merge --run {run_dir}    # merge to bf16",
        "```",
        "",
        "### Export to llama.cpp (GGUF)\n",
        "```bash",
        f"uv run rocm-qlora merge --run {run_dir}",
        "python llama.cpp/convert_hf_to_gguf.py <merged-dir> --outfile model-f16.gguf --outtype f16",
        "llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M",
        "```",
    ]
