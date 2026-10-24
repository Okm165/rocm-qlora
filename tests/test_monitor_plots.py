"""Observability + plotting smoke tests (synthetic metrics, Agg backend)."""

from __future__ import annotations

import math

import pytest

from rocm_qlora.metrics import (
    MetricsWriter,
    chart_meta,
    ema,
    read_metrics,
    rolling_median,
    summarize,
    trend_arrow,
)
from rocm_qlora.plots import render_charts, write_report


def synth_records(n=60, epochs=3):
    records = []
    per_epoch = n // epochs
    records.append({"type": "event", "level": "info", "event": "train_start", "max_steps": n})
    for i in range(1, n + 1):
        loss = 1.5 * math.exp(-i / 25) + 0.05
        rec = {
            "type": "train",
            "step": i,
            "epoch": round((i - 1) / per_epoch, 2),
            "loss": loss,
            "lr": 1e-4 * max(0.01, 1 - i / n),
            "grad_norm": 0.5 + 0.1 * math.sin(i),
            "tok_per_s": 800.0,
            "tokens_seen": i * 4096,
            "world_size": 3,
            "gpu": [
                {
                    "mem_used_gb": 19.0 + g * 0.1,
                    "mem_total_gb": 24.0,
                    "temp_edge_c": 60 + g,
                    "power_w": 250,
                }
                for g in range(3)
            ],
        }
        records.append(rec)
        if i % per_epoch == 0:
            records.append({"type": "eval", "step": i, "eval_loss": loss + 0.1})
            records.append(
                {
                    "type": "epoch_end",
                    "epoch": i // per_epoch,
                    "step": i,
                    "train_loss_mean": loss,
                    "elapsed_s": i * 2.0,
                }
            )
    records.append({"type": "event", "level": "warn", "event": "grad_spike", "step": 30, "value": 9.9, "median": 0.5})
    records.append({"type": "run_end", "total_seconds": n * 2.0, "dry_run": True})
    return records


def test_ema_and_trend():
    series = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6]
    smoothed = ema(series, span=5)
    assert len(smoothed) == len(series)
    assert smoothed[-1] < smoothed[0]
    assert trend_arrow(series) == "↘ improving"
    assert trend_arrow(list(reversed(series))) == "↗ rising"
    assert trend_arrow([1.0] * 10) == "→ flat"


def test_rolling_median():
    out = rolling_median([1.0, 2.0, 100.0, 2.0, 1.0], window=3)
    assert out[2] == 2.0  # spike suppressed


def test_health_guard_catches_broken_training():
    """The three numeric failure modes must be loud.

    Regression context: a real 27B run reported loss 1.07 with grad_norm NaN
    at step 6 (finite loss, poisoned backward), then exactly-0.0 losses for
    every following step while the EMA kept "improving".
    """
    from rocm_qlora.metrics import health_violation

    assert health_violation({"step": 1, "loss": 1.25, "grad_norm": 0.5}) is None

    event, message = health_violation({"step": 6, "loss": 1.07, "grad_norm": float("nan")})
    assert event == "nan_grad_norm" and "NaN" in message

    event, _ = health_violation({"step": 7, "loss": 0.0, "grad_norm": float("nan")})
    assert event == "zero_loss"

    event, _ = health_violation({"step": 8, "loss": float("inf"), "grad_norm": 1.0})
    assert event == "nan_loss"

    event, _ = health_violation({"step": 9, "loss": 1.0, "grad_norm": float("inf")})
    assert event == "nan_grad_norm"


def test_metrics_roundtrip_and_summary(tmp_path):
    records = synth_records()
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    for r in records:
        writer.write(r)
    writer.close()
    back = read_metrics(tmp_path)
    assert len(back) == len(records)
    summary = summarize(back)
    assert summary.steps == 60
    assert summary.first_loss > summary.last_loss
    assert summary.best_eval_loss is not None
    assert summary.peak_vram_gb == 19.2  # max mem_used_gb across the samples
    # world size comes from any record; dry-run + wall time from the run_end marker
    assert summary.world_size == 3
    assert summary.dry_run is True
    assert summary.total_seconds == 120.0


def test_metrics_writer_refuses_an_existing_stream(tmp_path):
    """Two runs in one directory would interleave two streams — fail loudly."""
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"type": "train", "step": 1, "loss": 1.0})
    writer.close()
    with pytest.raises(FileExistsError, match="two runs in one directory"):
        MetricsWriter(path)


def test_truncated_last_line_tolerated(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"type": "train", "step": 1, "loss": 1.0})
    writer.close()
    with open(path, "a") as fh:
        fh.write('{"type": "train", "step": 2, "loss":')  # killed mid-write
    assert len(read_metrics(tmp_path)) == 1


def test_render_charts_and_report(tmp_path):
    records = synth_records()
    meta = {"run_name": "test", "model": "toy", "max_grad_norm": 0.3, "warmup_end_step": 5}
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    (charts_dir / "stale-chart.png").write_bytes(b"old")  # a chart removed from the code
    written = render_charts(tmp_path, records=records, meta=meta, live=False)
    names = {p.name for p in written}
    assert {"loss.png", "overview.png", "vram.png"} <= names
    assert not (charts_dir / "stale-chart.png").exists()  # the full pass owns the set
    assert all(p.exists() and p.stat().st_size > 1000 for p in written)
    # live subset is smaller and cheap
    live = render_charts(tmp_path, records=records, meta=meta, live=True)
    assert {p.name for p in live} == {"overview.png", "loss.png"}
    # the report derives from the stream alone — write it, then report
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    for r in records:
        writer.write(r)
    writer.close()
    report = write_report(tmp_path)
    text = report.read_text()
    assert "Headline numbers" in text and "loss.png" in text
    assert "*GPUs:* 3" in text and "*dry run:* True" in text


def test_chart_meta_uses_the_trainers_warmup_rule():
    """ceil(max_steps * ratio) — flooring it would shade one step short."""
    records = synth_records(n=6, epochs=1)
    records.append({"type": "event", "level": "info", "event": "train_start", "max_steps": 252})
    meta = chart_meta(records, run_name="r", model="m", max_grad_norm=0.3, warmup_ratio=0.03)
    assert meta["warmup_end_step"] == 8  # ceil(7.56) = the trainer's value; floor gives 7
    assert meta["step"] == "step 6"


def test_summarize_does_not_let_nan_decide_the_best():
    """A NaN record is the abort being visible: last_loss keeps it, best_loss skips it."""
    records = [
        {
            "type": "train",
            "step": step,
            "epoch": 0.0,
            "loss": loss,
            "lr": 1e-4,
            "grad_norm": 1.0,
            "tok_per_s": 800.0,
            "tokens_seen": step * 100,
            "gpu": [{"mem_used_gb": mem, "mem_total_gb": 24.0, "temp_edge_c": 60, "power_w": 250}],
        }
        # the abort case: the NaN step is the last record in the stream
        for step, loss, mem in ((1, 1.5, 20.0), (2, 1.2, 21.0), (3, float("nan"), float("nan")))
    ]
    s = summarize(records)
    assert s.best_loss == 1.2 and s.best_step == 2
    assert s.peak_vram_gb == 21.0
    assert s.last_loss != s.last_loss  # NaN: the last step was the broken one


def test_metrics_writer_reopens_after_close_for_a_late_event(tmp_path):
    """The crash path: _finalize_run closes the stream, then _report_failure writes."""
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"type": "train", "step": 1, "loss": 1.0})
    writer.close()
    writer.write({"type": "event", "level": "error", "event": "abort", "detail": "late failure"})
    writer.close()
    assert [r["type"] for r in read_metrics(path)] == ["train", "event"]
