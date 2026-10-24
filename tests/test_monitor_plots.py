"""Observability + plotting smoke tests (synthetic metrics, Agg backend)."""

from __future__ import annotations

import math

from rocm_qlora.monitor import (
    MetricsWriter,
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
    for i in range(1, n + 1):
        loss = 1.5 * math.exp(-i / 25) + 0.05
        rec = {
            "type": "train",
            "step": i,
            "epoch": round((i - 1) / per_epoch, 2),
            "loss": loss,
            "loss_ema": loss,
            "lr": 1e-4 * max(0.01, 1 - i / n),
            "grad_norm": 0.5 + 0.1 * math.sin(i),
            "tok_per_s": 800.0,
            "samples_per_s": 0.5,
            "tokens_seen": i * 4096,
            "elapsed_s": i * 2.0,
            "eta_s": (n - i) * 2.0,
            "vram_peak_gb": 19.5,
            "gpu": [
                {
                    "index": g,
                    "mem_used_gb": 19.0 + g * 0.1,
                    "mem_total_gb": 24.0,
                    "temp_edge_c": 60 + g,
                    "power_w": 250,
                    "util_pct": 90,
                }
                for g in range(3)
            ],
        }
        records.append(rec)
        if i % per_epoch == 0:
            records.append({"type": "eval", "step": i, "epoch": i / per_epoch, "eval_loss": loss + 0.1})
            records.append(
                {
                    "type": "epoch_end",
                    "epoch": i // per_epoch,
                    "step": i,
                    "train_loss_mean": loss,
                    "eval_loss": loss + 0.1,
                    "elapsed_s": i * 2.0,
                }
            )
    records.append({"type": "event", "level": "info", "event": "train_start", "step": 0})
    records.append({"type": "event", "level": "warn", "event": "grad_spike", "step": 30, "value": 9.9})
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
    from rocm_qlora.monitor import health_violation

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
    assert summary.peak_vram_gb == 19.5


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
    written = render_charts(tmp_path, records=records, meta=meta, live=False)
    names = {p.name for p in written}
    assert {"loss.png", "eval.png", "dashboard.png", "vram.png"} <= names
    assert all(p.exists() and p.stat().st_size > 1000 for p in written)
    # live subset is smaller and cheap
    live = render_charts(tmp_path, records=records, meta=meta, live=True)
    assert {p.name for p in live} == {"dashboard.png", "loss.png"}
    # report builds from the same records
    (tmp_path / "summary.json").write_text("{}")
    report = write_report(tmp_path)
    text = report.read_text()
    assert "Headline numbers" in text and "loss.png" in text


def test_run_summary_round_trip():
    """summary.json is a typed round-trip: RunSummary -> dict -> RunSummary."""
    from rocm_qlora.monitor import RunSummary

    records = synth_records(n=30)
    summary = summarize(records)
    summary.model = "org/model"
    summary.world_size = 3
    summary.dry_run = True

    restored = RunSummary.from_dict({**summary.to_dict(), "unknown_future_key": 1})
    assert restored == summary  # unknown keys ignored, every field preserved
    assert restored.model == "org/model" and restored.world_size == 3 and restored.dry_run is True
