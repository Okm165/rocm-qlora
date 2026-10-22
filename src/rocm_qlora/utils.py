"""Small shared helpers: formatting, timing, files, seeds.

Everything here is dependency-light on purpose: these functions are used by
the dashboard, the charts, the reports and the tests alike.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Human-friendly formatting (used everywhere in the terminal UI)
# ---------------------------------------------------------------------------


def human_bytes(n: float | int | None) -> str:
    """12345678 -> '11.8 GiB'."""
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TiB"  # pragma: no cover - unreachable


def human_duration(seconds: float | None) -> str:
    """3672 -> '1h 01m 12s'. Handles negatives and None gracefully."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "n/a"
    if seconds < 0:
        return "-" + human_duration(-seconds)
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"


def human_rate(value: float | None, unit: str = "/s", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}{unit}"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def fmt_eta(seconds: float | None) -> str:
    """ETA + wall-clock completion time, e.g. '4m 12s (finishes ~14:32)'."""
    if seconds is None or seconds < 0:
        return "n/a"
    done = datetime.now() + timedelta(seconds=seconds)
    return f"{human_duration(seconds)} (finishes ~{done.strftime('%H:%M')})"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    """Sortable run-directory slug, e.g. '2026-10-22_0137'."""
    return datetime.now().strftime("%Y-%m-%d_%H%M")


class Stopwatch:
    """Tiny monotonic stopwatch with laps."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._last = self._t0

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def lap(self) -> float:
        now = time.monotonic()
        dt, self._last = now - self._last, now
        return dt


# ---------------------------------------------------------------------------
# Files / JSON
# ---------------------------------------------------------------------------


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    return a / b if b else default


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (like numpy.percentile, q in [0,100])."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (q / 100.0) * (len(sorted_values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def linear_slope(values: Iterable[float]) -> float:
    """Least-squares slope of y over x=0..n-1. Used for loss-trend arrows."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n < 2:
        return float("nan")
    x_mean = (n - 1) / 2.0
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (torch imported lazily so this module stays
    importable without the ML stack — e.g. in plotting-only tests)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is always present in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch may be unavailable in tests
        pass
