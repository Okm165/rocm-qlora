"""Small shared helpers: formatting, timing, files, seeds.

Everything here is dependency-light on purpose: these functions are used by
the progress lines, the charts, the reports and the tests alike.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Human-friendly formatting (used everywhere in the terminal UI)
# ---------------------------------------------------------------------------


def human_bytes(n: float | int | None) -> str:
    """12345678 -> '12.3 MB'. Decimal units.

    Every size this tool measures (rocm-smi VRAM, disk space, model files)
    and every size the docs quote is decimal, so file sizes are too. Host RAM
    is the one exception and is reported as GiB where it is shown.
    """
    if n is None:
        return "n/a"
    n = float(n)
    if not math.isfinite(n):
        return "n/a"  # a failed measurement is unknown, not "nan TB"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000.0
    return f"{n:.1f} TB"  # pragma: no cover - unreachable


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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    """Sortable run-directory slug, e.g. '2026-10-22_0137'."""
    return datetime.now().strftime("%Y-%m-%d_%H%M")


# ---------------------------------------------------------------------------
# Files / JSON
# ---------------------------------------------------------------------------


def read_json(path: str | Path) -> Any:
    """Read a JSON file (raises on parse errors)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write(path: str | Path, write: Callable[[TextIO], None]) -> None:
    """Write via a sibling tmp file + fsync + rename (crash-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON (UTF-8, indent=2), creating parent dirs."""

    def dump(fh: TextIO) -> None:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)

    _atomic_write(path, dump)


def write_text(path: str | Path, text: str) -> None:
    """Write text (UTF-8), creating parent dirs."""

    def dump(fh: TextIO) -> None:
        fh.write(text)

    _atomic_write(path, dump)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def as_float(value: float | None) -> float:
    """Coerce a filtered optional metric to float (callers drop the Nones)."""
    assert value is not None  # the caller's comprehension filtered Nones
    return value


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy semantics, q in [0, 100])."""
    if not values:
        return float("nan")
    import numpy as np

    return float(np.percentile(values, q))


def linear_slope(values: Iterable[float | None]) -> float:
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
    """Seed python/numpy/torch.

    torch is imported lazily: the import costs seconds and the plotting-only
    paths (report, charts) never need it. numpy is a hard dependency, so its
    seed is not optional — a swallowed import error would silently drop it.

    ``PYTHONHASHSEED`` is *not* set here: hash randomization is fixed at
    interpreter start, so only the launcher can export it in time (it does).
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
