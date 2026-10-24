"""Formatting + statistics atoms (pure functions, no GPU)."""

from __future__ import annotations

import math

from rocm_qlora.utils import (
    as_float,
    fmt_float,
    fmt_int,
    human_bytes,
    human_duration,
    linear_slope,
    percentile,
    timestamp_slug,
)


def test_human_bytes_is_decimal():
    """Sizes match rocm-smi / the docs (decimal), not binary units."""
    assert human_bytes(None) == "n/a"
    assert human_bytes(0) == "0 B"
    assert human_bytes(1536) == "1.5 KB"
    assert human_bytes(25_753_026_560) == "25.8 GB"  # the 24 GiB card as rocm-smi reports it


def test_human_duration():
    assert human_duration(None) == "n/a"
    assert human_duration(float("nan")) == "n/a"
    assert human_duration(45) == "45s"
    assert human_duration(90) == "1m 30s"
    assert human_duration(3725) == "1h 02m 05s"
    assert human_duration(-45) == "-45s"


def test_fmt_float_and_int():
    assert fmt_float(None) == "n/a"
    assert fmt_float(1.2345, 2) == "1.23"
    assert fmt_float(float("inf")) == "inf"
    assert fmt_float(1e-4) == "1.00e-04"
    assert fmt_int(1234567) == "1,234,567"
    assert fmt_int(None) == "n/a"


def test_percentile_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == 2.5
    assert math.isnan(percentile([], 50))


def test_linear_slope_drops_none_and_nan():
    assert linear_slope([1.0, 2.0, 3.0]) == 1.0
    assert linear_slope([1.0, None, 3.0]) == 2.0  # None dropped, x re-indexed
    assert math.isnan(linear_slope([1.0]))
    assert math.isnan(linear_slope([float("nan"), float("nan")]))


def test_as_float_narrows_optionals():
    assert as_float(2.5) == 2.5
    try:
        as_float(None)
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("as_float(None) must assert (callers filter Nones)")


def test_timestamp_slug():
    slug = timestamp_slug()
    assert len(slug) == 15 and slug[10] == "_"  # YYYY-MM-DD_HHMM
