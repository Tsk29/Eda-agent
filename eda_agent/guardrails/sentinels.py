"""Sentinel-value detection gate.

This is the deterministic gate that must run — and can flag/block — *before
any aggregate is computed* on a column. It is intentionally narrow and
literal, per the guardrails spec:

- Numeric sentinels -999, 9999, -1 count only if the rest of the column
  (i.e. excluding occurrences of that sentinel) is non-negative.
- String sentinels are exact matches on "NA", "N/A", and "" (empty string).

This is a different, stricter check than the profiler's broad
outlier-isolation heuristic in `eda_agent/profiler/profile.py`, which
additionally requires the value to be an isolated statistical outlier. The
two are deliberately not unified.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

NUMERIC_SENTINELS: tuple[int, ...] = (-999, 9999, -1)
STRING_SENTINELS: frozenset[str] = frozenset({"NA", "N/A", ""})


@dataclass(frozen=True)
class SentinelResult:
    """Sentinel values found in a single column, if any."""

    column: str
    has_sentinels: bool
    sentinel_values_found: list[str]
    affected_row_count: int


def detect_sentinels(series: pd.Series, column_name: str | None = None) -> SentinelResult:
    """Detect sentinel values in a column before any aggregate is computed.

    Args:
        series: The column to check.
        column_name: Name to report; defaults to `series.name`.

    Returns:
        A SentinelResult naming which sentinel tokens/values are present and
        how many rows they affect. Nulls are ignored (they are handled by
        `null_fraction`, not this gate).
    """
    name = column_name if column_name is not None else series.name
    name = str(name) if name is not None else "<unnamed>"

    non_null = series.dropna()
    if non_null.empty:
        return SentinelResult(
            column=name, has_sentinels=False, sentinel_values_found=[], affected_row_count=0
        )

    if pd.api.types.is_numeric_dtype(non_null):
        return _detect_numeric_sentinels(non_null, name)
    return _detect_string_sentinels(non_null, name)


def _detect_numeric_sentinels(series: pd.Series, name: str) -> SentinelResult:
    found: list[str] = []
    affected = 0
    for sentinel in NUMERIC_SENTINELS:
        mask = series == sentinel
        if not mask.any():
            continue
        rest = series[~mask]
        rest_is_non_negative = rest.empty or bool((rest >= 0).all())
        if rest_is_non_negative:
            found.append(str(sentinel))
            affected += int(mask.sum())
    return SentinelResult(
        column=name,
        has_sentinels=bool(found),
        sentinel_values_found=found,
        affected_row_count=affected,
    )


def _detect_string_sentinels(series: pd.Series, name: str) -> SentinelResult:
    str_series = series.astype(str)
    found: list[str] = []
    affected = 0
    for sentinel in sorted(STRING_SENTINELS):
        mask = str_series == sentinel
        if not mask.any():
            continue
        found.append('""' if sentinel == "" else sentinel)
        affected += int(mask.sum())
    return SentinelResult(
        column=name,
        has_sentinels=bool(found),
        sentinel_values_found=found,
        affected_row_count=affected,
    )
