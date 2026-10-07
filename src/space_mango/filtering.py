"""Filter expressions shared by the server (dataset.py) and the client cache (cache path).

Both sides call build_filter_exprs, so a query filters identically wherever it runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

import polars as pl

from space_mango.errors import QueryError
from space_mango.models import Region, filters_for

RESERVED_PARAMS = frozenset({
    "columns", "spacecraft", "sc", "start", "stop", "time_min", "time_max",
    "sw_paired_only", "normalized_only", "limit", "format",
})


def parse_time(value: str, *, param: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise QueryError(
            "bad_time", f"{param}={value!r} is not an ISO 8601 time (e.g. 2017-01-12T10:00:00)."
        ) from None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def time_window(
    start: str | None, stop: str | None, time_min: str | None, time_max: str | None
) -> tuple[datetime | None, datetime | None, bool]:
    """Resolve new (half-open) and legacy (inclusive) time parameters."""
    lo = start if start is not None else time_min
    start_dt = parse_time(lo, param="start") if lo is not None else None
    if stop is not None:
        return start_dt, parse_time(stop, param="stop"), False
    if time_max is not None:
        return start_dt, parse_time(time_max, param="time_max"), True
    return start_dt, None, False


def parse_range_params(region: Region | str, raw: Mapping[str, str | None]) -> dict[str, float]:
    allowed = filters_for(region)
    valid = [f"{n}_{s}" for n in allowed for s in ("min", "max")]
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key in RESERVED_PARAMS or value is None:
            continue
        name, _, suffix = key.rpartition("_")
        if suffix not in ("min", "max") or name not in allowed:
            raise QueryError(
                "unknown_filter",
                f"'{key}' is not a valid parameter for region '{Region(region).value}'.",
                valid,
            )
        try:
            out[key] = float(value)
        except ValueError:
            raise QueryError("bad_filter_value", f"Filter '{key}' must be numeric, got {value!r}.") from None
    return out


_FLAGS = {"sw_paired_only": "SW_pairing", "normalized_only": "Norma_pos"}


def build_filter_exprs(
    region: Region | str,
    available: set[str],
    *,
    spacecraft: Iterable[str] | None = None,
    start: datetime | None = None,
    stop: datetime | None = None,
    stop_inclusive: bool = False,
    sw_paired_only: bool = False,
    normalized_only: bool = False,
    ranges: Mapping[str, float] | None = None,
) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    if spacecraft:
        exprs.append(pl.col("SC").is_in(list(spacecraft)))
    if start is not None:
        exprs.append(pl.col("Time") >= start)
    if stop is not None:
        exprs.append(pl.col("Time") <= stop if stop_inclusive else pl.col("Time") < stop)
    for flag, wanted in (("sw_paired_only", sw_paired_only), ("normalized_only", normalized_only)):
        if not wanted:
            continue
        column = _FLAGS[flag]
        if column not in available:
            raise QueryError(
                "flag_unavailable",
                f"{flag}=true needs column '{column}', which region '{Region(region).value}' does not have.",
            )
        exprs.append(pl.col(column))
    catalog = filters_for(region)
    for key, value in (ranges or {}).items():
        name, _, suffix = key.rpartition("_")
        filt = catalog[name]
        if filt.column not in available:
            raise QueryError(
                "filter_column_missing",
                f"Filter '{name}' needs column '{filt.column}', "
                f"which region '{Region(region).value}' does not have.",
            )
        col = pl.col(filt.column)
        exprs.append(col >= value if suffix == "min" else col <= value)
    return exprs
