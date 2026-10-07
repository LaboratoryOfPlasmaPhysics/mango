"""Turn the time types physicists use into the ISO strings the server expects (naive UTC)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from space_mango.errors import TimeParseError

TimeLike = str | datetime | date | None

_PARTIAL_FORMATS = ("%Y-%m", "%Y")


def _naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo is not None else dt


def to_iso(value: object, *, param: str) -> str | None:
    """Accept str (ISO 8601, or 'YYYY-MM' / 'YYYY'), datetime, date, pandas.Timestamp,
    numpy.datetime64. Timezone-aware values are converted to UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):  # includes pandas.Timestamp
        return _naive_utc(value).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).isoformat()
    if type(value).__name__ == "datetime64":  # numpy, without importing numpy
        return to_iso(str(value.astype("datetime64[us]")), param=param)  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(value, str):
        try:
            return _naive_utc(datetime.fromisoformat(value)).isoformat()
        except ValueError:
            pass
        for fmt in _PARTIAL_FORMATS:
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue
    raise TimeParseError(
        f"{param}={value!r} is not a time. Use an ISO 8601 string ('2017-01-12T10:00'), "
        "'2017-01', a datetime, a pandas.Timestamp or a numpy.datetime64."
    )
