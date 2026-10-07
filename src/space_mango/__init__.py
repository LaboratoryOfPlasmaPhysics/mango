from __future__ import annotations

from space_mango.client import MangoClient
from space_mango.errors import (
    CacheMissError,
    MangoError,
    MangoFilterError,
    ServerError,
    TimeParseError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
)
from space_mango.result import MangoResult
from space_mango.timeparse import TimeLike

__all__ = [
    "MangoClient",
    "MangoResult",
    "CacheMissError",
    "MangoError",
    "MangoFilterError",
    "ServerError",
    "TimeParseError",
    "UnknownColumnError",
    "UnknownRegionError",
    "UnknownSpacecraftError",
    "get_data",
    "regions",
    "columns",
    "filters",
]

_default_client: MangoClient | None = None


def _get_default_client() -> MangoClient:
    global _default_client
    if _default_client is None:
        _default_client = MangoClient()
    return _default_client


def get_data(
    region: str,
    *,
    columns: list[str] | None = None,
    spacecraft: list[str] | None = None,
    start: TimeLike = None,
    stop: TimeLike = None,
    sw_paired_only: bool = False,
    normalized_only: bool = False,
    limit: int | None = None,
    time_min: TimeLike = None,
    time_max: TimeLike = None,
    **filters: float,
) -> MangoResult:
    """Query the MANGO dataset and return a MangoResult (data plus metadata).

    Range filters are passed as keyword arguments:
        mango.get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
    """
    legacy: dict[str, TimeLike] = {}
    if time_min is not None:
        legacy["time_min"] = time_min
    if time_max is not None:
        legacy["time_max"] = time_max
    return _get_default_client().get_data(
        region,
        columns=columns,
        spacecraft=spacecraft,
        start=start,
        stop=stop,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        limit=limit,
        **legacy,  # pyright: ignore[reportArgumentType]
        **filters,
    )


def regions() -> list[str]:
    """List available regions."""
    return _get_default_client().regions()


def columns(region: str) -> list[str]:
    """List columns available in a region."""
    return _get_default_client().columns(region)


def filters(region: str) -> list[dict[str, object]]:
    """List available filters for a region."""
    return _get_default_client().filters(region)
