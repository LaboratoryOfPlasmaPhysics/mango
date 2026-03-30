from __future__ import annotations

from typing import TYPE_CHECKING

from space_mango.client import MangoClient, MangoFilterError

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "MangoClient",
    "MangoFilterError",
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
    time_min: str | None = None,
    time_max: str | None = None,
    sw_paired_only: bool = False,
    normalized_only: bool = False,
    limit: int | None = None,
    **filters: float,
) -> pl.DataFrame:
    """Query the MANGO dataset and return a polars DataFrame.

    Range filters are passed as keyword arguments:
        mango.get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
    """
    return _get_default_client().get_data(
        region,
        columns=columns,
        spacecraft=spacecraft,
        time_min=time_min,
        time_max=time_max,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        limit=limit,
        **filters,
    )


def regions() -> list[str]:
    """List available regions."""
    return _get_default_client().regions()


def columns(region: str) -> list[str]:
    """List columns available in a region."""
    return _get_default_client().columns(region)


def filters(region: str) -> list[dict]:
    """List available filters for a region."""
    return _get_default_client().filters(region)
