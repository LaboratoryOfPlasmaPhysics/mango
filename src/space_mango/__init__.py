"""MANGO: Magnetospheric Atlas of Normalized Geospace Observations.

    import space_mango as mango
    mango.describe("magnetosheath")
    r = mango.magnetosheath.get_data(bz_imf_max=-2, d_msh_max=0.3)
    r.to_pandas()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from space_mango.cache import CacheInfo
from space_mango.client import DEFAULT_URL, MangoClient
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

if TYPE_CHECKING:
    import polars as pl

    from space_mango._regions_generated import MagnetosheathAPI, MagnetosphereAPI, SolarWindAPI
    from space_mango.timeparse import TimeLike

    magnetosphere: MagnetosphereAPI
    magnetosheath: MagnetosheathAPI
    solar_wind: SolarWindAPI

__all__ = [
    "DEFAULT_URL", "MangoClient", "MangoResult",
    "MangoError", "UnknownRegionError", "UnknownSpacecraftError", "UnknownColumnError",
    "MangoFilterError", "TimeParseError", "ServerError", "CacheMissError",
    "get_data", "regions", "columns", "filters", "describe", "spacecraft", "count",
    "search", "timeline", "cite", "dataset_info", "cache",
    "magnetosphere", "magnetosheath", "solar_wind",
]

_default_client: MangoClient | None = None
_REGION_NAMES = ("magnetosphere", "magnetosheath", "solar_wind")


def _get_default_client() -> MangoClient:
    global _default_client
    if _default_client is None:
        _default_client = MangoClient()
    return _default_client


def get_data(region: str, **kwargs: Any) -> MangoResult:
    """Query one region; see MangoClient.get_data, or use mango.<region>.get_data for
    tab-completion of every filter."""
    return _get_default_client().get_data(region, **kwargs)


def regions() -> list[str]:
    """Region names. Their definitions: mango.<region> or describe()."""
    return _get_default_client().regions()


def columns(region: str) -> list[str]:
    return _get_default_client().columns(region)


def filters(region: str) -> list[dict[str, object]]:
    return _get_default_client().filters(region)


def describe(region: str) -> pl.DataFrame:
    """Columns of a region with unit, frame, description and matching filter."""
    return _get_default_client().describe(region)


def spacecraft(region: str) -> pl.DataFrame:
    """Spacecraft in a region with first/last sample and row count."""
    return _get_default_client().spacecraft(region)


def count(region: str, **kwargs: Any) -> dict[str, float]:
    """n_rows and est_mb (after filtering) and download_mb_estimate (what the cached
    get_data would download) of a get_data call, without downloading. See MangoClient.count."""
    return _get_default_client().count(region, **kwargs)


def search(text: str) -> pl.DataFrame:
    """Find columns and filters by name, unit or description."""
    return _get_default_client().search(text)


def timeline(sc: str, start: TimeLike, stop: TimeLike, columns: list[str] | None = None) -> MangoResult:
    """All samples of one spacecraft over [start, stop) across regions (≤ 31 days)."""
    return _get_default_client().timeline(sc, start, stop, columns)


def cite() -> str:
    """BibTeX for the dataset version served."""
    return _get_default_client().cite()


def dataset_info() -> dict[str, Any]:
    return _get_default_client().dataset_info()


class _CacheHandle:
    """mango.cache.info() / mango.cache.clear() for the default client's on-disk cache."""

    def info(self) -> CacheInfo:
        return _get_default_client().cache_info()

    def clear(self) -> None:
        _get_default_client().cache_clear()


cache = _CacheHandle()


def __getattr__(name: str) -> Any:
    if name in _REGION_NAMES:
        from space_mango._regions_generated import REGION_APIS

        return REGION_APIS[name](_get_default_client)
    raise AttributeError(f"module 'space_mango' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REGION_NAMES))
