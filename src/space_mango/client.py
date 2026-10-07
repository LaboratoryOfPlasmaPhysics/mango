from __future__ import annotations

import warnings
from collections.abc import Mapping

import httpx
import polars as pl

from space_mango.errors import (
    MangoFilterError,
    ServerError,
    UnknownRegionError,
    did_you_mean,
    error_from_response,
)
from space_mango.timeparse import TimeLike, to_iso

DEFAULT_URL = "http://sciqlop.lpp.polytechnique.fr/mango/"


def _validate_filters(
    filters: Mapping[str, object],
    region: str,
    valid_names: set[str],
    other_regions: dict[str, set[str]],
) -> dict[str, float]:
    """Validate filter kwargs and return cleaned {param: float_value} dict."""
    cleaned: dict[str, float] = {}
    for key, value in filters.items():
        if not (key.endswith("_min") or key.endswith("_max")):
            raise MangoFilterError(
                f"Filter parameter '{key}' must end with '_min' or '_max' "
                f"(e.g. '{key}_min' or '{key}_max')."
            )
        name = key.rsplit("_", 1)[0]
        if name not in valid_names:
            hint = ""
            for other_region, other_names in other_regions.items():
                if name in other_names:
                    hint = f"\nHint: '{name}' is available for region '{other_region}'."
                    break
            available = ", ".join(sorted(valid_names))
            valid_params = [f"{n}_{s}" for n in sorted(valid_names) for s in ("min", "max")]
            raise MangoFilterError(
                f"'{key}' is not a valid filter for region '{region}'."
                f"{did_you_mean(key, valid_params)}\n"
                f"Available filters: {available}{hint}"
            )
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise MangoFilterError(f"Filter '{key}' value must be numeric, got {value!r}.")
        try:
            cleaned[key] = float(value)
        except ValueError:
            raise MangoFilterError(
                f"Filter '{key}' value must be numeric, got {value!r}."
            ) from None
    return cleaned


class MangoClient:
    """Client for the MANGO dataset API."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout, transport=transport)
        self._filter_cache: dict[str, set[str]] = {}
        self._regions: list[str] | None = None

    def _get(self, path: str, params: Mapping[str, object] | None = None) -> httpx.Response:
        try:
            r = self._http.get(path, params=params)  # pyright: ignore[reportArgumentType]
        except httpx.TransportError as e:
            raise ServerError(f"Could not reach the MANGO server at {self._base_url}: {e}") from e
        if r.status_code == 400:
            raise error_from_response(400, r.json())
        if r.status_code >= 400:
            raise ServerError(
                f"MANGO server answered HTTP {r.status_code} for {path}: {r.text[:300]}"
            )
        return r

    def _check_region(self, region: str) -> str:
        known = self.regions()
        if region not in known:
            raise UnknownRegionError(
                f"'{region}' is not a MANGO region.{did_you_mean(region, known)} "
                f"Regions: {', '.join(known)}."
            )
        return region

    def _ensure_filters_cached(self, region: str) -> None:
        if region not in self._filter_cache:
            r = self._get(f"/api/v1/regions/{region}/filters")
            self._filter_cache[region] = {f["name"] for f in r.json()}

    def _other_region_filters(self, exclude: str) -> dict[str, set[str]]:
        for region in self.regions():
            self._ensure_filters_cached(region)
        return {r: names for r, names in self._filter_cache.items() if r != exclude}

    def regions(self) -> list[str]:
        """List available regions."""
        if self._regions is None:
            self._regions = self._get("/api/v1/regions").json()
        return list(self._regions or [])

    def columns(self, region: str) -> list[str]:
        """List columns available in a region."""
        r = self._get(f"/api/v1/regions/{region}/columns")
        return r.json()

    def filters(self, region: str) -> list[dict[str, object]]:
        """List available filters for a region (name, column, unit, description)."""
        r = self._get(f"/api/v1/regions/{region}/filters")
        return r.json()

    def get_data(
        self,
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
    ) -> pl.DataFrame:
        """Query one region. Range filters are keyword arguments: bz_imf_max=-2, d_msh_max=0.3.

        start is inclusive, stop is exclusive. time_min/time_max are deprecated aliases
        (both inclusive).
        """
        self._check_region(region)
        if time_min is not None:
            warnings.warn("time_min is deprecated, use start=", FutureWarning, stacklevel=2)
            start = start if start is not None else time_min
        if time_max is not None:
            warnings.warn("time_max is deprecated, use stop= (exclusive)", FutureWarning, stacklevel=2)
        self._ensure_filters_cached(region)
        cleaned = _validate_filters(
            filters, region, self._filter_cache[region], self._other_region_filters(exclude=region)
        )
        params: dict[str, object] = {"format": "arrow", **{k: str(v) for k, v in cleaned.items()}}
        if limit is not None:
            params["limit"] = str(limit)
        if columns:
            params["columns"] = columns
        if spacecraft:
            params["spacecraft"] = spacecraft
        if (s := to_iso(start, param="start")) is not None:
            params["start"] = s
        if (s := to_iso(stop, param="stop")) is not None:
            params["stop"] = s
        if (s := to_iso(time_max, param="time_max")) is not None:
            params["time_max"] = s
        if sw_paired_only:
            params["sw_paired_only"] = "true"
        if normalized_only:
            params["normalized_only"] = "true"
        r = self._get(f"/api/v1/regions/{region}/data", params)
        return pl.read_ipc(r.content)
