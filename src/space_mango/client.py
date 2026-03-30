from __future__ import annotations

import polars as pl
import httpx

DEFAULT_URL = "http://sciqlop.lpp.polytechnique.fr/mango/"


class MangoFilterError(Exception):
    pass


def _validate_filters(
    filters: dict[str, object],
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
            raise MangoFilterError(
                f"'{key}' is not a valid filter for region '{region}'.\n"
                f"Available filters: {available}{hint}"
            )
        try:
            cleaned[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise MangoFilterError(
                f"Filter '{key}' value must be numeric, got {value!r}."
            ) from None
    return cleaned


class MangoClient:
    """Client for the MANGO dataset API."""

    def __init__(self, base_url: str = DEFAULT_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=120.0)
        self._filter_cache: dict[str, set[str]] = {}

    def _ensure_filters_cached(self, region: str) -> None:
        if region not in self._filter_cache:
            r = self._http.get(f"/api/v1/regions/{region}/filters")
            r.raise_for_status()
            self._filter_cache[region] = {f["name"] for f in r.json()}

    def _other_region_filters(self, exclude: str) -> dict[str, set[str]]:
        for region in self.regions():
            self._ensure_filters_cached(region)
        return {r: names for r, names in self._filter_cache.items() if r != exclude}

    def regions(self) -> list[str]:
        """List available regions."""
        r = self._http.get("/api/v1/regions")
        r.raise_for_status()
        return r.json()

    def columns(self, region: str) -> list[str]:
        """List columns available in a region."""
        r = self._http.get(f"/api/v1/regions/{region}/columns")
        r.raise_for_status()
        return r.json()

    def filters(self, region: str) -> list[dict]:
        """List available filters for a region (name, column, unit, description)."""
        r = self._http.get(f"/api/v1/regions/{region}/filters")
        r.raise_for_status()
        return r.json()

    def get_data(
        self,
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
            get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
        """
        self._ensure_filters_cached(region)
        valid_names = self._filter_cache[region]
        other = self._other_region_filters(exclude=region)
        cleaned = _validate_filters(filters, region, valid_names, other)

        params: dict[str, str | list[str]] = {"format": "arrow"}
        if limit is not None:
            params["limit"] = str(limit)
        if columns:
            params["columns"] = columns
        if spacecraft:
            params["spacecraft"] = spacecraft
        if time_min:
            params["time_min"] = time_min
        if time_max:
            params["time_max"] = time_max
        if sw_paired_only:
            params["sw_paired_only"] = "true"
        if normalized_only:
            params["normalized_only"] = "true"
        for key, value in cleaned.items():
            params[key] = str(value)

        r = self._http.get(f"/api/v1/regions/{region}/data", params=params)
        r.raise_for_status()
        return pl.read_ipc(r.content)
