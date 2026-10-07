from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import polars as pl

from space_mango.cache import (
    CacheInfo,
    FragmentCache,
    contiguous_runs,
    default_cache_dir,
    default_max_bytes,
    month_start,
    months_between,
    next_month,
)
from space_mango.errors import (
    CacheMissError,
    MangoError,
    MangoFilterError,
    QueryError,
    ServerError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
    did_you_mean,
    error_from_query,
    error_from_response,
)
from space_mango.filtering import build_filter_exprs, parse_time
from space_mango.models import filters_for
from space_mango.result import MangoResult
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


def _progress(items: list[list[date]], desc: str) -> Iterable[list[date]]:
    """Progress bar over fragment downloads when tqdm is installed; silent otherwise."""
    if len(items) < 2:
        return items
    try:
        from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        return items
    return tqdm(items, desc=desc, unit="request")


_DATETIME_DTYPE = re.compile(r"Datetime\(time_unit='(ns|us|ms)', time_zone=(None|'[^']*')\)")
_TIME_UNITS: dict[str, Literal["ns", "us", "ms"]] = {"ns": "ns", "us": "us", "ms": "ms"}


def _parse_dtype(text: str) -> pl.DataType:
    """Polars dtype from its str() as served by /describe (e.g. 'Float64', 'Datetime(...)')."""
    m = _DATETIME_DTYPE.fullmatch(text)
    if m:
        tz = None if m[2] == "None" else m[2].strip("'")
        return pl.Datetime(_TIME_UNITS[m[1]], tz)
    dtype = getattr(pl, text, None)
    if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
        return dtype()
    raise MangoError(f"Cannot interpret column dtype {text!r} served by /describe.")


def _range_filters(region: str, query: Mapping[str, object]) -> dict[str, float]:
    """The validated range filters ({"bz_imf_max": -2.0, ...}) recorded in a query."""
    names = filters_for(region)
    out: dict[str, float] = {}
    for key, value in query.items():
        name, _, suffix = key.rpartition("_")
        if suffix in ("min", "max") and name in names and isinstance(value, float):
            out[key] = value
    return out


class MangoClient:
    """Client for the MANGO dataset API."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        cache_dir: Path | str | None = None,
        cache: bool = True,
        offline: bool = False,
    ) -> None:
        """cache_dir defaults to $SPACE_MANGO_CACHE_DIR or the platform user cache directory.
        cache=False sends every get_data to the server. offline=True serves get_data only from
        the cache and raises CacheMissError for anything not cached."""
        self._cache: FragmentCache = FragmentCache(
            Path(cache_dir) if cache_dir else default_cache_dir(), default_max_bytes()
        )
        self._cache_enabled: bool = cache
        self._offline: bool = offline
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout, transport=transport)
        self._filter_cache: dict[str, set[str]] = {}
        self._regions: list[str] | None = None
        self._dataset_info: dict[str, Any] | None = None
        self._describe_cache: dict[str, dict[str, Any]] = {}
        self._spacecraft_cache: dict[str, pl.DataFrame] = {}

    def _get(self, path: str, params: Mapping[str, object] | None = None) -> httpx.Response:
        if self._offline:
            raise CacheMissError(
                f"Offline mode: {path} needs the MANGO server and is not answered from the "
                "cache. Drop offline=True to use it."
            )
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

    def _get_meta(self, path: str) -> Any:
        """JSON of a metadata endpoint. Online: fetched and stored under the dataset version
        in the cache. Offline: read back from the cache, never from the network."""
        name = path.removeprefix("/api/v1/").replace("/", "__") + ".json"
        if self._offline:
            version = (
                str(self._dataset_info["version"])
                if self._dataset_info is not None
                else self._cache.latest_version()
            )
            data = self._cache.read_meta(version, name) if version else None
            if data is None:
                raise CacheMissError(
                    f"Offline mode: {path} is not in the cache at {self._cache.root}. "
                    "Run the same query once online, or drop offline=True."
                )
            return data
        data = self._get(path).json()
        version = data["version"] if path == "/api/v1/dataset" else self.dataset_info()["version"]
        self._cache.write_meta(str(version), name, data)
        return data

    def _ensure_filters_cached(self, region: str) -> None:
        if region not in self._filter_cache:
            infos = self._get_meta(f"/api/v1/regions/{region}/filters")
            self._filter_cache[region] = {f["name"] for f in infos}

    def _other_region_filters(self, exclude: str) -> dict[str, set[str]]:
        for region in self.regions():
            self._ensure_filters_cached(region)
        return {r: names for r, names in self._filter_cache.items() if r != exclude}

    def regions(self) -> list[str]:
        """List available regions."""
        if self._regions is None:
            self._regions = self._get_meta("/api/v1/regions")
        return list(self._regions or [])

    def columns(self, region: str) -> list[str]:
        """List columns available in a region."""
        r = self._get(f"/api/v1/regions/{region}/columns")
        return r.json()

    def filters(self, region: str) -> list[dict[str, object]]:
        """List available filters for a region (name, column, unit, description)."""
        return self._get_meta(f"/api/v1/regions/{region}/filters")

    def dataset_info(self) -> dict[str, Any]:
        """Dataset version, title, citation (BibTeX), DOI and schema checksum."""
        if self._dataset_info is None:
            self._dataset_info = self._get_meta("/api/v1/dataset")
        return self._dataset_info or {}

    def _describe_raw(self, region: str) -> dict[str, Any]:
        if region not in self._describe_cache:
            self._describe_cache[region] = self._get_meta(f"/api/v1/regions/{region}/describe")
        return self._describe_cache[region]

    def _result(
        self, region: str | None, df: pl.DataFrame, query: dict[str, object]
    ) -> MangoResult:
        regions = [region] if region else self.regions()
        info: dict[str, dict[str, str]] = {}
        for r in regions:
            for c in self._describe_raw(r)["columns"]:
                info.setdefault(c["name"], {k: c[k] for k in ("unit", "frame", "description")})
        ds = self.dataset_info()
        return MangoResult(df, info, ds["version"], query, ds["citation"], region)

    def _request_params(
        self,
        region: str,
        *,
        columns: list[str] | None,
        spacecraft: list[str] | None,
        start: TimeLike,
        stop: TimeLike,
        sw_paired_only: bool,
        normalized_only: bool,
        time_min: TimeLike,
        time_max: TimeLike,
        filters: Mapping[str, object],
        limit: int | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Validate and build (HTTP params, query record), shared by get_data and count."""
        self._check_region(region)
        if time_min is not None:
            warnings.warn("time_min is deprecated, use start=", FutureWarning, stacklevel=3)
            start = start if start is not None else time_min
        if time_max is not None:
            warnings.warn(
                "time_max is deprecated, use stop= (exclusive)", FutureWarning, stacklevel=3
            )
        self._ensure_filters_cached(region)
        cleaned = _validate_filters(
            filters, region, self._filter_cache[region], self._other_region_filters(exclude=region)
        )
        start_iso = to_iso(start, param="start")
        stop_iso = to_iso(stop, param="stop")
        max_iso = to_iso(time_max, param="time_max")
        params: dict[str, object] = {"format": "arrow", **{k: str(v) for k, v in cleaned.items()}}
        if limit is not None:
            params["limit"] = str(limit)
        if columns:
            params["columns"] = columns
        if spacecraft:
            params["spacecraft"] = spacecraft
        if start_iso is not None:
            params["start"] = start_iso
        if stop_iso is not None:
            params["stop"] = stop_iso
        if max_iso is not None:
            params["time_max"] = max_iso
        if sw_paired_only:
            params["sw_paired_only"] = "true"
        if normalized_only:
            params["normalized_only"] = "true"
        query: dict[str, object] = {
            "region": region,
            "columns": columns,
            "spacecraft": spacecraft,
            "start": start_iso,
            "stop": stop_iso,
            "time_max": max_iso,
            "sw_paired_only": sw_paired_only,
            "normalized_only": normalized_only,
            "limit": limit,
            **cleaned,
        }
        return params, query

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
        cache: bool | None = None,
        **filters: float,
    ) -> MangoResult:
        """Query one region. Range filters are keyword arguments: bz_imf_max=-2, d_msh_max=0.3.

        start is inclusive, stop is exclusive. time_min/time_max are deprecated aliases
        (both inclusive).

        By default data are fetched as monthly per-column fragments, kept in a local cache
        keyed by dataset version, and filtered locally (same filtering code as the server).
        cache=False, or a limit, sends the query to the server instead.
        """
        params, query = self._request_params(
            region,
            columns=columns,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            sw_paired_only=sw_paired_only,
            normalized_only=normalized_only,
            time_min=time_min,
            time_max=time_max,
            filters=filters,
            limit=limit,
        )
        use_cache = self._offline or (
            (self._cache_enabled if cache is None else cache) and limit is None
        )
        if use_cache:
            df = self._get_data_cached(
                region,
                columns,
                spacecraft,
                query,
                _range_filters(region, query),
                sw_paired_only,
                normalized_only,
            )
            if limit is not None:
                df = df.head(limit)
        else:
            df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        return self._result(region, df, query)

    def _fetch_months(
        self, version: str, region: str, sc: str, months: list[date], columns: list[str]
    ) -> None:
        params: dict[str, object] = {
            "format": "arrow",
            "spacecraft": [sc],
            "columns": columns,
            "start": month_start(months[0]).isoformat(),
            "stop": month_start(next_month(months[-1])).isoformat(),
        }
        df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        self._cache.write_months(version, region, sc, months, df)

    def _get_data_cached(
        self,
        region: str,
        columns: list[str] | None,
        spacecraft: list[str] | None,
        query: Mapping[str, object],
        ranges: dict[str, float],
        sw_paired_only: bool,
        normalized_only: bool,
    ) -> pl.DataFrame:
        version = str(self.dataset_info()["version"])
        served = [c["name"] for c in self._describe_raw(region)["columns"]]
        for c in columns or []:
            if c not in served:
                raise UnknownColumnError(
                    f"'{c}' is not a column of region '{region}'.{did_you_mean(c, served)}"
                )
        coverage: dict[str, tuple[datetime, datetime]] = {
            row["sc"]: (row["start"], row["stop"]) for row in self.spacecraft(region).to_dicts()
        }
        for sc in spacecraft or []:
            if sc not in coverage:
                raise UnknownSpacecraftError(
                    f"'{sc}' is not a spacecraft in region '{region}'."
                    f"{did_you_mean(sc, coverage)}"
                )
        start_raw, stop_raw, max_raw = query.get("start"), query.get("stop"), query.get("time_max")
        start = parse_time(str(start_raw), param="start") if start_raw else None
        stop_inclusive = not stop_raw and bool(max_raw)
        stop_any = stop_raw or max_raw
        stop = parse_time(str(stop_any), param="stop") if stop_any else None
        needed = {"Time"} | {c for c in (columns or served) if c != "SC"}
        catalog = filters_for(region)
        needed |= {catalog[k.rpartition("_")[0]].column for k in ranges}
        if sw_paired_only:
            needed.add("SW_pairing")
        if normalized_only:
            needed.add("Norma_pos")
        # Columns the region does not serve are left out: build_filter_exprs then refuses the
        # flag/filter exactly as the server does, before anything is downloaded.
        cols = sorted(needed & set(served))
        try:
            exprs = build_filter_exprs(
                region,
                set(cols) | {"SC"},
                start=start,
                stop=stop,
                stop_inclusive=stop_inclusive,
                sw_paired_only=sw_paired_only,
                normalized_only=normalized_only,
                ranges=ranges,
            )
        except QueryError as e:
            raise error_from_query(e) from None
        keep = pl.all_horizontal(exprs) if exprs else None

        frames: list[pl.DataFrame] = []
        for sc in spacecraft or sorted(coverage):
            sc_start, sc_stop = coverage[sc]
            lo = max(start, sc_start) if start else sc_start
            hi = min(stop, sc_stop) if stop else sc_stop
            if lo > hi:
                continue
            months = months_between(lo, hi)
            missing = [m for m in months if not self._cache.has(version, region, sc, m, cols)]
            if missing and self._offline:
                raise CacheMissError(
                    f"Offline mode: {len(missing)} month(s) of {sc}/{region} are not cached "
                    f"(first: {missing[0]:%Y-%m}). Run once online, or drop offline=True."
                )
            for run in _progress(contiguous_runs(missing), f"Downloading {region}/{sc}"):
                self._fetch_months(version, region, sc, run, cols)
            for m in months:
                frame = self._read_month(version, region, sc, m, cols).with_columns(SC=pl.lit(sc))
                # Filter month by month (all filters are row-wise) to bound peak memory.
                frames.append(frame if keep is None else frame.filter(keep))
        self._cache.evict()
        if not frames:
            return self._empty_frame(region, columns or served)
        return pl.concat(frames, how="vertical_relaxed").select(columns or served)

    def _read_month(
        self, version: str, region: str, sc: str, month: date, columns: list[str]
    ) -> pl.DataFrame:
        try:
            return self._cache.read_month(version, region, sc, month, columns)
        except FileNotFoundError:
            # Another process evicted a fragment after has() said it was there: fetch it again.
            if self._offline:
                raise CacheMissError(
                    f"Offline mode: {month:%Y-%m} of {sc}/{region} was evicted from the cache."
                ) from None
            self._fetch_months(version, region, sc, [month], columns)
            return self._cache.read_month(version, region, sc, month, columns)

    def _empty_frame(self, region: str, columns: list[str]) -> pl.DataFrame:
        """A zero-row frame with the served schema, built from the region's described dtypes
        (no request, so it also works offline)."""
        dtypes = {c["name"]: c["dtype"] for c in self._describe_raw(region)["columns"]}
        return pl.DataFrame(schema={c: _parse_dtype(dtypes[c]) for c in columns})

    def cache_info(self) -> CacheInfo:
        """Cache location, number of fragment files, size and size cap (bytes)."""
        return self._cache.info()

    def cache_clear(self) -> None:
        """Delete every cached fragment (all dataset versions)."""
        self._cache.clear()

    def describe(self, region: str) -> pl.DataFrame:
        """Every column of a region: unit, coordinate frame, description, matching filter."""
        self._check_region(region)
        cols = self._describe_raw(region)["columns"]
        return pl.DataFrame(
            [
                {
                    "column": c["name"],
                    "unit": c["unit"],
                    "frame": c["frame"],
                    "description": c["description"],
                    "filter": c["filter"],
                    "dtype": c["dtype"],
                }
                for c in cols
            ],
            schema={
                "column": pl.String,
                "unit": pl.String,
                "frame": pl.String,
                "description": pl.String,
                "filter": pl.String,
                "dtype": pl.String,
            },
        )

    def region_definition(self, region: str) -> str:
        """Plain-language definition of a region."""
        self._check_region(region)
        return self._describe_raw(region)["definition"]

    def spacecraft(self, region: str) -> pl.DataFrame:
        """Spacecraft present in a region, with first/last sample time and row count."""
        self._check_region(region)
        if region not in self._spacecraft_cache:
            rows = self._get_meta(f"/api/v1/regions/{region}/spacecraft")
            self._spacecraft_cache[region] = pl.DataFrame(
                rows,
                schema={"sc": pl.String, "start": pl.String, "stop": pl.String, "n_rows": pl.Int64},
            ).with_columns(pl.col("start", "stop").str.to_datetime(time_unit="us"))
        return self._spacecraft_cache[region]

    def count(
        self,
        region: str,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        **filters: float,
    ) -> dict[str, float]:
        """Rows and estimated download size (MB) of the matching get_data call. Downloads nothing."""
        params, _ = self._request_params(
            region,
            columns=columns,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            sw_paired_only=sw_paired_only,
            normalized_only=normalized_only,
            time_min=None,
            time_max=None,
            filters=filters,
        )
        params.pop("format", None)
        c = self._get(f"/api/v1/regions/{region}/count", params).json()
        return {"n_rows": c["n_rows"], "est_mb": c["est_bytes"] / 1e6}

    def search(self, text: str) -> pl.DataFrame:
        """Columns and filters whose name, unit or description contains `text` (case-insensitive)."""
        needle = text.lower()
        rows: list[dict[str, str]] = []
        for region in self.regions():
            d = self._describe_raw(region)
            for c in d["columns"]:
                rows.append(
                    {
                        "region": region,
                        "kind": "column",
                        "name": c["name"],
                        "unit": c["unit"],
                        "description": c["description"],
                    }
                )
            for f in d["filters"]:
                rows.append(
                    {
                        "region": region,
                        "kind": "filter",
                        "name": f["name"],
                        "unit": f["unit"],
                        "description": f["description"],
                    }
                )
        schema = {k: pl.String for k in ("region", "kind", "name", "unit", "description")}
        df = pl.DataFrame(rows, schema=schema)
        hay = pl.concat_str(["name", "unit", "description"], separator=" ").str.to_lowercase()
        return df.filter(hay.str.contains(needle, literal=True))

    def timeline(
        self, sc: str, start: TimeLike, stop: TimeLike, columns: list[str] | None = None
    ) -> MangoResult:
        """Every sample of one spacecraft over [start, stop) across all regions (up to 31 days),
        with a 'region' column. Use .to_intervals() for region crossings."""
        params: dict[str, object] = {
            "sc": sc,
            "start": to_iso(start, param="start"),
            "stop": to_iso(stop, param="stop"),
            "format": "arrow",
        }
        if columns:
            params["columns"] = columns
        r = self._get("/api/v1/timeline", params)
        query: dict[str, object] = {
            "timeline": sc,
            "start": params["start"],
            "stop": params["stop"],
            "columns": columns,
        }
        return self._result(None, pl.read_ipc(r.content), query)

    def cite(self) -> str:
        """BibTeX for the dataset version served."""
        return self.dataset_info()["citation"]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MangoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
