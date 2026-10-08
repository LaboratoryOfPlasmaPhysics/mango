from __future__ import annotations

import math
import numbers
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
import polars as pl

from space_mango.cache import (
    CacheInfo,
    FragmentCache,
    contiguous_runs,
    default_cache_dir,
    default_max_bytes,
    month_slice,
    month_start,
    months_between,
    next_month,
)
from space_mango.errors import (
    CacheMissError,
    MangoError,
    MangoFilterError,
    PgsmError,
    QueryError,
    ServerError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
    did_you_mean,
    error_from_query,
    error_from_response,
)
from space_mango.filtering import build_filter_exprs, time_window
from space_mango.models import filters_for
from space_mango.pgsm import (
    COLUMN_INFO as PGSM_COLUMN_INFO,
)
from space_mango.pgsm import (
    OUTPUT_COLUMNS as PGSM_OUTPUT_COLUMNS,
)
from space_mango.pgsm import (
    REQUIRED_COLUMNS as PGSM_REQUIRED_COLUMNS,
)
from space_mango.pgsm import (
    THESIS,
    PgsmSpec,
    make_spec,
    to_pgsm,
)
from space_mango.result import MangoResult
from space_mango.timeparse import TimeLike, to_iso

DEFAULT_URL = "http://sciqlop.lpp.polytechnique.fr/mango/"
MAX_MONTHS_PER_REQUEST = 6
"""Cache path: a download request covers at most this many months of one spacecraft."""

_T = TypeVar("_T")

Names = str | Iterable[str] | None


def _as_names(value: Names) -> list[str] | None:
    """One name or several, as a list without duplicates (order kept). "THA" -> ["THA"]."""
    if value is None:
        return None
    names = [value] if isinstance(value, str) else list(value)
    return list(dict.fromkeys(names)) or None


def _is_number(value: object) -> bool:
    """int, float and numpy integer/float scalars (registered as numbers.Real by numpy,
    so numpy is never imported here); bool and numpy.bool_ are refused."""
    if isinstance(value, bool) or type(value).__name__ in ("bool_", "bool"):
        return False
    return isinstance(value, numbers.Real)


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
        if not (_is_number(value) or isinstance(value, str)):
            raise MangoFilterError(f"Filter '{key}' value must be numeric, got {value!r}.")
        try:
            number = float(value)  # pyright: ignore[reportArgumentType]
        except ValueError:
            raise MangoFilterError(
                f"Filter '{key}' value must be numeric, got {value!r}."
            ) from None
        if not math.isfinite(number):
            raise MangoFilterError(f"Filter '{key}' value must be finite, got {value!r}.")
        cleaned[key] = number
    return cleaned


def _progress(items: list[_T], desc: str) -> Iterable[_T]:
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


class _CacheUnusable(Exception):
    """A cache write failed (already warned): get_data falls back to the server path."""


def _cache_key_of(info: Mapping[str, Any]) -> str:
    """Cache directory name: <version>-<12 hex chars of the schema checksum>."""
    checksum = str(info.get("schema_checksum") or "")[:12]
    version = str(info["version"])
    return f"{version}-{checksum}" if checksum else version


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _range_filters(region: str, query: Mapping[str, object]) -> dict[str, float]:
    """The validated range filters ({"bz_imf_max": -2.0, ...}) recorded in a query."""
    names = filters_for(region)
    out: dict[str, float] = {}
    for key, value in query.items():
        name, _, suffix = key.rpartition("_")
        if suffix in ("min", "max") and name in names and isinstance(value, float):
            out[key] = value
    return out


def _pgsm_spec(
    region: str,
    frame: str | None,
    cone: object,
    clock: object,
    tilt: object,
    *,
    require_clock: bool = True,
) -> PgsmSpec | None:
    try:
        return make_spec(region, frame, cone, clock, tilt, require_clock=require_clock)  # pyright: ignore[reportArgumentType]
    except QueryError as e:
        raise error_from_query(e) from None


def _pgsm_fetch_args(
    spec: PgsmSpec,
    columns: list[str] | None,
    filters: Mapping[str, object],
) -> tuple[list[str] | None, dict[str, object]]:
    """Columns and range filters to fetch for a PGSM query: the requested columns plus the
    transform inputs; for the magnetosphere, |tilt| <= max(|t1|, |t2|) (radians) narrows the
    download (the exact selection is done by the transform)."""
    if spec.region == "magnetosphere" and {"tilt_min", "tilt_max"} & set(filters):
        raise PgsmError(
            "tilt_min/tilt_max (radians, plain selection) cannot be combined with "
            "frame='pgsm'; use tilt=[min, max] in degrees."
        )
    fetch = (
        None
        if columns is None
        else list(
            dict.fromkeys(
                [
                    *(c for c in columns if c not in PGSM_OUTPUT_COLUMNS[spec.region]),
                    *PGSM_REQUIRED_COLUMNS[spec.region],
                ]
            )
        )
    )
    extra: dict[str, object] = dict(filters)
    if spec.tilt is not None:
        t = math.radians(max(abs(spec.tilt[0]), abs(spec.tilt[1])))
        extra |= {"tilt_min": -t, "tilt_max": t}
    return fetch, extra


def _pgsm_count_params(spec: PgsmSpec) -> dict[str, object]:
    params: dict[str, object] = {"frame": "pgsm"}
    if spec.cone is not None:
        params |= {"pgsm_cone_min": str(spec.cone[0]), "pgsm_cone_max": str(spec.cone[1])}
    if spec.tilt is not None:
        params |= {"pgsm_tilt_min": str(spec.tilt[0]), "pgsm_tilt_max": str(spec.tilt[1])}
    return params


class MangoClient:
    """Client for the MANGO dataset API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        cache_dir: Path | str | None = None,
        cache: bool = True,
        offline: bool = False,
    ) -> None:
        """base_url defaults to $SPACE_MANGO_URL, else the public MANGO server.
        cache_dir defaults to $SPACE_MANGO_CACHE_DIR or the platform user cache directory.
        cache=False sends every get_data to the server. offline=True serves get_data only from
        the cache and raises CacheMissError for anything not cached."""
        if base_url is None:
            base_url = os.environ.get("SPACE_MANGO_URL") or DEFAULT_URL
        self._cache: FragmentCache = FragmentCache(
            Path(cache_dir) if cache_dir else default_cache_dir(), default_max_bytes()
        )
        self._cache_enabled: bool = cache
        self._cache_usable: bool = True  # False after a failed write (warned once)
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
            try:
                body = r.json()
            except ValueError:  # not JSON, e.g. an HTML page from a proxy
                raise ServerError(
                    f"MANGO server answered HTTP 400 for {path}: {r.text[:300]}"
                ) from None
            raise error_from_response(400, body)
        if r.status_code == 404 and path == "/api/v1/dataset":
            raise ServerError(
                f"MANGO server at {self._base_url} is older than 0.2; upgrade the server "
                "or use space-mango<0.2"
            )
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

    def _cache_failed(self, exc: OSError) -> None:
        """Warn (once per client) that the cache cannot be written, and stop using it."""
        if self._cache_usable:
            warnings.warn(
                f"space-mango cannot write its cache at {self._cache.root} ({exc}); "
                "continuing without caching. Set SPACE_MANGO_CACHE_DIR to a writable "
                "directory to cache downloads.",
                UserWarning,
                stacklevel=4,
            )
        self._cache_usable = False

    def _cache_key(self) -> str:
        return _cache_key_of(self.dataset_info())

    def _get_meta(self, path: str) -> Any:
        """JSON of a metadata endpoint. Online: fetched, and stored under the cache key when
        caching is enabled. Offline: read back from the cache, never from the network."""
        name = path.removeprefix("/api/v1/").replace("/", "__") + ".json"
        if self._offline:
            key = (
                self._cache_key() if self._dataset_info is not None else self._cache.latest_key()
            )
            data = self._cache.read_meta(key, name) if key else None
            if data is None:
                raise CacheMissError(
                    f"Offline mode: {path} is not in the cache at {self._cache.root}. "
                    "Run the same query once online, or drop offline=True."
                )
            return data
        data = self._get(path).json()
        if self._cache_enabled and self._cache_usable:
            key = _cache_key_of(data) if path == "/api/v1/dataset" else self._cache_key()
            try:
                self._cache.write_meta(key, name, data)
            except OSError as e:
                self._cache_failed(e)
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
        try:
            time_window(start_iso, stop_iso, None, max_iso)  # refuse an empty window early
        except QueryError as e:
            raise error_from_query(e) from None
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
        columns: Names = None,
        spacecraft: Names = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int | None = None,
        time_min: TimeLike = None,
        time_max: TimeLike = None,
        cache: bool | None = None,
        frame: str | None = None,
        cone: Sequence[float] | None = None,
        clock: float | None = None,
        tilt: Sequence[float] | None = None,
        **filters: float,
    ) -> MangoResult:
        """Query one region. Range filters are keyword arguments: bz_imf_max=-2, d_msh_max=0.3.

        start is inclusive, stop is exclusive. time_min/time_max are deprecated aliases
        (both inclusive).

        By default data are fetched as monthly per-column fragments, kept in a local cache
        keyed by dataset version, and filtered locally (same filtering code as the server).
        cache=False, or a limit, sends the query to the server instead.
        columns and spacecraft take one name or a list of names.

        frame="pgsm" returns PGSM data (Michotte de Welle 2024): magnetosheath with
        cone=[min, max] (degrees, IMF angle to X_GSM, 0 = sunward) and clock=<target degrees>;
        magnetosphere with tilt=[min, max] (degrees). Implies normalized_only (and
        sw_paired_only in the magnetosheath). limit applies to the rows fetched, before the
        transform. See the PGSM section of the user guide.
        """
        if frame is not None or cone is not None or clock is not None or tilt is not None:
            self._check_region(region)
            spec = _pgsm_spec(region, frame, cone, clock, tilt)
            if spec is not None:
                return self._get_data_pgsm(
                    spec,
                    _as_names(columns),
                    spacecraft=spacecraft,
                    start=start,
                    stop=stop,
                    sw_paired_only=sw_paired_only,
                    limit=limit,
                    time_min=time_min,
                    time_max=time_max,
                    cache=cache,
                    filters=filters,
                )
        columns, spacecraft = _as_names(columns), _as_names(spacecraft)
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
            (self._cache_enabled if cache is None else cache)
            and limit is None
            and self._cache_usable
        )
        df: pl.DataFrame | None = None
        if use_cache:
            try:
                df = self._get_data_cached(
                    region,
                    columns,
                    spacecraft,
                    query,
                    _range_filters(region, query),
                    sw_paired_only,
                    normalized_only,
                )
            except _CacheUnusable:
                df = None  # warned already; ask the server instead
        if df is None:
            df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        return self._result(region, df, query)

    def _write_month(self, key: str, region: str, sc: str, month: date, part: pl.DataFrame) -> None:
        try:
            self._cache.write_month(key, region, sc, month, part)
        except OSError as e:
            self._cache_failed(e)
            raise _CacheUnusable from e

    def _same_time(self, key: str, region: str, sc: str, month: date, part: pl.DataFrame) -> bool:
        """Whether the cached Time fragment of this month equals the freshly fetched Time."""
        try:
            cached = self._cache.read_month(key, region, sc, month, ["Time"])["Time"]
        except FileNotFoundError:
            return False
        return cached.len() == part.height and cached.equals(part["Time"])

    def _fetch_months(
        self,
        key: str,
        region: str,
        sc: str,
        months: list[date],
        missing: list[str],
        needed: list[str],
    ) -> None:
        """Download Time plus the missing columns of these months and store them. New
        fragments are only added next to a cached Time fragment equal to the fetched Time;
        a month where they differ is fetched again with every needed column."""
        params: dict[str, object] = {
            "format": "arrow",
            "spacecraft": [sc],
            "columns": ["Time", *[c for c in missing if c != "Time"]],
            "start": month_start(months[0]).isoformat(),
            "stop": month_start(next_month(months[-1])).isoformat(),
        }
        df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        df = df.sort("Time", maintain_order=True)
        stale: list[date] = []
        for m in months:
            part = month_slice(df, m)
            if "Time" in missing:
                self._write_month(key, region, sc, m, part)
            elif self._same_time(key, region, sc, m, part):
                self._write_month(key, region, sc, m, part.drop("Time"))
            else:
                stale.append(m)
        for m in stale:
            self._fetch_months(key, region, sc, [m], needed, needed)

    def _fetch_plan(
        self,
        key: str,
        region: str,
        sc: str,
        months: list[date],
        columns: list[str],
        max_months: int | None = MAX_MONTHS_PER_REQUEST,
    ) -> list[tuple[list[date], list[str]]]:
        """Requests filling the missing fragments of these months: (months, missing columns).
        Months missing the same columns are grouped in contiguous runs of at most max_months.
        A month without its Time fragment is fetched with every column."""
        by_missing: dict[tuple[str, ...], list[date]] = {}
        for m in months:
            missing = self._cache.missing(key, region, sc, m, columns)
            if missing:
                by_missing.setdefault(tuple(columns if "Time" in missing else missing), []).append(m)
        plan: list[tuple[list[date], list[str]]] = []
        for missing, ms in by_missing.items():
            for run in contiguous_runs(ms):
                step = max_months or len(run)
                plan += [(run[i : i + step], list(missing)) for i in range(0, len(run), step)]
        return sorted(plan, key=lambda p: p[0][0])

    def _cache_layout(
        self,
        region: str,
        columns: list[str] | None,
        spacecraft: list[str] | None,
        query: Mapping[str, object],
        ranges: dict[str, float],
        sw_paired_only: bool,
        normalized_only: bool,
    ) -> tuple[list[str], list[str], dict[str, list[date]], pl.Expr | None]:
        """(served columns, columns to cache, months per spacecraft, row filter) of a query."""
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
        start, stop, stop_inclusive = time_window(
            _opt_str(query.get("start")), _opt_str(query.get("stop")), None,
            _opt_str(query.get("time_max")),
        )
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
        # Last instant asked for: an exclusive stop on a month boundary must not pull that month.
        last = stop if stop is None or stop_inclusive else stop - timedelta(microseconds=1)
        months: dict[str, list[date]] = {}
        for sc in spacecraft or sorted(coverage):
            sc_start, sc_stop = coverage[sc]
            lo = max(start, sc_start) if start else sc_start
            hi = min(last, sc_stop) if last else sc_stop
            if lo <= hi:
                months[sc] = months_between(lo, hi)
        return served, cols, months, keep

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
        key = self._cache_key()
        served, cols, months_by_sc, keep = self._cache_layout(
            region, columns, spacecraft, query, ranges, sw_paired_only, normalized_only
        )
        frames: list[pl.DataFrame] = []
        for sc, months in months_by_sc.items():
            plan = self._fetch_plan(key, region, sc, months, cols)
            if plan and self._offline:
                n = sum(len(chunk) for chunk, _ in plan)
                raise CacheMissError(
                    f"Offline mode: {n} month(s) of {sc}/{region} are not fully cached "
                    f"(first: {plan[0][0][0]:%Y-%m}). Run once online, or drop offline=True."
                )
            for chunk, missing in _progress(plan, f"Downloading {region}/{sc}"):
                self._fetch_months(key, region, sc, chunk, missing, cols)
            for m in months:
                frame = self._read_month(key, region, sc, m, cols).with_columns(SC=pl.lit(sc))
                # Filter month by month (all filters are row-wise) to bound peak memory.
                frames.append(frame if keep is None else frame.filter(keep))
        self._evict()
        if not frames:
            return self._empty_frame(region, columns or served)
        return pl.concat(frames, how="vertical_relaxed").select(columns or served)

    def _evict(self) -> None:
        try:
            self._cache.evict()
        except FileNotFoundError:
            pass  # another process removed a fragment while we were evicting
        except OSError as e:
            self._cache_failed(e)

    def _read_month(
        self, key: str, region: str, sc: str, month: date, columns: list[str]
    ) -> pl.DataFrame:
        try:
            return self._cache.read_month(key, region, sc, month, columns)
        except FileNotFoundError:
            # Another process evicted a fragment after has() said it was there: fetch it again.
            if self._offline:
                raise CacheMissError(
                    f"Offline mode: {month:%Y-%m} of {sc}/{region} was evicted from the cache."
                ) from None
            self._fetch_months(key, region, sc, [month], columns, columns)
            return self._cache.read_month(key, region, sc, month, columns)

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

    def _get_data_pgsm(
        self,
        spec: PgsmSpec,
        columns: list[str] | None,
        *,
        spacecraft: Names,
        start: TimeLike,
        stop: TimeLike,
        sw_paired_only: bool,
        limit: int | None,
        time_min: TimeLike,
        time_max: TimeLike,
        cache: bool | None,
        filters: Mapping[str, object],
    ) -> MangoResult:
        fetch, extra = _pgsm_fetch_args(spec, columns, filters)
        res = self.get_data(
            spec.region,
            columns=fetch,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            sw_paired_only=sw_paired_only or spec.region == "magnetosheath",
            normalized_only=True,
            limit=limit,
            time_min=time_min,
            time_max=time_max,
            cache=cache,
            **extra,  # pyright: ignore[reportArgumentType]
        )
        try:
            df = to_pgsm(res.data, spec)
        except QueryError as e:
            raise error_from_query(e) from None
        if columns is not None:
            outs = PGSM_OUTPUT_COLUMNS[spec.region]
            df = df.select(
                [*dict.fromkeys(c for c in columns if c not in outs), *outs]
            )
        prefilter = {k: v for k, v in extra.items() if k not in filters}
        query = {
            **{k: v for k, v in res.query.items() if k not in prefilter},
            "columns": columns,
            "prefilter": prefilter,
            "frame": "pgsm",
            "cone": spec.cone,
            "clock": spec.clock,
            "tilt": spec.tilt,
            "reference": THESIS,
        }
        info = {**res.columns_info, **PGSM_COLUMN_INFO}
        return MangoResult(df, info, res.version, query, res.citation, spec.region)

    def count(
        self,
        region: str,
        *,
        columns: Names = None,
        spacecraft: Names = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        frame: str | None = None,
        cone: Sequence[float] | None = None,
        clock: float | None = None,
        tilt: Sequence[float] | None = None,
        **filters: float,
    ) -> dict[str, float]:
        """What the matching get_data call would return and download. Downloads nothing.

        n_rows: rows after filtering. est_mb: their size in MB (what a cache=False call
        transfers). download_mb_estimate: MB the default cached get_data would download, i.e.
        whole months of every needed column (requested + filter + flag columns) over the
        requested time span, unfiltered, minus the fragments already in the cache. It equals
        est_mb when caching is disabled for this client.

        With frame="pgsm", n_rows is the exact number of PGSM rows (computed by the server).
        """
        spec = None
        if frame is not None or cone is not None or clock is not None or tilt is not None:
            self._check_region(region)
            spec = _pgsm_spec(region, frame, cone, clock, tilt, require_clock=False)
        if spec is not None:
            if "pgsm_count" not in self.dataset_info().get("features", []):
                raise ServerError(
                    f"The MANGO server at {self._base_url} cannot count PGSM rows (it needs "
                    "space-mango >= 0.3 on the server); get_data(frame='pgsm') still works."
                )
            fetch, extra = _pgsm_fetch_args(spec, _as_names(columns), filters)
            columns, filters = fetch, extra  # pyright: ignore[reportAssignmentType]
            normalized_only = True
            sw_paired_only = sw_paired_only or spec.region == "magnetosheath"
        columns, spacecraft = _as_names(columns), _as_names(spacecraft)
        params, query = self._request_params(
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
        if spec is not None:
            params |= _pgsm_count_params(spec)
        c = self._get(f"/api/v1/regions/{region}/count", params).json()
        out = {"n_rows": c["n_rows"], "est_mb": c["est_bytes"] / 1e6}
        if self._cache_enabled and self._cache_usable:
            download = self._download_bytes(
                region, columns, spacecraft, query, sw_paired_only, normalized_only
            )
            out["download_mb_estimate"] = download / 1e6
        else:
            out["download_mb_estimate"] = out["est_mb"]
        return out

    def _count_bytes(
        self, region: str, spacecraft: list[str], columns: list[str], months: list[date]
    ) -> int:
        """Server estimate of the bytes of these columns over whole months, unfiltered."""
        params: dict[str, object] = {
            "spacecraft": spacecraft,
            "columns": columns,
            "start": month_start(months[0]).isoformat(),
            "stop": month_start(next_month(months[-1])).isoformat(),
        }
        return int(self._get(f"/api/v1/regions/{region}/count", params).json()["est_bytes"])

    def _download_bytes(
        self,
        region: str,
        columns: list[str] | None,
        spacecraft: list[str] | None,
        query: Mapping[str, object],
        sw_paired_only: bool,
        normalized_only: bool,
    ) -> int:
        """Bytes the cache path would download for this query (see count)."""
        key = self._cache_key()
        _, cols, months_by_sc, _ = self._cache_layout(
            region, columns, spacecraft, query, _range_filters(region, query),
            sw_paired_only, normalized_only,
        )
        plans = {
            sc: self._fetch_plan(key, region, sc, months, cols, max_months=None)
            for sc, months in months_by_sc.items()
        }
        if not any(plans.values()):
            return 0
        cold = all(
            len(plan) == 1 and plan[0] == (months_by_sc[sc], cols) for sc, plan in plans.items()
        )
        if cold:  # nothing cached: one request over the month-expanded span of every SC
            every = sorted({m for months in months_by_sc.values() for m in months})
            return self._count_bytes(region, list(months_by_sc), cols, [every[0], every[-1]])
        return sum(
            self._count_bytes(region, [sc], ["Time", *[c for c in missing if c != "Time"]], chunk)
            for sc, plan in plans.items()
            for chunk, missing in plan
        )

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
        self, sc: str, start: TimeLike, stop: TimeLike, columns: Names = None
    ) -> MangoResult:
        """Every sample of one spacecraft over [start, stop) across all regions (up to 31 days),
        with a 'region' column. Use .to_intervals() for region crossings."""
        columns = _as_names(columns)
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
