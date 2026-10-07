import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from space_mango.errors import QueryError, did_you_mean
from space_mango.filtering import COUNT_PARAMS, DATA_PARAMS, build_filter_exprs, parse_range_params
from space_mango.models import Region, columns_for, filters_for

_DEFAULT_DATA_DIR = Path("/data/mango")
MAX_TIMELINE_SPAN = timedelta(days=31)
_DTYPE_BYTES: dict[type[pl.DataType], int] = {pl.Boolean: 1, pl.String: 4}


def _dtype_bytes(dt: pl.DataType) -> int:
    return _DTYPE_BYTES.get(type(dt), 8)  # floats, ints, datetimes: 8 bytes


class MangoDataset:
    """Lazy Polars interface over the MANGO Parquet files."""

    def __init__(self, data_dir: Path):
        self._dir = data_dir
        self._frames: dict[str, pl.LazyFrame] = {}
        self._coverage: dict[str, pl.DataFrame] = {}

    def _lazy(self, region: str) -> pl.LazyFrame:
        if region not in self._frames:
            path = self._dir / region
            self._frames[region] = pl.scan_parquet(
                path, hive_partitioning=True
            )
        return self._frames[region]

    def __getitem__(self, region: str) -> pl.LazyFrame:
        return self._lazy(region)

    def spacecraft(self, region: Region | str) -> list[str]:
        path = self._dir / Region(region).value
        if not path.is_dir():
            return []
        return sorted(
            p.name.split("=", 1)[1]
            for p in path.iterdir()
            if p.is_dir() and p.name.startswith("SC=")
        )

    def columns(self, region: Region | str) -> list[str]:
        return self._lazy(Region(region).value).collect_schema().names()

    def _plan(
        self,
        region: Region,
        raw_params: Mapping[str, str | None],
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: datetime | None = None,
        stop: datetime | None = None,
        stop_inclusive: bool = False,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        params: frozenset[str] = DATA_PARAMS,
    ) -> pl.LazyFrame:
        lf = self._lazy(region)
        if columns:
            columns = list(dict.fromkeys(columns))
        available = set(lf.collect_schema().names())
        if spacecraft:
            known = self.spacecraft(region)
            for sc in spacecraft:
                if sc not in known:
                    raise QueryError(
                        "unknown_spacecraft",
                        f"'{sc}' is not a spacecraft in region '{region.value}'."
                        f"{did_you_mean(sc, known)}",
                        known,
                    )
        if columns:
            for c in columns:
                if c not in available:
                    raise QueryError(
                        "unknown_column",
                        f"'{c}' is not a column of region '{region.value}'."
                        f"{did_you_mean(c, available)}",
                        available,
                    )
        exprs = build_filter_exprs(
            region,
            available,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            stop_inclusive=stop_inclusive,
            sw_paired_only=sw_paired_only,
            normalized_only=normalized_only,
            ranges=parse_range_params(region, raw_params, params),
        )
        if exprs:
            lf = lf.filter(pl.all_horizontal(exprs))
        if columns:
            lf = lf.select(columns)
        return lf

    def query(
        self,
        region: Region,
        raw_params: Mapping[str, str | None],
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: datetime | None = None,
        stop: datetime | None = None,
        stop_inclusive: bool = False,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int | None = None,
    ) -> pl.DataFrame:
        lf = self._plan(
            region,
            raw_params,
            columns=columns,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            stop_inclusive=stop_inclusive,
            sw_paired_only=sw_paired_only,
            normalized_only=normalized_only,
        )
        if limit is not None:
            lf = lf.limit(limit)
        return lf.collect()

    def exists(self) -> bool:
        return self._dir.is_dir()

    def coverage(self, region: Region | str) -> pl.DataFrame:
        key = Region(region).value
        if key not in self._coverage:
            self._coverage[key] = (
                self._lazy(key)
                .group_by("SC")
                .agg(start=pl.col("Time").min(), stop=pl.col("Time").max(), n_rows=pl.len())
                .sort("SC")
                .collect()
            )
        return self._coverage[key]

    def count(
        self,
        region: Region,
        raw_params: Mapping[str, str | None],
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: datetime | None = None,
        stop: datetime | None = None,
        stop_inclusive: bool = False,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
    ) -> tuple[int, int]:
        lf = self._plan(
            region,
            raw_params,
            columns=columns,
            spacecraft=spacecraft,
            start=start,
            stop=stop,
            stop_inclusive=stop_inclusive,
            sw_paired_only=sw_paired_only,
            normalized_only=normalized_only,
            params=COUNT_PARAMS,
        )
        n_rows = int(lf.select(pl.len()).collect().item())
        row_bytes = sum(_dtype_bytes(dt) for dt in lf.collect_schema().dtypes())
        return n_rows, n_rows * row_bytes

    def timeline(
        self, sc: str, start: datetime, stop: datetime, columns: list[str] | None
    ) -> pl.DataFrame:
        if stop <= start:
            raise QueryError("bad_time", "stop must be after start.")
        if stop - start > MAX_TIMELINE_SPAN:
            raise QueryError(
                "span_too_long",
                f"A timeline covers at most {MAX_TIMELINE_SPAN.days} days; got {stop - start}. "
                "Use get_data() per region for longer periods.",
            )
        regions = [r for r in Region if sc in self.spacecraft(r)]
        if not regions:
            every = sorted({s for r in Region for s in self.spacecraft(r)})
            raise QueryError(
                "unknown_spacecraft",
                f"'{sc}' is not a MANGO spacecraft.{did_you_mean(sc, every)}",
                every,
            )
        available = {c for r in regions for c in self.columns(r)}
        if columns:
            columns = list(dict.fromkeys(columns))
        for c in columns or []:
            if c not in available:
                raise QueryError(
                    "unknown_column",
                    f"'{c}' is not a MANGO column.{did_you_mean(c, available)}",
                    available,
                )
        frames: list[pl.DataFrame] = []
        for r in regions:
            lf = self._lazy(r).filter(
                (pl.col("SC") == sc) & (pl.col("Time") >= start) & (pl.col("Time") < stop)
            )
            if columns:
                have = set(self.columns(r))
                lf = lf.select(
                    ["Time", "SC", *[c for c in columns if c in have and c not in ("Time", "SC")]]
                )
            frames.append(lf.with_columns(region=pl.lit(r.value)).collect())
        return pl.concat(frames, how="diagonal_relaxed").sort("Time")

    def schema_checksum(self) -> str:
        parts: list[str] = []
        for r in Region:
            if (self._dir / r.value).is_dir():
                schema = self._lazy(r.value).collect_schema()
                parts += [f"{r.value}:{name}:{dtype}" for name, dtype in schema.items()]
        return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()

    def check_catalog(self) -> list[str]:
        """Differences between the catalog (models.py) and the served schema."""
        problems: list[str] = []
        for r in Region:
            if not (self._dir / r.value).is_dir():
                continue
            served = set(self.columns(r))
            documented = set(columns_for(r))
            problems += [
                f"{r.value}: catalog column '{c}' is not served"
                for c in sorted(documented - served)
            ]
            problems += [
                f"{r.value}: served column '{c}' is not in the catalog"
                for c in sorted(served - documented)
            ]
            problems += [
                f"{r.value}: filter '{n}' needs missing column '{f.column}'"
                for n, f in filters_for(r).items()
                if f.column not in served
            ]
        return problems


_dataset: MangoDataset | None = None


def get_dataset() -> MangoDataset:
    global _dataset
    if _dataset is None:
        import os

        data_dir = Path(os.environ.get("MANGO_DATA_DIR", str(_DEFAULT_DATA_DIR)))
        _dataset = MangoDataset(data_dir)
    return _dataset
