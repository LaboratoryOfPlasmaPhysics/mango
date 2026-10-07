from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl

from space_mango.errors import QueryError, did_you_mean
from space_mango.filtering import build_filter_exprs, parse_range_params
from space_mango.models import Region

_DEFAULT_DATA_DIR = Path("/data/mango")


class MangoDataset:
    """Lazy Polars interface over the MANGO Parquet files."""

    def __init__(self, data_dir: Path):
        self._dir = data_dir
        self._frames: dict[str, pl.LazyFrame] = {}

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
    ) -> pl.LazyFrame:
        lf = self._lazy(region)
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
            ranges=parse_range_params(region, raw_params),
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


_dataset: MangoDataset | None = None


def get_dataset() -> MangoDataset:
    global _dataset
    if _dataset is None:
        import os

        data_dir = Path(os.environ.get("MANGO_DATA_DIR", str(_DEFAULT_DATA_DIR)))
        _dataset = MangoDataset(data_dir)
    return _dataset
