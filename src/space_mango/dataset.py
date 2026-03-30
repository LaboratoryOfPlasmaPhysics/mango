from pathlib import Path

import polars as pl

from space_mango.models import RANGE_FILTERS, Region

_DEFAULT_DATA_DIR = Path("/data/mango")


def _apply_range_filters(
    lf: pl.LazyFrame, region: Region, raw_params: dict[str, str | None]
) -> pl.LazyFrame:
    """Apply all range filters from the catalog that have min/max values set."""
    available = set(lf.collect_schema().names())
    filters: list[pl.Expr] = []

    for name, filt in RANGE_FILTERS.items():
        if region not in filt.regions:
            continue
        if filt.column not in available:
            continue
        lo = raw_params.get(f"{name}_min")
        hi = raw_params.get(f"{name}_max")
        if lo is not None:
            filters.append(pl.col(filt.column) >= float(lo))
        if hi is not None:
            filters.append(pl.col(filt.column) <= float(hi))

    if filters:
        lf = lf.filter(pl.all_horizontal(filters))
    return lf


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

    def query(
        self,
        region: Region,
        raw_params: dict[str, str | None],
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int | None = None,
    ) -> pl.DataFrame:
        lf = self._lazy(region)
        available = set(lf.collect_schema().names())
        filters: list[pl.Expr] = []

        if spacecraft:
            filters.append(pl.col("SC").is_in(spacecraft))
        if time_min:
            filters.append(pl.col("Time") >= time_min)
        if time_max:
            filters.append(pl.col("Time") <= time_max)
        if sw_paired_only and "SW_pairing" in available:
            filters.append(pl.col("SW_pairing"))
        if normalized_only and "Norma_pos" in available:
            filters.append(pl.col("Norma_pos"))

        if filters:
            lf = lf.filter(pl.all_horizontal(filters))

        lf = _apply_range_filters(lf, region, raw_params)

        if columns:
            cols = [c for c in columns if c in available]
            if cols:
                lf = lf.select(cols)

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
