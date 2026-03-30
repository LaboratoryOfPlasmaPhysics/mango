from pathlib import Path

import polars as pl

from mango.models import SubsetParams

# Default path; override with MANGO_DATA_DIR env var.
_DEFAULT_DATA_DIR = Path("/data/mango")


def _range_filter(col: str, lo: float | None, hi: float | None) -> list[pl.Expr]:
    filters: list[pl.Expr] = []
    if lo is not None:
        filters.append(pl.col(col) >= lo)
    if hi is not None:
        filters.append(pl.col(col) <= hi)
    return filters


class MangoDataset:
    """Lazy Polars interface over the MANGO Parquet files."""

    def __init__(self, data_dir: Path):
        self._dir = data_dir
        self._frames: dict[str, pl.LazyFrame] = {}

    def _lazy(self, region: str) -> pl.LazyFrame:
        if region not in self._frames:
            path = self._dir / f"{region}.parquet"
            self._frames[region] = pl.scan_parquet(path)
        return self._frames[region]

    def __getitem__(self, region: str) -> pl.LazyFrame:
        return self._lazy(region)

    def query(self, region: str, params: SubsetParams) -> pl.DataFrame:
        lf = self._lazy(region)

        filters: list[pl.Expr] = []

        if params.spacecraft:
            filters.append(pl.col("SC").is_in(params.spacecraft))

        if params.time_min:
            filters.append(pl.col("Time") >= params.time_min)
        if params.time_max:
            filters.append(pl.col("Time") <= params.time_max)

        filters.extend(_range_filter("X_gsm", params.x_gsm_min, params.x_gsm_max))
        filters.extend(_range_filter("Y_gsm", params.y_gsm_min, params.y_gsm_max))
        filters.extend(_range_filter("Z_gsm", params.z_gsm_min, params.z_gsm_max))
        filters.extend(_range_filter("Bz_imf", params.bz_imf_min, params.bz_imf_max))
        filters.extend(_range_filter("Pd_sw", params.pd_sw_min, params.pd_sw_max))

        if filters:
            lf = lf.filter(pl.all_horizontal(filters))

        if params.columns:
            available = set(lf.collect_schema().names())
            cols = [c for c in params.columns if c in available]
            if cols:
                lf = lf.select(cols)

        return lf.limit(params.limit).collect()


_dataset: MangoDataset | None = None


def get_dataset() -> MangoDataset:
    global _dataset
    if _dataset is None:
        import os

        data_dir = Path(os.environ.get("MANGO_DATA_DIR", str(_DEFAULT_DATA_DIR)))
        _dataset = MangoDataset(data_dir)
    return _dataset
