"""The docs sample must stay small, complete and in the production layout."""

from pathlib import Path

import polars as pl
import pytest
from conftest import SERVED_COLUMNS

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
MAX_BYTES = 10 * 1024 * 1024
EVENT_SC = "THA"


def _region(region: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / region, hive_partitioning=True)


def test_sample_size_cap():
    total = sum(p.stat().st_size for p in DATA.rglob("*.parquet"))
    assert 0 < total <= MAX_BYTES, total


@pytest.mark.parametrize("region", ["magnetosphere", "magnetosheath", "solar_wind"])
def test_region_schema_and_spacecraft(region):
    df = _region(region)
    assert set(df.columns) == set(SERVED_COLUMNS[region]) | {"SC"}
    assert df["SC"].n_unique() >= 2
    assert df.height > 1000


def test_event_window_crosses_all_regions():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_docs_sample", DATA.parent.parent / "scripts" / "make_docs_sample.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start = pl.lit(mod.EVENT_START).str.to_datetime()
    stop = pl.lit(mod.EVENT_STOP).str.to_datetime()
    for region in ("magnetosphere", "magnetosheath", "solar_wind"):
        rows = _region(region).filter(
            (pl.col("SC") == EVENT_SC) & (pl.col("Time") >= start) & (pl.col("Time") < stop))
        assert rows.height > 100, region
