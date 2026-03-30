import tempfile
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from mango.app import create_app
from mango.dataset import MangoDataset
from mango.models import Region

client = TestClient(create_app())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_regions():
    r = client.get("/api/v1/regions")
    assert r.status_code == 200
    assert set(r.json()) == {"magnetosphere", "magnetosheath", "solar_wind"}


def test_filters_magnetosheath():
    r = client.get("/api/v1/regions/magnetosheath/filters")
    assert r.status_code == 200
    filters = r.json()
    names = {f["name"] for f in filters}
    # Solar wind conditions should be available
    assert "bz_imf" in names
    assert "pd_sw" in names
    assert "beta_sw" in names
    assert "ma_sw" in names
    # Sheath-specific
    assert "d_msh" in names
    # Magnetosphere-specific should NOT be here
    assert "d_msp" not in names


def test_filters_magnetosphere():
    r = client.get("/api/v1/regions/magnetosphere/filters")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()}
    assert "d_msp" in names
    assert "d_msh" not in names
    assert "bz_imf" in names


def test_filters_solar_wind():
    r = client.get("/api/v1/regions/solar_wind/filters")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()}
    # Solar wind region has no upstream pairing
    assert "bz_imf" not in names
    assert "pd_sw" not in names
    # But spatial and local plasma filters still apply
    assert "x_gsm" in names
    assert "np" in names


def _write_test_region(base: Path, region: str, rows: list[dict]) -> None:
    """Write a small Hive-partitioned Parquet dataset for testing."""
    df = pl.DataFrame(rows)
    for sc in df["SC"].unique().to_list():
        sc_dir = base / region / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        part = df.filter(pl.col("SC") == sc).drop("SC")
        part.write_parquet(sc_dir / "part-0.parquet")


MAGNETOSHEATH_ROWS = [
    {"Time": "2010-01-01T00:00:00", "SC": "THA", "Bx": 1.0, "By": 2.0, "Bz": 3.0,
     "Np": 10.0, "Vx": -200.0, "Vy": 0.0, "Vz": 0.0, "Tp": 1e6,
     "X_gsm": 8.0, "Y_gsm": 3.0, "Z_gsm": 0.0,
     "SW_pairing": True, "Bz_imf": -5.0, "Pd_sw": 3.0, "Norma_pos": True},
    {"Time": "2010-01-01T00:00:05", "SC": "MMS", "Bx": 2.0, "By": 3.0, "Bz": 4.0,
     "Np": 20.0, "Vx": -300.0, "Vy": 1.0, "Vz": 1.0, "Tp": 2e6,
     "X_gsm": 9.0, "Y_gsm": 4.0, "Z_gsm": 1.0,
     "SW_pairing": False, "Bz_imf": 2.0, "Pd_sw": 1.0, "Norma_pos": False},
]


def test_hive_dataset_query_all():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(Region.magnetosheath, {}, limit=100)
        assert len(df) == 2
        assert "SC" in df.columns
        assert "Time" in df.columns


def test_hive_dataset_query_spacecraft_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(Region.magnetosheath, {}, spacecraft=["THA"], limit=100)
        assert len(df) == 1
        assert df["SC"][0] == "THA"


def test_hive_dataset_query_time_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(
            Region.magnetosheath, {},
            time_min="2010-01-01T00:00:03",
            limit=100,
        )
        assert len(df) == 1
