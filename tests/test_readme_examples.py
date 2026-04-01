"""Tests covering the README Quick Start examples using synthetic fixture data."""

import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
from fastapi.testclient import TestClient

import space_mango as sm
from space_mango.app import create_app
from space_mango.dataset import MangoDataset, get_dataset


def _write_region(base: Path, region: str, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    for sc in df["SC"].unique().to_list():
        sc_dir = base / region / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        part = df.filter(pl.col("SC") == sc).drop("SC")
        part.write_parquet(sc_dir / "part-0.parquet")


def _make_client(data_dir: Path) -> sm.MangoClient:
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    tc = TestClient(app)
    client = sm.MangoClient.__new__(sm.MangoClient)
    client._base_url = "http://testserver"
    client._http = httpx.Client(transport=tc._transport, base_url="http://testserver")
    client._filter_cache = {}
    return client


# Use datetime objects so Polars writes Datetime[ns] columns (matching real data)
MAGNETOSHEATH_ROWS = [
    {
        "Time": datetime(2016, 3, 15, 10, 0, 0), "SC": "THA",
        "Bx": 1.0, "By": 2.0, "Bz": 3.0, "Np": 10.0,
        "Vx": -200.0, "Vy": 0.0, "Vz": 0.0, "Tp": 1e6,
        "X_gsm": 8.0, "Y_gsm": 3.0, "Z_gsm": 0.0,
        "D_msh": 0.4,
        "SW_pairing": True, "Bz_imf": -5.0, "By_imf": 1.0, "Bx_imf": 0.5,
        "Pd_sw": 4.0, "Np_sw": 8.0, "Tp_sw": 1e5, "Vx_sw": -400.0,
        "Beta_sw": 1.2, "Ma_sw": 7.0, "Tilt": 0.1,
        "Norma_pos": True,
    },
    {
        "Time": datetime(2018, 7, 20, 14, 30, 0), "SC": "MMS",
        "Bx": 2.0, "By": 3.0, "Bz": 4.0, "Np": 20.0,
        "Vx": -300.0, "Vy": 1.0, "Vz": 1.0, "Tp": 2e6,
        "X_gsm": 9.0, "Y_gsm": 4.0, "Z_gsm": 1.0,
        "D_msh": 0.8,
        "SW_pairing": False, "Bz_imf": 2.0, "By_imf": -1.0, "Bx_imf": -0.3,
        "Pd_sw": 1.0, "Np_sw": 5.0, "Tp_sw": 5e4, "Vx_sw": -350.0,
        "Beta_sw": 0.8, "Ma_sw": 5.0, "Tilt": -0.2,
        "Norma_pos": False,
    },
    {
        "Time": datetime(2019, 1, 5, 8, 0, 0), "SC": "C1",
        "Bx": 0.5, "By": -1.0, "Bz": -2.0, "Np": 15.0,
        "Vx": -250.0, "Vy": -0.5, "Vz": 0.5, "Tp": 1.5e6,
        "X_gsm": 10.0, "Y_gsm": -2.0, "Z_gsm": 0.5,
        "D_msh": 0.2,
        "SW_pairing": True, "Bz_imf": -8.0, "By_imf": 3.0, "Bx_imf": 1.0,
        "Pd_sw": 6.0, "Np_sw": 12.0, "Tp_sw": 2e5, "Vx_sw": -500.0,
        "Beta_sw": 2.0, "Ma_sw": 10.0, "Tilt": 0.3,
        "Norma_pos": True,
    },
]

MAGNETOSPHERE_ROWS = [
    {
        "Time": datetime(2015, 6, 10, 12, 0, 0), "SC": "MMS",
        "Bx": 10.0, "By": -5.0, "Bz": -20.0, "Np": 1.0,
        "Vx": -50.0, "Vy": 10.0, "Vz": 5.0, "Tp": 5e7,
        "X_gsm": -5.0, "Y_gsm": 2.0, "Z_gsm": 1.0,
        "D_msp": 0.3,
        "SW_pairing": True, "Bz_imf": -3.0, "By_imf": 0.0, "Bx_imf": 0.0,
        "Pd_sw": 2.0, "Np_sw": 6.0, "Tp_sw": 1e5, "Vx_sw": -380.0,
        "Beta_sw": 1.0, "Ma_sw": 6.0, "Tilt": 0.15,
        "Norma_pos": True,
    },
    {
        "Time": datetime(2017, 11, 3, 6, 0, 0), "SC": "THA",
        "Bx": 15.0, "By": 3.0, "Bz": -30.0, "Np": 0.5,
        "Vx": -30.0, "Vy": 5.0, "Vz": -2.0, "Tp": 8e7,
        "X_gsm": -8.0, "Y_gsm": -1.0, "Z_gsm": -0.5,
        "D_msp": 0.6,
        "SW_pairing": True, "Bz_imf": 1.0, "By_imf": 2.0, "Bx_imf": -1.0,
        "Pd_sw": 3.0, "Np_sw": 7.0, "Tp_sw": 1.5e5, "Vx_sw": -420.0,
        "Beta_sw": 1.5, "Ma_sw": 8.0, "Tilt": -0.1,
        "Norma_pos": True,
    },
    {
        "Time": datetime(2021, 2, 14, 18, 0, 0), "SC": "C3",
        "Bx": 5.0, "By": -2.0, "Bz": -10.0, "Np": 2.0,
        "Vx": -80.0, "Vy": 0.0, "Vz": 0.0, "Tp": 3e7,
        "X_gsm": -3.0, "Y_gsm": 5.0, "Z_gsm": 2.0,
        "D_msp": 0.9,
        "SW_pairing": False, "Bz_imf": -1.0, "By_imf": -3.0, "Bx_imf": 0.5,
        "Pd_sw": 1.5, "Np_sw": 4.0, "Tp_sw": 8e4, "Vx_sw": -360.0,
        "Beta_sw": 0.6, "Ma_sw": 4.0, "Tilt": 0.05,
        "Norma_pos": False,
    },
]

SOLAR_WIND_ROWS = [
    {
        "Time": datetime(2016, 5, 1, 0, 0, 0), "SC": "THA",
        "Bx": 0.1, "By": -0.5, "Bz": -1.0, "Np": 5.0,
        "Vx": -400.0, "Vy": 0.0, "Vz": 0.0, "Tp": 1e5,
        "X_gsm": 20.0, "Y_gsm": 0.0, "Z_gsm": 0.0,
    },
]


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> sm.MangoClient:
    base = tmp_path_factory.mktemp("mango_data")
    _write_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
    _write_region(base, "magnetosphere", MAGNETOSPHERE_ROWS)
    _write_region(base, "solar_wind", SOLAR_WIND_ROWS)
    return _make_client(base)


# --- README example: sm.regions() ---

def test_regions_returns_all_three(client):
    regions = client.regions()
    assert set(regions) == {"magnetosphere", "magnetosheath", "solar_wind"}


# --- README example: sm.get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3) ---

def test_get_data_magnetosheath_southward_imf_high_pressure(client):
    df = client.get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0
    assert all(df["Bz_imf"] <= -2)
    assert all(df["Pd_sw"] >= 3)


# --- README example: sm.get_data("magnetosphere", columns=..., spacecraft=..., time_min/max=...) ---

def test_get_data_magnetosphere_columns_spacecraft_time(client):
    df = client.get_data(
        "magnetosphere",
        columns=["X_gsm", "Y_gsm", "Z_gsm", "Np", "Bz"],
        spacecraft=["MMS", "THA"],
        time_min="2015-01-01",
        time_max="2020-12-31",
    )
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2  # MMS(2015) and THA(2017), not C3(2021)
    assert set(df.columns) == {"X_gsm", "Y_gsm", "Z_gsm", "Np", "Bz"}


def test_get_data_time_min_filters_correctly(client):
    df = client.get_data("magnetosphere", time_min="2018-01-01")
    assert len(df) == 1
    assert df["SC"][0] == "C3"


def test_get_data_time_max_filters_correctly(client):
    df = client.get_data("magnetosphere", time_max="2016-01-01")
    assert len(df) == 1
    assert df["SC"][0] == "MMS"


def test_get_data_time_range_excludes_outside(client):
    df = client.get_data("magnetosphere", time_min="2017-01-01", time_max="2018-01-01")
    assert len(df) == 1
    assert df["SC"][0] == "THA"


# --- README example: sm.filters("magnetosheath") ---

def test_filters_magnetosheath(client):
    filters = client.filters("magnetosheath")
    assert isinstance(filters, list)
    assert len(filters) > 0
    names = {f["name"] for f in filters}
    assert "bz_imf" in names
    assert "pd_sw" in names
    assert "d_msh" in names
    # d_msp is magnetosphere-only
    assert "d_msp" not in names


# --- README example: sm.columns("magnetosphere") ---

def test_columns_magnetosphere(client):
    cols = client.columns("magnetosphere")
    assert isinstance(cols, list)
    for expected in ["X_gsm", "Y_gsm", "Z_gsm", "Np", "Bz", "Time"]:
        assert expected in cols


# --- Additional coverage: spacecraft filter ---

def test_get_data_spacecraft_single(client):
    df = client.get_data("magnetosheath", spacecraft=["C1"])
    assert len(df) == 1
    assert df["SC"][0] == "C1"


def test_get_data_spacecraft_multiple(client):
    df = client.get_data("magnetosheath", spacecraft=["THA", "MMS"])
    assert len(df) == 2
    assert set(df["SC"].to_list()) == {"THA", "MMS"}


def test_get_data_spacecraft_no_match(client):
    df = client.get_data("magnetosheath", spacecraft=["NONEXISTENT"])
    assert len(df) == 0


# --- Additional coverage: limit ---

def test_get_data_limit(client):
    df = client.get_data("magnetosheath", limit=1)
    assert len(df) == 1


# --- Additional coverage: combined filters ---

def test_get_data_combined_range_and_spacecraft(client):
    df = client.get_data("magnetosheath", spacecraft=["THA", "C1"], bz_imf_max=-2)
    assert len(df) > 0
    assert all(df["Bz_imf"] <= -2)
    assert all(sc in ("THA", "C1") for sc in df["SC"].to_list())


# --- Additional coverage: sw_paired_only, normalized_only ---

def test_get_data_sw_paired_only(client):
    df = client.get_data("magnetosheath", sw_paired_only=True)
    assert all(df["SW_pairing"].to_list())


def test_get_data_normalized_only(client):
    df = client.get_data("magnetosheath", normalized_only=True)
    assert all(df["Norma_pos"].to_list())


# --- Additional coverage: columns selection ---

def test_get_data_column_subset(client):
    df = client.get_data("magnetosheath", columns=["Np", "Bz"])
    assert set(df.columns) == {"Np", "Bz"}


def test_get_data_column_nonexistent_ignored(client):
    df = client.get_data("magnetosheath", columns=["Np", "DOES_NOT_EXIST"])
    assert "Np" in df.columns
    assert "DOES_NOT_EXIST" not in df.columns


# --- Additional coverage: all regions queryable ---

def test_get_data_solar_wind(client):
    df = client.get_data("solar_wind")
    assert len(df) == 1


def test_columns_solar_wind(client):
    cols = client.columns("solar_wind")
    assert "Time" in cols
    assert "X_gsm" in cols


def test_filters_magnetosphere_has_d_msp(client):
    filters = client.filters("magnetosphere")
    names = {f["name"] for f in filters}
    assert "d_msp" in names
    assert "d_msh" not in names


# --- Error handling ---

def test_get_data_invalid_region(client):
    with pytest.raises(Exception):
        client.get_data("invalid_region")
