import tempfile
from pathlib import Path

import httpx
import polars as pl
import pytest
from fastapi.testclient import TestClient

import mango
from mango.app import create_app
from mango.client import MangoFilterError, _validate_filters
from mango.dataset import MangoDataset, get_dataset


MAGNETOSHEATH_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
    "x_gsm", "y_gsm", "z_gsm", "d_msh", "np", "tp", "bz",
}

MAGNETOSPHERE_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
    "x_gsm", "y_gsm", "z_gsm", "d_msp", "np", "tp", "bz",
}


def test_validate_filters_valid():
    _validate_filters(
        {"bz_imf_max": -2.0, "pd_sw_min": 3.0},
        "magnetosheath",
        MAGNETOSHEATH_FILTERS,
        {"magnetosphere": MAGNETOSPHERE_FILTERS},
    )


def test_validate_filters_bad_suffix():
    with pytest.raises(MangoFilterError, match="must end with '_min' or '_max'"):
        _validate_filters(
            {"bz_imf": -2.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_unknown_filter():
    with pytest.raises(MangoFilterError, match="not a valid filter"):
        _validate_filters(
            {"fake_min": 1.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_wrong_region_with_hint():
    with pytest.raises(MangoFilterError, match="available for region 'magnetosphere'"):
        _validate_filters(
            {"d_msp_min": 0.5},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {"magnetosphere": MAGNETOSPHERE_FILTERS},
        )


def test_validate_filters_non_numeric():
    with pytest.raises(MangoFilterError, match="must be numeric"):
        _validate_filters(
            {"bz_imf_max": "not_a_number"},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def _write_test_region(base: Path, region: str, rows: list[dict]) -> None:
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


def _make_mango_client(data_dir: Path) -> mango.MangoClient:
    """Create a MangoClient backed by a test server with fixture data."""
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    tc = TestClient(app)
    client = mango.MangoClient.__new__(mango.MangoClient)
    client._base_url = "http://testserver"
    client._http = httpx.Client(transport=tc._transport, base_url="http://testserver")
    client._filter_cache = {}
    return client


def test_client_get_data_all():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_mango_client(base)

        df = client.get_data("magnetosheath", limit=10)
        assert len(df) == 2
        assert "SC" in df.columns
        assert "Bz_imf" in df.columns


def test_client_get_data_spacecraft_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_mango_client(base)

        df = client.get_data("magnetosheath", spacecraft=["THA"], limit=10)
        assert len(df) == 1
        assert df["SC"][0] == "THA"


def test_client_get_data_range_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_mango_client(base)

        df = client.get_data("magnetosheath", bz_imf_max=-1.0, limit=10)
        assert len(df) == 1
        assert df["Bz_imf"][0] == -5.0


def test_client_regions():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_mango_client(base)

        assert "magnetosheath" in client.regions()
