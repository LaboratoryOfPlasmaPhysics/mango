from datetime import datetime, timedelta

import polars as pl
import pytest

import space_mango as sm
from space_mango.result import MangoResult


def _result(rows, region="magnetosheath"):
    df = pl.DataFrame(rows)
    info = {"Np": {"unit": "cm⁻³", "frame": "", "description": "Local ion density"}}
    return MangoResult(df, info, "2026.0", {"region": region}, "@misc{x}", region)


def test_get_data_returns_result_with_metadata(client):
    r = client.get_data("magnetosheath", columns=["Time", "Np", "Bz_imf"], limit=10)
    assert isinstance(r, sm.MangoResult)
    assert r.version == "2026.0"
    assert r.metadata["Bz_imf"]["unit"] == "nT"
    assert set(r.metadata) == {"Time", "Np", "Bz_imf"}
    assert len(r) == 3 and r.columns == ["Time", "Np", "Bz_imf"]
    assert r.query["region"] == "magnetosheath"
    assert "2026.0" in r.cite()
    assert "MangoResult" in repr(r) and "2026.0" in repr(r)


def test_to_pandas_keeps_metadata():
    pytest.importorskip("pandas")
    pdf = _result({"Time": [datetime(2016, 1, 1)], "Np": [3.0]}).to_pandas()
    assert pdf.attrs["mango"]["version"] == "2026.0"
    assert pdf.attrs["mango"]["columns"]["Np"]["unit"] == "cm⁻³"


def test_to_xarray_keeps_units():
    pytest.importorskip("xarray")
    ds = _result({"Time": [datetime(2016, 1, 1), datetime(2016, 1, 1)],
                  "SC": ["THA", "C1"], "Np": [3.0, 4.0]}).to_xarray()
    assert ds["Np"].attrs["units"] == "cm⁻³"
    assert ds.attrs["mango_version"] == "2026.0"
    assert ds.sizes["index"] == 2  # same Time on two spacecraft: no unique time index


def test_to_intervals_splits_on_region_sc_and_gaps():
    t0 = datetime(2017, 1, 12, 10)
    s = timedelta(seconds=5)
    df = pl.DataFrame({
        "Time": [t0, t0 + s, t0 + 2 * s, t0 + 3 * s, t0 + 3 * s + timedelta(minutes=5), t0],
        "SC": ["THA"] * 5 + ["C1"],
        "region": ["magnetosphere", "magnetosphere", "magnetosheath", "magnetosheath",
                   "magnetosheath", "solar_wind"],
    })
    iv = MangoResult(df, {}, "2026.0", {}, "", None).to_intervals()
    assert iv.columns == ["sc", "region", "start", "stop", "n_points"]
    assert iv.rows() == [
        ("C1", "solar_wind", t0, t0, 1),
        ("THA", "magnetosphere", t0, t0 + s, 2),
        ("THA", "magnetosheath", t0 + 2 * s, t0 + 3 * s, 2),
        ("THA", "magnetosheath", t0 + 3 * s + timedelta(minutes=5), t0 + 3 * s + timedelta(minutes=5), 1),
    ]


def test_to_intervals_on_single_region_result(client):
    iv = client.get_data("magnetosheath", limit=10).to_intervals()
    assert set(iv["region"].to_list()) == {"magnetosheath"}
    assert iv["n_points"].sum() == 3


def test_to_intervals_without_time_or_sc_raises_mango_error(client):
    r = client.get_data("magnetosheath", columns=["Np"])
    with pytest.raises(sm.MangoError, match="Time and SC"):
        r.to_intervals()
