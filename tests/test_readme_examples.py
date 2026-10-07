"""Tests covering the README Quick Start examples using synthetic fixture data."""

import polars as pl
import pytest

import space_mango as sm

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
    with pytest.warns(FutureWarning):
        df = client.get_data(
            "magnetosphere",
            columns=["Time", "X_gsm", "Y_gsm", "Z_gsm", "Np", "Bz"],
            spacecraft=["MMS", "THA"],
            time_min="2015-01-01",
            time_max="2020-12-31",
        )
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2  # MMS(2015) and THA(2017), not C3(2021)
    assert set(df.columns) == {"Time", "X_gsm", "Y_gsm", "Z_gsm", "Np", "Bz"}


def test_get_data_time_min_filters_correctly(client):
    with pytest.warns(FutureWarning):
        df = client.get_data("magnetosphere", time_min="2018-01-01")
    assert len(df) == 1
    assert df["SC"][0] == "C3"


def test_get_data_time_max_filters_correctly(client):
    with pytest.warns(FutureWarning):
        df = client.get_data("magnetosphere", time_max="2016-01-01")
    assert len(df) == 1
    assert df["SC"][0] == "MMS"


def test_get_data_time_range_excludes_outside(client):
    with pytest.warns(FutureWarning):
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
    with pytest.raises(sm.UnknownSpacecraftError):
        client.get_data("magnetosheath", spacecraft=["NONEXISTENT"])


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


def test_get_data_column_nonexistent_is_error(client):
    with pytest.raises(sm.UnknownColumnError):
        client.get_data("magnetosheath", columns=["Np", "DOES_NOT_EXIST"])


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
    with pytest.raises(sm.UnknownRegionError):
        client.get_data("invalid_region")
