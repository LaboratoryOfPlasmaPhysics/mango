from datetime import datetime

import polars as pl
import pytest

from space_mango.errors import QueryError
from space_mango.filtering import (
    build_filter_exprs,
    parse_range_params,
    parse_time,
    time_window,
)


def test_parse_time_accepts_iso_and_converts_tz_to_naive_utc():
    assert parse_time("2017-01-12T10:00:00", param="start") == datetime(2017, 1, 12, 10)
    assert parse_time("2017-01-12T12:00:00+02:00", param="start") == datetime(2017, 1, 12, 10)


def test_parse_time_rejects_garbage():
    with pytest.raises(QueryError) as e:
        parse_time("yesterday", param="start")
    assert e.value.code == "bad_time"


def test_time_window_new_and_legacy():
    assert time_window("2017-01-01", "2018-01-01", None, None) == (
        datetime(2017, 1, 1), datetime(2018, 1, 1), False)
    assert time_window(None, None, "2017-01-01", "2018-01-01") == (
        datetime(2017, 1, 1), datetime(2018, 1, 1), True)


def test_parse_range_params_skips_reserved_and_validates():
    raw = {"limit": "3", "spacecraft": "THA", "bz_imf_max": "-2", "d_msh_min": "0"}
    assert parse_range_params("magnetosheath", raw) == {"bz_imf_max": -2.0, "d_msh_min": 0.0}


@pytest.mark.parametrize("key", ["foo_min", "tilt_min", "bz_imf", "d_msp_max"])
def test_parse_range_params_unknown_is_error(key):
    with pytest.raises(QueryError) as e:
        parse_range_params("magnetosheath", {key: "1"})
    assert e.value.code == "unknown_filter"
    assert "bz_imf_max" in e.value.valid


def test_parse_range_params_non_numeric():
    with pytest.raises(QueryError) as e:
        parse_range_params("magnetosheath", {"bz_imf_max": "south"})
    assert e.value.code == "bad_filter_value"


def test_build_filter_exprs_applies_everything():
    df = pl.DataFrame({
        "Time": [datetime(2016, 1, 1), datetime(2017, 1, 1), datetime(2018, 1, 1)],
        "SC": ["THA", "THA", "C1"],
        "Bz_imf": [-5.0, -1.0, -8.0],
        "SW_pairing": [True, True, False],
        "Norma_pos": [True, True, True],
    })
    exprs = build_filter_exprs(
        "magnetosheath", set(df.columns),
        spacecraft=["THA", "C1"], start=datetime(2016, 1, 1), stop=datetime(2018, 1, 1),
        sw_paired_only=True, ranges={"bz_imf_max": -2.0},
    )
    out = df.filter(pl.all_horizontal(exprs))
    assert out["Time"].to_list() == [datetime(2016, 1, 1)]


def test_stop_inclusive_flag():
    df = pl.DataFrame({"Time": [datetime(2018, 1, 1)]})
    excl = build_filter_exprs("solar_wind", {"Time"}, stop=datetime(2018, 1, 1))
    incl = build_filter_exprs("solar_wind", {"Time"}, stop=datetime(2018, 1, 1), stop_inclusive=True)
    assert df.filter(pl.all_horizontal(excl)).height == 0
    assert df.filter(pl.all_horizontal(incl)).height == 1


def test_flag_unavailable_in_region():
    with pytest.raises(QueryError) as e:
        build_filter_exprs("solar_wind", {"Time"}, sw_paired_only=True)
    assert e.value.code == "flag_unavailable"


def test_server_rejects_unknown_spacecraft_with_suggestion(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"spacecraft": "MMS1"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["error"] == "unknown_spacecraft"
    assert "Did you mean 'MMS'?" in d["message"]


def test_server_rejects_unknown_column(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"columns": ["Np", "Nope"]})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unknown_column"


def test_server_rejects_unknown_query_param(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"foo_min": 1})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unknown_filter"


def test_server_rejects_bad_time_with_400_not_500(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"time_min": "not-a-date"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "bad_time"


def test_server_start_stop_half_open(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={
        "format": "csv", "start": "2016-03-15T10:00:00", "stop": "2018-07-20T14:30:00"})
    assert r.status_code == 200
    assert len(r.text.strip().splitlines()) == 2  # header + THA only (MMS row is at stop)
