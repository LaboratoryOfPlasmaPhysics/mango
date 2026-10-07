import io
from datetime import datetime

import polars as pl
import pytest

from space_mango.dataset import MangoDataset


def test_describe_lists_columns_with_units_and_filters(api):
    d = api.get("/api/v1/regions/magnetosheath/describe").json()
    assert d["region"] == "magnetosheath"
    assert "bow shock" in d["definition"]
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["Bz_imf"]["unit"] == "nT" and cols["Bz_imf"]["filter"] == "bz_imf"
    assert cols["R_norm"]["filter"] == "d_msh"
    assert "bow shock" in cols["R_norm"]["description"]
    assert cols["SC"]["dtype"] == "String"
    assert {f["name"] for f in d["filters"]} >= {"bz_imf", "d_msh"}


def test_spacecraft_coverage(api):
    rows = api.get("/api/v1/regions/magnetosphere/spacecraft").json()
    assert [r["sc"] for r in rows] == ["C3", "MMS", "THA"]
    mms = rows[1]
    assert mms["start"].startswith("2015-06-10T12:00") and mms["n_rows"] == 1


def test_count_matches_data(api):
    c = api.get("/api/v1/regions/magnetosheath/count", params={"bz_imf_max": -2}).json()
    assert c["n_rows"] == 2
    assert c["est_bytes"] > 0
    c2 = api.get(
        "/api/v1/regions/magnetosheath/count",
        params={"bz_imf_max": -2, "columns": ["Np"]},
    ).json()
    assert c2["est_bytes"] == 2 * 8


def test_count_rejects_unknown_filter(api):
    r = api.get("/api/v1/regions/magnetosheath/count", params={"tilt_min": 0})
    assert r.status_code == 400


def test_timeline_spans_regions(make_row, make_dataset, make_api):
    rows = {
        "magnetosphere": [make_row("magnetosphere", "THA", datetime(2017, 1, 12, 10, 0))],
        "magnetosheath": [make_row("magnetosheath", "THA", datetime(2017, 1, 12, 10, 30))],
        "solar_wind": [
            make_row("solar_wind", "THA", datetime(2017, 1, 12, 11, 0)),
            make_row("solar_wind", "C1", datetime(2017, 1, 12, 11, 0)),
        ],
    }
    api = make_api(make_dataset(rows))
    r = api.get(
        "/api/v1/timeline",
        params={"sc": "THA", "start": "2017-01-12T09:00", "stop": "2017-01-12T12:00"},
    )
    assert r.status_code == 200
    df = pl.read_ipc(io.BytesIO(r.content))
    assert df["region"].to_list() == ["magnetosphere", "magnetosheath", "solar_wind"]
    assert set(df["SC"].to_list()) == {"THA"}
    assert "R_bs" in df.columns  # magnetosheath-only column, null elsewhere


def test_timeline_limits(api):
    r = api.get(
        "/api/v1/timeline",
        params={"sc": "THA", "start": "2016-01-01", "stop": "2017-01-01"},
    )
    assert r.status_code == 400 and r.json()["detail"]["error"] == "span_too_long"
    r = api.get(
        "/api/v1/timeline",
        params={"sc": "MMS1", "start": "2016-01-01", "stop": "2016-01-02"},
    )
    assert r.status_code == 400 and "Did you mean 'MMS'" in r.json()["detail"]["message"]


def test_dataset_endpoint_and_version_header(api):
    r = api.get("/api/v1/dataset")
    d = r.json()
    assert d["version"] == "2026.0"
    assert d["citation"].startswith("@") and d["doi"] is None
    assert len(d["schema_checksum"]) == 64
    assert r.headers["X-Mango-Dataset-Version"] == "2026.0"
    assert api.get("/health").headers["X-Mango-Dataset-Version"] == "2026.0"


def test_check_catalog_reports_drift(make_row, make_dataset):
    row = make_row("magnetosheath", "THA", datetime(2016, 1, 1))
    row["Extra"] = 1.0
    del row["R_bs"]
    problems = MangoDataset(make_dataset({"magnetosheath": [row]})).check_catalog()
    assert any("R_bs" in p for p in problems)
    assert any("Extra" in p for p in problems)


def test_check_catalog_clean_on_served_schema(dataset_dir):
    assert MangoDataset(dataset_dir).check_catalog() == []


# --- final-review fix wave: the server never silently ignores a parameter ------------------

@pytest.mark.parametrize("endpoint", ["data", "count"])
def test_data_and_count_reject_sc(api, endpoint):
    r = api.get(f"/api/v1/regions/magnetosheath/{endpoint}", params={"sc": "THA"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_filter" and "'sc'" in detail["message"]
    assert "spacecraft" in detail["valid"] and "bz_imf_max" in detail["valid"]


@pytest.mark.parametrize("param", ["limit", "format"])
def test_count_rejects_data_only_params(api, param):
    r = api.get("/api/v1/regions/magnetosheath/count", params={param: "1"})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "unknown_filter"


@pytest.mark.parametrize("extra", [{"bz_imf_max": "-2"}, {"spacecraft": "THA"}, {"limit": "3"}])
def test_timeline_rejects_unknown_params(api, extra):
    params = {"sc": "THA", "start": "2016-03-15", "stop": "2016-03-16", **extra}
    r = api.get("/api/v1/timeline", params=params)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_parameter"
    assert detail["valid"] == ["columns", "format", "sc", "start", "stop"]


@pytest.mark.parametrize("endpoint", ["data", "count"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "NaN"])
def test_non_finite_filter_values_rejected(api, endpoint, value):
    r = api.get(f"/api/v1/regions/magnetosheath/{endpoint}", params={"bz_imf_max": value})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "bad_filter_value"


@pytest.mark.parametrize("endpoint", ["data", "count"])
@pytest.mark.parametrize(
    "params",
    [
        {"start": "2017-01-01", "stop": "2017-01-01"},
        {"start": "2018-01-01", "stop": "2017-01-01"},
        {"time_min": "2018-01-01", "time_max": "2017-01-01"},
    ],
)
def test_start_not_before_stop_rejected(api, endpoint, params):
    r = api.get(f"/api/v1/regions/magnetosheath/{endpoint}", params=params)
    assert r.status_code == 400 and r.json()["detail"]["error"] == "bad_time"


def test_legacy_inclusive_single_instant_is_allowed(api):
    t = "2016-03-15T10:00:00"
    r = api.get("/api/v1/regions/magnetosheath/count", params={"time_min": t, "time_max": t})
    assert r.status_code == 200 and r.json()["n_rows"] == 1


def test_duplicate_columns_are_deduplicated_by_server(api):
    r = api.get("/api/v1/regions/magnetosheath/data",
                params={"columns": ["Time", "Np", "Time"], "format": "arrow"})
    assert r.status_code == 200
    assert pl.read_ipc(io.BytesIO(r.content)).columns == ["Time", "Np"]
    c = api.get("/api/v1/regions/magnetosheath/count", params={"columns": ["Np", "Np"]})
    assert c.status_code == 200
    t = api.get("/api/v1/timeline", params={"sc": "THA", "start": "2016-03-15",
                                            "stop": "2016-03-16", "columns": ["Np", "Np"]})
    assert t.status_code == 200
    assert pl.read_ipc(io.BytesIO(t.content)).columns == ["Time", "SC", "Np", "region"]
