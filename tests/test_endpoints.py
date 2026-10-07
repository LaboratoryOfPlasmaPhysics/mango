import io
from datetime import datetime

import polars as pl

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
