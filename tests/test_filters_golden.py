"""Each of the four filters that were dead on the live server must actually filter."""

import pytest


@pytest.mark.parametrize(
    ("region", "params", "expected_sc"),
    [
        ("magnetosheath", {"d_msh_max": 0.3}, {"C1"}),
        ("magnetosphere", {"d_msp_min": 0.5}, {"THA", "C3"}),
        ("magnetosphere", {"tilt_min": 0.1}, {"MMS"}),
        ("magnetosphere", {"tilt_max": 0.0}, {"THA"}),
    ],
)
def test_previously_dead_filter_filters(client, region, params, expected_sc):
    df = client.get_data(region, limit=100, **params)
    assert set(df["SC"].to_list()) == expected_sc


def test_tilt_is_not_a_magnetosheath_filter(client):
    names = {f["name"] for f in client.filters("magnetosheath")}
    assert "tilt" not in names


def test_filter_on_missing_column_is_400(make_row, make_dataset, make_api):
    from datetime import datetime

    row = make_row("magnetosheath", "THA", datetime(2016, 1, 1))
    del row["R_norm"]
    api = make_api(make_dataset({"magnetosheath": [row]}))
    r = api.get("/api/v1/regions/magnetosheath/data", params={"d_msh_max": 0.3})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "filter_column_missing"
    assert "R_norm" in detail["message"]
