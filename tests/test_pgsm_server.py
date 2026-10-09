"""Server side of PGSM: exact counts on /count, refused elsewhere."""

import math

import pytest


@pytest.fixture
def pgsm_api(make_row, make_dataset, make_api):
    from datetime import datetime

    msh, msp = "magnetosheath", "magnetosphere"
    rows = {
        # make_row defaults every float to 1.0 and every flag to True:
        # IMF (1, 1, 1), folded cone acos(1/sqrt 3) = 54.7 deg.
        msh: [
            make_row(msh, "THA", datetime(2016, 1, 1, 0, 0, 0)),
            make_row(msh, "THA", datetime(2016, 1, 1, 0, 0, 5), Bx_imf=0.0, By_imf=0.0, Bz_imf=2.0),
            make_row(msh, "THA", datetime(2016, 1, 1, 0, 0, 10), Norma_pos=False),
        ],
        msp: [
            make_row(msp, "THA", datetime(2016, 1, 1, 0, 0, 0), tilt=math.radians(2.0)),
            make_row(msp, "THA", datetime(2016, 1, 1, 0, 0, 5), tilt=math.radians(-12.0)),
            make_row(msp, "THA", datetime(2016, 1, 1, 0, 0, 10), tilt=math.radians(12.0),
                     Norma_pos=False),
        ],
        "solar_wind": [make_row("solar_wind", "THA", datetime(2016, 1, 1))],
    }
    return make_api(make_dataset(rows))


def _count(api, region, **params):
    r = api.get(f"/api/v1/regions/{region}/count", params=params)
    return r.status_code, r.json()


@pytest.mark.parametrize(
    ("cone", "n"),
    [((50, 60), 1), ((120, 130), 1), ((0, 180), 4), ((90, 90), 2), ((85, 95), 2), ((60, 80), 0)],
)
def test_count_magnetosheath(pgsm_api, cone, n):
    # row 1: f = 54.7 -> one row in (50,60), one in (120,130), two in (0,180)
    # row 2: f = 90 -> two rows whenever 90 is in range; row 3 is not normalized
    status, body = _count(pgsm_api, "magnetosheath", frame="pgsm",
                          cone_min=cone[0], cone_max=cone[1])
    assert status == 200 and body["n_rows"] == n


@pytest.mark.parametrize(("tilt", "n"), [((-5, 5), 2), ((10, 15), 1), ((-15, -10), 1), ((20, 30), 0)])
def test_count_magnetosphere(pgsm_api, tilt, n):
    status, body = _count(pgsm_api, "magnetosphere", frame="pgsm",
                          tilt_deg_min=tilt[0], tilt_deg_max=tilt[1])
    assert status == 200 and body["n_rows"] == n


@pytest.mark.parametrize(
    ("region", "params"),
    [
        ("magnetosheath", {"cone_min": 0, "cone_max": 90}),
        ("solar_wind", {"frame": "pgsm"}),
        ("magnetosheath", {"frame": "pgsm", "cone_min": 0}),
        ("magnetosphere", {"frame": "pgsm", "cone_min": 0, "cone_max": 9}),
    ],
)
def test_count_refuses_bad_frame(pgsm_api, region, params):
    status, body = _count(pgsm_api, region, **params)
    assert status == 400 and body["detail"]["error"] == "bad_frame"


def test_data_endpoint_refuses_frame(pgsm_api):
    r = pgsm_api.get("/api/v1/regions/magnetosheath/data", params={"frame": "pgsm"})
    assert r.status_code == 400


def test_dataset_advertises_frames(pgsm_api):
    assert "frames" in pgsm_api.get("/api/v1/dataset").json()["features"]
