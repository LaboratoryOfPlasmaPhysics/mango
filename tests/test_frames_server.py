"""Server side of frames: selections and projection on /data, exact counts on /count."""

import math
from datetime import datetime

import polars as pl
import pytest

MSH, MSP = "magnetosheath", "magnetosphere"


@pytest.fixture
def frames_api(make_row, make_dataset, make_api):
    t = datetime(2016, 1, 1)
    def at(s):
        return t.replace(second=s)
    rows = {
        MSH: [
            # clock 90 deg (By>0, Bz=0), GSM cone acos(1/sqrt2) = 45 deg
            make_row(MSH, "THA", at(0), Bx_imf=1.0, By_imf=1.0, Bz_imf=0.0,
                     Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0),
            # clock 0 deg (northward), GSM cone 90 deg
            make_row(MSH, "THA", at(5), Bx_imf=0.0, By_imf=0.0, Bz_imf=3.0,
                     Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0),
            # clock 350 deg, cone ~ 90; not paired with the solar wind
            make_row(MSH, "THA", at(10), Bx_imf=0.0, By_imf=-0.5, Bz_imf=3.0, SW_pairing=False),
        ],
        MSP: [
            make_row(MSP, "THA", at(0), tilt=math.radians(12.0)),
            make_row(MSP, "THA", at(5), tilt=math.radians(-12.0)),
        ],
        "solar_wind": [make_row("solar_wind", "THA", t)],
    }
    return make_api(make_dataset(rows))


def _data(api, region, **params):
    r = api.get(f"/api/v1/regions/{region}/data", params=params)
    assert r.status_code == 200, r.text
    return pl.read_ipc(r.content)


def _count(api, region, **params):
    r = api.get(f"/api/v1/regions/{region}/count", params=params)
    return r.status_code, r.json()


@pytest.mark.parametrize(
    ("params", "n"),
    [({"cone_min": 40, "cone_max": 50}, 1), ({"clock_min": 330, "clock_max": 30}, 1),
     ({"clock_min": 0, "clock_max": 360}, 2), ({"cone_min": 0, "cone_max": 180}, 2),
     ({"frame": "swi", "cone_min": 40, "cone_max": 50}, 1), ({"frame": "swi"}, 2)],
)
def test_magnetosheath_selection_data_and_count(frames_api, params, n):
    assert _data(frames_api, MSH, **params).height == n
    status, body = _count(frames_api, MSH, **params)
    assert status == 200 and body["n_rows"] == n


def test_tilt_selection_in_degrees(frames_api):
    df = _data(frames_api, MSP, tilt_deg_min=10, tilt_deg_max=15)
    assert df["tilt"].to_list() == pytest.approx([math.radians(12.0)])


def test_projection_by_frame(frames_api):
    swi = _data(frames_api, MSH, frame="swi")
    assert "Bx_swi" in swi.columns and "Bx" not in swi.columns and "Bx_imf" not in swi.columns
    gsm = _data(frames_api, MSH, frame="gsm")
    assert "Bx" in gsm.columns and not [c for c in gsm.columns if c.endswith("_swi") or "swi_" in c]
    assert "Np" in swi.columns and "Np" in gsm.columns


def test_explicit_columns_are_honoured(frames_api):
    df = _data(frames_api, MSH, frame="swi", columns=["Time", "Bx_imf"], cone_min=0, cone_max=90)
    assert df.columns == ["Time", "Bx_imf"]


def test_no_frame_returns_every_column(frames_api):
    plain = _data(frames_api, MSH)
    selected = _data(frames_api, MSH, cone_min=0, cone_max=180)
    assert selected.columns == plain.columns


@pytest.mark.parametrize(
    ("region", "params"),
    [(MSH, {"frame": "pgsm", "cone_min": 0, "cone_max": 90}), (MSP, {"frame": "swi"}),
     (MSH, {"tilt_deg_min": 0, "tilt_deg_max": 5}), (MSH, {"cone_min": 0})],
)
def test_data_refuses_bad_frames(frames_api, region, params):
    r = frames_api.get(f"/api/v1/regions/{region}/data", params=params)
    assert r.status_code == 400 and r.json()["detail"]["error"] == "bad_frame"


def test_pgsm_count_still_works(frames_api):
    status, body = _count(frames_api, MSH, frame="pgsm", cone_min=0, cone_max=180)
    assert status == 200 and body["n_rows"] == 4  # two paired rows, each twice


@pytest.mark.parametrize(
    ("path", "params", "message"),
    [
        ("data", {"frame": "pgsm"}, "computed by the client"),
        ("data", {"frame": "pgsm", "cone_min": 0}, "computed by the client"),
        ("count", {"cone_min": 0}, "must be given together"),
        ("count", {"frame": "pgsm", "tilt_deg_min": 0, "tilt_deg_max": 5, "tilt_min": 0.1}, "tilt_min"),
    ],
)
def test_frame_param_messages(frames_api, path, params, message):
    region = MSP if "tilt_deg_min" in params else MSH
    r = frames_api.get(f"/api/v1/regions/{region}/{path}", params=params)
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "bad_frame" and message in r.json()["detail"]["message"]
