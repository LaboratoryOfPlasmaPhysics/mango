"""Client frames: local selection and projection, exact counts, errors before download."""

import math
from datetime import datetime

import httpx
import pytest

from space_mango._regions_generated import MagnetosheathAPI, MagnetosphereAPI, SolarWindAPI
from space_mango.client import MangoClient
from space_mango.errors import FrameError

MSH, MSP, SW = "magnetosheath", "magnetosphere", "solar_wind"


@pytest.fixture
def fdir(make_row, make_dataset):
    t = datetime(2016, 1, 1)

    def at(s):
        return t.replace(second=s)

    return make_dataset({
        MSH: [
            make_row(MSH, "THA", at(0), Bx_imf=1.0, By_imf=1.0, Bz_imf=0.0,
                     Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0),            # clock 90, cone 45
            make_row(MSH, "THA", at(5), Bx_imf=0.0, By_imf=0.0, Bz_imf=3.0,
                     Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0),            # clock 0, cone 90
            make_row(MSH, "THA", at(10), Bx_imf=0.0, By_imf=-0.5, Bz_imf=3.0,
                     SW_pairing=False),                              # unpaired
        ],
        MSP: [make_row(MSP, "THA", at(0), tilt=math.radians(12.0)),
              make_row(MSP, "THA", at(5), tilt=math.radians(-12.0))],
        SW: [make_row(SW, "THA", t)],
    })


@pytest.fixture
def fc(fdir, make_client) -> MangoClient:
    return make_client(fdir)


def test_no_frame_no_selection_is_unchanged(fc):
    a = fc.get_data(MSH)
    assert len(a) == 3 and "Bx_swi" in a.columns and "Bx" in a.columns


@pytest.mark.parametrize(
    ("kwargs", "n"),
    [({"cone": [40, 50]}, 1), ({"clock": [330, 30]}, 1), ({"clock": [0, 360]}, 2),
     ({"frame": "swi", "cone": [40, 50]}, 1), ({"frame": "swi"}, 2), ({"frame": "gsm"}, 3)],
)
def test_selection_and_count(fc, kwargs, n):
    assert len(fc.get_data(MSH, **kwargs)) == n
    assert fc.count(MSH, **kwargs)["n_rows"] == n


def test_selection_server_path_matches_cache_path(fc):
    a = fc.get_data(MSH, frame="swi", cone=[0, 90])
    b = fc.get_data(MSH, frame="swi", cone=[0, 90], cache=False)
    assert a.to_polars().sort("Time").equals(b.to_polars().sort("Time"))


def test_columns_by_frame(fc):
    swi = fc.get_data(MSH, frame="swi").columns
    assert "Bx_swi" in swi and "Bx" not in swi and "Bx_imf" not in swi and "Np" in swi
    gsm = fc.get_data(MSH, frame="gsm").columns
    assert "Bx" in gsm and "Bx_imf" in gsm and not [c for c in gsm if "swi" in c]
    sel = fc.get_data(MSH, frame="swi", cone=[0, 90]).columns
    assert sel == swi  # the selection inputs (Bx_imf, V_sw) are not returned


def test_requested_columns_in_frame(fc):
    r = fc.get_data(MSH, frame="swi", cone=[0, 90], columns=["Time", "Bx_swi"])
    assert r.columns == ["Time", "Bx_swi"]


def test_column_from_another_frame_is_an_error(fc):
    with pytest.raises(FrameError, match="gsm"):
        fc.get_data(MSH, frame="swi", columns=["Time", "Bx"])


def test_tilt_selection(fc):
    r = fc.get_data(MSP, frame="gsm", tilt=[10, 15])
    assert r["tilt"].to_list() == pytest.approx([math.radians(12.0)])
    with pytest.raises(FrameError, match="tilt_min"):
        fc.get_data(MSP, tilt=[10, 15], tilt_min=0.1)


def test_query_records_frame_and_selection(fc):
    r = fc.get_data(MSH, frame="swi", cone=[20, 50], columns=["Time"])
    assert r.query["frame"] == "swi" and r.query["cone"] == (20.0, 50.0)
    assert r.query["columns"] == ["Time"]


def test_get_data_works_without_server_feature(fc, monkeypatch):
    info = dict(fc.dataset_info())
    info.pop("features", None)
    monkeypatch.setattr(fc, "dataset_info", lambda: info)
    assert len(fc.get_data(MSH, cone=[40, 50], cache=False)) == 1
    with pytest.raises(Exception, match="0.3"):
        fc.count(MSH, cone=[40, 50])


@pytest.mark.parametrize(
    ("region", "kwargs"),
    [(SW, {"cone": [0, 90]}), (MSP, {"frame": "swi"}), (MSH, {"frame": "swi", "clock": [0, 9]}),
     (MSH, {"clock": 30}), (MSH, {"frame": "pgsm", "cone": [0, 90], "clock": [0, 9]}),
     (MSH, {"tilt": [0, 5]})],
)
def test_errors_before_any_data_request(fdir, make_api, tmp_path, region, kwargs):
    calls: list[str] = []
    inner = make_api(fdir)._transport

    class Counting(httpx.BaseTransport):
        def handle_request(self, request):
            calls.append(request.url.path)
            return inner.handle_request(request)

    c = MangoClient("http://testserver", transport=Counting(), cache_dir=tmp_path / "c")
    with pytest.raises(FrameError):
        c.get_data(region, **kwargs)
    assert not [p for p in calls if p.endswith("/data")]


def test_region_objects(fc):
    assert len(MagnetosheathAPI(lambda: fc).get_data(frame="swi", cone=[40, 50])) == 1
    assert len(MagnetosphereAPI(lambda: fc).get_data(tilt=[10, 15])) == 1
    assert "Bx" in SolarWindAPI(lambda: fc).get_data(frame="gsm").columns
