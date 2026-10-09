"""Client PGSM: validation before any request, local transform, exact counts."""

import math
from datetime import datetime

import httpx
import pytest

from space_mango._regions_generated import MagnetosheathAPI, MagnetosphereAPI
from space_mango.client import MangoClient
from space_mango.errors import FrameError, ServerError
from space_mango.frames import OUTPUT_COLUMNS

MSH, MSP = "magnetosheath", "magnetosphere"


@pytest.fixture
def pgsm_dir(make_row, make_dataset):
    return make_dataset({
        MSH: [
            make_row(MSH, "THA", datetime(2016, 1, 1, 0, 0, 0), Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0),  # IMF (1,1,1): f = 54.7 deg
            make_row(MSH, "THA", datetime(2016, 1, 1, 0, 0, 5), Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0, Bx_imf=-2.0, By_imf=3.0,
                     Bz_imf=-4.0),  # f = 68.2 deg, original Bx < 0
            make_row(MSH, "THA", datetime(2016, 1, 1, 0, 0, 10), Vx_sw=-400.0, Vy_sw=0.0, Vz_sw=0.0, SW_pairing=False),
        ],
        MSP: [
            make_row(MSP, "THA", datetime(2016, 1, 1, 0, 0, 0), tilt=math.radians(8.0)),
            make_row(MSP, "THA", datetime(2016, 1, 1, 0, 0, 5), tilt=math.radians(-6.0)),
            make_row(MSP, "THA", datetime(2016, 1, 1, 0, 0, 10), tilt=math.radians(7.0),
                     Norma_pos=False),
        ],
        "solar_wind": [make_row("solar_wind", "THA", datetime(2016, 1, 1))],
    })


@pytest.fixture
def pc(pgsm_dir, make_client) -> MangoClient:
    return make_client(pgsm_dir)


def test_magnetosheath_rows_and_columns(pc):
    r = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=180)
    assert len(r) == 4  # two paired, normalized rows, each with bx_sign +1 and -1
    assert set(OUTPUT_COLUMNS[MSH]) <= set(r.columns)
    assert r.query["frame"] == "pgsm" and r.query["clock"] == 180.0
    assert r.query["cone"] == (0.0, 180.0) and "tel-04661957" in str(r.query["reference"])
    assert r.metadata["Bx_pgsm"]["frame"] == "PGSM"


def test_cone_selection_through_the_client(pc):
    r = pc.get_data(MSH, frame="pgsm", cone=[100, 120], clock=0)
    assert r["bx_sign"].to_list() == [-1]  # only f = 68.2 -> 111.8 deg
    assert r["mirrored"].to_list() == [False]  # that row was measured with Bx < 0


def test_requested_columns_plus_output_columns(pc):
    r = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=0, columns=["Time", "Np"])
    assert r.columns == ["Time", "Np", *OUTPUT_COLUMNS[MSH]]


def test_empty_selection(pc):
    r = pc.get_data(MSH, frame="pgsm", cone=[170, 180], clock=0, columns=["Time"])
    assert len(r) == 0 and r.columns == ["Time", *OUTPUT_COLUMNS[MSH]]


def test_magnetosphere_tilt(pc):
    r = pc.get_data(MSP, frame="pgsm", tilt=[5, 10])
    assert sorted(r["mirrored"].to_list()) == [False, True]  # 8 deg as is, -6 deg mirrored
    assert sorted(r["tilt_pgsm"].to_list()) == pytest.approx([6.0, 8.0])


def test_server_path_matches_cache_path(pc):
    a = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=30)
    b = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=30, cache=False)
    assert a.to_polars().sort(["Time", "bx_sign"]).equals(b.to_polars().sort(["Time", "bx_sign"]))


def test_new_clock_reuses_the_cache(pgsm_dir, make_api, tmp_path):
    calls: list[str] = []
    inner = make_api(pgsm_dir)._transport

    class Counting(httpx.BaseTransport):
        def handle_request(self, request):
            calls.append(request.url.path)
            return inner.handle_request(request)

    c = MangoClient("http://testserver", transport=Counting(), cache_dir=tmp_path / "c")
    c.get_data(MSH, frame="pgsm", cone=[0, 180], clock=0)
    before = sum(p.endswith("/data") for p in calls)
    c.get_data(MSH, frame="pgsm", cone=[0, 180], clock=135)
    assert sum(p.endswith("/data") for p in calls) == before


@pytest.mark.parametrize(
    ("region", "kwargs"),
    [
        (MSP, {"frame": "pgsm", "cone": [0, 90], "clock": 0}),
        (MSH, {"frame": "pgsm", "tilt": [0, 5]}),
        ("solar_wind", {"frame": "pgsm"}),
        (MSH, {"cone": [0, 90]}),
        (MSP, {"frame": "pgsm", "tilt": [0, 5], "tilt_min": 0.0}),
    ],
)
def test_errors_before_any_data_request(pgsm_dir, make_api, tmp_path, region, kwargs):
    calls: list[str] = []
    inner = make_api(pgsm_dir)._transport

    class Counting(httpx.BaseTransport):
        def handle_request(self, request):
            calls.append(request.url.path)
            return inner.handle_request(request)

    c = MangoClient("http://testserver", transport=Counting(), cache_dir=tmp_path / "c")
    with pytest.raises(FrameError):
        c.get_data(region, **kwargs)
    assert not [p for p in calls if p.endswith("/data")]


@pytest.mark.parametrize(
    ("region", "kwargs"),
    [(MSH, {"cone": [0, 180]}), (MSH, {"cone": [100, 120]}), (MSH, {"cone": [170, 180]}),
     (MSP, {"tilt": [5, 10]}), (MSP, {"tilt": [-10, 10]})],
)
def test_count_equals_rows(pc, region, kwargs):
    rows = len(pc.get_data(region, frame="pgsm", **({"clock": 0} if region == MSH else {}), **kwargs))
    assert pc.count(region, frame="pgsm", **kwargs)["n_rows"] == rows


def test_count_needs_a_recent_server(pc, monkeypatch):
    info = dict(pc.dataset_info())
    info.pop("features", None)
    monkeypatch.setattr(pc, "dataset_info", lambda: info)
    with pytest.raises(ServerError, match="get_data"):
        pc.count(MSH, frame="pgsm", cone=[0, 90])


def test_region_objects(pc):
    msh = MagnetosheathAPI(lambda: pc)
    assert len(msh.get_data(frame="pgsm", cone=[0, 180], clock=0)) == 4
    assert msh.count(frame="pgsm", cone=[0, 180])["n_rows"] == 4
    msp = MagnetosphereAPI(lambda: pc)
    assert len(msp.get_data(frame="pgsm", tilt=[5, 10])) == 2


def test_query_records_the_request_not_the_fetch(pc):
    r = pc.get_data(MSP, frame="pgsm", tilt=[5, 10], columns=["Time"])
    assert r.query["columns"] == ["Time"]
    assert "tilt_min" not in r.query
    assert r.query["prefilter"]["tilt_max"] == pytest.approx(math.radians(10))
    r = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=0, columns=["Time", "Np"], pd_sw_max=10)
    assert r.query["columns"] == ["Time", "Np"]
    assert r.query["pd_sw_max"] == 10.0
    assert r.query["prefilter"] == {}


def test_requesting_an_output_column_does_not_fail(pc):
    r = pc.get_data(MSH, frame="pgsm", cone=[0, 180], clock=0, columns=["Time", "Bx_pgsm"])
    assert r.columns == ["Time", *OUTPUT_COLUMNS[MSH]]
