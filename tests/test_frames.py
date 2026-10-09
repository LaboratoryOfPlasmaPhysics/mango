"""Frame parameters, angle expressions, selections and column sets (no I/O)."""

import math

import numpy as np
import polars as pl
import pytest

from space_mango.errors import QueryError
from space_mango.frames import (
    FrameSpec,
    frame_columns,
    gsm_clock_deg,
    gsm_cone_deg,
    implied_flags,
    make_spec,
    required_columns,
    selection,
    spec_from_params,
    swi_cone_deg,
)

MSH, MSP, SW = "magnetosheath", "magnetosphere", "solar_wind"


def imf(rows):
    """rows: (Bx_imf, By_imf, Bz_imf, Vx_sw, Vy_sw, Vz_sw[, tilt_rad])."""
    cols = ["Bx_imf", "By_imf", "Bz_imf", "Vx_sw", "Vy_sw", "Vz_sw", "tilt"]
    return pl.DataFrame([dict(zip(cols, (*r, 0.0)[:7], strict=False)) for r in rows])


# ---- make_spec -------------------------------------------------------------------------

def test_no_frame_no_selection_is_none():
    assert make_spec(MSH, None) is None


@pytest.mark.parametrize(
    ("region", "kwargs", "expected"),
    [
        (MSH, {"cone": [20, 40]}, FrameSpec(MSH, None, cone=(20.0, 40.0))),
        (MSH, {"clock": [330, 30]}, FrameSpec(MSH, None, clock_range=(330.0, 30.0))),
        (MSP, {"frame": "gsm", "tilt": [10, 15]}, FrameSpec(MSP, "gsm", tilt=(10.0, 15.0))),
        (SW, {"frame": "gsm"}, FrameSpec(SW, "gsm")),
        (MSH, {"frame": "swi"}, FrameSpec(MSH, "swi")),
        (MSH, {"frame": "swi", "cone": [0, 30]}, FrameSpec(MSH, "swi", cone=(0.0, 30.0))),
        (MSH, {"frame": "pgsm", "cone": [80, 100], "clock": 180}, FrameSpec(MSH, "pgsm", cone=(80.0, 100.0), clock=180.0)),
        (MSP, {"frame": "pgsm", "tilt": [-5, 5]}, FrameSpec(MSP, "pgsm", tilt=(-5.0, 5.0))),
    ],
)
def test_valid_specs(region, kwargs, expected):
    assert make_spec(region, **{"frame": None, **kwargs}) == expected  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("region", "kwargs", "message"),
    [
        (MSH, {"frame": "sm"}, "not a frame"),
        (MSH, {"tilt": [0, 5]}, "tilt does not apply to the magnetosheath"),
        (SW, {"cone": [0, 90]}, "no IMF columns"),
        (SW, {"frame": "swi"}, "only defined for the magnetosheath"),
        (MSP, {"frame": "swi"}, "only defined for the magnetosheath"),
        (MSH, {"frame": "swi", "clock": [0, 90]}, "clock makes no sense in SWI"),
        (MSH, {"clock": 30}, "clock in GSM is a range"),
        (MSH, {"frame": "pgsm", "cone": [0, 90], "clock": [0, 90]}, "clock in PGSM is one target value"),
        (MSH, {"frame": "pgsm", "cone": [0, 90]}, "needs cone"),
        (MSP, {"frame": "pgsm", "cone": [0, 9], "tilt": [0, 5]}, "do not apply to the magnetosphere"),
        (MSP, {"frame": "pgsm"}, "needs tilt"),
        (SW, {"frame": "pgsm"}, "not defined for region 'solar_wind'"),
        (MSH, {"cone": [100, 80]}, "min <= max"),
        (MSH, {"cone": [0, 181]}, "must lie in"),
        (MSH, {"clock": [0, 400]}, r"\[-360, 360\]"),
        (MSH, {"clock": "0,90"}, r"\[min, max\]"),
        (MSP, {"tilt": [-40, 0]}, "must lie in"),
    ],
)
def test_bad_specs(region, kwargs, message):
    with pytest.raises(QueryError, match=message) as info:
        make_spec(region, **{"frame": None, **kwargs})  # pyright: ignore[reportArgumentType]
    assert info.value.code == "bad_frame"


def test_count_may_omit_pgsm_clock():
    assert make_spec(MSH, "pgsm", cone=[0, 90], require_clock=False) == FrameSpec(
        MSH, "pgsm", cone=(0.0, 90.0))


# ---- angles ----------------------------------------------------------------------------

def test_gsm_cone_and_clock():
    df = imf([(1.0, 0.0, 0.0, -400, 0, 0), (0.0, 1.0, 0.0, -400, 0, 0), (0.0, 0.0, -1.0, -400, 0, 0),
              (-1.0, -1.0, 0.0, -400, 0, 0), (0.0, 0.0, 0.0, -400, 0, 0), (2.0, 0.0, 0.0, -400, 0, 0)])
    out = df.select(cone=gsm_cone_deg(), clock=gsm_clock_deg())
    assert out["cone"].to_list()[:4] == pytest.approx([0.0, 90.0, 90.0, 135.0])
    assert math.isnan(out["cone"][4])
    clocks = out["clock"].to_list()
    assert clocks[1:4] == pytest.approx([90.0, 180.0, 270.0])
    assert clocks[4] is None and clocks[5] is None  # By = Bz = 0: undefined


def test_swi_cone_matches_numpy_oracle():
    rng = np.random.default_rng(1)
    b = rng.normal(size=(50, 3)) * 5
    v = np.column_stack([-rng.uniform(300, 700, 50), rng.normal(0, 40, 50), rng.normal(0, 40, 50)])
    df = imf([(*bi, *vi) for bi, vi in zip(b, v, strict=True)])
    s = np.sign(b[:, 0])
    x = -v / np.linalg.norm(v, axis=1)[:, None]
    expected = np.degrees(np.arccos(np.clip(s * (b * x).sum(1) / np.linalg.norm(b, axis=1), -1, 1)))
    assert df.select(swi_cone_deg()).to_series().to_numpy() == pytest.approx(expected)


def test_swi_cone_can_exceed_90_with_aberration():
    # IMF nearly perpendicular to X_GSM with Bx > 0, flow strongly deflected: s*B.X_swi < 0
    df = imf([(0.05, 1.0, 0.0, -400.0, 100.0, 0.0)])
    assert df.select(swi_cone_deg()).item() > 90.0


# ---- selections ------------------------------------------------------------------------

CLOCKS = [0.0, 20.0, 90.0, 180.0, 300.0, 340.0]


def clock_frame():
    rows = [(0.0, math.sin(math.radians(c)), math.cos(math.radians(c)), -400, 0, 0) for c in CLOCKS]
    return imf(rows + [(1.0, 0.0, 0.0, -400, 0, 0)])  # last row: clock undefined


@pytest.mark.parametrize(
    ("clock", "kept"),
    [([10, 100], [20.0, 90.0]), ([330, 30], [0.0, 20.0, 340.0]), ([-30, 30], [0.0, 20.0, 340.0]),
     ([0, 360], CLOCKS), ([179.9, 180.1], [180.0])],
)
def test_clock_range(clock, kept):
    spec = make_spec(MSH, None, clock=clock)
    assert spec is not None
    sel = selection(spec)
    assert sel is not None
    got = clock_frame().filter(sel).select(gsm_clock_deg()).to_series().to_list()
    assert sorted(got) == pytest.approx(sorted(kept))


def test_cone_selection_gsm_vs_swi_axis():
    # Bx>0 IMF at 30 deg from X_GSM; flow deflected by 10 deg so the SWI cone is 20 deg
    by = math.tan(math.radians(30.0))
    df = imf([(1.0, by, 0.0, -400.0, -400.0 * math.tan(math.radians(10.0)), 0.0)])
    gsm = selection(FrameSpec(MSH, None, cone=(25.0, 35.0)))
    swi = selection(FrameSpec(MSH, "swi", cone=(25.0, 35.0)))
    assert gsm is not None and swi is not None
    assert df.filter(gsm).height == 1 and df.filter(swi).height == 0


def test_tilt_selection_in_degrees():
    df = imf([(1, 0, 0, -400, 0, 0, math.radians(12.0)), (1, 0, 0, -400, 0, 0, math.radians(-12.0))])
    sel = selection(FrameSpec(MSP, "gsm", tilt=(10.0, 15.0)))
    assert sel is not None and df.filter(sel)["tilt"].to_list() == pytest.approx([math.radians(12.0)])


def test_no_selection_is_none():
    assert selection(FrameSpec(MSH, "swi")) is None


@pytest.mark.parametrize(
    ("spec", "flags"),
    [
        (FrameSpec(MSH, "gsm"), (False, False)),
        (FrameSpec(MSP, "gsm", tilt=(0.0, 5.0)), (False, False)),
        (FrameSpec(MSH, None, cone=(0.0, 30.0)), (True, False)),
        (FrameSpec(MSH, None, clock_range=(0.0, 30.0)), (True, False)),
        (FrameSpec(MSH, "swi"), (True, True)),
        (FrameSpec(MSH, "pgsm", cone=(0.0, 30.0), clock=0.0), (True, True)),
        (FrameSpec(MSP, "pgsm", tilt=(0.0, 5.0)), (False, True)),
    ],
)
def test_implied_flags(spec, flags):
    assert implied_flags(spec) == flags


def test_required_columns():
    assert required_columns(FrameSpec(MSH, None, cone=(0.0, 30.0))) == ["Bx_imf", "By_imf", "Bz_imf"]
    assert required_columns(FrameSpec(MSH, "swi", cone=(0.0, 30.0))) == [
        "Bx_imf", "By_imf", "Bz_imf", "Vx_sw", "Vy_sw", "Vz_sw"]
    assert required_columns(FrameSpec(MSP, "gsm", tilt=(0.0, 5.0))) == ["tilt"]
    assert required_columns(FrameSpec(MSH, "swi")) == []
    assert "Vx_sw" in required_columns(FrameSpec(MSH, "pgsm", cone=(0.0, 30.0), clock=0.0))


# ---- column sets -----------------------------------------------------------------------

MSH_SERVED = ["Time", "Bx", "By", "Bz", "Np", "Vx", "Vy", "Vz", "Tp", "X_gsm", "Y_gsm", "Z_gsm",
              "SW_pairing", "Bx_imf", "By_imf", "Bz_imf", "Np_sw", "Vx_sw", "Vy_sw", "Vz_sw",
              "Tp_sw", "Pd_sw", "Beta_sw", "Ma_sw", "R_mp", "R_bs", "R_norm", "Norma_pos",
              "X_gsm_norm", "Y_gsm_norm", "Z_gsm_norm", "Bx_swi", "By_swi", "Bz_swi",
              "Vx_swi", "Vy_swi", "Vz_swi", "X_swi_norm", "Y_swi_norm", "Z_swi_norm", "SC"]
SCALARS = ["Time", "Np", "Tp", "SW_pairing", "Np_sw", "Tp_sw", "Pd_sw", "Beta_sw", "Ma_sw",
           "R_mp", "R_bs", "R_norm", "Norma_pos", "SC"]


def test_frame_columns():
    assert frame_columns(MSH, None, MSH_SERVED) == MSH_SERVED
    swi = frame_columns(MSH, "swi", MSH_SERVED)
    assert [c for c in swi if c not in SCALARS] == [
        "Bx_swi", "By_swi", "Bz_swi", "Vx_swi", "Vy_swi", "Vz_swi",
        "X_swi_norm", "Y_swi_norm", "Z_swi_norm"]
    gsm = frame_columns(MSH, "gsm", MSH_SERVED)
    assert not [c for c in gsm if "swi" in c] and "Bx_imf" in gsm and "X_gsm_norm" in gsm
    assert [c for c in swi if c in SCALARS] == SCALARS  # served order kept
    pgsm = frame_columns(MSH, "pgsm", MSH_SERVED)
    assert pgsm[-11:] == ["X_pgsm_norm", "Y_pgsm_norm", "Z_pgsm_norm", "Bx_pgsm", "By_pgsm",
                          "Bz_pgsm", "Vx_pgsm", "Vy_pgsm", "Vz_pgsm", "mirrored", "bx_sign"]


# ---- server parameters -----------------------------------------------------------------

def test_spec_from_params():
    assert spec_from_params(MSH, {"cone_min": "20", "cone_max": "40"}, for_count=False) == FrameSpec(
        MSH, None, cone=(20.0, 40.0))
    assert spec_from_params(MSH, {"frame": "swi"}, for_count=False) == FrameSpec(MSH, "swi")
    assert spec_from_params(MSH, {}, for_count=False) is None
    assert spec_from_params(MSH, {"frame": "pgsm", "cone_min": "0", "cone_max": "90"}, for_count=True) == FrameSpec(
        MSH, "pgsm", cone=(0.0, 90.0))


@pytest.mark.parametrize(
    ("raw", "for_count", "message"),
    [
        ({"frame": "pgsm", "cone_min": "0", "cone_max": "90"}, False, "computed by the client"),
        ({"cone_min": "0"}, False, r"\[min, max\]"),
        ({"tilt_deg_min": "0", "tilt_deg_max": "5", "tilt_min": "0.1"}, False, "tilt_min"),
    ],
)
def test_spec_from_params_errors(raw, for_count, message):
    with pytest.raises(QueryError, match=message):
        spec_from_params(MSP if "tilt_deg_min" in raw else MSH, raw, for_count=for_count)
