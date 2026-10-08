import math

import numpy as np
import polars as pl
import pytest

from space_mango.errors import MangoError, PgsmError, QueryError, error_from_query
from space_mango.pgsm import (
    COLUMN_INFO,
    OUTPUT_COLUMNS,
    PgsmSpec,
    candidates,
    folded_cone_deg,
    make_spec,
    to_pgsm,
)


def test_no_frame_no_spec():
    assert make_spec("magnetosheath", None) is None


def test_magnetosheath_spec():
    assert make_spec("magnetosheath", "pgsm", cone=[80, 100], clock=180) == PgsmSpec(
        "magnetosheath", cone=(80.0, 100.0), clock=180.0
    )


def test_magnetosphere_spec():
    assert make_spec("magnetosphere", "pgsm", tilt=(-5, 5)) == PgsmSpec(
        "magnetosphere", tilt=(-5.0, 5.0)
    )


def test_count_may_omit_clock():
    spec = make_spec("magnetosheath", "pgsm", cone=[0, 90], require_clock=False)
    assert spec == PgsmSpec("magnetosheath", cone=(0.0, 90.0))


@pytest.mark.parametrize(
    ("region", "kwargs", "message"),
    [
        ("magnetosheath", {"cone": [0, 90]}, "need frame='pgsm'"),
        ("magnetosphere", {"frame": "gsm", "tilt": [0, 5]}, "not supported"),
        ("solar_wind", {"frame": "pgsm"}, "not defined for region 'solar_wind'"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0, 90]}, "needs cone"),
        ("magnetosheath", {"frame": "pgsm", "clock": 0}, "needs cone"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0, 90], "clock": 0, "tilt": [0, 5]},
         "tilt does not apply"),
        ("magnetosphere", {"frame": "pgsm", "tilt": [0, 5], "cone": [0, 90]},
         "do not apply to the magnetosphere"),
        ("magnetosphere", {"frame": "pgsm"}, "needs tilt"),
        ("magnetosheath", {"frame": "pgsm", "cone": [100, 80], "clock": 0}, "min <= max"),
        ("magnetosheath", {"frame": "pgsm", "cone": [-1, 80], "clock": 0}, "0 <= min"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0, 181], "clock": 0}, "<= 180"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0], "clock": 0}, "[min, max]"),
        ("magnetosheath", {"frame": "pgsm", "cone": "0,90", "clock": 0}, "[min, max]"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0, 90], "clock": float("nan")}, "clock"),
        ("magnetosheath", {"frame": "pgsm", "cone": [0, 90], "clock": 400}, "clock"),
        ("magnetosphere", {"frame": "pgsm", "tilt": [-40, 0]}, "-35 <= min"),
    ],
)
def test_bad_parameters(region, kwargs, message):
    with pytest.raises(QueryError, match=message) as info:
        make_spec(region, **{"frame": None, **kwargs})  # pyright: ignore[reportArgumentType]
    assert info.value.code == "bad_pgsm"


def test_bad_pgsm_maps_to_pgsm_error():
    err = error_from_query(QueryError("bad_pgsm", "x"))
    assert isinstance(err, PgsmError) and isinstance(err, MangoError)


MSH = "magnetosheath"


def swi_basis(v_sw, b_imf):
    """Test oracle, as spok.coordinates.swi_base: X = -V/|V|, Z = X x (s B)/|.|, Y = Z x X."""
    s = float(np.sign(b_imf[0]) or np.sign(b_imf[1]) or np.sign(b_imf[2]))
    x = -np.asarray(v_sw, float)
    x /= np.linalg.norm(x)
    z = np.cross(x, s * np.asarray(b_imf, float))
    z /= np.linalg.norm(z)
    return s, np.array([x, np.cross(z, x), z])


def msh_frame(b_imf_list, *, b=None, v=(-200.0, 50.0, 20.0), r=(10.0, 3.0, 4.0),
              v_sw=(-400.0, 0.0, 0.0)):
    """Magnetosheath rows whose SWI columns are built as the MANGO pipeline does.
    b=None puts the local field equal to the IMF (B_swi is then the IMF in SWI)."""
    rows = []
    for b_imf in b_imf_list:
        s, rot = swi_basis(v_sw, b_imf)
        bs = s * rot @ np.asarray(b_imf if b is None else b, float)
        vs = rot @ (np.asarray(v, float) - np.array([0.0, 29.8, 0.0]))
        rs = rot @ np.asarray(r, float)
        rows.append({
            "Bx_imf": b_imf[0], "By_imf": b_imf[1], "Bz_imf": b_imf[2],
            "Bx_swi": bs[0], "By_swi": bs[1], "Bz_swi": bs[2],
            "Vx_swi": vs[0], "Vy_swi": vs[1], "Vz_swi": vs[2],
            "X_swi_norm": rs[0], "Y_swi_norm": rs[1], "Z_swi_norm": rs[2],
            "Np": 10.0,
        })
    return pl.DataFrame(rows)


def clock_cone(row, prefix="B", suffix="_pgsm"):
    bx, by, bz = (row[f"{prefix}{c}{suffix}"] for c in "xyz")
    clock = math.degrees(math.atan2(by, bz)) % 360.0
    cone = math.degrees(math.acos(bx / math.sqrt(bx * bx + by * by + bz * bz)))
    return clock, cone


IMF = (-2.0, 3.0, -4.0)  # Bx < 0: folded cone f = acos(2/sqrt(29)) = 68.2 deg
F = math.degrees(math.acos(2 / math.sqrt(29)))


def spec(cone, clock):
    return PgsmSpec(MSH, cone=cone, clock=clock)


def test_folded_cone():
    df = pl.DataFrame(
        {"Bx_imf": [-2.0, 1.0, 0.0], "By_imf": [3.0, 0.0, 0.0], "Bz_imf": [-4.0, 0.0, -3.0]}
    )
    got = df.select(folded_cone_deg()).to_series().to_list()
    assert got == pytest.approx([F, 0.0, 90.0])


@pytest.mark.parametrize("clock", [0.0, 45.0, 150.0, 270.0])
def test_imf_lands_at_target_clock_and_cone(clock):
    out = to_pgsm(msh_frame([IMF]), spec((0.0, 180.0), clock))
    assert out.height == 2
    plus = out.filter(pl.col("bx_sign") == 1).row(0, named=True)
    minus = out.filter(pl.col("bx_sign") == -1).row(0, named=True)
    for row, cone in [(plus, F), (minus, 180.0 - F)]:
        got_clock, got_cone = clock_cone(row)
        # compare angles modulo 360 (0 and 359.9999999 are the same clock)
        assert math.cos(math.radians(got_clock - clock)) == pytest.approx(1.0, abs=1e-12)
        assert got_cone == pytest.approx(cone)


def test_original_bx_sign_is_not_mirrored():
    out = to_pgsm(msh_frame([IMF]), spec((0.0, 180.0), 90.0))
    flags = dict(zip(out["bx_sign"].to_list(), out["mirrored"].to_list(), strict=True))
    assert flags == {1: True, -1: False}  # IMF Bx < 0: the bx_sign = -1 row is the measurement


def test_clock_90_bx_positive_reproduces_swi():
    df = msh_frame([(2.0, 3.0, -4.0)], b=(1.0, -2.0, 5.0))
    out = to_pgsm(df, spec((0.0, 89.0), 90.0))
    assert out.height == 1
    for pgsm, swi in [("X_pgsm_norm", "X_swi_norm"), ("Y_pgsm_norm", "Y_swi_norm"),
                      ("Z_pgsm_norm", "Z_swi_norm"), ("Bx_pgsm", "Bx_swi"),
                      ("By_pgsm", "By_swi"), ("Bz_pgsm", "Bz_swi"), ("Vx_pgsm", "Vx_swi"),
                      ("Vy_pgsm", "Vy_swi"), ("Vz_pgsm", "Vz_swi")]:
        assert out[pgsm][0] == pytest.approx(df[swi][0], abs=1e-12)


def test_bx_negative_is_the_y_mirror_at_clock_90():
    df = msh_frame([IMF], b=(1.0, -2.0, 5.0))
    out = to_pgsm(df, spec((0.0, 180.0), 90.0))
    p = out.filter(pl.col("bx_sign") == 1).row(0, named=True)
    m = out.filter(pl.col("bx_sign") == -1).row(0, named=True)
    assert (m["X_pgsm_norm"], m["Y_pgsm_norm"], m["Z_pgsm_norm"]) == pytest.approx(
        (p["X_pgsm_norm"], -p["Y_pgsm_norm"], p["Z_pgsm_norm"]))
    assert (m["Bx_pgsm"], m["By_pgsm"], m["Bz_pgsm"]) == pytest.approx(
        (-p["Bx_pgsm"], p["By_pgsm"], -p["Bz_pgsm"]))
    assert (m["Vx_pgsm"], m["Vy_pgsm"], m["Vz_pgsm"]) == pytest.approx(
        (p["Vx_pgsm"], -p["Vy_pgsm"], p["Vz_pgsm"]))


def test_rotation_preserves_norms():
    df = msh_frame([IMF], b=(1.0, -2.0, 5.0))
    out = to_pgsm(df, spec((0.0, 180.0), 200.0))
    r_in = sum(df[f"{c}_swi_norm"][0] ** 2 for c in "XYZ")
    b_in = sum(df[f"B{c}_swi"][0] ** 2 for c in "xyz")
    v_in = sum(df[f"V{c}_swi"][0] ** 2 for c in "xyz")
    for row in out.iter_rows(named=True):
        assert sum(row[f"{c}_pgsm_norm"] ** 2 for c in "XYZ") == pytest.approx(r_in)
        assert sum(row[f"B{c}_pgsm"] ** 2 for c in "xyz") == pytest.approx(b_in)
        assert sum(row[f"V{c}_pgsm"] ** 2 for c in "xyz") == pytest.approx(v_in)


@pytest.mark.parametrize(
    ("cone", "signs"),
    [((60.0, 80.0), [1]), ((100.0, 120.0), [-1]), ((0.0, 180.0), [1, -1]), ((85.0, 95.0), [])],
)
def test_cone_selection(cone, signs):
    out = to_pgsm(msh_frame([IMF]), spec(cone, 0.0))
    assert sorted(out["bx_sign"].to_list(), reverse=True) == signs


def test_cone_90_selects_perpendicular_imf_twice():
    out = to_pgsm(msh_frame([(0.0, 0.0, -3.0)]), spec((90.0, 90.0), 0.0))
    assert sorted(out["bx_sign"].to_list()) == [-1, 1]


def test_zero_or_null_imf_is_never_selected():
    df = pl.concat([
        msh_frame([IMF]),
        msh_frame([IMF]).with_columns(Bx_imf=pl.lit(0.0), By_imf=pl.lit(0.0), Bz_imf=pl.lit(0.0)),
        msh_frame([IMF]).with_columns(By_imf=pl.lit(None, dtype=pl.Float64)),
    ])
    out = to_pgsm(df, spec((0.0, 180.0), 0.0))
    assert out.height == 2  # only the first row, twice
    assert not out["Bx_pgsm"].is_nan().any()


@pytest.mark.parametrize(("a", "b"), [(0.0, 360.0), (-90.0, 270.0)])
def test_clock_is_periodic(a, b):
    df = msh_frame([IMF], b=(1.0, -2.0, 5.0))
    out_a = to_pgsm(df, spec((0.0, 180.0), a)).select(OUTPUT_COLUMNS[MSH])
    out_b = to_pgsm(df, spec((0.0, 180.0), b)).select(OUTPUT_COLUMNS[MSH])
    for c in OUTPUT_COLUMNS[MSH]:
        if out_a[c].dtype == pl.Float64:
            assert out_a[c].to_list() == pytest.approx(out_b[c].to_list(), abs=1e-9)


def test_empty_selection_keeps_output_columns():
    out = to_pgsm(msh_frame([IMF]), spec((85.0, 95.0), 0.0))
    assert out.height == 0
    assert set(OUTPUT_COLUMNS[MSH]) <= set(out.columns)


def test_original_columns_are_kept_as_measured():
    df = msh_frame([IMF])
    out = to_pgsm(df, spec((0.0, 180.0), 30.0))
    assert out["Bx_swi"].to_list() == [df["Bx_swi"][0]] * 2
    assert out["Np"].to_list() == [10.0, 10.0]


def test_missing_input_column():
    with pytest.raises(QueryError, match="Bx_swi"):
        to_pgsm(msh_frame([IMF]).drop("Bx_swi"), spec((0.0, 90.0), 0.0))


def test_candidates_shape():
    preds = candidates(spec((0.0, 90.0), 0.0))
    assert [s for _, s in preds] == [1, -1]


MSP = "magnetosphere"


def dipole(r, psi):
    """Earth dipole (unit moment) at GSM position r for tilt psi (rad): m = -(sin psi, 0, cos psi)."""
    m = -np.array([math.sin(psi), 0.0, math.cos(psi)])
    rn = np.linalg.norm(r)
    u = np.asarray(r, float) / rn
    return (3.0 * np.dot(m, u) * u - m) / rn**3


def msp_frame(points, psi):
    rows = []
    for r in points:
        b = dipole(r, psi)
        rows.append({
            "tilt": psi, "Bx": b[0], "By": b[1], "Bz": b[2],
            "Vx": 10.0, "Vy": 20.0, "Vz": 30.0,
            "X_gsm_norm": r[0], "Y_gsm_norm": r[1], "Z_gsm_norm": r[2],
        })
    return pl.DataFrame(rows)


POINTS = [(5.0, 2.0, 3.0), (-3.0, -4.0, 6.0), (8.0, 0.5, -2.0)]


def test_tilt_mirror_maps_dipole_at_psi_to_dipole_at_minus_psi():
    psi = math.radians(24.0)
    out = to_pgsm(msp_frame(POINTS, psi), PgsmSpec(MSP, tilt=(-25.0, -23.0)))
    assert out.height == len(POINTS) and out["mirrored"].all()
    for row in out.iter_rows(named=True):
        r = (row["X_pgsm_norm"], row["Y_pgsm_norm"], row["Z_pgsm_norm"])
        b = (row["Bx_pgsm"], row["By_pgsm"], row["Bz_pgsm"])
        assert b == pytest.approx(tuple(dipole(r, -psi)))
        assert row["tilt_pgsm"] == pytest.approx(-24.0)


def test_mirror_vectors_and_originals():
    df = msp_frame([POINTS[0]], math.radians(2.0))
    out = to_pgsm(df, PgsmSpec(MSP, tilt=(-5.0, 5.0)))
    orig = out.filter(~pl.col("mirrored")).row(0, named=True)
    mirr = out.filter(pl.col("mirrored")).row(0, named=True)
    assert (orig["X_pgsm_norm"], orig["Y_pgsm_norm"], orig["Z_pgsm_norm"]) == POINTS[0]
    assert (mirr["X_pgsm_norm"], mirr["Y_pgsm_norm"], mirr["Z_pgsm_norm"]) == (5.0, -2.0, -3.0)
    assert (mirr["Bx_pgsm"], mirr["By_pgsm"], mirr["Bz_pgsm"]) == pytest.approx(
        (-orig["Bx_pgsm"], orig["By_pgsm"], orig["Bz_pgsm"]))
    assert (mirr["Vx_pgsm"], mirr["Vy_pgsm"], mirr["Vz_pgsm"]) == (10.0, -20.0, -30.0)
    assert (orig["tilt_pgsm"], mirr["tilt_pgsm"]) == pytest.approx((2.0, -2.0))
    assert mirr["tilt"] == pytest.approx(math.radians(2.0))  # original column as measured


@pytest.mark.parametrize(
    ("psi_deg", "tilt", "mirrored"),
    [(12.0, (10.0, 15.0), [False]), (-12.0, (10.0, 15.0), [True]), (2.0, (-5.0, 5.0), [False, True]),
     (20.0, (10.0, 15.0), [])],
)
def test_tilt_selection(psi_deg, tilt, mirrored):
    out = to_pgsm(msp_frame([POINTS[0]], math.radians(psi_deg)), PgsmSpec(MSP, tilt=tilt))
    assert sorted(out["mirrored"].to_list()) == mirrored


def test_magnetosphere_empty_selection_keeps_output_columns():
    out = to_pgsm(msp_frame([POINTS[0]], 0.5), PgsmSpec(MSP, tilt=(0.0, 1.0)))
    assert out.height == 0 and set(OUTPUT_COLUMNS[MSP]) <= set(out.columns)


def test_column_info_covers_every_output_column():
    for region in (MSH, MSP):
        for c in OUTPUT_COLUMNS[region]:
            assert set(COLUMN_INFO[c]) == {"unit", "frame", "description"}
