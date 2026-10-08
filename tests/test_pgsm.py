import pytest

from space_mango.errors import MangoError, PgsmError, QueryError, error_from_query
from space_mango.pgsm import PgsmSpec, make_spec


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
