from datetime import UTC, date, datetime, timedelta, timezone

import pytest

import space_mango as sm
from space_mango.errors import (
    MangoError,
    MangoFilterError,
    TimeParseError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
    error_from_response,
)
from space_mango.timeparse import to_iso


def test_unknown_region_suggests(client):
    with pytest.raises(UnknownRegionError, match="Did you mean 'magnetosheath'"):
        client.get_data("magnetoshealth", limit=1)


def test_unknown_spacecraft_mms1_suggests_mms(client):
    with pytest.raises(UnknownSpacecraftError, match="Did you mean 'MMS'"):
        client.get_data("magnetosheath", spacecraft=["MMS1"], limit=1)


def test_unknown_column(client):
    with pytest.raises(UnknownColumnError):
        client.get_data("magnetosheath", columns=["Nope"], limit=1)


def test_filter_typo_suggests(client):
    with pytest.raises(MangoFilterError, match="Did you mean 'bz_imf_max'"):
        client.get_data("magnetosheath", bzimf_max=-2, limit=1)


def test_all_errors_are_mango_errors():
    for cls in (UnknownRegionError, UnknownSpacecraftError, UnknownColumnError,
                MangoFilterError, TimeParseError):
        assert issubclass(cls, MangoError) and issubclass(cls, ValueError)
    assert sm.MangoFilterError is MangoFilterError


def test_error_from_response_maps_codes():
    body = {"detail": {"error": "unknown_spacecraft", "message": "nope", "valid": ["MMS"]}}
    err = error_from_response(400, body)
    assert isinstance(err, UnknownSpacecraftError) and "nope" in str(err)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2017-01-12T10:00", "2017-01-12T10:00:00"),
        ("2017-01", "2017-01-01T00:00:00"),
        ("2017", "2017-01-01T00:00:00"),
        (date(2017, 1, 12), "2017-01-12T00:00:00"),
        (datetime(2017, 1, 12, 12, tzinfo=timezone(timedelta(hours=2))), "2017-01-12T10:00:00"),
        (datetime(2017, 1, 12, 10, tzinfo=UTC), "2017-01-12T10:00:00"),
        (None, None),
    ],
)
def test_to_iso(value, expected):
    assert to_iso(value, param="start") == expected


def test_to_iso_pandas_timestamp_with_tz():
    pd = pytest.importorskip("pandas")
    assert to_iso(pd.Timestamp("2017-01-12 12:00", tz="Europe/Paris"), param="start") == "2017-01-12T11:00:00"


def test_to_iso_numpy_datetime64():
    np = pytest.importorskip("numpy")
    assert to_iso(np.datetime64("2017-01-12T10:00:00.5"), param="start") == "2017-01-12T10:00:00.500000"


def test_to_iso_rejects_garbage():
    with pytest.raises(TimeParseError, match="start"):
        to_iso("yesterday", param="start")


def test_start_stop_on_client(client):
    df = client.get_data("magnetosphere", start="2017-01", stop=datetime(2018, 1, 1))
    assert df["SC"].to_list() == ["THA"]


def test_time_min_is_deprecated_but_works(client):
    with pytest.warns(FutureWarning, match="start"):
        df = client.get_data("magnetosphere", time_min="2018-01-01")
    assert df["SC"].to_list() == ["C3"]
