import pytest

from space_mango.client import MangoFilterError, _validate_filters

MAGNETOSHEATH_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw",
    "x_gsm", "y_gsm", "z_gsm", "d_msh", "np", "tp", "bz",
}

MAGNETOSPHERE_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
    "x_gsm", "y_gsm", "z_gsm", "d_msp", "np", "tp", "bz",
}


def test_validate_filters_valid():
    _validate_filters(
        {"bz_imf_max": -2.0, "pd_sw_min": 3.0},
        "magnetosheath",
        MAGNETOSHEATH_FILTERS,
        {"magnetosphere": MAGNETOSPHERE_FILTERS},
    )


def test_validate_filters_bad_suffix():
    with pytest.raises(MangoFilterError, match="must end with '_min' or '_max'"):
        _validate_filters(
            {"bz_imf": -2.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_unknown_filter():
    with pytest.raises(MangoFilterError, match="not a valid filter"):
        _validate_filters(
            {"fake_min": 1.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_wrong_region_with_hint():
    with pytest.raises(MangoFilterError, match="available for region 'magnetosphere'"):
        _validate_filters(
            {"d_msp_min": 0.5},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {"magnetosphere": MAGNETOSPHERE_FILTERS},
        )


def test_validate_filters_non_numeric():
    with pytest.raises(MangoFilterError, match="must be numeric"):
        _validate_filters(
            {"bz_imf_max": "not_a_number"},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_client_get_data_all(client):
    df = client.get_data("magnetosheath", limit=10)
    assert len(df) == 3
    assert "SC" in df.columns
    assert "Bz_imf" in df.columns


def test_client_get_data_spacecraft_filter(client):
    df = client.get_data("magnetosheath", spacecraft=["THA"], limit=10)
    assert len(df) == 1
    assert df["SC"][0] == "THA"


def test_client_get_data_range_filter(client):
    df = client.get_data("magnetosheath", bz_imf_max=-1.0, limit=10)
    assert set(df["SC"].to_list()) == {"THA", "C1"}


def test_client_regions(client):
    assert "magnetosheath" in client.regions()
