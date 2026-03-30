import pytest

from mango.client import MangoFilterError, _validate_filters


MAGNETOSHEATH_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
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
