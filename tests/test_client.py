import httpx
import pytest

from space_mango.client import MangoClient, _validate_filters
from space_mango.errors import MangoFilterError, ServerError

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


# --- final-review fix wave: client input normalisation -------------------------------------


@pytest.mark.parametrize("cache", [True, False])
def test_plain_string_spacecraft_and_columns(client, cache):
    r = client.get_data("magnetosheath", spacecraft="THA", columns="Np", cache=cache)
    assert r.columns == ["Np"] and r["Np"].to_list() == [10.0]
    assert client.count("magnetosheath", spacecraft="THA", columns="Np")["n_rows"] == 1


@pytest.mark.parametrize("cache", [True, False])
def test_duplicate_names_are_deduplicated(client, cache):
    r = client.get_data(
        "magnetosheath", spacecraft=["THA", "THA"], columns=["Time", "Np", "Time"], cache=cache
    )
    assert r.columns == ["Time", "Np"] and len(r) == 1


def test_numpy_scalar_filter_values_are_accepted(client):
    np = pytest.importorskip("numpy")
    r = client.get_data("magnetosheath", bz_imf_max=np.float32(-2), np_min=np.int64(5))
    assert sorted(r["SC"].to_list()) == ["C1", "THA"]


@pytest.mark.parametrize("bad", [True, "nan", float("nan"), float("inf"), -float("inf")])
def test_bool_and_non_finite_filter_values_rejected(bad):
    with pytest.raises(MangoFilterError):
        _validate_filters({"bz_imf_max": bad}, "magnetosheath", MAGNETOSHEATH_FILTERS, {})


def test_numpy_bool_filter_value_rejected():
    np = pytest.importorskip("numpy")
    with pytest.raises(MangoFilterError, match="must be numeric"):
        _validate_filters(
            {"bz_imf_max": np.bool_(True)}, "magnetosheath", MAGNETOSHEATH_FILTERS, {}
        )


def test_non_json_400_is_a_server_error(tmp_path):
    def proxy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="<html>Bad Request (proxy)</html>")

    c = MangoClient("http://testserver", transport=httpx.MockTransport(proxy), cache_dir=tmp_path)
    with pytest.raises(ServerError, match="proxy"):
        c.regions()


def test_old_server_without_dataset_endpoint(tmp_path):
    def old(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/dataset":
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json=["magnetosheath"])

    c = MangoClient("http://testserver", transport=httpx.MockTransport(old), cache_dir=tmp_path)
    with pytest.raises(ServerError, match=r"older than 0\.2.*space-mango<0\.2"):
        c.dataset_info()
