from datetime import datetime

from fastapi.testclient import TestClient

from space_mango.app import create_app
from space_mango.dataset import MangoDataset
from space_mango.models import Region

client = TestClient(create_app())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_regions():
    r = client.get("/api/v1/regions")
    assert r.status_code == 200
    assert set(r.json()) == {"magnetosphere", "magnetosheath", "solar_wind"}


def test_filters_magnetosheath():
    r = client.get("/api/v1/regions/magnetosheath/filters")
    assert r.status_code == 200
    filters = r.json()
    names = {f["name"] for f in filters}
    # Solar wind conditions should be available
    assert "bz_imf" in names
    assert "pd_sw" in names
    assert "beta_sw" in names
    assert "ma_sw" in names
    # Sheath-specific
    assert "d_msh" in names
    # Magnetosphere-specific should NOT be here
    assert "d_msp" not in names


def test_filters_magnetosphere():
    r = client.get("/api/v1/regions/magnetosphere/filters")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()}
    assert "d_msp" in names
    assert "d_msh" not in names
    assert "bz_imf" in names


def test_filters_solar_wind():
    r = client.get("/api/v1/regions/solar_wind/filters")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()}
    # Solar wind region has no upstream pairing
    assert "bz_imf" not in names
    assert "pd_sw" not in names
    # But spatial and local plasma filters still apply
    assert "x_gsm" in names
    assert "np" in names


def test_hive_dataset_query_all(dataset_dir):
    df = MangoDataset(dataset_dir).query(Region.magnetosheath, {}, limit=100)
    assert len(df) == 3
    assert {"SC", "Time"} <= set(df.columns)


def test_hive_dataset_query_spacecraft_filter(dataset_dir):
    df = MangoDataset(dataset_dir).query(Region.magnetosheath, {}, spacecraft=["THA"], limit=100)
    assert df["SC"].to_list() == ["THA"]


def test_hive_dataset_query_time_filter(dataset_dir):
    df = MangoDataset(dataset_dir).query(
        Region.magnetosheath, {}, start=datetime(2017, 1, 1), limit=100
    )
    assert set(df["SC"].to_list()) == {"MMS", "C1"}


def test_data_endpoint_returns_csv(api):
    r = api.get("/api/v1/regions/magnetosheath/data?format=csv&limit=10")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert len(r.text.strip().splitlines()) == 4  # header + 3 rows


def test_data_endpoint_spacecraft_filter(api):
    r = api.get("/api/v1/regions/magnetosheath/data?format=csv&spacecraft=THA&limit=10")
    lines = r.text.strip().splitlines()
    assert len(lines) == 2
    assert "THA" in lines[1]
