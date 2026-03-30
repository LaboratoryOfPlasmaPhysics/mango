from fastapi.testclient import TestClient

from mango.app import create_app

client = TestClient(create_app())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_regions():
    r = client.get("/api/v1/regions")
    assert r.status_code == 200
    assert set(r.json()) == {"magnetosphere", "magnetosheath", "solar_wind"}
