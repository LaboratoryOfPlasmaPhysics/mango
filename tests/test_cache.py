from datetime import date, datetime

import polars as pl
import pytest

from space_mango.cache import FragmentCache, contiguous_runs, months_between, next_month
from space_mango.client import MangoClient
from space_mango.errors import CacheMissError, UnknownSpacecraftError


def test_month_helpers():
    assert months_between(datetime(2016, 11, 20), datetime(2017, 2, 1)) == [
        date(2016, 11, 1), date(2016, 12, 1), date(2017, 1, 1), date(2017, 2, 1)]
    assert next_month(date(2016, 12, 1)) == date(2017, 1, 1)
    assert contiguous_runs([date(2016, 1, 1), date(2016, 2, 1), date(2016, 5, 1)]) == [
        [date(2016, 1, 1), date(2016, 2, 1)], [date(2016, 5, 1)]]


def test_fragment_roundtrip_and_empty_months(tmp_path):
    fc = FragmentCache(tmp_path, max_bytes=10**9)
    df = pl.DataFrame({"Time": [datetime(2016, 1, 3), datetime(2016, 3, 9)], "Np": [1.0, 2.0]})
    months = [date(2016, 1, 1), date(2016, 2, 1), date(2016, 3, 1)]
    fc.write_months("2026.0", "magnetosheath", "THA", months, df)
    assert fc.has("2026.0", "magnetosheath", "THA", date(2016, 2, 1), ["Time", "Np"])
    assert fc.read_month("2026.0", "magnetosheath", "THA", date(2016, 2, 1), ["Time", "Np"]).height == 0
    assert fc.read_month("2026.0", "magnetosheath", "THA", date(2016, 3, 1), ["Time", "Np"])["Np"].to_list() == [2.0]
    assert not fc.has("2026.1", "magnetosheath", "THA", date(2016, 1, 1), ["Time"])
    assert not list(tmp_path.rglob("*.tmp"))


def test_eviction_removes_oldest(tmp_path):
    fc = FragmentCache(tmp_path, max_bytes=1)
    df = pl.DataFrame({"Time": [datetime(2016, 1, 3)], "Np": [1.0]})
    fc.write_months("v", "magnetosheath", "THA", [date(2016, 1, 1)], df)
    fc.evict()
    assert fc.info()["size_bytes"] <= 1


QUERIES = [
    {},
    {"bz_imf_max": -2},
    {"columns": ["Time", "Np", "R_norm"], "d_msh_max": 0.5},
    {"spacecraft": ["THA", "C1"], "start": "2016-01-01", "stop": "2019-01-05T08:00:00"},
    {"sw_paired_only": True, "normalized_only": True},
]


@pytest.mark.parametrize("q", QUERIES)
def test_cache_path_equals_server_path(client, q):
    cached = client.get_data("magnetosheath", **q).to_polars()
    remote = client.get_data("magnetosheath", cache=False, **q).to_polars()
    key = ["SC", "Time"] if "SC" in remote.columns else ["Time"]
    assert cached.sort(key).equals(remote.select(cached.columns).sort(key))
    assert cached.columns == remote.columns


def test_legacy_time_max_inclusive_on_cache_path(client):
    with pytest.warns(FutureWarning):
        df = client.get_data("magnetosheath", time_max="2018-07-20T14:30:00")
    assert set(df["SC"].to_list()) == {"THA", "MMS"}


def test_second_call_is_served_from_cache(dataset_dir, make_client, monkeypatch):
    c = make_client(dataset_dir)
    c.get_data("magnetosphere", spacecraft=["THA"])
    calls: list[str] = []
    real = c._fetch_months
    monkeypatch.setattr(c, "_fetch_months", lambda *a, **k: calls.append("x") or real(*a, **k))
    c.get_data("magnetosphere", spacecraft=["THA"])
    assert calls == []


def test_offline_raises_on_missing_fragment(dataset_dir, make_client, tmp_path):
    c = make_client(dataset_dir)
    off = MangoClient("http://testserver", transport=c._http._transport,
                      cache_dir=tmp_path / "empty", offline=True)
    with pytest.raises(CacheMissError):
        off.get_data("magnetosphere", spacecraft=["THA"])


def test_new_dataset_version_uses_new_directory(dataset_dir, make_client, monkeypatch):
    c = make_client(dataset_dir)
    c.get_data("solar_wind")
    root = c._cache.root
    assert (root / "2026.0").is_dir()
    c._dataset_info = {**c.dataset_info(), "version": "2027.0"}
    c.get_data("solar_wind")
    assert (root / "2027.0").is_dir()


def test_cache_path_rejects_mms1_with_suggestion(client):
    with pytest.raises(UnknownSpacecraftError, match="Did you mean 'MMS'"):
        client.get_data("magnetosheath", spacecraft=["MMS1"])


def test_cache_info_and_clear(dataset_dir, make_client):
    c = make_client(dataset_dir)
    c.get_data("solar_wind")
    assert c.cache_info()["n_files"] > 0
    c.cache_clear()
    assert c.cache_info()["n_files"] == 0
