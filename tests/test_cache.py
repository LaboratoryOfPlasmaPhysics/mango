from datetime import date, datetime
from typing import Any

import httpx
import polars as pl
import pytest

from space_mango.cache import FragmentCache, contiguous_runs, months_between, next_month
from space_mango.client import MangoClient
from space_mango.errors import CacheMissError, MangoFilterError, UnknownSpacecraftError


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


def _no_network(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("network is down", request=request)


def _fresh(dataset_dir, make_client, cache_dir, **kw) -> MangoClient:
    """A client on its own cache directory (the default one is shared with other tests)."""
    return MangoClient("http://testserver", transport=make_client(dataset_dir)._http._transport,
                       cache_dir=cache_dir, **kw)


OFFLINE_QUERIES: list[dict[str, Any]] = [
    {"spacecraft": ["THA"]},
    {"spacecraft": ["THA"], "columns": ["Time", "Bz", "SC"], "bz_imf_max": 2},
    {"spacecraft": ["THA"], "start": "2030-01-01"},  # nothing overlaps: empty result
]


def test_offline_works_with_network_down(dataset_dir, make_client, tmp_path):
    online = _fresh(dataset_dir, make_client, tmp_path / "c")
    expected = []
    for q in OFFLINE_QUERIES:
        online.get_data("magnetosphere", **q)  # fills the cache
        expected.append(online.get_data("magnetosphere", cache=False, **q).to_polars())
    off = MangoClient("http://testserver", transport=httpx.MockTransport(_no_network),
                      cache_dir=tmp_path / "c", offline=True)
    for q, exp in zip(OFFLINE_QUERIES, expected, strict=True):
        got = off.get_data("magnetosphere", **q)
        assert got.to_polars().equals(exp)
    assert off.dataset_info()["version"] == online.dataset_info()["version"]


def test_offline_with_empty_cache_raises_cache_miss_not_server_error(tmp_path):
    off = MangoClient("http://testserver", transport=httpx.MockTransport(_no_network),
                      cache_dir=tmp_path / "empty", offline=True)
    with pytest.raises(CacheMissError, match="online"):
        off.get_data("magnetosphere", spacecraft=["THA"])


def test_offline_missing_fragment_with_metadata_cached(dataset_dir, make_client, tmp_path):
    _fresh(dataset_dir, make_client, tmp_path / "c").get_data("magnetosphere", spacecraft=["THA"])
    off = MangoClient("http://testserver", transport=httpx.MockTransport(_no_network),
                      cache_dir=tmp_path / "c", offline=True)
    with pytest.raises(CacheMissError, match="C3"):
        off.get_data("magnetosphere", spacecraft=["C3"])


def test_eviction_keeps_metadata(tmp_path):
    fc = FragmentCache(tmp_path, max_bytes=1)
    fc.write_meta("v", "dataset.json", {"version": "v"})
    fc.write_months("v", "solar_wind", "THA", [date(2016, 1, 1)],
                    pl.DataFrame({"Time": [datetime(2016, 1, 3)], "Np": [1.0]}))
    fc.evict()
    assert fc.read_meta("v", "dataset.json") == {"version": "v"}
    assert fc.info()["n_files"] == 0
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("flag", ["sw_paired_only", "normalized_only"])
def test_flag_on_region_without_flag_column(client, cache, flag):
    with pytest.raises(MangoFilterError, match="needs column"):
        client.get_data("solar_wind", cache=cache, **{flag: True})


def test_fragment_deleted_between_has_and_read_is_refetched(
    dataset_dir, make_client, tmp_path, monkeypatch
):
    c = _fresh(dataset_dir, make_client, tmp_path / "c")
    expected = c.get_data("magnetosphere", spacecraft=["THA"]).to_polars()
    victim = next(c._cache.root.rglob("SC=THA/Np/*.parquet"))
    victim.unlink()  # as if another process evicted it after has() said yes
    monkeypatch.setattr(c._cache, "has", lambda *a, **k: True)
    assert c.get_data("magnetosphere", spacecraft=["THA"]).to_polars().equals(expected)
    assert victim.is_file()
