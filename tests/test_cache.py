import os
import warnings
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
    checksum = c.dataset_info()["schema_checksum"]
    assert (root / f"2026.0-{checksum[:12]}").is_dir()
    c._dataset_info = {**c.dataset_info(), "version": "2027.0"}
    c.get_data("solar_wind")
    assert (root / f"2027.0-{checksum[:12]}").is_dir()


def test_new_schema_checksum_uses_new_directory(dataset_dir, make_client, tmp_path):
    c = _fresh(dataset_dir, make_client, tmp_path / "c")
    c.get_data("solar_wind")
    c._dataset_info = {**c.dataset_info(), "schema_checksum": "abcdef0123456789" * 4}
    c.get_data("solar_wind")
    assert (tmp_path / "c" / "2026.0-abcdef012345" / "solar_wind").is_dir()


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
    monkeypatch.setattr(c._cache, "missing", lambda *a, **k: [])
    assert c.get_data("magnetosphere", spacecraft=["THA"]).to_polars().equals(expected)
    assert victim.is_file()


# --- final-review fix wave: fetch planning, request size, unwritable cache, count -----------


class _Recorder(httpx.BaseTransport):
    """Passes requests to the in-process server and remembers them."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self.inner = inner
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.inner.handle_request(request)

    def data(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/data")]

    def clear(self) -> None:
        self.requests.clear()


def _sw_rows(make_row, *days: datetime):
    return {"solar_wind": [make_row("solar_wind", "THA", t, Np=float(i), Bz=-float(i))
                           for i, t in enumerate(days)]}


@pytest.fixture
def year_dataset(make_row, make_dataset):
    """THA in the solar wind on 2016-01-10, 2016-03-10 and 2016-12-10."""
    return make_dataset(_sw_rows(make_row, datetime(2016, 1, 10), datetime(2016, 3, 10),
                                 datetime(2016, 12, 10)))


def _recording(data_dir, make_api, cache_dir, **kw) -> tuple[MangoClient, _Recorder]:
    rec = _Recorder(make_api(data_dir)._transport)
    return MangoClient("http://testserver", transport=rec, cache_dir=cache_dir, **kw), rec


def _window(r: httpx.Request) -> tuple[str | None, str | None]:
    return r.url.params.get("start"), r.url.params.get("stop")


def test_exclusive_stop_does_not_fetch_the_stop_month(year_dataset, make_api, tmp_path):
    c, rec = _recording(year_dataset, make_api, tmp_path / "c")
    r = c.get_data("solar_wind", start="2016-01", stop="2016-02")
    assert r["Np"].to_list() == [0.0]
    assert [_window(q) for q in rec.data()] == [("2016-01-01T00:00:00", "2016-02-01T00:00:00")]
    assert not list((tmp_path / "c").rglob("2016-02.parquet"))


def test_long_run_is_split_into_six_month_requests(year_dataset, make_api, tmp_path):
    from space_mango.client import MAX_MONTHS_PER_REQUEST

    assert MAX_MONTHS_PER_REQUEST == 6
    c, rec = _recording(year_dataset, make_api, tmp_path / "c")
    got = c.get_data("solar_wind")
    assert [_window(q) for q in rec.data()] == [
        ("2016-01-01T00:00:00", "2016-07-01T00:00:00"),
        ("2016-07-01T00:00:00", "2017-01-01T00:00:00"),
    ]
    assert got.to_polars().equals(c.get_data("solar_wind", cache=False).to_polars())


def test_adding_a_column_fetches_only_that_column(year_dataset, make_api, tmp_path):
    c, rec = _recording(year_dataset, make_api, tmp_path / "c")
    c.get_data("solar_wind", columns=["Time", "Np"])
    rec.clear()
    got = c.get_data("solar_wind", columns=["Time", "Np", "Bz"])
    assert rec.data() and all(q.url.params.get_list("columns") == ["Time", "Bz"]
                              for q in rec.data())
    remote = c.get_data("solar_wind", columns=["Time", "Np", "Bz"], cache=False)
    assert got.to_polars().equals(remote.to_polars())


def test_time_mismatch_refetches_all_columns_of_the_month(year_dataset, make_api, tmp_path):
    c, rec = _recording(year_dataset, make_api, tmp_path / "c")
    c.get_data("solar_wind", columns=["Time", "Np"])
    jan = date(2016, 1, 1)
    stale = c._cache.path(c._cache_key(), "solar_wind", "THA", "Time", jan)
    pl.DataFrame({"Time": [datetime(2016, 1, 11)]}).cast({"Time": pl.Datetime("ns")}) \
        .write_parquet(stale)  # same length, other values: fragments no longer line up
    rec.clear()
    got = c.get_data("solar_wind", columns=["Time", "Np", "Bz"])
    cols = [(_window(q), q.url.params.get_list("columns")) for q in rec.data()]
    assert (("2016-01-01T00:00:00", "2016-02-01T00:00:00"), ["Time", "Bz", "Np"]) in cols
    remote = c.get_data("solar_wind", columns=["Time", "Np", "Bz"], cache=False)
    assert got.to_polars().equals(remote.to_polars())


def test_count_reports_cache_download_estimate(dataset_dir, make_api, tmp_path):
    c, rec = _recording(dataset_dir, make_api, tmp_path / "c")
    cold = c.count("magnetosheath", bz_imf_max=-2)
    assert cold["n_rows"] == 2
    assert cold["download_mb_estimate"] >= cold["est_mb"] > 0
    assert rec.data() == []  # count downloads nothing
    c.get_data("magnetosheath", bz_imf_max=-2)
    warm = c.count("magnetosheath", bz_imf_max=-2)
    assert warm["download_mb_estimate"] == 0 and warm["n_rows"] == 2


def test_count_download_estimate_counts_only_missing_columns(year_dataset, make_api, tmp_path):
    c, _ = _recording(year_dataset, make_api, tmp_path / "c")
    cold = c.count("solar_wind", columns=["Time", "Np", "Bz"])["download_mb_estimate"]
    c.get_data("solar_wind", columns=["Time", "Np"])
    partial = c.count("solar_wind", columns=["Time", "Np", "Bz"])["download_mb_estimate"]
    assert 0 < partial < cold


def test_count_without_cache_estimates_the_server_download(dataset_dir, make_api, tmp_path):
    c, _ = _recording(dataset_dir, make_api, tmp_path / "c", cache=False)
    n = c.count("magnetosheath", bz_imf_max=-2)
    assert n["download_mb_estimate"] == n["est_mb"]


@pytest.fixture
def read_only_dir(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    yield ro
    ro.chmod(0o700)


def test_unwritable_cache_dir_with_cache_disabled(dataset_dir, make_api, read_only_dir):
    c, _ = _recording(dataset_dir, make_api, read_only_dir / "cache", cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(c.get_data("magnetosheath")) == 3
        assert c.count("magnetosheath")["n_rows"] == 3


def test_unwritable_cache_dir_warns_once_and_still_returns_data(
    dataset_dir, make_api, read_only_dir
):
    c, _ = _recording(dataset_dir, make_api, read_only_dir / "cache")
    with pytest.warns(UserWarning, match="SPACE_MANGO_CACHE_DIR"):
        got = c.get_data("magnetosheath", bz_imf_max=-2)
    assert got.to_polars().equals(c.get_data("magnetosheath", bz_imf_max=-2, cache=False).to_polars())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(c.get_data("magnetosheath")) == 3


def test_fragment_write_failure_warns_and_returns_data(
    dataset_dir, make_api, tmp_path, monkeypatch
):
    c, _ = _recording(dataset_dir, make_api, tmp_path / "c")
    c.dataset_info()  # metadata is writable

    def disk_full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(c._cache, "write_month", disk_full)
    with pytest.warns(UserWarning, match="SPACE_MANGO_CACHE_DIR"):
        got = c.get_data("magnetosphere", spacecraft="THA")
    assert got.to_polars().equals(
        c.get_data("magnetosphere", spacecraft="THA", cache=False).to_polars())
