import polars as pl
import pytest

from space_mango.errors import UnknownRegionError


def test_describe(client):
    d = client.describe("magnetosheath")
    assert d.columns == ["column", "unit", "frame", "description", "filter", "dtype"]
    row = d.filter(pl.col("column") == "R_norm").row(0, named=True)
    assert row["filter"] == "d_msh" and "bow shock" in row["description"]


def test_region_definition(client):
    assert "magnetopause" in client.region_definition("magnetosphere")


def test_spacecraft(client):
    sc = client.spacecraft("magnetosheath")
    assert sc["sc"].to_list() == ["C1", "MMS", "THA"]
    assert sc.schema["start"] == pl.Datetime("us")


def test_count(client):
    c = client.count("magnetosheath", bz_imf_max=-2)
    assert c["n_rows"] == 2 and c["est_mb"] > 0


def test_count_validates_like_get_data(client):
    with pytest.raises(UnknownRegionError):
        client.count("nowhere")


def test_search(client):
    hits = client.search("density")
    assert {"Np", "Np_sw"} <= set(hits["name"].to_list())
    assert set(hits["kind"].to_list()) <= {"column", "filter"}
    assert client.search("zzzz").height == 0


def test_timeline(client):
    t = client.timeline("THA", "2016-03-15T09:00", "2016-03-15T11:00")
    assert t["region"].to_list() == ["magnetosheath"]
    assert t.region is None and t.version == "2026.0"


def test_cite(client):
    assert client.cite().startswith("@misc")


def test_context_manager(dataset_dir, make_client):
    with make_client(dataset_dir) as c:
        assert "magnetosheath" in c.regions()
