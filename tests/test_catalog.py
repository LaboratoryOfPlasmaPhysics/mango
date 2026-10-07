"""The catalog must describe exactly what the server serves (SERVED_COLUMNS in conftest)."""

import subprocess
import sys

from conftest import SERVED_COLUMNS

from space_mango.models import (
    COLUMNS,
    RANGE_FILTERS,
    REGIONS,
    Region,
    citation_bibtex,
    columns_for,
    filter_for_column,
    filters_for,
)


def test_every_region_has_a_definition():
    assert set(REGIONS) == set(Region)
    assert all(info.definition for info in REGIONS.values())


def test_catalog_columns_match_served_columns():
    for region in Region:
        assert set(columns_for(region)) == set(SERVED_COLUMNS[region.value]) | {"SC"}, region


def test_every_column_is_documented():
    for name, col in COLUMNS.items():
        for region in col.regions:
            assert col.description_for(region), name


def test_filters_point_at_documented_columns_in_their_regions():
    for name, f in RANGE_FILTERS.items():
        col = COLUMNS[f.column]
        assert f.regions <= col.regions, name
        assert f.unit == col.unit, name


def test_r_norm_is_described_per_region():
    msh = COLUMNS["R_norm"].description_for(Region.magnetosheath)
    msp = COLUMNS["R_norm"].description_for(Region.magnetosphere)
    assert "bow shock" in msh
    assert msh != msp


def test_filter_lookup_helpers():
    assert "d_msh" in filters_for("magnetosheath")
    assert "tilt" not in filters_for(Region.magnetosheath)
    assert filter_for_column("magnetosheath", "R_norm") == "d_msh"
    assert filter_for_column("magnetosphere", "R_norm") == "d_msp"
    assert filter_for_column("magnetosheath", "Bx_swi") is None


def test_citation_mentions_version_and_doi():
    text = citation_bibtex("2026.0", "10.5281/zenodo.1")
    assert "2026.0" in text and "10.5281/zenodo.1" in text and text.startswith("@")


def test_client_import_does_not_pull_server_dependencies():
    code = (
        "import sys, space_mango, space_mango.models; "
        "bad = {'pydantic', 'fastapi', 'pandas'} & set(sys.modules); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
