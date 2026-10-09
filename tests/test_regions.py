import inspect
from pathlib import Path

import space_mango._regions_generated as generated
from space_mango._codegen import render_regions_module
from space_mango._regions_generated import REGION_APIS, MagnetosheathAPI
from space_mango.models import filters_for


def test_generated_module_is_up_to_date():
    expected = render_regions_module()
    actual = Path(generated.__file__).read_text()
    assert actual == expected, "run: uv run python -m space_mango._codegen"


def test_signatures_list_every_filter():
    for name, cls in REGION_APIS.items():
        method = cls.get_data  # pyright: ignore[reportAttributeAccessIssue]
        params = inspect.signature(method).parameters
        for f in filters_for(name):
            assert f"{f}_min" in params and f"{f}_max" in params
        assert "tilt_min" not in inspect.signature(MagnetosheathAPI.get_data).parameters


def test_docstring_shows_units():
    doc = MagnetosheathAPI.get_data.__doc__ or ""
    assert "bz_imf_min / bz_imf_max" in doc and "[nT]" in doc


def test_region_object_delegates(client):
    msh = MagnetosheathAPI(lambda: client)
    r = msh.get_data(spacecraft=["THA"], bz_imf_max=-2)
    assert r["SC"].to_list() == ["THA"]
    assert msh.count(bz_imf_max=-2)["n_rows"] == 2
    assert "column" in msh.describe().columns
    assert "bow shock" in repr(msh)


def test_frame_parameters_only_where_they_apply():
    from space_mango._regions_generated import MagnetosphereAPI, SolarWindAPI

    msh = inspect.signature(MagnetosheathAPI.get_data).parameters
    msp = inspect.signature(MagnetosphereAPI.get_data).parameters
    sw = inspect.signature(SolarWindAPI.get_data).parameters
    assert {"frame", "cone", "clock"} <= set(msh) and "tilt" not in msh
    assert {"frame", "cone", "clock", "tilt"} <= set(msp)
    assert "frame" in sw and "cone" not in sw

