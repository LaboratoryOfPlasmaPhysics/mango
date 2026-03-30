from enum import StrEnum

from pydantic import BaseModel


class Format(StrEnum):
    arrow = "arrow"
    csv = "csv"


class Region(StrEnum):
    magnetosphere = "magnetosphere"
    magnetosheath = "magnetosheath"
    solar_wind = "solar_wind"


class RangeFilter(BaseModel):
    column: str
    unit: str
    description: str
    # Which regions this filter applies to
    regions: frozenset[Region] = frozenset(Region)


# ---- Filter catalog: single source of truth ----

RANGE_FILTERS: dict[str, RangeFilter] = {
    # Upstream solar wind conditions (paired)
    "bz_imf": RangeFilter(
        column="Bz_imf", unit="nT",
        description="IMF Bz — southward (<0) drives reconnection",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "by_imf": RangeFilter(
        column="By_imf", unit="nT",
        description="IMF By — controls reconnection geometry and asymmetry",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "bx_imf": RangeFilter(
        column="Bx_imf", unit="nT",
        description="IMF Bx — cone angle / Parker spiral orientation",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "pd_sw": RangeFilter(
        column="Pd_sw", unit="nPa",
        description="Solar wind dynamic pressure",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "np_sw": RangeFilter(
        column="Np_sw", unit="cm⁻³",
        description="Solar wind proton density",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "tp_sw": RangeFilter(
        column="Tp_sw", unit="K",
        description="Solar wind proton temperature",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "vx_sw": RangeFilter(
        column="Vx_sw", unit="km/s",
        description="Solar wind velocity X (GSM) — bulk speed",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "beta_sw": RangeFilter(
        column="Beta_sw", unit="",
        description="Solar wind plasma beta",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    "ma_sw": RangeFilter(
        column="Ma_sw", unit="",
        description="Solar wind Alfvén Mach number",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    # Dipole tilt
    "tilt": RangeFilter(
        column="Tilt", unit="rad",
        description="Dipole tilt angle — seasonal / hemispheric asymmetry",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
    # Spatial — raw GSM position
    "x_gsm": RangeFilter(
        column="X_gsm", unit="Re",
        description="X GSM coordinate",
    ),
    "y_gsm": RangeFilter(
        column="Y_gsm", unit="Re",
        description="Y GSM coordinate",
    ),
    "z_gsm": RangeFilter(
        column="Z_gsm", unit="Re",
        description="Z GSM coordinate",
    ),
    # Spatial — normalized relative position
    "d_msp": RangeFilter(
        column="D_msp", unit="",
        description="Relative distance Earth(0)–magnetopause(1)",
        regions=frozenset({Region.magnetosphere}),
    ),
    "d_msh": RangeFilter(
        column="D_msh", unit="",
        description="Relative distance magnetopause(0)–bow shock(1)",
        regions=frozenset({Region.magnetosheath}),
    ),
    # Local plasma measurements
    "np": RangeFilter(
        column="Np", unit="cm⁻³",
        description="Local plasma density",
    ),
    "tp": RangeFilter(
        column="Tp", unit="K",
        description="Local plasma temperature",
    ),
    "bz": RangeFilter(
        column="Bz", unit="nT",
        description="Local Bz (GSM)",
    ),
}


class DatasetInfo(BaseModel):
    region: str
    row_count: int
    columns: list[str]


class FilterInfo(BaseModel):
    name: str
    column: str
    unit: str
    description: str
    params: str  # "min=..&max=.."
