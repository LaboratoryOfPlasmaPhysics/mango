"""MANGO catalog: the single source of truth for regions, columns and range filters.

Plain dataclasses on purpose: the client imports this module and must not need pydantic.
Column descriptions follow docs/superpowers/specs/2026-10-07-mango-column-dictionary-draft.md;
wording stays cautious where the dictionary says INFERRED/UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Format(StrEnum):
    arrow = "arrow"
    csv = "csv"


class Region(StrEnum):
    magnetosphere = "magnetosphere"
    magnetosheath = "magnetosheath"
    solar_wind = "solar_wind"


@dataclass(frozen=True)
class RangeFilter:
    column: str
    unit: str
    description: str
    # Which regions this filter applies to
    regions: frozenset[Region] = frozenset(Region)


@dataclass(frozen=True)
class RegionInfo:
    definition: str


@dataclass(frozen=True)
class ColumnInfo:
    unit: str
    frame: str
    description: str
    computed: str
    regions: frozenset[Region]
    per_region: tuple[tuple[Region, str], ...] = ()

    def description_for(self, region: Region | str) -> str:
        for r, text in self.per_region:
            if r == region:
                return text
        return self.description


_ALL = frozenset(Region)
_PAIRED = frozenset({Region.magnetosphere, Region.magnetosheath})
_MSH = frozenset({Region.magnetosheath})
_MSP = frozenset({Region.magnetosphere})

REGIONS: dict[Region, RegionInfo] = {
    Region.magnetosphere: RegionInfo(
        "Inside the magnetopause: closed-field-line magnetosphere sampled by the spacecraft "
        "(machine-learning region classification, Nguyen et al. 2022)."
    ),
    Region.magnetosheath: RegionInfo(
        "Between the bow shock and the magnetopause: shocked solar wind "
        "(machine-learning region classification, Nguyen et al. 2022)."
    ),
    Region.solar_wind: RegionInfo(
        "Upstream of the bow shock: pristine solar wind measured in situ by the same spacecraft "
        "(machine-learning region classification, Nguyen et al. 2022)."
    ),
}


def _col(
    unit: str, frame: str, description: str, computed: str,
    regions: frozenset[Region] = _ALL, per_region: tuple[tuple[Region, str], ...] = (),
) -> ColumnInfo:
    return ColumnInfo(unit, frame, description, computed, regions, per_region)


# SWI construction, verified against the upstream notebook MSH_GSM_to_SWI.ipynb and on
# 132k served rows (2026-10-08). R = rows (X, Y, Z) of the SWI basis of each row.
_SWI_BASIS = ("SWI: X = -V_sw/|V_sw| (served V_sw), Z = X x (s B_imf)/|.|, Y = Z x X, "
              "s = sgn(Bx_imf) (else sgn(By_imf), else sgn(Bz_imf))")
_SWI_B = _SWI_BASIS + "; B_swi = s R B, so the IMF has Bx > 0 along +Y_SWI (sgn(Bx_imf) factor)"
_SWI_V = _SWI_BASIS + "; V_swi = R (V - 29.8 km/s along Y_GSM) (Earth orbital motion removed)"
_SWI_R = (_SWI_BASIS + "; R applied to X/Y/Z_gsm_norm, then re-normalized between the mean "
          "Shue98/Jelinek2012 boundaries at the SWI angles and clipped to [0, 1]: the radius "
          "can differ from |r_gsm_norm| near a boundary")

COLUMNS: dict[str, ColumnInfo] = {
    "Time": _col("", "", "Sample time (UTC assumed), on a 5 s grid",
                 "5 s averages of the mission data"),
    "SC": _col("", "", "Spacecraft: THA–THE (THEMIS), C1, C3 (Cluster), MMS, DS1 (Double Star)",
               "Hive partition key SC=<name>"),
    "Bx": _col("nT", "GSM", "Local magnetic field, X", "Fluxgate magnetometer, 5 s mean"),
    "By": _col("nT", "GSM", "Local magnetic field, Y", "Fluxgate magnetometer, 5 s mean"),
    "Bz": _col("nT", "GSM", "Local magnetic field, Z", "Fluxgate magnetometer, 5 s mean"),
    "Np": _col("cm⁻³", "", "Local ion density", "Ion moments (THEMIS ESA, Cluster HIA, MMS FPI)"),
    "Vx": _col("km/s", "GSM", "Local ion bulk velocity, X", "Ion moments"),
    "Vy": _col("km/s", "GSM", "Local ion bulk velocity, Y", "Ion moments"),
    "Vz": _col("km/s", "GSM", "Local ion bulk velocity, Z", "Ion moments"),
    "Tp": _col("K", "", "Local ion temperature (T∥ + 2T⊥)/3", "Ion moments, converted from eV"),
    "X_gsm": _col("R_E", "GSM", "Spacecraft position, X", "Orbit data, 5 s mean"),
    "Y_gsm": _col("R_E", "GSM", "Spacecraft position, Y", "Orbit data, 5 s mean"),
    "Z_gsm": _col("R_E", "GSM", "Spacecraft position, Z", "Orbit data, 5 s mean"),
    "SW_pairing": _col("", "", "True when an upstream solar-wind sample is associated to this row",
                       "OMNI propagated to the spacecraft", _PAIRED),
    "Bx_imf": _col("nT", "GSM", "Upstream IMF Bx paired with this row", "OMNI, time-shifted", _PAIRED),
    "By_imf": _col("nT", "GSM", "Upstream IMF By paired with this row", "OMNI, time-shifted", _PAIRED),
    "Bz_imf": _col("nT", "GSM", "Upstream IMF Bz paired with this row", "OMNI, time-shifted", _PAIRED),
    "Np_sw": _col("cm⁻³", "", "Upstream proton density", "OMNI, time-shifted", _PAIRED),
    "Vx_sw": _col("km/s", "GSM", "Upstream velocity X; negative (anti-sunward), not a speed",
                  "OMNI, time-shifted", _PAIRED),
    "Vy_sw": _col("km/s", "GSM", "Upstream velocity Y", "OMNI, time-shifted", _PAIRED),
    "Vz_sw": _col("km/s", "GSM", "Upstream velocity Z", "OMNI, time-shifted", _PAIRED),
    "Tp_sw": _col("K", "", "Upstream proton temperature", "OMNI, time-shifted", _PAIRED),
    "Pd_sw": _col("nPa", "", "Upstream dynamic pressure", "OMNI: 2e-6 · Np · V² (includes He)", _PAIRED),
    "Beta_sw": _col("", "", "Upstream plasma beta", "OMNI (electrons and He included)", _PAIRED),
    "Ma_sw": _col("", "", "Upstream Alfvén Mach number", "OMNI", _PAIRED),
    "tilt": _col("rad", "", "Dipole tilt angle (positive near June solstice)",
                 "Analytic approximation, not IGRF: 23.4°·cos(2π(doy−172)/365.25) "
                 "+ 11.2°·cos(2π(UT−16.72)/24) (spok.get_tilt)", _MSP),
    "R_mp": _col("R_E", "radial", "Magnetopause distance along the spacecraft direction",
                 "Magnetopause model driven by the paired solar wind", _PAIRED),
    "R_bs": _col("R_E", "radial", "Bow-shock distance along the spacecraft direction",
                 "Bow-shock model driven by the paired solar wind", _MSH),
    "R_norm": _col(
        "", "", "Normalized radial position", "Ratio of |r| to the model boundaries", _PAIRED,
        per_region=(
            (Region.magnetosheath, "Fractional position from magnetopause (0) to bow shock (1): "
                                   "(|r| - R_mp) / (R_bs - R_mp)"),
            (Region.magnetosphere, "Fractional position from Earth (0) to magnetopause (1): |r| / R_mp"),
        ),
    ),
    "Norma_pos": _col("", "", "True when the row has a normalized position (*_norm columns)",
                      "Requires solar-wind pairing", _PAIRED),
    "X_gsm_norm": _col("R_E", "GSM", "Normalized position, X: radially rescaled between fixed average boundaries",
                       "Same direction as the spacecraft position", _PAIRED),
    "Y_gsm_norm": _col("R_E", "GSM", "Normalized position, Y: radially rescaled between fixed average boundaries",
                       "Same direction as the spacecraft position", _PAIRED),
    "Z_gsm_norm": _col("R_E", "GSM", "Normalized position, Z: radially rescaled between fixed average boundaries",
                       "Same direction as the spacecraft position", _PAIRED),
    "Bx_swi": _col("nT", "SWI", "Local magnetic field in the SWI frame, X",
                   _SWI_B, _MSH),
    "By_swi": _col("nT", "SWI", "Local magnetic field in the SWI frame, Y",
                   _SWI_B, _MSH),
    "Bz_swi": _col("nT", "SWI", "Local magnetic field in the SWI frame, Z",
                   _SWI_B, _MSH),
    "Vx_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, X (aberration-corrected)",
                   _SWI_V, _MSH),
    "Vy_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, Y (aberration-corrected)",
                   _SWI_V, _MSH),
    "Vz_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, Z (aberration-corrected)",
                   _SWI_V, _MSH),
    "X_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, X", _SWI_R, _MSH),
    "Y_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, Y", _SWI_R, _MSH),
    "Z_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, Z", _SWI_R, _MSH),
}

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
        description="Solar wind velocity X (GSM); negative (anti-sunward), so faster wind is more negative",
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
        column="tilt", unit="rad",
        description="Dipole tilt angle (positive near June solstice)",
        regions=frozenset({Region.magnetosphere}),
    ),
    # Spatial — raw GSM position
    "x_gsm": RangeFilter(
        column="X_gsm", unit="R_E",
        description="X GSM coordinate",
    ),
    "y_gsm": RangeFilter(
        column="Y_gsm", unit="R_E",
        description="Y GSM coordinate",
    ),
    "z_gsm": RangeFilter(
        column="Z_gsm", unit="R_E",
        description="Z GSM coordinate",
    ),
    # Spatial — normalized relative position
    "d_msp": RangeFilter(
        column="R_norm", unit="",
        description="Relative distance Earth(0)–magnetopause(1): |r| / R_mp",
        regions=frozenset({Region.magnetosphere}),
    ),
    "d_msh": RangeFilter(
        column="R_norm", unit="",
        description="Relative distance magnetopause(0)–bow shock(1): (|r| - R_mp) / (R_bs - R_mp)",
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


def filters_for(region: Region | str) -> dict[str, RangeFilter]:
    r = Region(region)
    return {name: f for name, f in RANGE_FILTERS.items() if r in f.regions}


def columns_for(region: Region | str) -> dict[str, ColumnInfo]:
    r = Region(region)
    return {name: c for name, c in COLUMNS.items() if r in c.regions}


def filter_for_column(region: Region | str, column: str) -> str | None:
    for name, f in filters_for(region).items():
        if f.column == column:
            return name
    return None


DEFAULT_DATASET_VERSION = "2026.0"
DATASET_TITLE = "MANGO: Magnetospheric Atlas of Normalized Geospace Observations"


def citation_bibtex(version: str, doi: str | None) -> str:
    """BibTeX for the dataset. The data paper is in preparation (spec §9, question 8)."""
    doi_line = f"  doi    = {{{doi}}},\n" if doi else ""
    return (
        "@misc{mango_dataset,\n"
        f"  title  = {{{DATASET_TITLE}}},\n"
        "  author = {Michotte de Welle, B. and Aunai, N. and Ghisalberti, A. and Lavraud, B. and others},\n"
        "  year   = {2026},\n"
        f"  note   = {{Dataset version {version}. Data descriptor in preparation for Scientific Data.}},\n"
        f"{doi_line}"
        "  url    = {https://github.com/LaboratoryOfPlasmaPhysics/mango}\n"
        "}"
    )
