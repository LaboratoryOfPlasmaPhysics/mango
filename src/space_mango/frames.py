"""Coordinate frames of MANGO data: GSM, SWI and PGSM (B. Michotte de Welle, PhD thesis 2024, sections 2.7.3-2.7.4, eqs 2.13-2.20).

Magnetosheath: the served SWI columns, selected by IMF cone angle and rotated about X to a
target IMF clock angle (eqs 2.19-2.20). Magnetosphere: GSM plus the dipole-tilt symmetry
psi -> -psi (eqs 2.14-2.16). Pure polars: used by the client (transform, cache-path counts)
and by the server (/count). Angles are in degrees.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from space_mango.errors import QueryError
from space_mango.models import COLUMNS

FRAMES = ("gsm", "swi", "pgsm")
THESIS = (
    "B. Michotte de Welle (2024), PhD thesis, https://theses.hal.science/tel-04661957, "
    "sections 2.7.3-2.7.4"
)

_POS = ["X_pgsm_norm", "Y_pgsm_norm", "Z_pgsm_norm"]
_VEC = ["Bx_pgsm", "By_pgsm", "Bz_pgsm", "Vx_pgsm", "Vy_pgsm", "Vz_pgsm"]
_IMF = ["Bx_imf", "By_imf", "Bz_imf"]
_VSW = ["Vx_sw", "Vy_sw", "Vz_sw"]
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "magnetosheath": [
        *_IMF, *_VSW,
        "Bx_swi", "By_swi", "Bz_swi", "Vx_swi", "Vy_swi", "Vz_swi",
        "X_swi_norm", "Y_swi_norm", "Z_swi_norm",
    ],
    "magnetosphere": [
        "tilt", "Bx", "By", "Bz", "Vx", "Vy", "Vz", "X_gsm_norm", "Y_gsm_norm", "Z_gsm_norm",
    ],
}
"""Input columns of the PGSM transform, fetched even when not asked for."""
OUTPUT_COLUMNS: dict[str, list[str]] = {
    "magnetosheath": [*_POS, *_VEC, "mirrored", "bx_sign"],
    "magnetosphere": [*_POS, *_VEC, "mirrored", "tilt_pgsm"],
}


def _info(unit: str, description: str) -> dict[str, str]:
    return {"unit": unit, "frame": "PGSM", "description": description}


COLUMN_INFO: dict[str, dict[str, str]] = {
    **{f"{c}_pgsm_norm": _info("R_E", f"Normalized position in PGSM, {c}") for c in "XYZ"},
    **{f"B{c}_pgsm": _info("nT", f"Local magnetic field in PGSM, {c.upper()}") for c in "xyz"},
    **{f"V{c}_pgsm": _info("km/s", f"Local ion velocity in PGSM, {c.upper()}") for c in "xyz"},
    "mirrored": {"unit": "", "frame": "", "description":
                 "magnetosheath: IMF Bx sign of the row (bx_sign) differs from the measured sgn(Bx_imf); magnetosphere: tilt mirror psi -> -psi"},
    "bx_sign": {"unit": "", "frame": "", "description":
                "Sign of IMF Bx given to the row in PGSM (+1 or -1)"},
    "tilt_pgsm": {"unit": "deg", "frame": "", "description":
                  "Dipole tilt of the row in PGSM (negated on mirrored rows)"},
}


@dataclass(frozen=True)
class FrameSpec:
    region: str
    frame: str | None = None
    cone: tuple[float, float] | None = None
    clock: float | None = None
    clock_range: tuple[float, float] | None = None
    tilt: tuple[float, float] | None = None


def _bad(message: str) -> QueryError:
    return QueryError("bad_frame", message)


def _range(name: str, value: object, lo: float, hi: float, *, ordered: bool = True) -> tuple[float, float]:
    if isinstance(value, str | bytes):
        raise _bad(f"{name} must be [min, max] in degrees, got {value!r}.")
    try:
        a, b = (float(v) for v in value)  # pyright: ignore[reportGeneralTypeIssues]
    except (TypeError, ValueError):
        raise _bad(f"{name} must be [min, max] in degrees, got {value!r}.") from None
    if not (math.isfinite(a) and math.isfinite(b) and lo <= a <= hi and lo <= b <= hi):
        raise _bad(f"{name}=[{a}, {b}] must lie in [{lo:g}, {hi:g}] degrees.")
    if ordered and a > b:
        raise _bad(f"{name}=[{a}, {b}] must satisfy {lo:g} <= min <= max <= {hi:g} (degrees).")
    return a, b


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _clock_value(value: object) -> float:
    if not _is_number(value):
        raise _bad(f"clock in PGSM is one target value in degrees, got {value!r}.")
    c = float(value)  # pyright: ignore[reportArgumentType]
    if not (math.isfinite(c) and -360.0 <= c <= 360.0):
        raise _bad(f"clock={value!r} must be a finite angle in [-360, 360] degrees.")
    return c


def make_spec(
    region: str,
    frame: str | None,
    cone: Sequence[float] | None = None,
    clock: float | Sequence[float] | None = None,
    tilt: Sequence[float] | None = None,
    *,
    require_clock: bool = True,
) -> FrameSpec | None:
    """Validated frame and selection parameters (spec 2026-10-09 §4), or None when no frame
    and no selection is given. count() passes require_clock=False: the PGSM clock does not
    change which rows are selected."""
    region = str(getattr(region, "value", region))
    if frame is None and cone is None and clock is None and tilt is None:
        return None
    if frame is not None and frame not in FRAMES:
        raise _bad(f"frame={frame!r} is not a frame; use 'gsm', 'swi' or 'pgsm' (or no frame).")
    if frame == "pgsm":
        return _pgsm_spec(region, cone, clock, tilt, require_clock)
    if frame == "swi":
        if region != "magnetosheath":
            raise _bad("frame='swi' is only defined for the magnetosheath.")
        if clock is not None:
            raise _bad("clock makes no sense in SWI (the IMF is rotated to clock 90 deg); use cone.")
        if tilt is not None:
            raise _bad("tilt does not apply to the magnetosheath.")
        return FrameSpec(region, "swi", cone=None if cone is None else _range("cone", cone, 0.0, 180.0))
    # no frame, or frame="gsm": plain selections on the GSM values
    if region == "solar_wind" and (cone is not None or clock is not None):
        raise _bad("cone and clock need the IMF columns; region 'solar_wind' has no IMF columns.")
    if tilt is not None and region != "magnetosphere":
        raise _bad("tilt does not apply to the magnetosheath." if region == "magnetosheath"
                   else f"tilt is not served for region '{region}'.")
    if clock is not None and _is_number(clock):
        raise _bad("clock in GSM is a range [min, max] in degrees (e.g. [330, 30] wraps through north).")
    return FrameSpec(
        region, frame,
        cone=None if cone is None else _range("cone", cone, 0.0, 180.0),
        clock_range=None if clock is None else _range("clock", clock, -360.0, 360.0, ordered=False),
        tilt=None if tilt is None else _range("tilt", tilt, -35.0, 35.0),
    )


def _pgsm_spec(
    region: str, cone: object, clock: object, tilt: object, require_clock: bool
) -> FrameSpec:
    if region == "magnetosheath":
        if tilt is not None:
            raise _bad("tilt does not apply to the magnetosheath; PGSM there uses cone and clock.")
        if cone is None or (clock is None and require_clock):
            raise _bad("frame='pgsm' on the magnetosheath needs cone=[min, max] and clock=<degrees>.")
        return FrameSpec(
            region, "pgsm",
            cone=_range("cone", cone, 0.0, 180.0),
            clock=None if clock is None else _clock_value(clock),
        )
    if region == "magnetosphere":
        if cone is not None or clock is not None:
            raise _bad("cone and clock do not apply to the magnetosphere; PGSM there uses tilt.")
        if tilt is None:
            raise _bad("frame='pgsm' on the magnetosphere needs tilt=[min, max] in degrees.")
        return FrameSpec(region, "pgsm", tilt=_range("tilt", tilt, -35.0, 35.0))
    raise _bad(
        f"frame='pgsm' is not defined for region '{region}' (magnetosheath and magnetosphere only)."
    )


def imf_sign() -> pl.Expr:
    """s of the SWI basis: sgn(Bx_imf), else sgn(By_imf), else sgn(Bz_imf) (spok swi_base)."""
    bx, by, bz = pl.col("Bx_imf"), pl.col("By_imf"), pl.col("Bz_imf")
    return pl.when(bx != 0).then(bx.sign()).when(by != 0).then(by.sign()).otherwise(bz.sign())


def _imf_norm() -> pl.Expr:
    return (pl.col("Bx_imf") ** 2 + pl.col("By_imf") ** 2 + pl.col("Bz_imf") ** 2).sqrt()


def gsm_cone_deg() -> pl.Expr:
    """acos(Bx_imf/|B_imf|) in [0, 180] degrees; NaN when |B_imf| = 0."""
    return (pl.col("Bx_imf") / _imf_norm()).clip(-1.0, 1.0).arccos().degrees()


def gsm_clock_deg() -> pl.Expr:
    """atan2(By_imf, Bz_imf) in [0, 360) degrees; null when By_imf = Bz_imf = 0."""
    by, bz = pl.col("By_imf"), pl.col("Bz_imf")
    angle = (pl.arctan2(by, bz).degrees() + 360.0) % 360.0
    return pl.when((by != 0) | (bz != 0)).then(angle)


def swi_cone_deg() -> pl.Expr:
    """IMF cone in SWI: acos(s B_imf . X_swi / |B_imf|), X_swi = -V_sw/|V_sw|, in [0, 180]
    degrees (<= 90 except where aberration makes s B_imf . X_swi < 0); NaN when |B_imf| or
    |V_sw| is 0."""
    v = (pl.col("Vx_sw") ** 2 + pl.col("Vy_sw") ** 2 + pl.col("Vz_sw") ** 2).sqrt()
    b_dot_x = -(pl.col("Bx_imf") * pl.col("Vx_sw") + pl.col("By_imf") * pl.col("Vy_sw")
                + pl.col("Bz_imf") * pl.col("Vz_sw")) / v
    return (imf_sign() * b_dot_x / _imf_norm()).clip(-1.0, 1.0).arccos().degrees()


def tilt_deg() -> pl.Expr:
    return pl.col("tilt").degrees()


def _clock_in(clock: pl.Expr, c1: float, c2: float) -> pl.Expr:
    if c1 <= c2 and c2 - c1 >= 360.0:
        return clock.is_not_null() & clock.is_not_nan()
    a, b = c1 % 360.0, c2 % 360.0
    return clock.is_between(a, b) if a <= b else clock.is_not_nan() & ((clock >= a) | (clock <= b))


def selection(spec: FrameSpec) -> pl.Expr | None:
    """Row predicate of the non-PGSM selections (cone, clock range, tilt), None if none.
    Cone is measured from X_GSM, or from -V_sw when frame='swi'."""
    preds: list[pl.Expr] = []
    if spec.cone is not None and spec.frame != "pgsm":
        cone = swi_cone_deg() if spec.frame == "swi" else gsm_cone_deg()
        preds.append(cone.is_between(*spec.cone))
    if spec.clock_range is not None:
        preds.append(_clock_in(gsm_clock_deg(), *spec.clock_range))
    if spec.tilt is not None and spec.frame != "pgsm":
        preds.append(tilt_deg().is_between(*spec.tilt))
    return pl.all_horizontal(preds) if preds else None


def implied_flags(spec: FrameSpec) -> tuple[bool, bool]:
    """(sw_paired_only, normalized_only) a frame or selection implies (spec §4)."""
    msh = spec.region == "magnetosheath"
    sw_paired = (spec.frame == "swi" or (spec.frame == "pgsm" and msh)
                 or spec.cone is not None or spec.clock_range is not None)
    normalized = spec.frame in ("swi", "pgsm")
    return sw_paired, normalized


def required_columns(spec: FrameSpec) -> list[str]:
    """Served columns the selection and the PGSM transform read."""
    cols: list[str] = []
    if spec.frame == "pgsm":
        cols += REQUIRED_COLUMNS[spec.region]
    if spec.cone is not None or spec.clock_range is not None:
        cols += _IMF
    if spec.cone is not None and spec.frame in ("swi", "pgsm"):
        cols += _VSW
    if spec.tilt is not None:
        cols.append("tilt")
    return list(dict.fromkeys(cols))


def frame_of_column(name: str) -> str:
    """'gsm' or 'swi' for a served vector column, '' for scalars and unknown names."""
    info = COLUMNS.get(name)
    return info.frame.lower() if info is not None and info.frame in ("GSM", "SWI") else ""


def frame_columns(region: str, frame: str | None, served: Sequence[str]) -> list[str]:
    """Columns get_data returns for a frame (spec §5): every served column without a frame;
    otherwise the scalars plus the frame's vector columns, in served order; for PGSM the
    scalars plus OUTPUT_COLUMNS."""
    if frame is None:
        return list(served)
    scalars = [c for c in served if frame_of_column(c) == ""]
    if frame == "pgsm":
        return [*scalars, *OUTPUT_COLUMNS[region]]
    return [c for c in served if frame_of_column(c) in ("", frame)]


def spec_from_params(
    region: str, raw: Mapping[str, str | None], *, for_count: bool
) -> FrameSpec | None:
    """FrameSpec from the server query string (frame, cone_min/max, clock_min/max,
    tilt_deg_min/max). /data refuses frame=pgsm (the transform runs in the client)."""

    def pair(lo: str, hi: str) -> tuple[str | None, str | None] | None:
        a, b = raw.get(lo), raw.get(hi)
        return None if a is None and b is None else (a, b)

    tilt = pair("tilt_deg_min", "tilt_deg_max")
    # frame=pgsm: the client itself sends tilt_min/max (radians) as a pre-filter next to tilt_deg_*.
    if (tilt is not None and raw.get("frame") != "pgsm"
            and ("tilt_min" in raw or "tilt_max" in raw)):
        raise _bad("tilt_deg_min/max (degrees) cannot be combined with tilt_min/tilt_max (radians).")
    spec = make_spec(
        region, raw.get("frame"),
        cone=pair("cone_min", "cone_max"),  # pyright: ignore[reportArgumentType]
        clock=pair("clock_min", "clock_max"),  # pyright: ignore[reportArgumentType]
        tilt=tilt,  # pyright: ignore[reportArgumentType]
        require_clock=False,
    )
    if spec is not None and spec.frame == "pgsm" and not for_count:
        raise _bad("frame='pgsm' is computed by the client (space_mango); /data serves frames "
                   "'gsm' and 'swi', or no frame.")
    return spec


def candidates(spec: FrameSpec) -> list[tuple[pl.Expr, int]]:
    """PGSM (row predicate, sign) pairs; each row passing a predicate gives one output row.
    Magnetosheath: sign = target sgn(Bx_imf); with f the SWI cone, output cone f (+1) or
    180 - f (-1). Magnetosphere: +1 = the row as measured, -1 = its tilt mirror."""
    if spec.region == "magnetosheath":
        if spec.cone is None:
            raise _bad("PGSM on the magnetosheath needs cone=[min, max].")
        a, b = spec.cone
        f = swi_cone_deg()
        return [(f.is_between(a, b), 1), ((180.0 - f).is_between(a, b), -1)]
    if spec.tilt is None:
        raise _bad("PGSM on the magnetosphere needs tilt=[min, max].")
    t1, t2 = spec.tilt
    psi = tilt_deg()
    return [(psi.is_between(t1, t2), 1), ((-psi).is_between(t1, t2), -1)]


def _msh_rows(df: pl.DataFrame, sign: int, clock: float) -> pl.DataFrame:
    """Eqs 2.19-2.20 with the target sgn(Bx_imf) = sign and delta = clock - 90 deg.
    Written as the linear rotation of (Y, Z) by delta, azimuth from +Z towards +Y (eq 2.21):
    rho sin(a + d) = Y cos d + Z sin d, rho cos(a + d) = Z cos d - Y sin d."""
    d = math.radians(clock - 90.0)
    cos_d, sin_d = math.cos(d), math.sin(d)
    s = float(sign)

    def rot(y: pl.Expr, z: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
        return y * cos_d + z * sin_d, z * cos_d - y * sin_d

    # sign = -1 is the Y mirror before the rotation: positions and velocities (polar)
    # (X, -Y, Z); magnetic field (axial) (-Bx, By, -Bz).
    y, z = rot(s * pl.col("Y_swi_norm"), pl.col("Z_swi_norm"))
    vy, vz = rot(s * pl.col("Vy_swi"), pl.col("Vz_swi"))
    by, bz = rot(pl.col("By_swi"), s * pl.col("Bz_swi"))
    return df.with_columns(
        X_pgsm_norm=pl.col("X_swi_norm"), Y_pgsm_norm=y, Z_pgsm_norm=z,
        Bx_pgsm=s * pl.col("Bx_swi"), By_pgsm=by, Bz_pgsm=bz,
        Vx_pgsm=pl.col("Vx_swi"), Vy_pgsm=vy, Vz_pgsm=vz,
        # mirrored: the row's IMF Bx sign (bx_sign) differs from the measured sgn(Bx_imf),
        # i.e. the sample was moved to the other Parker-spiral orientation by symmetry.
        # Note: SWI already applies B -> -B to Bx < 0 measurements, and bx_sign = -1 rows
        # have positions reflected Y -> -Y relative to SWI (eq 2.19).
        mirrored=imf_sign() != s,
        bx_sign=pl.lit(sign, dtype=pl.Int8),
    )


def to_pgsm(df: pl.DataFrame, spec: FrameSpec) -> pl.DataFrame:
    """Select and transform df (which must hold REQUIRED_COLUMNS[spec.region]).
    Adds OUTPUT_COLUMNS[spec.region]; keeps every input column as measured."""
    missing = [c for c in REQUIRED_COLUMNS[spec.region] if c not in df.columns]
    if missing:
        raise _bad(f"PGSM ({spec.region}) needs the columns {missing}.")
    parts: list[pl.DataFrame] = []
    for keep, sign in candidates(spec):
        rows = df.filter(keep)
        if spec.region == "magnetosheath":
            if spec.clock is None:
                raise _bad("PGSM on the magnetosheath needs clock=<degrees>.")
            parts.append(_msh_rows(rows, sign, spec.clock))
        else:
            parts.append(_msp_rows(rows, sign))
    out = pl.concat(parts, how="vertical")
    return out.sort("Time", maintain_order=True) if "Time" in out.columns else out


def _msp_rows(df: pl.DataFrame, sign: int) -> pl.DataFrame:
    """sign = +1: the row as measured. sign = -1: eqs 2.14-2.16, the tilt mirror psi -> -psi,
    i.e. a rotation by pi about X together with B -> -B: positions (X, -Y, -Z),
    B (-Bx, By, Bz), V (Vx, -Vy, -Vz) (B -> -B does not act on V)."""
    m = float(sign)
    return df.with_columns(
        X_pgsm_norm=pl.col("X_gsm_norm"), Y_pgsm_norm=m * pl.col("Y_gsm_norm"),
        Z_pgsm_norm=m * pl.col("Z_gsm_norm"),
        Bx_pgsm=m * pl.col("Bx"), By_pgsm=pl.col("By"), Bz_pgsm=pl.col("Bz"),
        Vx_pgsm=pl.col("Vx"), Vy_pgsm=m * pl.col("Vy"), Vz_pgsm=m * pl.col("Vz"),
        mirrored=pl.lit(sign == -1),
        tilt_pgsm=m * pl.col("tilt").degrees(),
    )
