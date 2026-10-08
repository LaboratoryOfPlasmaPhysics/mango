"""PGSM frame (B. Michotte de Welle, PhD thesis 2024, sections 2.7.3-2.7.4, eqs 2.13-2.20).

Magnetosheath: the served SWI columns, selected by IMF cone angle and rotated about X to a
target IMF clock angle (eqs 2.19-2.20). Magnetosphere: GSM plus the dipole-tilt symmetry
psi -> -psi (eqs 2.14-2.16). Pure polars: used by the client (transform, cache-path counts)
and by the server (/count). Angles are in degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from space_mango.errors import QueryError

FRAMES = ("pgsm",)
THESIS = (
    "B. Michotte de Welle (2024), PhD thesis, https://theses.hal.science/tel-04661957, "
    "sections 2.7.3-2.7.4"
)

_POS = ["X_pgsm_norm", "Y_pgsm_norm", "Z_pgsm_norm"]
_VEC = ["Bx_pgsm", "By_pgsm", "Bz_pgsm", "Vx_pgsm", "Vy_pgsm", "Vz_pgsm"]
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "magnetosheath": [
        "Bx_imf", "By_imf", "Bz_imf",
        "Bx_swi", "By_swi", "Bz_swi", "Vx_swi", "Vy_swi", "Vz_swi",
        "X_swi_norm", "Y_swi_norm", "Z_swi_norm",
    ],
    "magnetosphere": [
        "tilt", "Bx", "By", "Bz", "Vx", "Vy", "Vz", "X_gsm_norm", "Y_gsm_norm", "Z_gsm_norm",
    ],
}
"""Input columns of the transform, fetched even when not asked for."""
OUTPUT_COLUMNS: dict[str, list[str]] = {
    "magnetosheath": [*_POS, *_VEC, "mirrored", "bx_sign"],
    "magnetosphere": [*_POS, *_VEC, "mirrored", "tilt_pgsm"],
}


@dataclass(frozen=True)
class PgsmSpec:
    region: str
    cone: tuple[float, float] | None = None
    clock: float | None = None
    tilt: tuple[float, float] | None = None


def _bad(message: str) -> QueryError:
    return QueryError("bad_pgsm", message)


def _range(name: str, value: object, lo: float, hi: float) -> tuple[float, float]:
    if isinstance(value, str | bytes):
        raise _bad(f"{name} must be [min, max] in degrees, got {value!r}.")
    try:
        a, b = (float(v) for v in value)  # pyright: ignore[reportGeneralTypeIssues]
    except (TypeError, ValueError):
        raise _bad(f"{name} must be [min, max] in degrees, got {value!r}.") from None
    if not (math.isfinite(a) and math.isfinite(b) and lo <= a <= b <= hi):
        raise _bad(f"{name}=[{a}, {b}] must satisfy {lo:g} <= min <= max <= {hi:g} (degrees).")
    return a, b


def _clock(value: object) -> float:
    try:
        c = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        raise _bad(f"clock must be a number of degrees, got {value!r}.") from None
    if not (math.isfinite(c) and -360.0 <= c <= 360.0):
        raise _bad(f"clock={value!r} must be a finite angle in [-360, 360] degrees.")
    return c


def make_spec(
    region: str,
    frame: str | None,
    cone: Sequence[float] | None = None,
    clock: float | None = None,
    tilt: Sequence[float] | None = None,
    *,
    require_clock: bool = True,
) -> PgsmSpec | None:
    """Validated PGSM parameters, or None when frame is None. count() passes
    require_clock=False: the clock angle does not change which rows are selected."""
    region = str(getattr(region, "value", region))
    given = [n for n, v in (("cone", cone), ("clock", clock), ("tilt", tilt)) if v is not None]
    if frame is None:
        if given:
            raise _bad(f"{', '.join(given)} need frame='pgsm'.")
        return None
    if frame not in FRAMES:
        raise _bad(f"frame={frame!r} is not supported; use frame='pgsm'.")
    if region == "magnetosheath":
        if tilt is not None:
            raise _bad("tilt does not apply to the magnetosheath; PGSM there uses cone and clock.")
        if cone is None or (clock is None and require_clock):
            raise _bad("frame='pgsm' on the magnetosheath needs cone=[min, max] and clock=<degrees>.")
        return PgsmSpec(
            region,
            cone=_range("cone", cone, 0.0, 180.0),
            clock=None if clock is None else _clock(clock),
        )
    if region == "magnetosphere":
        if cone is not None or clock is not None:
            raise _bad("cone and clock do not apply to the magnetosphere; PGSM there uses tilt.")
        if tilt is None:
            raise _bad("frame='pgsm' on the magnetosphere needs tilt=[min, max] in degrees.")
        return PgsmSpec(region, tilt=_range("tilt", tilt, -35.0, 35.0))
    raise _bad(
        f"frame='pgsm' is not defined for region '{region}' (magnetosheath and magnetosphere only)."
    )


def imf_sign() -> pl.Expr:
    """s of the SWI basis: sgn(Bx_imf), else sgn(By_imf), else sgn(Bz_imf) (spok swi_base)."""
    bx, by, bz = pl.col("Bx_imf"), pl.col("By_imf"), pl.col("Bz_imf")
    return pl.when(bx != 0).then(bx.sign()).when(by != 0).then(by.sign()).otherwise(bz.sign())


def folded_cone_deg() -> pl.Expr:
    """IMF cone angle once in SWI (Bx > 0): acos(|Bx_imf|/|B_imf|) in [0, 90] degrees.
    NaN when |B_imf| = 0, so such rows are never selected."""
    b = (pl.col("Bx_imf") ** 2 + pl.col("By_imf") ** 2 + pl.col("Bz_imf") ** 2).sqrt()
    return (pl.col("Bx_imf").abs() / b).clip(0.0, 1.0).arccos().degrees()


def candidates(spec: PgsmSpec) -> list[tuple[pl.Expr, int]]:
    """(row predicate, sign) pairs; each row passing a predicate gives one output row.
    Magnetosheath: sign = target sgn(Bx_imf), output cone f (+1) or 180 - f (-1).
    Magnetosphere: +1 = the row as measured, -1 = its tilt mirror."""
    if spec.region == "magnetosheath":
        assert spec.cone is not None
        a, b = spec.cone
        f = folded_cone_deg()
        return [(f.is_between(a, b), 1), ((180.0 - f).is_between(a, b), -1)]
    assert spec.tilt is not None
    t1, t2 = spec.tilt
    psi = pl.col("tilt").degrees()
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


def to_pgsm(df: pl.DataFrame, spec: PgsmSpec) -> pl.DataFrame:
    """Select and transform df (which must hold REQUIRED_COLUMNS[spec.region]).
    Adds OUTPUT_COLUMNS[spec.region]; keeps every input column as measured."""
    missing = [c for c in REQUIRED_COLUMNS[spec.region] if c not in df.columns]
    if missing:
        raise _bad(f"PGSM ({spec.region}) needs the columns {missing}.")
    parts: list[pl.DataFrame] = []
    for keep, sign in candidates(spec):
        rows = df.filter(keep)
        if spec.region == "magnetosheath":
            assert spec.clock is not None, "clock is required to transform"
            parts.append(_msh_rows(rows, sign, spec.clock))
        else:
            parts.append(_msp_rows(rows, sign))
    out = pl.concat(parts, how="vertical")
    return out.sort("Time", maintain_order=True) if "Time" in out.columns else out


def _msp_rows(df: pl.DataFrame, sign: int) -> pl.DataFrame:
    raise NotImplementedError("magnetosphere PGSM: Task 4")
