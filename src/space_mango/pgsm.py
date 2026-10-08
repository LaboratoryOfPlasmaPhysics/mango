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
