# Frames in space_mango: GSM, SWI, PGSM — design (2026-10-09)

Status: design approved in conversation with N. Aunai, 2026-10-09; this spec awaits review.
Extends and partly supersedes `2026-10-08-pgsm-design.md` (PGSM, merged on `main` at
`60d034d`, not released). Where the two disagree, this spec wins. Everything here ships in
0.3.0 together with PGSM.

## 1. Goal

Let users ask for MANGO data in one coordinate frame — GSM, SWI or PGSM — and select samples
by IMF cone angle, IMF clock angle and dipole tilt with plain `[min, max]` ranges, with each
angle measured in the frame the user asked for.

## 2. Decisions (from the conversation)

- `frame=` takes `"gsm"`, `"swi"` or `"pgsm"`. It decides **which vector columns are
  returned** as well as how selections are interpreted.
- Without `frame`, every served column is returned (as in 0.2.0), and `cone`/`clock`/`tilt`
  are plain selections on the GSM values — GSM is the only frame without assumptions.
- GSM: `cone`, `clock`, `tilt` are `[min, max]` ranges that select rows; nothing is
  transformed.
- SWI: magnetosheath only (an error elsewhere). The served SWI data, selected by a `cone`
  range. `clock` makes no sense in SWI and is refused.
- PGSM: unchanged from the PGSM spec — magnetosheath: `cone` range + one `clock` value
  (rotation of SWI data); magnetosphere: `tilt` range + the tilt symmetry.
- The cone angle is measured from each frame's own X axis: X_GSM in GSM, −V_sw (X_SWI) in SWI
  and PGSM. This replaces the PGSM spec's "cone from X_GSM in PGSM".

## 3. Angle definitions

All angles in the API are in degrees.

- **GSM cone** = acos(Bx_imf/|B_imf|) ∈ [0°, 180°], 0° = IMF sunward. NaN (row never
  selected) when |B_imf| = 0.
- **GSM clock** = atan2(By_imf, Bz_imf) mod 360° ∈ [0°, 360°), 0° = northward, 90° = +Y.
  Undefined (row never selected) when By_imf = Bz_imf = 0.
- **SWI cone** = acos(s·B_imf·X̂_SWI/|B_imf|) ∈ [0°, 180°], with X̂_SWI = −V_sw/|V_sw| (served
  `V*_sw`) and s = sgn(Bx_imf) (fallbacks sgn(By_imf), then sgn(Bz_imf)), as in the SWI
  construction. It is the cone of the IMF in the SWI frame; it is ≤ 90° except for a few
  rows where aberration makes s·B_imf·X̂_SWI < 0. NaN when |B_imf| = 0 or |V_sw| = 0.
- **Tilt** = served `tilt` (radians) converted to degrees; magnetosphere only.
- On the docs sample, −V_sw is a median 5.4° from X_GSM (99th percentile 13°) and the SWI
  cone differs from the GSM cone by a median 2.6° (max 14°): the same `cone` range selects
  slightly different samples in GSM and in SWI/PGSM. Documented.

## 4. User API

```python
mango.get_data("magnetosheath", cone=[20, 40], clock=[330, 30])              # all columns, GSM selections
mango.get_data("magnetosphere", frame="gsm", tilt=[10, 15], cone=[0, 30])    # GSM columns only
mango.get_data("magnetosheath", frame="swi", cone=[20, 40])                  # SWI columns only
mango.get_data("magnetosheath", frame="pgsm", cone=[80, 100], clock=180)     # PGSM (cone from -V_sw)
mango.get_data("magnetosphere", frame="pgsm", tilt=[10, 15])                 # PGSM tilt symmetry
mango.count(...)                                                             # same arguments, exact rows
```

Parameters by frame (✓ allowed, — refused with an error):

| frame | region | cone `[min,max]` | clock | tilt `[min,max]` |
|---|---|---|---|---|
| none / `"gsm"` | magnetosheath | ✓ GSM cone [0,180] | ✓ range | — |
| none / `"gsm"` | magnetosphere | ✓ GSM cone [0,180] | ✓ range | ✓ [−35,35] |
| none / `"gsm"` | solar_wind | — (no IMF columns) | — | — |
| `"swi"` | magnetosheath | ✓ SWI cone [0,180] | — | — |
| `"swi"` | magnetosphere, solar_wind | frame refused | | |
| `"pgsm"` | magnetosheath | ✓ required, SWI cone [0,180] | ✓ required, one value | — |
| `"pgsm"` | magnetosphere | — | — | ✓ required, with symmetry |
| `"pgsm"` | solar_wind | frame refused | | |

- Clock range `[c1, c2]`: values in [−360, 360], taken mod 360. If c1 ≤ c2 after that, rows
  with c1 ≤ clock ≤ c2; otherwise the range wraps through north: clock ≥ c1 or clock ≤ c2.
  A range spanning 360° or more selects every row with a defined clock.
- In GSM a single clock value is an error ("clock in GSM is a range [min, max]"); in PGSM a
  range is an error ("clock in PGSM is one target value").
- `frame="swi"` and `frame="pgsm"` (magnetosheath) imply `sw_paired_only` and
  `normalized_only` (SWI columns are null otherwise). `frame="pgsm"` (magnetosphere)
  implies `normalized_only`. Any `cone`/`clock` selection implies `sw_paired_only`.
- The legacy `tilt_min`/`tilt_max` range filter (radians, magnetosphere) stays for 0.2
  compatibility; combining it with `tilt=` is an error.
- All other arguments (`spacecraft`, `start`/`stop`, `columns`, range filters such as
  `pd_sw_max`, `limit`, `cache`) keep their meaning. Range filters always act on the
  measured (served) values, before any transform.

## 5. Columns returned

| frame | vector columns | plus |
|---|---|---|
| none | every served column (as in 0.2.0) | — |
| `"gsm"` | `Bx/By/Bz`, `Vx/Vy/Vz`, `X/Y/Z_gsm`, `X/Y/Z_gsm_norm`, `B*_imf`, `V*_sw` | scalars |
| `"swi"` | `B*_swi`, `V*_swi`, `X/Y/Z_swi_norm` | scalars |
| `"pgsm"` | `X/Y/Z_pgsm_norm`, `B*_pgsm`, `V*_pgsm`, `mirrored`, `bx_sign` (msh) / `tilt_pgsm` (msp) | scalars |

- **Scalars** (frame-free, returned in every frame where served): `Time`, `SC`, `Np`, `Tp`,
  `SW_pairing`, `Norma_pos`, `Np_sw`, `Tp_sw`, `Pd_sw`, `Beta_sw`, `Ma_sw`, `R_mp`, `R_bs`,
  `R_norm`, `tilt`.
- In SWI and PGSM the GSM IMF and solar-wind vectors are not returned (they are GSM). The
  IMF expressed in SWI/PGSM is not served; adding it is possible later (computable from the
  served columns) but out of scope.
- `columns=` with a frame: every listed column must belong to that frame's vector columns or
  to the scalars; otherwise a `FrameError` names the frame the column belongs to. PGSM
  output columns may be listed with `frame="pgsm"` (always returned anyway).
- The column sets live in one place (the frames module, §6) and are derived from the column
  catalog's `frame` field where possible, so a new served column is classified once.

## 6. Where things run

- New module `src/space_mango/frames.py`, absorbing `pgsm.py`: parameter validation
  (`make_spec` → a `FrameSpec` with `frame`, `cone`, `clock`, `tilt`), angle expressions
  (GSM cone, GSM clock, SWI cone, tilt), selection predicates (`candidates`), column sets per
  frame, and the PGSM transform. Pure polars, no I/O, no numpy.
- Selections (cone/clock/tilt, all frames) are row predicates built by the frames module
  and applied by the shared filtering path, so the cache path, the server `/data` and the
  server `/count` select the same rows. PGSM doubling stays: `/count` sums the PGSM
  candidates as today.
- Column projection by frame is applied by the server on `/data` (so non-Python consumers
  get it) and by the client on the cache path, with the same column sets.
- The PGSM transform stays in the client; `/data` with `frame=pgsm` is refused (400,
  message: computed by the client).
- Server parameters (query string), shared by `/data` and `/count`: `frame`, `cone_min`,
  `cone_max`, `clock_min`, `clock_max`, `tilt_deg_min`, `tilt_deg_max` (degrees). They
  replace the unreleased `pgsm_cone_*`/`pgsm_tilt_*`. `/count` with `frame=pgsm` also takes
  them. The feature flag in `/api/v1/dataset` becomes `"frames"` (replacing the unreleased
  `"pgsm_count"`).
- Errors: one `FrameError(MangoError, ValueError)` (code `bad_frame`), replacing the
  unreleased `PgsmError`/`bad_pgsm`.
- Client compatibility: against a server without the `"frames"` feature, the cache path
  still works (selection and projection are local); `cache=False`/`limit` queries and
  `count()` with any frame/cone/clock/tilt raise `ServerError` ("needs a 0.3 server").

## 7. Tests

- Angle expressions against a numpy oracle: GSM cone/clock, SWI cone (incl. a row with
  s·B_imf·X̂_SWI < 0), NaN/undefined rows never selected.
- Clock ranges: plain, wrapping through north, ≥ 360° span, values given negative.
- Per frame × region: allowed/refused parameters (table §4), required ones, error messages.
- Column sets: exact output columns per frame and region; `columns=` from another frame
  raises; default (no frame) unchanged from 0.2.0.
- PGSM: existing tests kept; selection now on the SWI cone: the transformed IMF cone equals
  the SWI cone of the row (s = +1) or its supplement (s = −1), exactly.
- Same rows from cache path, server `/data` and `/count` for each frame.
- Real docs sample: SWI cone vs a numpy recomputation; `frame="swi"` returns the served SWI
  values unchanged.

## 8. Docs and release

- User guide: a "Frames" section (GSM / SWI / PGSM, the parameter table, column sets, angle
  definitions with the measured GSM–SWI cone differences) replacing the current PGSM-only
  wording; example notebook updated; changelog 0.3. Release order unchanged: deploy server,
  then tag v0.3.0.
- Open, unchanged: thesis eq 2.17 (magnetosheath doubling) awaits B. Michotte de Welle; the
  thesis Y reflection for bx_sign = −1 is kept.
