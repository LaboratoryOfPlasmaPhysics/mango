# User guide

```python
import space_mango as mango
```

## Regions

| Region | Definition |
|---|---|
| `magnetosphere` | Inside the magnetopause: closed-field-line magnetosphere sampled by the spacecraft (machine-learning region classification, Nguyen et al. 2022). |
| `magnetosheath` | Between the bow shock and the magnetopause: shocked solar wind (machine-learning region classification, Nguyen et al. 2022). |
| `solar_wind` | Upstream of the bow shock: pristine solar wind measured in situ by the same spacecraft (machine-learning region classification, Nguyen et al. 2022). |

`mango.regions()` lists them; `mango.magnetosheath` (also `mango.magnetosphere`,
`mango.solar_wind`) is a region object whose `get_data` has a generated, documented
signature (tab-completion, `help()`).

## Columns and frames

`mango.describe("magnetosheath")` returns this information as a table for one region.
"all" means all three regions. A dash means no unit / no frame.

| Column | Regions | Unit | Frame | Description |
|---|---|---|---|---|
| `Time` | all | - | - | Sample time (UTC assumed), on a 5 s grid |
| `SC` | all | - | - | Spacecraft: THA–THE (THEMIS), C1, C3 (Cluster), MMS, DS1 (Double Star) |
| `Bx` | all | nT | GSM | Local magnetic field, X |
| `By` | all | nT | GSM | Local magnetic field, Y |
| `Bz` | all | nT | GSM | Local magnetic field, Z |
| `Np` | all | cm⁻³ | - | Local ion density |
| `Vx` | all | km/s | GSM | Local ion bulk velocity, X |
| `Vy` | all | km/s | GSM | Local ion bulk velocity, Y |
| `Vz` | all | km/s | GSM | Local ion bulk velocity, Z |
| `Tp` | all | K | - | Local ion temperature (T∥ + 2T⊥)/3 |
| `X_gsm` | all | R_E | GSM | Spacecraft position, X |
| `Y_gsm` | all | R_E | GSM | Spacecraft position, Y |
| `Z_gsm` | all | R_E | GSM | Spacecraft position, Z |
| `SW_pairing` | magnetosphere, magnetosheath | - | - | True when an upstream solar-wind sample is associated to this row |
| `Bx_imf` | magnetosphere, magnetosheath | nT | GSM | Upstream IMF Bx paired with this row |
| `By_imf` | magnetosphere, magnetosheath | nT | GSM | Upstream IMF By paired with this row |
| `Bz_imf` | magnetosphere, magnetosheath | nT | GSM | Upstream IMF Bz paired with this row |
| `Np_sw` | magnetosphere, magnetosheath | cm⁻³ | - | Upstream proton density |
| `Vx_sw` | magnetosphere, magnetosheath | km/s | GSM | Upstream velocity X; negative (anti-sunward), not a speed |
| `Vy_sw` | magnetosphere, magnetosheath | km/s | GSM | Upstream velocity Y |
| `Vz_sw` | magnetosphere, magnetosheath | km/s | GSM | Upstream velocity Z |
| `Tp_sw` | magnetosphere, magnetosheath | K | - | Upstream proton temperature |
| `Pd_sw` | magnetosphere, magnetosheath | nPa | - | Upstream dynamic pressure |
| `Beta_sw` | magnetosphere, magnetosheath | - | - | Upstream plasma beta |
| `Ma_sw` | magnetosphere, magnetosheath | - | - | Upstream Alfvén Mach number |
| `tilt` | magnetosphere | rad | - | Dipole tilt angle (positive near June solstice) |
| `R_mp` | magnetosphere, magnetosheath | R_E | radial | Magnetopause distance along the spacecraft direction |
| `R_bs` | magnetosheath | R_E | radial | Bow-shock distance along the spacecraft direction |
| `R_norm` | magnetosphere, magnetosheath | - | - | magnetosheath: Fractional position from magnetopause (0) to bow shock (1): (\|r\| - R_mp) / (R_bs - R_mp); magnetosphere: Fractional position from Earth (0) to magnetopause (1): \|r\| / R_mp |
| `Norma_pos` | magnetosphere, magnetosheath | - | - | True when the row has a normalized position (*_norm columns) |
| `X_gsm_norm` | magnetosphere, magnetosheath | R_E | GSM | Normalized position, X: radially rescaled between fixed average boundaries |
| `Y_gsm_norm` | magnetosphere, magnetosheath | R_E | GSM | Normalized position, Y: radially rescaled between fixed average boundaries |
| `Z_gsm_norm` | magnetosphere, magnetosheath | R_E | GSM | Normalized position, Z: radially rescaled between fixed average boundaries |
| `Bx_swi` | magnetosheath | nT | SWI | Local magnetic field in the SWI frame, X |
| `By_swi` | magnetosheath | nT | SWI | Local magnetic field in the SWI frame, Y |
| `Bz_swi` | magnetosheath | nT | SWI | Local magnetic field in the SWI frame, Z |
| `Vx_swi` | magnetosheath | km/s | SWI | Local ion velocity in the SWI frame, X (aberration-corrected) |
| `Vy_swi` | magnetosheath | km/s | SWI | Local ion velocity in the SWI frame, Y (aberration-corrected) |
| `Vz_swi` | magnetosheath | km/s | SWI | Local ion velocity in the SWI frame, Z (aberration-corrected) |
| `X_swi_norm` | magnetosheath | R_E | SWI | Normalized position in the SWI frame, X |
| `Y_swi_norm` | magnetosheath | R_E | SWI | Normalized position in the SWI frame, Y |
| `Z_swi_norm` | magnetosheath | R_E | SWI | Normalized position in the SWI frame, Z |

## Filters

Pass any filter as `<name>_min=` and/or `<name>_max=` keyword arguments to `get_data` or
`count`. A filter used in a region where it does not apply raises `MangoFilterError`.
Values are numbers (int, float or numpy scalars); NaN and infinity are refused.
`spacecraft=` and `columns=` take one name (`"THA"`) or a list; duplicates are dropped.

| Filter (with `_min` / `_max`) | Column | Regions | Unit | Description |
|---|---|---|---|---|
| `bz_imf` | `Bz_imf` | magnetosphere, magnetosheath | nT | IMF Bz — southward (<0) drives reconnection |
| `by_imf` | `By_imf` | magnetosphere, magnetosheath | nT | IMF By — controls reconnection geometry and asymmetry |
| `bx_imf` | `Bx_imf` | magnetosphere, magnetosheath | nT | IMF Bx — cone angle / Parker spiral orientation |
| `pd_sw` | `Pd_sw` | magnetosphere, magnetosheath | nPa | Solar wind dynamic pressure |
| `np_sw` | `Np_sw` | magnetosphere, magnetosheath | cm⁻³ | Solar wind proton density |
| `tp_sw` | `Tp_sw` | magnetosphere, magnetosheath | K | Solar wind proton temperature |
| `vx_sw` | `Vx_sw` | magnetosphere, magnetosheath | km/s | Solar wind velocity X (GSM); negative (anti-sunward), so faster wind is more negative |
| `beta_sw` | `Beta_sw` | magnetosphere, magnetosheath | - | Solar wind plasma beta |
| `ma_sw` | `Ma_sw` | magnetosphere, magnetosheath | - | Solar wind Alfvén Mach number |
| `tilt` | `tilt` | magnetosphere | rad | Dipole tilt angle (positive near June solstice) |
| `x_gsm` | `X_gsm` | all | R_E | X GSM coordinate |
| `y_gsm` | `Y_gsm` | all | R_E | Y GSM coordinate |
| `z_gsm` | `Z_gsm` | all | R_E | Z GSM coordinate |
| `d_msp` | `R_norm` | magnetosphere | - | Relative distance Earth(0)–magnetopause(1): \|r\| / R_mp |
| `d_msh` | `R_norm` | magnetosheath | - | Relative distance magnetopause(0)–bow shock(1): (\|r\| - R_mp) / (R_bs - R_mp) |
| `np` | `Np` | all | cm⁻³ | Local plasma density |
| `tp` | `Tp` | all | K | Local plasma temperature |
| `bz` | `Bz` | all | nT | Local Bz (GSM) |

## Time conventions

- Times are UTC, on a 5 s grid.
- `start` is inclusive, `stop` is exclusive. `stop` must be after `start`
  (else `TimeParseError`).
- `start`/`stop` accept strings (`"2016-01"`, `"2017-01-12T10:00"`), dates or datetimes.
  `time_min`/`time_max` are deprecated aliases and emit a `FutureWarning`.
- `mango.timeline(sc, start, stop)` returns all samples of one spacecraft across regions
  and is limited to 31 days.

## Choosing the server

The module-level functions use the public MANGO server. Set `$SPACE_MANGO_URL` to point them
at another server (for example a self-hosted one), or pass `base_url=` to
`MangoClient(...)`.

## Cache

`get_data` stores results on disk as per-column monthly Parquet fragments, so re-running a
notebook does not download again.

- Location: `$SPACE_MANGO_CACHE_DIR`, or the platform user cache directory
  (`~/.cache/space-mango` on Linux).
- Size cap: `$SPACE_MANGO_CACHE_SIZE` bytes (default 10 GB); least-recently-used fragments
  are evicted and re-fetched when needed.
- `mango.cache.info()` reports location, number of files, size and cap;
  `mango.cache.clear()` empties it.
- Layout: `<cache_dir>/<version>-<checksum>/<region>/SC=<sc>/<column>/<YYYY-MM>.parquet`,
  where `<checksum>` is the first 12 hex characters of the server's schema checksum
  (`mango.dataset_info()["schema_checksum"]`). A new dataset version or schema starts a new
  directory; old ones are never mixed in.
- Downloads: only the missing (month, column) fragments are fetched, at most 6 months of one
  spacecraft per request. Adding a column to a cached query downloads only that column.
- `MangoClient(cache=False)` sends every request to the server and writes nothing to the
  cache directory.
- If the cache directory cannot be written, the client warns once (`UserWarning`) and
  continues without caching; set `SPACE_MANGO_CACHE_DIR` to a writable directory.
- `MangoClient(offline=True)` serves only from the cache, including the metadata stored
  under `<cache_dir>/<version>-<checksum>/_meta`. It needs no network and raises
  `CacheMissError` as soon as anything it needs (data or metadata) is not cached.

**Two sizes in `count()`.** `mango.count(region, ...)` downloads nothing and returns:

- `n_rows`, `est_mb`: rows matching the query and their size after server-side filtering —
  what a `cache=False` call transfers.
- `download_mb_estimate`: what the default cached `get_data` would download — whole months
  of every needed column (requested, filter and flag columns), unfiltered, minus fragments
  already in the cache. It is usually larger than `est_mb` on a cold cache (filters run
  locally) and drops to 0 once everything is cached. It equals `est_mb` when the client
  does not cache.

**For server operators:** the cache trusts `MANGO_DATASET_VERSION`. Changing the served data
requires bumping `MANGO_DATASET_VERSION`; a schema change alone also starts a new cache
directory, but changed values with the same schema and version would not be re-downloaded.

## Errors

All inherit from `MangoError`.

- `UnknownRegionError`: the region name does not exist.
- `UnknownSpacecraftError`: the spacecraft is not in the region (with a "did you mean" hint, e.g. `MMS1` vs `MMS`).
- `UnknownColumnError`: the requested column is not served for the region.
- `MangoFilterError`: unknown filter, or a filter that applies to another region.
- `TimeParseError`: `start`/`stop` could not be interpreted.
- `ServerError`: the server answered with an error (including 404 from a server older than 0.2).
- `CacheMissError`: `offline=True` and the data or metadata is not in the cache.

## Citing

`mango.cite()` (or `result.cite()`) returns BibTeX for the dataset version served.

## Frames

MANGO data can be asked for in one of three coordinate frames with `frame=`: `"gsm"`,
`"swi"` (magnetosheath only) or `"pgsm"` (the "pseudo-GSM" of Michotte de Welle 2024, PhD
thesis, sections 2.7.3–2.7.4, <https://theses.hal.science/tel-04661957>). The frame decides
which vector columns are returned and how `cone`, `clock` and `tilt` are read. Without
`frame`, every served column is returned (as in 0.2) and the selections act on the GSM
values.

```python
mango.get_data("magnetosheath", cone=[20, 40], clock=[330, 30])             # all columns, GSM selections
mango.get_data("magnetosphere", frame="gsm", tilt=[10, 15], cone=[0, 30])   # GSM columns only
mango.get_data("magnetosheath", frame="swi", cone=[20, 40])                 # SWI columns only
mango.get_data("magnetosheath", frame="pgsm", cone=[80, 100], clock=180,
               spacecraft="THA", start="2008-08-01", stop="2008-09-01")      # PGSM
mango.get_data("magnetosphere", frame="pgsm", tilt=[10, 15],
               spacecraft="THA", start="2008-08-01", stop="2008-09-01")
mango.count("magnetosheath", frame="pgsm", cone=[80, 100])   # exact number of output rows
```

### Parameters by frame

A dash means the parameter is refused with a `FrameError`.

| frame | region | `cone=[min, max]` | `clock` | `tilt=[min, max]` |
|---|---|---|---|---|
| none / `"gsm"` | magnetosheath | GSM cone [0, 180] | range | - |
| none / `"gsm"` | magnetosphere | GSM cone [0, 180] | range | [−35, 35] |
| none / `"gsm"` | solar_wind | - (no IMF columns) | - | - |
| `"swi"` | magnetosheath | SWI cone [0, 180] | - | - |
| `"swi"` | magnetosphere, solar_wind | frame refused | | |
| `"pgsm"` | magnetosheath | required, SWI cone [0, 180] | required, one value | - |
| `"pgsm"` | magnetosphere | - | - | required, with symmetry |
| `"pgsm"` | solar_wind | frame refused | | |

- All angles are in degrees.
- A `clock` range `[c1, c2]` takes values in [−360, 360], taken mod 360. If c1 ≤ c2 after
  that, it selects c1 ≤ clock ≤ c2; otherwise it wraps through north (clock ≥ c1 or
  clock ≤ c2), so `[330, 30]` and `[-30, 30]` both select the sector around 0°. A range
  spanning 360° or more selects every row with a defined clock; because bounds are reduced
  mod 360 first, `[180, -180]` selects only 180°. In GSM `clock` is a range (a single value
  is an error); in PGSM it is one target value (a range is an error); SWI has no clock.
- `tilt=[min, max]` is in degrees and applies to the magnetosphere only. The legacy
  `tilt_min`/`tilt_max` filter (radians, no symmetry) cannot be combined with `tilt=`.
- Any `cone` or `clock` selection implies `sw_paired_only` (the IMF columns are filled on
  unpaired rows too, but are not meaningful there). `frame="swi"` and `frame="pgsm"` imply
  `normalized_only`, and `sw_paired_only` in the magnetosheath.
- Range filters (`bz_imf_max`, `y_gsm_min`, ...) and `spacecraft`/`start`/`stop` select on
  the measured values, before any transform: on mirrored or rotated rows they do not act on
  the PGSM quantities.
- With a frame or a selection, `limit` applies to the rows fetched, before the selection
  and the PGSM transform.

### Columns returned

| frame | vector columns | plus |
|---|---|---|
| none | every served column | - |
| `"gsm"` | `Bx/By/Bz`, `Vx/Vy/Vz`, `X/Y/Z_gsm`, `X/Y/Z_gsm_norm`, `B*_imf`, `V*_sw` | scalars |
| `"swi"` | `B*_swi`, `V*_swi`, `X/Y/Z_swi_norm` | scalars |
| `"pgsm"` | `X/Y/Z_pgsm_norm`, `B*_pgsm`, `V*_pgsm`, `mirrored`, `bx_sign` (magnetosheath) / `tilt_pgsm` (magnetosphere) | scalars |

Scalars (frame-free, returned in every frame where they are served): `Time`, `SC`, `Np`,
`Tp`, `SW_pairing`, `Norma_pos`, `Np_sw`, `Tp_sw`, `Pd_sw`, `Beta_sw`, `Ma_sw`, `R_mp`,
`R_bs`, `R_norm`, `tilt`. In SWI and PGSM the GSM IMF and solar-wind vectors are not
returned. With an explicit `frame`, asking in `columns=` for a column of another frame
raises a `FrameError` naming the frame it belongs to; without `frame`, every served column
may be requested.

### Angle definitions

- **GSM cone** = acos(Bx_imf/|B_imf|) ∈ [0°, 180°], measured from X_GSM; 0° = IMF
  sunward.
- **GSM clock** = atan2(By_imf, Bz_imf) ∈ [0°, 360°); 0° = northward, 90° = +Y.
- **SWI and PGSM cone** = acos(s·B_imf·X̂/|B_imf|), measured from −V_sw: X̂ = −V_sw/|V_sw|
  and s = sgn(Bx_imf) as in the SWI construction. It is at most 90° except for a few rows
  where aberration makes the scalar product negative (about 3.5 % of the magnetosheath
  sample rows), so it can slightly exceed 90°.
- Rows where an angle is undefined (zero IMF, or zero solar-wind speed for the SWI cone)
  are never selected.

SWI is organized around −V_sw, not X_GSM: on the docs sample the angle between −V_sw and
X_GSM has a median of 5.4° (99th percentile 13°; aberration accounts for about 4°, flow
deflection for the rest), and the cone measured from −V_sw differs from the GSM cone by a
median of 2.6° (90th percentile 7.4°, max 14°). The same `cone` range therefore selects
slightly different samples in GSM and in SWI/PGSM; narrow cone bins near 0° are the most
affected.

### GSM selections

Without a transform, `cone`, `clock` and `tilt` just select rows:

```python
mango.get_data("magnetosheath", cone=[20, 40])                 # IMF cone between 20 and 40 degrees
mango.get_data("magnetosheath", clock=[330, 30])               # clock wraps through north
mango.get_data("magnetosphere", tilt=[10, 15])                 # dipole tilt, degrees
```

### SWI

Magnetosheath only: the served SWI columns (`*_swi`), where the IMF lies in the X–Y plane
along +Y (By > 0) and, up to aberration, Bx > 0, so only the cone angle matters. `cone` is
measured from −V_sw; `clock` is refused.

### PGSM

PGSM pools measurements taken under many IMF orientations (magnetosheath) or dipole tilts
(magnetosphere) by symmetry. The result looks like a GSM map for one target IMF orientation
or tilt range, with much better coverage.

**Magnetosheath.** Rows are taken in the SWI frame.
- `cone=[min, max]`: the SWI cone defined above, measured from −V_sw.
- A row with SWI cone f is kept with `bx_sign` = +1 if f is in the range, and with
  `bx_sign` = −1 if 180° − f is (the Y mirror of eqs 2.19–2.20). A range containing both
  gives two rows. Rows with `bx_sign = −1` have their positions reflected Y → −Y relative
  to SWI (eq 2.19).
- `clock`, degrees: the target IMF clock angle atan2(By, Bz) (0° = northward). Every
  selected row is rotated about X to it (eqs 2.19–2.20). Changing `clock` only re-runs the
  local transform; nothing is downloaded again.

**Magnetosphere.** `tilt=[min, max]`, degrees. A row with dipole tilt ψ is kept as measured
if ψ is in the range, and its mirror image (eqs 2.14–2.16: positions (X, −Y, −Z),
B (−Bx, By, Bz), V (Vx, −Vy, −Vz)) is added if −ψ is. A range containing 0 can give a row
twice.

**Output.** Every requested column as measured (also on mirrored rows), plus
`X/Y/Z_pgsm_norm`, `Bx/By/Bz_pgsm`, `Vx/Vy/Vz_pgsm`, `mirrored` (magnetosheath: true when the row's IMF Bx sign, `bx_sign`, differs from the
measured sgn(Bx_imf); magnetosphere: true for the tilt mirror ψ → −ψ). Because SWI already
applies B → −B to samples measured with Bx_imf < 0, no magnetosheath PGSM row is the bare
measurement for those samples: `~mirrored` does not select unsymmetrized data. Also
`bx_sign` (magnetosheath) and `tilt_pgsm` (magnetosphere, degrees: the row's tilt, ψ as
measured or −ψ on mirrored rows; note the served `tilt` column is in radians).

### Server version

New in 0.3. `get_data` with a frame or a selection also works against a 0.2 server (the
selection and projection then run in the client); `count()` with a frame, `cone`, `clock`
or `tilt` needs a server running space-mango ≥ 0.3.

## Known caveats

- **SWI columns (magnetosheath).** The SWI basis of each row is built from that row's
  `V*_sw` and `B*_imf`; `B*_swi` includes the factor sgn(Bx_imf), so in SWI the IMF lies
  in the X–Y plane with By > 0 and, up to aberration, Bx > 0. `*_swi_norm` was re-normalized between mean boundaries after the
  rotation: near the magnetopause or bow shock (about 9 % of rows) its radius differs from
  that of `*_gsm_norm`, by up to about 1.7 R_E. SWI columns are null unless `Norma_pos`
  and `SW_pairing` are both true.

Column names are frozen for the data paper (in preparation); they will not be renamed.
