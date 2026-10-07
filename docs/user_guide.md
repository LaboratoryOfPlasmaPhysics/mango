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

## Known caveats

<!-- SWI-frame caveat: wording pending PI decision (see spec §9 q1) -->

Column names are frozen for the data paper (in preparation); they will not be renamed.
