# MANGO: a discoverable Python API for space plasma physicists — design

Date: 2026-10-07 · Status: draft for review · Supersedes the *ordering* of
`docs/superpowers/plans/2026-07-26-dataset-versioning-and-update-pipeline.md` (that plan's
versioning and regeneration content is kept as Phases 3–4, see §8).

Companion: `2026-10-07-mango-column-dictionary-draft.md` — per-column unit/frame/meaning,
verified against the live server where possible, with open questions for the PI.

---

## 1. Goal and users

MANGO should be adopted by space plasma physicists outside the LPP group. Success = a physicist who
just ran `pip install space-mango` can, without reading source code or asking us:

1. find out what regions, spacecraft, time coverage and columns exist, with units, frames and meaning;
2. pull a filtered subset for a **statistical study** (e.g. magnetosheath, IMF Bz<0, d_msh<0.3)
   into pandas, without accidentally downloading 84 M rows;
3. get **event context**: for one spacecraft and a time interval, which region it was in and the
   upstream conditions;
4. know which dataset version they used and how to cite it.

Primary use cases (PI, 2026-10-07): statistical studies and event context. ML training sets and
paper reproduction are secondary (served by versioning, Phase 3).

## 2. Decisions (2026-10-07)

| # | Decision | Why |
|---|---|---|
| D1 | API first; versioning and regeneration pipeline stay as later phases | user adoption is the bottleneck |
| D2 | **Served column names are frozen** (the Scientific Data paper uses them). Metadata is added on top; nothing is renamed. `MMS` stays `MMS`. | paper consistency |
| D3 | Dead filters are fixed **in the catalog**, not the data: `d_msh`, `d_msp` → column `R_norm`; `tilt` → `tilt`; `tilt` removed from magnetosheath | `R_norm` verified equal to the d_msh/d_msp definitions to 1e-15 (dictionary). Reverses the 2026-07-26 "emit D_msh/D_msp/Tilt in data" item, which conflicted with D2 |
| D4 | `get_data` returns a `MangoResult` (data + metadata + version + citation), converting to pandas / polars / xarray | units and frames must survive conversion |
| D5 | Event context: `timeline()` (rows across all regions with a `region` column) **and** `.to_intervals()` | both views needed |
| D6 | Large requests: `count()` before download + on-disk client cache inspired by speasy; no hard size limit | statistical studies need big pulls |
| D7 | Filters stay as kwargs (`bz_imf_max=-2`); discoverability via region objects with generated signatures | no new syntax |
| D8 | speasy/SciQLop integration is a later phase; Phase 1 only exports intervals in a speasy-readable format, no speasy dependency | MANGO's main query (rows where conditions hold) doesn't fit speasy's (product, start, stop) |
| D9 | Architecture A: metadata catalog lives in `models.py` and is served by the server | single source of truth, CSV users benefit, ships without touching data on disk |

Still in force from 2026-07-26 (not re-litigated): CalVer `YYYY.N` + `schema_revision` +
content-addressed `dataset_id`; existing data adopted as `2026.0`; every release retrains every
model; kaa and published versions are read-only; Zenodo DOI per version; Alexis Jeandet operates
the serving host.

## 3. Public API

```python
import space_mango as mango

# discovery — no data download
mango.regions()                       # table: region | definition (one line)
mango.describe("magnetosheath")       # table: column | unit | frame | description | filter
mango.spacecraft("magnetosheath")     # table: sc | start | stop | n_rows
mango.search("density")               # matching columns and filters across regions
mango.count("magnetosheath", spacecraft=["THA"], bz_imf_max=-2)  # n_rows, est_mb

# region objects: attributes for tab-completion; generated signatures for help()
msh = mango.magnetosheath             # also mango.magnetosphere, mango.solar_wind
r = msh.get_data(spacecraft=["THA", "MMS"], start="2017-01", stop=datetime(2018, 1, 1),
                 columns=["Bx_swi", "Np", "R_norm"], bz_imf_max=-2, d_msh_max=0.3)

r.to_pandas()      # df.attrs["mango"] = {version, columns: {name: {unit, frame, description}}}
r.to_polars()
r.to_xarray()      # dims (time,) ; per-variable attrs units/frame/description ; SC as variable
r.metadata         # same catalog entries as describe(), restricted to returned columns
r.version          # "2026.0"
r.query            # the canonical query, re-runnable
r.cite()           # BibTeX: dataset (Zenodo DOI when it exists) + data paper

# event context
t = mango.timeline("THA", "2017-01-12T10:00", "2017-01-12T12:00")   # MangoResult, + 'region'
t.to_intervals()   # table: sc | region | start | stop | n_points
t.to_intervals().write_csv("thA.csv")   # columns chosen to load as a speasy/SciQLop catalog

mango.cite()       # dataset-level BibTeX
mango.cache.info(); mango.cache.clear()
```

- `mango.get_data(region, ...)` keeps working (delegates to the region object).
- `start`/`stop` accept `str`, `datetime`, `numpy.datetime64`, `pandas.Timestamp`. Old
  `time_min`/`time_max` remain as deprecated aliases (warning).
- Region objects are thin: `describe()`, `spacecraft()`, `count()`, `get_data()`, `filters`.
  Their `get_data` signatures are **generated from `RANGE_FILTERS`** (explicit keyword-only
  `{name}_min`/`{name}_max: float | None` parameters with unit in the docstring). A test fails if the
  generated code drifts from the catalog.
- `MangoClient(base_url=..., cache_dir=..., offline=False)` stays the power-user entry point; it
  gains `close()` / context-manager support.

### Errors

All errors subclass `MangoError`, each with a "did you mean" suggestion (difflib) where relevant:
`UnknownRegionError`, `UnknownSpacecraftError`, `UnknownColumnError`, `MangoFilterError`
(existing; now also unknown/unsupported-in-region filters), `TimeParseError`,
`ServerError` (wraps HTTP failures with the server's message). No silent empty results for typos;
no silent dropping of unknown columns.

## 4. Server

### 4.1 Catalog (`models.py`)

- `REGIONS: dict[str, RegionInfo]` — name, one-line physical definition, classification
  reference (Nguyen et al. 2022).
- `COLUMNS: dict[str, ColumnInfo]` — name, regions, dtype, unit, frame, description,
  `computed` (how / source), per-region variants where the meaning differs (`R_norm` in MSH vs MSP).
  Filled from the companion dictionary; INFERRED/UNKNOWN entries are worded cautiously until the
  PI answers (§9).
- `RANGE_FILTERS` corrected per D3; `vx_sw` description corrected (signed, negative).
- Shared, dependency-light `filtering.py` (polars only) building the filter expressions from the
  catalog. Used by the server **and** the client cache, so both filter identically.

### 4.2 Behaviour changes

- Filter whose backing column is absent → **HTTP 400** (removes the silent skip at
  `dataset.py:21-22`). Unknown query parameters, spacecraft, columns → 400 with the valid names.
- Malformed times → 400 (today: 500, `dataset.py:73-75`).
- Every response carries `X-Mango-Dataset-Version`.
- FastAPI app version wired to the package version (`app.py` hard-codes `0.1.0`).

### 4.3 New endpoints (`/api/v1`, all read-only)

| endpoint | returns |
|---|---|
| `GET /regions` | regions with definitions (extends current list) |
| `GET /regions/{r}/describe` | column catalog for the region, intersected with the served schema |
| `GET /regions/{r}/spacecraft` | sc, start, stop, n_rows — computed once at startup |
| `GET /regions/{r}/count` | same params as `/data`; n_rows + estimated bytes |
| `GET /timeline?sc=&start=&stop=&columns=` | Arrow/CSV rows from all regions with a `region` column |
| `GET /dataset` | version, citation, DOI (null until Phase 2), schema checksum |

Startup self-check: every catalog column and filter must exist in the served schema of the region
it claims; otherwise the server logs the mismatch and the corresponding filter answers 400
(fail loudly per request, don't refuse to start).

## 5. Client cache

Design borrowed from speasy (`speasy/core/cache/`, studied 2026-10-07): fragment the data space,
fetch only missing fragments, merge locally. Simplified because MANGO releases are immutable.

- **Fragment** = (dataset version, region, SC, time chunk, column), one zstd Parquet file:
  `~/.cache/space-mango/<version>/<region>/SC=<sc>/<column>/<chunk>.parquet`
  (location via `platformdirs`, override `SPACE_MANGO_CACHE_DIR`).
- **Query path**: needed columns = requested ∪ filter columns ∪ flags. Determine missing fragments,
  fetch them via `/data` (no range filters, explicit columns, grouped contiguous chunks per request),
  then `pl.scan_parquet` locally + shared `filtering.py` + column selection.
- **Chunk length**: fixed per region, chosen so a single-column fragment is ~10–50 MB of rows
  (to be measured from per-SC row density; calendar-month is the starting guess).
- **Invalidation**: none needed — the version is in the path. Version read once per session from
  `/dataset`; a changed server version starts a new directory; old ones age out.
- **Eviction**: LRU on file access time, cap `SPACE_MANGO_CACHE_SIZE` (default 10 GB), run after writes.
- **Concurrency**: write to temp file + `os.replace` (atomic); worst case a duplicate download.
- **Bypasses**: `limit=` queries and `cache=False` go straight to the server with server-side
  filtering (for one-off huge, very selective pulls). `offline=True` serves only from cache and
  raises if a fragment is missing.
- **Progress**: a progress bar on fragment downloads (optional dependency `tqdm`, silent without it).

## 6. Code layout

```
src/space_mango/
  models.py        catalog: REGIONS, COLUMNS, RANGE_FILTERS (+ dataclasses)
  filtering.py     shared polars filter builder (client + server)
  errors.py        MangoError hierarchy
  result.py        MangoResult (+ to_pandas / to_xarray / to_intervals / cite)
  cache.py         fragment cache
  regions.py       region objects; generated signatures (_regions_generated.py, checked by test)
  client.py        MangoClient (HTTP, discovery calls, cache orchestration)
  __init__.py      module-level API on a lazy default client
  dataset.py / routes/ / app.py   server (as today + §4)
```

New client dependencies: `platformdirs` (small, pure Python). `pandas` and `xarray` are imported
lazily by `to_pandas`/`to_xarray` (extra `space-mango[pandas]`, `[xarray]`); `tqdm` optional.

## 7. Testing

- Extract the triplicated fixture builder into `tests/conftest.py`. Fixture data must use the
  **served** schema (`R_norm`, lowercase `tilt`, `MMS`, the `*_swi` columns) — today's fixtures use
  the intended schema, which is why tests never caught the dead filters.
- Golden test: each of the 4 previously dead filters actually filters.
- Catalog contract test: catalog ↔ fixture schema; region-object signatures ↔ `RANGE_FILTERS`.
- Error tests: each error type, with suggestion text.
- Cache tests: fragment computation, partial hits, offline, eviction, atomic write, local vs
  server filtering give identical rows.
- `MangoResult` conversions keep metadata (pandas attrs, xarray attrs).
- `test_readme_examples.py` updated with the new README quick start.
- Fix CI first (`devtools/lint.py` missing → call ruff/basedpyright/codespell directly).

## 8. Phases

| phase | content | est. |
|---|---|---|
| **1 — discoverable API** (this spec) | CI fix, conftest, catalog fixes + COLUMNS, errors, new endpoints, MangoResult, region objects, timeline/intervals, cache, README + docs page | 3–4 wk |
| **2 — citable dataset** | PI answers §9 → finalize column metadata; SWI anomaly decision; dataset card (rendered docs page); Zenodo deposit + DOI for `2026.0` (quota request early) | 1–2 wk + external |
| **3 — versioning & publish** | 2026-07-26 Phase 1, slimmed: versioned layout, manifest, contract gate, `mango data` CLI, migration with Jeandet; client `version=` pinning becomes meaningful | 2 wk |
| **4 — regeneration pipeline** | 2026-07-26 Phase 2, unchanged in substance (`packages/mango-pipeline/`) | 9–13 wk |
| **5 — speasy/SciQLop adapter** | `space-mango[speasy]`: intervals as speasy Catalogs; after discussion with speasy maintainers | 1 wk |

Phase 1 critical path to first user-visible value (~3 days): CI fix → conftest with served schema →
catalog fixes (D3) + HTTP 400 → `describe`/`spacecraft`/`count` → README fix (`MMS`).

## 9. Open questions (PI)

1. **SWI anomaly**: `*_swi` columns reproduce exactly from each row's own `*_sw`/`*_imf` for
   THA, C1, MMS-2015 but not THD-2012, THE-2019, DS1, C3; B and position not rotated by the same
   matrix there. Processing bug or undocumented step? Affects what we can promise in `describe()`.
   (Possible bug — evidence in the companion dictionary.)
2. The 10 column-semantics questions listed at the end of the companion dictionary (Time bin
   convention/UTC, pairing method, R_mp/R_bs models, `Norma_pos` criterion, `*_gsm_norm`
   reference surfaces, proton vs ion moments, …).
3. Scientific Data paper status (submitted?) and its exact citation for `cite()`.
4. Server capacity for the new endpoints and cache-driven traffic (ask Jeandet).

## 10. Non-goals (Phase 1)

Renaming columns; new regions or missions; dataset regeneration; multi-version serving; a web UI
(MANGO Explorer is a separate consumer of this API); expression-object filter syntax.
