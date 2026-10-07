# Making MANGO updatable: versioned dataset + regeneration pipeline

## Context

MANGO's published dataset stops at **2021-Q2** and cannot be extended without a person re-running
Jupyter notebooks on a colleague's account on `kaa`, with stale hardcoded paths, a dependency on a
personal `spok` fork, and no parameterisation by time range. MANGO is meanwhile load-bearing for a
Nature Scientific Data paper in preparation, for the DR 2026 dossier (which promises "continuous
enrichment with new and old missions"), and for the APR 2027 SMILE proposal. "Continuous
enrichment" is currently not possible.

Worse, nothing verifies that what is served is what was intended: **4 of the 18 advertised range
filters silently return unfiltered data** on the live server today. Users may have published
figures from data they believed was filtered.

Goal: make an update routine, fully provenanced, and always revertable, with the version being
replaced left frozen and untouched.

---

## Ground truth (verified against the live server, 2026-07-26)

| region | rows | coverage |
|---|---|---|
| magnetosphere | 84 305 926 | → 2021-Q2 |
| magnetosheath | 50 655 590 | → 2021-Q2 |
| solar_wind | 17 034 753 | → 2021-Q2 |

Spacecraft partitions: `THA THB THC THD THE C1 C3 MMS DS1`. The key is **`MMS`**, but `README.md:58`
and `routes/data.py:56` both tell users `MMS1` — that documented example returns zero rows.
THB/THC stop before 2010 (already cut at the ARTEMIS transition); C3 ends 2009, DS1 2007, C1 2019.
**Only THA, THD, THE, MMS1 and C1 actually extend past 2021.**

### The four dead filters

Verified by sending absurd bounds and still getting rows, with `bz_imf_min=1000` / `np_sw_min=99999`
/ `x_gsm_min=99999` as controls returning none.

| filter | region | catalog expects | data has | resolution |
|---|---|---|---|---|
| `d_msh` | magnetosheath | `D_msh` | `R_norm` | fix in data — emit `D_msh` |
| `d_msp` | magnetosphere | `D_msp` | `R_norm` | fix in data — emit `D_msp` |
| `tilt` | magnetosphere | `Tilt` | `tilt` | fix in data — casing |
| `tilt` | magnetosheath | `Tilt` | *(absent)* | **remove from catalog** — dipole tilt is not a meaningful organising parameter for the magnetosheath |

Root cause, `src/space_mango/dataset.py:21-22`:

```python
if filt.column not in available:
    continue
```

Schema drift degrades silently. `tests/test_readme_examples.py:45` asserts the *intended* schema
(`"D_msh": 0.4`), so the tests can never catch this.

### Other repo facts that shape the design

- **No dataset version, manifest, provenance or checksum anywhere.** The on-disk contract is
  implicit: `$MANGO_DATA_DIR/<region>/SC=<value>/part-0.parquet`.
- `get_dataset()` is a FastAPI `Depends`, so each of the 4 preforked gunicorn workers builds its
  own `MangoDataset` **on its first request**. With a fixed path that is harmless; with a moving
  pointer, workers would pin different versions permanently. Fix: resolve in `create_app()`, which
  runs once in the arbiter under `--preload`.
- `scripts/convert_pickles.py` writes in place, no staging, no atomic swap, no cleanup of stale
  `SC=` dirs — unsafe to re-run against a live directory.
- **CI is red**: `.github/workflows/ci.yml:60` and `Makefile:15` call `devtools/lint.py`, which does
  not exist. Nothing is currently enforced.

### Reusable assets

`spok` (`~/Documents/code/spok`, **your repo**, published, tested, CI) is a real dependency: 8
magnetopause models, 3 bow-shock models, `Magnetosheath.boundaries()`, `swi_base`, `get_tilt`.
`some_code/models_magnetic_field.py:374` holds the actual normalized-position algorithm (~40 lines).
`spok/models/planetary.py:26` holds the Šafránková pairing, **broken under pandas ≥ 2.0**
(`DataFrame.append` at :37/:41) — fix upstream in spok, cut 0.1.2.

Not available anywhere locally: the region classifier and its labels (kaa only), and any GSM→SM
transform (new code, against `geopack` — **not** spacepy, which needs Fortran toolchains and is
hostile to CI).

---

## Your six steps — validated

Correct chain, correct order. Five things missing, one framing correction.

**1. Download from speasy** — add a **coverage-inventory step before it** (read what exists from
the previous manifest, so fetch is resumable and append-only rather than `start, stop` → full
recompute). Give OMNI its own stage: it feeds both pairing and boundary training. Capture
provenance here (speasy version, product ids, fetch date, per-chunk checksums) or nothing
downstream is reproducible.

**2. Re-train region models** — correct as you scoped it. Two additions: the **label set becomes a
first-class versioned input artifact** copied read-only off kaa (the hardest external dependency in
the plan), and a **drift check**, because labels are ≤2021 and the classifier therefore encodes
≤2021 instrument behaviour applied at a different solar-cycle phase.

**3. Re-train boundary models** — **a stage is missing between 2 and 3: crossing extraction.**
Boundary models train on a BS/MP *crossing catalog* derived from region transitions, not on the
classified time series. That catalog is where most of the scientific judgement lives, is the
artifact that genuinely grows with new data, and is independently citable.

**4. Pairing to solar wind** — correct, and it is **two stages**: crossing-level pairing (~10⁵ rows)
must run *before* boundary training because the crossing feature vector is the upstream OMNI at the
crossing; row-level pairing (~1.5×10⁸ rows) runs after classification and is independent of the
boundary models, so it parallelises with training. Collapsing them creates a false serialisation.

**5. Re-position between average boundary models** — three products, and conflating two of them is a
reproducibility trap:
- `R_mp`, `R_bs` — instantaneous, from the **retrained ML** models at the row's upstream conditions;
- `D_msh` / `D_msp` ∈ [0,1] — **the quantity currently mis-emitted as `R_norm`**, from those same ML
  models, so it changes every release. That is intended.
- `R_norm` — radius remapped into the **average analytic Formisano parabolic frame**. This frame
  must be **frozen as a config constant across all releases**. If it floats, `R_norm` stops being
  comparable across versions and every figure in the 2022/2024 papers becomes unreproducible.
- The magnetosphere branch additionally needs GSM→SM, which exists nowhere locally.

**6. Remake the dataset** — becomes four steps, and this is where revertability actually lives:
assemble → **contract gate** → **scientific diff** → **atomic publish**.

### Framing correction

Because every release retrains every model, every release changes every value. "Did the old rows
stay identical?" is not available as a safety net, and semver's minor/patch distinction is
meaningless for the data. But the conclusion is *not* "no hard gate" — it is **two different
gates**:

- The **contract gate** is absolute and threshold-free: schema, dtypes, sortedness, filter-column
  presence. It runs on one version. It is what would have caught the four dead filters.
- The **scientific diff** has thresholds, and every one of them will trip on a retrain. Its purpose
  is not to block change but to **force each change to be seen and named**: `BLOCK` means a human
  types `--accept-block --reason '...'` and that reason is written into `history.jsonl` forever.
  The deliverable is an audit trail of deliberate scientific decisions.

One non-obvious cost consequence: retraining the region classifier from **frozen labels with a fixed
seed and a pinned sklearn is a deterministic function of unchanged inputs**, so its artifact id is
identical across releases and everything downstream of it stays cached. Retrain anyway and *assert*
the hash matches. The genuinely expensive consequence of "retrain everything" is boundary models →
full re-normalisation of 152 M+ rows, and that is irreducible — it is the improvement you ship.

### Decomposition: assets, not a chain

| asset | from | lifecycle |
|---|---|---|
| A1 raw mission series | speasy | append-only |
| A2 OMNI | speasy | append-only, occasionally revised upstream |
| A3 label set | kaa, read-only copy | frozen |
| A4 region classifier | A3 + code | retrained; **id stable** |
| A5 classified regions | A1 + A4 | cached while A4 holds |
| A6 BS/MP crossings | A5 | recompute |
| A7 boundary models | A6 + A2 | **always changes** |
| A8 SW pairing | A1 + A2 + A7 | incremental |
| A9 normalized positions | A1 + A7 + A8 | **full recompute, dominant cost** |
| A10 MANGO release | A5 + A8 + A9 | published, versioned |

A dataset version is then literally the tuple of input asset ids plus the pipeline git sha —
which makes the Scientific Data methods section one paragraph.

**Acyclicity, verified.** The classifier takes only `Bx,By,Bz,Np,Vx,Vy,Vz,Tp` — no position, no
boundary (`some_code/preprocessing_data.py:171`). The region post-filter uses *fixed literature*
boundaries (Shue 1998, Jerab 2005) via `clean_nightside_dipole`, not MANGO's own. SW pairing uses
OMNI's own `BSN_x`, not MANGO's BS model. All three keep the DAG acyclic. Any of the three could be
"improved" into a cycle. **Rule: a back-edge is legal only if it points at a pinned, immutable
artifact from a previous release, declared in `[pins]`.** Enforce with `test_registry_dag.py`.

---

## Decisions taken

| | decision |
|---|---|
| Release model | Retrain everything every release; region classifier from the **same frozen labels** |
| Old notebooks | Reference only — the pipeline must never depend on or invoke them |
| Existing artifacts | **Strictly read-only.** Nothing on kaa or in any published version is modified |
| First deliverable | **Phase 1: versioning + publish infrastructure**, usable with today's data |
| History | Append; previous versions frozen; retain ≥2 versions + permanent `history.jsonl` |
| Pipeline home | uv workspace member `packages/mango-pipeline/` in this repo |
| Serving host | Operated by **Alexis Jeandet** — publish is a safe atomic handoff, not assumed write access |
| Dead filters | Fix `D_msh`/`D_msp`/`Tilt` in the data; **drop `tilt` from the magnetosheath catalog** |
| Spacecraft | Rename `MMS` → `MMS1` in v2; extension covers THA/THD/THE/MMS1/C1; **THB/THC keep the existing ARTEMIS cut** |
| Filter errors | Requesting a filter with a missing column → **HTTP 400**, no escape hatch |
| Archival | Zenodo DOI per published version |

---

## Phase 1 — versioning and publish infrastructure

### Task 0 first: a 10-line spike

The recommended adoption strategy never moves a byte — it puts `versions/2026.0/<region>` **symlinks**
next to the existing data. That only works if polars scans through a symlinked directory with
`hive_partitioning=True` and still derives `SC` correctly. Unverified, and it decides the whole
migration. Fallback if it fails: `--mode move`, a `rename(2)` of the legacy region dirs — still zero
bytes rewritten, still instantly reversible.

Same spike confirms that a `=` anywhere in the data-root path injects a bogus hive column
(`/srv/data=mango` would silently add a column to every region) — justifying check `V000`.

### Layout

```
$MANGO_DATA_DIR/
├── current   -> versions/2026.1      # relative symlinks; absolute would break in the container
├── previous  -> versions/2026.0      # sole input to `revert`
├── history.jsonl                     # append-only, never pruned
├── .mango.lock                       # flock target for activate/revert/promote
├── versions/2026.0/{manifest.json, MANIFEST.sha256, <region> -> ../../<region>}   # adopted
├── versions/2026.1/{manifest.json, MANIFEST.sha256, <region>/SC=…/part-*.parquet} # owned
├── incoming/2026.2/                  # Phase 2 staging; never scanned. Must be under $D so
│                                     # promote is an atomic same-filesystem rename.
└── magnetosphere/ magnetosheath/ solar_wind/    # legacy originals, untouched, still work
```

No `=` at any level except `SC=`. Resolution precedence, in order: explicit `version` →
`<root>/manifest.json` present → `<root>/current` → `versions/` **without** `current` is a hard
error (never guess) → legacy region dirs at root → error. The legacy case is byte-for-byte today's
behaviour, which is what keeps every existing test and the live deployment green.

### Manifest

`manifest_schema_version: 1`. Key design points rather than the full field list:

- **`dataset_version` = `YYYY.N` CalVer**, plus a **separate `schema_revision: int`** bumped only on
  column add/remove/rename/dtype-change, plus **`dataset_id`** as content-addressed ground truth.
  One string cannot carry both "will my code break?" (machine-checkable, changes rarely) and "which
  numbers did I get?" (changes every release, only recency is meaningful). A methods section reads:
  *"MANGO version 2026.1 (`sha256:ab12cd34`), schema revision 3."*
- `dataset_id` = `sha256` over `sorted("{relpath}\0{size}\0{sha256}\n")` across all parquet files,
  excluding the manifest itself. Order-independent, reproducible with shell tools, algorithm name
  pinned so a future change is detectable.
- **`storage.mode: "owned" | "linked"`** — load-bearing, not cosmetic. The adopted 2026.0's bytes
  live *outside* its version dir; without this field a later `prune 2026.0` would `rm -rf` the
  original dataset through the symlinks.
- Per-column `source: "file" | "hive"` — `SC` comes from the directory name
  (`convert_pickles.py:64` drops it), so without this the manifest lies about the parquet schema.
- `provenance.inputs: [InputArtifact]` with role / name / version / vintage / uri / digest /
  time_range, and a `training` sub-object for model artifacts. For the adopted version every field
  is `null` and `method` is `"adopted-scan"` — *"provenance unknown, reconstructed by read-only
  scan"*, which is honest rather than invented. **The `inputs.toml` file that feeds this is the
  Phase-1/Phase-2 seam — define its schema now** so the pipeline has a target.
- `contract.known_failures` records the 4 findings verbatim for 2026.0. Adoption must not be
  blocked by the failures it exists to document.
- Per-region `layout` (partition keys, sort order, row-group size), per-(SC, year) row counts,
  and a `stats` ladder: `none` → `cheap` (footer min/max/null_count + yearly counts + boolean
  true-fractions; seconds even at 84 M rows) → `full` (21 quantiles, mean, std).

### New modules in `src/space_mango/` — no new dependencies

| file | purpose | imports |
|---|---|---|
| `versioning.py` | resolution, atomic pointer ops, history | **stdlib only** |
| `manifest_build.py` | build a manifest by scanning | polars, pyarrow (core deps) |
| `validate.py` | the contract gate | polars, pyarrow, models |
| `diff.py` | scientific diff — **manifest-in, report-out, no data access** | stdlib only |
| `cli_data.py` | `mango data …` subapp | cyclopts (server extra) |
| `routes/dataset.py` | version/manifest endpoints | fastapi |

`tomllib` is stdlib at `requires-python >=3.11`.

**`atomic_symlink` matters**: `ln -sfn` is *not* atomic (coreutils unlinks then symlinks, leaving a
window with no `current`). Use `os.symlink(target, tmp); os.replace(tmp, link)` — `rename(2)`,
and `os.replace` replaces the link itself without following it.

Making `diff` take **manifests only** is the constraint that forces the manifest to carry everything
the gate needs. It also makes the diff instant, CI-runnable from two JSON files, and testable with
pure JSON fixtures.

### Contract gate

`validate_dataset(data_root, *, sorted_check="off"|"quick"|"full", manifest=None, strict=False)`
returning `ValidationReport` with typed `Finding(code, severity, region, partition, message, detail)`.
Exit 0 pass / 1 failures / 2 usage.

Checks: `V000` root path contains `=` · `V002` disk regions == `Region` enum · `V003` orphan/empty
partition dirs · `V004` bad SC value · `V005` nested hive dir · `V010` `Time` present, Datetime,
tz-naive · `V011` `SC` resolvable from hive and **not** shadowed by a file column · **`V012` all
partitions of a region share one file schema** · `V013` dtypes · **`V014` every `RANGE_FILTERS`
column present for every region that declares it** · `V020` empty partition · `V021` `Time`
ascending (`quick` = row-group footer stats monotone and non-overlapping; `full` = real scan) ·
`V022`/`V023` null and implausible timestamps · `V025` checksum/rowcount vs manifest.

`V012` deserves emphasis: today a single spacecraft missing a column resolves the schema from
whichever file polars reads first, so results depend on file ordering and nothing surfaces it —
the same failure mode as the dead filters, one level down.

**Acceptance criterion**: `validate --legacy` against today's data emits *exactly* the four `V014`
failures, with `similar_columns: ["tilt"]` and a **"case mismatch"** message for magnetosphere-tilt,
and `["R_norm"]` for `D_msh`/`D_msp` via a three-line `KNOWN_ALIASES` table (fuzzy matching will not
find it). That turns a mystery into an instruction.

### Scientific diff

`diff_manifests(a, b, thresholds) -> DiffReport`, verdicts `info` / `review` / `block`.

Always `BLOCK`: column removed or dtype changed · contract failure introduced · spacecraft removed ·
`time_max` regressed. Thresholded: total rows per region (>10 % review / >25 % block) · rows per
(region, SC, year) **restricted to years present in both**, else a coverage extension shows +∞ % on
every new year and drowns the signal · null fractions · distributions as `maxₖ|Δqₖ| / IQRₐ` over 21
quantiles (>0.05 / >0.20), scale-free and needing no shared binning.

**The one threshold to keep if you keep only one: region share per (SC, year)** (>2 pp review,
>5 pp block). Total row counts can be flat while the retrained classifier reassigns 5 % of the
magnetosheath to the magnetosphere. Nothing else in the list catches that — it *is* the measurement
of the retrain.

When a stats level is insufficient the report populates `skipped` and prints it. A gate that
quietly does not run is worse than no gate.

### CLI — `mango data …`

`validate` · `manifest` · `adopt` · `install` · `promote` · `list` · `show` · `diff` · `activate` ·
`revert` · `history` · `prune`.

- `adopt` = `manifest --legacy --allow-contract-failures`. It gets its own name because it carries
  a different promise and because **it never writes into the dataset directory** — it emits to
  stdout for review.
- `diff` must accept `--a-path` / `--b-path`: the real workflow diffs a *staging* directory against
  the current version **before** promoting, when `b` has no version directory yet.
- Many-path commands (`install`, `promote`, `prune`) default to `--dry-run` and print the exact
  `mkdir`/`symlink`/`rename` operations. Single-pointer commands (`activate`, `revert`) act
  immediately and print their inverse — in an incident that must be one keystroke.
- `activate`, under `flock`: validate → set `previous` → set `current` → append history → print the
  restart command and the exact inverse → with `--health-url`, **poll `/health` until the new
  version is live**. "Tell the operator a reload is required" is weaker than verifying it happened.
- `prune` refuses below `keep=2`, refuses `current`/`previous`, and refuses to delete anything
  reachable through a `"linked"` version's symlinks.

### Server and client

Resolve the dataset in `create_app()` so all 4 preforked workers inherit one identical
`ResolvedDataset`. `init_dataset` must **never raise** — a missing mount stores the error, `/health`
reports `status: "error"`, data routes 503. Crashing 4 workers at boot because a volume did not
mount is worse than the status quo.

**On swap, deliberately nothing happens until a restart.** Each worker keeps serving its pinned
version — consistent, still on disk, and truthfully reported. `/health` returns
`{dataset_version, pointer_version, stale, contract_ok, dataset_id}`. Do **not** implement hot-swap
polling: it buys seconds at the cost of workers straddling versions mid-query-loop. The real answer
to reproducibility is letting the client **pin** a version, which the layout makes nearly free.

- `GET /api/v1/dataset/{version,versions,manifest}`; `X-Mango-Dataset-{Version,Id}` headers on every
  data response (arrow and csv); `?dataset_version=` on the data endpoint.
- `_apply_range_filters` raises `UnavailableFilterError` → HTTP 400, **only when the user actually
  requested that filter**. An absent column nobody asked about stays silent.
- `FilterInfo` gains `available: bool` and `unavailable_reason`. Keep unavailable filters *listed*
  so users learn why. Client reads `f.get("available", True)`, so it works against old servers.
- Client: `provenance()` → the dict you paste into a paper; `cite()`; `last_dataset_version` from
  the response header; `spacecraft(region)` from the manifest (which also fixes the `MMS1` doc bug).
  Polars frames carry no metadata slot, so provenance rides on the client, not the DataFrame.

### Migration (Jeandet runs this)

**Steps 0–1 work with the existing `:ro` mount. That is the proof they cannot damage anything** —
an enforced property, not a promise in a doc.

| # | command | writes | rollback |
|---|---|---|---|
| 0 | `mango data validate --root $D --legacy --json-out /tmp/v.json` | nothing | — |
| 1 | `mango data adopt --root $D --version 2026.0 --out /tmp/m.json`, then review | `/tmp` only | `rm` |
| 2 | `mango data install --root $D --manifest /tmp/m.json --mode link --yes` | first write, additive: `versions/2026.0/*`, `history.jsonl`. **Server unaffected** | `rm -rf $D/versions $D/history.jsonl` |
| 3 | `mango data validate --root $D --version 2026.0 --against-manifest` | nothing | — |
| 4 | `mango data activate 2026.0 --root $D` | `current` symlink | `rm $D/current` |
| 5 | restart the service | — | `rm $D/current` + restart |
| 6 | `curl $URL/health` → assert `dataset_version=2026.0, stale=false` | nothing | as above |

Because `current` and the legacy path point at the same inodes, there is **literally zero data
change across the restart**. That is why adoption is risk-free. Use a container/unit restart, not
`kill -HUP`: gunicorn `--preload` + HUP semantics are subtle, and step 6 verifies the outcome
regardless of mechanism.

Publishing 2026.1 later: pipeline writes `incoming/2026.1/` → `validate` (must be **0** failures) →
`manifest --inputs inputs.toml` → `diff 2026.0 2026.1 --b-path incoming/2026.1` → **a human reads
it** → `promote` (atomic rename) → `activate` → restart. 2026.0 is never touched; `revert` is one
command at every stage.

### Phase 1 tasks

0. **Spike**: polars through symlinked hive dirs; `=` in root path. *Decides the migration mode.*
1. **Fix CI** — replace `devtools/lint.py` with `ruff check .` / `basedpyright` / `codespell` in
   `.github/workflows/ci.yml` and `Makefile`. Until this is green, nothing below is enforced.
2. **Extract `tests/conftest.py`** — `_write_test_region` exists three times
   (`test_client.py:78`, `test_placeholder.py:65`, `test_readme_examples.py:17`). Add fixtures
   `legacy_root`, `versioned_root`, `linked_root`, and **`broken_root`** — magnetosheath with
   `R_norm` and no tilt column, magnetosphere with `R_norm` and lowercase `tilt`. That reproduces
   the live pathology in a temp dir and is the acceptance criterion for task 4.
3. `versioning.py` + tests (6-case resolution matrix, atomicity, rejected version strings).
4. **`validate.py` + the golden four-failure test.** Ship before anything writes to disk — highest
   value, zero risk, and it is what *proves* the bug.
5. `manifest_build.py` + tests (determinism, hand-computed Merkle, `source: "hive"` for `SC`).
6. `diff.py` + tests (pure JSON fixtures).
7. `MangoDataset` version awareness, `init_dataset` in `create_app()`, `UnavailableFilterError`.
8. API surface + `routes/dataset.py` + headers + `?dataset_version=`.
9. Client methods + `README.md` `MMS1` fix.
10. `cli_data.py` read-only commands.
11. `cli_data.py` mutating commands (flock, atomic rename, history, `--health-url`).
12. `docs/operations/publishing.md` runbook + `docs/operations/inputs-schema.md`.
13. Execute the migration with Jeandet.
14. Update `CLAUDE.md`; delete the "Known data caveat" section, superseded by the contract gate.

Tasks 0, 1, 2, 4 are the critical path to value and are roughly a day's work.
Whole phase ≈ 2–3 weeks.

---

## Phase 2 — `packages/mango-pipeline/` (roadmap)

Wire with `[tool.uv.workspace] members = ["packages/*"]` in the root `pyproject.toml`; the pipeline
declares `space-mango = { workspace = true }` and imports `Region`/`RANGE_FILTERS` from
`space_mango.models`, so producer and consumer of the schema cannot drift. **`uv` is not installed
on this machine** — an increment-0 prerequisite.

### 14 stages

`inventory` → `fetch` → `l1_clean` / `omni_l1` → `train_region_model` → `classify` → `crossings` →
`pair_crossings_sw` → `train_boundaries` ∥ `pair_full_sw` → `normalize` → `assemble` →
`contract_check` → `diff_report` → `manifest`.

`train_boundaries` and `pair_full_sw` are siblings — run them concurrently, that is hours per
release. `normalize` is the dominant cost: evaluating two GBRs (`n_estimators=1500, max_depth=10`)
over 152 M rows is ~4.6×10¹¹ node visits ≈ **1–10 h on 24 cores**. Measure it early; if it lands
badly, `HistGradientBoostingRegressor` is a much faster drop-in but the switch must be gated behind
the scientific diff since the papers reproduce the GBR models.

### Artifact store

`$MANGO_ARTIFACTS/{inputs,raw,stages,runs,candidates}/`, each artifact identified by
`blake2b(canonical_json({stage, stage_version, params, inputs, scope, env_key}))`.

Two opinionated details:
- **`stage_version` is a hand-bumped int, not a source hash and not `git describe`.** A whitespace
  commit must not invalidate a three-day fetch. `git_commit`/`git_dirty` go in the manifest for
  provenance but not in the identity hash; release builds use `--strict-provenance` to refuse a
  dirty tree. Exception: the resolved `uv.lock` sha256 *does* enter the hash for the two sklearn
  stages, because a sklearn bump changes model output.
- Raw fetch is append-only via a `_chunks.jsonl` ledger. Providers *do* reprocess, so a reissue
  appends a chunk with `revision+1` and readers take the max revision. Downstream stages read a
  **raw snapshot id** = hash of the ledger rows *restricted to the build's scope*, so appending 2027
  data next year does not invalidate this year's shards. That detail is what makes incremental
  extension actually work.

Per-shard `_SUCCESS` sentinels; writes to `.tmp-<pid>` promoted with `os.replace`; same atomicity
discipline as Phase 1 so the mental model is uniform.

### Orchestration: plain Python, not Snakemake, not Prefect

Snakemake's identity model is timestamps, not content hashes, so it cannot produce the provenance a
Scientific Data paper needs — you would write the manifest layer anyway, and once written *it is
the DAG engine*. The shard set is data-dependent (which years does each SC have?), forcing
`checkpoint`s, the most confusing feature in the tool. Prefect needs a server and agent and its
result cache fights a bespoke content-addressed store — two sources of truth about what is cached.

The decisive argument for this team: every stage stays an ordinary importable function callable from
a notebook cell with real arguments, and each stage's pure maths lives in `mango_pipeline.science.*`
with no artifact store involved. That is the migration path *off* notebooks that both tools
foreclose. ~400–600 lines of runner. The runner emits `runs/<id>/dag.json`, so a 50-line Snakemake
wrapper remains possible if cluster scheduling ever matters.

Process isolation via `ProcessPoolExecutor(max_tasks_per_child=1)` replaces the `os.fork()` hack in
`convert_pickles.py:86` — OOM containment and parallelism from one mechanism. The 25 GB frame was an
artifact of materialising one global pandas DataFrame; sharded by (SC, year) the largest working set
is ~1.5 GB.

### Two scientific decisions that must be made now

- **The label artifact must freeze the feature values alongside the labels**, not just
  `(time, sat, label)`. Otherwise `train_region_model` depends on `l1_clean`, and any improvement to
  L1 cleaning silently changes the retrained classifier — destroying the whole point of "regenerated
  reproducibly from code". Freezing `(time, sat, Bx…Tp, label, label_source)` at ingest makes the
  stage hermetic and its id deterministic.
- **Acceptance for the reimplemented classifier is not bit-identity.** Reproducing Nguyen et al.
  2022 bit-for-bit across a decade of sklearn is not achievable. Define instead: ≥99 % agreement
  with the inherited model on a frozen 10⁵-row probe set, plus a held-out confusion matrix within
  tolerance of the published numbers. Both go in the manifest and in the paper — a far stronger
  provenance artifact than an opaque pickle.

**Do this immediately, before sklearn drifts further**: in a throwaway pinned env
(`uv run --isolated --with 'scikit-learn==1.0.2' …`), load each inherited classifier pickle once and
export `predict`/`predict_proba` over that probe set to Parquet. The pickle may become unloadable;
the probe predictions never will.

### Getting inputs off kaa, read-only

`rsync -a --checksum` **pull only**, no `--delete`, `--dry-run` first; sha256 computed both remotely
and locally (catches truncation); `chmod -R a-w` after ingest; re-ingest creates a new dated dir so
existing builds keep pointing at the old one. Pull: the frozen labels, the four classifier pickles
(as **test fixtures**, never production dependencies), the notebooks as reference text under
`docs/reference/kaa/` marked never-executed, the v1 dataset (diff baseline), and the v1 crossing
lists (to validate the reimplemented extractor on the overlap). Refetch OMNI from CDAWeb rather than
ingesting it, so the pipeline is self-contained.

**Open question to resolve on kaa**: the notebooks are named `1T_*` (1-minute) but inference runs on
5 s data. Confirm the training cadence — it changes what cadence the frozen-label Parquet must be at,
and it is a legitimate methodological detail for the paper.

### Increments

| # | scope | effort |
|---|---|---|
| 0 | workspace, config schema, artifact store, registry/runner, CLI, ledger, CI | 0.5 wk |
| 1 | **reproduce v1 for one (SC, year) using v1's own inputs** — inherited classifier, v1 boundary models. Success: ≥99.9 % of timestamps reproduced, median \|Δ`R_norm`\| < 1e-6 R⊕ | 2 wk |
| 2 | `inventory` + `fetch` + ledger; validate by refetching a year already in v1 | 1–2 wk + multi-day runtime |
| 3 | region-model retraining, probe-agreement test, label churn on the overlap | 1–2 wk |
| 4 | crossings + boundary models; validate the extractor against v1's crossing lists *before* retraining | 2–3 wk |
| 5 | full normalize, assemble, release; emit `D_msh`, `D_msp`, `Tilt` | 1–2 wk + compute |
| 6 | one-command routine release, runbook, golden metrics | 1 wk |

≈ 9–13 weeks with the backfill overlapping. **Increment 1 is the one that matters** — it isolates
reimplementation risk from retraining risk from fetch risk, and it is where every undocumented step
in the notebooks surfaces. Budget generously.

Scope explicitly cut: no new science columns in the proving release · no MMS2–4 (formation
separation ~10–100 km, redundant at 5 s) · no new missions · no Snakemake/Prefect/Airflow/DVC ·
no plugin system.

---

## Verification

**Phase 1, without touching production:**

```bash
uv sync --all-extras && uv run pytest          # after task 1, CI is green for the first time
uv run pytest tests/test_validate.py -k golden # the 4 dead filters, reproduced in a temp dir
```

Then against the live data with a read-only mount, which is the real proof:

```bash
uvx --from 'space-mango[server]' mango data validate --root $D --legacy --json-out /tmp/v.json
jq '[.findings[] | select(.code=="V014")] | length' /tmp/v.json     # expect 4
```

End-to-end on a scratch copy before Jeandet runs anything: `adopt` → `install --mode link` →
`validate --against-manifest` → `activate` → serve → `curl /health` → `revert` → serve again, and
assert the served row counts are identical before and after the round trip.

**Phase 2, increment 1 is the gate**: the reproduced (SC, year) shard must match v1 within the
stated tolerances, and `mango data diff` between v1 and the reproduction must show only INFO
findings.

---

## Open items

1. **Disk on the serving host** — two versions ≈ 50–70 GB once extended to 2026 (my estimate:
   25–35 GB per version). Needs confirming with Jeandet before task 13, along with the restart
   mechanism.
2. **Zenodo quota** — a full version may exceed the default 50 GB deposit limit; a quota request
   takes time, so start it before the paper needs the DOI.
3. **kaa access** — reading the notebooks and pulling the label set is the hard external dependency
   for Phase 2. Nothing in Phase 1 needs it.
4. `space-mango` 0.1.1 is on PyPI with only two releases and the FastAPI `version="0.1.0"` string is
   hardcoded and already stale (`app.py:12`). Wire it to the package version while touching that file.
