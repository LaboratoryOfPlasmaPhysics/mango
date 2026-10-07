# MANGO documentation site with executed notebook examples — design

Date: 2026-10-07 · Status: draft for review · Branch: `docs/notebooks` (stacked on
`feat/discoverable-api`, PR #1). Builds on the 0.2 API described in
`2026-10-07-discoverable-api-design.md`.

## 1. Goal

A Read the Docs site for `space-mango` whose examples are Jupyter notebooks executed on every
build, so a broken example fails CI. Readers are space plasma physicists; the examples show
the two primary use cases (statistical studies, event context) on real MANGO data.

Success: `sphinx-build -W` in CI executes every notebook against a real-data sample with no
network access, and fails on any cell error or warning; the same build runs on Read the Docs.

## 2. Decisions (2026-10-07)

| # | Decision |
|---|---|
| E1 | Notebooks run against a **small real-data sample**, not synthetic data and not the live server (still 0.1, network-dependent). |
| E2 | The sample lives **in the repo**, `docs/data/`, **≤ 10 MB** total. |
| E3 | Work lands as a **stacked PR**: branch `docs/notebooks` off `feat/discoverable-api`, PR targets it, retargeted to `main` after #1 merges. |
| E4 | Stack: **Sphinx + myst-nb + furo** (Sphinx like speasy; myst-nb executes at build). |

## 3. The docs sample

- Layout identical to production: `docs/data/<region>/SC=<sc>/part-0.parquet`, zstd.
- **Statistical part:** ~300 one-hour windows at random times (fixed seed) across
  2007–2021, spread over spacecraft and all three regions, so histograms and maps cover the
  real orbit geometry.
- **Event part:** 2–3 contiguous days of THA crossing magnetosphere → magnetosheath → solar
  wind, chosen from the data so the timeline example shows real boundary crossings.
- **Extraction:** `scripts/make_docs_sample.py` — read-only `GET /api/v1/regions/{r}/data`
  calls to the current public server (0.1 API: `spacecraft`, `time_min`, `time_max`,
  `format=arrow`), written under `docs/data/`. No kaa, no other writes. The script is
  committed so the sample can be regenerated; it prints the total size and refuses to
  write more than 10 MB.
- **Labelling:** served as dataset version `2026.0-docs-sample`
  (`MANGO_DATASET_VERSION`), so `r.version` and `r.cite()` in the rendered docs say it is a
  sample. Each notebook's first markdown cell says the outputs come from a small sample.

## 4. How notebooks reach the data

- New client behaviour: the default client's base URL is `$SPACE_MANGO_URL` when set,
  else `DEFAULT_URL`. (Also useful to self-hosters.) Tested.
- `docs/conf.py` starts `mango serve --data-dir docs/data --host 127.0.0.1 --port <free>`
  as a subprocess with `MANGO_DATASET_VERSION=2026.0-docs-sample`, waits for `/health`, sets
  `SPACE_MANGO_URL` and `SPACE_MANGO_CACHE_DIR=<tmp>` for the kernels, and stops the server
  at build end. Notebook cells therefore contain only what a user would type
  (`import space_mango as mango`, …) — no hidden setup cells.

## 5. Notebooks (`docs/examples/`, stored with outputs stripped)

1. **Getting started** — `regions`, `mango.magnetosheath`, `describe`, `spacecraft`, `search`,
   `count`, a first `get_data`.
2. **Statistical study** — magnetosheath density and |B| vs IMF Bz, binned in d_msh
   (`R_norm`); matplotlib histograms and a map in normalized coordinates (`X/Y_gsm_norm`).
3. **Event context** — `timeline` for THA over the event window; |B| and Np with region
   shading; `to_intervals()` as a crossing list.
4. **Working with results** — `to_pandas`/`to_xarray`, `metadata` and units, `query`,
   `cite()`.
5. **Cache & offline** — `count()` sizes (`est_mb` vs `download_mb_estimate`),
   `cache.info()`, `offline=True`, errors with suggestions (e.g. `MMS1`).

Science text in the notebooks stays factual about what the code does; no physical claims
beyond the column dictionary. Cell outputs must be deterministic enough to build twice the
same way (fixed seeds, no wall-clock values printed).

## 6. Site structure (`docs/`)

`index` · Installation · Quick start · Examples (the 5 notebooks) · User guide (content of
the current `docs/usage.md`: regions, columns & frames, filters, time conventions, errors,
cache — `docs/usage.md` becomes `docs/user_guide.md` and the README link is updated) ·
API reference (`sphinx.ext.autodoc` + `napoleon` over the public API: module functions,
`MangoClient`, `MangoResult`, region objects, errors) · Citing MANGO · Changelog (0.2).

`docs/superpowers/` is excluded from the site.

## 7. Build, CI, Read the Docs

- `pyproject.toml`: extra `docs = [sphinx, myst-nb, furo, matplotlib, ipykernel]` (plus the
  `server`, `pandas`, `xarray` extras the notebooks need).
- `conf.py`: `nb_execution_mode = "force"`, `nb_execution_raise_on_error = True`,
  `nb_execution_timeout = 300`; `exclude_patterns` for `superpowers/`, `data/`.
- CI: new job `docs` (one Python version, 3.13): `uv sync --all-extras`, check notebooks have
  no stored outputs (`nbstripout --verify` or an equivalent check script), then
  `uv run sphinx-build -W --keep-going -b html docs docs/_build/html`.
- `.readthedocs.yaml`: Ubuntu, Python 3.13, install with the `docs` extra, `sphinx.configuration:
  docs/conf.py`, `fail_on_warning: true`.
- Activating the project on readthedocs.org needs a repository admin — outside this work.

## 8. Testing

- Unit test for `SPACE_MANGO_URL`.
- A test that the sample respects the layout and the 10 MB cap, and that every region,
  the THA event window and ≥ 2 spacecraft per region are present.
- The docs build itself is the integration test for the notebooks.

## 9. Non-goals

Gallery thumbnails, versioned docs, speasy/SciQLop integration examples, a hosted Binder,
translating the spec documents into site pages.

## 10. Open items

- Read the Docs project activation (repo admin).
- The SWI-frame caveat stays a placeholder until the PI decides (spec §9 of the API design).
