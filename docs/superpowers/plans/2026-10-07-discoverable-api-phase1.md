# MANGO Discoverable API (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `space-mango` discoverable and safe for space plasma physicists: describe/search/count before downloading, a result object that keeps units and citation, event-context timelines, clear errors, and an on-disk cache.

**Architecture:** The catalog (`models.py`: `REGIONS`, `COLUMNS`, `RANGE_FILTERS`) stays the single source of truth and becomes plain dataclasses usable by the client. A shared `filtering.py` builds polars filter expressions for both the server and the client cache. The server gains read-only discovery endpoints and stops ignoring anything silently (HTTP 400 with a machine-readable code). The client wraps results in `MangoResult`, caches per-column monthly Parquet fragments, and exposes region objects with generated signatures.

**Tech Stack:** Python ≥ 3.11, polars, pyarrow, httpx, platformdirs (client); FastAPI, pydantic (server extra); pytest, ruff, basedpyright, codespell; uv.

**Spec:** `docs/superpowers/specs/2026-10-07-discoverable-api-design.md` (+ `2026-10-07-mango-column-dictionary-draft.md`). Read both before starting any task.

## Global Constraints

- Work on branch `feat/discoverable-api`, created from `docs/discoverable-api-spec`. Never commit on `main`. One commit per task (more is fine).
- **Served column names are frozen.** Never rename a data column. `MMS` stays `MMS`.
- Client import path (`import space_mango`) must not import pydantic, fastapi or pandas. Client runtime deps: `polars`, `pyarrow`, `httpx`, `platformdirs` only. pandas/xarray are lazy-imported optional extras.
- Existing endpoints keep their response shapes (`/regions` → `list[str]`, `/regions/{r}/columns` → `list[str]`, `/regions/{r}/filters` → list of filter dicts, `/regions/{r}/info`). 0.1.1 clients are in the wild.
- The server never silently ignores a parameter, filter, spacecraft or column: HTTP 400, body `{"detail": {"error": <code>, "message": <str>, "valid": <list[str]>}}`.
- Time windows: `start` inclusive, `stop` exclusive. Legacy `time_min`/`time_max` remain inclusive on both ends.
- Each task ends with all four green: `uv run pytest`, `uv run ruff check .`, `uv run basedpyright -p pyproject.toml`, `uv run codespell`.
  - On pharemer `uv` is not on PATH: use `/tmp/claude-3255/-home-aunai-Documents-code-mango/54bc4848-b79b-4c98-ae31-b3dfd5768acd/scratchpad/uvenv/bin/uv` wherever this plan says `uv`.
  - Always pass `-p pyproject.toml` to basedpyright: a `pyrightconfig.json` in `~/Documents/code/` otherwise hijacks it and scans PHARE.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01T6RBwQQKRfYYqokQoJQWPP
  ```

## Review Focus

1. A user types the old documented name `spacecraft=["MMS1"]` → `UnknownSpacecraftError` whose message says `Did you mean 'MMS'?` (server and cache paths). Tests: Task 5 (server), Task 6 (client), Task 10 (cache path).
2. Timezone-aware times (`datetime(..., tzinfo=UTC+2)`, `pd.Timestamp(..., tz=...)`) and partial dates (`"2017-01"`) → converted to naive UTC, never a 500. Tests: Task 6.
3. Server dataset version changes between two sessions → cache never mixes versions (new directory). Test: Task 10.
4. A query range covering months where a spacecraft has no data → fragments recorded as empty, no re-download on the next call. Test: Task 10.
5. Cache path and server path return identical rows for the same query (same filters, flags, time bounds incl. legacy inclusive `time_max`). Test: Task 10.

---

## File structure (after Phase 1)

```
src/space_mango/
  models.py              catalog dataclasses: Region, Format, RangeFilter, RegionInfo, ColumnInfo,
                         REGIONS, COLUMNS, RANGE_FILTERS, filters_for(), columns_for(),
                         DEFAULT_DATASET_VERSION, citation_bibtex()          (no pydantic)
  errors.py              QueryError (server/shared) + MangoError hierarchy + did_you_mean()
  filtering.py           RESERVED_PARAMS, parse_time(), time_window(), parse_range_params(),
                         build_filter_exprs()                                (shared)
  timeparse.py           TimeLike, to_iso()                                  (client)
  result.py              MangoResult
  cache.py               FragmentCache, months_between(), next_month(), contiguous_runs()
  regions.py             RegionAPI base class
  _codegen.py            render_regions_module()
  _regions_generated.py  generated region classes + REGION_APIS (checked by a test)
  client.py              MangoClient
  __init__.py            module-level API
  dataset.py             MangoDataset (server)
  app.py                 create_app(): error handler, version header, lifespan check
  routes/schemas.py      pydantic response models (server)
  routes/data.py         region routes
  routes/dataset.py      /timeline, /dataset
tests/
  conftest.py            served-schema fixtures + factories
  test_ci_config.py, test_filters_golden.py, test_catalog.py, test_filtering.py,
  test_errors.py, test_endpoints.py, test_result.py, test_discovery.py, test_cache.py,
  test_regions.py, test_client.py, test_placeholder.py, test_readme_examples.py
docs/usage.md
```

---

### Task 1: Make CI and local lint green

CI calls `devtools/lint.py`, which does not exist, so nothing has been enforced. Replace it with direct tool calls and fix the 12 ruff, 8 basedpyright and 3 codespell findings.

**Files:**
- Modify: `pyproject.toml`, `Makefile:15`, `.github/workflows/ci.yml:59-60`
- Modify: `src/space_mango/client.py`, `src/space_mango/__init__.py:69`, `src/space_mango/dataset.py`, `src/space_mango/routes/data.py`, `scripts/convert_pickles.py:32`
- Modify: `tests/test_client.py`, `tests/test_placeholder.py`, `tests/test_readme_examples.py`
- Create: `tests/test_ci_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MangoDataset.query(..., raw_params: Mapping[str, str | None], ...)`; `_validate_filters(filters: Mapping[str, object], ...)`; `MangoClient.filters() -> list[dict[str, object]]`.

- [ ] **Step 1: Create the branch**

```bash
git switch docs/discoverable-api-spec && git switch -c feat/discoverable-api
```

- [ ] **Step 2: Write the failing test** — `tests/test_ci_config.py`

```python
"""CI must call tools that exist (it used to call a missing devtools/lint.py)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ci_does_not_call_missing_lint_script():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    for text in (ci, makefile):
        assert "devtools/lint.py" not in text
        assert "ruff check" in text
        assert "basedpyright -p pyproject.toml" in text
        assert "codespell" in text
```

- [ ] **Step 3: Run it, expect FAIL**

Run: `uv run pytest tests/test_ci_config.py -v` → FAIL (`devtools/lint.py` found).

- [ ] **Step 4: Fix CI and Makefile**

`.github/workflows/ci.yml`: replace the step

```yaml
      - name: Run linting
        run: uv run python devtools/lint.py
```

with

```yaml
      - name: Lint (ruff)
        run: uv run ruff check .

      - name: Type check (basedpyright)
        run: uv run basedpyright -p pyproject.toml

      - name: Spelling (codespell)
        run: uv run codespell
```

`Makefile`: replace the `lint:` recipe with

```make
lint:
	uv run ruff check .
	uv run basedpyright -p pyproject.toml
	uv run codespell
```

- [ ] **Step 5: Configure ruff and codespell** — in `pyproject.toml`

Add after the `[tool.ruff.lint]` table:

```toml
[tool.ruff.lint.flake8-bugbear]
# FastAPI's dependency-injection idiom puts Depends()/Query() in defaults on purpose.
extend-immutable-calls = ["fastapi.Depends", "fastapi.Query", "fastapi.params.Depends", "fastapi.params.Query"]
```

Change `[tool.codespell]` to:

```toml
[tool.codespell]
ignore-words-list = "msh,msp,tha"
```

- [ ] **Step 6: Auto-fix and fix the rest by hand**

Run: `uv run ruff check . --fix` (fixes I001, F401, F541).

Then, in `tests/test_readme_examples.py`, the last test becomes:

```python
def test_get_data_invalid_region(client):
    with pytest.raises(httpx.HTTPStatusError):
        client.get_data("invalid_region")
```

basedpyright fixes:
- `src/space_mango/__init__.py`: `def filters(region: str) -> list[dict[str, object]]:`
- `src/space_mango/client.py`:
  - add `from collections.abc import Mapping`;
  - `_validate_filters(filters: Mapping[str, object], ...)`;
  - replace the `try: cleaned[key] = float(value)` block with:
    ```python
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise MangoFilterError(f"Filter '{key}' value must be numeric, got {value!r}.")
        try:
            cleaned[key] = float(value)
        except ValueError:
            raise MangoFilterError(
                f"Filter '{key}' value must be numeric, got {value!r}."
            ) from None
    ```
  - `def filters(self, region: str) -> list[dict[str, object]]:`
- `src/space_mango/dataset.py`: add `from collections.abc import Mapping`; type `raw_params: Mapping[str, str | None]` in both `_apply_range_filters` and `MangoDataset.query`.
- tests: every `rows: list[dict]` → `rows: list[dict[str, object]]`.

- [ ] **Step 7: Run everything, expect all green**

Run: `uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell`
Expected: 41 passed; ruff "All checks passed!"; basedpyright "0 errors"; codespell no output.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "ci: call ruff/basedpyright/codespell directly; fix existing lint findings"
```

---

### Task 2: Shared test fixtures with the served schema

The three test files each copy a fixture builder, and their fixture data uses the *intended* schema (`D_msh`, `Tilt`), which is why tests never caught the dead filters. Replace them with one `conftest.py` whose schema is copied from the live server. This task does not change behaviour.

**Files:**
- Create: `tests/conftest.py`
- Modify: `src/space_mango/client.py` (constructor gains `timeout`, `transport`)
- Rewrite: `tests/test_client.py`, `tests/test_placeholder.py`
- Modify: `tests/test_readme_examples.py` (delete local helpers, rows and `client` fixture)

**Interfaces:**
- Consumes: Task 1.
- Produces (fixtures, used by all later tasks):
  - `SERVED_COLUMNS: dict[str, list[str]]` (module constant in conftest; *not* derived from `space_mango.models`)
  - `make_row(region, sc, time, **values) -> dict[str, object]` (fixture returning the function)
  - `make_dataset(rows_by_region: dict[str, list[dict[str, object]]]) -> Path` (function-scoped factory fixture)
  - `make_api(data_dir: Path) -> TestClient`, `make_client(data_dir: Path) -> MangoClient` (factory fixtures)
  - `dataset_dir: Path`, `api: TestClient`, `client: MangoClient` (session fixtures on the standard data below)
  - `MangoClient(base_url=DEFAULT_URL, *, timeout: float = 120.0, transport: httpx.BaseTransport | None = None)`

Standard data (rows per region, one parquet file per spacecraft):

| region | SC | Time | key values |
|---|---|---|---|
| magnetosheath | THA | 2016-03-15 10:00 | R_norm 0.4, Bz_imf −5, Pd_sw 4, SW_pairing T, Norma_pos T |
| magnetosheath | MMS | 2018-07-20 14:30 | R_norm 0.8, Bz_imf 2, Pd_sw 1, SW_pairing F, Norma_pos F |
| magnetosheath | C1 | 2019-01-05 08:00 | R_norm 0.2, Bz_imf −8, Pd_sw 6, SW_pairing T, Norma_pos T |
| magnetosphere | MMS | 2015-06-10 12:00 | R_norm 0.3, tilt 0.15, Bz_imf −3 |
| magnetosphere | THA | 2017-11-03 06:00 | R_norm 0.6, tilt −0.1, Bz_imf 1 |
| magnetosphere | C3 | 2021-02-14 18:00 | R_norm 0.9, tilt 0.05, Bz_imf −1, SW_pairing F, Norma_pos F |
| solar_wind | THA | 2016-05-01 00:00 | Np 5 |

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared fixtures: a small Hive-partitioned dataset with the schema the live server serves.

SERVED_COLUMNS is copied from GET /api/v1/regions/{r}/columns on the public server
(2026-10-07). Do NOT derive it from space_mango.models: it is the ground truth the
catalog is tested against.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
from fastapi.testclient import TestClient

from space_mango.app import create_app
from space_mango.client import MangoClient
from space_mango.dataset import MangoDataset, get_dataset

Row = dict[str, object]

_BASE = ["Time", "Bx", "By", "Bz", "Np", "Vx", "Vy", "Vz", "Tp", "X_gsm", "Y_gsm", "Z_gsm"]
_PAIRED = [
    "SW_pairing", "Bx_imf", "By_imf", "Bz_imf", "Np_sw", "Vx_sw", "Vy_sw", "Vz_sw",
    "Tp_sw", "Pd_sw", "Beta_sw", "Ma_sw", "R_mp", "R_norm", "Norma_pos",
    "X_gsm_norm", "Y_gsm_norm", "Z_gsm_norm",
]
_SWI = [
    "R_bs", "Bx_swi", "By_swi", "Bz_swi", "Vx_swi", "Vy_swi", "Vz_swi",
    "X_swi_norm", "Y_swi_norm", "Z_swi_norm",
]
# "SC" is not listed: it comes from the SC=<name> directory (Hive partition key).
SERVED_COLUMNS: dict[str, list[str]] = {
    "solar_wind": _BASE,
    "magnetosphere": _BASE + _PAIRED + ["tilt"],
    "magnetosheath": _BASE + _PAIRED + _SWI,
}
_BOOL_COLUMNS = {"SW_pairing", "Norma_pos"}


def _make_row(region: str, sc: str, time: datetime, **values: object) -> Row:
    cols = SERVED_COLUMNS[region]
    unknown = set(values) - set(cols)
    if unknown:
        raise KeyError(f"{sorted(unknown)} are not served in region {region!r}")
    row: Row = {c: (True if c in _BOOL_COLUMNS else 1.0) for c in cols}
    row["Time"] = time
    row["SC"] = sc
    row.update(values)
    return row


def _write_region(base: Path, region: str, rows: list[Row]) -> None:
    df = pl.DataFrame(rows).with_columns(pl.col("Time").cast(pl.Datetime("ns")))
    for sc in df["SC"].unique().to_list():
        sc_dir = base / region / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("SC") == sc).drop("SC").write_parquet(sc_dir / "part-0.parquet")


def _write_dataset(base: Path, rows_by_region: dict[str, list[Row]]) -> Path:
    for region, rows in rows_by_region.items():
        _write_region(base, region, rows)
    return base


def _api(data_dir: Path) -> TestClient:
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    return TestClient(app)


def _client(data_dir: Path) -> MangoClient:
    tc = _api(data_dir)
    return MangoClient("http://testserver", transport=tc._transport)


def standard_rows() -> dict[str, list[Row]]:
    r = _make_row
    msh = "magnetosheath"
    msp = "magnetosphere"
    return {
        msh: [
            r(msh, "THA", datetime(2016, 3, 15, 10), Bx=1.0, By=2.0, Bz=3.0, Np=10.0,
              Vx=-200.0, Vy=0.0, Vz=0.0, Tp=1e6, X_gsm=8.0, Y_gsm=3.0, Z_gsm=0.0,
              R_norm=0.4, SW_pairing=True, Bz_imf=-5.0, By_imf=1.0, Bx_imf=0.5,
              Pd_sw=4.0, Np_sw=8.0, Tp_sw=1e5, Vx_sw=-400.0, Beta_sw=1.2, Ma_sw=7.0,
              Norma_pos=True),
            r(msh, "MMS", datetime(2018, 7, 20, 14, 30), Bx=2.0, By=3.0, Bz=4.0, Np=20.0,
              Vx=-300.0, Vy=1.0, Vz=1.0, Tp=2e6, X_gsm=9.0, Y_gsm=4.0, Z_gsm=1.0,
              R_norm=0.8, SW_pairing=False, Bz_imf=2.0, By_imf=-1.0, Bx_imf=-0.3,
              Pd_sw=1.0, Np_sw=5.0, Tp_sw=5e4, Vx_sw=-350.0, Beta_sw=0.8, Ma_sw=5.0,
              Norma_pos=False),
            r(msh, "C1", datetime(2019, 1, 5, 8), Bx=0.5, By=-1.0, Bz=-2.0, Np=15.0,
              Vx=-250.0, Vy=-0.5, Vz=0.5, Tp=1.5e6, X_gsm=10.0, Y_gsm=-2.0, Z_gsm=0.5,
              R_norm=0.2, SW_pairing=True, Bz_imf=-8.0, By_imf=3.0, Bx_imf=1.0,
              Pd_sw=6.0, Np_sw=12.0, Tp_sw=2e5, Vx_sw=-500.0, Beta_sw=2.0, Ma_sw=10.0,
              Norma_pos=True),
        ],
        msp: [
            r(msp, "MMS", datetime(2015, 6, 10, 12), Bx=10.0, By=-5.0, Bz=-20.0, Np=1.0,
              Vx=-50.0, Vy=10.0, Vz=5.0, Tp=5e7, X_gsm=-5.0, Y_gsm=2.0, Z_gsm=1.0,
              R_norm=0.3, tilt=0.15, SW_pairing=True, Bz_imf=-3.0, By_imf=0.0, Bx_imf=0.0,
              Pd_sw=2.0, Np_sw=6.0, Tp_sw=1e5, Vx_sw=-380.0, Beta_sw=1.0, Ma_sw=6.0,
              Norma_pos=True),
            r(msp, "THA", datetime(2017, 11, 3, 6), Bx=15.0, By=3.0, Bz=-30.0, Np=0.5,
              Vx=-30.0, Vy=5.0, Vz=-2.0, Tp=8e7, X_gsm=-8.0, Y_gsm=-1.0, Z_gsm=-0.5,
              R_norm=0.6, tilt=-0.1, SW_pairing=True, Bz_imf=1.0, By_imf=2.0, Bx_imf=-1.0,
              Pd_sw=3.0, Np_sw=7.0, Tp_sw=1.5e5, Vx_sw=-420.0, Beta_sw=1.5, Ma_sw=8.0,
              Norma_pos=True),
            r(msp, "C3", datetime(2021, 2, 14, 18), Bx=5.0, By=-2.0, Bz=-10.0, Np=2.0,
              Vx=-80.0, Vy=0.0, Vz=0.0, Tp=3e7, X_gsm=-3.0, Y_gsm=5.0, Z_gsm=2.0,
              R_norm=0.9, tilt=0.05, SW_pairing=False, Bz_imf=-1.0, By_imf=-3.0, Bx_imf=0.5,
              Pd_sw=1.5, Np_sw=4.0, Tp_sw=8e4, Vx_sw=-360.0, Beta_sw=0.6, Ma_sw=4.0,
              Norma_pos=False),
        ],
        "solar_wind": [
            r("solar_wind", "THA", datetime(2016, 5, 1), Bx=0.1, By=-0.5, Bz=-1.0, Np=5.0,
              Vx=-400.0, Vy=0.0, Vz=0.0, Tp=1e5, X_gsm=20.0, Y_gsm=0.0, Z_gsm=0.0),
        ],
    }


@pytest.fixture
def make_row() -> Callable[..., Row]:
    return _make_row


@pytest.fixture
def make_dataset(tmp_path: Path) -> Callable[[dict[str, list[Row]]], Path]:
    counter = iter(range(1_000))

    def factory(rows_by_region: dict[str, list[Row]]) -> Path:
        return _write_dataset(tmp_path / f"data{next(counter)}", rows_by_region)

    return factory


@pytest.fixture
def make_api() -> Callable[[Path], TestClient]:
    return _api


@pytest.fixture
def make_client() -> Callable[[Path], MangoClient]:
    return _client


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("mango_data"), standard_rows())


@pytest.fixture(scope="session")
def api(dataset_dir: Path) -> TestClient:
    return _api(dataset_dir)


@pytest.fixture(scope="session")
def client(dataset_dir: Path) -> MangoClient:
    return _client(dataset_dir)
```

- [ ] **Step 2: Give `MangoClient` a proper constructor** — `src/space_mango/client.py`

```python
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout, transport=transport)
        self._filter_cache: dict[str, set[str]] = {}
```

- [ ] **Step 3: Rewrite `tests/test_client.py`**

Keep the five `test_validate_filters_*` tests and the two `*_FILTERS` sets unchanged. Delete `_write_test_region`, `MAGNETOSHEATH_ROWS`, `_make_test_mango_client` and the four `test_client_*` tests, then add:

```python
def test_client_get_data_all(client):
    df = client.get_data("magnetosheath", limit=10)
    assert len(df) == 3
    assert "SC" in df.columns
    assert "Bz_imf" in df.columns


def test_client_get_data_spacecraft_filter(client):
    df = client.get_data("magnetosheath", spacecraft=["THA"], limit=10)
    assert len(df) == 1
    assert df["SC"][0] == "THA"


def test_client_get_data_range_filter(client):
    df = client.get_data("magnetosheath", bz_imf_max=-1.0, limit=10)
    assert set(df["SC"].to_list()) == {"THA", "C1"}


def test_client_regions(client):
    assert "magnetosheath" in client.regions()
```

Remove now-unused imports (`tempfile`, `datetime`, `Path`, `httpx`, `polars`, `TestClient`, `sm`, `create_app`, `MangoDataset`, `get_dataset`).

- [ ] **Step 4: Rewrite `tests/test_placeholder.py`**

Keep the module-level `client = TestClient(create_app())` and the five tests that use it (`test_health`, `test_list_regions`, the three `test_filters_*`). Delete `_write_test_region`, `MAGNETOSHEATH_ROWS`, `_make_test_client` and the five `test_hive_*`/`test_data_endpoint_*` tests, then add:

```python
def test_hive_dataset_query_all(dataset_dir):
    df = MangoDataset(dataset_dir).query(Region.magnetosheath, {}, limit=100)
    assert len(df) == 3
    assert {"SC", "Time"} <= set(df.columns)


def test_hive_dataset_query_spacecraft_filter(dataset_dir):
    df = MangoDataset(dataset_dir).query(Region.magnetosheath, {}, spacecraft=["THA"], limit=100)
    assert df["SC"].to_list() == ["THA"]


def test_hive_dataset_query_time_filter(dataset_dir):
    df = MangoDataset(dataset_dir).query(
        Region.magnetosheath, {}, time_min="2017-01-01T00:00:00", limit=100
    )
    assert set(df["SC"].to_list()) == {"MMS", "C1"}


def test_data_endpoint_returns_csv(api):
    r = api.get("/api/v1/regions/magnetosheath/data?format=csv&limit=10")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert len(r.text.strip().splitlines()) == 4  # header + 3 rows


def test_data_endpoint_spacecraft_filter(api):
    r = api.get("/api/v1/regions/magnetosheath/data?format=csv&spacecraft=THA&limit=10")
    lines = r.text.strip().splitlines()
    assert len(lines) == 2
    assert "THA" in lines[1]
```

Note: the module-level variable `client` in this file shadows the conftest `client` fixture only inside this module; the new tests use `api` and `dataset_dir`, not `client`.

- [ ] **Step 5: Trim `tests/test_readme_examples.py`**

Delete `_write_region`, `_make_client`, `MAGNETOSHEATH_ROWS`, `MAGNETOSPHERE_ROWS`, `SOLAR_WIND_ROWS` and the module-level `client` fixture. All tests keep their bodies; they now receive the session `client` fixture from conftest, whose data is the same rows (with `R_norm`/`tilt` instead of `D_msh`/`D_msp`/`Tilt`). Remove unused imports.

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell` → all green.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "test: shared conftest fixtures using the served schema"
```

---

### Task 3: Fix the four dead filters; a filter on a missing column is HTTP 400

`d_msh`, `d_msp` and `tilt` (both regions) silently return unfiltered data on the live server, because `dataset.py:21-22` skips filters whose column is missing. `R_norm` is verified to be the d_msh/d_msp quantity (spec D3).

**Files:**
- Create: `src/space_mango/errors.py`, `tests/test_filters_golden.py`
- Modify: `src/space_mango/models.py` (`RANGE_FILTERS`), `src/space_mango/dataset.py`, `src/space_mango/app.py`
- Modify: `tests/test_client.py` (`MAGNETOSHEATH_FILTERS` loses `tilt`)

**Interfaces:**
- Consumes: Task 2 fixtures.
- Produces: `class QueryError(ValueError)` with `.code: str`, `.message: str`, `.valid: list[str]`, `.to_dict() -> dict[str, object]`; `did_you_mean(name: str, choices: Iterable[str]) -> str`; app-wide handler turning `QueryError` into HTTP 400.

- [ ] **Step 1: Write the failing golden tests** — `tests/test_filters_golden.py`

```python
"""Each of the four filters that were dead on the live server must actually filter."""

import pytest


@pytest.mark.parametrize(
    ("region", "params", "expected_sc"),
    [
        ("magnetosheath", {"d_msh_max": 0.3}, {"C1"}),
        ("magnetosphere", {"d_msp_min": 0.5}, {"THA", "C3"}),
        ("magnetosphere", {"tilt_min": 0.1}, {"MMS"}),
        ("magnetosphere", {"tilt_max": 0.0}, {"THA"}),
    ],
)
def test_previously_dead_filter_filters(client, region, params, expected_sc):
    df = client.get_data(region, limit=100, **params)
    assert set(df["SC"].to_list()) == expected_sc


def test_tilt_is_not_a_magnetosheath_filter(client):
    names = {f["name"] for f in client.filters("magnetosheath")}
    assert "tilt" not in names


def test_filter_on_missing_column_is_400(make_row, make_dataset, make_api):
    from datetime import datetime

    row = make_row("magnetosheath", "THA", datetime(2016, 1, 1))
    del row["R_norm"]
    api = make_api(make_dataset({"magnetosheath": [row]}))
    r = api.get("/api/v1/regions/magnetosheath/data", params={"d_msh_max": 0.3})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "filter_column_missing"
    assert "R_norm" in detail["message"]
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_filters_golden.py -v` → the parametrized cases return all rows; `tilt` still listed; missing column returns 200.

- [ ] **Step 3: Create `src/space_mango/errors.py`**

```python
"""Errors shared by the MANGO server and client."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def did_you_mean(name: str, choices: Iterable[str]) -> str:
    """Return " Did you mean 'x'?" for the closest choice, or "" if nothing is close."""
    match = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f" Did you mean '{match[0]}'?" if match else ""


class QueryError(ValueError):
    """A request the server refuses (HTTP 400) instead of silently ignoring."""

    def __init__(self, code: str, message: str, valid: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.valid = sorted(valid)

    def to_dict(self) -> dict[str, object]:
        return {"error": self.code, "message": self.message, "valid": self.valid}
```

- [ ] **Step 4: Fix the catalog** — in `src/space_mango/models.py` replace the `tilt`, `vx_sw`, `d_msp`, `d_msh` entries:

```python
    "vx_sw": RangeFilter(
        column="Vx_sw", unit="km/s",
        description="Solar wind velocity X (GSM); negative (anti-sunward), so faster wind is more negative",
        regions=frozenset({Region.magnetosphere, Region.magnetosheath}),
    ),
```

```python
    "tilt": RangeFilter(
        column="tilt", unit="rad",
        description="Dipole tilt angle (positive near June solstice)",
        regions=frozenset({Region.magnetosphere}),
    ),
```

```python
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
```

- [ ] **Step 5: Stop skipping, raise instead** — `src/space_mango/dataset.py`, in `_apply_range_filters` replace

```python
        if filt.column not in available:
            continue
        lo = raw_params.get(f"{name}_min")
        hi = raw_params.get(f"{name}_max")
```

with

```python
        lo = raw_params.get(f"{name}_min")
        hi = raw_params.get(f"{name}_max")
        if (lo is not None or hi is not None) and filt.column not in available:
            raise QueryError(
                "filter_column_missing",
                f"Filter '{name}' needs column '{filt.column}', "
                f"which region '{region}' does not have.",
            )
```

and add `from space_mango.errors import QueryError`.

- [ ] **Step 6: Map `QueryError` to HTTP 400** — `src/space_mango/app.py`

```python
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from space_mango.errors import QueryError
from space_mango.routes import data, health


async def _query_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, QueryError)
    return JSONResponse(status_code=400, content={"detail": exc.to_dict()})


def create_app() -> FastAPI:
    app = FastAPI(
        title="MANGO",
        description="Magnetosphere Atlas from Normalized Geospace Observations — data subsetting API",
        version="0.1.0",
        root_path=os.environ.get("MANGO_ROOT_PATH", ""),
    )
    app.add_exception_handler(QueryError, _query_error_handler)
    app.include_router(health.router)
    app.include_router(data.router, prefix="/api/v1")
    return app
```

- [ ] **Step 7: Update the expected filter set** — `tests/test_client.py`: remove `"tilt"` from `MAGNETOSHEATH_FILTERS`.

- [ ] **Step 8: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "fix: d_msh/d_msp/tilt filters point at served columns; missing filter column is HTTP 400"
```

---

### Task 4: Catalog as plain dataclasses, with regions and column metadata

**Files:**
- Rewrite: `src/space_mango/models.py`
- Create: `src/space_mango/routes/schemas.py`, `tests/test_catalog.py`
- Modify: `src/space_mango/routes/data.py` (import `DatasetInfo`, `FilterInfo` from `routes.schemas`)

**Interfaces:**
- Consumes: Task 3 `RANGE_FILTERS` content (keep every entry and its text exactly; only the container type changes).
- Produces:
  - `@dataclass(frozen=True) RangeFilter(column: str, unit: str, description: str, regions: frozenset[Region] = frozenset(Region))`
  - `@dataclass(frozen=True) RegionInfo(definition: str)`; `REGIONS: dict[Region, RegionInfo]`
  - `@dataclass(frozen=True) ColumnInfo(unit, frame, description, computed: str, regions: frozenset[Region], per_region: tuple[tuple[Region, str], ...] = ())` with `.description_for(region) -> str`
  - `COLUMNS: dict[str, ColumnInfo]`
  - `filters_for(region: Region | str) -> dict[str, RangeFilter]`; `columns_for(region: Region | str) -> dict[str, ColumnInfo]`; `filter_for_column(region, column) -> str | None`
  - `DEFAULT_DATASET_VERSION = "2026.0"`; `DATASET_TITLE: str`; `citation_bibtex(version: str, doi: str | None) -> str`
  - `routes/schemas.py`: pydantic `DatasetInfo`, `FilterInfo` (moved, unchanged fields)

- [ ] **Step 1: Write the failing contract tests** — `tests/test_catalog.py`

```python
"""The catalog must describe exactly what the server serves (SERVED_COLUMNS in conftest)."""

import subprocess
import sys

from conftest import SERVED_COLUMNS

from space_mango.models import (
    COLUMNS,
    RANGE_FILTERS,
    REGIONS,
    Region,
    citation_bibtex,
    columns_for,
    filter_for_column,
    filters_for,
)


def test_every_region_has_a_definition():
    assert set(REGIONS) == set(Region)
    assert all(info.definition for info in REGIONS.values())


def test_catalog_columns_match_served_columns():
    for region in Region:
        assert set(columns_for(region)) == set(SERVED_COLUMNS[region.value]) | {"SC"}, region


def test_every_column_is_documented():
    for name, col in COLUMNS.items():
        for region in col.regions:
            assert col.description_for(region), name


def test_filters_point_at_documented_columns_in_their_regions():
    for name, f in RANGE_FILTERS.items():
        col = COLUMNS[f.column]
        assert f.regions <= col.regions, name
        assert f.unit == col.unit, name


def test_r_norm_is_described_per_region():
    msh = COLUMNS["R_norm"].description_for(Region.magnetosheath)
    msp = COLUMNS["R_norm"].description_for(Region.magnetosphere)
    assert "bow shock" in msh
    assert msh != msp


def test_filter_lookup_helpers():
    assert "d_msh" in filters_for("magnetosheath")
    assert "tilt" not in filters_for(Region.magnetosheath)
    assert filter_for_column("magnetosheath", "R_norm") == "d_msh"
    assert filter_for_column("magnetosphere", "R_norm") == "d_msp"
    assert filter_for_column("magnetosheath", "Bx_swi") is None


def test_citation_mentions_version_and_doi():
    text = citation_bibtex("2026.0", "10.5281/zenodo.1")
    assert "2026.0" in text and "10.5281/zenodo.1" in text and text.startswith("@")


def test_client_import_does_not_pull_server_dependencies():
    code = (
        "import sys, space_mango, space_mango.models; "
        "bad = {'pydantic', 'fastapi', 'pandas'} & set(sys.modules); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

(`from conftest import ...` works because pytest prepends `tests/` to `sys.path`; if basedpyright reports it as unresolved, add `extraPaths = ["tests"]` under `[tool.basedpyright]`.)

- [ ] **Step 2: Run, expect FAIL** (`ImportError: cannot import name 'COLUMNS'`).

- [ ] **Step 3: Rewrite `src/space_mango/models.py`**

Keep `Format`, `Region` and every `RANGE_FILTERS` entry from Task 3 verbatim, except that the `x_gsm`/`y_gsm`/`z_gsm` units become `"R_E"`. Replace the pydantic classes:

```python
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
                 "Analytic approximation (spok.get_tilt), not IGRF", _MSP),
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
                   "SWI: X = -V_sw/|V_sw|, IMF in the X-Y plane", _MSH),
    "By_swi": _col("nT", "SWI", "Local magnetic field in the SWI frame, Y",
                   "SWI: X = -V_sw/|V_sw|, IMF in the X-Y plane", _MSH),
    "Bz_swi": _col("nT", "SWI", "Local magnetic field in the SWI frame, Z",
                   "SWI: X = -V_sw/|V_sw|, IMF in the X-Y plane", _MSH),
    "Vx_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, X (aberration-corrected)",
                   "Rotated after removing Earth's 29.8 km/s orbital motion", _MSH),
    "Vy_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, Y (aberration-corrected)",
                   "Rotated after removing Earth's 29.8 km/s orbital motion", _MSH),
    "Vz_swi": _col("km/s", "SWI", "Local ion velocity in the SWI frame, Z (aberration-corrected)",
                   "Rotated after removing Earth's 29.8 km/s orbital motion", _MSH),
    "X_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, X", "SWI rotation of X/Y/Z_gsm_norm", _MSH),
    "Y_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, Y", "SWI rotation of X/Y/Z_gsm_norm", _MSH),
    "Z_swi_norm": _col("R_E", "SWI", "Normalized position in the SWI frame, Z", "SWI rotation of X/Y/Z_gsm_norm", _MSH),
}

# ---- Filter catalog: single source of truth ----

RANGE_FILTERS: dict[str, RangeFilter] = {
    # ... every entry from Task 3, unchanged, x/y/z_gsm unit "R_E" ...
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
```

(Write out the full `RANGE_FILTERS` dict — the comment above stands for "copy the Task 3 entries here", which the implementer does literally from the current file.)

- [ ] **Step 4: Create `src/space_mango/routes/schemas.py`** with the two pydantic models moved from `models.py`:

```python
"""Pydantic response models (server only)."""

from pydantic import BaseModel


class DatasetInfo(BaseModel):
    region: str
    row_count: int
    columns: list[str]


class FilterInfo(BaseModel):
    name: str
    column: str
    unit: str
    description: str
    params: str  # "min=..&max=.."
```

In `routes/data.py`: `from space_mango.models import RANGE_FILTERS, Format, Region` and `from space_mango.routes.schemas import DatasetInfo, FilterInfo`.

- [ ] **Step 5: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: catalog as dataclasses with REGIONS and COLUMNS metadata"
```

---

### Task 5: Shared filtering; the server refuses unknown parameters, spacecraft, columns and bad times

**Files:**
- Create: `src/space_mango/filtering.py`, `tests/test_filtering.py`
- Modify: `src/space_mango/dataset.py`, `src/space_mango/routes/data.py`
- Modify: `tests/test_readme_examples.py` (`test_get_data_spacecraft_no_match`, `test_get_data_column_nonexistent_ignored`)

**Interfaces:**
- Consumes: `QueryError`, `did_you_mean` (Task 3); `filters_for`, `Region` (Task 4).
- Produces:
  - `RESERVED_PARAMS: frozenset[str]`
  - `parse_time(value: str, *, param: str) -> datetime` (naive UTC; raises `QueryError("bad_time")`)
  - `time_window(start: str | None, stop: str | None, time_min: str | None, time_max: str | None) -> tuple[datetime | None, datetime | None, bool]` (last item: `stop_inclusive`)
  - `parse_range_params(region: Region | str, raw: Mapping[str, str | None]) -> dict[str, float]`
  - `build_filter_exprs(region, available: set[str], *, spacecraft=None, start=None, stop=None, stop_inclusive=False, sw_paired_only=False, normalized_only=False, ranges=None) -> list[pl.Expr]`
  - `MangoDataset.spacecraft(region) -> list[str]`; `MangoDataset.columns(region) -> list[str]`
  - `MangoDataset.query(region, raw_params, *, columns=None, spacecraft=None, start: datetime | None = None, stop: datetime | None = None, stop_inclusive: bool = False, sw_paired_only=False, normalized_only=False, limit=None) -> pl.DataFrame`
  - `MangoDataset._plan(...same minus limit...) -> pl.LazyFrame` (used by Task 7 `count`)
  - `/data` accepts `start`/`stop` (half-open) besides legacy `time_min`/`time_max`.

- [ ] **Step 1: Write the failing tests** — `tests/test_filtering.py`

```python
from datetime import datetime

import polars as pl
import pytest

from space_mango.errors import QueryError
from space_mango.filtering import (
    build_filter_exprs,
    parse_range_params,
    parse_time,
    time_window,
)


def test_parse_time_accepts_iso_and_converts_tz_to_naive_utc():
    assert parse_time("2017-01-12T10:00:00", param="start") == datetime(2017, 1, 12, 10)
    assert parse_time("2017-01-12T12:00:00+02:00", param="start") == datetime(2017, 1, 12, 10)


def test_parse_time_rejects_garbage():
    with pytest.raises(QueryError) as e:
        parse_time("yesterday", param="start")
    assert e.value.code == "bad_time"


def test_time_window_new_and_legacy():
    assert time_window("2017-01-01", "2018-01-01", None, None) == (
        datetime(2017, 1, 1), datetime(2018, 1, 1), False)
    assert time_window(None, None, "2017-01-01", "2018-01-01") == (
        datetime(2017, 1, 1), datetime(2018, 1, 1), True)


def test_parse_range_params_skips_reserved_and_validates():
    raw = {"limit": "3", "spacecraft": "THA", "bz_imf_max": "-2", "d_msh_min": "0"}
    assert parse_range_params("magnetosheath", raw) == {"bz_imf_max": -2.0, "d_msh_min": 0.0}


@pytest.mark.parametrize("key", ["foo_min", "tilt_min", "bz_imf", "d_msp_max"])
def test_parse_range_params_unknown_is_error(key):
    with pytest.raises(QueryError) as e:
        parse_range_params("magnetosheath", {key: "1"})
    assert e.value.code == "unknown_filter"
    assert "bz_imf_max" in e.value.valid


def test_parse_range_params_non_numeric():
    with pytest.raises(QueryError) as e:
        parse_range_params("magnetosheath", {"bz_imf_max": "south"})
    assert e.value.code == "bad_filter_value"


def test_build_filter_exprs_applies_everything():
    df = pl.DataFrame({
        "Time": [datetime(2016, 1, 1), datetime(2017, 1, 1), datetime(2018, 1, 1)],
        "SC": ["THA", "THA", "C1"],
        "Bz_imf": [-5.0, -1.0, -8.0],
        "SW_pairing": [True, True, False],
        "Norma_pos": [True, True, True],
    })
    exprs = build_filter_exprs(
        "magnetosheath", set(df.columns),
        spacecraft=["THA", "C1"], start=datetime(2016, 1, 1), stop=datetime(2018, 1, 1),
        sw_paired_only=True, ranges={"bz_imf_max": -2.0},
    )
    out = df.filter(pl.all_horizontal(exprs))
    assert out["Time"].to_list() == [datetime(2016, 1, 1)]


def test_stop_inclusive_flag():
    df = pl.DataFrame({"Time": [datetime(2018, 1, 1)]})
    excl = build_filter_exprs("solar_wind", {"Time"}, stop=datetime(2018, 1, 1))
    incl = build_filter_exprs("solar_wind", {"Time"}, stop=datetime(2018, 1, 1), stop_inclusive=True)
    assert df.filter(pl.all_horizontal(excl)).height == 0
    assert df.filter(pl.all_horizontal(incl)).height == 1


def test_flag_unavailable_in_region():
    with pytest.raises(QueryError) as e:
        build_filter_exprs("solar_wind", {"Time"}, sw_paired_only=True)
    assert e.value.code == "flag_unavailable"


def test_server_rejects_unknown_spacecraft_with_suggestion(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"spacecraft": "MMS1"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["error"] == "unknown_spacecraft"
    assert "Did you mean 'MMS'?" in d["message"]


def test_server_rejects_unknown_column(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"columns": ["Np", "Nope"]})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unknown_column"


def test_server_rejects_unknown_query_param(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"foo_min": 1})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unknown_filter"


def test_server_rejects_bad_time_with_400_not_500(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={"time_min": "not-a-date"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "bad_time"


def test_server_start_stop_half_open(api):
    r = api.get("/api/v1/regions/magnetosheath/data", params={
        "format": "csv", "start": "2016-03-15T10:00:00", "stop": "2018-07-20T14:30:00"})
    assert r.status_code == 200
    assert len(r.text.strip().splitlines()) == 2  # header + THA only (MMS row is at stop)
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: space_mango.filtering`).

- [ ] **Step 3: Create `src/space_mango/filtering.py`**

```python
"""Filter expressions shared by the server (dataset.py) and the client cache (cache path).

Both sides call build_filter_exprs, so a query filters identically wherever it runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

import polars as pl

from space_mango.errors import QueryError
from space_mango.models import Region, filters_for

RESERVED_PARAMS = frozenset({
    "columns", "spacecraft", "sc", "start", "stop", "time_min", "time_max",
    "sw_paired_only", "normalized_only", "limit", "format",
})


def parse_time(value: str, *, param: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise QueryError(
            "bad_time", f"{param}={value!r} is not an ISO 8601 time (e.g. 2017-01-12T10:00:00)."
        ) from None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def time_window(
    start: str | None, stop: str | None, time_min: str | None, time_max: str | None
) -> tuple[datetime | None, datetime | None, bool]:
    """Resolve new (half-open) and legacy (inclusive) time parameters."""
    lo = start if start is not None else time_min
    start_dt = parse_time(lo, param="start") if lo is not None else None
    if stop is not None:
        return start_dt, parse_time(stop, param="stop"), False
    if time_max is not None:
        return start_dt, parse_time(time_max, param="time_max"), True
    return start_dt, None, False


def parse_range_params(region: Region | str, raw: Mapping[str, str | None]) -> dict[str, float]:
    allowed = filters_for(region)
    valid = [f"{n}_{s}" for n in allowed for s in ("min", "max")]
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key in RESERVED_PARAMS or value is None:
            continue
        name, _, suffix = key.rpartition("_")
        if suffix not in ("min", "max") or name not in allowed:
            raise QueryError(
                "unknown_filter",
                f"'{key}' is not a valid parameter for region '{Region(region).value}'.",
                valid,
            )
        try:
            out[key] = float(value)
        except ValueError:
            raise QueryError("bad_filter_value", f"Filter '{key}' must be numeric, got {value!r}.") from None
    return out


_FLAGS = {"sw_paired_only": "SW_pairing", "normalized_only": "Norma_pos"}


def build_filter_exprs(
    region: Region | str,
    available: set[str],
    *,
    spacecraft: Iterable[str] | None = None,
    start: datetime | None = None,
    stop: datetime | None = None,
    stop_inclusive: bool = False,
    sw_paired_only: bool = False,
    normalized_only: bool = False,
    ranges: Mapping[str, float] | None = None,
) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    if spacecraft:
        exprs.append(pl.col("SC").is_in(list(spacecraft)))
    if start is not None:
        exprs.append(pl.col("Time") >= start)
    if stop is not None:
        exprs.append(pl.col("Time") <= stop if stop_inclusive else pl.col("Time") < stop)
    for flag, wanted in (("sw_paired_only", sw_paired_only), ("normalized_only", normalized_only)):
        if not wanted:
            continue
        column = _FLAGS[flag]
        if column not in available:
            raise QueryError(
                "flag_unavailable",
                f"{flag}=true needs column '{column}', which region '{Region(region).value}' does not have.",
            )
        exprs.append(pl.col(column))
    catalog = filters_for(region)
    for key, value in (ranges or {}).items():
        name, _, suffix = key.rpartition("_")
        filt = catalog[name]
        if filt.column not in available:
            raise QueryError(
                "filter_column_missing",
                f"Filter '{name}' needs column '{filt.column}', "
                f"which region '{Region(region).value}' does not have.",
            )
        col = pl.col(filt.column)
        exprs.append(col >= value if suffix == "min" else col <= value)
    return exprs
```

- [ ] **Step 4: Rewrite the query part of `src/space_mango/dataset.py`**

Delete `_apply_range_filters`. Replace `MangoDataset.query` with:

```python
    def spacecraft(self, region: Region | str) -> list[str]:
        path = self._dir / Region(region).value
        if not path.is_dir():
            return []
        return sorted(p.name.split("=", 1)[1] for p in path.iterdir()
                      if p.is_dir() and p.name.startswith("SC="))

    def columns(self, region: Region | str) -> list[str]:
        return self._lazy(Region(region).value).collect_schema().names()

    def _plan(
        self,
        region: Region,
        raw_params: Mapping[str, str | None],
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: datetime | None = None,
        stop: datetime | None = None,
        stop_inclusive: bool = False,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
    ) -> pl.LazyFrame:
        lf = self._lazy(region)
        available = set(lf.collect_schema().names())
        if spacecraft:
            known = self.spacecraft(region)
            for sc in spacecraft:
                if sc not in known:
                    raise QueryError(
                        "unknown_spacecraft",
                        f"'{sc}' is not a spacecraft in region '{region.value}'.{did_you_mean(sc, known)}",
                        known,
                    )
        if columns:
            for c in columns:
                if c not in available:
                    raise QueryError(
                        "unknown_column",
                        f"'{c}' is not a column of region '{region.value}'.{did_you_mean(c, available)}",
                        available,
                    )
        exprs = build_filter_exprs(
            region, available,
            spacecraft=spacecraft, start=start, stop=stop, stop_inclusive=stop_inclusive,
            sw_paired_only=sw_paired_only, normalized_only=normalized_only,
            ranges=parse_range_params(region, raw_params),
        )
        if exprs:
            lf = lf.filter(pl.all_horizontal(exprs))
        if columns:
            lf = lf.select(columns)
        return lf

    def query(
        self,
        region: Region,
        raw_params: Mapping[str, str | None],
        *,
        limit: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        lf = self._plan(region, raw_params, **kwargs)
        if limit is not None:
            lf = lf.limit(limit)
        return lf.collect()
```

Imports: `from typing import Any`, `from space_mango.errors import QueryError, did_you_mean`, `from space_mango.filtering import build_filter_exprs, parse_range_params`. `_lazy` takes `str`; `Region` is a `StrEnum` so passing it works.

Update `tests/test_placeholder.py::test_hive_dataset_query_time_filter` to call `query(..., start=datetime(2017, 1, 1), limit=100)`.

- [ ] **Step 5: Wire the route** — `src/space_mango/routes/data.py` `region_data`:

Add parameters after `time_max`:

```python
    start: str | None = Query(None, description="Start time, inclusive (ISO 8601)"),
    stop: str | None = Query(None, description="Stop time, exclusive (ISO 8601)"),
```

change the `spacecraft` description to `"Filter by spacecraft (e.g. THA, C1, MMS)"`, change `time_min`/`time_max` descriptions to `"Legacy inclusive start (use start)"` / `"Legacy inclusive end (use stop)"`, and replace the `ds.query(...)` call with:

```python
    start_dt, stop_dt, stop_inclusive = time_window(start, stop, time_min, time_max)
    df = ds.query(
        region,
        dict(request.query_params),
        limit=limit,
        columns=columns,
        spacecraft=spacecraft,
        start=start_dt,
        stop=stop_dt,
        stop_inclusive=stop_inclusive,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
    )
```

with `from space_mango.filtering import time_window`. Also update the docstring example list (keep `d_msh_max=0.3`, it now works).

- [ ] **Step 6: Adjust two README tests to the new contract** — `tests/test_readme_examples.py`

```python
def test_get_data_spacecraft_no_match(client):
    with pytest.raises(httpx.HTTPStatusError) as e:
        client.get_data("magnetosheath", spacecraft=["NONEXISTENT"])
    assert e.value.response.status_code == 400


def test_get_data_column_nonexistent_is_error(client):
    with pytest.raises(httpx.HTTPStatusError) as e:
        client.get_data("magnetosheath", columns=["Np", "DOES_NOT_EXIST"])
    assert e.value.response.status_code == 400
```

(Task 6 turns these into typed errors.)

- [ ] **Step 7: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: shared filtering; server rejects unknown params, spacecraft, columns, bad times (400)"
```

---

### Task 6: Client error hierarchy, time parsing, `start`/`stop`

**Files:**
- Modify: `src/space_mango/errors.py` (append client errors)
- Create: `src/space_mango/timeparse.py`, `tests/test_errors.py`
- Modify: `src/space_mango/client.py`, `src/space_mango/__init__.py`
- Modify: `tests/test_readme_examples.py`, `tests/test_client.py`

**Interfaces:**
- Consumes: server error codes from Tasks 3 and 5.
- Produces:
  - `MangoError(Exception)`; `UnknownRegionError`, `UnknownSpacecraftError`, `UnknownColumnError`, `MangoFilterError`, `TimeParseError` (each `(MangoError, ValueError)`); `ServerError(MangoError)`; `CacheMissError(MangoError)`
  - `error_from_response(status: int, body: object) -> MangoError`
  - `TimeLike = str | datetime | date | None` (plus duck-typed `numpy.datetime64`); `to_iso(value: object, *, param: str) -> str | None`
  - `MangoClient._get(path: str, params: Mapping[str, object] | None = None) -> httpx.Response` (raises mapped errors)
  - `MangoClient._check_region(region: str) -> str`
  - `MangoClient.get_data(region, *, columns=None, spacecraft=None, start=None, stop=None, sw_paired_only=False, normalized_only=False, limit=None, time_min=None, time_max=None, **filters) -> pl.DataFrame` (returns `MangoResult` from Task 8)
  - `space_mango.MangoFilterError` re-exported from `errors` (same class object as `space_mango.client.MangoFilterError`)

- [ ] **Step 1: Write the failing tests** — `tests/test_errors.py`

```python
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

import space_mango as sm
from space_mango.errors import (
    MangoError,
    MangoFilterError,
    TimeParseError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
    error_from_response,
)
from space_mango.timeparse import to_iso


def test_unknown_region_suggests(client):
    with pytest.raises(UnknownRegionError, match="Did you mean 'magnetosheath'"):
        client.get_data("magnetoshealth", limit=1)


def test_unknown_spacecraft_mms1_suggests_mms(client):
    with pytest.raises(UnknownSpacecraftError, match="Did you mean 'MMS'"):
        client.get_data("magnetosheath", spacecraft=["MMS1"], limit=1)


def test_unknown_column(client):
    with pytest.raises(UnknownColumnError):
        client.get_data("magnetosheath", columns=["Nope"], limit=1)


def test_filter_typo_suggests(client):
    with pytest.raises(MangoFilterError, match="Did you mean 'bz_imf_max'"):
        client.get_data("magnetosheath", bzimf_max=-2, limit=1)


def test_all_errors_are_mango_errors():
    for cls in (UnknownRegionError, UnknownSpacecraftError, UnknownColumnError,
                MangoFilterError, TimeParseError):
        assert issubclass(cls, MangoError) and issubclass(cls, ValueError)
    assert sm.MangoFilterError is MangoFilterError


def test_error_from_response_maps_codes():
    body = {"detail": {"error": "unknown_spacecraft", "message": "nope", "valid": ["MMS"]}}
    err = error_from_response(400, body)
    assert isinstance(err, UnknownSpacecraftError) and "nope" in str(err)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2017-01-12T10:00", "2017-01-12T10:00:00"),
        ("2017-01", "2017-01-01T00:00:00"),
        ("2017", "2017-01-01T00:00:00"),
        (date(2017, 1, 12), "2017-01-12T00:00:00"),
        (datetime(2017, 1, 12, 12, tzinfo=timezone(timedelta(hours=2))), "2017-01-12T10:00:00"),
        (datetime(2017, 1, 12, 10, tzinfo=UTC), "2017-01-12T10:00:00"),
        (None, None),
    ],
)
def test_to_iso(value, expected):
    assert to_iso(value, param="start") == expected


def test_to_iso_pandas_timestamp_with_tz():
    pd = pytest.importorskip("pandas")
    assert to_iso(pd.Timestamp("2017-01-12 12:00", tz="Europe/Paris"), param="start") == "2017-01-12T11:00:00"


def test_to_iso_numpy_datetime64():
    np = pytest.importorskip("numpy")
    assert to_iso(np.datetime64("2017-01-12T10:00:00.5"), param="start") == "2017-01-12T10:00:00.500000"


def test_to_iso_rejects_garbage():
    with pytest.raises(TimeParseError, match="start"):
        to_iso("yesterday", param="start")


def test_start_stop_on_client(client):
    df = client.get_data("magnetosphere", start="2017-01", stop=datetime(2018, 1, 1))
    assert df["SC"].to_list() == ["THA"]


def test_time_min_is_deprecated_but_works(client):
    with pytest.warns(FutureWarning, match="start"):
        df = client.get_data("magnetosphere", time_min="2018-01-01")
    assert df["SC"].to_list() == ["C3"]
```

Add to the `dev` dependency group in `pyproject.toml`: `"pandas>=2.0"`, `"xarray>=2024.1"`, `"numpy>=1.26"` (tests only; Task 8 also needs them) and run `uv sync --all-extras`.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Append to `src/space_mango/errors.py`**

```python
class MangoError(Exception):
    """Base class for every error raised by space_mango."""


class UnknownRegionError(MangoError, ValueError):
    pass


class UnknownSpacecraftError(MangoError, ValueError):
    pass


class UnknownColumnError(MangoError, ValueError):
    pass


class MangoFilterError(MangoError, ValueError):
    pass


class TimeParseError(MangoError, ValueError):
    pass


class ServerError(MangoError):
    pass


class CacheMissError(MangoError):
    pass


_CODE_TO_ERROR: dict[str, type[MangoError]] = {
    "unknown_spacecraft": UnknownSpacecraftError,
    "unknown_column": UnknownColumnError,
    "unknown_filter": MangoFilterError,
    "bad_filter_value": MangoFilterError,
    "filter_column_missing": MangoFilterError,
    "flag_unavailable": MangoFilterError,
    "bad_time": TimeParseError,
}


def error_from_response(status: int, body: object) -> MangoError:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        code = str(detail.get("error", ""))
        message = str(detail.get("message", ""))
        return _CODE_TO_ERROR.get(code, ServerError)(message)
    return ServerError(f"MANGO server answered HTTP {status}: {body!r}")
```

- [ ] **Step 4: Create `src/space_mango/timeparse.py`**

```python
"""Turn the time types physicists use into the ISO strings the server expects (naive UTC)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from space_mango.errors import TimeParseError

TimeLike = str | datetime | date | None

_PARTIAL_FORMATS = ("%Y-%m", "%Y")


def _naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo is not None else dt


def to_iso(value: object, *, param: str) -> str | None:
    """Accept str (ISO 8601, or 'YYYY-MM' / 'YYYY'), datetime, date, pandas.Timestamp,
    numpy.datetime64. Timezone-aware values are converted to UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):  # includes pandas.Timestamp
        return _naive_utc(value).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).isoformat()
    if type(value).__name__ == "datetime64":  # numpy, without importing numpy
        return to_iso(str(value.astype("datetime64[us]")), param=param)  # type: ignore[attr-defined]
    if isinstance(value, str):
        try:
            return _naive_utc(datetime.fromisoformat(value)).isoformat()
        except ValueError:
            pass
        for fmt in _PARTIAL_FORMATS:
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue
    raise TimeParseError(
        f"{param}={value!r} is not a time. Use an ISO 8601 string ('2017-01-12T10:00'), "
        "'2017-01', a datetime, a pandas.Timestamp or a numpy.datetime64."
    )
```

- [ ] **Step 5: Rework `src/space_mango/client.py`**

- Delete the local `class MangoFilterError`; `from space_mango.errors import (MangoFilterError, ServerError, UnknownRegionError, did_you_mean, error_from_response)`.
- In `_validate_filters`, the unknown-name message gets a suggestion: compute `valid_params = [f"{n}_{s}" for n in sorted(valid_names) for s in ("min", "max")]` and append `did_you_mean(key, valid_params)` to the message (keep the region hint).
- Add:

```python
    def _get(self, path: str, params: Mapping[str, object] | None = None) -> httpx.Response:
        try:
            r = self._http.get(path, params=params)  # type: ignore[arg-type]
        except httpx.TransportError as e:
            raise ServerError(f"Could not reach the MANGO server at {self._base_url}: {e}") from e
        if r.status_code == 400:
            raise error_from_response(400, r.json())
        if r.status_code >= 400:
            raise ServerError(f"MANGO server answered HTTP {r.status_code} for {path}: {r.text[:300]}")
        return r

    def _check_region(self, region: str) -> str:
        known = self.regions()
        if region not in known:
            raise UnknownRegionError(
                f"'{region}' is not a MANGO region.{did_you_mean(region, known)} "
                f"Regions: {', '.join(known)}."
            )
        return region
```

- Replace every `self._http.get(...); r.raise_for_status()` pair with `self._get(...)`. Cache `regions()` result on the instance (`self._regions: list[str] | None`).
- New `get_data` signature and body:

```python
    def get_data(
        self,
        region: str,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int | None = None,
        time_min: TimeLike = None,
        time_max: TimeLike = None,
        **filters: float,
    ) -> pl.DataFrame:
        """Query one region. Range filters are keyword arguments: bz_imf_max=-2, d_msh_max=0.3.

        start is inclusive, stop is exclusive. time_min/time_max are deprecated aliases
        (both inclusive).
        """
        self._check_region(region)
        if time_min is not None:
            warnings.warn("time_min is deprecated, use start=", FutureWarning, stacklevel=2)
            start = start if start is not None else time_min
        if time_max is not None:
            warnings.warn("time_max is deprecated, use stop= (exclusive)", FutureWarning, stacklevel=2)
        self._ensure_filters_cached(region)
        cleaned = _validate_filters(
            filters, region, self._filter_cache[region], self._other_region_filters(exclude=region)
        )
        params: dict[str, object] = {"format": "arrow", **{k: str(v) for k, v in cleaned.items()}}
        if limit is not None:
            params["limit"] = str(limit)
        if columns:
            params["columns"] = columns
        if spacecraft:
            params["spacecraft"] = spacecraft
        if (s := to_iso(start, param="start")) is not None:
            params["start"] = s
        if (s := to_iso(stop, param="stop")) is not None:
            params["stop"] = s
        if (s := to_iso(time_max, param="time_max")) is not None:
            params["time_max"] = s
        if sw_paired_only:
            params["sw_paired_only"] = "true"
        if normalized_only:
            params["normalized_only"] = "true"
        r = self._get(f"/api/v1/regions/{region}/data", params)
        return pl.read_ipc(r.content)
```

Imports: `import warnings`, `from collections.abc import Mapping`, `from space_mango.timeparse import TimeLike, to_iso`.

- [ ] **Step 6: Update `src/space_mango/__init__.py`**: import `MangoFilterError` (and the other error classes) from `space_mango.errors`; add them to `__all__`; module-level `get_data` gains `start`, `stop` (and passes `time_min`/`time_max` through only when not `None`).

- [ ] **Step 7: Update the old tests to typed errors**

- `tests/test_readme_examples.py`: `test_get_data_spacecraft_no_match` → `pytest.raises(sm.UnknownSpacecraftError)`; `test_get_data_column_nonexistent_is_error` → `pytest.raises(sm.UnknownColumnError)`; `test_get_data_invalid_region` → `pytest.raises(sm.UnknownRegionError)`. README tests calling `time_min`/`time_max` wrap the call in `with pytest.warns(FutureWarning):`.
- `tests/test_client.py`: import `MangoFilterError` from `space_mango.errors`.

- [ ] **Step 8: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: MangoError hierarchy with suggestions; start/stop accept datetime/pandas/numpy"
```

---

### Task 7: Discovery endpoints, version header, app version, catalog self-check

**Files:**
- Modify: `src/space_mango/dataset.py`, `src/space_mango/routes/data.py`, `src/space_mango/routes/schemas.py`, `src/space_mango/app.py`
- Create: `src/space_mango/routes/dataset.py`, `tests/test_endpoints.py`

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces (HTTP, all under `/api/v1`, all read-only):
  - `GET /regions/{r}/describe` → `{"region", "definition", "columns": [{"name","dtype","unit","frame","description","computed","filter"}], "filters": [FilterInfo]}`; columns in served-schema order; undocumented served columns appear with empty strings.
  - `GET /regions/{r}/spacecraft` → `[{"sc","start","stop","n_rows"}]` (ISO strings)
  - `GET /regions/{r}/count` (same params as `/data` minus `limit`/`format`) → `{"n_rows": int, "est_bytes": int}`
  - `GET /timeline?sc=&start=&stop=&columns=&format=` → Arrow/CSV rows from every region, extra `region` column, sorted by Time; span ≤ 31 days
  - `GET /dataset` → `{"version","title","citation","doi","schema_checksum"}`
  - Header `X-Mango-Dataset-Version` on every response; FastAPI `version` = installed package version.
  - Python: `MangoDataset.coverage(region) -> pl.DataFrame` (cached), `count(...) -> tuple[int, int]`, `timeline(sc, start, stop, columns) -> pl.DataFrame`, `schema_checksum() -> str`, `check_catalog() -> list[str]`, `exists() -> bool`; `MAX_TIMELINE_SPAN = timedelta(days=31)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_endpoints.py`

```python
import io
from datetime import datetime

import polars as pl

from space_mango.dataset import MangoDataset


def test_describe_lists_columns_with_units_and_filters(api):
    d = api.get("/api/v1/regions/magnetosheath/describe").json()
    assert d["region"] == "magnetosheath"
    assert "bow shock" in d["definition"]
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["Bz_imf"]["unit"] == "nT" and cols["Bz_imf"]["filter"] == "bz_imf"
    assert cols["R_norm"]["filter"] == "d_msh"
    assert "bow shock" in cols["R_norm"]["description"]
    assert cols["SC"]["dtype"] == "String"
    assert {f["name"] for f in d["filters"]} >= {"bz_imf", "d_msh"}


def test_spacecraft_coverage(api):
    rows = api.get("/api/v1/regions/magnetosphere/spacecraft").json()
    assert [r["sc"] for r in rows] == ["C3", "MMS", "THA"]
    mms = rows[1]
    assert mms["start"].startswith("2015-06-10T12:00") and mms["n_rows"] == 1


def test_count_matches_data(api):
    c = api.get("/api/v1/regions/magnetosheath/count", params={"bz_imf_max": -2}).json()
    assert c["n_rows"] == 2
    assert c["est_bytes"] > 0
    c2 = api.get("/api/v1/regions/magnetosheath/count",
                 params={"bz_imf_max": -2, "columns": ["Np"]}).json()
    assert c2["est_bytes"] == 2 * 8


def test_count_rejects_unknown_filter(api):
    r = api.get("/api/v1/regions/magnetosheath/count", params={"tilt_min": 0})
    assert r.status_code == 400


def test_timeline_spans_regions(make_row, make_dataset, make_api):
    rows = {
        "magnetosphere": [make_row("magnetosphere", "THA", datetime(2017, 1, 12, 10, 0))],
        "magnetosheath": [make_row("magnetosheath", "THA", datetime(2017, 1, 12, 10, 30))],
        "solar_wind": [make_row("solar_wind", "THA", datetime(2017, 1, 12, 11, 0)),
                       make_row("solar_wind", "C1", datetime(2017, 1, 12, 11, 0))],
    }
    api = make_api(make_dataset(rows))
    r = api.get("/api/v1/timeline", params={
        "sc": "THA", "start": "2017-01-12T09:00", "stop": "2017-01-12T12:00"})
    assert r.status_code == 200
    df = pl.read_ipc(io.BytesIO(r.content))
    assert df["region"].to_list() == ["magnetosphere", "magnetosheath", "solar_wind"]
    assert set(df["SC"].to_list()) == {"THA"}
    assert "R_bs" in df.columns  # magnetosheath-only column, null elsewhere


def test_timeline_limits(api):
    r = api.get("/api/v1/timeline", params={"sc": "THA", "start": "2016-01-01", "stop": "2017-01-01"})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "span_too_long"
    r = api.get("/api/v1/timeline", params={"sc": "MMS1", "start": "2016-01-01", "stop": "2016-01-02"})
    assert r.status_code == 400 and "Did you mean 'MMS'" in r.json()["detail"]["message"]


def test_dataset_endpoint_and_version_header(api):
    r = api.get("/api/v1/dataset")
    d = r.json()
    assert d["version"] == "2026.0"
    assert d["citation"].startswith("@") and d["doi"] is None
    assert len(d["schema_checksum"]) == 64
    assert r.headers["X-Mango-Dataset-Version"] == "2026.0"
    assert api.get("/health").headers["X-Mango-Dataset-Version"] == "2026.0"


def test_check_catalog_reports_drift(make_row, make_dataset):
    row = make_row("magnetosheath", "THA", datetime(2016, 1, 1))
    row["Extra"] = 1.0
    del row["R_bs"]
    problems = MangoDataset(make_dataset({"magnetosheath": [row]})).check_catalog()
    assert any("R_bs" in p for p in problems)
    assert any("Extra" in p for p in problems)


def test_check_catalog_clean_on_served_schema(dataset_dir):
    assert MangoDataset(dataset_dir).check_catalog() == []
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Extend `src/space_mango/dataset.py`**

```python
MAX_TIMELINE_SPAN = timedelta(days=31)
_DTYPE_BYTES = {pl.Boolean: 1, pl.String: 4}


def _dtype_bytes(dt: pl.DataType) -> int:
    return _DTYPE_BYTES.get(type(dt), 8)  # floats, ints, datetimes: 8 bytes
```

Add to `MangoDataset` (`__init__` also sets `self._coverage: dict[str, pl.DataFrame] = {}`):

```python
    def exists(self) -> bool:
        return self._dir.is_dir()

    def coverage(self, region: Region | str) -> pl.DataFrame:
        key = Region(region).value
        if key not in self._coverage:
            self._coverage[key] = (
                self._lazy(key)
                .group_by("SC")
                .agg(start=pl.col("Time").min(), stop=pl.col("Time").max(), n_rows=pl.len())
                .sort("SC")
                .collect()
            )
        return self._coverage[key]

    def count(self, region: Region, raw_params: Mapping[str, str | None], **kwargs: Any) -> tuple[int, int]:
        lf = self._plan(region, raw_params, **kwargs)
        n_rows = int(lf.select(pl.len()).collect().item())
        row_bytes = sum(_dtype_bytes(dt) for dt in lf.collect_schema().dtypes())
        return n_rows, n_rows * row_bytes

    def timeline(
        self, sc: str, start: datetime, stop: datetime, columns: list[str] | None
    ) -> pl.DataFrame:
        if stop <= start:
            raise QueryError("bad_time", "stop must be after start.")
        if stop - start > MAX_TIMELINE_SPAN:
            raise QueryError(
                "span_too_long",
                f"A timeline covers at most {MAX_TIMELINE_SPAN.days} days; got {stop - start}. "
                "Use get_data() per region for longer periods.",
            )
        regions = [r for r in Region if sc in self.spacecraft(r)]
        if not regions:
            every = sorted({s for r in Region for s in self.spacecraft(r)})
            raise QueryError(
                "unknown_spacecraft", f"'{sc}' is not a MANGO spacecraft.{did_you_mean(sc, every)}", every
            )
        available = {c for r in regions for c in self.columns(r)}
        for c in columns or []:
            if c not in available:
                raise QueryError("unknown_column", f"'{c}' is not a MANGO column.{did_you_mean(c, available)}", available)
        frames: list[pl.DataFrame] = []
        for r in regions:
            lf = self._lazy(r).filter(
                (pl.col("SC") == sc) & (pl.col("Time") >= start) & (pl.col("Time") < stop)
            )
            if columns:
                have = set(self.columns(r))
                lf = lf.select(["Time", "SC", *[c for c in columns if c in have and c not in ("Time", "SC")]])
            frames.append(lf.with_columns(region=pl.lit(r.value)).collect())
        return pl.concat(frames, how="diagonal_relaxed").sort("Time")

    def schema_checksum(self) -> str:
        parts = []
        for r in Region:
            if (self._dir / r.value).is_dir():
                schema = self._lazy(r.value).collect_schema()
                parts += [f"{r.value}:{name}:{dtype}" for name, dtype in schema.items()]
        return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()

    def check_catalog(self) -> list[str]:
        """Differences between the catalog (models.py) and the served schema."""
        problems: list[str] = []
        for r in Region:
            if not (self._dir / r.value).is_dir():
                continue
            served = set(self.columns(r))
            documented = set(columns_for(r))
            problems += [f"{r.value}: catalog column '{c}' is not served" for c in sorted(documented - served)]
            problems += [f"{r.value}: served column '{c}' is not in the catalog" for c in sorted(served - documented)]
            problems += [
                f"{r.value}: filter '{n}' needs missing column '{f.column}'"
                for n, f in filters_for(r).items() if f.column not in served
            ]
        return problems
```

Imports: `hashlib`, `timedelta`, `columns_for`, `filters_for`.

- [ ] **Step 4: Schemas** — append to `src/space_mango/routes/schemas.py`:

```python
class ColumnDescription(BaseModel):
    name: str
    dtype: str
    unit: str
    frame: str
    description: str
    computed: str
    filter: str | None


class RegionDescription(BaseModel):
    region: str
    definition: str
    columns: list[ColumnDescription]
    filters: list[FilterInfo]


class SpacecraftCoverage(BaseModel):
    sc: str
    start: datetime
    stop: datetime
    n_rows: int


class CountResult(BaseModel):
    n_rows: int
    est_bytes: int


class DatasetDescription(BaseModel):
    version: str
    title: str
    citation: str
    doi: str | None
    schema_checksum: str
```

- [ ] **Step 5: Region routes** — in `src/space_mango/routes/data.py`

Factor the filter list into a helper `_filter_infos(region) -> list[FilterInfo]` (used by `/filters` and `/describe`). Then:

```python
@router.get("/regions/{region}/describe")
def region_describe(region: Region, ds: MangoDataset = Depends(get_dataset)) -> RegionDescription:
    schema = ds[region].collect_schema()
    catalog = columns_for(region)
    cols = []
    for name, dtype in schema.items():
        info = catalog.get(name)
        cols.append(ColumnDescription(
            name=name,
            dtype=str(dtype),
            unit=info.unit if info else "",
            frame=info.frame if info else "",
            description=info.description_for(region) if info else "",
            computed=info.computed if info else "",
            filter=filter_for_column(region, name),
        ))
    return RegionDescription(
        region=region.value, definition=REGIONS[region].definition,
        columns=cols, filters=_filter_infos(region),
    )


@router.get("/regions/{region}/spacecraft")
def region_spacecraft(region: Region, ds: MangoDataset = Depends(get_dataset)) -> list[SpacecraftCoverage]:
    return [SpacecraftCoverage(**row) for row in ds.coverage(region).rename({"SC": "sc"}).to_dicts()]


@router.get("/regions/{region}/count")
def region_count(
    request: Request,
    region: Region,
    columns: list[str] | None = Query(None),
    spacecraft: list[str] | None = Query(None),
    start: str | None = Query(None),
    stop: str | None = Query(None),
    time_min: str | None = Query(None),
    time_max: str | None = Query(None),
    sw_paired_only: bool = Query(False),
    normalized_only: bool = Query(False),
    ds: MangoDataset = Depends(get_dataset),
) -> CountResult:
    """Rows a /data request with the same parameters would return, and an estimated size."""
    start_dt, stop_dt, stop_inclusive = time_window(start, stop, time_min, time_max)
    n_rows, est_bytes = ds.count(
        region, dict(request.query_params),
        columns=columns, spacecraft=spacecraft, start=start_dt, stop=stop_dt,
        stop_inclusive=stop_inclusive, sw_paired_only=sw_paired_only, normalized_only=normalized_only,
    )
    return CountResult(n_rows=n_rows, est_bytes=est_bytes)
```

Factor the Arrow/CSV response code at the end of `region_data` into `def frame_response(df: pl.DataFrame, fmt: Format, name: str) -> StreamingResponse` (in `routes/data.py`), reused by `/timeline`.

- [ ] **Step 6: `src/space_mango/routes/dataset.py`**

```python
"""Cross-region routes: /timeline and /dataset."""

from fastapi import APIRouter, Depends, Query, Request

from space_mango.dataset import MangoDataset, get_dataset
from space_mango.filtering import parse_time
from space_mango.models import DATASET_TITLE, Format, citation_bibtex
from space_mango.routes.data import frame_response
from space_mango.routes.schemas import DatasetDescription

router = APIRouter(tags=["dataset"])


@router.get("/timeline")
def timeline(
    sc: str = Query(..., description="Spacecraft, e.g. THA"),
    start: str = Query(..., description="Start time, inclusive (ISO 8601)"),
    stop: str = Query(..., description="Stop time, exclusive (ISO 8601); at most 31 days after start"),
    columns: list[str] | None = Query(None, description="Columns (default: all); Time, SC, region always included"),
    format: Format = Query(Format.arrow),
    ds: MangoDataset = Depends(get_dataset),
):
    """All rows of one spacecraft over an interval, across regions, with a `region` column."""
    df = ds.timeline(sc, parse_time(start, param="start"), parse_time(stop, param="stop"), columns)
    return frame_response(df, format, f"timeline_{sc}")


@router.get("/dataset")
def dataset_info(request: Request, ds: MangoDataset = Depends(get_dataset)) -> DatasetDescription:
    version: str = request.app.state.dataset_version
    return DatasetDescription(
        version=version, title=DATASET_TITLE, citation=citation_bibtex(version, None),
        doi=None, schema_checksum=ds.schema_checksum(),
    )
```

- [ ] **Step 7: `src/space_mango/app.py`** — version header, package version, lifespan check

```python
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from space_mango.dataset import get_dataset
from space_mango.errors import QueryError
from space_mango.models import DEFAULT_DATASET_VERSION
from space_mango.routes import data, dataset, health

logger = logging.getLogger("space_mango")


def _package_version() -> str:
    try:
        return version("space-mango")
    except PackageNotFoundError:
        return "0+unknown"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ds = get_dataset()
    if ds.exists():
        for problem in ds.check_catalog():
            logger.warning("catalog/schema mismatch: %s", problem)
    yield


async def _query_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, QueryError)
    return JSONResponse(status_code=400, content={"detail": exc.to_dict()})


def create_app() -> FastAPI:
    dataset_version = os.environ.get("MANGO_DATASET_VERSION", DEFAULT_DATASET_VERSION)
    app = FastAPI(
        title="MANGO",
        description="Magnetospheric Atlas of Normalized Geospace Observations — data subsetting API",
        version=_package_version(),
        root_path=os.environ.get("MANGO_ROOT_PATH", ""),
        lifespan=_lifespan,
    )
    app.state.dataset_version = dataset_version

    @app.middleware("http")
    async def _version_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Mango-Dataset-Version"] = dataset_version
        return response

    app.add_exception_handler(QueryError, _query_error_handler)
    app.include_router(health.router)
    app.include_router(data.router, prefix="/api/v1")
    app.include_router(dataset.router, prefix="/api/v1")
    return app
```

(The title text now matches the README: "Magnetospheric Atlas of …".)

- [ ] **Step 8: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: describe/spacecraft/count/timeline/dataset endpoints; version header; catalog self-check"
```

---

### Task 8: `MangoResult`

**Files:**
- Create: `src/space_mango/result.py`, `tests/test_result.py`
- Modify: `src/space_mango/client.py` (`get_data` returns `MangoResult`; add `_describe_raw`, `dataset_info`), `src/space_mango/__init__.py`, `pyproject.toml` (extras)
- Modify: `tests/test_readme_examples.py` (`isinstance(df, pl.DataFrame)` → `isinstance(df, sm.MangoResult)`)

**Interfaces:**
- Consumes: `/describe`, `/dataset` (Task 7).
- Produces:
  - `MangoResult(data: pl.DataFrame, columns_info: dict[str, dict[str, str]], version: str, query: dict[str, object], citation: str, region: str | None)`; methods `to_polars()`, `to_pandas()`, `to_xarray()`, `cite() -> str`, `to_intervals(max_gap: timedelta = timedelta(seconds=30)) -> pl.DataFrame`; properties `metadata`, `columns`; `__len__`, `__getitem__`, `__repr__`.
  - `MangoClient._describe_raw(region) -> dict[str, Any]` (cached), `MangoClient.dataset_info() -> dict[str, Any]` (cached per client), `MangoClient._result(region: str | None, df, query) -> MangoResult`.
  - extras: `pandas = ["pandas>=2.0"]`, `xarray = ["xarray>=2024.1", "pandas>=2.0"]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_result.py`

```python
from datetime import datetime, timedelta

import polars as pl
import pytest

import space_mango as sm
from space_mango.result import MangoResult


def _result(rows, region="magnetosheath"):
    df = pl.DataFrame(rows)
    info = {"Np": {"unit": "cm⁻³", "frame": "", "description": "Local ion density"}}
    return MangoResult(df, info, "2026.0", {"region": region}, "@misc{x}", region)


def test_get_data_returns_result_with_metadata(client):
    r = client.get_data("magnetosheath", columns=["Time", "Np", "Bz_imf"], limit=10)
    assert isinstance(r, sm.MangoResult)
    assert r.version == "2026.0"
    assert r.metadata["Bz_imf"]["unit"] == "nT"
    assert set(r.metadata) == {"Time", "Np", "Bz_imf"}
    assert len(r) == 3 and r.columns == ["Time", "Np", "Bz_imf"]
    assert r.query["region"] == "magnetosheath"
    assert "2026.0" in r.cite()
    assert "MangoResult" in repr(r) and "2026.0" in repr(r)


def test_to_pandas_keeps_metadata():
    pytest.importorskip("pandas")
    pdf = _result({"Time": [datetime(2016, 1, 1)], "Np": [3.0]}).to_pandas()
    assert pdf.attrs["mango"]["version"] == "2026.0"
    assert pdf.attrs["mango"]["columns"]["Np"]["unit"] == "cm⁻³"


def test_to_xarray_keeps_units():
    pytest.importorskip("xarray")
    ds = _result({"Time": [datetime(2016, 1, 1), datetime(2016, 1, 1)],
                  "SC": ["THA", "C1"], "Np": [3.0, 4.0]}).to_xarray()
    assert ds["Np"].attrs["units"] == "cm⁻³"
    assert ds.attrs["mango_version"] == "2026.0"
    assert ds.sizes["index"] == 2  # same Time on two spacecraft: no unique time index


def test_to_intervals_splits_on_region_sc_and_gaps():
    t0 = datetime(2017, 1, 12, 10)
    s = timedelta(seconds=5)
    df = pl.DataFrame({
        "Time": [t0, t0 + s, t0 + 2 * s, t0 + 3 * s, t0 + 3 * s + timedelta(minutes=5), t0],
        "SC": ["THA"] * 5 + ["C1"],
        "region": ["magnetosphere", "magnetosphere", "magnetosheath", "magnetosheath",
                   "magnetosheath", "solar_wind"],
    })
    iv = MangoResult(df, {}, "2026.0", {}, "", None).to_intervals()
    assert iv.columns == ["sc", "region", "start", "stop", "n_points"]
    assert iv.rows() == [
        ("C1", "solar_wind", t0, t0, 1),
        ("THA", "magnetosphere", t0, t0 + s, 2),
        ("THA", "magnetosheath", t0 + 2 * s, t0 + 3 * s, 2),
        ("THA", "magnetosheath", t0 + 3 * s + timedelta(minutes=5), t0 + 3 * s + timedelta(minutes=5), 1),
    ]


def test_to_intervals_on_single_region_result(client):
    iv = client.get_data("magnetosheath", limit=10).to_intervals()
    assert set(iv["region"].to_list()) == {"magnetosheath"}
    assert iv["n_points"].sum() == 3
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Create `src/space_mango/result.py`**

```python
"""MangoResult: data plus the metadata needed to use and cite it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


@dataclass
class MangoResult:
    data: pl.DataFrame
    columns_info: dict[str, dict[str, str]]
    version: str
    query: dict[str, object] = field(default_factory=dict)
    citation: str = ""
    region: str | None = None

    @property
    def metadata(self) -> dict[str, dict[str, str]]:
        """Unit, frame and description of each returned column."""
        return {c: self.columns_info.get(c, {}) for c in self.data.columns}

    @property
    def columns(self) -> list[str]:
        return self.data.columns

    def __len__(self) -> int:
        return self.data.height

    def __getitem__(self, key: str) -> pl.Series:
        return self.data[key]

    def __repr__(self) -> str:
        where = self.region or "timeline"
        return f"MangoResult({where}, dataset {self.version}, {self.data.height} rows)\n{self.data}"

    def cite(self) -> str:
        return self.citation

    def to_polars(self) -> pl.DataFrame:
        return self.data

    def _attrs(self) -> dict[str, Any]:
        return {"version": self.version, "query": self.query, "columns": self.metadata,
                "citation": self.citation}

    def to_pandas(self) -> pd.DataFrame:
        """pandas DataFrame; metadata in df.attrs['mango'] (needs space-mango[pandas])."""
        pdf = self.data.to_pandas()
        pdf.attrs["mango"] = self._attrs()
        return pdf

    def to_xarray(self) -> xr.Dataset:
        """xarray Dataset on an 'index' dimension (Time is not unique across spacecraft);
        per-variable attrs units/frame/description (needs space-mango[xarray])."""
        import xarray as xr

        meta = self.metadata
        ds = xr.Dataset({
            name: ("index", self.data[name].to_numpy(), {
                "units": meta[name].get("unit", ""),
                "frame": meta[name].get("frame", ""),
                "description": meta[name].get("description", ""),
            })
            for name in self.data.columns
        })
        ds.attrs.update({"mango_version": self.version, "mango_citation": self.citation,
                         "mango_region": self.region or "timeline"})
        return ds

    def to_intervals(self, max_gap: timedelta = timedelta(seconds=30)) -> pl.DataFrame:
        """Contiguous stretches per spacecraft and region: sc | region | start | stop | n_points.

        A new interval starts when the region or spacecraft changes, or when consecutive
        samples are more than max_gap apart. stop is the last sample time.
        """
        df = self.data
        if "region" not in df.columns:
            df = df.with_columns(region=pl.lit(self.region))
        df = df.select("SC", "Time", "region").sort("SC", "Time")
        new = (
            (pl.col("region") != pl.col("region").shift())
            | (pl.col("SC") != pl.col("SC").shift())
            | ((pl.col("Time") - pl.col("Time").shift()) > max_gap)
        ).fill_null(True)
        return (
            df.with_columns(interval=new.cum_sum())
            .group_by("interval", maintain_order=True)
            .agg(sc=pl.col("SC").first(), region=pl.col("region").first(),
                 start=pl.col("Time").min(), stop=pl.col("Time").max(), n_points=pl.len())
            .drop("interval")
        )
```

- [ ] **Step 4: Client** — in `src/space_mango/client.py`

```python
    def dataset_info(self) -> dict[str, Any]:
        """Dataset version, title, citation (BibTeX), DOI and schema checksum."""
        if self._dataset_info is None:
            self._dataset_info = self._get("/api/v1/dataset").json()
        return self._dataset_info

    def _describe_raw(self, region: str) -> dict[str, Any]:
        if region not in self._describe_cache:
            self._describe_cache[region] = self._get(f"/api/v1/regions/{region}/describe").json()
        return self._describe_cache[region]

    def _result(self, region: str | None, df: pl.DataFrame, query: dict[str, object]) -> MangoResult:
        regions = [region] if region else self.regions()
        info: dict[str, dict[str, str]] = {}
        for r in regions:
            for c in self._describe_raw(r)["columns"]:
                info.setdefault(c["name"], {k: c[k] for k in ("unit", "frame", "description")})
        ds = self.dataset_info()
        return MangoResult(df, info, ds["version"], query, ds["citation"], region)
```

(`__init__` adds `self._dataset_info: dict[str, Any] | None = None` and `self._describe_cache: dict[str, dict[str, Any]] = {}`.)

`get_data` now ends with `return self._result(region, pl.read_ipc(r.content), query)` where `query = {"region": region, "columns": columns, "spacecraft": spacecraft, "start": <iso>, "stop": <iso>, "sw_paired_only": ..., "normalized_only": ..., "limit": limit, **cleaned}`; return annotation `-> MangoResult`.

- [ ] **Step 5: Export and extras** — `__init__.py` exports `MangoResult`; `pyproject.toml` `[project.optional-dependencies]` adds `pandas = ["pandas>=2.0"]` and `xarray = ["xarray>=2024.1", "pandas>=2.0"]`. In `tests/test_readme_examples.py` replace `isinstance(df, pl.DataFrame)` with `isinstance(df, sm.MangoResult)`.

- [ ] **Step 6: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: MangoResult with pandas/xarray conversion, metadata, citation, intervals"
```

---

### Task 9: Client discovery: `describe`, `spacecraft`, `count`, `search`, `timeline`, `cite`, context manager

**Files:**
- Modify: `src/space_mango/client.py`
- Create: `tests/test_discovery.py`

**Interfaces:**
- Consumes: Tasks 7–8.
- Produces on `MangoClient`:
  - `describe(region) -> pl.DataFrame` columns `column, unit, frame, description, filter, dtype`
  - `region_definition(region) -> str`
  - `spacecraft(region) -> pl.DataFrame` columns `sc, start, stop, n_rows` (Datetime)
  - `count(region, *, columns=None, spacecraft=None, start=None, stop=None, sw_paired_only=False, normalized_only=False, **filters) -> dict[str, float]` keys `n_rows`, `est_mb`
  - `search(text: str) -> pl.DataFrame` columns `region, kind ("column"|"filter"), name, unit, description`
  - `timeline(sc: str, start: TimeLike, stop: TimeLike, columns: list[str] | None = None) -> MangoResult`
  - `cite() -> str`; `close()`; `__enter__`/`__exit__`
  - internal `_request_params(region, *, columns, spacecraft, start, stop, sw_paired_only, normalized_only, time_min, time_max, filters) -> tuple[dict[str, object], dict[str, object]]` (HTTP params, query record) shared by `get_data` and `count`. It validates the region and filters, emits the deprecation warnings, and the query record always has the keys `region, columns, spacecraft, start, stop, time_max, sw_paired_only, normalized_only, limit` (times as ISO strings or `None`) plus the cleaned filters; Task 10 reads `start`/`stop`/`time_max` from it.

- [ ] **Step 1: Write the failing tests** — `tests/test_discovery.py`

```python
from datetime import datetime

import polars as pl
import pytest

from space_mango.errors import UnknownRegionError


def test_describe(client):
    d = client.describe("magnetosheath")
    assert d.columns == ["column", "unit", "frame", "description", "filter", "dtype"]
    row = d.filter(pl.col("column") == "R_norm").row(0, named=True)
    assert row["filter"] == "d_msh" and "bow shock" in row["description"]


def test_region_definition(client):
    assert "magnetopause" in client.region_definition("magnetosphere")


def test_spacecraft(client):
    sc = client.spacecraft("magnetosheath")
    assert sc["sc"].to_list() == ["C1", "MMS", "THA"]
    assert sc.schema["start"] == pl.Datetime("us")


def test_count(client):
    c = client.count("magnetosheath", bz_imf_max=-2)
    assert c["n_rows"] == 2 and c["est_mb"] > 0


def test_count_validates_like_get_data(client):
    with pytest.raises(UnknownRegionError):
        client.count("nowhere")


def test_search(client):
    hits = client.search("density")
    assert {"Np", "Np_sw"} <= set(hits["name"].to_list())
    assert set(hits["kind"].to_list()) <= {"column", "filter"}
    assert client.search("zzzz").height == 0


def test_timeline(client):
    t = client.timeline("THA", "2016-03-15T09:00", "2016-03-15T11:00")
    assert t["region"].to_list() == ["magnetosheath"]
    assert t.region is None and t.version == "2026.0"


def test_cite(client):
    assert client.cite().startswith("@misc")


def test_context_manager(dataset_dir, make_client):
    with make_client(dataset_dir) as c:
        assert "magnetosheath" in c.regions()
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — `src/space_mango/client.py`

Move the parameter-building part of `get_data` (deprecation warnings, filter validation, `params` and `query` dicts) into `_request_params`, used by both. Then:

```python
    def describe(self, region: str) -> pl.DataFrame:
        """Every column of a region: unit, coordinate frame, description, matching filter."""
        self._check_region(region)
        cols = self._describe_raw(region)["columns"]
        return pl.DataFrame(
            [{"column": c["name"], "unit": c["unit"], "frame": c["frame"],
              "description": c["description"], "filter": c["filter"], "dtype": c["dtype"]}
             for c in cols],
            schema={"column": pl.String, "unit": pl.String, "frame": pl.String,
                    "description": pl.String, "filter": pl.String, "dtype": pl.String},
        )

    def region_definition(self, region: str) -> str:
        self._check_region(region)
        return self._describe_raw(region)["definition"]

    def spacecraft(self, region: str) -> pl.DataFrame:
        """Spacecraft present in a region, with first/last sample time and row count."""
        self._check_region(region)
        if region not in self._spacecraft_cache:
            rows = self._get(f"/api/v1/regions/{region}/spacecraft").json()
            self._spacecraft_cache[region] = pl.DataFrame(
                rows, schema={"sc": pl.String, "start": pl.String, "stop": pl.String, "n_rows": pl.Int64}
            ).with_columns(pl.col("start", "stop").str.to_datetime(time_unit="us"))
        return self._spacecraft_cache[region]

    def count(self, region: str, *, columns: list[str] | None = None,
              spacecraft: list[str] | None = None, start: TimeLike = None, stop: TimeLike = None,
              sw_paired_only: bool = False, normalized_only: bool = False,
              **filters: float) -> dict[str, float]:
        """Rows and estimated download size (MB) of the matching get_data call. Downloads nothing."""
        params, _ = self._request_params(
            region, columns=columns, spacecraft=spacecraft, start=start, stop=stop,
            sw_paired_only=sw_paired_only, normalized_only=normalized_only,
            time_min=None, time_max=None, filters=filters,
        )
        params.pop("format", None)
        c = self._get(f"/api/v1/regions/{region}/count", params).json()
        return {"n_rows": c["n_rows"], "est_mb": c["est_bytes"] / 1e6}

    def search(self, text: str) -> pl.DataFrame:
        """Columns and filters whose name, unit or description contains `text` (case-insensitive)."""
        needle = text.lower()
        rows: list[dict[str, str]] = []
        for region in self.regions():
            d = self._describe_raw(region)
            for c in d["columns"]:
                rows.append({"region": region, "kind": "column", "name": c["name"],
                             "unit": c["unit"], "description": c["description"]})
            for f in d["filters"]:
                rows.append({"region": region, "kind": "filter", "name": f["name"],
                             "unit": f["unit"], "description": f["description"]})
        schema = {k: pl.String for k in ("region", "kind", "name", "unit", "description")}
        df = pl.DataFrame(rows, schema=schema)
        hay = pl.concat_str(["name", "unit", "description"], separator=" ").str.to_lowercase()
        return df.filter(hay.str.contains(needle, literal=True))

    def timeline(self, sc: str, start: TimeLike, stop: TimeLike,
                 columns: list[str] | None = None) -> MangoResult:
        """Every sample of one spacecraft over [start, stop) across all regions (≤ 31 days),
        with a 'region' column. Use .to_intervals() for region crossings."""
        params: dict[str, object] = {"sc": sc, "start": to_iso(start, param="start"),
                                     "stop": to_iso(stop, param="stop"), "format": "arrow"}
        if columns:
            params["columns"] = columns
        r = self._get("/api/v1/timeline", params)
        query = {"timeline": sc, "start": params["start"], "stop": params["stop"], "columns": columns}
        return self._result(None, pl.read_ipc(r.content), query)

    def cite(self) -> str:
        """BibTeX for the dataset version served."""
        return self.dataset_info()["citation"]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MangoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

(`__init__` adds `self._spacecraft_cache: dict[str, pl.DataFrame] = {}`.)

- [ ] **Step 4: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: client discovery (describe, spacecraft, count, search, timeline, cite)"
```

---

### Task 10: Fragment cache, used by `get_data` by default

Design from spec §5: one Parquet file per (dataset version, region, spacecraft, calendar month, column) under `<cache_dir>/<version>/<region>/SC=<sc>/<column>/<YYYY-MM>.parquet`. Fetch only missing months, filter locally with `filtering.build_filter_exprs`.

**Files:**
- Create: `src/space_mango/cache.py`, `tests/test_cache.py`
- Modify: `src/space_mango/client.py`, `tests/conftest.py` (clients get a temp cache dir), `pyproject.toml` (`platformdirs>=4.0` in `dependencies`)

**Interfaces:**
- Consumes: `build_filter_exprs`, `filters_for` (Tasks 4–5); `spacecraft()`, `_describe_raw()`, `dataset_info()` (Tasks 8–9); `CacheMissError` (Task 6).
- Produces:
  - `months_between(lo: datetime, hi: datetime) -> list[date]` (first-of-month dates covering [lo, hi], hi inclusive); `next_month(m: date) -> date`; `contiguous_runs(months: list[date]) -> list[list[date]]`
  - `default_cache_dir() -> Path` (`$SPACE_MANGO_CACHE_DIR` or `platformdirs.user_cache_dir("space-mango")`); `default_max_bytes() -> int` (`$SPACE_MANGO_CACHE_SIZE` bytes, default `10 * 1024**3`)
  - `FragmentCache(root: Path, max_bytes: int)` with `path(version, region, sc, column, month) -> Path`, `has(version, region, sc, month, columns) -> bool`, `write_months(version, region, sc, months, df) -> None`, `read_month(version, region, sc, month, columns) -> pl.DataFrame`, `evict() -> None`, `info() -> dict[str, object]`, `clear() -> None`
  - `MangoClient(..., cache_dir: Path | str | None = None, cache: bool = True, offline: bool = False)`; `get_data(..., cache: bool | None = None)`; `cache_info()`, `cache_clear()`

- [ ] **Step 1: Give test clients a temp cache** — `tests/conftest.py`

```python
def _client(data_dir: Path) -> MangoClient:
    tc = _api(data_dir)
    return MangoClient("http://testserver", transport=tc._transport,
                       cache_dir=data_dir.parent / f"{data_dir.name}-cache")
```

- [ ] **Step 2: Write the failing tests** — `tests/test_cache.py`

```python
from datetime import date, datetime

import polars as pl
import pytest

from space_mango.cache import FragmentCache, contiguous_runs, months_between, next_month
from space_mango.client import MangoClient
from space_mango.errors import CacheMissError, UnknownSpacecraftError


def test_month_helpers():
    assert months_between(datetime(2016, 11, 20), datetime(2017, 2, 1)) == [
        date(2016, 11, 1), date(2016, 12, 1), date(2017, 1, 1), date(2017, 2, 1)]
    assert next_month(date(2016, 12, 1)) == date(2017, 1, 1)
    assert contiguous_runs([date(2016, 1, 1), date(2016, 2, 1), date(2016, 5, 1)]) == [
        [date(2016, 1, 1), date(2016, 2, 1)], [date(2016, 5, 1)]]


def test_fragment_roundtrip_and_empty_months(tmp_path):
    fc = FragmentCache(tmp_path, max_bytes=10**9)
    df = pl.DataFrame({"Time": [datetime(2016, 1, 3), datetime(2016, 3, 9)], "Np": [1.0, 2.0]})
    months = [date(2016, 1, 1), date(2016, 2, 1), date(2016, 3, 1)]
    fc.write_months("2026.0", "magnetosheath", "THA", months, df)
    assert fc.has("2026.0", "magnetosheath", "THA", date(2016, 2, 1), ["Time", "Np"])
    assert fc.read_month("2026.0", "magnetosheath", "THA", date(2016, 2, 1), ["Time", "Np"]).height == 0
    assert fc.read_month("2026.0", "magnetosheath", "THA", date(2016, 3, 1), ["Time", "Np"])["Np"].to_list() == [2.0]
    assert not fc.has("2026.1", "magnetosheath", "THA", date(2016, 1, 1), ["Time"])
    assert not list(tmp_path.rglob("*.tmp"))


def test_eviction_removes_oldest(tmp_path):
    fc = FragmentCache(tmp_path, max_bytes=1)
    df = pl.DataFrame({"Time": [datetime(2016, 1, 3)], "Np": [1.0]})
    fc.write_months("v", "magnetosheath", "THA", [date(2016, 1, 1)], df)
    fc.evict()
    assert fc.info()["size_bytes"] <= 1


QUERIES = [
    {},
    {"bz_imf_max": -2},
    {"columns": ["Time", "Np", "R_norm"], "d_msh_max": 0.5},
    {"spacecraft": ["THA", "C1"], "start": "2016-01-01", "stop": "2019-01-05T08:00:00"},
    {"sw_paired_only": True, "normalized_only": True},
]


@pytest.mark.parametrize("q", QUERIES)
def test_cache_path_equals_server_path(client, q):
    cached = client.get_data("magnetosheath", **q).to_polars()
    remote = client.get_data("magnetosheath", cache=False, **q).to_polars()
    key = ["SC", "Time"] if "SC" in remote.columns else ["Time"]
    assert cached.sort(key).equals(remote.select(cached.columns).sort(key))
    assert cached.columns == remote.columns


def test_legacy_time_max_inclusive_on_cache_path(client):
    with pytest.warns(FutureWarning):
        df = client.get_data("magnetosheath", time_max="2018-07-20T14:30:00")
    assert set(df["SC"].to_list()) == {"THA", "MMS"}


def test_second_call_is_served_from_cache(dataset_dir, make_client, monkeypatch):
    c = make_client(dataset_dir)
    c.get_data("magnetosphere", spacecraft=["THA"])
    calls: list[str] = []
    real = c._fetch_months
    monkeypatch.setattr(c, "_fetch_months", lambda *a, **k: calls.append("x") or real(*a, **k))
    c.get_data("magnetosphere", spacecraft=["THA"])
    assert calls == []


def test_offline_raises_on_missing_fragment(dataset_dir, make_client, tmp_path):
    c = make_client(dataset_dir)
    off = MangoClient("http://testserver", transport=c._http._transport,
                      cache_dir=tmp_path / "empty", offline=True)
    with pytest.raises(CacheMissError):
        off.get_data("magnetosphere", spacecraft=["THA"])


def test_new_dataset_version_uses_new_directory(dataset_dir, make_client, monkeypatch):
    c = make_client(dataset_dir)
    c.get_data("solar_wind")
    root = c._cache.root
    assert (root / "2026.0").is_dir()
    c._dataset_info = {**c.dataset_info(), "version": "2027.0"}
    c.get_data("solar_wind")
    assert (root / "2027.0").is_dir()


def test_cache_path_rejects_mms1_with_suggestion(client):
    with pytest.raises(UnknownSpacecraftError, match="Did you mean 'MMS'"):
        client.get_data("magnetosheath", spacecraft=["MMS1"])


def test_cache_info_and_clear(dataset_dir, make_client):
    c = make_client(dataset_dir)
    c.get_data("solar_wind")
    assert c.cache_info()["n_files"] > 0
    c.cache_clear()
    assert c.cache_info()["n_files"] == 0
```

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Create `src/space_mango/cache.py`**

```python
"""On-disk fragment cache, modelled on speasy's (speasy/core/cache/), simplified because a
published MANGO version never changes: the version is part of the path, so nothing expires.

Layout: <root>/<version>/<region>/SC=<sc>/<column>/<YYYY-MM>.parquet
A month with no data is stored as an empty file, so it is not fetched again.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime
from pathlib import Path

import platformdirs
import polars as pl

from space_mango.errors import MangoError


def default_cache_dir() -> Path:
    env = os.environ.get("SPACE_MANGO_CACHE_DIR")
    return Path(env) if env else Path(platformdirs.user_cache_dir("space-mango"))


def default_max_bytes() -> int:
    env = os.environ.get("SPACE_MANGO_CACHE_SIZE")
    return int(env) if env else 10 * 1024**3


def next_month(m: date) -> date:
    return date(m.year + m.month // 12, m.month % 12 + 1, 1)


def months_between(lo: datetime, hi: datetime) -> list[date]:
    m, last = date(lo.year, lo.month, 1), date(hi.year, hi.month, 1)
    out: list[date] = []
    while m <= last:
        out.append(m)
        m = next_month(m)
    return out


def contiguous_runs(months: list[date]) -> list[list[date]]:
    runs: list[list[date]] = []
    for m in sorted(months):
        if runs and next_month(runs[-1][-1]) == m:
            runs[-1].append(m)
        else:
            runs.append([m])
    return runs


class FragmentCache:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes

    def path(self, version: str, region: str, sc: str, column: str, month: date) -> Path:
        return self.root / version / region / f"SC={sc}" / column / f"{month:%Y-%m}.parquet"

    def has(self, version: str, region: str, sc: str, month: date, columns: list[str]) -> bool:
        return all(self.path(version, region, sc, c, month).is_file() for c in columns)

    def write_months(self, version: str, region: str, sc: str, months: list[date], df: pl.DataFrame) -> None:
        df = df.sort("Time", maintain_order=True)
        for m in months:
            part = df.filter((pl.col("Time") >= datetime(m.year, m.month, 1))
                             & (pl.col("Time") < datetime(*next_month(m).timetuple()[:3])))
            for column in df.columns:
                target = self.path(version, region, sc, column, m)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(f".{os.getpid()}.tmp")
                part.select(column).write_parquet(tmp, compression="zstd")
                os.replace(tmp, target)  # atomic: readers never see a half-written file

    def read_month(self, version: str, region: str, sc: str, month: date, columns: list[str]) -> pl.DataFrame:
        parts = []
        for c in columns:
            p = self.path(version, region, sc, c, month)
            os.utime(p)  # mark as recently used for eviction
            parts.append(pl.read_parquet(p))
        if len({p.height for p in parts}) > 1:
            raise MangoError(
                f"Inconsistent cache fragments in {self.path(version, region, sc, '*', month).parent.parent}; "
                "run space_mango.cache.clear() and retry."
            )
        return pl.concat(parts, how="horizontal")

    def _files(self) -> list[Path]:
        return [p for p in self.root.rglob("*.parquet") if p.is_file()] if self.root.exists() else []

    def evict(self) -> None:
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)

    def info(self) -> dict[str, object]:
        files = self._files()
        return {"root": str(self.root), "n_files": len(files),
                "size_bytes": sum(p.stat().st_size for p in files), "max_bytes": self.max_bytes}

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
```

- [ ] **Step 5: Client integration** — `src/space_mango/client.py`

Constructor gains `cache_dir: Path | str | None = None, cache: bool = True, offline: bool = False`:

```python
        self._cache = FragmentCache(Path(cache_dir) if cache_dir else default_cache_dir(), default_max_bytes())
        self._cache_enabled = cache
        self._offline = offline
```

`get_data` gains `cache: bool | None = None`; after `_request_params`:

```python
        use_cache = self._offline or ((self._cache_enabled if cache is None else cache) and limit is None)
        if use_cache:
            df = self._get_data_cached(region, columns, spacecraft, query, cleaned,
                                       sw_paired_only, normalized_only)
            if limit is not None:
                df = df.head(limit)
        else:
            df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        return self._result(region, df, query)
```

and the cached path:

```python
    def _fetch_months(self, version: str, region: str, sc: str, months: list[date], columns: list[str]) -> None:
        params = {"format": "arrow", "spacecraft": [sc], "columns": columns,
                  "start": datetime(months[0].year, months[0].month, 1).isoformat(),
                  "stop": datetime(*next_month(months[-1]).timetuple()[:3]).isoformat()}
        df = pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)
        self._cache.write_months(version, region, sc, months, df)

    def _get_data_cached(
        self, region: str, columns: list[str] | None, spacecraft: list[str] | None,
        query: dict[str, object], ranges: dict[str, float],
        sw_paired_only: bool, normalized_only: bool,
    ) -> pl.DataFrame:
        version = self.dataset_info()["version"]
        served = [c["name"] for c in self._describe_raw(region)["columns"]]
        for c in columns or []:
            if c not in served:
                raise UnknownColumnError(
                    f"'{c}' is not a column of region '{region}'.{did_you_mean(c, served)}")
        coverage = {row["sc"]: (row["start"], row["stop"]) for row in self.spacecraft(region).to_dicts()}
        for sc in spacecraft or []:
            if sc not in coverage:
                raise UnknownSpacecraftError(
                    f"'{sc}' is not a spacecraft in region '{region}'.{did_you_mean(sc, coverage)}")
        start = datetime.fromisoformat(str(query["start"])) if query.get("start") else None
        stop_raw = query.get("stop") or query.get("time_max")
        stop = datetime.fromisoformat(str(stop_raw)) if stop_raw else None
        stop_inclusive = query.get("stop") is None and query.get("time_max") is not None
        needed = {"Time"} | {c for c in (columns or served) if c != "SC"}
        needed |= {filters_for(region)[k.rpartition("_")[0]].column for k in ranges}
        needed |= {"SW_pairing"} if sw_paired_only else set()
        needed |= {"Norma_pos"} if normalized_only else set()
        cols = sorted(needed)

        frames: list[pl.DataFrame] = []
        for sc in spacecraft or sorted(coverage):
            sc_start, sc_stop = coverage[sc]
            lo = max(start, sc_start) if start else sc_start
            hi = min(stop, sc_stop) if stop else sc_stop
            if lo > hi:
                continue
            months = months_between(lo, hi)
            missing = [m for m in months if not self._cache.has(version, region, sc, m, cols)]
            if missing and self._offline:
                raise CacheMissError(
                    f"Offline mode: {len(missing)} month(s) of {sc}/{region} are not cached "
                    f"(first: {missing[0]:%Y-%m}). Run once online, or drop offline=True.")
            for run in _progress(contiguous_runs(missing), f"Downloading {region}/{sc}"):
                self._fetch_months(version, region, sc, run, cols)
            for m in months:
                frames.append(self._cache.read_month(version, region, sc, m, cols)
                              .with_columns(SC=pl.lit(sc)))
        self._cache.evict()
        if not frames:
            return self._empty_frame(region, columns)
        df = pl.concat(frames, how="vertical_relaxed")
        exprs = build_filter_exprs(
            region, set(df.columns), start=start, stop=stop, stop_inclusive=stop_inclusive,
            sw_paired_only=sw_paired_only, normalized_only=normalized_only, ranges=ranges)
        if exprs:
            df = df.filter(pl.all_horizontal(exprs))
        return df.select(columns or served)

    def _empty_frame(self, region: str, columns: list[str] | None) -> pl.DataFrame:
        params: dict[str, object] = {"format": "arrow", "limit": "1", "start": "1900-01-01", "stop": "1900-01-02"}
        if columns:
            params["columns"] = columns
        return pl.read_ipc(self._get(f"/api/v1/regions/{region}/data", params).content)

    def cache_info(self) -> dict[str, object]:
        return self._cache.info()

    def cache_clear(self) -> None:
        self._cache.clear()
```

`served` is in server order (`Time…`, `SC` last), so `df.select(columns or served)` reproduces the server's column order. Module-level helper:

```python
def _progress(items: list[list[date]], desc: str) -> Iterable[list[date]]:
    if len(items) < 2:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, unit="request")
```

`_request_params` must record the ISO `start`, `stop` and `time_max` in `query` (Task 9 already does). Imports: `date`, `datetime`, `Path`, `Iterable`, cache helpers, `build_filter_exprs`, `filters_for`, error classes.

- [ ] **Step 6: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: per-column monthly Parquet fragment cache used by get_data by default"
```

---

### Task 11: Region objects with generated signatures

`help(mango.magnetosheath.get_data)` and tab-completion must list every filter with its unit. Signatures are generated from `RANGE_FILTERS` into a checked-in module; a test fails if it drifts.

**Files:**
- Create: `src/space_mango/regions.py`, `src/space_mango/_codegen.py`, `src/space_mango/_regions_generated.py` (generated), `tests/test_regions.py`
- Modify: `pyproject.toml` (exclude generated file from ruff line-length only if needed; it is formatted to ≤100 chars, so nothing should be needed)

**Interfaces:**
- Consumes: `MangoClient.get_data`, `count`, `describe`, `spacecraft`, `filters`, `region_definition` (Tasks 6–10); `filters_for`, `REGIONS`.
- Produces: `RegionAPI(client_factory: Callable[[], MangoClient])` with `name: ClassVar[str]`, `definition: ClassVar[str]`, `describe()`, `spacecraft()`, `filters` (property → `pl.DataFrame`), `_call(method: str, args: dict[str, object])`; generated classes `MagnetosphereAPI`, `MagnetosheathAPI`, `SolarWindAPI` each with explicit `get_data(...) -> MangoResult` and `count(...) -> dict[str, float]`; `REGION_APIS: dict[str, type[RegionAPI]]`; `render_regions_module() -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_regions.py`

```python
import inspect
from pathlib import Path

import space_mango._regions_generated as generated
from space_mango._codegen import render_regions_module
from space_mango._regions_generated import REGION_APIS
from space_mango.models import filters_for


def test_generated_module_is_up_to_date():
    expected = render_regions_module()
    actual = Path(generated.__file__).read_text()
    assert actual == expected, "run: uv run python -m space_mango._codegen"


def test_signatures_list_every_filter():
    for name, cls in REGION_APIS.items():
        params = inspect.signature(cls.get_data).parameters
        for f in filters_for(name):
            assert f"{f}_min" in params and f"{f}_max" in params
        assert "tilt_min" not in inspect.signature(REGION_APIS["magnetosheath"].get_data).parameters


def test_docstring_shows_units():
    doc = REGION_APIS["magnetosheath"].get_data.__doc__ or ""
    assert "bz_imf_min / bz_imf_max" in doc and "[nT]" in doc


def test_region_object_delegates(client):
    msh = REGION_APIS["magnetosheath"](lambda: client)
    r = msh.get_data(spacecraft=["THA"], bz_imf_max=-2)
    assert r["SC"].to_list() == ["THA"]
    assert msh.count(bz_imf_max=-2)["n_rows"] == 2
    assert "column" in msh.describe().columns
    assert "bow shock" in repr(msh)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: `src/space_mango/regions.py`**

```python
"""Base class of the per-region objects (mango.magnetosheath, ...)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

if TYPE_CHECKING:
    from space_mango.client import MangoClient

_FIXED = ("columns", "spacecraft", "start", "stop", "sw_paired_only", "normalized_only", "limit", "cache")


class RegionAPI:
    name: ClassVar[str]
    definition: ClassVar[str]

    def __init__(self, client_factory: Callable[[], MangoClient]) -> None:
        self._client_factory = client_factory

    def __repr__(self) -> str:
        return f"<MANGO region '{self.name}': {self.definition}>"

    def describe(self) -> pl.DataFrame:
        return self._client_factory().describe(self.name)

    def spacecraft(self) -> pl.DataFrame:
        return self._client_factory().spacecraft(self.name)

    @property
    def filters(self) -> pl.DataFrame:
        return pl.DataFrame(self._client_factory().filters(self.name)).drop("params")

    def _call(self, method: str, args: dict[str, object]) -> Any:
        args = {k: v for k, v in args.items() if k != "self"}
        fixed = {k: args.pop(k) for k in _FIXED if k in args}
        filters = {k: v for k, v in args.items() if v is not None}
        return getattr(self._client_factory(), method)(self.name, **fixed, **filters)
```

- [ ] **Step 4: `src/space_mango/_codegen.py`**

```python
"""Generate _regions_generated.py from the catalog: `uv run python -m space_mango._codegen`."""

from __future__ import annotations

from pathlib import Path

from space_mango.models import REGIONS, Region, filters_for

_HEADER = '''"""GENERATED by space_mango._codegen from models.py. Do not edit by hand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from space_mango.regions import RegionAPI
from space_mango.timeparse import TimeLike

if TYPE_CHECKING:
    from space_mango.result import MangoResult
'''


def _class_name(region: Region) -> str:
    return "".join(part.capitalize() for part in region.value.split("_")) + "API"


def _filter_doc(region: Region) -> list[str]:
    lines = []
    for name, f in filters_for(region).items():
        unit = f" [{f.unit}]" if f.unit else ""
        lines.append(f"        {name}_min / {name}_max : {f.description}{unit}")
    return lines


def _filter_params(region: Region) -> list[str]:
    out = []
    for name in filters_for(region):
        out.append(f"        {name}_min: float | None = None,")
        out.append(f"        {name}_max: float | None = None,")
    return out


def _render_class(region: Region) -> str:
    definition = REGIONS[region].definition.replace('"', "'")
    doc = "\n".join(_filter_doc(region))
    params = "\n".join(_filter_params(region))
    cls = _class_name(region)
    return f'''

class {cls}(RegionAPI):
    """{region.value}: {definition}"""

    name = "{region.value}"
    definition = "{definition}"

    def get_data(
        self,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int | None = None,
        cache: bool | None = None,
{params}
    ) -> MangoResult:
        """Download {region.value} data. start is inclusive, stop exclusive.

        Range filters (inclusive bounds):
{doc}
        """
        return self._call("get_data", locals())

    def count(
        self,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
{params}
    ) -> dict[str, float]:
        """Rows and estimated MB that get_data would return with the same arguments."""
        return self._call("count", locals())
'''


def render_regions_module() -> str:
    body = "".join(_render_class(r) for r in Region)
    mapping = "\n".join(f'    "{r.value}": {_class_name(r)},' for r in Region)
    return f"{_HEADER}{body}\n\nREGION_APIS: dict[str, type[RegionAPI]] = {{\n{mapping}\n}}\n"


if __name__ == "__main__":
    target = Path(__file__).with_name("_regions_generated.py")
    target.write_text(render_regions_module())
    print(f"wrote {target}")
```

- [ ] **Step 5: Generate and check**

Run: `uv run python -m space_mango._codegen && uv run ruff check src/space_mango/_regions_generated.py`
If ruff flags line length on a long definition string, shorten the `RegionInfo` definition in `models.py` (keep each under ~80 chars) and regenerate — do not hand-edit the generated file.

- [ ] **Step 6: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: region objects with generated, documented signatures"
```

---

### Task 12: Module-level API, README, usage docs

**Files:**
- Rewrite: `src/space_mango/__init__.py`
- Modify: `README.md` (Quick Start), `tests/test_readme_examples.py`
- Create: `docs/usage.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `space_mango.{get_data, regions, columns, filters, describe, spacecraft, count, search, timeline, cite, dataset_info, cache, magnetosphere, magnetosheath, solar_wind, MangoClient, MangoResult, MangoError, UnknownRegionError, UnknownSpacecraftError, UnknownColumnError, MangoFilterError, TimeParseError, ServerError, CacheMissError}`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_readme_examples.py`, and replace the old Quick-Start-mirroring tests so they follow the new README (Step 4). Module-level tests point the default client at the test server:

```python
@pytest.fixture
def default_client(client, monkeypatch):
    import space_mango

    monkeypatch.setattr(space_mango, "_default_client", client)
    return client


def test_module_level_discovery(default_client):
    import space_mango as mango

    assert "bow shock" in repr(mango.magnetosheath)
    assert "magnetosheath" in dir(mango)
    assert mango.describe("magnetosheath").height > 0
    assert mango.spacecraft("magnetosheath")["sc"].to_list() == ["C1", "MMS", "THA"]
    assert mango.count("magnetosheath", bz_imf_max=-2)["n_rows"] == 2
    assert mango.search("density").height > 0
    assert mango.cite().startswith("@misc")
    assert mango.cache.info()["n_files"] >= 0


def test_readme_statistical_example(default_client):
    import space_mango as mango

    r = mango.magnetosheath.get_data(spacecraft=["THA", "MMS", "C1"], start="2016-01",
                                     columns=["Time", "Np", "R_norm"], bz_imf_max=-2, d_msh_max=0.3)
    assert r["R_norm"].to_list() == [0.2]
    assert r.metadata["Np"]["unit"] == "cm⁻³"


def test_readme_event_example(default_client):
    import space_mango as mango

    t = mango.timeline("THA", "2016-03-15T09:00", "2016-03-15T11:00")
    assert t.to_intervals()["region"].to_list() == ["magnetosheath"]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Rewrite `src/space_mango/__init__.py`**

```python
"""MANGO: Magnetospheric Atlas of Normalized Geospace Observations.

    import space_mango as mango
    mango.describe("magnetosheath")
    r = mango.magnetosheath.get_data(bz_imf_max=-2, d_msh_max=0.3)
    r.to_pandas()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from space_mango.client import DEFAULT_URL, MangoClient
from space_mango.errors import (
    CacheMissError,
    MangoError,
    MangoFilterError,
    ServerError,
    TimeParseError,
    UnknownColumnError,
    UnknownRegionError,
    UnknownSpacecraftError,
)
from space_mango.result import MangoResult

if TYPE_CHECKING:
    import polars as pl

    from space_mango._regions_generated import MagnetosheathAPI, MagnetosphereAPI, SolarWindAPI
    from space_mango.timeparse import TimeLike

    magnetosphere: MagnetosphereAPI
    magnetosheath: MagnetosheathAPI
    solar_wind: SolarWindAPI

__all__ = [
    "DEFAULT_URL", "MangoClient", "MangoResult",
    "MangoError", "UnknownRegionError", "UnknownSpacecraftError", "UnknownColumnError",
    "MangoFilterError", "TimeParseError", "ServerError", "CacheMissError",
    "get_data", "regions", "columns", "filters", "describe", "spacecraft", "count",
    "search", "timeline", "cite", "dataset_info", "cache",
    "magnetosphere", "magnetosheath", "solar_wind",
]

_default_client: MangoClient | None = None
_REGION_NAMES = ("magnetosphere", "magnetosheath", "solar_wind")


def _get_default_client() -> MangoClient:
    global _default_client
    if _default_client is None:
        _default_client = MangoClient()
    return _default_client


def get_data(region: str, **kwargs: Any) -> MangoResult:
    """Query one region; see MangoClient.get_data, or use mango.<region>.get_data for
    tab-completion of every filter."""
    return _get_default_client().get_data(region, **kwargs)


def regions() -> list[str]:
    """Region names. Their definitions: mango.<region> or describe()."""
    return _get_default_client().regions()


def columns(region: str) -> list[str]:
    return _get_default_client().columns(region)


def filters(region: str) -> list[dict[str, object]]:
    return _get_default_client().filters(region)


def describe(region: str) -> pl.DataFrame:
    """Columns of a region with unit, frame, description and matching filter."""
    return _get_default_client().describe(region)


def spacecraft(region: str) -> pl.DataFrame:
    """Spacecraft in a region with first/last sample and row count."""
    return _get_default_client().spacecraft(region)


def count(region: str, **kwargs: Any) -> dict[str, float]:
    """Rows and estimated MB of a get_data call, without downloading."""
    return _get_default_client().count(region, **kwargs)


def search(text: str) -> pl.DataFrame:
    """Find columns and filters by name, unit or description."""
    return _get_default_client().search(text)


def timeline(sc: str, start: TimeLike, stop: TimeLike, columns: list[str] | None = None) -> MangoResult:
    """All samples of one spacecraft over [start, stop) across regions (≤ 31 days)."""
    return _get_default_client().timeline(sc, start, stop, columns)


def cite() -> str:
    """BibTeX for the dataset version served."""
    return _get_default_client().cite()


def dataset_info() -> dict[str, Any]:
    return _get_default_client().dataset_info()


class _CacheHandle:
    """mango.cache.info() / mango.cache.clear() for the default client's on-disk cache."""

    def info(self) -> dict[str, object]:
        return _get_default_client().cache_info()

    def clear(self) -> None:
        _get_default_client().cache_clear()


cache = _CacheHandle()


def __getattr__(name: str) -> Any:
    if name in _REGION_NAMES:
        from space_mango._regions_generated import REGION_APIS

        return REGION_APIS[name](_get_default_client)
    raise AttributeError(f"module 'space_mango' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REGION_NAMES))
```

- [ ] **Step 4: README Quick Start** — replace the `## Quick Start` section of `README.md` with:

````markdown
## Quick Start

```python
import space_mango as mango

# What is there? (nothing below downloads data)
mango.regions()                     # ['magnetosphere', 'magnetosheath', 'solar_wind']
mango.magnetosheath                 # <MANGO region 'magnetosheath': Between the bow shock and ...>
mango.describe("magnetosheath")     # column | unit | frame | description | filter | dtype
mango.spacecraft("magnetosheath")   # sc | start | stop | n_rows
mango.search("density")             # columns and filters matching a word
mango.count("magnetosheath", bz_imf_max=-2)   # {'n_rows': ..., 'est_mb': ...}

# Statistical study: southward IMF, inner magnetosheath
r = mango.magnetosheath.get_data(           # tab-complete the filters; help() lists their units
    spacecraft=["THA", "MMS"],              # spacecraft names: see mango.spacecraft(...)
    start="2016-01", stop="2021-01",        # start inclusive, stop exclusive
    columns=["Time", "Np", "Bx_swi", "R_norm"],
    bz_imf_max=-2, d_msh_max=0.3,
)
df = r.to_pandas()                          # or r.to_polars(), r.to_xarray()
r.metadata["Np"]                            # {'unit': 'cm⁻³', 'frame': '', 'description': ...}
print(r.cite())                             # BibTeX, with the dataset version

# Event context: where was THA, and when did it cross a boundary?
t = mango.timeline("THA", "2017-01-12T10:00", "2017-01-12T12:00")
t.to_intervals()                            # sc | region | start | stop | n_points
```

Results are cached on disk (`~/.cache/space-mango`, size cap `SPACE_MANGO_CACHE_SIZE`
bytes, default 10 GB); re-running a notebook does not download again.
`mango.cache.info()` / `mango.cache.clear()` manage it.

**Changes in 0.2:** `get_data` returns a `MangoResult` (use `.to_polars()` for the previous
polars DataFrame); `time_min`/`time_max` are deprecated in favour of `start`/`stop`;
unknown spacecraft, columns or filters now raise an error instead of returning empty or
unfiltered data. The spacecraft name for MMS is `MMS` (not `MMS1`).
````

Update the remaining README tests that mirror the old snippets (`spacecraft=["MMS1", ...]` becomes `["MMS", ...]`).

- [ ] **Step 5: `docs/usage.md`** — a reference page with sections *Regions*, *Columns and frames* (a table generated from `COLUMNS`: name, regions, unit, frame, description — write it out by hand from `models.py`, one row per column), *Filters* (from `RANGE_FILTERS`), *Time conventions* (UTC, 5 s grid, start inclusive/stop exclusive), *Cache*, *Errors* (one line per error class), *Citing* (`mango.cite()`), *Known caveats* (column names frozen for the data paper; SWI-frame values under investigation for some spacecraft and epochs — **ask Nicolas before publishing this sentence**). Link it from the README.

- [ ] **Step 6: Run everything, expect green; commit**

```bash
uv run pytest -q && uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell
git add -A && git commit -m "feat: module-level discovery API; README and usage docs for 0.2"
```

---

## After the last task

- Whole-branch review (superpowers:requesting-code-review), then superpowers:finishing-a-development-branch.
- Not in this plan (spec §8, later phases): Zenodo DOI and finished column metadata (Phase 2), versioned layout (Phase 3), regeneration pipeline (Phase 4), speasy adapter (Phase 5).
- Deployment of the new server endpoints needs Alexis Jeandet (spec §9 q4); the client degrades loudly (`ServerError` on 404) against an old server, so release the server before publishing 0.2 to PyPI.
