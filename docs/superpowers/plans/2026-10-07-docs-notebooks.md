# MANGO Docs Site with Executed Notebooks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Sphinx/Read the Docs site for `space-mango` whose five example notebooks run on every build against a ≤10 MB real-data sample, failing CI on any error.

**Architecture:** A read-only extraction script pulls a small real sample from today's public server into `docs/data/` (production Hive layout). `docs/conf.py` starts a local uvicorn server on that sample and points the client at it through a new `SPACE_MANGO_URL` environment variable, so notebook cells are plain user code. myst-nb executes the notebooks during `sphinx-build -W`; a CI job and Read the Docs run the same build.

**Tech Stack:** Sphinx, myst-nb, furo, matplotlib, ipykernel, nbformat; existing space-mango (polars, httpx, FastAPI/uvicorn); uv.

**Spec:** `docs/superpowers/specs/2026-10-07-docs-notebooks-design.md` (builds on `2026-10-07-discoverable-api-design.md`).

## Global Constraints

- Branch `docs/notebooks` (already created off `feat/discoverable-api`). Never commit on `main` or `feat/discoverable-api`. Never push.
- Sample: `docs/data/<region>/SC=<sc>/part-0.parquet`, zstd, **total ≤ 10 MB**, extracted only with read-only `GET /api/v1/regions/{r}/data` calls to `http://sciqlop.lpp.polytechnique.fr/mango/` (0.1 API: params `spacecraft`, `time_min`, `time_max`, `columns`, `format=arrow`, `limit`). No other host, no writes outside `docs/data/`.
- The sample is served as dataset version `2026.0-docs-sample`.
- Notebooks are stored **without outputs** and contain only user-facing code (no hidden setup cells).
- Served column names are frozen; the sample keeps every served column of each region.
- Each task ends with exit code 0 from: `uv run pytest -q`, `uv run ruff check .`, `uv run basedpyright -p pyproject.toml`, `uv run codespell`; from Task 3 on also `uv run sphinx-build -W --keep-going -b html docs docs/_build/html`. **Check exit codes (`; echo $?`), not summary lines.**
- uv is not on PATH on pharemer: use `/tmp/claude-3255/-home-aunai-Documents-code-mango/54bc4848-b79b-4c98-ae31-b3dfd5768acd/scratchpad/uvenv/bin/uv`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01T6RBwQQKRfYYqokQoJQWPP
  ```

## Review Focus

1. The docs server fails to start (port taken, missing extra) → the build must fail with a clear message, not hang. Test: Task 3 (`wait_for_health` timeout test).
2. A notebook cell raises → `sphinx-build` exits non-zero (no silently rendered traceback). Test: Task 3 (a throwaway failing notebook built in a temp dir).
3. Someone commits a notebook with outputs → CI fails. Test: Task 4 (`tests/test_notebooks_stripped.py`).
4. The sample silently exceeds 10 MB or loses a region / the THA event window when regenerated → test fails. Test: Task 2 (`tests/test_docs_sample.py`).
5. `SPACE_MANGO_URL` set with a trailing slash or unset → client still builds correct URLs / falls back to the public URL. Test: Task 1.

---

## File structure

```
src/space_mango/client.py          MangoClient base_url defaults to $SPACE_MANGO_URL, else DEFAULT_URL
scripts/make_docs_sample.py        read-only extraction of docs/data (+ --find-event helper)
docs/data/<region>/SC=<sc>/part-0.parquet
docs/_docs_server.py               start/stop a local uvicorn server on docs/data (used by conf.py)
docs/conf.py, docs/index.md, docs/installation.md, docs/quickstart.md, docs/user_guide.md (moved from usage.md),
docs/api.md, docs/citing.md, docs/changelog.md, docs/examples/index.md
docs/examples/01_getting_started.ipynb … 05_cache_and_offline.ipynb
.readthedocs.yaml
.github/workflows/ci.yml           new `docs` job
tests/test_default_url.py, tests/test_docs_sample.py, tests/test_docs_server.py, tests/test_notebooks_stripped.py
```

---

### Task 1: `SPACE_MANGO_URL` selects the default server

**Files:**
- Modify: `src/space_mango/client.py` (`MangoClient.__init__`)
- Create: `tests/test_default_url.py`

**Interfaces:**
- Produces: `MangoClient(base_url: str | None = None, ...)`; when `base_url is None` it uses `os.environ.get("SPACE_MANGO_URL") or DEFAULT_URL`. Everything else unchanged (`space_mango._get_default_client()` calls `MangoClient()` and therefore honours the variable).

- [ ] **Step 1: Write the failing tests** — `tests/test_default_url.py`

```python
from space_mango.client import DEFAULT_URL, MangoClient


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "http://127.0.0.1:8765/")
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == "http://127.0.0.1:8765"


def test_unset_env_falls_back_to_public_server(monkeypatch, tmp_path):
    monkeypatch.delenv("SPACE_MANGO_URL", raising=False)
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == DEFAULT_URL.rstrip("/")


def test_empty_env_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "")
    c = MangoClient(cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == DEFAULT_URL.rstrip("/")


def test_explicit_base_url_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACE_MANGO_URL", "http://ignored:1")
    c = MangoClient("http://explicit:2", cache_dir=tmp_path)
    assert str(c._http.base_url).rstrip("/") == "http://explicit:2"
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_default_url.py -v` (first test fails: base URL is the public server).

- [ ] **Step 3: Implement** — in `MangoClient.__init__` change the signature to `base_url: str | None = None` and, first thing in the body:

```python
        if base_url is None:
            base_url = os.environ.get("SPACE_MANGO_URL") or DEFAULT_URL
```

(add `import os` if absent). Update the constructor docstring: "base_url defaults to $SPACE_MANGO_URL, else the public MANGO server."

- [ ] **Step 4: Run all checks, expect exit 0; commit**

```bash
uv run pytest -q; echo $?
uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell; echo $?
git add src/space_mango/client.py tests/test_default_url.py
git commit -m "feat: SPACE_MANGO_URL selects the default server"
```

---

### Task 2: Real-data docs sample

**Files:**
- Create: `scripts/make_docs_sample.py`, `docs/data/**` (generated), `tests/test_docs_sample.py`
- Modify: `pyproject.toml` (`[tool.codespell] skip`/`[tool.ruff] extend-exclude` only if the parquet files trip them — they should not)

**Interfaces:**
- Produces: `docs/data/{magnetosphere,magnetosheath,solar_wind}/SC=<sc>/part-0.parquet`; module constants in the script `EVENT_SC = "THA"`, `EVENT_START`, `EVENT_STOP` (ISO strings, the chosen event window), documented in its docstring; `docs/data/README.md` stating origin, date of extraction, command, and that it is a sample.

- [ ] **Step 1: Write the failing test** — `tests/test_docs_sample.py`

```python
"""The docs sample must stay small, complete and in the production layout."""

from pathlib import Path

import polars as pl
import pytest

from conftest import SERVED_COLUMNS

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
MAX_BYTES = 10 * 1024 * 1024
EVENT_SC = "THA"


def _region(region: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / region, hive_partitioning=True)


def test_sample_size_cap():
    total = sum(p.stat().st_size for p in DATA.rglob("*.parquet"))
    assert 0 < total <= MAX_BYTES, total


@pytest.mark.parametrize("region", ["magnetosphere", "magnetosheath", "solar_wind"])
def test_region_schema_and_spacecraft(region):
    df = _region(region)
    assert set(df.columns) == set(SERVED_COLUMNS[region]) | {"SC"}
    assert df["SC"].n_unique() >= 2
    assert df.height > 1000


def test_event_window_crosses_all_regions():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_docs_sample", DATA.parent.parent / "scripts" / "make_docs_sample.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    start = pl.lit(mod.EVENT_START).str.to_datetime()
    stop = pl.lit(mod.EVENT_STOP).str.to_datetime()
    for region in ("magnetosphere", "magnetosheath", "solar_wind"):
        rows = _region(region).filter(
            (pl.col("SC") == EVENT_SC) & (pl.col("Time") >= start) & (pl.col("Time") < stop))
        assert rows.height > 100, region
```

Run: `uv run pytest tests/test_docs_sample.py -v` → FAIL (no `docs/data`).

- [ ] **Step 2: Write `scripts/make_docs_sample.py`**

```python
"""Extract the small real-data sample used by the docs notebooks (docs/data/).

Read-only: only GET /api/v1/regions/{region}/data on the public MANGO server (0.1 API).
Writes only under docs/data/. Refuses to write more than 10 MB.

Usage:
    uv run python scripts/make_docs_sample.py --find-event 2008-06-01 2008-10-01
    uv run python scripts/make_docs_sample.py            # writes docs/data/

Contents:
    - statistical part: N_WINDOWS short windows per region at random times (SEED) in
      2007-2021, all served columns;
    - event part: THA between EVENT_START and EVENT_STOP in every region.
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

SERVER = "http://sciqlop.lpp.polytechnique.fr/mango/api/v1"
REGIONS = ("magnetosphere", "magnetosheath", "solar_wind")
OUT = Path(__file__).resolve().parent.parent / "docs" / "data"
MAX_BYTES = 10 * 1024 * 1024
SEED = 20261007
N_WINDOWS = 150                       # tuned so the total stays under MAX_BYTES
WINDOW = timedelta(minutes=10)
PERIOD = (datetime(2007, 1, 1), datetime(2021, 6, 1))
EVENT_SC = "THA"
EVENT_START = "2008-07-01T00:00:00"   # replaced by the window found with --find-event
EVENT_STOP = "2008-07-03T00:00:00"


def fetch(client: httpx.Client, region: str, **params: object) -> pl.DataFrame:
    r = client.get(f"{SERVER}/regions/{region}/data", params={"format": "arrow", **params})
    r.raise_for_status()
    return pl.read_ipc(io.BytesIO(r.content))


def find_event(client: httpx.Client, first: str, last: str) -> None:
    """Print days where EVENT_SC has rows in all three regions (columns=Time only)."""
    day = datetime.fromisoformat(first)
    end = datetime.fromisoformat(last)
    while day < end:
        nxt = day + timedelta(days=1)
        counts = {
            reg: fetch(client, reg, spacecraft=EVENT_SC, columns="Time",
                       time_min=day.isoformat(), time_max=nxt.isoformat()).height
            for reg in REGIONS
        }
        if all(counts.values()):
            print(day.date(), counts)
        day = nxt


def build(client: httpx.Client) -> None:
    rng = random.Random(SEED)
    span = (PERIOD[1] - PERIOD[0]).total_seconds()
    parts: dict[str, list[pl.DataFrame]] = {reg: [] for reg in REGIONS}
    for reg in REGIONS:
        got = 0
        attempts = 0
        while got < N_WINDOWS and attempts < 20 * N_WINDOWS:
            attempts += 1
            t0 = PERIOD[0] + timedelta(seconds=rng.uniform(0, span))
            df = fetch(client, reg, time_min=t0.isoformat(), time_max=(t0 + WINDOW).isoformat())
            if df.height:
                parts[reg].append(df)
                got += 1
        parts[reg].append(fetch(client, reg, spacecraft=EVENT_SC,
                                time_min=EVENT_START, time_max=EVENT_STOP))
    tmp = OUT.with_name("data.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for reg, frames in parts.items():
        df = pl.concat(frames, how="vertical_relaxed").unique(maintain_order=True).sort("SC", "Time")
        for (sc,), part in df.group_by("SC", maintain_order=True):
            d = tmp / reg / f"SC={sc}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop("SC").write_parquet(d / "part-0.parquet", compression="zstd")
    total = sum(p.stat().st_size for p in tmp.rglob("*.parquet"))
    print(f"sample size: {total / 1e6:.2f} MB")
    if total > MAX_BYTES:
        shutil.rmtree(tmp)
        sys.exit(f"sample is {total} bytes > {MAX_BYTES}; lower N_WINDOWS or WINDOW")
    readme = OUT / "README.md"
    keep = readme.read_text() if readme.exists() else None
    shutil.rmtree(OUT, ignore_errors=True)
    tmp.rename(OUT)
    if keep is not None:
        readme.write_text(keep)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--find-event", nargs=2, metavar=("FIRST_DAY", "LAST_DAY"))
    args = p.parse_args()
    with httpx.Client(timeout=300) as client:
        if args.find_event:
            find_event(client, *args.find_event)
        else:
            build(client)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Choose the event window.** Run `--find-event` over a THEMIS dayside season (try 2008-06-01 → 2008-10-01; if nothing, 2009-06-01 → 2009-10-01). Pick 2–3 consecutive days where THA has rows in all three regions; set `EVENT_START`/`EVENT_STOP` to that window and record the choice (and the `--find-event` output line) in the script docstring.

- [ ] **Step 4: Build the sample.** Run `uv run python scripts/make_docs_sample.py`. If it exits for size, lower `N_WINDOWS` (keep `WINDOW` ≥ 5 min) and rerun; record the final values. Write `docs/data/README.md`:

```markdown
# Docs sample (not the full dataset)

Small extract of the MANGO dataset used to execute the documentation notebooks.
Produced on <YYYY-MM-DD> with `uv run python scripts/make_docs_sample.py`
from http://sciqlop.lpp.polytechnique.fr/mango/ (read-only). Served by the docs build as
dataset version `2026.0-docs-sample`. Do not use for science.
```

- [ ] **Step 5: Run checks (exit codes 0), commit**

```bash
uv run pytest -q; echo $?
uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell; echo $?
git add scripts/make_docs_sample.py docs/data tests/test_docs_sample.py
git commit -m "docs: real-data sample for executed notebooks (<size> MB, read-only extraction)"
```

---

### Task 3: Sphinx site skeleton, local docs server, Read the Docs config

**Files:**
- Create: `docs/_docs_server.py`, `docs/conf.py`, `docs/index.md`, `docs/installation.md`, `docs/quickstart.md`, `docs/api.md`, `docs/citing.md`, `docs/changelog.md`, `docs/examples/index.md`, `.readthedocs.yaml`, `tests/test_docs_server.py`
- Move: `docs/usage.md` → `docs/user_guide.md` (`git mv`); fix its title and links; update the README link (`docs/usage.md` → `docs/user_guide.md`).
- Modify: `pyproject.toml` (extra `docs`), `.gitignore` (`docs/_build/`)

**Interfaces:**
- Consumes: `SPACE_MANGO_URL` (Task 1); `docs/data` (Task 2).
- Produces: `docs/_docs_server.py` with `start(data_dir: Path, version: str = "2026.0-docs-sample", timeout: float = 30.0) -> tuple[subprocess.Popen[bytes], str]` (returns process and base URL) and `stop(proc) -> None`; raises `RuntimeError("docs server did not become healthy …")` on timeout. `conf.py` sets `os.environ["SPACE_MANGO_URL"]` and `os.environ["SPACE_MANGO_CACHE_DIR"]` before notebooks execute.

- [ ] **Step 1: Failing tests** — `tests/test_docs_server.py`

```python
import importlib.util
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _server_module():
    spec = importlib.util.spec_from_file_location("_docs_server", DOCS / "_docs_server.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_server_serves_the_sample_with_sample_version():
    srv = _server_module()
    proc, url = srv.start(DOCS / "data")
    try:
        r = httpx.get(f"{url}/api/v1/dataset", timeout=30)
        assert r.json()["version"] == "2026.0-docs-sample"
        assert r.headers["X-Mango-Dataset-Version"] == "2026.0-docs-sample"
    finally:
        srv.stop(proc)
    assert proc.poll() is not None


def test_start_times_out_with_clear_error(tmp_path, monkeypatch):
    srv = _server_module()
    monkeypatch.setattr(srv, "_command", lambda port: [sys.executable, "-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="did not become healthy"):
        srv.start(tmp_path, timeout=2)


@pytest.mark.slow
def test_failing_notebook_fails_the_build(tmp_path):
    """A cell error must make sphinx-build exit non-zero."""
    import nbformat

    src = tmp_path / "src"
    src.mkdir()
    (src / "conf.py").write_text(
        "extensions = ['myst_nb']\nnb_execution_mode = 'force'\nnb_execution_raise_on_error = True\n")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise ValueError('boom')")])
    nbformat.write(nb, src / "index.ipynb")
    res = subprocess.run([sys.executable, "-m", "sphinx", "-W", "-b", "html", str(src), str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert res.returncode != 0
```

Register the marker in `pyproject.toml` `[tool.pytest.ini_options]`: `markers = ["slow: builds a throwaway Sphinx site"]`.

Run → FAIL (no `_docs_server.py`, no sphinx installed yet).

- [ ] **Step 2: `pyproject.toml` docs extra** — add to `[project.optional-dependencies]`:

```toml
docs = [
    "space-mango[server,pandas,xarray]",
    "sphinx>=8.0",
    "myst-nb>=1.1",
    "furo>=2024.8.6",
    "matplotlib>=3.8",
    "ipykernel>=6.29",
    "nbformat>=5.10",
]
```

and `"space-mango[docs]"` to the `dev` group. Run `uv sync --all-extras`; commit `uv.lock` with this task.

- [ ] **Step 3: `docs/_docs_server.py`**

```python
"""Start a local MANGO server on the docs sample for notebook execution (docs build only)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _command(port: int) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "space_mango.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]


def start(data_dir: Path, version: str = "2026.0-docs-sample",
          timeout: float = 30.0) -> tuple[subprocess.Popen[bytes], str]:
    port = _free_port()
    env = {**os.environ, "MANGO_DATA_DIR": str(data_dir), "MANGO_DATASET_VERSION": version}
    proc = subprocess.Popen(_command(port), env=env)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return proc, url
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    stop(proc)
    raise RuntimeError(f"docs server did not become healthy within {timeout} s (data_dir={data_dir})")


def stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
```

- [ ] **Step 4: `docs/conf.py`**

```python
"""Sphinx configuration for space-mango. Notebooks run against a local server on docs/data."""

import atexit
import os
import sys
import tempfile
from importlib.metadata import version as _version
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS))
import _docs_server  # noqa: E402

project = "MANGO"
author = "Laboratoire de Physique des Plasmas"
release = _version("space-mango")
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx"]
html_theme = "furo"
html_title = "MANGO"
exclude_patterns = ["_build", "superpowers", "data", "**/.ipynb_checkpoints"]
myst_enable_extensions = ["colon_fence"]
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_execution_show_tb = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

_proc, _url = _docs_server.start(DOCS / "data")
atexit.register(_docs_server.stop, _proc)
os.environ["SPACE_MANGO_URL"] = _url
os.environ["SPACE_MANGO_CACHE_DIR"] = tempfile.mkdtemp(prefix="mango-docs-cache-")
```

If intersphinx inventories cannot be fetched offline and that produces a warning that fails `-W`, drop `intersphinx` (record it in the report) — the build must not need the network.

- [ ] **Step 5: Pages.** Write each page in MyST Markdown:
  - `index.md` — one-paragraph description of MANGO (from README "What is MANGO" text), then a `toctree` (maxdepth 2): installation, quickstart, examples/index, user_guide, api, citing, changelog.
  - `installation.md` — `pip install space-mango`, extras `[pandas]`, `[xarray]`, `[server]`, `[docs]`; Python ≥ 3.11.
  - `quickstart.md` — the README Quick Start code block and its paragraphs, verbatim.
  - `user_guide.md` — the moved `usage.md`; keep content; first line `# User guide`.
  - `api.md` — `# API reference`, then `{eval-rst}` blocks with `.. autofunction::` for `space_mango.get_data, regions, columns, filters, describe, spacecraft, count, search, timeline, cite, dataset_info`; `.. autoclass:: space_mango.MangoClient :members:`; `.. autoclass:: space_mango.MangoResult :members:`; `.. autoclass:: space_mango._regions_generated.MagnetosheathAPI :members: get_data, count, describe, spacecraft` (and the two other region classes); `.. automodule:: space_mango.errors :members: MangoError, UnknownRegionError, UnknownSpacecraftError, UnknownColumnError, MangoFilterError, TimeParseError, ServerError, CacheMissError`.
  - `citing.md` — "Use `mango.cite()`; it returns BibTeX for the dataset version you used", plus the README References list moved here by link (keep README unchanged except the usage link).
  - `changelog.md` — `## 0.2` with the README "Changes in 0.2" bullets verbatim.
  - `examples/index.md` — `# Examples`, one sentence saying the notebooks run on a small real-data sample, and a `toctree` (glob) `*`.

- [ ] **Step 6: `.readthedocs.yaml`**

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"
python:
  install:
    - method: pip
      path: .
      extra_requirements: [docs]
sphinx:
  configuration: docs/conf.py
  fail_on_warning: true
```

Add `docs/_build/` to `.gitignore`.

- [ ] **Step 7: Build and test (exit codes 0), commit**

```bash
uv run pytest -q; echo $?
uv run sphinx-build -W --keep-going -b html docs docs/_build/html; echo $?
uv run ruff check . && uv run basedpyright -p pyproject.toml && uv run codespell; echo $?
git add -A docs .readthedocs.yaml pyproject.toml uv.lock .gitignore README.md tests/test_docs_server.py
git commit -m "docs: Sphinx site with local sample server and Read the Docs config"
```

(`git add -A docs` must not pick up `docs/_build` — check `git status` first.)

---

### Task 4: Example notebooks 1, 4, 5 (discovery, results, cache) + stripped-output check

**Files:**
- Create: `docs/examples/01_getting_started.ipynb`, `docs/examples/04_working_with_results.ipynb`, `docs/examples/05_cache_and_offline.ipynb`, `tests/test_notebooks_stripped.py`

**Interfaces:**
- Consumes: the docs build from Task 3 (kernel env has `SPACE_MANGO_URL`, `SPACE_MANGO_CACHE_DIR`).
- Produces: notebooks with no outputs and `execution_count: null`; kernelspec `python3`.

Notebooks are created with a throwaway `nbformat` script (not committed): one markdown or code cell per bullet below, in order. Keep markdown short and factual; no physics claims beyond `docs/user_guide.md`.

- [ ] **Step 1: Failing test** — `tests/test_notebooks_stripped.py`

```python
"""Notebooks are stored without outputs; the docs build executes them."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "docs" / "examples").glob("*.ipynb"))


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"{path.name}: strip outputs (cell {cell.get('id')})"
            assert cell.get("execution_count") is None
```

- [ ] **Step 2: `01_getting_started.ipynb`**
  1. md: `# Getting started` — what MANGO is (one paragraph); note: "These pages run on a small real-data sample served locally (dataset version `2026.0-docs-sample`); with the public server, the same code returns the full dataset."
  2. code: `import space_mango as mango` / `mango.regions()`
  3. code: `mango.magnetosheath`
  4. code: `mango.describe("magnetosheath")`
  5. md: what `R_norm` and the `*_swi` columns are, pointing to the user guide.
  6. code: `mango.spacecraft("magnetosheath")`
  7. code: `mango.search("density")`
  8. code: `mango.count("magnetosheath", bz_imf_max=-2)`
  9. code: `r = mango.magnetosheath.get_data(columns=["Time", "Np", "Bz_imf", "R_norm"], bz_imf_max=-2)` / `r`
  10. code: `help(mango.magnetosheath.get_data)`

- [ ] **Step 3: `04_working_with_results.ipynb`**
  1. md: `# Working with results`
  2. code: `import space_mango as mango` / `r = mango.magnetosphere.get_data(columns=["Time", "Bx", "By", "Bz", "Np", "R_norm"], d_msp_min=0.8)`
  3. code: `r.version, len(r), r.columns`
  4. code: `r.metadata["Bz"]`
  5. code: `df = r.to_pandas()` / `df.head()` / `df.attrs["mango"]["columns"]["Np"]`
  6. code: `ds = r.to_xarray()` / `ds`
  7. code: `r.query`
  8. code: `print(r.cite())`

- [ ] **Step 4: `05_cache_and_offline.ipynb`**
  1. md: `# Cache and offline use` — how the cache works (two sentences, from user guide).
  2. code: `import space_mango as mango` / `mango.count("magnetosheath", bz_imf_max=-2, columns=["Time", "Np"])`
  3. md: `est_mb` vs `download_mb_estimate`.
  4. code: `r = mango.magnetosheath.get_data(bz_imf_max=-2, columns=["Time", "Np"])` / `mango.cache.info()`
  5. code: `mango.count("magnetosheath", bz_imf_max=-2, columns=["Time", "Np"])["download_mb_estimate"]  # 0: already cached`
  6. code: `offline = mango.MangoClient(offline=True)` / `offline.get_data("magnetosheath", bz_imf_max=-2, columns=["Time", "Np"]).to_polars().height`
  7. md: errors explain themselves.
  8. code: `try:` / `    mango.magnetosheath.get_data(spacecraft="MMS1")` / `except mango.UnknownSpacecraftError as e:` / `    print(e)`

- [ ] **Step 5: Build (exit 0), run tests, commit**

```bash
uv run pytest -q tests/test_notebooks_stripped.py; echo $?
uv run sphinx-build -W --keep-going -b html docs docs/_build/html; echo $?
git add docs/examples/01_getting_started.ipynb docs/examples/04_working_with_results.ipynb docs/examples/05_cache_and_offline.ipynb tests/test_notebooks_stripped.py
git commit -m "docs: getting-started, results and cache notebooks"
```

---

### Task 5: Science notebooks 2 and 3 (statistical study, event context)

**Files:**
- Create: `docs/examples/02_statistical_study.ipynb`, `docs/examples/03_event_context.ipynb`

**Interfaces:**
- Consumes: `scripts/make_docs_sample.py` constants `EVENT_SC`, `EVENT_START`, `EVENT_STOP` (copy the literal values into notebook 3 — notebooks must not import repo scripts).

- [ ] **Step 1: `02_statistical_study.ipynb`**
  1. md: `# A statistical study: magnetosheath density vs IMF Bz` — states the question; notes the sample is small, so the figures illustrate the method, not a result.
  2. code: `import matplotlib.pyplot as plt` / `import numpy as np` / `import polars as pl` / `import space_mango as mango`
  3. code: `mango.count("magnetosheath", normalized_only=True, columns=["Np", "Bz_imf", "R_norm", "X_gsm_norm", "Y_gsm_norm"])`
  4. code: `r = mango.magnetosheath.get_data(normalized_only=True, columns=["Np", "Bx", "By", "Bz", "Np_sw", "Bz_imf", "R_norm", "X_gsm_norm", "Y_gsm_norm"])` / `df = r.to_polars().with_columns(B=(pl.col("Bx")**2 + pl.col("By")**2 + pl.col("Bz")**2).sqrt(), n_ratio=pl.col("Np") / pl.col("Np_sw"))`
  5. code (figure): two panels — median `n_ratio` and median `B` in 10 bins of `R_norm` (0–1), for `Bz_imf < 0` vs `Bz_imf > 0`; axis labels with units from `r.metadata`; legend.
  6. code (figure): scatter or 2-D histogram of `X_gsm_norm` vs `Y_gsm_norm` coloured by `n_ratio`, `aspect="equal"`, labelled `R_E`.
  7. md: how to run the same notebook on the full dataset (unset `SPACE_MANGO_URL` / use default server) and that `count()` first is advisable.

- [ ] **Step 2: `03_event_context.ipynb`**
  1. md: `# Event context: where was THA, and when did it cross a boundary?`
  2. code: `import matplotlib.pyplot as plt` / `import polars as pl` / `import space_mango as mango` / `SC, START, STOP = "THA", "<EVENT_START>", "<EVENT_STOP>"`
  3. code: `t = mango.timeline(SC, START, STOP, columns=["Bx", "By", "Bz", "Np"])` / `t`
  4. code: `iv = t.to_intervals()` / `iv`
  5. code (figure): two stacked panels sharing x — |B| and Np vs Time; `axvspan` per interval coloured by region (fixed colour per region, legend); labels with units from `t.metadata`.
  6. md: what to do next — e.g. pass the interval list to speasy (`spz.get_data`) for higher-resolution data (text only, no speasy import).

- [ ] **Step 3: Build (exit 0), run tests, commit**

```bash
uv run pytest -q; echo $?
uv run sphinx-build -W --keep-going -b html docs docs/_build/html; echo $?
git add docs/examples/02_statistical_study.ipynb docs/examples/03_event_context.ipynb
git commit -m "docs: statistical-study and event-context notebooks"
```

Open `docs/_build/html/examples/02_statistical_study.html` and `03_event_context.html` and check the figures render with labelled axes; describe them in the report.

---

### Task 6: CI `docs` job

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add the job** (same checkout/uv setup steps as `build`, one Python):

```yaml
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v7
        with:
          version: "0.10.2"
          enable-cache: true
          python-version: "3.13"
      - run: uv python install
      - run: uv sync --all-extras
      - name: Notebooks are stored without outputs
        run: uv run pytest -q tests/test_notebooks_stripped.py
      - name: Build docs (executes notebooks, warnings are errors)
        run: uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

In the `build` job's test step, skip the slow Sphinx test: `uv run pytest -m "not slow"` (the docs job covers the build; the slow test runs locally).

- [ ] **Step 2: Validate the YAML locally** — `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` (add `pyyaml` only if missing; it is usually present via other deps); exit 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build docs and execute notebooks"
```

---

## After the last task

Whole-branch review; then push `docs/notebooks` (over SSH: `git push git@github.com:LaboratoryOfPlasmaPhysics/mango.git docs/notebooks:docs/notebooks`) and open a PR **targeting `feat/discoverable-api`** — only after the user approves pushing. Read the Docs project activation is for a repository admin.
