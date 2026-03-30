# Client Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python client to `mango` so researchers can query the MANGO dataset with `mango.get_data("magnetosheath", bz_imf_max=-2)`.

**Architecture:** `MangoClient` in `client.py` talks to the server over HTTP (httpx), returns polars DataFrames via Arrow IPC. Filter validation happens client-side using the cached filter catalog. Module-level convenience functions in `__init__.py` delegate to a lazy global instance. Server deps move to `mango[server]` optional extra.

**Tech Stack:** httpx, polars, pyarrow

**File structure:**
```
src/mango/
├── __init__.py    — re-exports MangoClient + module-level get_data, regions, columns, filters
├── client.py      — MangoClient class, MangoFilterError, filter validation
├── models.py      — unchanged (already has FilterInfo)
├── (server files unchanged)
```

---

### Task 1: Restructure dependencies (client-first)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move server deps to optional extra**

In `pyproject.toml`, change `dependencies` and add `[project.optional-dependencies]`:

```toml
dependencies = [
    "polars>=1.0",
    "pyarrow>=18.0",
    "httpx>=0.28",
]

[project.optional-dependencies]
server = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "cyclopts>=4.10.1",
]
```

Also update `description` to:

```toml
description = "Python client and server for the MANGO magnetospheric dataset"
```

Move `httpx` from `[dependency-groups] dev` to core `dependencies` (it's now a client dep, not dev-only).

- [ ] **Step 2: Run uv sync and verify**

```bash
uv sync --all-extras
uv run pytest tests/ -v
```

Expected: all 10 tests pass (server extras are installed in dev).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "refactor: make server deps optional, client deps are core"
```

---

### Task 2: MangoClient with filter validation

**Files:**
- Create: `src/mango/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write failing tests for filter validation**

Create `tests/test_client.py`:

```python
import pytest

from mango.client import MangoFilterError, _validate_filters


MAGNETOSHEATH_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
    "x_gsm", "y_gsm", "z_gsm", "d_msh", "np", "tp", "bz",
}

MAGNETOSPHERE_FILTERS = {
    "bz_imf", "by_imf", "bx_imf", "pd_sw", "np_sw", "tp_sw",
    "vx_sw", "beta_sw", "ma_sw", "tilt",
    "x_gsm", "y_gsm", "z_gsm", "d_msp", "np", "tp", "bz",
}


def test_validate_filters_valid():
    _validate_filters(
        {"bz_imf_max": -2.0, "pd_sw_min": 3.0},
        "magnetosheath",
        MAGNETOSHEATH_FILTERS,
        {"magnetosphere": MAGNETOSPHERE_FILTERS},
    )


def test_validate_filters_bad_suffix():
    with pytest.raises(MangoFilterError, match="must end with '_min' or '_max'"):
        _validate_filters(
            {"bz_imf": -2.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_unknown_filter():
    with pytest.raises(MangoFilterError, match="not a valid filter"):
        _validate_filters(
            {"fake_min": 1.0},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )


def test_validate_filters_wrong_region_with_hint():
    with pytest.raises(MangoFilterError, match="available for region 'magnetosphere'"):
        _validate_filters(
            {"d_msp_min": 0.5},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {"magnetosphere": MAGNETOSPHERE_FILTERS},
        )


def test_validate_filters_non_numeric():
    with pytest.raises(MangoFilterError, match="must be numeric"):
        _validate_filters(
            {"bz_imf_max": "not_a_number"},
            "magnetosheath",
            MAGNETOSHEATH_FILTERS,
            {},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_client.py -v
```

Expected: FAIL — `mango.client` does not exist yet.

- [ ] **Step 3: Write `client.py` with filter validation and MangoClient**

Create `src/mango/client.py`:

```python
from __future__ import annotations

import polars as pl
import httpx

DEFAULT_URL = "http://sciqlop.lpp.polytechnique.fr/mango/"


class MangoFilterError(Exception):
    pass


def _validate_filters(
    filters: dict[str, object],
    region: str,
    valid_names: set[str],
    other_regions: dict[str, set[str]],
) -> dict[str, float]:
    """Validate filter kwargs and return cleaned {param: float_value} dict."""
    cleaned: dict[str, float] = {}
    for key, value in filters.items():
        if not (key.endswith("_min") or key.endswith("_max")):
            raise MangoFilterError(
                f"Filter parameter '{key}' must end with '_min' or '_max' "
                f"(e.g. '{key}_min' or '{key}_max')."
            )
        name = key.rsplit("_", 1)[0]
        if name not in valid_names:
            hint = ""
            for other_region, other_names in other_regions.items():
                if name in other_names:
                    hint = f"\nHint: '{name}' is available for region '{other_region}'."
                    break
            available = ", ".join(sorted(valid_names))
            raise MangoFilterError(
                f"'{key}' is not a valid filter for region '{region}'.\n"
                f"Available filters: {available}{hint}"
            )
        try:
            cleaned[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise MangoFilterError(
                f"Filter '{key}' value must be numeric, got {value!r}."
            ) from None
    return cleaned


class MangoClient:
    """Client for the MANGO dataset API."""

    def __init__(self, base_url: str = DEFAULT_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=120.0)
        self._filter_cache: dict[str, set[str]] = {}

    def _ensure_filters_cached(self, region: str) -> None:
        if region not in self._filter_cache:
            r = self._http.get(f"/api/v1/regions/{region}/filters")
            r.raise_for_status()
            self._filter_cache[region] = {f["name"] for f in r.json()}

    def _other_region_filters(self, exclude: str) -> dict[str, set[str]]:
        for region in self.regions():
            self._ensure_filters_cached(region)
        return {r: names for r, names in self._filter_cache.items() if r != exclude}

    def regions(self) -> list[str]:
        """List available regions."""
        r = self._http.get("/api/v1/regions")
        r.raise_for_status()
        return r.json()

    def columns(self, region: str) -> list[str]:
        """List columns available in a region."""
        r = self._http.get(f"/api/v1/regions/{region}/columns")
        r.raise_for_status()
        return r.json()

    def filters(self, region: str) -> list[dict]:
        """List available filters for a region (name, column, unit, description)."""
        r = self._http.get(f"/api/v1/regions/{region}/filters")
        r.raise_for_status()
        return r.json()

    def get_data(
        self,
        region: str,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int = 100_000,
        **filters: float,
    ) -> pl.DataFrame:
        """Query the MANGO dataset and return a polars DataFrame.

        Range filters are passed as keyword arguments:
            get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
        """
        self._ensure_filters_cached(region)
        valid_names = self._filter_cache[region]
        other = self._other_region_filters(exclude=region)
        cleaned = _validate_filters(filters, region, valid_names, other)

        params: dict[str, str | list[str]] = {
            "format": "arrow",
            "limit": str(limit),
        }
        if columns:
            params["columns"] = columns
        if spacecraft:
            params["spacecraft"] = spacecraft
        if time_min:
            params["time_min"] = time_min
        if time_max:
            params["time_max"] = time_max
        if sw_paired_only:
            params["sw_paired_only"] = "true"
        if normalized_only:
            params["normalized_only"] = "true"
        for key, value in cleaned.items():
            params[key] = str(value)

        r = self._http.get(f"/api/v1/regions/{region}/data", params=params)
        r.raise_for_status()
        return pl.read_ipc(r.content)
```

- [ ] **Step 4: Run filter validation tests**

```bash
uv run pytest tests/test_client.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mango/client.py tests/test_client.py
git commit -m "feat: add MangoClient with filter validation"
```

---

### Task 3: Module-level convenience API

**Files:**
- Modify: `src/mango/__init__.py`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write failing test for module-level get_data**

Add to `tests/test_client.py`:

```python
import tempfile
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

import mango
from mango.app import create_app
from mango.dataset import MangoDataset, get_dataset


def _write_test_region(base: Path, region: str, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    for sc in df["SC"].unique().to_list():
        sc_dir = base / region / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        part = df.filter(pl.col("SC") == sc).drop("SC")
        part.write_parquet(sc_dir / "part-0.parquet")


MAGNETOSHEATH_ROWS = [
    {"Time": "2010-01-01T00:00:00", "SC": "THA", "Bx": 1.0, "By": 2.0, "Bz": 3.0,
     "Np": 10.0, "Vx": -200.0, "Vy": 0.0, "Vz": 0.0, "Tp": 1e6,
     "X_gsm": 8.0, "Y_gsm": 3.0, "Z_gsm": 0.0,
     "SW_pairing": True, "Bz_imf": -5.0, "Pd_sw": 3.0, "Norma_pos": True},
    {"Time": "2010-01-01T00:00:05", "SC": "MMS", "Bx": 2.0, "By": 3.0, "Bz": 4.0,
     "Np": 20.0, "Vx": -300.0, "Vy": 1.0, "Vz": 1.0, "Tp": 2e6,
     "X_gsm": 9.0, "Y_gsm": 4.0, "Z_gsm": 1.0,
     "SW_pairing": False, "Bz_imf": 2.0, "Pd_sw": 1.0, "Norma_pos": False},
]


def _start_test_server(data_dir: Path) -> str:
    """Start a test server and return its base URL.

    Uses FastAPI TestClient with transport-level integration
    so no real network is needed.
    """
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    tc = TestClient(app)
    return tc


def test_client_get_data_via_test_server():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        tc = _start_test_server(base)

        client = mango.MangoClient.__new__(mango.MangoClient)
        client._base_url = "http://testserver"
        client._http = httpx.Client(transport=tc.transport, base_url="http://testserver")
        client._filter_cache = {}

        df = client.get_data("magnetosheath", limit=10)
        assert len(df) == 2
        assert "SC" in df.columns
        assert "Bz_imf" in df.columns


def test_client_get_data_with_spacecraft_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        tc = _start_test_server(base)

        client = mango.MangoClient.__new__(mango.MangoClient)
        client._base_url = "http://testserver"
        client._http = httpx.Client(transport=tc.transport, base_url="http://testserver")
        client._filter_cache = {}

        df = client.get_data("magnetosheath", spacecraft=["THA"], limit=10)
        assert len(df) == 1
        assert df["SC"][0] == "THA"


def test_client_get_data_with_range_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        tc = _start_test_server(base)

        client = mango.MangoClient.__new__(mango.MangoClient)
        client._base_url = "http://testserver"
        client._http = httpx.Client(transport=tc.transport, base_url="http://testserver")
        client._filter_cache = {}

        df = client.get_data("magnetosheath", bz_imf_max=-1.0, limit=10)
        assert len(df) == 1
        assert df["Bz_imf"][0] == -5.0


def test_client_regions():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        tc = _start_test_server(base)

        client = mango.MangoClient.__new__(mango.MangoClient)
        client._base_url = "http://testserver"
        client._http = httpx.Client(transport=tc.transport, base_url="http://testserver")
        client._filter_cache = {}

        assert "magnetosheath" in client.regions()
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_client.py::test_client_get_data_via_test_server -v
```

Expected: FAIL — `mango.MangoClient` not importable from `__init__.py`.

- [ ] **Step 3: Write `__init__.py` with lazy global client and re-exports**

Replace `src/mango/__init__.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from mango.client import MangoClient, MangoFilterError

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "MangoClient",
    "MangoFilterError",
    "get_data",
    "regions",
    "columns",
    "filters",
]

_default_client: MangoClient | None = None


def _get_default_client() -> MangoClient:
    global _default_client
    if _default_client is None:
        _default_client = MangoClient()
    return _default_client


def get_data(
    region: str,
    *,
    columns: list[str] | None = None,
    spacecraft: list[str] | None = None,
    time_min: str | None = None,
    time_max: str | None = None,
    sw_paired_only: bool = False,
    normalized_only: bool = False,
    limit: int = 100_000,
    **filters: float,
) -> pl.DataFrame:
    """Query the MANGO dataset and return a polars DataFrame.

    Range filters are passed as keyword arguments:
        mango.get_data("magnetosheath", bz_imf_max=-2, pd_sw_min=3)
    """
    return _get_default_client().get_data(
        region,
        columns=columns,
        spacecraft=spacecraft,
        time_min=time_min,
        time_max=time_max,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        limit=limit,
        **filters,
    )


def regions() -> list[str]:
    """List available regions."""
    return _get_default_client().regions()


def columns(region: str) -> list[str]:
    """List columns available in a region."""
    return _get_default_client().columns(region)


def filters(region: str) -> list[dict]:
    """List available filters for a region."""
    return _get_default_client().filters(region)
```

- [ ] **Step 4: Add `import httpx` to the test file header**

The test file needs `import httpx` for the `httpx.Client(transport=...)` calls.

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (existing server tests + new client tests).

- [ ] **Step 6: Commit**

```bash
git add src/mango/__init__.py tests/test_client.py
git commit -m "feat: add module-level get_data, regions, columns, filters"
```

---

### Task 4: Guard server imports

**Files:**
- Modify: `src/mango/cli.py`
- Modify: `src/mango/app.py`

The CLI and app modules import fastapi/uvicorn/cyclopts which are now optional. They only run when the user has `mango[server]` installed, but the imports will fail at import time if someone accidentally imports them.

- [ ] **Step 1: Add a server-deps guard to cli.py**

Replace `src/mango/cli.py`:

```python
from pathlib import Path

try:
    import cyclopts
except ImportError:
    raise ImportError(
        "The MANGO server requires extra dependencies.\n"
        "Install them with: pip install mango[server]"
    ) from None

app = cyclopts.App(name="mango", help="MANGO dataset service")


@app.command
def serve(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    data_dir: Path | None = None,
) -> None:
    """Start the MANGO API server."""
    import os

    import uvicorn

    if data_dir:
        os.environ["MANGO_DATA_DIR"] = str(data_dir)

    uvicorn.run("mango.main:app", host=host, port=port, reload=reload)


def main() -> None:
    app()
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass (server extras are installed in dev environment).

- [ ] **Step 3: Commit**

```bash
git add src/mango/cli.py
git commit -m "feat: guard server imports with helpful error for client-only install"
```
