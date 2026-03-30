# Hive-Partitioned Parquet Conversion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert MANGO pickle files to Hive-partitioned Parquet (by spacecraft), update the service to use them, and reduce per-request RAM.

**Architecture:** One-time conversion script reads each pickle, renames `time`→`Time`, sorts by time within each SC group, and writes Hive-partitioned Parquet (`SC=XXX/data.parquet`). The service switches from `scan_parquet(file)` to `scan_parquet(dir, hive_partitioning=True)`. The magnetosphere file (~25 GB in memory) is converted SC-by-SC to stay within RAM budget.

**Tech Stack:** Polars, pandas (conversion only), pyarrow, Parquet

**Data layout:**
```
<data_dir>/
├── magnetosheath/
│   ├── SC=C1/part-0.parquet
│   ├── SC=C3/part-0.parquet
│   └── ...
├── magnetosphere/
│   └── SC=.../part-0.parquet
└── solar_wind/
    └── SC=.../part-0.parquet
```

**Column rename during conversion:** `time` → `Time` (pickles use lowercase, service code uses uppercase).

**Known pre-existing issue (out of scope):** `RANGE_FILTERS` references `D_msh`/`D_msp` columns but magnetosheath pickle has `R_norm` instead. The filter silently no-ops. Not fixing here.

---

### Task 1: Conversion script

**Files:**
- Create: `scripts/convert_pickles.py`

- [ ] **Step 1: Write the conversion script**

```python
"""Convert MANGO pickle files to Hive-partitioned Parquet (by spacecraft).

Usage:
    python scripts/convert_pickles.py /path/to/pickle/dir /path/to/output/dir

Converts each region one at a time to limit peak RAM. The magnetosphere
file (~25 GB) is loaded once and written per-SC partition, then freed.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = {
    "magnetosheath": "Magnetosheath_dataset.pkl",
    "magnetosphere": "Magnetosphere_dataset.pkl",
    "solar_wind": "SolarWind_dataset.pkl",
}

ROW_GROUP_SIZE = 500_000


def convert_region(pickle_path: Path, out_dir: Path, region: str) -> None:
    print(f"Loading {pickle_path} ...")
    import pickle

    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    print(f"  shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Rename time → Time to match service code
    if "time" in df.columns:
        df.rename(columns={"time": "Time"}, inplace=True)

    region_dir = out_dir / region

    for sc, group in df.groupby("SC", sort=False):
        sc_dir = region_dir / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        part = group.drop(columns=["SC"]).sort_values("Time").reset_index(drop=True)
        table = pa.Table.from_pandas(part, preserve_index=False)
        pq.write_table(table, sc_dir / "part-0.parquet", row_group_size=ROW_GROUP_SIZE)
        print(f"  SC={sc}: {len(part)} rows → {sc_dir / 'part-0.parquet'}")
        del part, table

    del df
    gc.collect()
    print(f"  done: {region}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MANGO pickles to Hive Parquet")
    parser.add_argument("pickle_dir", type=Path, help="Directory containing .pkl files")
    parser.add_argument("output_dir", type=Path, help="Output directory for Parquet files")
    args = parser.parse_args()

    for region, filename in REGIONS.items():
        pkl = args.pickle_dir / filename
        if not pkl.exists():
            print(f"SKIP {pkl} (not found)")
            continue
        convert_region(pkl, args.output_dir, region)

    print("All done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the conversion on the solar wind file (smallest) to validate**

```bash
mkdir -p /tmp/mango_parquet
uv run --with pandas scripts/convert_pickles.py /home/jeandet/Documents/DATA/Mango /tmp/mango_parquet
```

Expected: three region directories, each with `SC=XXX/part-0.parquet` subdirectories. Solar wind should complete in ~30s, magnetosheath in ~5 min, magnetosphere in ~10 min.

- [ ] **Step 3: Verify the output with Polars**

```bash
uv run python -c "
import polars as pl
for region in ['solar_wind', 'magnetosheath', 'magnetosphere']:
    lf = pl.scan_parquet(f'/tmp/mango_parquet/{region}/', hive_partitioning=True)
    schema = lf.collect_schema()
    print(f'{region}: {schema.names()}')
    n = lf.select(pl.len()).collect().item()
    print(f'  rows: {n}')
    print(f'  SC values: {lf.select(\"SC\").unique().collect()[\"SC\"].to_list()}')
    # Verify Time column is present and datetime
    print(f'  Time dtype: {schema[\"Time\"]}')
"
```

Expected: each region has `Time` (Datetime), `SC` (String from hive), and all original columns.

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_pickles.py
git commit -m "feat: add pickle-to-hive-parquet conversion script"
```

---

### Task 2: Update `MangoDataset` for Hive partitioning

**Files:**
- Modify: `src/mango/dataset.py`
- Test: `tests/test_placeholder.py`

- [ ] **Step 1: Write failing test — dataset loads hive-partitioned directory**

Add to `tests/test_placeholder.py`:

```python
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from mango.dataset import MangoDataset
from mango.models import Region


def _write_test_region(base: Path, region: str, rows: list[dict]) -> None:
    """Write a small Hive-partitioned Parquet dataset for testing."""
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


def test_hive_dataset_query_all():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(Region.magnetosheath, {}, limit=100)
        assert len(df) == 2
        assert "SC" in df.columns
        assert "Time" in df.columns


def test_hive_dataset_query_spacecraft_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(Region.magnetosheath, {}, spacecraft=["THA"], limit=100)
        assert len(df) == 1
        assert df["SC"][0] == "THA"


def test_hive_dataset_query_time_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)

        ds = MangoDataset(base)
        df = ds.query(
            Region.magnetosheath, {},
            time_min="2010-01-01T00:00:03",
            limit=100,
        )
        assert len(df) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_placeholder.py::test_hive_dataset_query_all -v
```

Expected: FAIL — `MangoDataset` tries to open `magnetosheath.parquet` (a file), not `magnetosheath/` (a directory).

- [ ] **Step 3: Update `MangoDataset._lazy` to use Hive partitioning**

In `src/mango/dataset.py`, change `_lazy`:

```python
def _lazy(self, region: str) -> pl.LazyFrame:
    if region not in self._frames:
        path = self._dir / region
        self._frames[region] = pl.scan_parquet(
            path, hive_partitioning=True
        )
    return self._frames[region]
```

The only change: `f"{region}.parquet"` → `region` (directory), plus `hive_partitioning=True`.

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass including the 3 new ones and the existing metadata tests.

- [ ] **Step 5: Commit**

```bash
git add src/mango/dataset.py tests/test_placeholder.py
git commit -m "feat: switch to Hive-partitioned Parquet (by spacecraft)"
```

---

### Task 3: Update data dir default and CLI

**Files:**
- Modify: `src/mango/dataset.py`
- Modify: `src/mango/cli.py`

- [ ] **Step 1: Update the default data dir constant**

In `src/mango/dataset.py`, the default is already `/data/mango` which works — the conversion script just writes the new layout under it. No change needed here.

- [ ] **Step 2: Add a note to the CLI help for `--data-dir`**

In `src/mango/cli.py`, update the help text:

```python
serve_parser.add_argument(
    "--data-dir", default=None,
    help="Path to Hive-partitioned Parquet directory (overrides MANGO_DATA_DIR)"
)
```

- [ ] **Step 3: Commit**

```bash
git add src/mango/cli.py
git commit -m "docs: update --data-dir help text for Hive layout"
```

---

### Task 4: Add data endpoint test with fixture data

**Files:**
- Modify: `tests/test_placeholder.py`

- [ ] **Step 1: Write test for the `/data` endpoint with range filters**

Add to `tests/test_placeholder.py`:

```python
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from mango.app import create_app
from mango.dataset import MangoDataset


def _make_test_client(data_dir: Path) -> TestClient:
    app = create_app()
    ds = MangoDataset(data_dir)
    app.dependency_overrides[get_dataset] = lambda: ds
    return TestClient(app)


def test_data_endpoint_returns_csv():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_test_client(base)

        r = client.get("/api/v1/regions/magnetosheath/data?format=csv&limit=10")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        lines = r.text.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows


def test_data_endpoint_spacecraft_filter():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_test_region(base, "magnetosheath", MAGNETOSHEATH_ROWS)
        client = _make_test_client(base)

        r = client.get("/api/v1/regions/magnetosheath/data?format=csv&spacecraft=THA&limit=10")
        assert r.status_code == 200
        lines = r.text.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "THA" in lines[1]
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_placeholder.py
git commit -m "test: add data endpoint tests with fixture Parquet data"
```
