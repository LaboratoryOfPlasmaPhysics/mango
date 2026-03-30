"""Convert MANGO pickle files to Hive-partitioned Parquet (by spacecraft).

Usage:
    python scripts/convert_pickles.py /path/to/pickle/dir /path/to/output/dir

Each region is converted in a forked subprocess so that memory is fully
released to the OS between regions (Python/glibc don't return arena memory
after large allocations).
"""

import argparse
import gc
import os
import pickle
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

    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    print(f"  shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

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


def _convert_in_subprocess(pickle_path: Path, out_dir: Path, region: str) -> None:
    """Fork a child process to convert one region, so memory is fully reclaimed."""
    pid = os.fork()
    if pid == 0:
        # Child: convert and exit
        try:
            convert_region(pickle_path, out_dir, region)
        except Exception as e:
            print(f"ERROR converting {region}: {e}", file=sys.stderr)
            os._exit(1)
        os._exit(0)
    else:
        # Parent: wait for child
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
            print(f"FAILED: {region} (exit code {os.WEXITSTATUS(status)})", file=sys.stderr)
            sys.exit(1)
        if os.WIFSIGNALED(status):
            print(f"KILLED: {region} (signal {os.WTERMSIG(status)})", file=sys.stderr)
            sys.exit(1)


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
        _convert_in_subprocess(pkl, args.output_dir, region)

    print("All done.")


if __name__ == "__main__":
    main()
