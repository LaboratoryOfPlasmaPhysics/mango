"""Convert MANGO pickle files to Hive-partitioned Parquet (by spacecraft).

Two-pass strategy for large files:
  1. Load pickle → write single temporary Parquet (peak RAM = pickle size)
  2. Free all memory, then use Polars lazy scan to repartition by SC

Each region runs in a forked subprocess so memory is fully returned to the OS.
"""

import gc
import os
import pickle
import sys
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = {
    "magnetosheath": "Magnetosheath_dataset.pkl",
    "magnetosphere": "Magnetosphere_dataset.pkl",
    "solar_wind": "SolarWind_dataset.pkl",
}

ROW_GROUP_SIZE = 500_000


def _pickle_to_tmp_parquet(pickle_path: Path) -> Path:
    """Pass 1: load pickle, write a single temporary Parquet file, free memory."""
    print(f"  pass 1: loading pickle ...")
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    print(f"  shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    if "time" in df.columns:
        df.rename(columns={"time": "Time"}, inplace=True)

    tmp = Path(tempfile.mktemp(suffix=".parquet"))
    print(f"  writing temporary parquet → {tmp}")
    table = pa.Table.from_pandas(df, preserve_index=False)
    del df
    gc.collect()
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_SIZE)
    del table
    gc.collect()
    return tmp


def _repartition_by_sc(tmp_parquet: Path, out_dir: Path) -> None:
    """Pass 2: lazy-scan the temp Parquet file and write Hive partitions."""
    print("  pass 2: repartitioning by SC ...")
    lf = pl.scan_parquet(tmp_parquet)
    sc_values = lf.select("SC").unique().collect()["SC"].to_list()

    for sc in sorted(sc_values):
        sc_dir = out_dir / f"SC={sc}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        part = (
            lf.filter(pl.col("SC") == sc)
            .drop("SC")
            .sort("Time")
            .collect()
        )
        part.write_parquet(sc_dir / "part-0.parquet", row_group_size=ROW_GROUP_SIZE)
        print(f"  SC={sc}: {len(part)} rows")
        del part


def convert_region(pickle_path: Path, out_dir: Path, region: str) -> None:
    print(f"Converting {region} ({pickle_path}) ...")
    region_dir = out_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)

    tmp_parquet = _pickle_to_tmp_parquet(pickle_path)
    try:
        _repartition_by_sc(tmp_parquet, region_dir)
    finally:
        tmp_parquet.unlink(missing_ok=True)

    print(f"  done: {region}")


def _convert_in_subprocess(pickle_path: Path, out_dir: Path, region: str) -> None:
    """Fork a child process to convert one region, so memory is fully reclaimed."""
    pid = os.fork()
    if pid == 0:
        try:
            convert_region(pickle_path, out_dir, region)
        except Exception as e:
            print(f"ERROR converting {region}: {e}", file=sys.stderr)
            os._exit(1)
        os._exit(0)
    else:
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
            print(f"FAILED: {region} (exit code {os.WEXITSTATUS(status)})", file=sys.stderr)
            sys.exit(1)
        if os.WIFSIGNALED(status):
            print(f"KILLED: {region} (signal {os.WTERMSIG(status)})", file=sys.stderr)
            sys.exit(1)


def main(pickle_dir: Path, output_dir: Path) -> None:
    """Convert MANGO pickle files to Hive-partitioned Parquet."""
    for region, filename in REGIONS.items():
        pkl = pickle_dir / filename
        if not pkl.exists():
            print(f"SKIP {pkl} (not found)")
            continue
        _convert_in_subprocess(pkl, output_dir, region)

    print("All done.")


if __name__ == "__main__":
    import cyclopts

    app = cyclopts.App(help=__doc__)
    app.default(main)
    app()
