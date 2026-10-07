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

Event choice: scanning 2008-06-01..2008-10-01 (THA, one-day windows, every 3rd day, then
daily 2008-08-07..11), the only day with THA rows in all three regions is 2008-08-09
({'magnetosphere': 4237, 'magnetosheath': 7971, 'solar_wind': 431}); neighbours have
solar_wind == 0. Window: 2008-08-09 .. 2008-08-11 (EVENT_START inclusive, EVENT_STOP exclusive).

Final tuning: N_WINDOWS = 120, WINDOW = 10 min (see docs/data/README.md for the size).
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
N_WINDOWS = 120                       # tuned so the total stays under MAX_BYTES
WINDOW = timedelta(minutes=10)
PERIOD = (datetime(2007, 1, 1), datetime(2021, 6, 1))
EVENT_SC = "THA"
EVENT_START = "2008-08-09T00:00:00"   # found with --find-event, see docstring
EVENT_STOP = "2008-08-11T00:00:00"


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
