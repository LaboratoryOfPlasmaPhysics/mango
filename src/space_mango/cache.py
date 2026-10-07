"""On-disk fragment cache, modelled on speasy's (speasy/core/cache/), simplified because a
published MANGO version never changes: the version is part of the path, so nothing expires.

Layout: <root>/<version>/<region>/SC=<sc>/<column>/<YYYY-MM>.parquet
A month with no data is stored as an empty file, so it is not fetched again.
Server metadata (dataset, regions, filters, describe, spacecraft) is kept as JSON in
<root>/<version>/_meta/ so offline=True works with no network. Eviction only removes fragments.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypedDict

import platformdirs
import polars as pl

from space_mango.errors import MangoError


def default_cache_dir() -> Path:
    env = os.environ.get("SPACE_MANGO_CACHE_DIR")
    return Path(env) if env else Path(platformdirs.user_cache_dir("space-mango"))


def default_max_bytes() -> int:
    env = os.environ.get("SPACE_MANGO_CACHE_SIZE")
    return int(env) if env else 10 * 1024**3


def month_start(m: date) -> datetime:
    return datetime(m.year, m.month, 1)


def next_month(m: date) -> date:
    return date(m.year + m.month // 12, m.month % 12 + 1, 1)


def months_between(lo: datetime, hi: datetime) -> list[date]:
    """First-of-month dates covering [lo, hi], hi inclusive."""
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


def _atomic_write(target: Path, write: Callable[[Path], object]) -> None:
    """Write via a temp file + os.replace: readers never see a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, target)


class CacheInfo(TypedDict):
    root: str
    n_files: int
    size_bytes: int
    max_bytes: int


class FragmentCache:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root: Path = root
        self.max_bytes: int = max_bytes

    def path(self, version: str, region: str, sc: str, column: str, month: date) -> Path:
        return self.root / version / region / f"SC={sc}" / column / f"{month:%Y-%m}.parquet"

    def has(self, version: str, region: str, sc: str, month: date, columns: list[str]) -> bool:
        return all(self.path(version, region, sc, c, month).is_file() for c in columns)

    def write_months(
        self, version: str, region: str, sc: str, months: list[date], df: pl.DataFrame
    ) -> None:
        df = df.sort("Time", maintain_order=True)
        for m in months:
            part = df.filter(
                (pl.col("Time") >= month_start(m)) & (pl.col("Time") < month_start(next_month(m)))
            )
            for column in df.columns:
                frag = part.select(column)
                _atomic_write(
                    self.path(version, region, sc, column, m),
                    lambda tmp, frag=frag: frag.write_parquet(tmp, compression="zstd"),
                )

    def read_month(
        self, version: str, region: str, sc: str, month: date, columns: list[str]
    ) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for c in columns:
            p = self.path(version, region, sc, c, month)
            os.utime(p)  # mark as recently used for eviction
            parts.append(pl.read_parquet(p))
        if len({p.height for p in parts}) > 1:
            raise MangoError(
                "Inconsistent cache fragments in "
                f"{self.path(version, region, sc, '*', month).parent.parent}; "
                "call MangoClient.cache_clear() and retry."
            )
        return pl.concat(parts, how="horizontal")

    def _meta_path(self, version: str, name: str) -> Path:
        return self.root / version / "_meta" / name

    def write_meta(self, version: str, name: str, data: object) -> None:
        text = json.dumps(data)
        _atomic_write(self._meta_path(version, name), lambda tmp: tmp.write_text(text))

    def read_meta(self, version: str, name: str) -> Any | None:
        """Stored JSON, or None if this version has no such metadata file."""
        p = self._meta_path(version, name)
        return json.loads(p.read_text()) if p.is_file() else None

    def latest_version(self) -> str | None:
        """The version whose dataset metadata was stored most recently, if any."""
        if not self.root.exists():
            return None
        metas = list(self.root.glob("*/_meta/dataset.json"))
        if not metas:
            return None
        return max(metas, key=lambda p: p.stat().st_mtime).parent.parent.name

    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [p for p in self.root.rglob("*.parquet") if p.is_file()]

    def evict(self) -> None:
        """Delete least recently used fragments until the cache fits in max_bytes."""
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)

    def info(self) -> CacheInfo:
        files = self._files()
        return {
            "root": str(self.root),
            "n_files": len(files),
            "size_bytes": sum(p.stat().st_size for p in files),
            "max_bytes": self.max_bytes,
        }

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
