"""On-disk fragment cache, modelled on speasy's (speasy/core/cache/), simplified because a
published MANGO version never changes: the version is part of the path, so nothing expires.

Layout: <root>/<key>/<region>/SC=<sc>/<column>/<YYYY-MM>.parquet, where <key> is
<version>-<first 12 hex chars of the schema checksum> (see MangoClient._cache_key).
A month with no data is stored as an empty file, so it is not fetched again.
Server metadata (dataset, regions, filters, describe, spacecraft) is kept as JSON in
<root>/<key>/_meta/ so offline=True works with no network. Eviction only removes fragments.
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


def month_slice(df: pl.DataFrame, m: date) -> pl.DataFrame:
    """Rows of df whose Time falls in month m."""
    return df.filter(
        (pl.col("Time") >= month_start(m)) & (pl.col("Time") < month_start(next_month(m)))
    )


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
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class CacheInfo(TypedDict):
    root: str
    n_files: int
    size_bytes: int
    max_bytes: int


class FragmentCache:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root: Path = root
        self.max_bytes: int = max_bytes

    def path(self, key: str, region: str, sc: str, column: str, month: date) -> Path:
        return self.root / key / region / f"SC={sc}" / column / f"{month:%Y-%m}.parquet"

    def missing(
        self, key: str, region: str, sc: str, month: date, columns: list[str]
    ) -> list[str]:
        """The columns of this month that have no fragment on disk."""
        return [c for c in columns if not self.path(key, region, sc, c, month).is_file()]

    def has(self, key: str, region: str, sc: str, month: date, columns: list[str]) -> bool:
        return not self.missing(key, region, sc, month, columns)

    def write_month(self, key: str, region: str, sc: str, month: date, part: pl.DataFrame) -> None:
        """One fragment per column of part (rows of this month only, sorted by Time)."""
        for column in part.columns:
            frag = part.select(column)
            _atomic_write(
                self.path(key, region, sc, column, month),
                lambda tmp, frag=frag: frag.write_parquet(tmp, compression="zstd"),
            )

    def write_months(
        self, key: str, region: str, sc: str, months: list[date], df: pl.DataFrame
    ) -> None:
        df = df.sort("Time", maintain_order=True)
        for m in months:
            self.write_month(key, region, sc, m, month_slice(df, m))

    def read_month(
        self, key: str, region: str, sc: str, month: date, columns: list[str]
    ) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for c in columns:
            p = self.path(key, region, sc, c, month)
            try:
                os.utime(p)  # mark as recently used for eviction
            except PermissionError:
                pass  # read-only shared cache: still readable
            parts.append(pl.read_parquet(p))
        if len({p.height for p in parts}) > 1:
            raise MangoError(
                "Inconsistent cache fragments in "
                f"{self.path(key, region, sc, '*', month).parent.parent}; "
                "call MangoClient.cache_clear() and retry."
            )
        return pl.concat(parts, how="horizontal")

    def _meta_path(self, key: str, name: str) -> Path:
        return self.root / key / "_meta" / name

    def write_meta(self, key: str, name: str, data: object) -> None:
        text = json.dumps(data)
        _atomic_write(self._meta_path(key, name), lambda tmp: tmp.write_text(text))

    def read_meta(self, key: str, name: str) -> Any | None:
        """Stored JSON, or None if this cache key has no such metadata file."""
        p = self._meta_path(key, name)
        return json.loads(p.read_text()) if p.is_file() else None

    def latest_key(self) -> str | None:
        """The cache key (directory) whose dataset metadata was stored most recently, if any."""
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
