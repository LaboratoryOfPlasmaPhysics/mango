"""MangoResult: data plus the metadata needed to use and cite it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

from space_mango.errors import MangoError

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
        if not {"Time", "SC"} <= set(df.columns):
            raise MangoError(
                "to_intervals needs the Time and SC columns; request them in columns=…"
            )
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
