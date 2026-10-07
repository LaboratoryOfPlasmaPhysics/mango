"""Pydantic response models (server only)."""

from datetime import datetime

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
