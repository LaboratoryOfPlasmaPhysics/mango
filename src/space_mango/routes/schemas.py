"""Pydantic response models (server only)."""

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
