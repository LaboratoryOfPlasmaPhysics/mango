from enum import StrEnum

from fastapi import Query
from pydantic import BaseModel


class Format(StrEnum):
    arrow = "arrow"
    csv = "csv"


class SubsetParams:
    """Query parameters for subsetting a MANGO region dataset."""

    def __init__(
        self,
        columns: list[str] | None = Query(None, description="Columns to include (default: all)"),
        spacecraft: list[str] | None = Query(None, description="Filter by spacecraft (e.g. THA, C1, MMS1)"),
        time_min: str | None = Query(None, description="Start time (ISO 8601)"),
        time_max: str | None = Query(None, description="End time (ISO 8601)"),
        x_gsm_min: float | None = Query(None, description="Min X_gsm (Re)"),
        x_gsm_max: float | None = Query(None, description="Max X_gsm (Re)"),
        y_gsm_min: float | None = Query(None, description="Min Y_gsm (Re)"),
        y_gsm_max: float | None = Query(None, description="Max Y_gsm (Re)"),
        z_gsm_min: float | None = Query(None, description="Min Z_gsm (Re)"),
        z_gsm_max: float | None = Query(None, description="Max Z_gsm (Re)"),
        bz_imf_min: float | None = Query(None, description="Min Bz_imf (nT)"),
        bz_imf_max: float | None = Query(None, description="Max Bz_imf (nT)"),
        pd_sw_min: float | None = Query(None, description="Min dynamic pressure (nPa)"),
        pd_sw_max: float | None = Query(None, description="Max dynamic pressure (nPa)"),
        limit: int = Query(100_000, ge=1, le=10_000_000, description="Max rows to return"),
        format: Format = Query(Format.arrow, description="Output format: arrow or csv"),
    ):
        self.columns = columns
        self.spacecraft = spacecraft
        self.time_min = time_min
        self.time_max = time_max
        self.x_gsm_min = x_gsm_min
        self.x_gsm_max = x_gsm_max
        self.y_gsm_min = y_gsm_min
        self.y_gsm_max = y_gsm_max
        self.z_gsm_min = z_gsm_min
        self.z_gsm_max = z_gsm_max
        self.bz_imf_min = bz_imf_min
        self.bz_imf_max = bz_imf_max
        self.pd_sw_min = pd_sw_min
        self.pd_sw_max = pd_sw_max
        self.limit = limit
        self.format = format


class DatasetInfo(BaseModel):
    region: str
    row_count: int
    columns: list[str]
