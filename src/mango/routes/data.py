import io

import polars as pl
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from mango.dataset import MangoDataset, get_dataset
from mango.models import RANGE_FILTERS, DatasetInfo, FilterInfo, Format, Region

router = APIRouter(tags=["data"])


@router.get("/regions")
def list_regions() -> list[str]:
    return [r.value for r in Region]


@router.get("/regions/{region}/info")
def region_info(region: Region, ds: MangoDataset = Depends(get_dataset)) -> DatasetInfo:
    lf = ds[region]
    schema = lf.collect_schema()
    return DatasetInfo(
        region=region,
        row_count=lf.select(pl.len()).collect().item(),
        columns=schema.names(),
    )


@router.get("/regions/{region}/columns")
def region_columns(region: Region, ds: MangoDataset = Depends(get_dataset)) -> list[str]:
    return ds[region].collect_schema().names()


@router.get("/regions/{region}/filters")
def region_filters(region: Region) -> list[FilterInfo]:
    """List available range filters for this region."""
    return [
        FilterInfo(
            name=name,
            column=f.column,
            unit=f.unit,
            description=f.description,
            params=f"{name}_min=..&{name}_max=..",
        )
        for name, f in RANGE_FILTERS.items()
        if region in f.regions
    ]


@router.get("/regions/{region}/data")
def region_data(
    request: Request,
    region: Region,
    # General filters
    columns: list[str] | None = Query(None, description="Columns to include (default: all)"),
    spacecraft: list[str] | None = Query(None, description="Filter by spacecraft (e.g. THA, C1, MMS1)"),
    time_min: str | None = Query(None, description="Start time (ISO 8601)"),
    time_max: str | None = Query(None, description="End time (ISO 8601)"),
    sw_paired_only: bool = Query(False, description="Only return points with upstream SW pairing"),
    normalized_only: bool = Query(False, description="Only return spatially normalized points"),
    limit: int = Query(100_000, ge=1, le=10_000_000, description="Max rows to return"),
    format: Format = Query(Format.arrow, description="Output format: arrow or csv"),
    # Range filters are passed as arbitrary query params (e.g. bz_imf_min=-5&pd_sw_max=4)
    # and extracted from the raw query string below.
    ds: MangoDataset = Depends(get_dataset),
):
    """Return a subset of the dataset, filtered by solar wind conditions and more.

    Range filters use `{name}_min` / `{name}_max` query params.
    See `GET /regions/{region}/filters` for the full list.

    Examples:
    - Southward IMF: `bz_imf_max=-2`
    - High pressure: `pd_sw_min=5`
    - Near magnetopause in sheath: `d_msh_max=0.3`
    - Specific clock angle range: combine `by_imf_min/max` + `bz_imf_min/max`
    """
    raw_params = dict(request.query_params)

    df = ds.query(
        region,
        raw_params,
        columns=columns,
        spacecraft=spacecraft,
        time_min=time_min,
        time_max=time_max,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        limit=limit,
    )

    if format == Format.csv:
        buf = df.write_csv()
        return StreamingResponse(
            iter([buf]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mango_{region}.csv"},
        )

    buf = io.BytesIO()
    df.write_ipc(buf)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/vnd.apache.arrow.stream",
        headers={"Content-Disposition": f"attachment; filename=mango_{region}.arrow"},
    )
