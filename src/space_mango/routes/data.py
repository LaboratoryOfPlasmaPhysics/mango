import io

import polars as pl
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from space_mango.dataset import MangoDataset, get_dataset
from space_mango.filtering import time_window
from space_mango.frames import make_spec
from space_mango.models import (
    REGIONS,
    Format,
    Region,
    columns_for,
    filter_for_column,
    filters_for,
)
from space_mango.routes.schemas import (
    ColumnDescription,
    CountResult,
    DatasetInfo,
    FilterInfo,
    RegionDescription,
    SpacecraftCoverage,
)

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


def _filter_infos(region: Region) -> list[FilterInfo]:
    return [
        FilterInfo(
            name=name,
            column=f.column,
            unit=f.unit,
            description=f.description,
            params=f"{name}_min=..&{name}_max=..",
        )
        for name, f in filters_for(region).items()
    ]


@router.get("/regions/{region}/filters")
def region_filters(region: Region) -> list[FilterInfo]:
    """List available range filters for this region."""
    return _filter_infos(region)


@router.get("/regions/{region}/describe")
def region_describe(region: Region, ds: MangoDataset = Depends(get_dataset)) -> RegionDescription:
    schema = ds[region].collect_schema()
    catalog = columns_for(region)
    cols: list[ColumnDescription] = []
    for name, dtype in schema.items():
        info = catalog.get(name)
        cols.append(
            ColumnDescription(
                name=name,
                dtype=str(dtype),
                unit=info.unit if info else "",
                frame=info.frame if info else "",
                description=info.description_for(region) if info else "",
                computed=info.computed if info else "",
                filter=filter_for_column(region, name),
            )
        )
    return RegionDescription(
        region=region.value,
        definition=REGIONS[region].definition,
        columns=cols,
        filters=_filter_infos(region),
    )


@router.get("/regions/{region}/spacecraft")
def region_spacecraft(
    region: Region, ds: MangoDataset = Depends(get_dataset)
) -> list[SpacecraftCoverage]:
    return [
        SpacecraftCoverage(**row)
        for row in ds.coverage(region).rename({"SC": "sc"}).to_dicts()
    ]


@router.get("/regions/{region}/count")
def region_count(
    request: Request,
    region: Region,
    columns: list[str] | None = Query(None),
    spacecraft: list[str] | None = Query(None),
    start: str | None = Query(None),
    stop: str | None = Query(None),
    time_min: str | None = Query(None),
    time_max: str | None = Query(None),
    sw_paired_only: bool = Query(False),
    normalized_only: bool = Query(False),
    frame: str | None = Query(None, description="'pgsm': count the rows of the PGSM transform"),
    cone_min: float | None = Query(None, description="PGSM magnetosheath cone, degrees"),
    cone_max: float | None = Query(None),
    tilt_deg_min: float | None = Query(None, description="PGSM magnetosphere tilt, degrees"),
    tilt_deg_max: float | None = Query(None),
    ds: MangoDataset = Depends(get_dataset),
) -> CountResult:
    """Rows a /data request with the same parameters would return, and an estimated size.
    With frame=pgsm: the rows get_data(frame='pgsm') returns (selected twice = counted twice)."""
    start_dt, stop_dt, stop_inclusive = time_window(start, stop, time_min, time_max)
    cone = None if cone_min is None and cone_max is None else (cone_min, cone_max)
    tilt = None if tilt_deg_min is None and tilt_deg_max is None else (tilt_deg_min, tilt_deg_max)
    pgsm = make_spec(region.value, frame, cone=cone, tilt=tilt, require_clock=False)  # pyright: ignore[reportArgumentType]
    n_rows, est_bytes = ds.count(
        region,
        dict(request.query_params),
        columns=columns,
        spacecraft=spacecraft,
        start=start_dt,
        stop=stop_dt,
        stop_inclusive=stop_inclusive,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        pgsm=pgsm,
    )
    return CountResult(n_rows=n_rows, est_bytes=est_bytes)


def frame_response(df: pl.DataFrame, fmt: Format, name: str) -> StreamingResponse:
    if fmt == Format.csv:
        return StreamingResponse(
            iter([df.write_csv()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mango_{name}.csv"},
        )
    buf = io.BytesIO()
    df.write_ipc(buf)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/vnd.apache.arrow.stream",
        headers={"Content-Disposition": f"attachment; filename=mango_{name}.arrow"},
    )


@router.get("/regions/{region}/data")
def region_data(
    request: Request,
    region: Region,
    # General filters
    columns: list[str] | None = Query(None, description="Columns to include (default: all)"),
    spacecraft: list[str] | None = Query(None, description="Filter by spacecraft (e.g. THA, C1, MMS)"),
    time_min: str | None = Query(None, description="Legacy inclusive start (use start)"),
    time_max: str | None = Query(None, description="Legacy inclusive end (use stop)"),
    start: str | None = Query(None, description="Start time, inclusive (ISO 8601)"),
    stop: str | None = Query(None, description="Stop time, exclusive (ISO 8601)"),
    sw_paired_only: bool = Query(False, description="Only return points with upstream SW pairing"),
    normalized_only: bool = Query(False, description="Only return spatially normalized points"),
    limit: int | None = Query(None, ge=1, le=10_000_000, description="Max rows to return (default: all)"),
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
    start_dt, stop_dt, stop_inclusive = time_window(start, stop, time_min, time_max)
    df = ds.query(
        region,
        dict(request.query_params),
        columns=columns,
        spacecraft=spacecraft,
        start=start_dt,
        stop=stop_dt,
        stop_inclusive=stop_inclusive,
        sw_paired_only=sw_paired_only,
        normalized_only=normalized_only,
        limit=limit,
    )

    return frame_response(df, format, region.value)
