from enum import StrEnum

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from mango.dataset import get_dataset
from mango.models import DatasetInfo, SubsetParams

router = APIRouter(tags=["data"])


class Region(StrEnum):
    magnetosphere = "magnetosphere"
    magnetosheath = "magnetosheath"
    solar_wind = "solar_wind"


@router.get("/regions")
def list_regions() -> list[str]:
    return [r.value for r in Region]


@router.get("/regions/{region}/info")
def region_info(region: Region, ds=Depends(get_dataset)) -> DatasetInfo:
    df = ds[region]
    return DatasetInfo(
        region=region,
        row_count=df.height,
        columns=df.columns,
    )


@router.get("/regions/{region}/columns")
def region_columns(region: Region, ds=Depends(get_dataset)) -> list[str]:
    return ds[region].columns


@router.get("/regions/{region}/data")
def region_data(
    region: Region,
    params: SubsetParams = Depends(),
    ds=Depends(get_dataset),
):
    """Return a subset of the dataset as Parquet (default) or CSV."""
    df = ds.query(region, params)

    if params.format == "csv":
        buf = df.write_csv()
        return StreamingResponse(
            iter([buf]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mango_{region}.csv"},
        )

    # Default: Parquet (compact, typed, fast)
    buf = df.write_ipc()
    return StreamingResponse(
        iter([buf]),
        media_type="application/vnd.apache.arrow.stream",
        headers={"Content-Disposition": f"attachment; filename=mango_{region}.arrow"},
    )
