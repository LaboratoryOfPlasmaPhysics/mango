"""Cross-region routes: /timeline and /dataset."""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from space_mango.dataset import MangoDataset, get_dataset
from space_mango.filtering import TIMELINE_PARAMS, parse_time, reject_unknown_params
from space_mango.models import DATASET_TITLE, Format, citation_bibtex
from space_mango.routes.data import frame_response
from space_mango.routes.schemas import DatasetDescription

router = APIRouter(tags=["dataset"])


@router.get("/timeline")
def timeline(
    request: Request,
    sc: str = Query(..., description="Spacecraft, e.g. THA"),
    start: str = Query(..., description="Start time, inclusive (ISO 8601)"),
    stop: str = Query(..., description="Stop time, exclusive (ISO 8601); at most 31 days after start"),
    columns: list[str] | None = Query(None, description="Columns (default: all); Time, SC, region always included"),
    format: Format = Query(Format.arrow),
    ds: MangoDataset = Depends(get_dataset),
) -> StreamingResponse:
    """All rows of one spacecraft over an interval, across regions, with a `region` column."""
    reject_unknown_params("/timeline", request.query_params.keys(), TIMELINE_PARAMS)
    df = ds.timeline(sc, parse_time(start, param="start"), parse_time(stop, param="stop"), columns)
    return frame_response(df, format, f"timeline_{sc}")


@router.get("/dataset")
def dataset_info(request: Request, ds: MangoDataset = Depends(get_dataset)) -> DatasetDescription:
    version: str = request.app.state.dataset_version
    return DatasetDescription(
        version=version,
        title=DATASET_TITLE,
        citation=citation_bibtex(version, None),
        doi=None,
        schema_checksum=ds.schema_checksum(),
        features=["pgsm_count"],
    )
