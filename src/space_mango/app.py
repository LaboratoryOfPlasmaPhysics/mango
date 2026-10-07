import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from space_mango.dataset import get_dataset
from space_mango.errors import QueryError
from space_mango.models import DEFAULT_DATASET_VERSION
from space_mango.routes import data, dataset, health

logger = logging.getLogger("space_mango")


def _package_version() -> str:
    try:
        return version("space-mango")
    except PackageNotFoundError:
        return "0+unknown"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ds = get_dataset()
    if ds.exists():
        for problem in ds.check_catalog():
            logger.warning("catalog/schema mismatch: %s", problem)
    yield


async def _query_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, QueryError)
    return JSONResponse(status_code=400, content={"detail": exc.to_dict()})


def create_app() -> FastAPI:
    dataset_version = os.environ.get("MANGO_DATASET_VERSION", DEFAULT_DATASET_VERSION)
    app = FastAPI(
        title="MANGO",
        description="Magnetospheric Atlas of Normalized Geospace Observations — data subsetting API",
        version=_package_version(),
        root_path=os.environ.get("MANGO_ROOT_PATH", ""),
        lifespan=_lifespan,
    )
    app.state.dataset_version = dataset_version

    @app.middleware("http")
    async def _version_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Mango-Dataset-Version"] = dataset_version
        return response

    app.add_exception_handler(QueryError, _query_error_handler)
    app.include_router(health.router)
    app.include_router(data.router, prefix="/api/v1")
    app.include_router(dataset.router, prefix="/api/v1")
    return app
