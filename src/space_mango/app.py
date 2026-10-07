import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from space_mango.errors import QueryError
from space_mango.routes import data, health


async def _query_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, QueryError)
    return JSONResponse(status_code=400, content={"detail": exc.to_dict()})


def create_app() -> FastAPI:
    app = FastAPI(
        title="MANGO",
        description="Magnetosphere Atlas from Normalized Geospace Observations — data subsetting API",
        version="0.1.0",
        root_path=os.environ.get("MANGO_ROOT_PATH", ""),
    )
    app.add_exception_handler(QueryError, _query_error_handler)
    app.include_router(health.router)
    app.include_router(data.router, prefix="/api/v1")
    return app
