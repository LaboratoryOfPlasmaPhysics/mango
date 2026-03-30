from fastapi import FastAPI

from mango.routes import data, health


def create_app() -> FastAPI:
    app = FastAPI(
        title="MANGO",
        description="Magnetosphere Atlas from Normalized Geospace Observations — data subsetting API",
        version="0.1.0",
    )
    app.include_router(health.router)
    app.include_router(data.router, prefix="/api/v1")
    return app
