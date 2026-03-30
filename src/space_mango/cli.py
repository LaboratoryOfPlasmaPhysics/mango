from pathlib import Path

try:
    import cyclopts
except ImportError:
    raise ImportError(
        "The MANGO server requires extra dependencies.\n"
        "Install them with: pip install space-mango[server]"
    ) from None

app = cyclopts.App(name="mango", help="MANGO dataset service")


@app.command
def serve(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    data_dir: Path | None = None,
) -> None:
    """Start the MANGO API server."""
    import os

    import uvicorn

    if data_dir:
        os.environ["MANGO_DATA_DIR"] = str(data_dir)

    root_path = os.environ.get("MANGO_ROOT_PATH", "")
    uvicorn.run("space_mango.main:app", host=host, port=port, reload=reload, root_path=root_path)


def main() -> None:
    app()
