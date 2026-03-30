from pathlib import Path

import cyclopts

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

    uvicorn.run("mango.main:app", host=host, port=port, reload=reload)


def main() -> None:
    app()
