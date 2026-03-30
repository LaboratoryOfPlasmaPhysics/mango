import os

from uvicorn_worker import UvicornWorker


class MangoUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {
        "root_path": os.environ.get("MANGO_ROOT_PATH", ""),
        "host": "0.0.0.0",
        "port": int(os.environ.get("PORT", 8000)),
        "proxy_headers": True,
    }
