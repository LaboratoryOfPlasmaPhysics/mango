"""scripts/check_server.py passes on a 0.2 server and reports an 0.1 server clearly."""

import importlib.util
from pathlib import Path

import httpx

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_server.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_server", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_new_server_passes(api):
    assert _module().check(api) == []


def test_old_server_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "Not Found"})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://old") as http:
        problems = _module().check(http)
    assert problems == ["/api/v1/dataset is missing: the server still runs the 0.1 code"]
