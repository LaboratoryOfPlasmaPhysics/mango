import importlib.util
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _server_module():
    spec = importlib.util.spec_from_file_location("_docs_server", DOCS / "_docs_server.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_server_serves_the_sample_with_sample_version():
    srv = _server_module()
    proc, url = srv.start(DOCS / "data")
    try:
        r = httpx.get(f"{url}/api/v1/dataset", timeout=30)
        assert r.json()["version"] == "2026.0-docs-sample"
        assert r.headers["X-Mango-Dataset-Version"] == "2026.0-docs-sample"
    finally:
        srv.stop(proc)
    assert proc.poll() is not None


def test_start_times_out_with_clear_error(tmp_path, monkeypatch):
    srv = _server_module()
    monkeypatch.setattr(srv, "_command", lambda port: [sys.executable, "-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="did not become healthy"):
        srv.start(tmp_path, timeout=2)


@pytest.mark.slow
def test_failing_notebook_fails_the_build(tmp_path):
    """A cell error must make sphinx-build exit non-zero."""
    import nbformat

    src = tmp_path / "src"
    src.mkdir()
    (src / "conf.py").write_text(
        "extensions = ['myst_nb']\nnb_execution_mode = 'force'\nnb_execution_raise_on_error = True\n")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise ValueError('boom')")])
    nbformat.write(nb, src / "index.ipynb")
    res = subprocess.run([sys.executable, "-m", "sphinx", "-W", "-b", "html", str(src), str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert res.returncode != 0
