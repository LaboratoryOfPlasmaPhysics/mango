"""Start a local MANGO server on the docs sample for notebook execution (docs build only)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _command(port: int) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "space_mango.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]


def start(data_dir: Path, version: str = "2026.0-docs-sample",
          timeout: float = 30.0) -> tuple[subprocess.Popen[bytes], str]:
    port = _free_port()
    env = {**os.environ, "MANGO_DATA_DIR": str(data_dir), "MANGO_DATASET_VERSION": version}
    proc = subprocess.Popen(_command(port), env=env)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return proc, url
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    stop(proc)
    raise RuntimeError(f"docs server did not become healthy within {timeout} s (data_dir={data_dir})")


def stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
