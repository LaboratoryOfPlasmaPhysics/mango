"""Smoke-test a MANGO server before publishing a client release (read-only).

Checks that the server speaks the 0.2 API: health, dataset version header, discovery
endpoints, filters that actually filter, and HTTP 400 on unknown names.

Usage:
    uv run python scripts/check_server.py                       # public server
    uv run python scripts/check_server.py http://localhost:8000 # any server

Exit code 0 when every check passes, 1 otherwise. Only GET requests are sent.
"""

from __future__ import annotations

import sys

import httpx

PUBLIC_URL = "http://sciqlop.lpp.polytechnique.fr/mango"
REGIONS = ("magnetosphere", "magnetosheath", "solar_wind")


def check(http: httpx.Client) -> list[str]:
    """Run every check against `http` (base URL already set). Return the failures."""
    problems: list[str] = []

    def get(path: str, **params: object) -> httpx.Response:
        return http.get(path, params=params)  # pyright: ignore[reportArgumentType]

    r = get("/health")
    if r.status_code != 200:
        return [f"/health answered {r.status_code}; is the server running?"]

    r = get("/api/v1/dataset")
    if r.status_code == 404:
        return ["/api/v1/dataset is missing: the server still runs the 0.1 code"]
    version = r.json().get("version")
    if not version:
        problems.append("/api/v1/dataset has no version")
    if r.headers.get("X-Mango-Dataset-Version") != version:
        problems.append("X-Mango-Dataset-Version header missing or different from /dataset")

    if sorted(get("/api/v1/regions").json()) != sorted(REGIONS):
        problems.append("/api/v1/regions does not list the three regions")

    for region in REGIONS:
        d = get(f"/api/v1/regions/{region}/describe")
        if d.status_code != 200 or not d.json().get("columns"):
            problems.append(f"/regions/{region}/describe failed ({d.status_code})")
        s = get(f"/api/v1/regions/{region}/spacecraft")
        if s.status_code != 200 or not s.json():
            problems.append(f"/regions/{region}/spacecraft failed or empty ({s.status_code})")

    total = get("/api/v1/regions/magnetosheath/count")
    near_mp = get("/api/v1/regions/magnetosheath/count", d_msh_max=0.3)
    if total.status_code != 200 or near_mp.status_code != 200:
        problems.append("/regions/magnetosheath/count failed")
    elif not 0 < near_mp.json()["n_rows"] < total.json()["n_rows"]:
        problems.append("d_msh_max=0.3 does not reduce the magnetosheath row count (dead filter?)")

    bad = get("/api/v1/regions/magnetosheath/data", spacecraft="MMS1", limit=1)
    if bad.status_code != 400 or bad.json()["detail"]["error"] != "unknown_spacecraft":
        problems.append("unknown spacecraft is not rejected with HTTP 400 unknown_spacecraft")

    return problems


def main() -> None:
    url = (sys.argv[1] if len(sys.argv) > 1 else PUBLIC_URL).rstrip("/")
    with httpx.Client(base_url=url, timeout=300) as http:
        problems = check(http)
    if problems:
        print(f"{url}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"{url}: OK, MANGO 0.2 API")


if __name__ == "__main__":
    main()
