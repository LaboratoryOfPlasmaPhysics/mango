# Deploying the MANGO 0.2 server

For whoever operates `sciqlop.lpp.polytechnique.fr/mango`. Deploy this before
`space-mango` 0.2 is published to PyPI: the 0.2 client needs the new endpoints.

## What changes

- **Code only.** Same data directory, same Parquet files, no new data files, nothing to
  migrate. Served column names are unchanged.
- **Same deployment:** `docker/build.sh` and `docker/entry_point.sh` are unchanged
  (gunicorn, 4 workers, `--preload`).
- **Old clients keep working.** `space-mango` 0.1.x uses `/regions`, `/columns`, `/filters`,
  `/info` and `/data`, whose responses are unchanged.
- **Stricter requests.** Typos, unknown parameters, NaN filter values and inverted time
  windows now get HTTP 400 with a JSON explanation instead of empty or unfiltered data.
- **Bug fix:** the `d_msh`, `d_msp` and `tilt` filters now filter. They were silently ignored.

## New endpoints (`/api/v1`, all read-only)

| endpoint | cost |
|---|---|
| `/regions/{r}/describe` | schema only |
| `/regions/{r}/spacecraft` | scans the `Time` column of the region once per worker process, then cached in memory |
| `/regions/{r}/count` | same work as a `/data` query, without sending the rows |
| `/timeline?sc=&start=&stop=` | one spacecraft, at most 31 days, all regions |
| `/dataset` | version, citation, schema checksum (reads schemas only) |

The first `/spacecraft` call per worker on the full data may take several seconds
(not measured on production data).

## Environment variables

| variable | default | notes |
|---|---|---|
| `MANGO_DATA_DIR` | `/data/mango` | unchanged |
| `MANGO_ROOT_PATH` | `""` | unchanged (reverse-proxy prefix) |
| `PORT` | `8000` | unchanged |
| `MANGO_DATASET_VERSION` | `2026.0` | **new.** Sent in every response (`X-Mango-Dataset-Version`). Clients cache data under this version plus a schema checksum, so **any change to the served data must come with a new value** (e.g. `2026.1`). Leave the default for this deployment. |

## Steps

1. Check out `main` at the release commit and build the image: `./docker/build.sh`.
2. Restart the container exactly as today, with the same volume and environment.
3. Look at the startup log. The server compares its column catalog with the served
   schema and logs `catalog/schema mismatch: …` lines if they differ. None are expected.
   If some appear, send them to the MANGO team.
4. Verify from any machine with the repository:
   ```bash
   uv run python scripts/check_server.py http://sciqlop.lpp.polytechnique.fr/mango
   ```
   Expected output: `… OK, MANGO 0.2 API`. The script only sends GET requests.

## Rollback

Redeploy the previous image (built from tag `v0.1.1`). Clients 0.1.x work with either
server. Clients 0.2 need this one.
