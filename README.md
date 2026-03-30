# MANGO

Web service for subsetting and distributing the **MANGO** dataset
(Magnetosphere Atlas from Normalized Geospace Observations).

The full dataset is ~40 GB. This API lets users request only the subset they need — filtered by region, time range, spacecraft, spatial position, and upstream solar wind conditions — and receive it as Apache Arrow IPC or CSV.

## Quickstart

```bash
# Install
uv sync

# Point to your Parquet files
export MANGO_DATA_DIR=/path/to/mango/parquet

# Run
uv run mango
```

The API is at `http://localhost:8000`. Interactive docs at `/docs`.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /api/v1/regions` | List available regions |
| `GET /api/v1/regions/{region}/info` | Row count and columns |
| `GET /api/v1/regions/{region}/columns` | Column names |
| `GET /api/v1/regions/{region}/data` | Subset data (with filters) |

### Filters (query params on `/data`)

- `columns` — select specific columns
- `spacecraft` — filter by spacecraft (THA, C1, MMS1, ...)
- `time_min`, `time_max` — ISO 8601 time range
- `x_gsm_min/max`, `y_gsm_min/max`, `z_gsm_min/max` — spatial box (Re)
- `bz_imf_min/max` — IMF Bz filter (nT)
- `pd_sw_min/max` — dynamic pressure filter (nPa)
- `limit` — max rows (default 100k, max 10M)
- `format` — `arrow` (default) or `csv`

## Development

```bash
uv sync --group dev
uv run pytest
```

## Data

The service expects Parquet files in `MANGO_DATA_DIR`:
- `magnetosphere.parquet`
- `magnetosheath.parquet`
- `solar_wind.parquet`

*This project was built from [simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
