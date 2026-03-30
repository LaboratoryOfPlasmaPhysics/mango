# MANGO Client Package Design

**Goal:** Add a Python client to the `mango` package so astrophysics researchers can query the MANGO dataset with a single function call. Server dependencies become optional (`mango[server]`).

## User-facing API

```python
import mango

# Main entry point — returns polars DataFrame
df = mango.get_data("magnetosheath", spacecraft=["MMS"], bz_imf_max=-2)

# With time range and column selection
df = mango.get_data(
    "magnetosphere",
    time_min="2015-01-01",
    time_max="2016-01-01",
    columns=["Time", "SC", "Bx", "By", "Bz"],
    limit=500_000,
)

# Discovery
mango.regions()                    # → ["magnetosphere", "magnetosheath", "solar_wind"]
mango.columns("magnetosheath")    # → ["Time", "SC", "Bx", ...]
mango.filters("magnetosheath")    # → list of FilterInfo (name, unit, description)
```

All module-level functions delegate to a lazily-created global `MangoClient` instance
pointing at `http://sciqlop.lpp.polytechnique.fr/mango/`.

Power users can instantiate their own client:

```python
client = mango.MangoClient("http://localhost:8000")
df = client.get_data("magnetosheath")
```

## `MangoClient` class

```python
class MangoClient:
    def __init__(self, base_url: str = DEFAULT_URL) -> None: ...

    def get_data(
        self,
        region: str,
        *,
        columns: list[str] | None = None,
        spacecraft: list[str] | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        sw_paired_only: bool = False,
        normalized_only: bool = False,
        limit: int = 100_000,
        **filters: float,          # e.g. bz_imf_max=-2, pd_sw_min=5
    ) -> pl.DataFrame: ...

    def regions(self) -> list[str]: ...
    def columns(self, region: str) -> list[str]: ...
    def filters(self, region: str) -> list[FilterInfo]: ...
```

### Filter validation

`get_data` validates `**filters` kwargs before sending the request:

1. Each kwarg must match `{filter_name}_{min|max}` pattern.
2. `filter_name` must exist in the region's filter catalog (fetched from server, cached).
3. Value must be numeric.

Invalid filter → `MangoFilterError` with a helpful message:

```
MangoFilterError: 'd_msh_min' is not a valid filter for region 'magnetosphere'.
Available filters: bz_imf, by_imf, bx_imf, pd_sw, ..., d_msp
Hint: 'd_msh' is available for region 'magnetosheath'.
```

### Data transfer

Requests use Arrow IPC format (default server format). Response bytes are read into
a polars DataFrame via `pl.read_ipc(response.content)`. This is the most efficient
path — no CSV parsing overhead.

### Filter cache

The filter catalog for each region is fetched once per `MangoClient` instance and
cached in a dict. No TTL — the catalog is static for the lifetime of a deployment.

## Package structure

```
src/mango/
├── __init__.py          # re-exports MangoClient, get_data, regions, columns, filters
├── client.py            # MangoClient class + MangoFilterError
├── models.py            # shared (Region, RangeFilter, FilterInfo) — exists
├── app.py               # server — unchanged
├── dataset.py           # server — unchanged
├── cli.py               # server — unchanged
├── main.py              # server — unchanged
└── routes/              # server — unchanged
```

## Dependencies

```toml
dependencies = ["polars>=1.0", "pyarrow>=18.0", "httpx>=0.28"]

[project.optional-dependencies]
server = ["fastapi>=0.115", "uvicorn[standard]>=0.34", "cyclopts>=4.0"]
```

Base install (`pip install mango`) gets the client only. Server extras
(`pip install mango[server]`) adds FastAPI, uvicorn, cyclopts.

The `mango` CLI entry point requires `mango[server]`.

## Error handling

- `MangoFilterError` — invalid filter name or pattern (raised client-side before request)
- `httpx.HTTPStatusError` — server returned non-2xx (propagated as-is from httpx)
- Region validation: `get_data("invalid_region")` gets a 422 from the server, which
  httpx surfaces as `HTTPStatusError`. No client-side region validation needed since
  the server already validates via the `Region` enum.

## Testing

- Unit tests for filter validation logic (no server needed)
- Integration tests using `pytest` + `TestClient` (FastAPI's test client, no real HTTP)
  by constructing a `MangoClient` that points at a `TestClient`-wrapped app with
  fixture Parquet data
