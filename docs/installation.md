# Installation

MANGO requires Python 3.11 or later.

```bash
pip install space-mango
```

Optional extras:

```bash
pip install "space-mango[pandas]"   # r.to_pandas()
pip install "space-mango[xarray]"   # r.to_xarray()
pip install "space-mango[server]"   # self-hosting the server
```

The `[docs]` extra is for building this documentation from a clone of the repository
(`pip install -e ".[docs]"`, then `sphinx-build -b html docs docs/_build/html`).
