# Quickstart

```python
import space_mango as mango

# What is there? (nothing below downloads data)
mango.regions()                     # ['magnetosphere', 'magnetosheath', 'solar_wind']
mango.magnetosheath                 # <MANGO region 'magnetosheath': Between the bow shock and ...>
mango.describe("magnetosheath")     # column | unit | frame | description | filter | dtype
mango.spacecraft("magnetosheath")   # sc | start | stop | n_rows
mango.search("density")             # columns and filters matching a word
mango.count("magnetosheath", bz_imf_max=-2)   # {'n_rows', 'est_mb', 'download_mb_estimate'}

# Statistical study: southward IMF, inner magnetosheath
r = mango.magnetosheath.get_data(           # tab-complete the filters; help() lists their units
    spacecraft=["THA", "MMS"],              # spacecraft names: see mango.spacecraft(...)
    start="2016-01", stop="2021-01",        # start inclusive, stop exclusive
    columns=["Time", "Np", "Bx_swi", "R_norm"],
    bz_imf_max=-2, d_msh_max=0.3,
)
df = r.to_pandas()                          # or r.to_polars(), r.to_xarray()
r.metadata["Np"]                            # {'unit': 'cm⁻³', 'frame': '', 'description': ...}
print(r.cite())                             # BibTeX, with the dataset version

# Event context: where was THA, and when did it cross a boundary?
t = mango.timeline("THA", "2017-01-12T10:00", "2017-01-12T12:00")
t.to_intervals()                            # sc | region | start | stop | n_points
```

Results are cached on disk (`~/.cache/space-mango`, size cap `SPACE_MANGO_CACHE_SIZE`
bytes, default 10 GB); re-running a notebook does not download again.
`mango.cache.info()` / `mango.cache.clear()` manage it.

Full reference: {doc}`user_guide`.
