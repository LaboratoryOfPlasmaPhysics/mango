# Changelog

## 0.3

- **PGSM frame:** `get_data(region, frame="pgsm", ...)` returns magnetosheath data selected
  by IMF cone angle and rotated to a target IMF clock angle, or magnetosphere data with the
  dipole-tilt symmetry (Michotte de Welle 2024). `count(..., frame="pgsm")` gives the exact
  number of rows; it needs the 0.3 server. See the user guide.
- **Release order:** deploy the 0.3 server before publishing the client, for
  `count(frame="pgsm")`. `get_data(frame="pgsm")` also works against a 0.2 server.

## 0.2

The default server URL can be overridden with the `SPACE_MANGO_URL` environment variable
(for example to use a self-hosted server).

`get_data` returns a `MangoResult` (use `.to_polars()` for the previous
polars DataFrame); `time_min`/`time_max` are deprecated in favour of `start`/`stop`;
unknown spacecraft, columns or filters now raise an error instead of returning empty or
unfiltered data. The spacecraft name for MMS is `MMS` (not `MMS1`).

- **Re-check analyses that used `d_msh`, `d_msp` or `tilt`:** before this release the
  server silently ignored these three filters and returned unfiltered data.
- **Release order:** the 0.2 client needs a 0.2 server — deploy the server first. Against
  an older server the client raises `ServerError` ("older than 0.2"); keep
  `space-mango<0.2` until the server is upgraded.
