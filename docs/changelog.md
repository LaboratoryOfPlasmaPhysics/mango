# Changelog

## 0.3

- **Frames:** `frame="gsm"`, `"swi"` (magnetosheath) or `"pgsm"` choose the returned vector
  columns; `cone`, `clock` and `tilt` select samples (GSM ranges; in SWI/PGSM the cone is
  measured from −V_sw); PGSM rotates magnetosheath data to a target clock angle or applies
  the dipole-tilt symmetry (Michotte de Welle 2024). Without `frame`, results are unchanged.
- **Release order:** deploy the 0.3 server before publishing the client, for
  `count(...)` with a frame or selection. `get_data` with a frame also works against a 0.2
  server.

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
