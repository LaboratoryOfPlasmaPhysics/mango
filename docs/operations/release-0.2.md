# Releasing space-mango 0.2

Order matters: the 0.2 client fails against the 0.1 server.

1. **Merge PR #1** into `main` (CI green: tests on Python 3.11–3.14, docs build).
2. **Deploy the server** from `main`. See [deploy-0.2.md](deploy-0.2.md).
3. **Smoke-test the public server:**
   ```bash
   uv run python scripts/check_server.py
   ```
   It must print `OK, MANGO 0.2 API` and exit 0. Do not continue otherwise.
4. **Create the GitHub release** `v0.2.0` on `main` (this also creates the tag; the version
   comes from it via uv-dynamic-versioning):
   ```bash
   gh release create v0.2.0 --target main --title v0.2.0 --notes-file <notes>
   ```
5. **PyPI publishing is automatic:** publishing the release triggers
   `.github/workflows/publish.yml`, which builds and uploads to PyPI with trusted publishing
   (no credentials needed). Check that the "Publish to PyPI" run succeeds.
6. **Check the published package** in a fresh environment against the public server:
   ```bash
   uv run --isolated --with space-mango==0.2.0 python -c \
     "import space_mango as m; print(m.dataset_info()['version']); print(m.describe('magnetosheath').height)"
   ```
7. **Read the Docs:** the project is active at https://space-mango.readthedocs.io and rebuilds
   on every push to `main` from `.readthedocs.yaml` (no network access during the build).

Afterwards: announce the dead-filter fix (`d_msh`, `d_msp`, `tilt` were silently ignored
before 0.2), so users can re-check analyses that relied on them.
