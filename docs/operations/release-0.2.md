# Releasing space-mango 0.2

Order matters: the 0.2 client fails against the 0.1 server.

1. **Merge PR #1** into `main` (CI green: tests on Python 3.11–3.14, docs build).
2. **Deploy the server** from `main`. See [deploy-0.2.md](deploy-0.2.md).
3. **Smoke-test the public server:**
   ```bash
   uv run python scripts/check_server.py
   ```
   It must print `OK, MANGO 0.2 API` and exit 0. Do not continue otherwise.
4. **Tag and build** on `main`. The version comes from the git tag (uv-dynamic-versioning):
   ```bash
   git tag -a v0.2.0 -m "space-mango 0.2.0"
   uv build
   ```
5. **Publish to PyPI** with the project's PyPI credentials (`uv publish`), then push the tag.
6. **Check the published package** in a fresh environment against the public server:
   ```bash
   uv run --isolated --with space-mango==0.2.0 python -c \
     "import space_mango as m; print(m.dataset_info()['version']); print(m.describe('magnetosheath').height)"
   ```
7. **Read the Docs:** activate the project for the repository (needs a repository admin).
   The site builds from `.readthedocs.yaml` and needs no network access during the build.

Afterwards: announce the dead-filter fix (`d_msh`, `d_msp`, `tilt` were silently ignored
before 0.2), so users can re-check analyses that relied on them.
