#!/usr/bin/env bash
uv run gunicorn space_mango.main:app --preload --timeout 120 -w 4 --threads=4 -k space_mango.worker.MangoUvicornWorker --bind 0.0.0.0:${PORT}
