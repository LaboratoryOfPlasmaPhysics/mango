"""Sphinx configuration for space-mango. Notebooks run against a local server on docs/data."""

import atexit
import os
import sys
import tempfile
from importlib.metadata import version as _version
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS))
import _docs_server  # noqa: E402

project = "MANGO"
author = "Laboratoire de Physique des Plasmas"
release = _version("space-mango")
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "furo"
html_title = "MANGO"
exclude_patterns = ["_build", "superpowers", "operations", "data", "**/.ipynb_checkpoints"]
myst_enable_extensions = ["colon_fence"]
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_execution_show_tb = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

_proc, _url = _docs_server.start(DOCS / "data")
atexit.register(_docs_server.stop, _proc)
os.environ["SPACE_MANGO_URL"] = _url
os.environ["SPACE_MANGO_CACHE_DIR"] = tempfile.mkdtemp(prefix="mango-docs-cache-")


def _escape_pipes(app, what, name, obj, options, lines):
    """Generated docstrings contain |r| (a norm), which reST reads as a substitution reference."""
    lines[:] = [line.replace("|r|", r"\|r\|") for line in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _escape_pipes)
