"""Notebooks are stored without outputs; the docs build executes them."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "docs" / "examples").glob("*.ipynb"))


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"{path.name}: strip outputs (cell {cell.get('id')})"
            assert cell.get("execution_count") is None
