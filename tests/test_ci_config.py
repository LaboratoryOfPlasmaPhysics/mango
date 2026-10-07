"""CI must call tools that exist (it used to call a missing devtools/lint.py)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ci_does_not_call_missing_lint_script():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    for text in (ci, makefile):
        assert "devtools/lint.py" not in text
        assert "ruff check" in text
        assert "basedpyright -p pyproject.toml" in text
        assert "codespell" in text
