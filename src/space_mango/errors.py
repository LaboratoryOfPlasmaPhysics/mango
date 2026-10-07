"""Errors shared by the MANGO server and client."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def did_you_mean(name: str, choices: Iterable[str]) -> str:
    """Return " Did you mean 'x'?" for the closest choice, or "" if nothing is close."""
    match = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f" Did you mean '{match[0]}'?" if match else ""


class QueryError(ValueError):
    """A request the server refuses (HTTP 400) instead of silently ignoring."""

    def __init__(self, code: str, message: str, valid: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.valid = sorted(valid)

    def to_dict(self) -> dict[str, object]:
        return {"error": self.code, "message": self.message, "valid": self.valid}
