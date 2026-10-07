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


class MangoError(Exception):
    """Base class for every error raised by space_mango."""


class UnknownRegionError(MangoError, ValueError):
    pass


class UnknownSpacecraftError(MangoError, ValueError):
    pass


class UnknownColumnError(MangoError, ValueError):
    pass


class MangoFilterError(MangoError, ValueError):
    pass


class TimeParseError(MangoError, ValueError):
    pass


class ServerError(MangoError):
    pass


class CacheMissError(MangoError):
    pass


_CODE_TO_ERROR: dict[str, type[MangoError]] = {
    "unknown_spacecraft": UnknownSpacecraftError,
    "unknown_column": UnknownColumnError,
    "unknown_filter": MangoFilterError,
    "bad_filter_value": MangoFilterError,
    "filter_column_missing": MangoFilterError,
    "flag_unavailable": MangoFilterError,
    "bad_time": TimeParseError,
}


def error_from_response(status: int, body: object) -> MangoError:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        code = str(detail.get("error", ""))
        message = str(detail.get("message", ""))
        return _CODE_TO_ERROR.get(code, ServerError)(message)
    return ServerError(f"MANGO server answered HTTP {status}: {body!r}")
