"""Base class of the per-region objects (mango.magnetosheath, ...)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

if TYPE_CHECKING:
    from space_mango.client import MangoClient

_FIXED = (
    "columns",
    "spacecraft",
    "start",
    "stop",
    "sw_paired_only",
    "normalized_only",
    "limit",
    "cache",
)


class RegionAPI:
    name: ClassVar[str]
    definition: ClassVar[str]

    def __init__(self, client_factory: Callable[[], MangoClient]) -> None:
        self._client_factory = client_factory

    def __repr__(self) -> str:
        return f"<MANGO region '{self.name}': {self.definition}>"

    def describe(self) -> pl.DataFrame:
        return self._client_factory().describe(self.name)

    def spacecraft(self) -> pl.DataFrame:
        return self._client_factory().spacecraft(self.name)

    @property
    def filters(self) -> pl.DataFrame:
        return pl.DataFrame(self._client_factory().filters(self.name)).drop("params")

    def _call(self, method: str, args: dict[str, object]) -> Any:
        args = {k: v for k, v in args.items() if k != "self"}
        fixed = {k: args.pop(k) for k in _FIXED if k in args}
        filters = {k: v for k, v in args.items() if v is not None}
        return getattr(self._client_factory(), method)(self.name, **fixed, **filters)
