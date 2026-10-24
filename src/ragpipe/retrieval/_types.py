from __future__ import annotations

from typing import Any, Protocol


class Table(Protocol):
    """Minimal interface of ragpipe.pgstore.PgTable used by retrievers. Rows are
    dicts of the table's columns plus ``score`` (higher is better)."""

    def nearest(self, vector: list[float], k: int) -> list[dict[str, Any]]: ...

    def bm25(self, query: str, k: int) -> list[dict[str, Any]]: ...

    def scan(self) -> list[dict[str, Any]]: ...
