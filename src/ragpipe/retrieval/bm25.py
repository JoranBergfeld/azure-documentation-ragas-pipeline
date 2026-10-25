from __future__ import annotations

from typing import Any

from ragpipe.models import Chunk
from ragpipe.retrieval._types import Table


def _to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        title=row.get("title", ""),
        url=row.get("url", ""),
        content=row.get("content", ""),
        score=float(row.get("score", 0.0)),
        context=row.get("context") or "",
    )


class BM25Retriever:
    def __init__(self, table: Table, top_k: int = 5) -> None:
        self._table = table
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        return [_to_chunk(r) for r in self._table.bm25(query, self._top_k)]
