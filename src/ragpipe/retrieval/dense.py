from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval._types import Table
from ragpipe.retrieval.bm25 import _to_chunk


class DenseRetriever:
    def __init__(
        self,
        table: Table,
        embed_fn: Callable[[str], list[float]],
        top_k: int = 5,
    ) -> None:
        self._table = table
        self._embed = embed_fn
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        return [_to_chunk(r) for r in self._table.nearest(self._embed(query), self._top_k)]
