from __future__ import annotations

from ragpipe.models import Chunk


class PassthroughReranker:
    """Rerank by existing candidate score, no external call. Used for substrates
    whose candidates span multiple indexes and can't use Azure's id-filtered
    semantic ranker (e.g. GraphRAG)."""

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def rerank(self, query: str, fused: list[Chunk], top_k: int | None = None) -> list[Chunk]:
        k = top_k or self._top_k
        return sorted(fused, key=lambda c: c.score, reverse=True)[:k]
