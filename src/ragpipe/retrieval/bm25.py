from __future__ import annotations

from typing import Any, Protocol

from ragpipe.models import Chunk


class _Searchable(Protocol):
    def search(self, search_text: str | None = None, **kwargs: Any): ...


def _to_chunk(doc: dict[str, Any]) -> Chunk:
    return Chunk(
        id=doc["id"],
        title=doc.get("title", ""),
        url=doc.get("url", ""),
        content=doc.get("content", ""),
        score=float(doc.get("@search.score", 0.0)),
    )


class BM25Retriever:
    def __init__(self, client: _Searchable, top_k: int = 5) -> None:
        self._client = client
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        results = self._client.search(
            search_text=query,
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_chunk(d) for d in results]
