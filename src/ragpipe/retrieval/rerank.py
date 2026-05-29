from __future__ import annotations

from typing import Any

from ragpipe.models import Chunk
from ragpipe.retrieval._types import Searchable


def _to_reranked_chunk(doc: dict[str, Any]) -> Chunk:
    return Chunk(
        id=doc["id"],
        title=doc.get("title", ""),
        url=doc.get("url", ""),
        content=doc.get("content", ""),
        score=float(doc.get("@search.rerankerScore", 0.0)),
    )


def _quote_ids(ids: list[str]) -> str:
    # OData search.in filter: search.in(id, 'a,b,c', ',')
    joined = ",".join(ids)
    return f"search.in(id, '{joined}', ',')"


class SemanticReranker:
    def __init__(
        self, client: Searchable, semantic_config: str, top_k: int = 5
    ) -> None:
        self._client = client
        self._semantic_config = semantic_config
        self._top_k = top_k

    def rerank(self, query: str, fused: list[Chunk]) -> list[Chunk]:
        if not fused:
            return []
        ids = [c.id for c in fused]
        results = self._client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
            filter=_quote_ids(ids),
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_reranked_chunk(d) for d in results]
