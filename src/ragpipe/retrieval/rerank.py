from __future__ import annotations

from typing import Any, Callable

from azure.search.documents.models import VectorizedQuery

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
        self,
        client: Searchable,
        semantic_config: str,
        top_k: int = 5,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._client = client
        self._semantic_config = semantic_config
        self._top_k = top_k
        self._embed = embed_fn

    def rerank(
        self, query: str, fused: list[Chunk], top_k: int | None = None
    ) -> list[Chunk]:
        if not fused:
            return []
        k = top_k or self._top_k
        ids = [c.id for c in fused]
        # Semantic ranking is two-stage: stage 1 retrieves candidates, stage 2
        # re-scores them. With search_text alone, stage 1 is BM25 — a fused
        # candidate with zero lexical overlap (dense-only) never matches and is
        # silently dropped despite passing the id filter. Adding the vector leg
        # makes stage 1 hybrid, so every fused candidate is reachable.
        vector_queries = None
        if self._embed is not None:
            vector_queries = [
                VectorizedQuery(
                    vector=self._embed(query),
                    k=len(ids),
                    fields="content_vector",
                )
            ]
        results = self._client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
            filter=_quote_ids(ids),
            vector_queries=vector_queries,
            top=k,
            select=["id", "title", "url", "content"],
        )
        return [_to_reranked_chunk(d) for d in results][:k]
