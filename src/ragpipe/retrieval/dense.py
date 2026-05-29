from __future__ import annotations

from typing import Callable

from azure.search.documents.models import VectorizedQuery

from ragpipe.models import Chunk
from ragpipe.retrieval._types import Searchable
from ragpipe.retrieval.bm25 import _to_chunk


class DenseRetriever:
    def __init__(
        self,
        client: Searchable,
        embed_fn: Callable[[str], list[float]],
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._embed = embed_fn
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        vector = self._embed(query)
        vq = VectorizedQuery(
            vector=vector, k_nearest_neighbors=self._top_k, fields="content_vector"
        )
        results = self._client.search(
            search_text=None,
            vector_queries=[vq],
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_chunk(d) for d in results]
