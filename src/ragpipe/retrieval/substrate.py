from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    """What a substrate returns: the final candidate list fed to rerank, plus
    named intermediate stages captured for the dashboard and eval (e.g. dense,
    bm25, fused). The substrate owns its own fusion; the pipeline does not."""

    candidates: list[Chunk]
    stages: dict[str, list[Chunk]] = field(default_factory=dict)


@runtime_checkable
class RetrievalSubstrate(Protocol):
    name: str

    async def retrieve(self, query: str, k: int) -> RetrievalResult: ...


class HybridSubstrate:
    """Dense + BM25 hybrid with RRF fusion -- the original pipeline topology,
    now owned by the substrate. `dense` and `bm25` are objects with a sync
    `.retrieve(query) -> list[Chunk]` (DenseRetriever / BM25Retriever)."""

    def __init__(self, *, name: str, dense, bm25, rrf_k: int = 60) -> None:
        self.name = name
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        dense = self._dense.retrieve(query)
        bm25 = self._bm25.retrieve(query)
        fused = reciprocal_rank_fusion(dense, bm25, k=self._rrf_k)
        return RetrievalResult(
            candidates=fused,
            stages={"dense": dense, "bm25": bm25, "fused": fused},
        )
