from __future__ import annotations

from typing import Callable, Protocol

from ragpipe.config import RetrievalMode, Settings
from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever
from ragpipe.retrieval.substrate import HybridSubstrate, RetrievalSubstrate


class SubstrateCtx(Protocol):
    def search_client(self, index: str): ...
    def embed(self, text: str) -> list[float]: ...


def _hybrid(index_attr: str, name: str):
    def factory(settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
        index = getattr(settings, index_attr)
        client = ctx.search_client(index)
        return HybridSubstrate(
            name=name,
            dense=DenseRetriever(client, ctx.embed, settings.candidate_pool),
            bm25=BM25Retriever(client, settings.candidate_pool),
            rrf_k=settings.rrf_k,
        )
    return factory


_REGISTRY: dict[RetrievalMode, Callable[[Settings, SubstrateCtx], RetrievalSubstrate]] = {
    RetrievalMode.CONTEXTUAL: _hybrid("search_index", "contextual"),
    RetrievalMode.BASELINE: _hybrid("baseline_index", "baseline"),
}


def registered_modes() -> list[RetrievalMode]:
    return list(_REGISTRY)


def build_substrate(mode: RetrievalMode, settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    if mode not in _REGISTRY:
        raise ValueError(f"mode {mode.value!r} is not registered yet (phase not built)")
    return _REGISTRY[mode](settings, ctx)
