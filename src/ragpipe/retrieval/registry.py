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


def _graphrag(settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    from ragpipe.retrieval.graph_substrate import (
        GraphRAGSubstrate,
        build_adjacency,
        build_community_search,
        build_entity_search,
    )

    ent = build_entity_search(ctx.search_client(settings.graph_entities_index), ctx.embed, settings.candidate_pool)
    com = build_community_search(ctx.search_client(settings.graph_communities_index), ctx.embed, settings.candidate_pool)
    adjacency = build_adjacency(ctx.search_client(settings.graph_relationships_index))
    return GraphRAGSubstrate(
        name="graphrag",
        entity_search=ent,
        community_search=com,
        adjacency=adjacency,
        rrf_k=settings.rrf_k,
    )


_REGISTRY: dict[RetrievalMode, Callable[[Settings, SubstrateCtx], RetrievalSubstrate]] = {
    RetrievalMode.CONTEXTUAL: _hybrid("search_index", "contextual"),
    RetrievalMode.BASELINE: _hybrid("baseline_index", "baseline"),
    RetrievalMode.RAPTOR_SAC: _hybrid("raptor_sac_index", "raptor_sac"),
    RetrievalMode.GRAPHRAG: _graphrag,
}


def registered_modes() -> list[RetrievalMode]:
    return list(_REGISTRY)


def build_substrate(mode: RetrievalMode, settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    if mode not in _REGISTRY:
        raise ValueError(f"mode {mode.value!r} is not registered yet (phase not built)")
    return _REGISTRY[mode](settings, ctx)
