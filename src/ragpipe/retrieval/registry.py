from __future__ import annotations

from typing import Callable, Protocol

from ragpipe.config import RetrievalMode, Settings
from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever
from ragpipe.retrieval.substrate import HybridSubstrate, RetrievalSubstrate


class SubstrateCtx(Protocol):
    def search_client(self, index: str): ...
    def embed(self, text: str) -> list[float]: ...
    def plan(self, query: str) -> list[str]: ...


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


# The four *_agentic wrapper modes have no committed eval coverage (no
# eval_results_*_agentic.json) and wrap a single-shot bounded planner fan-out
# (ADR-0015), so they are surfaced as experimental / unevaluated (ADR-0021).
_EXPERIMENTAL_MODES: frozenset[RetrievalMode] = frozenset(
    {
        RetrievalMode.BASELINE_AGENTIC,
        RetrievalMode.RAPTOR_SAC_AGENTIC,
        RetrievalMode.GRAPHRAG_AGENTIC,
        RetrievalMode.COMBINED_AGENTIC,
    }
)


def experimental_modes() -> list[RetrievalMode]:
    """Registered modes with no committed eval coverage (ADR-0021)."""
    return [m for m in _REGISTRY if m in _EXPERIMENTAL_MODES]


def is_experimental(mode: RetrievalMode | str) -> bool:
    """True for unevaluated/experimental modes (ADR-0021). Accepts a mode or its .value."""
    return RetrievalMode(mode) in _EXPERIMENTAL_MODES


def build_substrate(mode: RetrievalMode, settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    if mode not in _REGISTRY:
        raise ValueError(f"mode {mode.value!r} is not registered yet (phase not built)")
    return _REGISTRY[mode](settings, ctx)


def _combined(settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    raptor = build_substrate(RetrievalMode.RAPTOR_SAC, settings, ctx)
    graph = build_substrate(RetrievalMode.GRAPHRAG, settings, ctx)
    from ragpipe.retrieval.combined import CombinedSubstrate
    return CombinedSubstrate(name="combined", substrates=[raptor, graph], rrf_k=settings.rrf_k)


_REGISTRY[RetrievalMode.COMBINED] = _combined


def _agentic(base_mode: RetrievalMode, name: str):
    def factory(settings, ctx):
        inner = build_substrate(base_mode, settings, ctx)
        from ragpipe.retrieval.agentic import AgenticSubstrate
        return AgenticSubstrate(
            name=name, inner=inner, plan_fn=ctx.plan,
            max_iterations=settings.agentic_max_iterations,
        )
    return factory


_REGISTRY[RetrievalMode.BASELINE_AGENTIC] = _agentic(RetrievalMode.BASELINE, "baseline_agentic")
_REGISTRY[RetrievalMode.RAPTOR_SAC_AGENTIC] = _agentic(RetrievalMode.RAPTOR_SAC, "raptor_sac_agentic")
_REGISTRY[RetrievalMode.GRAPHRAG_AGENTIC] = _agentic(RetrievalMode.GRAPHRAG, "graphrag_agentic")
_REGISTRY[RetrievalMode.COMBINED_AGENTIC] = _agentic(RetrievalMode.COMBINED, "combined_agentic")
