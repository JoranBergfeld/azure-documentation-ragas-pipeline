from __future__ import annotations

from typing import Awaitable, Callable

from ragpipe.config import Settings
from ragpipe.models import PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def make_deps(settings, retrieve, reranker, generator, scorer) -> PipelineDeps:
    return PipelineDeps(
        retrieve=retrieve,
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        top_k=settings.top_k,
        candidate_pool=settings.candidate_pool,
    )


def build_pipeline_fn(
    settings: Settings,
    mode=None,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
    from functools import lru_cache

    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from agent_framework.foundry import FoundryAgent

    from ragpipe.config import RetrievalMode
    from ragpipe.embeddings import build_embed_fn
    from ragpipe.generate import Generator
    from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
    from ragpipe.retrieval.registry import build_substrate
    from ragpipe.retrieval.rerank import SemanticReranker
    from ragpipe.search_index import SEMANTIC_CONFIG_NAME

    mode = mode or settings.default_mode
    if isinstance(mode, str):
        mode = RetrievalMode(mode)
    cred = DefaultAzureCredential()
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    _clients: dict[str, SearchClient] = {}

    class _Ctx:
        def search_client(self, index: str):
            if index not in _clients:
                _clients[index] = SearchClient(settings.search_endpoint, index, cred)
            return _clients[index]

        def embed(self, text: str):
            return embed(text)

    ctx = _Ctx()
    substrate = build_substrate(mode, settings, ctx)

    # GraphRAG fuses candidates from multiple heterogeneous indexes; Azure's
    # id-filtered semantic reranker can't span them. Use PassthroughReranker
    # (score-sorted truncation) instead.
    if substrate.name == "graphrag":
        from ragpipe.retrieval.passthrough import PassthroughReranker
        reranker = PassthroughReranker(settings.top_k)
    else:
        # The reranker re-scores within whatever the substrate returns, using the
        # substrate's own index for the hybrid stage-1 retrieval.
        rerank_index_attr = {
            "contextual": "search_index",
            "baseline": "baseline_index",
            "raptor_sac": "raptor_sac_index",
        }.get(substrate.name, "search_index")
        reranker = SemanticReranker(
            ctx.search_client(getattr(settings, rerank_index_attr)),
            SEMANTIC_CONFIG_NAME,
            settings.top_k,
            embed_fn=embed,
        )

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        retrieve=substrate.retrieve,
        reranker=reranker,
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str) -> PipelineState:
        return await run_pipeline(query, deps)

    return pipeline_fn
