from __future__ import annotations

from typing import Any, Awaitable, Callable

from ragpipe.config import Settings
from ragpipe.models import PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def make_deps(
    settings: Settings,
    dense: Any,
    bm25: Any,
    reranker: Any,
    generator: Any,
    scorer: Any,
) -> PipelineDeps:
    return PipelineDeps(
        dense=lambda q: dense.retrieve(q),
        bm25=lambda q: bm25.retrieve(q),
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        rrf_k=settings.rrf_k,
        top_k=settings.top_k,
    )


def build_pipeline_fn(
    settings: Settings,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
    from functools import lru_cache

    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from agent_framework.foundry import FoundryAgent

    from ragpipe.embeddings import build_embed_fn
    from ragpipe.generate import Generator
    from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
    from ragpipe.retrieval.bm25 import BM25Retriever
    from ragpipe.retrieval.dense import DenseRetriever
    from ragpipe.retrieval.rerank import SemanticReranker
    from ragpipe.search_index import SEMANTIC_CONFIG_NAME

    cred = DefaultAzureCredential()
    search = SearchClient(settings.search_endpoint, settings.search_index, cred)

    # One embedding per query string, shared by the dense leg and the hybrid
    # rerank call (and across guardrail retries).
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        # Legs fetch the wider candidate pool; rerank narrows to top_k (widened
        # per retry by the workflow).
        dense=DenseRetriever(search, embed, settings.candidate_pool),
        bm25=BM25Retriever(search, settings.candidate_pool),
        reranker=SemanticReranker(
            search, SEMANTIC_CONFIG_NAME, settings.top_k, embed_fn=embed
        ),
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str) -> PipelineState:
        return await run_pipeline(query, deps)

    return pipeline_fn
