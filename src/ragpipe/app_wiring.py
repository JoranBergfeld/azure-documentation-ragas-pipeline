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
        rerank=lambda q, fused: reranker.rerank(q, fused),
        generate=lambda q, chunks: generator.generate(q, chunks),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        rrf_k=settings.rrf_k,
        top_k=settings.top_k,
    )


def build_pipeline_fn(
    settings: Settings,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
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
    # Synchronous embed callable over the Azure OpenAI endpoint (Entra auth). The
    # underlying openai client is sync, so it is safe to call from inside the
    # running run_pipeline event loop (no nested-loop bridging needed).
    embed = build_embed_fn(settings)

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        dense=DenseRetriever(search, embed, settings.top_k),
        bm25=BM25Retriever(search, settings.top_k),
        reranker=SemanticReranker(search, SEMANTIC_CONFIG_NAME, settings.top_k),
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str) -> PipelineState:
        return await run_pipeline(query, deps)

    return pipeline_fn
