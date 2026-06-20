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
    *,
    mode,
) -> Callable[..., Awaitable[PipelineState]]:  # pragma: no cover - live wiring
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

    complete_fn = None  # built lazily inside plan()

    class _Ctx:
        def search_client(self, index: str):
            if index not in _clients:
                _clients[index] = SearchClient(settings.search_endpoint, index, cred)
            return _clients[index]

        def embed(self, text: str):
            return embed(text)

        def plan(self, query: str) -> list[str]:  # pragma: no cover
            import sys

            from ragpipe.context_gen import build_context_complete_fn

            nonlocal complete_fn
            if complete_fn is None:
                complete_fn = build_context_complete_fn(settings)
            PLAN_PROMPT = (
                "Decompose the following question into 2-4 focused search sub-queries "
                "that together cover what the user needs. Output one sub-query per line "
                "with no numbering, bullets, or preamble.\n\nQuestion: {query}"
            )
            prompt = PLAN_PROMPT.format(query=query)
            for attempt in range(settings.max_retries + 1):
                try:
                    raw = complete_fn(prompt)
                    lines = [line.strip() for line in raw.splitlines() if line.strip()]
                    if lines:
                        return lines
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"plan attempt {attempt}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            return []

    ctx = _Ctx()
    substrate = build_substrate(mode, settings, ctx)

    def _uses_passthrough(name: str) -> bool:
        return name in {"graphrag", "combined"} or name.endswith("_agentic")

    # GraphRAG and combined fuse candidates from multiple heterogeneous indexes;
    # Azure's id-filtered semantic reranker can't span them. Agentic substrates
    # wrap these same modes. All use PassthroughReranker (score-sorted truncation).
    if _uses_passthrough(substrate.name):
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

    async def pipeline_fn(query: str, *, on_event=None) -> PipelineState:
        return await run_pipeline(query, deps, on_event=on_event)

    return pipeline_fn
