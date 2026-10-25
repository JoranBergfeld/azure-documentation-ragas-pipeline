from __future__ import annotations

from typing import Awaitable, Callable

from ragpipe.config import (
    RERANK_MODES,
    RERANKER_OFFLINE,
    ModeUnavailable,
    RetrievalMode,
    Settings,
    unavailable_reason,
)
from ragpipe.models import PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def _score_callable(scorer):
    """Prefer the scorer's per-claim path so the trace carries verdicts (ADR-0018);
    a scalar-only scorer still works."""
    detailed = getattr(scorer, "score_detailed", None)
    if detailed is not None:
        return lambda q, a, c: detailed(q, a, c)
    return lambda q, a, c: scorer.score(q, a, c)


# Which of the old semantic configs' content fields each rerank mode reads (issue
# #40): ms_docs and raptor_sac had title + content + context, baseline had title +
# content. Keeps the cross-encoder swap like for like.
_RERANK_WITH_CONTEXT: dict[RetrievalMode, bool] = {
    RetrievalMode.CONTEXTUAL: True,
    RetrievalMode.BASELINE: False,
    RetrievalMode.RAPTOR_SAC: True,
}


def build_reranker(settings: Settings, mode: RetrievalMode):
    """The cross-encoder for the rerank modes, passthrough for the rest."""
    from ragpipe.retrieval.passthrough import PassthroughReranker

    if mode not in RERANK_MODES:
        # graphrag and combined fuse heterogeneous tables no single cross-encoder
        # pass over one table spans; the agentic wrappers keep the ranking they
        # were evaluated with.
        return PassthroughReranker(settings.top_k)

    from ragpipe.retrieval.cross_encoder import CrossEncoderReranker, RerankerConfig

    config = RerankerConfig.from_env()
    if config is None:
        raise ModeUnavailable(mode, RERANKER_OFFLINE)
    return CrossEncoderReranker(
        config, mode, top_k=settings.top_k, with_context=_RERANK_WITH_CONTEXT[mode]
    )


def make_deps(settings, retrieve, reranker, generator, scorer) -> PipelineDeps:
    return PipelineDeps(
        retrieve=retrieve,
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        score=_score_callable(scorer),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        top_k=settings.top_k,
        candidate_pool=settings.candidate_pool,
    )


_pools: dict[str, object] = {}


def _table_pool(database_url: str):  # pragma: no cover - needs a database
    """One read-only pool per database, shared by every mode's pipeline."""
    from ragpipe.pgstore import open_pool

    if database_url not in _pools:
        _pools[database_url] = open_pool(database_url)
    return _pools[database_url]


def build_pipeline_fn(
    settings: Settings,
    *,
    mode,
) -> Callable[..., Awaitable[PipelineState]]:  # pragma: no cover - live wiring
    from functools import lru_cache

    from azure.identity import DefaultAzureCredential
    from agent_framework.foundry import FoundryAgent

    from ragpipe.embeddings import build_embed_fn
    from ragpipe.generate import Generator
    from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
    from ragpipe.pgstore import PgTable
    from ragpipe.retrieval.registry import build_substrate

    mode = RetrievalMode(mode)
    reason = unavailable_reason(mode)
    if reason is not None:
        raise ModeUnavailable(mode, reason)
    if not settings.rag_database_url:
        raise ValueError("Required environment variable RAG_DATABASE_URL is not set")
    pool = _table_pool(settings.rag_database_url)
    # Generation and embeddings only: retrieval no longer touches Azure (ADR-0021).
    cred = DefaultAzureCredential()
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    complete_fn = None  # built lazily inside plan()

    class _Ctx:
        def table(self, spec):
            return PgTable(pool, spec)

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

    reranker = build_reranker(settings, mode)

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
        # setup_agents.py registers the agent against FOUNDRY_CHAT_MODEL.
        generator=Generator(agent, deployment=settings.foundry_chat_model),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str, *, on_event=None) -> PipelineState:
        return await run_pipeline(query, deps, on_event=on_event)

    return pipeline_fn
