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
from ragpipe.model_roles import ModelRole
from ragpipe.models import PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


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

    config = RerankerConfig.from_endpoint(settings.endpoint(ModelRole.RERANKER))
    if config is None:
        raise ModeUnavailable(mode, RERANKER_OFFLINE)
    return CrossEncoderReranker(
        config, mode, top_k=settings.top_k, with_context=_RERANK_WITH_CONTEXT[mode]
    )


def make_deps(settings, retrieve, reranker, generator, gate) -> PipelineDeps:
    return PipelineDeps(
        retrieve=retrieve,
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        # The judge reads what the generator read: chunk content, never the
        # retrieval-only situating context (ADR-0003).
        score=lambda q, a, chunks: gate.score(q, a, [c.content for c in chunks]),
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

    from ragpipe.embeddings import build_embed_fn
    from ragpipe.generate import Generator
    from ragpipe.faithfulness_gate import build_faithfulness_gate
    from ragpipe.model_roles import complete_fn, generator_agent
    from ragpipe.pgstore import PgTable
    from ragpipe.retrieval.registry import build_substrate

    mode = RetrievalMode(mode)
    reason = unavailable_reason(mode)
    if reason is not None:
        raise ModeUnavailable(mode, reason)
    if not settings.rag_database_url:
        raise ValueError("Required environment variable RAG_DATABASE_URL is not set")
    pool = _table_pool(settings.rag_database_url)
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    plan_complete = None  # built lazily inside plan()

    class _Ctx:
        def table(self, spec):
            return PgTable(pool, spec)

        def embed(self, text: str):
            return embed(text)

        def plan(self, query: str) -> list[str]:  # pragma: no cover
            import sys

            nonlocal plan_complete
            if plan_complete is None:
                plan_complete = complete_fn(settings, ModelRole.PLANNER)
            PLAN_PROMPT = (
                "Decompose the following question into 2-4 focused search sub-queries "
                "that together cover what the user needs. Output one sub-query per line "
                "with no numbering, bullets, or preamble.\n\nQuestion: {query}"
            )
            prompt = PLAN_PROMPT.format(query=query)
            for attempt in range(settings.max_retries + 1):
                try:
                    raw = plan_complete(prompt)
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

    deps = make_deps(
        settings,
        retrieve=substrate.retrieve,
        reranker=reranker,
        # Reported with each generation's usage: on Foundry, the deployment
        # setup_agents.py registered the agent against.
        generator=Generator(
            generator_agent(settings), deployment=settings.endpoint(ModelRole.GENERATOR).model
        ),
        gate=build_faithfulness_gate(settings),
    )

    async def pipeline_fn(query: str, *, on_event=None) -> PipelineState:
        return await run_pipeline(query, deps, on_event=on_event)

    return pipeline_fn
