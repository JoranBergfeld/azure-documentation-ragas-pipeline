from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ragpipe.guardrail import LoopDecision, decide_next
from ragpipe.models import Chunk, PipelineState
from ragpipe.retrieval.rrf import reciprocal_rank_fusion

# Callable stage signatures (sync or async tolerated via _maybe_await).
DenseFn = Callable[[str], list[Chunk]]
Bm25Fn = Callable[[str], list[Chunk]]
RerankFn = Callable[[str, list[Chunk]], list[Chunk]]
GenerateFn = Callable[[str, list[Chunk]], object]
ScoreFn = Callable[[str, str, list[Chunk]], object]


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass
class PipelineDeps:
    dense: DenseFn
    bm25: Bm25Fn
    rerank: RerankFn
    generate: GenerateFn
    score: ScoreFn
    threshold: float = 0.7
    max_retries: int = 2
    rrf_k: int = 60
    top_k: int = 5


async def run_pipeline(query: str, deps: PipelineDeps) -> PipelineState:
    state = PipelineState(query=query)

    state.dense = await _maybe_await(deps.dense(query))
    state.add_trace("dense", {"ids": [c.id for c in state.dense]})
    state.bm25 = await _maybe_await(deps.bm25(query))
    state.add_trace("bm25", {"ids": [c.id for c in state.bm25]})

    while True:
        state.fused = reciprocal_rank_fusion(state.dense, state.bm25, k=deps.rrf_k)
        state.add_trace("rrf", {"ids": [c.id for c in state.fused]})

        state.reranked = await _maybe_await(deps.rerank(query, state.fused))
        state.add_trace("rerank", {"ids": [c.id for c in state.reranked]})

        state.answer = await _maybe_await(deps.generate(query, state.reranked))
        state.add_trace("generate", {"answer": state.answer})

        try:
            score = await _maybe_await(deps.score(query, state.answer, state.reranked))
        except Exception:  # judge failure -> fail-closed
            score = None
        state.faithfulness = score
        state.add_trace("faithfulness", {"score": score, "attempt": state.attempt})

        decision = decide_next(
            score=score,
            threshold=deps.threshold,
            attempt=state.attempt,
            max_retries=deps.max_retries,
        )
        if decision is LoopDecision.PASS:
            return state
        if decision is LoopDecision.EXHAUSTED:
            state.low_confidence = True
            return state
        state.next_attempt()  # RETRY
