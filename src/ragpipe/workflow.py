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


def build_viz_workflow():
    """Build an Agent Framework Workflow purely for WorkflowViz diagram export.

    Executors are no-op passthroughs; their only purpose is to make the graph
    topology (incl. the conditional loop edge faithfulness->rrf) renderable.
    Runtime behavior lives in run_pipeline().
    """
    from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

    # NOTE: this module uses `from __future__ import annotations`, which turns
    # every annotation into a string at runtime. The Agent Framework `@handler`
    # decorator inspects the live `ctx` annotation and rejects the string
    # "WorkflowContext[str]". We therefore define the no-op handler with real
    # (non-stringified) annotation objects so the decorator's validation passes.
    async def _go(self, msg: str, ctx) -> None:
        await ctx.send_message(msg)

    _go.__annotations__ = {
        "msg": str,
        "ctx": WorkflowContext[str],
        "return": None,
    }

    class _Stage(Executor):
        go = handler(_go)

    # A dispatch start node so BOTH dense and bm25 are reachable from the start
    # (WorkflowBuilder requires every executor be reachable from the start node).
    start = _Stage(id="start")
    dense = _Stage(id="dense")
    bm25 = _Stage(id="bm25")
    rrf = _Stage(id="rrf")
    rerank = _Stage(id="rerank")
    generate = _Stage(id="generate")
    faithfulness = _Stage(id="faithfulness")
    answer = _Stage(id="answer")

    def low_faithfulness(_msg: str) -> bool:
        return True  # label-only; real decision is in run_pipeline()

    builder = WorkflowBuilder(start_executor=start)
    builder.add_edge(start, dense)
    builder.add_edge(start, bm25)
    builder.add_edge(dense, rrf)
    builder.add_edge(bm25, rrf)
    builder.add_edge(rrf, rerank)
    builder.add_edge(rerank, generate)
    builder.add_edge(generate, faithfulness)
    builder.add_edge(faithfulness, rrf, condition=low_faithfulness)
    builder.add_edge(faithfulness, answer)
    return builder.build()
