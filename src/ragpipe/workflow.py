from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from ragpipe.guardrail import LoopDecision, decide_next
from ragpipe.models import Chunk, PipelineState
from ragpipe.progress import ProgressSink, emit
from ragpipe.retrieval.substrate import RetrievalResult

# Callable stage signatures (sync or async tolerated via _maybe_await).
RetrieveFn = Callable[..., object]  # (query, pool, *, on_event=None) -> awaitable RetrievalResult
RerankFn = Callable[[str, list[Chunk], int], list[Chunk]]
GenerateFn = Callable[[str, list[Chunk], str | None], object]
ScoreFn = Callable[[str, str, list[Chunk]], object]

# Returned verbatim when the guardrail exhausts: the directive abstention
# (ADR-0009). Consumers get this text instead of the unfaithful answer.
ABSTENTION_ANSWER = (
    "I don't have enough grounded information in the indexed documentation "
    "to answer this question reliably."
)


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass
class PipelineDeps:
    retrieve: RetrieveFn
    rerank: RerankFn
    generate: GenerateFn
    score: ScoreFn
    threshold: float = 0.7
    max_retries: int = 2
    top_k: int = 5
    candidate_pool: int = 15
    # Each retry widens the rerank window by this many chunks: the most common
    # faithfulness failure is the needed chunk sitting just below the cut.
    rerank_widen_step: int = 3


async def run_pipeline(
    query: str, deps: PipelineDeps, *, on_event: ProgressSink | None = None
) -> PipelineState:
    state = PipelineState(query=query)

    emit(on_event, "retrieve", "start", message="Retrieving candidates")
    result: RetrievalResult = await _maybe_await(
        deps.retrieve(query, deps.candidate_pool, on_event=on_event)
    )
    for name, chunks in result.stages.items():
        state.set_stage(name, chunks)
        state.add_trace(name, {"ids": [c.id for c in chunks]})
    state.candidates = result.candidates
    emit(
        on_event,
        "retrieve",
        "complete",
        message=f"Retrieved {len(result.candidates)} candidates",
        stages=list(result.stages),
        candidates=len(result.candidates),
    )

    previous_answer: str | None = None
    while True:
        k = deps.top_k + deps.rerank_widen_step * state.attempt
        emit(
            on_event,
            "rerank",
            "start",
            attempt=state.attempt,
            message=f"Reranking (attempt {state.attempt + 1}, window {k})",
            k=k,
        )
        reranked = await _maybe_await(deps.rerank(query, state.candidates, k))
        state.set_reranked(reranked)
        state.add_trace("rerank", {"ids": [c.id for c in state.reranked], "top_k": k})
        emit(
            on_event,
            "rerank",
            "complete",
            attempt=state.attempt,
            message=f"Reranked to top {len(state.reranked)}",
            k=k,
            n=len(state.reranked),
        )

        emit(
            on_event,
            "generate",
            "start",
            attempt=state.attempt,
            message=f"Generating answer (attempt {state.attempt + 1})",
        )
        state.answer = await _maybe_await(
            deps.generate(query, state.reranked, previous_answer)
        )
        state.add_trace("generate", {"answer": state.answer})
        previous_answer = state.answer
        emit(
            on_event,
            "generate",
            "complete",
            attempt=state.attempt,
            message="Answer generated",
        )

        emit(
            on_event,
            "faithfulness",
            "start",
            attempt=state.attempt,
            message=f"Scoring faithfulness (attempt {state.attempt + 1})",
        )
        try:
            score = await _maybe_await(deps.score(query, state.answer, state.reranked))
        except Exception as exc:  # judge failure -> fail-closed
            # Logged so operators can tell an outage from a scorer bug; the
            # decision path is identical either way (abstain immediately).
            print(
                f"guardrail: scorer failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            score = None
            emit(
                on_event,
                "faithfulness",
                "error",
                attempt=state.attempt,
                message="Faithfulness judge failed",
                error=type(exc).__name__,
            )
        else:
            emit(
                on_event,
                "faithfulness",
                "complete",
                attempt=state.attempt,
                message=(f"Faithfulness {score:.2f}" if score is not None else "Faithfulness n/a"),
                score=score,
                threshold=deps.threshold,
            )
        state.faithfulness = score
        state.add_trace("faithfulness", {"score": score, "attempt": state.attempt})

        decision = decide_next(
            score=score,
            threshold=deps.threshold,
            attempt=state.attempt,
            max_retries=deps.max_retries,
        )
        emit(
            on_event,
            "decision",
            "complete",
            attempt=state.attempt,
            message=f"Decision: {decision.name.lower()}",
            decision=decision.name.lower(),
            score=score,
            threshold=deps.threshold,
        )
        if decision is LoopDecision.PASS:
            return state
        if decision is LoopDecision.EXHAUSTED:
            state.low_confidence = True
            state.abstained = True
            state.add_trace(
                "abstain", {"suppressed_answer": state.answer, "score": score}
            )
            emit(
                on_event,
                "abstain",
                "complete",
                attempt=state.attempt,
                message="Abstained: not enough grounded context",
                suppressed_answer=state.answer,
                score=score,
            )
            state.answer = ABSTENTION_ANSWER
            return state
        state.next_attempt()  # RETRY


def build_viz_workflow():
    """Build an Agent Framework Workflow purely for WorkflowViz diagram export.

    Executors are no-op passthroughs; their only purpose is to make the graph
    topology (incl. the conditional loop edge faithfulness->rerank) renderable.
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

    # A single retrieval node stands in for the pluggable substrate (ADR-0012).
    start = _Stage(id="start")
    retrieve = _Stage(id="retrieve")
    rerank = _Stage(id="rerank")
    generate = _Stage(id="generate")
    faithfulness = _Stage(id="faithfulness")
    answer = _Stage(id="answer")

    def low_faithfulness(_msg: str) -> bool:
        return True  # label-only; real decision is in run_pipeline()

    builder = WorkflowBuilder(start_executor=start)
    builder.add_edge(start, retrieve)
    builder.add_edge(retrieve, rerank)
    builder.add_edge(rerank, generate)
    builder.add_edge(generate, faithfulness)
    # Retries re-enter at rerank (widened window over the fixed candidate set),
    # never re-running retrieval — retrieval is one substrate call per query
    # (ADR-0009/0012).
    builder.add_edge(faithfulness, rerank, condition=low_faithfulness)
    builder.add_edge(faithfulness, answer)
    return builder.build()
