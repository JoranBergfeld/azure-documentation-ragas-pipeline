from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import RetrievalResult
from ragpipe.progress import ProgressSink, emit


class AgenticSubstrate:
    """Wrap any substrate with a bounded plan->retrieve loop. `plan_fn` decomposes
    the query into sub-queries; we run inner.retrieve over the first
    `max_iterations` of them, accumulate+dedupe candidates by id (keeping the max
    score), and record each iteration as a stage. The faithfulness gate downstream
    stays the final arbiter; this only amplifies retrieval."""

    def __init__(self, *, name, inner, plan_fn: Callable[[str], list[str]], max_iterations: int = 3):
        self.name = name
        self._inner = inner
        self._plan_fn = plan_fn
        self._max_iterations = max_iterations

    async def retrieve(self, query: str, k: int, on_event: ProgressSink | None = None) -> RetrievalResult:
        subqueries = self._plan_fn(query)[: self._max_iterations] or [query]
        n = len(subqueries)
        emit(on_event, "retrieve.plan", "complete",
             message=f"Planned {n} sub-quer{'y' if n == 1 else 'ies'}", total=n)
        accumulated: dict[str, Chunk] = {}
        stages: dict = {}
        for i, sq in enumerate(subqueries):
            emit(on_event, "retrieve.iter", "start",
                 message=f"Retrieving sub-query {i + 1}/{n}",
                 index=i, total=n, subquery=sq)
            result = await self._inner.retrieve(sq, k)
            stages[f"iter_{i}"] = result.candidates
            for c in result.candidates:
                prev = accumulated.get(c.id)
                if prev is None or c.score > prev.score:
                    accumulated[c.id] = c
            emit(on_event, "retrieve.iter", "complete",
                 message=f"Sub-query {i + 1}/{n} → {len(result.candidates)} candidates",
                 index=i, total=n, candidates=len(result.candidates))
        candidates = sorted(accumulated.values(), key=lambda c: c.score, reverse=True)
        stages["fused"] = candidates
        emit(on_event, "retrieve.fuse", "complete",
             message=f"Fused to {len(candidates)} unique candidates", fused=len(candidates))
        return RetrievalResult(candidates=candidates, stages=stages)
