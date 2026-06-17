from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import RetrievalResult


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

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        subqueries = self._plan_fn(query)[: self._max_iterations] or [query]
        accumulated: dict[str, Chunk] = {}
        stages: dict = {}
        for i, sq in enumerate(subqueries):
            result = await self._inner.retrieve(sq, k)
            stages[f"iter_{i}"] = result.candidates
            for c in result.candidates:
                prev = accumulated.get(c.id)
                if prev is None or c.score > prev.score:
                    accumulated[c.id] = c
        candidates = sorted(accumulated.values(), key=lambda c: c.score, reverse=True)
        stages["fused"] = candidates
        return RetrievalResult(candidates=candidates, stages=stages)
