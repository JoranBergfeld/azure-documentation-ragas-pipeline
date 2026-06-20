from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.agentic import AgenticSubstrate
from ragpipe.retrieval.substrate import RetrievalResult


class _RecordingSub:
    name = "baseline"

    def __init__(self):
        self.queries = []

    async def retrieve(self, query, k):
        self.queries.append(query)
        return RetrievalResult(
            candidates=[Chunk(id=query, title="", url="", content="", score=1.0)],
            stages={"fused": []},
        )


@pytest.mark.asyncio
async def test_agentic_loops_planned_subqueries_bounded_and_dedupes():
    inner = _RecordingSub()
    sub = AgenticSubstrate(
        name="baseline_agentic", inner=inner,
        plan_fn=lambda q: ["sub a", "sub b", "sub c", "sub d"],
        max_iterations=2,
    )
    result = await sub.retrieve("original", k=5)
    assert sub.name == "baseline_agentic"
    assert inner.queries == ["sub a", "sub b"]
    ids = {c.id for c in result.candidates}
    assert ids == {"sub a", "sub b"}
    assert "iter_0" in result.stages and "iter_1" in result.stages
    assert "fused" in result.stages


@pytest.mark.asyncio
async def test_agentic_falls_back_to_original_query_when_plan_empty():
    inner = _RecordingSub()
    sub = AgenticSubstrate(name="x_agentic", inner=inner, plan_fn=lambda q: [], max_iterations=3)
    await sub.retrieve("original", k=5)
    assert inner.queries == ["original"]


@pytest.mark.asyncio
async def test_agentic_emits_plan_iter_fuse_events():
    inner = _RecordingSub()
    events: list = []
    sub = AgenticSubstrate(
        name="baseline_agentic", inner=inner,
        plan_fn=lambda q: ["sub a", "sub b"], max_iterations=3,
    )
    await sub.retrieve("original", k=5, on_event=events.append)

    phases = [(e.phase, e.status) for e in events]
    assert ("retrieve.plan", "complete") in phases
    iters = [e for e in events if e.phase == "retrieve.iter" and e.status == "complete"]
    assert [e.detail["index"] for e in iters] == [0, 1]
    assert all(e.detail["total"] == 2 for e in iters)
    assert ("retrieve.fuse", "complete") in phases


@pytest.mark.asyncio
async def test_agentic_unchanged_when_on_event_none():
    inner = _RecordingSub()
    sub = AgenticSubstrate(name="x_agentic", inner=inner, plan_fn=lambda q: ["a"], max_iterations=2)
    result = await sub.retrieve("original", k=5)  # no sink
    assert "iter_0" in result.stages and "fused" in result.stages
