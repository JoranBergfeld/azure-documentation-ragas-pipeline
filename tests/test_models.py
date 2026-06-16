from __future__ import annotations

from ragpipe.models import Chunk, PipelineState, TraceEvent


def test_chunk_holds_identity_and_score():
    c = Chunk(id="doc1#0", title="T", url="http://x", content="body", score=1.5)
    assert c.id == "doc1#0"
    assert c.score == 1.5


def test_pipeline_state_records_trace_in_order():
    state = PipelineState(query="what is RRF?")
    state.add_trace("dense", {"hits": 3})
    state.add_trace("bm25", {"hits": 2})

    assert [e.stage for e in state.trace] == ["dense", "bm25"]
    assert isinstance(state.trace[0], TraceEvent)
    assert state.trace[0].data == {"hits": 3}


def test_pipeline_state_attempt_increments():
    state = PipelineState(query="q")
    assert state.attempt == 0
    state.next_attempt()
    assert state.attempt == 1


def test_set_stage_records_and_orders():
    s = PipelineState(query="q")
    s.set_stage("dense", [Chunk(id="1", title="", url="", content="")])
    s.set_stage("bm25", [Chunk(id="2", title="", url="", content="")])
    assert list(s.stages.keys()) == ["dense", "bm25"]
    assert [c.id for c in s.stages["dense"]] == ["1"]


def test_set_reranked_mirrors_into_stages():
    s = PipelineState(query="q")
    s.set_reranked([Chunk(id="9", title="", url="", content="")])
    assert [c.id for c in s.reranked] == ["9"]
    assert [c.id for c in s.stages["reranked"]] == ["9"]
