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
