from ragpipe.models import Chunk, PipelineState
from app.dashboard import stage_rows


def test_stage_rows_summarizes_each_stage():
    state = PipelineState(query="q")
    state.dense = [Chunk(id="a", title="t", url="u", content="x", score=0.5)]
    state.reranked = [Chunk(id="a", title="t", url="u", content="x", score=3.2)]
    state.answer = "final"
    state.faithfulness = 0.81

    rows = stage_rows(state)

    labels = [r["stage"] for r in rows]
    assert "dense" in labels
    assert "reranked" in labels
    faith = next(r for r in rows if r["stage"] == "faithfulness")
    assert faith["detail"] == "0.81"
