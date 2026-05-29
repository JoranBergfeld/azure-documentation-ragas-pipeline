import pytest

from ragpipe.models import Chunk, PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=f"content-{cid}")


def _deps(score_sequence):
    """Build deps whose scorer returns scores from a sequence per attempt."""
    scores = iter(score_sequence)
    return PipelineDeps(
        dense=lambda q: [_chunk("a"), _chunk("b")],
        bm25=lambda q: [_chunk("b"), _chunk("c")],
        rerank=lambda q, fused: fused[:2],
        generate=lambda q, chunks: f"answer for {q}",
        score=lambda q, answer, chunks: next(scores),
        threshold=0.7,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_pipeline_passes_first_try():
    state = await run_pipeline("what is RRF?", _deps([0.9]))
    assert isinstance(state, PipelineState)
    assert state.answer == "answer for what is RRF?"
    assert state.faithfulness == 0.9
    assert state.attempt == 0
    assert state.low_confidence is False
    stages = [e.stage for e in state.trace]
    assert stages[:4] == ["dense", "bm25", "rrf", "rerank"]


@pytest.mark.asyncio
async def test_pipeline_loops_then_passes():
    state = await run_pipeline("q", _deps([0.4, 0.85]))
    assert state.attempt == 1
    assert state.faithfulness == 0.85
    assert state.low_confidence is False


@pytest.mark.asyncio
async def test_pipeline_exhausts_and_flags_low_confidence():
    state = await run_pipeline("q", _deps([0.1, 0.2, 0.3]))
    assert state.attempt == 2
    assert state.low_confidence is True
