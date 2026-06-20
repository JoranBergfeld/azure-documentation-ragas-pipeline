import pytest

from ragpipe.models import Chunk, PipelineState
from ragpipe.retrieval.substrate import RetrievalResult
from ragpipe.workflow import ABSTENTION_ANSWER, PipelineDeps, run_pipeline


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=f"content-{cid}")


def _fake_retrieve(chunks):
    async def retrieve(query, k, on_event=None):
        return RetrievalResult(
            candidates=chunks,
            stages={"dense": chunks, "bm25": [], "fused": chunks},
        )
    return retrieve


def _deps(score_sequence, rerank_calls=None, generate_calls=None):
    """Deps whose scorer returns scores from a sequence per attempt; optionally
    records the requested top_k per rerank call and the previous_answer per
    generate call."""
    scores = iter(score_sequence)
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]

    def rerank(q, candidates, k):
        if rerank_calls is not None:
            rerank_calls.append(k)
        return candidates[:k]

    def generate(q, chunks, previous_answer):
        if generate_calls is not None:
            generate_calls.append(previous_answer)
        return f"answer for {q}"

    return PipelineDeps(
        retrieve=_fake_retrieve(chunks),
        rerank=rerank,
        generate=generate,
        score=lambda q, answer, chunks: next(scores),
        threshold=0.7,
        max_retries=2,
        top_k=2,
        rerank_widen_step=3,
    )


@pytest.mark.asyncio
async def test_pipeline_passes_first_try():
    state = await run_pipeline("what is RRF?", _deps([0.9]))
    assert isinstance(state, PipelineState)
    assert state.answer == "answer for what is RRF?"
    assert state.faithfulness == 0.9
    assert state.attempt == 0
    assert state.low_confidence is False
    assert state.abstained is False
    stages = [e.stage for e in state.trace]
    assert stages[:4] == ["dense", "bm25", "fused", "rerank"]
    # substrate stages are populated and reranked is mirrored into stages
    assert len(state.stages["fused"]) > 0
    assert state.reranked == state.stages["reranked"]


@pytest.mark.asyncio
async def test_pipeline_loops_then_passes():
    state = await run_pipeline("q", _deps([0.4, 0.85]))
    assert state.attempt == 1
    assert state.faithfulness == 0.85
    assert state.low_confidence is False
    assert state.abstained is False


@pytest.mark.asyncio
async def test_retry_widens_rerank_window():
    calls = []
    await run_pipeline("q", _deps([0.4, 0.85], rerank_calls=calls))
    assert calls == [2, 5]  # top_k + widen_step * attempt


@pytest.mark.asyncio
async def test_retry_threads_previous_answer():
    calls = []
    await run_pipeline("q", _deps([0.4, 0.85], generate_calls=calls))
    assert calls[0] is None
    assert calls[1] is not None and "answer for q" in calls[1]


@pytest.mark.asyncio
async def test_exhaustion_abstains_with_directive_answer():
    state = await run_pipeline("q", _deps([0.1, 0.2, 0.3]))
    assert state.attempt == 2
    assert state.low_confidence is True
    assert state.abstained is True
    assert state.answer == ABSTENTION_ANSWER
    # the suppressed answer is preserved in the trace for debugging
    abstain_events = [e for e in state.trace if e.stage == "abstain"]
    assert len(abstain_events) == 1
    assert "answer for q" in abstain_events[0].data["suppressed_answer"]


@pytest.mark.asyncio
async def test_judge_exception_abstains_without_retry():
    def boom(q, a, c):
        raise RuntimeError("judge down")

    deps = _deps([0.9])
    deps.score = boom
    state = await run_pipeline("q", deps)
    assert state.attempt == 0  # no retries burned
    assert state.abstained is True
    assert state.answer == ABSTENTION_ANSWER


def _events():
    captured: list = []
    return captured, captured.append


@pytest.mark.asyncio
async def test_emits_phase_events_on_pass():
    events, sink = _events()
    await run_pipeline("q", _deps([0.9]), on_event=sink)
    seq = [(e.phase, e.status) for e in events]
    assert ("retrieve", "start") in seq and ("retrieve", "complete") in seq
    assert ("rerank", "start") in seq and ("rerank", "complete") in seq
    assert ("generate", "start") in seq and ("generate", "complete") in seq
    assert ("faithfulness", "complete") in seq
    decision = [e for e in events if e.phase == "decision"][-1]
    assert decision.detail["decision"] == "pass"
    assert decision.detail["score"] == 0.9


@pytest.mark.asyncio
async def test_emits_per_attempt_events_on_retry():
    events, sink = _events()
    await run_pipeline("q", _deps([0.4, 0.85]), on_event=sink)
    gen_starts = [e.attempt for e in events if e.phase == "generate" and e.status == "start"]
    assert gen_starts == [0, 1]
    decisions = [e.detail["decision"] for e in events if e.phase == "decision"]
    assert decisions == ["retry", "pass"]


@pytest.mark.asyncio
async def test_emits_abstain_event_on_exhaustion():
    events, sink = _events()
    await run_pipeline("q", _deps([0.1, 0.2, 0.3]), on_event=sink)
    assert any(e.phase == "abstain" and e.status == "complete" for e in events)
    assert [e for e in events if e.phase == "decision"][-1].detail["decision"] == "exhausted"


@pytest.mark.asyncio
async def test_emits_faithfulness_error_event_on_judge_failure():
    events, sink = _events()
    deps = _deps([0.9])

    def boom(q, a, c):
        raise RuntimeError("judge down")

    deps.score = boom
    await run_pipeline("q", deps, on_event=sink)
    assert any(e.phase == "faithfulness" and e.status == "error" for e in events)


@pytest.mark.asyncio
async def test_no_sink_keeps_behavior_unchanged():
    state = await run_pipeline("q", _deps([0.9]))  # default on_event=None
    assert state.faithfulness == 0.9 and state.abstained is False
