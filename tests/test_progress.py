from __future__ import annotations

from ragpipe.progress import ProgressEvent, emit


def test_event_to_dict_round_trips():
    ev = ProgressEvent(phase="rerank", status="complete", attempt=1, message="ok", detail={"k": 8})
    assert ev.to_dict() == {
        "phase": "rerank",
        "status": "complete",
        "attempt": 1,
        "message": "ok",
        "detail": {"k": 8},
    }


def test_emit_is_noop_when_sink_none():
    emit(None, "retrieve", "start")  # must not raise


def test_emit_builds_event_and_calls_sink():
    seen: list[ProgressEvent] = []
    emit(seen.append, "generate", "complete", attempt=2, message="done", score=0.9)
    assert len(seen) == 1
    ev = seen[0]
    assert (ev.phase, ev.status, ev.attempt, ev.message) == ("generate", "complete", 2, "done")
    assert ev.detail == {"score": 0.9}
