# Pipeline Progress Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit per-phase progress from `run_pipeline` via an optional `on_event` sink (default `None` = unchanged) and surface it as a live green-ticked checklist in the Streamlit Run tab and a `POST /run/stream` SSE endpoint, with agentic retrieve emitting `plan/iter/fuse` sub-rounds.

**Architecture:** A small serializable `ProgressEvent` + `ProgressSink = Callable[[ProgressEvent], None]` is the single contract. `run_pipeline` and the `retrieve` protocol gain a keyword-only `on_event=None`; `run_pipeline` emits at the boundaries it already traces, and `AgenticSubstrate` emits sub-rounds. A pure async-generator adapter (`pipeline_event_stream`) bridges the sync sink to SSE for the API. The dashboard drives an `st.status` checklist from the same sink.

**Tech Stack:** Python 3.11, FastAPI (`StreamingResponse`/SSE), Streamlit (`st.status`), pytest (`asyncio_mode=auto`), `uv`, ruff.

**Spec:** `docs/superpowers/specs/2026-06-20-pipeline-progress-feedback-design.md`

**Conventions to honor:**
- `from __future__ import annotations` at the top of every new module.
- Run tests with `uv run pytest -q`; lint with `uv run ruff check .` (line-length 100).
- Live wiring / UI entry points stay `# pragma: no cover`; pure logic is the tested seam.
- Commit trailer on every commit: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

---

## File Structure

**Create:**
- `src/ragpipe/progress.py` — `ProgressEvent`, `ProgressSink`, `emit()` helper. One responsibility: the event contract.
- `src/ragpipe/streaming.py` — `pipeline_event_stream()` async-generator adapter (sink → async iterator). One responsibility: the SSE bridge, testable without HTTP.
- `tests/test_progress.py`, `tests/test_streaming.py`.

**Modify:**
- `src/ragpipe/workflow.py` — `run_pipeline(..., *, on_event=None)` + emit calls at phase boundaries.
- `src/ragpipe/retrieval/substrate.py` — protocol + `HybridSubstrate.retrieve` accept `on_event`.
- `src/ragpipe/retrieval/combined.py`, `src/ragpipe/retrieval/graph_substrate.py` — accept-and-ignore `on_event`.
- `src/ragpipe/retrieval/agentic.py` — emit `retrieve.plan` / `retrieve.iter` / `retrieve.fuse`.
- `src/ragpipe/app_wiring.py` — returned `pipeline_fn(query, *, on_event=None)`.
- `app/api.py` — `POST /run/stream` + `_sse()` helper.
- `app/dashboard.py` — `progress_step_view()` helper + Run-tab `st.status` rewire.
- `tests/test_workflow.py`, `tests/retrieval/test_agentic.py`, `tests/retrieval/test_substrate.py`, `tests/test_api.py`, `tests/test_dashboard_helpers.py` — extend.
- `README.md` — document `POST /run/stream`.
- `docs/adr/0017-pipeline-progress-events.md` — record the seam + SSE decision.

---

## Task 1: Progress event model

**Files:**
- Create: `src/ragpipe/progress.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_progress.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragpipe.progress'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ragpipe/progress.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    """A single pipeline progress event. Serializable (``to_dict``) so the same
    object drives the Streamlit checklist and the SSE payload.

    ``phase`` is one of retrieve | rerank | generate | faithfulness | decision |
    abstain, or a nested agentic sub-round: retrieve.plan | retrieve.iter |
    retrieve.fuse. ``status`` is "start" | "complete" | "error". ``detail`` carries
    phase-specific data (e.g. {"score": .82, "threshold": .7, "decision": "retry"})."""

    phase: str
    status: str
    attempt: int = 0
    message: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


ProgressSink = Callable[[ProgressEvent], None]


def emit(
    sink: ProgressSink | None,
    phase: str,
    status: str,
    *,
    attempt: int = 0,
    message: str = "",
    **detail,
) -> None:
    """None-tolerant emit: build a ProgressEvent and call ``sink`` if present."""
    if sink is None:
        return
    sink(ProgressEvent(phase=phase, status=status, attempt=attempt, message=message, detail=dict(detail)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_progress.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/progress.py tests/test_progress.py
git commit -m "feat(progress): serializable ProgressEvent + None-tolerant emit helper

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Instrument `run_pipeline` with `on_event`

**Files:**
- Modify: `src/ragpipe/workflow.py:1-100`
- Test: `tests/test_workflow.py` (extend)

- [ ] **Step 1: Update the existing fake so it accepts the new kwarg, then write failing tests**

In `tests/test_workflow.py`, change `_fake_retrieve` (line 12-18) so its inner `retrieve` accepts `on_event` — `run_pipeline` will now pass it:

```python
def _fake_retrieve(chunks):
    async def retrieve(query, k, on_event=None):
        return RetrievalResult(
            candidates=chunks,
            stages={"dense": chunks, "bm25": [], "fused": chunks},
        )
    return retrieve
```

Then append these tests to `tests/test_workflow.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workflow.py -q`
Expected: the 5 new tests FAIL with `TypeError: run_pipeline() got an unexpected keyword argument 'on_event'`. (The existing 6 tests still pass; the `_fake_retrieve` edit is forward-compatible.)

- [ ] **Step 3: Add `on_event` + emit calls to `run_pipeline`**

In `src/ragpipe/workflow.py`, add the import near the top (after the existing `from ragpipe...` imports, around line 9):

```python
from ragpipe.progress import ProgressSink, emit
```

Replace the whole `run_pipeline` function body (lines 46-100) with:

```python
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
```

Also update the `RetrieveFn` comment/type at line 12 to note the optional kwarg:

```python
RetrieveFn = Callable[..., object]  # (query, pool, *, on_event=None) -> awaitable RetrievalResult
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workflow.py -q`
Expected: PASS (11 passed — 6 original + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/workflow.py tests/test_workflow.py
git commit -m "feat(workflow): emit per-phase progress events from run_pipeline

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Thread `on_event` through the retrieve protocol (non-agentic substrates accept-and-ignore)

**Files:**
- Modify: `src/ragpipe/retrieval/substrate.py:20-45`, `src/ragpipe/retrieval/combined.py:17`, `src/ragpipe/retrieval/graph_substrate.py:130`
- Test: `tests/retrieval/test_substrate.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/retrieval/test_substrate.py`:

```python
@pytest.mark.asyncio
async def test_hybrid_accepts_and_ignores_on_event():
    dense = _FakeLeg([Chunk(id="a", title="", url="", content="x")])
    bm25 = _FakeLeg([Chunk(id="b", title="", url="", content="y")])
    sub = HybridSubstrate(name="baseline", dense=dense, bm25=bm25, rrf_k=60)

    seen: list = []
    result = await sub.retrieve("q", k=10, on_event=seen.append)

    assert set(result.stages) == {"dense", "bm25", "fused"}
    assert seen == []  # hybrid does not emit; it just accepts the contract kwarg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_substrate.py::test_hybrid_accepts_and_ignores_on_event -q`
Expected: FAIL with `TypeError: retrieve() got an unexpected keyword argument 'on_event'`.

- [ ] **Step 3: Add `on_event` to the protocol and the three non-agentic substrates**

In `src/ragpipe/retrieval/substrate.py`, add the import after line 6 (`from ragpipe.models import Chunk`):

```python
from ragpipe.progress import ProgressSink
```

Change the protocol method (line 24) to:

```python
    async def retrieve(self, query: str, k: int, on_event: ProgressSink | None = None) -> RetrievalResult: ...
```

Change `HybridSubstrate.retrieve` signature (line 38) to:

```python
    async def retrieve(self, query: str, k: int, on_event: ProgressSink | None = None) -> RetrievalResult:
```

(The body is unchanged — Hybrid does not emit.)

In `src/ragpipe/retrieval/combined.py`, change line 17 to:

```python
    async def retrieve(self, query: str, k: int, on_event=None) -> RetrievalResult:
```

In `src/ragpipe/retrieval/graph_substrate.py`, change line 130 to:

```python
    async def retrieve(self, query: str, k: int, on_event=None) -> RetrievalResult:
```

(Bodies unchanged — both ignore `on_event`; they call `sub.retrieve(query, k)` / leaf searches without it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_substrate.py tests/retrieval/test_combined.py -q`
Expected: PASS (all green — existing tests call `retrieve(query, k)` and still work via the default; `test_protocol_is_runtime_checkable` still passes because `@runtime_checkable` checks method presence, not signature).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/substrate.py src/ragpipe/retrieval/combined.py src/ragpipe/retrieval/graph_substrate.py tests/retrieval/test_substrate.py
git commit -m "feat(retrieval): add optional on_event to the retrieve protocol

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Agentic sub-round emission

**Files:**
- Modify: `src/ragpipe/retrieval/agentic.py`
- Test: `tests/retrieval/test_agentic.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/retrieval/test_agentic.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_agentic.py -q`
Expected: `test_agentic_emits_plan_iter_fuse_events` FAILs with `TypeError: retrieve() got an unexpected keyword argument 'on_event'`.

- [ ] **Step 3: Emit sub-round events in `AgenticSubstrate.retrieve`**

In `src/ragpipe/retrieval/agentic.py`, add after line 6 (`from ragpipe.retrieval.substrate import RetrievalResult`):

```python
from ragpipe.progress import ProgressSink, emit
```

Replace the `retrieve` method (lines 22-35) with:

```python
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
```

(Note: `on_event` is intentionally **not** forwarded into `self._inner.retrieve` — that would surface dense/bm25 sub-stages under every iteration. v1 shows plan/iter/fuse only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_agentic.py -q`
Expected: PASS (4 passed — 2 original + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/agentic.py tests/retrieval/test_agentic.py
git commit -m "feat(agentic): emit plan/iter/fuse retrieve sub-round progress

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: SSE async-generator adapter

**Files:**
- Create: `src/ragpipe/streaming.py`
- Test: `tests/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_streaming.py`:

```python
from __future__ import annotations

import pytest

from ragpipe.progress import ProgressEvent
from ragpipe.streaming import pipeline_event_stream


@pytest.mark.asyncio
async def test_stream_yields_progress_then_result():
    async def fake_pipeline(query, *, on_event=None):
        on_event(ProgressEvent(phase="retrieve", status="start"))
        on_event(ProgressEvent(phase="generate", status="complete"))
        return {"state": query}

    items = [item async for item in pipeline_event_stream(fake_pipeline, "q")]
    assert [kind for kind, _ in items] == ["progress", "progress", "result"]
    assert items[-1][1] == {"state": "q"}
    assert items[0][1].phase == "retrieve"


@pytest.mark.asyncio
async def test_stream_yields_error_on_pipeline_failure():
    async def boom(query, *, on_event=None):
        raise RuntimeError("down")

    items = [item async for item in pipeline_event_stream(boom, "q")]
    assert items[-1][0] == "error"
    assert isinstance(items[-1][1], RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streaming.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragpipe.streaming'`.

- [ ] **Step 3: Write the adapter**

Create `src/ragpipe/streaming.py`:

```python
from __future__ import annotations

import asyncio
from typing import AsyncIterator


async def pipeline_event_stream(pipeline_fn, query: str) -> AsyncIterator[tuple[str, object]]:
    """Bridge ``run_pipeline``'s synchronous ``on_event`` sink to an async iterator.

    Yields ``("progress", ProgressEvent)`` for each emitted event, then exactly one
    terminal ``("result", state)`` on success or ``("error", exc)`` on failure.
    The queue and sink are created per call, so concurrent callers are isolated."""
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def sink(event) -> None:
        queue.put_nowait(("progress", event))

    async def runner() -> None:
        try:
            state = await pipeline_fn(query, on_event=sink)
            queue.put_nowait(("result", state))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error frame
            queue.put_nowait(("error", exc))
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item
    finally:
        await task
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streaming.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/streaming.py tests/test_streaming.py
git commit -m "feat(streaming): async-generator adapter bridging sink to SSE

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Wiring — `pipeline_fn` forwards `on_event`

**Files:**
- Modify: `src/ragpipe/app_wiring.py:23-139`

This is `# pragma: no cover` live wiring (needs Azure), so there is no unit test; the contract is exercised by the Task 5/7 fakes. Verify with ruff + the full suite.

- [ ] **Step 1: Make the returned closure accept `on_event`**

In `src/ragpipe/app_wiring.py`, change the return annotation on line 27 to:

```python
) -> Callable[..., Awaitable[PipelineState]]:  # pragma: no cover - live wiring
```

Replace the `pipeline_fn` closure (lines 136-137) with:

```python
    async def pipeline_fn(query: str, *, on_event=None) -> PipelineState:
        return await run_pipeline(query, deps, on_event=on_event)
```

- [ ] **Step 2: Loosen the API cache type hints to match**

In `app/api.py`, change line 22 and the two `Depends`/factory annotations (lines 25, 67, 79) from `Callable[[str], Awaitable[PipelineState]]` to `Callable[..., Awaitable[PipelineState]]`:

```python
_pipeline_fns: dict[str, Callable[..., Awaitable[PipelineState]]] = {}
```

```python
def get_pipeline_fn_for_mode() -> Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]]:
    async def factory(mode: str) -> Callable[..., Awaitable[PipelineState]]:
```

(Update both `run` and `compare` parameter annotations the same way: `Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]]`.)

- [ ] **Step 3: Verify lint + full suite still green**

Run: `uv run ruff check . && uv run pytest -q`
Expected: ruff clean; all tests pass (no behavior change — `/run` still calls `pipeline_fn(req.query)`).

- [ ] **Step 4: Commit**

```bash
git add src/ragpipe/app_wiring.py app/api.py
git commit -m "feat(wiring): pipeline_fn forwards optional on_event sink

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: API `POST /run/stream` (SSE)

**Files:**
- Modify: `app/api.py`
- Test: `tests/test_api.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_run_stream_emits_progress_and_result():
    from ragpipe.progress import ProgressEvent

    async def fake_pipeline(query, *, on_event=None):
        on_event(ProgressEvent(phase="retrieve", status="start", message="Retrieving"))
        on_event(ProgressEvent(phase="generate", status="complete", attempt=0, message="Answer generated"))
        return _state()

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        res = TestClient(api.app).post("/run/stream", json={"query": "what is RRF?", "mode": "contextual"})
    finally:
        api.app.dependency_overrides.clear()

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    body = res.text
    assert "event: progress" in body
    assert "event: result" in body
    assert '"phase": "retrieve"' in body
    assert "RRF merges ranked lists." in body  # serialized final state


def test_run_stream_emits_error_frame_on_failure():
    async def boom_pipeline(query, *, on_event=None):
        raise RuntimeError("kaboom")

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(boom_pipeline)
    try:
        res = TestClient(api.app).post("/run/stream", json={"query": "x", "mode": "contextual"})
    finally:
        api.app.dependency_overrides.clear()

    assert res.status_code == 200
    assert "event: error" in res.text
    assert "RuntimeError" in res.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k run_stream -q`
Expected: FAIL with 404 (no `/run/stream` route) → assertions fail.

- [ ] **Step 3: Add the endpoint**

In `app/api.py`, add to the imports (after line 7, `from fastapi import Depends, FastAPI`):

```python
from fastapi.responses import StreamingResponse
```

and after line 17 (`from ragpipe.models import PipelineState`):

```python
from ragpipe.streaming import pipeline_event_stream
```

Add an `_sse` helper after `_state_payload` (after line 56):

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```

Add the endpoint after the `/run` route (after line 73):

```python
@app.post("/run/stream")
async def run_stream(
    req: RunRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> StreamingResponse:
    pipeline_fn = await factory(req.mode)

    async def frames():
        async for kind, payload in pipeline_event_stream(pipeline_fn, req.query):
            if kind == "progress":
                yield _sse("progress", payload.to_dict())
            elif kind == "result":
                yield _sse("result", _state_payload(req.mode, payload))
            else:  # "error"
                yield _sse("error", {"error": type(payload).__name__})

    return StreamingResponse(frames(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -q`
Expected: PASS (all green, including the 2 new stream tests).

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "feat(api): POST /run/stream Server-Sent-Events progress endpoint

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Dashboard Run-tab progress checklist

**Files:**
- Modify: `app/dashboard.py` (add `progress_step_view`; rewire Run tab at lines 212-218)
- Test: `tests/test_dashboard_helpers.py` (extend)

- [ ] **Step 1: Write the failing test**

In `tests/test_dashboard_helpers.py`, add `progress_step_view` to the import block (lines 4-13):

```python
from app.dashboard import (
    available_architecture_diagrams,
    chunk_label,
    eval_rows,
    is_agentic_mode,
    per_stage_chart_data,
    progress_step_view,
    stage_chunk_tables,
    stage_expanded,
    stage_rows,
)
```

Append these tests (define a tiny stand-in so the test does not import `ragpipe.progress` unless desired — but importing the real event is fine and preferred):

```python
def test_progress_step_view_icons_by_status():
    from ragpipe.progress import ProgressEvent

    start = ProgressEvent(phase="generate", status="start", message="Generating answer (attempt 1)")
    done = ProgressEvent(phase="faithfulness", status="complete", message="Faithfulness 0.82")
    err = ProgressEvent(phase="faithfulness", status="error", message="Faithfulness judge failed")

    assert progress_step_view(start) == ("⏳", "Generating answer (attempt 1)")
    assert progress_step_view(done) == ("✅", "Faithfulness 0.82")
    assert progress_step_view(err) == ("⚠️", "Faithfulness judge failed")


def test_progress_step_view_falls_back_to_phase_when_no_message():
    from ragpipe.progress import ProgressEvent

    ev = ProgressEvent(phase="retrieve.fuse", status="complete")
    assert progress_step_view(ev) == ("✅", "retrieve.fuse")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard_helpers.py -k progress_step_view -q`
Expected: FAIL with `ImportError: cannot import name 'progress_step_view'`.

- [ ] **Step 3: Add the pure helper**

In `app/dashboard.py`, add this function near the other pure helpers (e.g. after `is_agentic_mode`, around line 32):

```python
def progress_step_view(event) -> tuple[str, str]:
    """Render a ProgressEvent as (icon, label) for the Run-tab live checklist."""
    icon = {"start": "⏳", "complete": "✅", "error": "⚠️"}.get(event.status, "•")
    return icon, event.message or event.phase
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard_helpers.py -q`
Expected: PASS (all green).

- [ ] **Step 5: Rewire the Run tab to drive an `st.status` checklist**

In `app/dashboard.py`, replace the spinner block (lines 216-218):

```python
            pipeline_fn = build_pipeline_fn(settings, mode=mode)
            with st.spinner("Running pipeline (retrieve → rerank → generate → faithfulness)…"):
                state = asyncio.run(pipeline_fn(query))
```

with:

```python
            pipeline_fn = build_pipeline_fn(settings, mode=mode)
            status = st.status("Running pipeline…", expanded=True)
            placeholders: dict[tuple, Any] = {}

            def on_event(ev) -> None:
                icon, label = progress_step_view(ev)
                key = (ev.phase, ev.attempt, ev.detail.get("index", -1))
                ph = placeholders.get(key)
                if ph is None:
                    ph = status.empty()
                    placeholders[key] = ph
                ph.markdown(f"{icon} {label}")

            state = asyncio.run(pipeline_fn(query, on_event=on_event))
            status.update(
                label="Abstained — insufficient grounded context" if state.abstained else "Pipeline complete",
                state="error" if state.abstained else "complete",
                expanded=False,
            )
```

(`Any` is already imported at the top of `app/dashboard.py`. The keying on `(phase, attempt, index)` makes each `start` create a line that its matching `complete` overwrites — the line flips ⏳→✅ in place; agentic `retrieve.iter` rounds get one line each via the `index` component.)

- [ ] **Step 6: Smoke-test the dashboard renders without error**

Run:

```bash
uv run python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app/dashboard.py'); at.run(timeout=30); print('exception:', at.exception); print('tabs:', len(at.tabs))"
```

Expected: `exception: None` and `tabs: 3`. (The Run button is not clicked, so no Azure call happens; this only verifies the script imports and renders cleanly with the new helper and rewired block.)

- [ ] **Step 7: Commit**

```bash
git add app/dashboard.py tests/test_dashboard_helpers.py
git commit -m "feat(dashboard): live st.status progress checklist on the Run tab

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: Docs — README endpoint + ADR-0017

**Files:**
- Modify: `README.md:85-94`
- Create: `docs/adr/0017-pipeline-progress-events.md`

- [ ] **Step 1: Document `POST /run/stream` in the README**

In `README.md`, insert this bullet immediately after the `/run` bullet block (after line 90, before the `/compare` bullet):

```markdown
- `POST /run/stream` — same body as `/run`; returns a `text/event-stream` (Server-Sent Events).
  Frames: `event: progress` (one per phase boundary — `retrieve`, `rerank`, `generate`,
  `faithfulness`, `decision`; agentic modes also emit `retrieve.plan` / `retrieve.iter` /
  `retrieve.fuse` sub-rounds), then a terminal `event: result` carrying the same payload as
  `/run`, or `event: error` on failure. Drives the dashboard's live step checklist and external
  web clients.
```

- [ ] **Step 2: Write ADR-0017**

Create `docs/adr/0017-pipeline-progress-events.md`:

```markdown
# 0017 — Pipeline progress events (in-band sink + SSE)

**Status:** Accepted (2026-06-20)

## Context

`run_pipeline` is one `await`: retrieve once, then a loop of rerank → generate → faithfulness →
decide, up to three attempts. The slow phases (LLM generation, the faithfulness judge, and the
multi-round agentic retrieve) dominate wall-clock time, but every caller — the Streamlit Run tab
and the FastAPI `/run` endpoint — blocks with no feedback until the whole pipeline finishes. We
want the run presented as a composition of steps that complete incrementally, on both the
dashboard and an external web client driven by the API.

## Decision

Introduce a single serializable `ProgressEvent` (`src/ragpipe/progress.py`) and a
`ProgressSink = Callable[[ProgressEvent], None]`. `run_pipeline` gains a keyword-only
`on_event=None` and emits at the boundaries it already traces; the `RetrievalSubstrate.retrieve`
protocol gains the same optional `on_event` so `AgenticSubstrate` can emit `retrieve.plan` /
`retrieve.iter` / `retrieve.fuse` sub-rounds. The default `None` makes emission a no-op, so every
existing caller and test is unchanged.

For the API, a pure async-generator adapter (`src/ragpipe/streaming.py::pipeline_event_stream`)
bridges the synchronous sink to an async iterator via a per-call `asyncio.Queue`; a new
`POST /run/stream` endpoint maps it to `text/event-stream` frames. The Streamlit Run tab drives an
`st.status` checklist from the same sink. One event model, two renderers.

We rejected an async-generator core (`run_pipeline` becomes a generator) because sub-events from
the substrate would still need a callback, forcing a mixed style; and we rejected a `contextvars`
sink as implicit "magic" that cuts against this repo's explicit-callback convention. The callback
is concurrency-safe because the sink and queue are created per call — important once an external
site drives concurrent `/run/stream` requests.

Scope: only agentic retrieve emits sub-rounds in v1 (`combined`/`graphrag` retrieve as one step;
their stages already show in the post-run trace). SSE only — no job/poll, reconnect, or persistence.
No determinate percentage bar (the retry count is unknown up front); overall progress is the
status container's running → complete/error state plus the growing green checklist.

## Consequences

- Additive and backward-compatible: `on_event` is keyword-only with a `None` default through
  `run_pipeline`, the substrate protocol, and `pipeline_fn`.
- New public API surface (`POST /run/stream`); `/run`, `/compare`, `/eval` are unchanged.
- The substrate protocol now documents a "substrates MAY emit progress" seam that `combined` /
  `graphrag` can adopt later without breaking the contract.
- Slightly more events than phases (start+complete per long step, plus agentic sub-rounds) — this
  is intentional: the per-attempt detail and retrieve sub-rounds are exactly what explains the
  long waits.

## Sources

- Spec: `docs/superpowers/specs/2026-06-20-pipeline-progress-feedback-design.md`
- `src/ragpipe/progress.py`, `src/ragpipe/streaming.py`, `src/ragpipe/workflow.py`
- `src/ragpipe/retrieval/substrate.py`, `src/ragpipe/retrieval/agentic.py`
- `app/api.py` (`/run/stream`), `app/dashboard.py` (`progress_step_view` + Run tab)
- ADR-0009 (directive guardrail / abstention), ADR-0012 (retrieval substrate seam),
  ADR-0015 (agentic retrieval wrapper)
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/adr/0017-pipeline-progress-events.md
git commit -m "docs: document /run/stream + ADR-0017 pipeline progress events

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 10: Full-suite verification

- [ ] **Step 1: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass (≈213 prior + ~14 new) and ruff reports "All checks passed!".

- [ ] **Step 2: If green, the feature is complete.** Push is handled at the end of the session per the user's request.

---

## Self-Review Notes (author checklist — already applied)

- **Spec coverage:** event model (Task 1), `run_pipeline` instrumentation incl. per-attempt +
  abstain + judge-error (Task 2), substrate protocol (Task 3), agentic sub-rounds (Task 4), SSE
  adapter (Task 5), wiring (Task 6), `/run/stream` (Task 7), dashboard checklist (Task 8),
  README + ADR (Task 9), full verification (Task 10). All spec sections map to a task.
- **No placeholders:** every code step shows complete code and exact commands with expected output.
- **Type consistency:** `ProgressEvent(phase,status,attempt,message,detail)`, `emit(sink,phase,
  status,*,attempt,message,**detail)`, `progress_step_view(event)->(icon,label)`,
  `pipeline_event_stream(pipeline_fn,query)->AsyncIterator[tuple[str,object]]`, and
  `pipeline_fn(query,*,on_event=None)` are used identically across all tasks.
```
