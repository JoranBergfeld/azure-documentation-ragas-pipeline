# Pipeline progress feedback (Streamlit step checklist + SSE stream)

**Date:** 2026-06-20
**Status:** Approved
**Origin:** Follow-on to the agentic-wrapper UI work (`2026-06-18-run-tab-mode-selector-design.md`,
ADR-0015). The Run tab and `/run` now expose every substrate; the next gap is *feedback during
a run* — both surfaces block until the whole pipeline finishes.

## Problem

`run_pipeline` (`src/ragpipe/workflow.py:46`) is one opaque `await`: retrieve once, then a loop of
rerank → generate → faithfulness → `decide_next`, up to three attempts. The slow phases (LLM
generation, the faithfulness judge, and the multi-round agentic retrieve) dominate wall-clock
time, but the caller sees nothing until the end:

- **Streamlit Run tab** (`app/dashboard.py`, in `main()`): the whole thing runs inside a single
  `st.spinner(...)` wrapping `asyncio.run(pipeline_fn(query))`. No incremental feedback.
- **FastAPI `/run`** (`app/api.py`): returns one JSON blob after the pipeline finishes — fine for a
  script, useless for a live progress UI on an external website.

We want the run presented as a **composition of steps that each turn green as they complete, with
overall progress visible**, on both surfaces. The API must expose the same progress so an external
web client (a personal website) can render it live. For agentic modes, the retrieve loop
(plan → fan-out iterations → fuse) is often the slowest part and should show its sub-rounds.

## Goals

- `run_pipeline` emits a structured progress event at every phase boundary it already traces, with
  per-attempt granularity (each retry shows its own rerank/generate/verify and the faithfulness
  score + retry/abstain decision).
- Agentic retrieve emits nested sub-round events (`plan` → `iter_i` → `fuse`) under Retrieve.
- The Streamlit Run tab shows a live, green-ticked step checklist that ends green on PASS / red on
  abstain.
- The API exposes a `POST /run/stream` Server-Sent-Events endpoint emitting the same events live,
  terminating with the final result (or an error).
- **Zero behavior change when no consumer is attached.** The event sink defaults to `None`; existing
  callers (`/run`, `/compare`, the eval harness, every current test) are byte-for-byte unaffected.
- The whole emission path is pure-testable with a recording sink — no network, no live server.

## Non-goals

- Only **agentic** retrieve gets sub-round detail. `combined` and `graphrag` retrieve as a single
  "Retrieve" step in v1; their internal stages already show in the post-run trace table.
- SSE only — no job-id/poll endpoint, no reconnect/resume, no auth, no server-side job persistence.
- No determinate percentage bar: the retry count is unknown up front, so a `0–100%` bar would
  either lie or jump backwards. Overall progress is conveyed by the status container's
  running → complete/error state plus the growing green checklist. (A bar is a possible fast-follow.)
- No change to `/compare`, `/eval`, or the Evaluation/Architecture tabs.

## Design

### 1. Event model — new `src/ragpipe/progress.py`

The single serializable contract shared by both surfaces.

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable

@dataclass(frozen=True)
class ProgressEvent:
    phase: str            # retrieve | rerank | generate | faithfulness | decision | abstain
                          # nested (agentic): retrieve.plan | retrieve.iter | retrieve.fuse
    status: str           # "start" | "complete" | "error"
    attempt: int = 0      # 0-based loop attempt; 0 for retrieve and its sub-rounds
    message: str = ""     # human label, e.g. "Generating answer (attempt 2)"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

ProgressSink = Callable[[ProgressEvent], None]

def emit(sink: ProgressSink | None, phase: str, status: str, *,
         attempt: int = 0, message: str = "", **detail) -> None:
    """None-tolerant helper: builds a ProgressEvent and calls the sink if present."""
    if sink is None:
        return
    sink(ProgressEvent(phase=phase, status=status, attempt=attempt,
                       message=message, detail=dict(detail)))
```

`detail` carries phase-specific data: `{"k": 8}` (rerank window), `{"score": 0.82, "threshold":
0.7, "decision": "retry"}` (decision), `{"index": 0, "total": 3, "subquery": "…"}` (agentic iter).
`status="start"` → the UI shows a spinner line; `status="complete"` → the line flips to green ✓ and
attaches `detail`. `error` is used for the faithfulness fail-closed path and stream failures.

### 2. Instrumentation in `run_pipeline` (`src/ragpipe/workflow.py`)

Add a keyword-only `on_event: ProgressSink | None = None` to `run_pipeline`. Default `None` keeps
today's behavior exactly. Emit around the boundaries that already call `state.add_trace`:

- **retrieve:** `emit(on_event, "retrieve", "start")` before the call; pass the sink *into* the
  substrate: `deps.retrieve(query, deps.candidate_pool, on_event=on_event)`; then
  `emit(on_event, "retrieve", "complete", stages=list(result.stages), candidates=len(result.candidates))`.
- **rerank (per attempt):** start with `attempt=state.attempt, k=k`; complete with
  `n=len(state.reranked)`.
- **generate (per attempt):** start/complete, `attempt=state.attempt`.
- **faithfulness (per attempt):** complete with `score`; on the existing judge-exception
  fail-closed branch emit `status="error"` (`detail={"error": type(exc).__name__}`) so the UI shows
  the judge fault rather than a silent NaN.
- **decision (per attempt):** one event, `detail={"decision": "pass"|"retry"|"exhausted",
  "score": …, "threshold": deps.threshold}`. On EXHAUSTED additionally emit
  `emit(on_event, "abstain", "complete", suppressed_answer=…)`.

`RetrieveFn` in `workflow.py` widens to accept the optional `on_event` kwarg.

### 3. Substrate protocol + agentic emission

`RetrievalSubstrate.retrieve` (`src/ragpipe/retrieval/substrate.py:24`) gains an optional
`on_event: ProgressSink | None = None`. The three non-agentic substrates (`HybridSubstrate`,
`CombinedSubstrate`, `GraphSubstrate`) accept-and-ignore it (one-line signature change each).

`AgenticSubstrate.retrieve` (`src/ragpipe/retrieval/agentic.py:22`) uses it:

- after planning: `emit(on_event, "retrieve.plan", "complete", total=len(subqueries))`;
- per iteration `i`: `emit(..., "retrieve.iter", "start", index=i, total=n, subquery=sq)` then
  `"complete"` with `candidates=len(result.candidates)`;
- after fusion: `emit(..., "retrieve.fuse", "complete", fused=len(candidates))`.

Agentic does **not** forward `on_event` into `inner.retrieve` — that would surface dense/bm25
sub-stages under every iteration (noise). v1 shows plan/iter/fuse only.

### 4. Wiring (`src/ragpipe/app_wiring.py`)

`build_pipeline_fn` returns a callable that accepts the optional sink:

```python
async def pipeline_fn(query: str, *, on_event=None) -> PipelineState:
    return await run_pipeline(query, deps, on_event=on_event)
```

`pipeline_fn(query)` still works (the dashboard's old call site and any test). `make_deps` is
unchanged — `deps.retrieve` is the substrate's bound `retrieve`, which now accepts `on_event`.

### 5. SSE adapter — new `src/ragpipe/streaming.py` (pure, testable without HTTP)

```python
async def pipeline_event_stream(pipeline_fn, query) -> AsyncIterator[tuple[str, object]]:
    """Yields ("progress", ProgressEvent) … then a terminal ("result", PipelineState)
    or ("error", Exception). Runs the pipeline as a task; a queue bridges the sync sink
    to async iteration. Concurrency-safe: queue + sink are per-call."""
```

It builds an `asyncio.Queue`, a `sink` that `put_nowait`s `("progress", event)`, runs
`pipeline_fn(query, on_event=sink)` in a task that finally enqueues `("result", state)` /
`("error", exc)` and a sentinel, and async-drains the queue. No shared mutable state, so concurrent
SSE requests are isolated.

### 6. API endpoint (`app/api.py`)

New `POST /run/stream` taking the existing `RunRequest` (query + required `mode`). It builds
`pipeline_fn` exactly as `/run` does, then returns a `StreamingResponse(..., media_type=
"text/event-stream")` that maps the adapter's tuples to SSE frames:

- `("progress", ev)` → `event: progress\ndata: {ev.to_dict() json}\n\n`
- `("result", state)` → `event: result\ndata: {_state_payload(mode, state) json}\n\n`
- `("error", exc)` → `event: error\ndata: {"error": "<type>"}\n\n`

It reuses `/run`'s existing serializer `_state_payload(mode, state)` (`app/api.py:46`). `/run`,
`/compare`, `/eval` are untouched.

### 7. Streamlit Run tab (`app/dashboard.py`)

Replace the single `st.spinner` + `asyncio.run(pipeline_fn(query))` with an
`st.status("Running pipeline…", expanded=True)` container holding a green-ticked checklist. A pure,
unit-tested helper maps an event to its rendered line:

```python
def progress_step_view(event) -> tuple[str, str]:
    """(icon, label) for a progress event: ⏳ on start, ✅ on complete, ⚠️ on error,
    with attempt number and key detail (score, k, subquery) folded into the label."""
```

The sink keeps a per-step placeholder (keyed by `(phase, attempt)` / agentic `index`): `start`
writes `⏳ {label}`; `complete` rewrites it to `✅ {label}`. The container ends
`state="complete"` (green) on PASS, `state="error"` (red) when `state.abstained`. The run is driven
by `asyncio.run(pipeline_fn(query, on_event=sink))` on the Streamlit script thread, where the
ScriptRunContext is present, so element deltas stream to the browser live. `main()` and the sink
glue stay `# pragma: no cover`; `progress_step_view` is the tested seam.

## Testing

- **`tests/test_progress.py`** — `ProgressEvent.to_dict()` round-trip; `emit(None, …)` is a no-op;
  `emit(sink, …)` builds the expected event.
- **`tests/test_workflow.py`** (extend the existing file) — `run_pipeline` with fake deps + a recording
  sink: assert the ordered event sequence for (a) a clean PASS, (b) a forced retry (score below
  threshold once, then above) showing two attempts, and (c) the abstain path (always below
  threshold → `decision` exhausted + `abstain`). Assert the judge-exception branch emits a
  faithfulness `error` event. **Regression guard:** `on_event=None` emits nothing and returns a
  state identical to the pre-change behavior.
- **`tests/retrieval/test_agentic.py`** (extend) — `AgenticSubstrate` with a fake inner + plan_fn
  and a recording sink: assert `retrieve.plan` + one `retrieve.iter` start/complete per subquery
  (with `index`/`total`) + `retrieve.fuse`; and that behavior/result is unchanged when
  `on_event=None`.
- **`tests/test_streaming.py`** — drive `pipeline_event_stream` with a fake `pipeline_fn` that
  calls its sink a few times then returns a stub state: assert the yielded tuple order ends in
  `("result", state)`, and that a raising `pipeline_fn` yields `("error", exc)`. No live server.
- **`app/dashboard.py`** — unit-test `progress_step_view` for start/complete/error and a couple of
  detail shapes; an `AppTest` smoke that a run path doesn't raise.

Validate with `uv run pytest -q` and `uv run ruff check .`.

## Consequences

- **Additive, backward-compatible.** The new `on_event` is keyword-only with a `None` default
  through `run_pipeline`, the substrate protocol, and `pipeline_fn`; every existing caller and test
  is unchanged. The substrate protocol gains an optional kwarg — a documented "substrates MAY emit
  progress" seam that `combined`/`graph` can adopt later without breaking the contract.
- **New public API surface.** `POST /run/stream` (SSE) is added; `/run` stays as the non-streaming
  blob. The README API section gains the streaming endpoint and its event shapes.
- **One event model, two renderers.** `ProgressEvent` serializes straight to SSE JSON and drives
  the Streamlit checklist, so the website and the dashboard stay in lockstep.
- **Concurrency-safe.** The sink and queue are created per call, so concurrent SSE clients don't
  interfere — important once an external site drives `/run/stream`.
- **Slightly more events than phases** (start+complete per long step, plus agentic sub-rounds). This
  is intentional: the per-attempt detail and the retrieve sub-rounds are exactly what explains the
  long waits the user is trying to see.
