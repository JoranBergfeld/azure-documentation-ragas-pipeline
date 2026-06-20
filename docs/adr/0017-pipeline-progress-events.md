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
