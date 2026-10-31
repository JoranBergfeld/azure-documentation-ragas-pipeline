from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.dashboard import (
    EVAL_RESULTS_PATH,
    eval_rows,
    per_stage_chart_data,
    stage_chunk_tables,
)
from ragpipe.config import (
    ModeUnavailable,
    RetrievalMode,
    Settings,
    is_experimental_mode,
    unavailable_reason,
)
from ragpipe.retrieval.registry import registered_modes
from ragpipe.models import PipelineState
from ragpipe.streaming import pipeline_event_stream

app = FastAPI(title="ragpipe API")


@app.exception_handler(ModeUnavailable)
async def _mode_unavailable(request: Request, exc: ModeUnavailable) -> JSONResponse:
    """A mode that went away after the up-front check (the reranker dropping
    mid-query, issue #40) gets the same 503 body ``_require_available`` sends."""
    return JSONResponse(status_code=503, content={"detail": {"mode": exc.mode, "reason": exc.reason}})

# Per-mode cache — building the clients is expensive so we reuse per mode.
_pipeline_fns: dict[str, Callable[..., Awaitable[PipelineState]]] = {}


def get_pipeline_fn_for_mode() -> Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]]:
    async def factory(mode: str) -> Callable[..., Awaitable[PipelineState]]:
        if mode not in _pipeline_fns:
            from ragpipe.app_wiring import build_pipeline_fn

            _pipeline_fns[mode] = build_pipeline_fn(Settings.from_env(), mode=RetrievalMode(mode))
        return _pipeline_fns[mode]

    return factory


class RunRequest(BaseModel):
    query: str
    mode: RetrievalMode


class CompareRequest(BaseModel):
    query: str
    modes: list[RetrievalMode]


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Infinity) with None.

    RAGAS faithfulness can be NaN (an unparseable judge response). ``json.dumps``
    emits that as the literal ``NaN`` token, which JS ``JSON.parse`` rejects — the
    SSE stream feeds external web clients, so the frames must be strict JSON.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _state_payload(mode: str, state: PipelineState) -> dict[str, Any]:
    return {
        "mode": mode,
        "experimental": is_experimental_mode(mode),
        "query": state.query,
        "answer": state.answer,
        "faithfulness": _json_safe(state.faithfulness),
        "attempt": state.attempt,
        "lowConfidence": state.low_confidence,
        "abstained": state.abstained,
        "stages": stage_chunk_tables(state),
        # Every metered call the run made, in call order; [] when none (ADR-0020).
        "usage": [entry.to_dict() for entry in state.usage],
        # The gate's result per attempt, claim verdicts included (ADR-0018), the
        # same facts the stream's faithfulness and decision steps carry.
        "attempts": [_gate_result(e.data) for e in state.trace if e.stage == "faithfulness"],
    }


def _gate_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": data["attempt"],
        "score": _json_safe(data["score"]),
        "threshold": data["threshold"],
        "claims": data["claims"],
        "decision": data["decision"],
    }


def _require_available(*modes: RetrievalMode) -> None:
    """503 for a mode /modes reports as unavailable, before anything is built."""
    for mode in modes:
        reason = unavailable_reason(mode)
        if reason is not None:
            raise HTTPException(status_code=503, detail={"mode": mode.value, "reason": reason})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(_json_safe(data))}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/modes")
def modes() -> dict[str, Any]:
    """Discoverable contract for the runnable retrieval modes. ``experimental`` flags
    the ``*_agentic`` wrappers, which are exposed without eval coverage (issue #11).
    ``available``/``reason`` say whether a mode can run right now (issue #35): the
    rerank modes are ``reranker-offline`` while the Spark's reranker isn't answering (#40)."""
    items = [
        {
            "mode": m.value,
            "experimental": is_experimental_mode(m),
            "available": unavailable_reason(m) is None,
            "reason": unavailable_reason(m),
        }
        for m in registered_modes()
    ]
    return {
        "modes": items,
        "experimental": [i["mode"] for i in items if i["experimental"]],
    }


@app.post("/run")
async def run(
    req: RunRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> dict[str, Any]:
    _require_available(req.mode)
    pipeline_fn = await factory(req.mode)
    state = await pipeline_fn(req.query)
    return _state_payload(req.mode, state)


@app.post("/run/stream")
async def run_stream(
    req: RunRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> StreamingResponse:
    _require_available(req.mode)
    pipeline_fn = await factory(req.mode)

    async def frames():
        async for kind, payload in pipeline_event_stream(pipeline_fn, req.query):
            if kind == "progress":
                yield _sse("progress", payload.to_dict())
            elif kind == "result":
                yield _sse("result", _state_payload(req.mode, payload))
            elif isinstance(payload, ModeUnavailable):
                yield _sse(
                    "error",
                    {"error": type(payload).__name__, "mode": payload.mode, "reason": payload.reason},
                )
            else:  # "error"
                yield _sse("error", {"error": type(payload).__name__})

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.post("/compare")
async def compare(
    req: CompareRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> dict[str, Any]:
    _require_available(*req.modes)
    results = []
    for mode in req.modes:
        pipeline_fn = await factory(mode)
        results.append(_state_payload(mode, await pipeline_fn(req.query)))
    return {"query": req.query, "results": results}


@app.get("/eval")
def evaluate() -> dict[str, Any]:
    path = Path(EVAL_RESULTS_PATH)
    if not path.exists():
        return {"overall": [], "perStage": {}, "nRecords": 0}
    results = json.loads(path.read_text())
    # New shape: multi-mode eval with means_by_mode key.
    if "means_by_mode" in results:
        mode_results = results.get("modes", {})
        return {
            "meansByMode": results["means_by_mode"],
            "modes": list(mode_results.keys()),
            "ci": {m: r.get("ci", {}) for m, r in mode_results.items()},
            "baselineMode": results.get("baseline_mode"),
            "pairedVsBaseline": results.get("paired_vs_baseline", {}),
        }
    # Old shape: single-run eval.
    overall = [
        {"metric": r["metric"], "meanScore": r["mean_score"], "coverage": r.get("coverage")}
        for r in eval_rows(results)
    ]
    return {
        "overall": overall,
        "perStage": per_stage_chart_data(results),
        "nRecords": len(results.get("records", [])),
    }
