from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dashboard import (
    EVAL_RESULTS_PATH,
    eval_rows,
    per_stage_chart_data,
    stage_chunk_tables,
)
from ragpipe.config import RetrievalMode, Settings
from ragpipe.models import PipelineState
from ragpipe.streaming import pipeline_event_stream

app = FastAPI(title="ragpipe API")

# Per-mode cache — building Azure clients is expensive so we reuse per mode.
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
        "query": state.query,
        "answer": state.answer,
        "faithfulness": _json_safe(state.faithfulness),
        "attempt": state.attempt,
        "lowConfidence": state.low_confidence,
        "abstained": state.abstained,
        "stages": stage_chunk_tables(state),
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(_json_safe(data))}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(
    req: RunRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> dict[str, Any]:
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


@app.post("/compare")
async def compare(
    req: CompareRequest,
    factory: Callable[[str], Awaitable[Callable[..., Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> dict[str, Any]:
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
        return {
            "meansByMode": results["means_by_mode"],
            "modes": list(results.get("modes", {}).keys()),
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
