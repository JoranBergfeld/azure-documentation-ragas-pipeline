from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.dashboard import (
    EVAL_RESULTS_PATH,
    eval_rows,
    per_stage_chart_data,
    stage_chunk_tables,
)
from ragpipe.config import RetrievalMode, Settings
from ragpipe.models import PipelineState

app = FastAPI(title="ragpipe API")

# Per-mode cache — building Azure clients is expensive so we reuse per mode.
_pipeline_fns: dict[str, Callable[[str], Awaitable[PipelineState]]] = {}


def get_pipeline_fn_for_mode() -> Callable[[str], Awaitable[Callable[[str], Awaitable[PipelineState]]]]:
    async def factory(mode: str) -> Callable[[str], Awaitable[PipelineState]]:
        if mode not in _pipeline_fns:
            from ragpipe.app_wiring import build_pipeline_fn

            _pipeline_fns[mode] = build_pipeline_fn(Settings.from_env(), mode=RetrievalMode(mode))
        return _pipeline_fns[mode]

    return factory


class RunRequest(BaseModel):
    query: str
    mode: str = "contextual"


class CompareRequest(BaseModel):
    query: str
    modes: list[str]


def _state_payload(mode: str, state: PipelineState) -> dict[str, Any]:
    return {
        "mode": mode,
        "query": state.query,
        "answer": state.answer,
        "faithfulness": state.faithfulness,
        "attempt": state.attempt,
        "lowConfidence": state.low_confidence,
        "abstained": state.abstained,
        "stages": stage_chunk_tables(state),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(
    req: RunRequest,
    factory: Callable[[str], Awaitable[Callable[[str], Awaitable[PipelineState]]]] = Depends(
        get_pipeline_fn_for_mode
    ),
) -> dict[str, Any]:
    pipeline_fn = await factory(req.mode)
    state = await pipeline_fn(req.query)
    return _state_payload(req.mode, state)


@app.post("/compare")
async def compare(
    req: CompareRequest,
    factory: Callable[[str], Awaitable[Callable[[str], Awaitable[PipelineState]]]] = Depends(
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
