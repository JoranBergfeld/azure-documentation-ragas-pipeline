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
from ragpipe.config import Settings
from ragpipe.models import PipelineState

app = FastAPI(title="ragpipe API")

# Built once on first /run (constructing Azure clients is expensive). Exposed as a
# FastAPI dependency so tests can override it without touching Azure.
_pipeline_fn: Callable[[str], Awaitable[PipelineState]] | None = None


def get_pipeline_fn() -> Callable[[str], Awaitable[PipelineState]]:
    global _pipeline_fn
    if _pipeline_fn is None:
        from ragpipe.app_wiring import build_pipeline_fn

        _pipeline_fn = build_pipeline_fn(Settings.from_env())
    return _pipeline_fn


class RunRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(
    req: RunRequest,
    pipeline_fn: Callable[[str], Awaitable[PipelineState]] = Depends(get_pipeline_fn),
) -> dict[str, Any]:
    state = await pipeline_fn(req.query)
    return {
        "query": state.query,
        "answer": state.answer,
        "faithfulness": state.faithfulness,
        "attempt": state.attempt,
        "lowConfidence": state.low_confidence,
        "abstained": state.abstained,
        "stages": stage_chunk_tables(state),
    }


@app.get("/eval")
def evaluate() -> dict[str, Any]:
    path = Path(EVAL_RESULTS_PATH)
    if not path.exists():
        return {"overall": [], "perStage": {}, "nRecords": 0}
    results = json.loads(path.read_text())
    overall = [
        {"metric": r["metric"], "meanScore": r["mean_score"], "coverage": r.get("coverage")}
        for r in eval_rows(results)
    ]
    return {
        "overall": overall,
        "perStage": per_stage_chart_data(results),
        "nRecords": len(results.get("records", [])),
    }
