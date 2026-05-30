"""Offline evaluation entry point: python -m ragpipe.eval.run"""
from __future__ import annotations

import asyncio
import json
import math

from ragpipe.app_wiring import build_pipeline_fn
from ragpipe.config import Settings
from ragpipe.eval.harness import (
    aggregate,
    build_per_stage_context_evaluator,
    build_ragas_evaluator,
    coverage,
    run_harness,
)
from ragpipe.eval.testset import load_testset


def _clean(value):
    """Replace non-finite floats (NaN/inf) with None so the output is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def main() -> None:  # pragma: no cover - integration entry point
    settings = Settings.from_env()
    items = load_testset(settings.testset_mode)
    pipeline_fn = build_pipeline_fn(settings)
    evaluator_fn = build_ragas_evaluator(settings)

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))

    if settings.per_stage_metrics:
        print("Per-stage metrics enabled: scoring context_precision/recall per stage…")
        per_stage_fn = build_per_stage_context_evaluator(settings)
        records = asyncio.run(per_stage_fn(records))

    means = aggregate(records)
    cov = {k: {"valid": v, "total": t} for k, (v, t) in coverage(records).items()}
    payload = _clean(
        {
            "means": means,
            "coverage": cov,
            "records": [r.__dict__ for r in records],
        }
    )
    with open("eval_results.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps({"means": means, "coverage": cov}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
