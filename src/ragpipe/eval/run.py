"""Offline evaluation entry point: python -m ragpipe.eval.run"""
from __future__ import annotations

import asyncio
import json

from ragpipe.app_wiring import build_pipeline_fn
from ragpipe.config import Settings
from ragpipe.eval.harness import aggregate, build_ragas_evaluator, run_harness
from ragpipe.eval.testset import load_testset


def main() -> None:  # pragma: no cover - integration entry point
    settings = Settings.from_env()
    items = load_testset(settings.testset_mode)
    pipeline_fn = build_pipeline_fn(settings)
    evaluator_fn = build_ragas_evaluator(settings)

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
    means = aggregate(records)
    with open("eval_results.json", "w") as f:
        json.dump(
            {"means": means, "records": [r.__dict__ for r in records]}, f, indent=2
        )
    print(json.dumps(means, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
