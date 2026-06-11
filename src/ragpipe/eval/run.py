"""Offline evaluation entry point: python -m ragpipe.eval.run"""
from __future__ import annotations

import asyncio
import json
import math

from ragpipe.app_wiring import build_pipeline_fn
from ragpipe.config import Settings, TestsetMode
from ragpipe.eval.harness import (
    aggregate,
    aggregate_by_tag,
    build_per_stage_context_evaluator,
    build_ragas_evaluator,
    coverage,
    run_harness,
)
from ragpipe.eval.testset import build_synthetic_generator, load_testset


def _clean(value):
    """Replace non-finite floats (NaN/inf) with None so the output is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _sample_corpus_docs(settings, limit: int = 40) -> list[dict]:  # pragma: no cover - live Azure
    """Pull a sample of indexed chunks to seed synthetic test-set generation.

    Reuses the already-ingested content in Azure AI Search so we don't re-fetch the
    corpus. Returns [{"content", "url"}].
    """
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        settings.search_endpoint, settings.search_index, DefaultAzureCredential()
    )
    results = client.search(search_text="*", top=limit, select=["content", "url"])
    return [{"content": r["content"], "url": r.get("url", "")} for r in results]


def main() -> None:  # pragma: no cover - integration entry point
    settings = Settings.from_env()
    if settings.testset_mode is TestsetMode.SYNTHETIC:
        print("TESTSET_MODE=synthetic: generating a test set from indexed corpus…")
        synthetic_fn = build_synthetic_generator(settings, _sample_corpus_docs(settings))
        items = load_testset(settings.testset_mode, synthetic_fn=synthetic_fn)
    else:
        items = load_testset(settings.testset_mode)
    pipeline_fn = build_pipeline_fn(settings)
    evaluator_fn = build_ragas_evaluator(settings)

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))

    if settings.per_stage_metrics:
        print("Per-stage metrics enabled: scoring context_precision/recall per stage…")
        per_stage_fn = build_per_stage_context_evaluator(settings)
        records = asyncio.run(per_stage_fn(records))

    means = aggregate(records)
    means_by_tag = aggregate_by_tag(records)
    cov = {k: {"valid": v, "total": t} for k, (v, t) in coverage(records).items()}
    payload = _clean(
        {
            "means": means,
            "means_by_tag": means_by_tag,
            "coverage": cov,
            "records": [r.__dict__ for r in records],
        }
    )
    with open("eval_results.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps({"means": means, "means_by_tag": means_by_tag, "coverage": cov}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
