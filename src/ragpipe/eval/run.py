"""Offline evaluation entry point: python -m ragpipe.eval.run"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os

from ragpipe.app_wiring import build_pipeline_fn
from ragpipe.config import RetrievalMode, Settings, TestsetMode
from ragpipe.eval.harness import (
    aggregate,
    aggregate_by_tag,
    build_per_stage_context_evaluator,
    build_ragas_evaluator,
    coverage,
    run_harness,
)
from ragpipe.eval.stats import mode_confidence_intervals, paired_tests_vs_baseline
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


def per_mode_filename(mode: str) -> str:
    """Filename for a single mode's standalone eval results, suffixed with the
    mode value (e.g. 'contextual' -> 'eval_results_contextual.json')."""
    return f"eval_results_{mode}.json"


def build_per_mode_payload(mode: str, mode_result: dict) -> dict:
    """Self-contained per-mode result: the mode's aggregates + records tagged
    with its name, so each file stands alone. Mirrors one entry of the combined
    eval_results.json `modes` map plus a top-level `mode` key. Does not mutate
    the input."""
    return {"mode": mode, **mode_result}


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


def _load_items(settings):  # pragma: no cover
    """Load (or generate) the testset items based on settings.testset_mode."""
    if settings.testset_mode is TestsetMode.SYNTHETIC:
        print("TESTSET_MODE=synthetic: generating a test set from indexed corpus…")
        synthetic_fn = build_synthetic_generator(settings, _sample_corpus_docs(settings))
        return load_testset(settings.testset_mode, synthetic_fn=synthetic_fn)
    return load_testset(settings.testset_mode)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="contextual,baseline")
    args = parser.parse_args()
    modes = [RetrievalMode(m.strip()) for m in args.modes.split(",") if m.strip()]

    settings = Settings.from_env()
    items = _load_items(settings)
    evaluator_fn = build_ragas_evaluator(settings)

    results_by_mode: dict[str, dict] = {}
    for mode in modes:
        mode_value = mode.value
        filename = per_mode_filename(mode_value)
        # Resume: a per-mode file left by an earlier (possibly interrupted) run is
        # a completed checkpoint. Reuse it instead of re-running the expensive live
        # pipeline + judges. Stripping the top-level "mode" key reconstructs this
        # mode's entry in the combined file's `modes` map exactly.
        if os.path.exists(filename):
            with open(filename) as f:
                cached = json.load(f)
            results_by_mode[mode_value] = {k: v for k, v in cached.items() if k != "mode"}
            print(f"=== mode: {mode_value} === (cached: {filename})", flush=True)
            continue

        print(f"=== mode: {mode_value} ===", flush=True)
        pipeline_fn = build_pipeline_fn(settings, mode=mode)
        records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
        if settings.per_stage_metrics:
            records = asyncio.run(build_per_stage_context_evaluator(settings)(records))
        mode_result = _clean({
            "means": aggregate(records),
            "means_by_tag": aggregate_by_tag(records),
            "ci": mode_confidence_intervals([r.metrics for r in records]),
            "coverage": {k: {"valid": v, "total": t} for k, (v, t) in coverage(records).items()},
            "records": [r.__dict__ for r in records],
        })
        results_by_mode[mode_value] = mode_result
        # Write the per-mode file the moment its mode finishes: each is a
        # standalone, committable artifact and a checkpoint that lets a re-run
        # resume at the next unfinished mode (this run can take hours).
        with open(filename, "w") as f:
            json.dump(build_per_mode_payload(mode_value, mode_result), f, indent=2, allow_nan=False)
        print(f"wrote {filename}", flush=True)

    # The combined eval_results.json the dashboard/API read. means_by_mode is just
    # each mode's `means` (aggregate_by_mode), so it rebuilds from the per-mode
    # aggregates without keeping every mode's records in memory.
    means_by_mode = {m: r["means"] for m, r in results_by_mode.items()}
    # Paired significance vs. the baseline mode (ADR-0018): same test set per
    # mode means record i is the same item everywhere, so per-item metric rows
    # line up for a paired test. Empty when the baseline mode is not in this run.
    baseline_mode = RetrievalMode.BASELINE.value
    rows_by_mode = {
        m: [rec["metrics"] for rec in r.get("records", [])]
        for m, r in results_by_mode.items()
    }
    payload = {
        "means_by_mode": means_by_mode,
        "modes": results_by_mode,
        "baseline_mode": baseline_mode,
        "paired_vs_baseline": paired_tests_vs_baseline(rows_by_mode, baseline_mode),
    }
    with open("eval_results.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(json.dumps({"means_by_mode": means_by_mode}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
