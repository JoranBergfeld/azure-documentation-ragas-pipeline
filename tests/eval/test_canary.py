from __future__ import annotations

import json
import math

import pytest

from ragpipe.eval.canary import (
    DEFAULT_CANARY_PATH,
    CanaryOutcome,
    canary_report,
    evaluate_outcome,
    load_canary,
    run_canary,
    score_to_prediction,
)
from ragpipe.guardrail import ClaimVerdict, FaithfulnessResult, _ensure_ragas_importable


def test_load_canary_integrity():
    items = load_canary()

    assert len(items) == 8
    assert sum(item.expected_faithful for item in items) == 4
    assert sum(not item.expected_faithful for item in items) == 4
    assert DEFAULT_CANARY_PATH.name == "faithfulness_canary.jsonl"
    assert len({item.id for item in items}) == 8
    assert all(item.question and item.answer and item.source_url and item.note for item in items)
    assert all(item.contexts and all(context for context in item.contexts) for item in items)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.7, True), (0.69, False), (None, None), (float("nan"), None)],
)
def test_score_to_prediction(score, expected):
    assert score_to_prediction(score, 0.7) is expected


def test_evaluate_outcome_passes_when_prediction_matches_label():
    item = load_canary()[0]
    claim = ClaimVerdict(statement="s", verdict=True, reason="r")

    outcome = evaluate_outcome(item, FaithfulnessResult(score=0.9, claims=(claim,)), 0.7)

    assert outcome == CanaryOutcome(
        id=item.id,
        expected_faithful=True,
        score=0.9,
        predicted_faithful=True,
        passed=True,
        claims=(claim,),
    )


def test_evaluate_outcome_fails_when_prediction_flips_or_is_indeterminate():
    faithful = load_canary()[0]

    flipped = evaluate_outcome(faithful, FaithfulnessResult(score=0.1), 0.7)
    indeterminate = evaluate_outcome(faithful, FaithfulnessResult(score=float("nan")), 0.7)

    assert flipped.passed is False
    assert indeterminate.predicted_faithful is None
    assert indeterminate.passed is False


@pytest.mark.asyncio
async def test_run_canary_all_pass_with_deterministic_scorer():
    items = load_canary()

    async def scorer(question, answer, contexts):
        item = next(i for i in items if i.question == question and i.answer == answer)
        return FaithfulnessResult(score=0.9 if item.expected_faithful else 0.1)

    outcomes = await run_canary(items, scorer, 0.7)

    assert all(outcome.passed for outcome in outcomes)


@pytest.mark.asyncio
async def test_run_canary_report_detects_drift_id():
    items = load_canary()
    drift_id = items[0].id

    async def scorer(question, answer, contexts):
        item = next(i for i in items if i.question == question and i.answer == answer)
        if item.id == drift_id:
            return FaithfulnessResult(score=0.1)
        return FaithfulnessResult(score=0.9 if item.expected_faithful else 0.1)

    outcomes = await run_canary(items, scorer, 0.7)
    report = canary_report(outcomes, {"ragas_version_pinned": True}, 0.7)

    assert report["summary"]["failed"] == 1
    assert report["summary"]["drift_ids"] == [drift_id]


def test_canary_report_is_json_safe_and_maps_nan_to_none():
    claim = ClaimVerdict(statement="s", verdict=False, reason="r")
    outcomes = [
        CanaryOutcome(
            id="id",
            expected_faithful=True,
            score=math.nan,
            predicted_faithful=None,
            passed=False,
            claims=(claim,),
        )
    ]

    report = canary_report(outcomes, {"ragas_version_pinned": True}, 0.7)

    assert report["items"][0]["score"] is None
    assert report["items"][0]["claims"] == [{"statement": "s", "verdict": False, "reason": "r"}]
    json.dumps(report, allow_nan=False)


def test_ragas_private_faithfulness_seam_is_still_available():
    _ensure_ragas_importable()
    from ragas.metrics import Faithfulness
    from ragas.metrics._faithfulness import StatementFaithfulnessAnswer

    metric = Faithfulness()
    assert callable(metric._create_statements)
    assert callable(metric._create_verdicts)
    assert callable(metric._compute_score)
    fields = StatementFaithfulnessAnswer.model_fields
    assert {"statement", "reason", "verdict"}.issubset(fields)
