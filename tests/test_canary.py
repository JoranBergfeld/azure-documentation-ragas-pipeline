from __future__ import annotations

from pathlib import Path

import pytest

from ragpipe.canary import (
    CanaryItem,
    evaluate_canary,
    load_canary_items,
    parse_claim_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANARY_FILE = REPO_ROOT / "data" / "faithfulness_canary.jsonl"


class _Stmt:
    """Stand-in for RAGAS StatementFaithfulnessAnswer (attribute access)."""

    def __init__(self, statement, verdict, reason=""):
        self.statement = statement
        self.verdict = verdict
        self.reason = reason


def test_parse_claim_verdicts_from_ragas_like_objects():
    claims = parse_claim_verdicts(
        [_Stmt("The sky is blue.", 1, "stated"), _Stmt("The sky is green.", 0, "contradicted")]
    )
    assert [c.faithful for c in claims] == [True, False]
    assert claims[0].claim == "The sky is blue."
    assert claims[1].reason == "contradicted"


def test_parse_claim_verdicts_from_dicts_and_coerces_verdict_forms():
    claims = parse_claim_verdicts(
        [
            {"claim": "a", "verdict": "1"},
            {"statement": "b", "verdict": False},
            {"statement": "c", "verdict": "yes"},
        ]
    )
    assert [c.faithful for c in claims] == [True, False, True]


def test_parse_claim_verdicts_rejects_uninterpretable_verdict():
    with pytest.raises(ValueError, match="cannot interpret"):
        parse_claim_verdicts([{"statement": "x", "verdict": "perhaps"}])


def _items():
    return [
        CanaryItem("f", "q", "a", ["ctx"], "faithful"),
        CanaryItem("u", "q", "a", ["ctx"], "unfaithful"),
    ]


def test_no_drift_when_both_judges_are_correct_and_agree():
    report = evaluate_canary(
        _items(),
        online_scores={"f": 0.9, "u": 0.2},
        offline_scores={"f": 0.85, "u": 0.3},
        threshold=0.7,
    )
    assert report.drifted is False
    assert report.reasons == []
    assert report.online_mismatches == 0 and report.offline_mismatches == 0
    assert report.mean_disagreement == pytest.approx((0.05 + 0.1) / 2)


def test_label_drift_when_a_judge_mislabels_an_obvious_item():
    report = evaluate_canary(
        _items(),
        online_scores={"f": 0.5, "u": 0.2},  # faithful item wrongly abstained
        offline_scores={"f": 0.85, "u": 0.3},
        threshold=0.7,
    )
    assert report.drifted is True
    assert report.online_mismatches == 1
    assert any("online judge mislabeled" in r for r in report.reasons)


def test_cross_family_drift_without_a_label_flip():
    # Both judges abstain on the unfaithful item (both correct), but their scores
    # diverge past the tolerance -> the consistency-check assumption has decayed.
    report = evaluate_canary(
        _items(),
        online_scores={"f": 0.9, "u": 0.1},
        offline_scores={"f": 0.85, "u": 0.5},
        threshold=0.7,
        tolerance=0.25,
    )
    assert report.online_mismatches == 0 and report.offline_mismatches == 0
    assert report.score_disagreements == 1
    assert report.drifted is True


def test_missing_score_is_treated_as_drift_fail_closed():
    report = evaluate_canary(
        _items(),
        online_scores={"u": 0.2},  # judge failed to score the faithful item
        offline_scores={"f": 0.85, "u": 0.3},
        threshold=0.7,
    )
    assert report.missing_scores == 1
    assert report.drifted is True
    assert report.results[0].online_score is None


def test_nan_score_is_treated_as_missing_not_a_silent_abstain():
    # A judge returning NaN on an unfaithful item must not be read as a correct
    # abstain — fail-closed, same as a missing score.
    report = evaluate_canary(
        _items(),
        online_scores={"f": 0.9, "u": float("nan")},
        offline_scores={"f": 0.85, "u": 0.3},
        threshold=0.7,
    )
    assert report.missing_scores == 1
    assert report.drifted is True
    assert report.results[1].online_score is None


def test_report_to_dict_is_json_serializable_shape():
    report = evaluate_canary(
        _items(),
        online_scores={"f": 0.9, "u": 0.2},
        offline_scores={"f": 0.85, "u": 0.3},
        threshold=0.7,
    )
    d = report.to_dict()
    assert d["summary"]["n_items"] == 2
    assert {"threshold", "drifted", "reasons", "summary", "results"} <= set(d)


def test_frozen_canary_file_loads_and_is_balanced():
    items = load_canary_items(CANARY_FILE)
    assert len(items) >= 6
    labels = [i.expected_label for i in items]
    assert "faithful" in labels and "unfaithful" in labels
    # Every item carries non-empty contexts so a judge can actually score it.
    assert all(i.contexts for i in items)
