from __future__ import annotations

import pytest

from ragpipe.calibration import (
    LabeledScore,
    candidate_thresholds,
    confusion_at,
    recommend_threshold,
    sweep_thresholds,
)


def _set(faithful, unfaithful):
    items = [LabeledScore(id=f"f{i}", score=s, label="faithful") for i, s in enumerate(faithful)]
    items += [LabeledScore(id=f"u{i}", score=s, label="unfaithful") for i, s in enumerate(unfaithful)]
    return items


def test_labeled_score_rejects_unknown_label():
    with pytest.raises(ValueError, match="label must be"):
        LabeledScore(id="x", score=0.5, label="maybe")  # type: ignore[arg-type]


def test_labeled_score_rejects_non_finite_score():
    with pytest.raises(ValueError, match="finite"):
        LabeledScore(id="x", score=float("nan"), label="faithful")


def test_confusion_splits_the_two_error_directions_separately():
    # faithful below threshold => false-abstain; unfaithful above => false-pass.
    labeled = _set(faithful=[0.9, 0.6], unfaithful=[0.8, 0.2])
    c = confusion_at(labeled, threshold=0.7)

    assert (c.true_pass, c.false_abstain) == (1, 1)  # 0.9 passes, 0.6 abstains
    assert (c.false_pass, c.true_abstain) == (1, 1)  # 0.8 passes, 0.2 abstains
    assert c.false_pass_rate == 0.5
    assert c.false_abstain_rate == 0.5


def test_confusion_uses_ge_so_a_score_equal_to_threshold_passes():
    c = confusion_at(_set(faithful=[0.7], unfaithful=[]), threshold=0.7)
    assert c.true_pass == 1 and c.false_abstain == 0


def test_candidate_thresholds_are_midpoints_plus_endpoints():
    labeled = _set(faithful=[0.8], unfaithful=[0.4])
    points = candidate_thresholds(labeled)
    assert points[0] == 0.0
    assert pytest.approx(0.6) == points[1]  # midpoint of 0.4 and 0.8
    assert points[-1] > 0.8  # one point above the max makes "abstain all" reachable


def test_sweep_is_sorted_and_covers_perfect_separation():
    labeled = _set(faithful=[0.9, 0.8, 0.75], unfaithful=[0.6, 0.4, 0.2])
    sweep = sweep_thresholds(labeled)
    assert [c.threshold for c in sweep] == sorted(c.threshold for c in sweep)
    # A perfectly separating threshold (0.6 < t <= 0.75) gives Youden's J == 1.
    assert any(c.youden_j == 1.0 for c in sweep)


def test_recommend_default_maximizes_youden_j():
    labeled = _set(faithful=[0.9, 0.8, 0.75], unfaithful=[0.6, 0.4, 0.2])
    rec = recommend_threshold(labeled)
    assert rec.confusion.youden_j == 1.0
    assert 0.6 < rec.threshold <= 0.75
    assert rec.confusion.false_pass_rate == 0.0
    assert rec.confusion.false_abstain_rate == 0.0


def test_recommend_with_zero_false_pass_budget_trades_for_more_false_abstains():
    # Overlapping classes: one unfaithful (0.7) outscores two faithful (0.6, 0.65).
    labeled = _set(faithful=[0.9, 0.65, 0.6], unfaithful=[0.7, 0.5, 0.3])
    rec = recommend_threshold(labeled, max_false_pass_rate=0.0)
    # The only way to admit zero unfaithful answers is to sit above 0.7, which
    # abstains on the two faithful answers that scored below it.
    assert rec.threshold > 0.7
    assert rec.confusion.false_pass_rate == 0.0
    assert rec.confusion.false_abstain_rate == pytest.approx(2 / 3)


def test_recommend_with_loose_budget_minimizes_false_abstains():
    labeled = _set(faithful=[0.9, 0.65, 0.6], unfaithful=[0.7, 0.5, 0.3])
    rec = recommend_threshold(labeled, max_false_pass_rate=0.34)
    # A threshold just under 0.6 lets every faithful answer through while only
    # admitting the single 0.7 unfaithful (false-pass 1/3, within budget).
    assert rec.confusion.false_abstain_rate == 0.0
    assert rec.confusion.false_pass_rate == pytest.approx(1 / 3)


def test_recommend_raises_when_budget_is_unreachable():
    labeled = _set(faithful=[0.9], unfaithful=[0.8])
    with pytest.raises(ValueError, match="no threshold meets"):
        recommend_threshold(labeled, max_false_pass_rate=-0.01)


def test_recommend_raises_on_empty_set():
    with pytest.raises(ValueError, match="empty"):
        recommend_threshold([])
