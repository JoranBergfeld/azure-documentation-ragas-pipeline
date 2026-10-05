"""Calibrate the faithfulness gate threshold against human labels (ADR-0018).

The online gate's accept/reject decision is a single comparison
``score >= threshold`` (``guardrail.decide_next``). A fixed 0.7 borrowed from the
RAGAS defaults is *uncalibrated*: it was never fit to human judgements of which
answers are actually grounded, and LLM-judge faithfulness scores are uncalibrated
and only weakly human-correlated, so one scalar mis-gates in both directions.

This module is the *measurement* half — pure, deterministic, no network. Given a
human-labeled set of (score, label) pairs it sweeps candidate thresholds and
reports the two error rates **separately**, because they are not interchangeable:

- **false-pass**  — an *unfaithful* answer scores ``>= threshold`` and is shown to
  the user. This is the dangerous error the gate exists to prevent.
- **false-abstain** — a *faithful* answer scores ``< threshold`` and is suppressed
  behind the directive abstention. This is the cost of being strict.

Lowering the threshold trades false-abstains for false-passes and vice versa, so a
single accuracy scalar hides the trade-off operators actually care about. The live
half (``scripts/calibrate_threshold.py``) scores a labeled set with the online
judge and feeds the results here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

# A human label for a (query, answer, contexts) triple: is the answer grounded in
# the retrieved context? This is a *grounding* label, not a correctness label —
# faithfulness measures grounding, not whether the grounded claim is true
# (ADR-0018).
Label = Literal["faithful", "unfaithful"]
FAITHFUL: Label = "faithful"
UNFAITHFUL: Label = "unfaithful"


@dataclass(frozen=True)
class LabeledScore:
    """One human-labeled item plus the judge score it received."""

    id: str
    score: float
    label: Label

    def __post_init__(self) -> None:
        if self.label not in (FAITHFUL, UNFAITHFUL):
            raise ValueError(
                f"label must be {FAITHFUL!r} or {UNFAITHFUL!r}, got {self.label!r}"
            )
        if not (isinstance(self.score, (int, float)) and math.isfinite(self.score)):
            raise ValueError(f"score must be a finite number, got {self.score!r}")


@dataclass(frozen=True)
class Confusion:
    """Gate outcomes at one threshold, split by the two error directions.

    A faithful item that scores >= threshold is a true-pass; a faithful item
    below it is a false-abstain. An unfaithful item >= threshold is a false-pass;
    below it, a true-abstain.
    """

    threshold: float
    true_pass: int
    false_abstain: int
    false_pass: int
    true_abstain: int

    @property
    def n_faithful(self) -> int:
        return self.true_pass + self.false_abstain

    @property
    def n_unfaithful(self) -> int:
        return self.false_pass + self.true_abstain

    @property
    def false_pass_rate(self) -> float:
        """Share of unfaithful answers wrongly accepted (the dangerous error)."""
        return self.false_pass / self.n_unfaithful if self.n_unfaithful else 0.0

    @property
    def false_abstain_rate(self) -> float:
        """Share of faithful answers wrongly suppressed (the strictness cost)."""
        return self.false_abstain / self.n_faithful if self.n_faithful else 0.0

    @property
    def youden_j(self) -> float:
        """Sensitivity + specificity - 1; 0 = chance, 1 = perfect separation.

        Threshold-choice statistic that weights the two error rates equally and
        is insensitive to the faithful/unfaithful class balance of the labeled
        set (unlike plain accuracy).
        """
        sensitivity = 1.0 - self.false_abstain_rate  # true-pass rate
        specificity = 1.0 - self.false_pass_rate  # true-abstain rate
        return sensitivity + specificity - 1.0


def confusion_at(labeled: Iterable[LabeledScore], threshold: float) -> Confusion:
    """Tally pass/abstain outcomes at ``threshold`` using the gate's own rule.

    Uses ``score >= threshold`` so the tally matches ``guardrail.decide_next``
    exactly (a missing/None score never passes there; callers must drop judge
    failures before calibrating — a NaN is not a label).
    """
    tp = fa = fp = ta = 0
    for item in labeled:
        passes = item.score >= threshold
        if item.label == FAITHFUL:
            if passes:
                tp += 1
            else:
                fa += 1
        else:
            if passes:
                fp += 1
            else:
                ta += 1
    return Confusion(
        threshold=threshold,
        true_pass=tp,
        false_abstain=fa,
        false_pass=fp,
        true_abstain=ta,
    )


def candidate_thresholds(labeled: Iterable[LabeledScore]) -> list[float]:
    """Thresholds worth evaluating: the midpoints between adjacent observed scores.

    Every threshold between two adjacent distinct scores yields the same
    confusion matrix, so only the boundaries matter. Midpoints (plus 0.0 and a
    just-above-max point) enumerate every reachable operating point without an
    arbitrary fixed grid.
    """
    scores = sorted({round(item.score, 6) for item in labeled})
    if not scores:
        return [0.0]
    points = [0.0]
    for lo, hi in zip(scores, scores[1:]):
        points.append((lo + hi) / 2.0)
    # One point strictly above the max score makes "abstain on everything"
    # reachable, so the sweep can express a fully conservative gate.
    points.append(scores[-1] + 1e-6)
    return points


def sweep_thresholds(
    labeled: Iterable[LabeledScore],
    thresholds: Iterable[float] | None = None,
) -> list[Confusion]:
    """Confusion matrix at each candidate threshold, ascending by threshold."""
    items = list(labeled)
    points = list(thresholds) if thresholds is not None else candidate_thresholds(items)
    return [confusion_at(items, t) for t in sorted(set(points))]


@dataclass(frozen=True)
class Recommendation:
    """A chosen operating point plus why it was chosen and what it costs."""

    threshold: float
    confusion: Confusion
    rationale: str


def recommend_threshold(
    labeled: Iterable[LabeledScore],
    *,
    max_false_pass_rate: float | None = None,
    thresholds: Iterable[float] | None = None,
) -> Recommendation:
    """Pick the threshold that best separates the labels, with an optional cap.

    The gate's job is to keep unfaithful answers out, so the default objective is
    to maximise Youden's J (balanced separation). When ``max_false_pass_rate`` is
    set the search is constrained to operating points at or under that false-pass
    budget and, among those, picks the one with the lowest false-abstain rate —
    i.e. "let through as few unfaithful answers as the budget allows, then abstain
    on as few faithful answers as possible". Ties break toward the *higher*
    threshold (the more conservative gate). Raises if no operating point meets the
    budget.
    """
    items = list(labeled)
    if not items:
        raise ValueError("cannot recommend a threshold from an empty labeled set")
    sweep = sweep_thresholds(items, thresholds)

    if max_false_pass_rate is None:
        best = max(sweep, key=lambda c: (c.youden_j, c.threshold))
        return Recommendation(
            threshold=best.threshold,
            confusion=best,
            rationale=(
                f"max Youden's J ({best.youden_j:.3f}); "
                f"false-pass {best.false_pass_rate:.3f}, "
                f"false-abstain {best.false_abstain_rate:.3f}"
            ),
        )

    feasible = [c for c in sweep if c.false_pass_rate <= max_false_pass_rate + 1e-9]
    if not feasible:
        best_fp = min(c.false_pass_rate for c in sweep)
        raise ValueError(
            f"no threshold meets false-pass budget {max_false_pass_rate:.3f}; "
            f"the lowest achievable false-pass rate is {best_fp:.3f}"
        )
    best = min(feasible, key=lambda c: (c.false_abstain_rate, -c.threshold))
    return Recommendation(
        threshold=best.threshold,
        confusion=best,
        rationale=(
            f"lowest false-abstain ({best.false_abstain_rate:.3f}) within "
            f"false-pass budget {max_false_pass_rate:.3f} "
            f"(achieved false-pass {best.false_pass_rate:.3f})"
        ),
    )
