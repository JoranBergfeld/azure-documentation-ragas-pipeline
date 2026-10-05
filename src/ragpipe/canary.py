"""Frozen drift canary for the faithfulness judges (ADR-0018).

Nothing in the pipeline watches the LLM judges for drift. The online gate (Claude)
and the offline RAGAS judge (DeepSeek) are different families read as a mutual
"consistency check" (ADR-0009), but a 0.8 from one was never calibrated to a 0.8
from the other, and either can silently shift when Azure rolls a model build, when
RAGAS changes a metric prompt, or when a deployment id is repointed. A judge that
drifts re-anchors every faithfulness number without any signal.

This module is the *measurement* half — pure, deterministic, no network. A small
**frozen** set of obviously-faithful and obviously-unfaithful (query, answer,
contexts) triples with known labels is re-scored on a schedule
(``scripts/faithfulness_canary.py`` + a scheduled workflow). ``evaluate_canary``
turns the two judges' scores into a drift verdict:

- **label drift** — a judge gets a *known-obvious* canary item wrong (an unfaithful
  answer scores at/above the gate threshold, or a faithful one below it). The
  canary set is deliberately easy, so any mislabel is a regression, not a hard call.
- **cross-family drift** — the two judges' scalar scores diverge by more than a
  tolerance, i.e. the consistency-check assumption (ADR-0009) has decayed.
- **judge outage** — a judge can't score a canary item at all (missing score).
  Treated as drift: fail-closed, same as the live gate.

Per-claim verdicts (``score_with_claims``) are captured alongside so a drift can be
read at the level of *which* decomposed claim flipped, not just the scalar
(uncalibrated scalars hide that). Extraction reproduces RAGAS's own two-call
faithfulness pass against the pinned ``ragas==0.4.3`` internals — the version pin
(ADR-0018) is what makes touching those internals safe.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ragpipe.calibration import FAITHFUL, UNFAITHFUL, Label

# Default tolerance on |online - offline| before the two families are considered
# to have diverged. The judges are different model families, so small spreads are
# expected; this flags a *systematic* gap, not per-item noise.
DEFAULT_DISAGREEMENT_TOLERANCE = 0.25


def _finite_score(value: float | None) -> float | None:
    """A judge score is usable only if it is a real, finite number; NaN/inf/None
    all collapse to None so they are treated as a missing score (fail-closed)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class ClaimVerdict:
    """One decomposed claim and whether the judge found it grounded in context."""

    claim: str
    faithful: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "faithful": self.faithful, "reason": self.reason}


def _coerce_verdict(value: Any) -> bool:
    """Normalise RAGAS's 0/1 verdict (or a stringy/boolean equivalent) to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"cannot interpret faithfulness verdict: {value!r}")
        return value >= 1
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "1.0", "true", "yes", "faithful", "supported"}:
            return True
        if token in {"0", "0.0", "false", "no", "unfaithful", "unsupported", ""}:
            return False
    raise ValueError(f"cannot interpret faithfulness verdict: {value!r}")


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute or mapping key from ``names``."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def parse_claim_verdicts(raw: Iterable[Any]) -> list[ClaimVerdict]:
    """Normalise RAGAS faithfulness statement-verdicts into ``ClaimVerdict``s.

    Accepts RAGAS ``StatementFaithfulnessAnswer`` objects (attributes
    ``statement`` / ``verdict`` / ``reason``) or plain dicts with the same keys
    (also tolerates ``claim`` for the statement text). This is the boundary that
    keeps per-claim logging independent of RAGAS's internal class names.
    """
    out: list[ClaimVerdict] = []
    for entry in raw:
        claim = _field(entry, "statement", "claim", default="")
        verdict = _field(entry, "verdict", "faithful")
        reason = _field(entry, "reason", default="") or ""
        out.append(
            ClaimVerdict(
                claim=str(claim),
                faithful=_coerce_verdict(verdict),
                reason=str(reason),
            )
        )
    return out


@dataclass(frozen=True)
class CanaryItem:
    """A frozen, known-label canary triple. ``expected_label`` is a grounding label."""

    id: str
    query: str
    answer: str
    contexts: list[str]
    expected_label: Label
    note: str = ""

    @property
    def expected_pass(self) -> bool:
        """A faithful answer should pass the gate; an unfaithful one should abstain."""
        return self.expected_label == FAITHFUL

    def __post_init__(self) -> None:
        if self.expected_label not in (FAITHFUL, UNFAITHFUL):
            raise ValueError(
                f"expected_label must be {FAITHFUL!r}/{UNFAITHFUL!r}, "
                f"got {self.expected_label!r}"
            )


def load_canary_items(path: str | Path) -> list[CanaryItem]:
    """Load the frozen canary set from a JSONL file (one item per line)."""
    items: list[CanaryItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(
            CanaryItem(
                id=row["id"],
                query=row["query"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                expected_label=row["expected_label"],
                note=row.get("note", ""),
            )
        )
    return items


@dataclass(frozen=True)
class CanaryResult:
    """Both judges' outcome on one canary item at the gate threshold."""

    id: str
    expected_label: Label
    online_score: float | None
    offline_score: float | None
    online_pass: bool | None
    offline_pass: bool | None
    online_correct: bool | None
    offline_correct: bool | None
    disagreement: float | None  # |online - offline| when both scored
    decision_disagree: bool | None  # judges land on different pass/abstain calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expected_label": self.expected_label,
            "online_score": self.online_score,
            "offline_score": self.offline_score,
            "online_pass": self.online_pass,
            "offline_pass": self.offline_pass,
            "online_correct": self.online_correct,
            "offline_correct": self.offline_correct,
            "disagreement": self.disagreement,
            "decision_disagree": self.decision_disagree,
        }


@dataclass(frozen=True)
class CanaryReport:
    threshold: float
    tolerance: float
    results: list[CanaryResult]
    online_mismatches: int
    offline_mismatches: int
    score_disagreements: int  # |online-offline| > tolerance
    decision_disagreements: int  # different pass/abstain calls
    missing_scores: int  # a judge failed to score a canary item
    mean_disagreement: float | None
    max_disagreement: float | None
    drifted: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "drifted": self.drifted,
            "reasons": self.reasons,
            "summary": {
                "n_items": len(self.results),
                "online_mismatches": self.online_mismatches,
                "offline_mismatches": self.offline_mismatches,
                "score_disagreements": self.score_disagreements,
                "decision_disagreements": self.decision_disagreements,
                "missing_scores": self.missing_scores,
                "mean_disagreement": self.mean_disagreement,
                "max_disagreement": self.max_disagreement,
            },
            "results": [r.to_dict() for r in self.results],
        }


def evaluate_canary(
    items: Iterable[CanaryItem],
    online_scores: Mapping[str, float | None],
    offline_scores: Mapping[str, float | None],
    *,
    threshold: float,
    tolerance: float = DEFAULT_DISAGREEMENT_TOLERANCE,
) -> CanaryReport:
    """Turn the two judges' canary scores into a drift verdict.

    ``online_scores`` / ``offline_scores`` map canary item id -> score (or None
    when that judge failed to score the item). Drift is declared when any judge
    mislabels a known-obvious item, when the two families' scores diverge beyond
    ``tolerance`` on any item, or when any canary item is left unscored — all
    fail-closed, matching the live gate's posture (ADR-0009).
    """
    results: list[CanaryResult] = []
    online_mismatch = offline_mismatch = 0
    score_disagree = decision_disagree = missing = 0
    disagreements: list[float] = []

    for item in items:
        on = _finite_score(online_scores.get(item.id))
        off = _finite_score(offline_scores.get(item.id))

        on_pass = None if on is None else on >= threshold
        off_pass = None if off is None else off >= threshold
        on_correct = None if on_pass is None else on_pass == item.expected_pass
        off_correct = None if off_pass is None else off_pass == item.expected_pass

        if on is None or off is None:
            missing += 1
        if on_correct is False:
            online_mismatch += 1
        if off_correct is False:
            offline_mismatch += 1

        disagreement = None
        decision_disag = None
        if on is not None and off is not None:
            disagreement = abs(on - off)
            disagreements.append(disagreement)
            if disagreement > tolerance:
                score_disagree += 1
            decision_disag = on_pass != off_pass
            if decision_disag:
                decision_disagree += 1

        results.append(
            CanaryResult(
                id=item.id,
                expected_label=item.expected_label,
                online_score=on,
                offline_score=off,
                online_pass=on_pass,
                offline_pass=off_pass,
                online_correct=on_correct,
                offline_correct=off_correct,
                disagreement=disagreement,
                decision_disagree=decision_disag,
            )
        )

    reasons: list[str] = []
    if online_mismatch:
        reasons.append(
            f"online judge mislabeled {online_mismatch} known-obvious canary item(s)"
        )
    if offline_mismatch:
        reasons.append(
            f"offline judge mislabeled {offline_mismatch} known-obvious canary item(s)"
        )
    if score_disagree:
        reasons.append(
            f"{score_disagree} item(s) exceed the cross-family score tolerance "
            f"{tolerance:.2f}"
        )
    if missing:
        reasons.append(f"{missing} canary item(s) left unscored by a judge")

    return CanaryReport(
        threshold=threshold,
        tolerance=tolerance,
        results=results,
        online_mismatches=online_mismatch,
        offline_mismatches=offline_mismatch,
        score_disagreements=score_disagree,
        decision_disagreements=decision_disagree,
        missing_scores=missing,
        mean_disagreement=(sum(disagreements) / len(disagreements)) if disagreements else None,
        max_disagreement=max(disagreements) if disagreements else None,
        drifted=bool(online_mismatch or offline_mismatch or score_disagree or missing),
        reasons=reasons,
    )


@dataclass(frozen=True)
class ScoredClaims:
    """A faithfulness score plus the per-claim verdicts behind it."""

    score: float
    claims: list[ClaimVerdict]


async def score_with_claims(metric, sample) -> ScoredClaims:  # pragma: no cover - live judge
    """Score a sample and capture the decomposed per-claim verdicts.

    Reproduces RAGAS 0.4.3 ``Faithfulness._ascore`` (decompose the answer into
    statements, NLI each against the context) but keeps the intermediate verdicts
    instead of discarding them — the same two judge calls, no extra cost. Relies
    on pinned ``ragas==0.4.3`` internals (``_create_statements`` /
    ``_create_verdicts`` / ``_compute_score``); the version pin (ADR-0018) is what
    makes that safe. On any failure it falls back to the public scalar path and
    returns empty claims rather than breaking the canary.
    """
    import math

    row = sample.to_dict()
    try:
        statements = (await metric._create_statements(row, None)).statements
        if not statements:
            return ScoredClaims(score=math.nan, claims=[])
        verdicts = await metric._create_verdicts(row, statements, None)
        score = float(metric._compute_score(verdicts))
        return ScoredClaims(score=score, claims=parse_claim_verdicts(verdicts.statements))
    except Exception:
        score = float(await metric.single_turn_ascore(sample))
        return ScoredClaims(score=score, claims=[])
