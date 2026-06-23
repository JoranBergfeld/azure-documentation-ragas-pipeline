from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ragpipe.config import Settings
from ragpipe.eval.judge_fingerprint import judge_fingerprint
from ragpipe.guardrail import (
    ClaimVerdict,
    FaithfulnessResult,
    build_ragas_faithfulness_detailed,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANARY_PATH = _REPO_ROOT / "data" / "faithfulness_canary.jsonl"


@dataclass(frozen=True)
class CanaryItem:
    id: str
    question: str
    answer: str
    contexts: tuple[str, ...]
    expected_faithful: bool
    source_url: str
    note: str


@dataclass(frozen=True)
class CanaryOutcome:
    id: str
    expected_faithful: bool
    score: float | None
    predicted_faithful: bool | None
    passed: bool
    claims: tuple[ClaimVerdict, ...]


def load_canary(path: Path | str = DEFAULT_CANARY_PATH) -> list[CanaryItem]:
    items: list[CanaryItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                CanaryItem(
                    id=row["id"],
                    question=row["question"],
                    answer=row["answer"],
                    contexts=tuple(row["contexts"]),
                    expected_faithful=bool(row["expected_faithful"]),
                    source_url=row["source_url"],
                    note=row["note"],
                )
            )
    return items


def score_to_prediction(score: float | None, threshold: float) -> bool | None:
    if score is None or math.isnan(score):
        return None
    return score >= threshold


def evaluate_outcome(
    item: CanaryItem, result: FaithfulnessResult, threshold: float
) -> CanaryOutcome:
    predicted = score_to_prediction(result.score, threshold)
    return CanaryOutcome(
        id=item.id,
        expected_faithful=item.expected_faithful,
        score=result.score,
        predicted_faithful=predicted,
        passed=predicted is not None and predicted == item.expected_faithful,
        claims=result.claims,
    )


ScoreDetailedFn = Callable[[str, str, list[str]], Awaitable[FaithfulnessResult]]


async def run_canary(
    items: list[CanaryItem], score_detailed_fn: ScoreDetailedFn, threshold: float
) -> list[CanaryOutcome]:
    outcomes: list[CanaryOutcome] = []
    for item in items:
        result = await score_detailed_fn(item.question, item.answer, list(item.contexts))
        outcomes.append(evaluate_outcome(item, result, threshold))
    return outcomes


def _json_score(score: float | None) -> float | None:
    if score is None or math.isnan(score):
        return None
    return score


def canary_report(outcomes: list[CanaryOutcome], fingerprint: dict, threshold: float) -> dict:
    failed = [outcome for outcome in outcomes if not outcome.passed]
    return {
        "fingerprint": fingerprint,
        "threshold": threshold,
        "summary": {
            "total": len(outcomes),
            "passed": len(outcomes) - len(failed),
            "failed": len(failed),
            "drift_ids": [outcome.id for outcome in failed],
        },
        "items": [
            {
                "id": outcome.id,
                "expected_faithful": outcome.expected_faithful,
                "score": _json_score(outcome.score),
                "predicted_faithful": outcome.predicted_faithful,
                "passed": outcome.passed,
                "claims": [asdict(claim) for claim in outcome.claims],
            }
            for outcome in outcomes
        ],
    }


def main() -> int:  # pragma: no cover - live wiring
    settings = Settings.from_env()
    metric_fn = build_ragas_faithfulness_detailed(settings)

    async def score_detailed_fn(
        question: str, answer: str, contexts: list[str]
    ) -> FaithfulnessResult:
        return await metric_fn(question=question, answer=answer, contexts=contexts)

    items = load_canary()
    outcomes = asyncio.run(run_canary(items, score_detailed_fn, settings.faithfulness_threshold))
    fingerprint = judge_fingerprint(settings)
    report = canary_report(outcomes, fingerprint, settings.faithfulness_threshold)
    output_path = _REPO_ROOT / "eval_results_canary.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    summary = report["summary"]
    print(
        f"Canary: {summary['passed']}/{summary['total']} passed; "
        f"failed={summary['failed']}"
    )
    if not fingerprint["ragas_version_pinned"]:
        print(
            f"WARNING: ragas version {fingerprint['ragas_version']} does not match "
            f"expected {fingerprint['expected_ragas_version']}"
        )
    if summary["failed"]:
        print("WARNING: drift detected for " + ", ".join(summary["drift_ids"]))
    return 0 if fingerprint["ragas_version_pinned"] and not summary["failed"] else 1


if __name__ == "__main__":  # pragma: no cover - live wiring
    raise SystemExit(main())
