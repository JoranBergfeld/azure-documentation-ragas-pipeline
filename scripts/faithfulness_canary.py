"""Re-score the frozen faithfulness canary with both judges and report drift.

LIVE entry point (needs Azure access), meant to run on a schedule (see
``.github/workflows/faithfulness-canary.yml``). It scores
``data/faithfulness_canary.jsonl`` with the online gate judge (``JUDGE_MODEL``,
Claude) and the offline RAGAS judge (``OFFLINE_JUDGE_MODEL``, DeepSeek), compares
them with ``ragpipe.canary.evaluate_canary`` against the calibrated gate
threshold, writes ``faithfulness_canary_report.json``, and **exits non-zero on
drift** so a scheduled run turns red when a judge moves.

Per-claim verdicts from the offline judge are logged per item: a drift is then
readable as *which decomposed claim flipped*, not just a scalar move. The online
gate is scored through the same code path the live pipeline uses
(``build_ragas_faithfulness``), so the canary exercises the real gate.
"""
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from ragpipe.canary import evaluate_canary, load_canary_items, score_with_claims
from ragpipe.config import Settings

CANARY_FILE = "data/faithfulness_canary.jsonl"
CALIBRATION_FILE = "data/faithfulness_calibration.json"
REPORT_FILE = "faithfulness_canary_report.json"


def _gate_threshold(settings) -> float:
    """The threshold the gate actually runs at: the pinned calibrated value if
    present, else the configured default."""
    path = Path(CALIBRATION_FILE)
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("threshold")
            if isinstance(value, (int, float)):
                return float(value)
        except (json.JSONDecodeError, OSError):
            pass
    return settings.faithfulness_threshold


def _finite_or_none(value):
    """A judge score is usable only if it is a real, finite number."""
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _run(coro):
    """Score one item, mapping any judge failure to None (fail-closed in the report)."""
    try:
        return asyncio.run(coro)
    except Exception as exc:  # noqa: BLE001 - one item failing must not abort the canary
        print(f"  judge error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None


def main() -> int:  # pragma: no cover - live judges
    settings = Settings.from_env()
    items = load_canary_items(CANARY_FILE)
    threshold = _gate_threshold(settings)
    print(f"canary: {len(items)} items, gate threshold {threshold:.3f}", flush=True)

    from ragpipe.guardrail import _ensure_ragas_importable, build_ragas_faithfulness

    _ensure_ragas_importable()
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    from ragpipe.eval.harness import _build_ragas_clients

    online_metric_fn = build_ragas_faithfulness(settings)
    offline_llm, _ = _build_ragas_clients(settings)
    offline_metric = Faithfulness(llm=offline_llm)

    online_scores: dict[str, float | None] = {}
    offline_scores: dict[str, float | None] = {}
    claims_by_id: dict[str, list[dict]] = {}

    for item in items:
        print(f"  scoring {item.id} ({item.expected_label})…", flush=True)
        online = _run(
            online_metric_fn(question=item.query, answer=item.answer, contexts=item.contexts)
        )
        online_scores[item.id] = _finite_or_none(online)

        sample = SingleTurnSample(
            user_input=item.query, response=item.answer, retrieved_contexts=item.contexts
        )
        scored = _run(score_with_claims(offline_metric, sample))
        if scored is None:
            offline_scores[item.id] = None
            claims_by_id[item.id] = []
        else:
            offline_scores[item.id] = _finite_or_none(scored.score)
            claims_by_id[item.id] = [c.to_dict() for c in scored.claims]

    report = evaluate_canary(
        items, online_scores, offline_scores, threshold=threshold
    )
    payload = report.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["online_judge"] = settings.judge_model
    payload["offline_judge"] = settings.offline_judge_model
    for result in payload["results"]:
        result["offline_claims"] = claims_by_id.get(result["id"], [])

    Path(REPORT_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {REPORT_FILE}", flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)

    if report.drifted:
        print("DRIFT DETECTED:", flush=True)
        for reason in report.reasons:
            print(f"  - {reason}", flush=True)
        return 1
    print("no drift detected", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
