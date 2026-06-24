"""Fit and pin the faithfulness gate threshold from human labels (ADR-0018).

LIVE entry point (needs Azure access). The online gate compares a RAGAS
faithfulness score against a single threshold (``guardrail.decide_next``); the
default 0.7 was never fit to human grounding judgements. This script closes that
gap:

1. Load a human-labeled set: JSONL of ``{id, query, answer, contexts, label}``
   where ``label`` is ``faithful`` / ``unfaithful`` (a *grounding* label).
2. Score each item with the live online gate judge (``JUDGE_MODEL``).
3. Sweep thresholds, tracking **false-pass and false-abstain rates separately**
   (``ragpipe.calibration``), and recommend an operating point — by default the
   one with the best label separation, or, with ``--max-false-pass``, the
   strictest gate that stays within a false-pass budget.
4. Write the pinned artifact ``data/faithfulness_calibration.json`` recording the
   RAGAS version, judge id, chosen threshold, and both error rates, so the gate's
   operating point is reproducible and re-anchors deliberately on a judge/RAGAS
   bump.

Replace ``data/faithfulness_calibration_set.example.jsonl`` with a real
human-labeled Azure-docs set before trusting the numbers.
"""
import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ragpipe.calibration import LabeledScore, recommend_threshold, sweep_thresholds
from ragpipe.config import Settings

DEFAULT_LABELED = "data/faithfulness_calibration_set.example.jsonl"
DEFAULT_OUT = "data/faithfulness_calibration.json"


def _load_labeled(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _print_sweep(sweep) -> None:
    print(f"{'threshold':>9}  {'false_pass':>10}  {'false_abstain':>13}  {'youden_J':>8}")
    for c in sweep:
        print(
            f"{c.threshold:>9.3f}  {c.false_pass_rate:>10.3f}  "
            f"{c.false_abstain_rate:>13.3f}  {c.youden_j:>8.3f}"
        )


def main() -> int:  # pragma: no cover - live judge
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default=DEFAULT_LABELED)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--max-false-pass",
        type=float,
        default=None,
        help="Optional false-pass-rate budget; recommend the strictest gate within it.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    rows = _load_labeled(args.labeled)
    print(f"calibrating on {len(rows)} labeled items from {args.labeled}", flush=True)

    from ragpipe.guardrail import build_ragas_faithfulness

    metric_fn = build_ragas_faithfulness(settings)

    labeled: list[LabeledScore] = []
    for row in rows:
        score = asyncio.run(
            metric_fn(
                question=row["query"], answer=row["answer"], contexts=list(row["contexts"])
            )
        )
        if not (isinstance(score, (int, float)) and math.isfinite(score)):
            print(f"  dropping {row['id']}: judge returned no usable score", flush=True)
            continue
        labeled.append(LabeledScore(id=row["id"], score=float(score), label=row["label"]))
        print(f"  {row['id']}: score={score:.3f} label={row['label']}", flush=True)

    if not labeled:
        print("no usable judge scores; aborting", file=sys.stderr)
        return 1

    sweep = sweep_thresholds(labeled)
    _print_sweep(sweep)
    rec = recommend_threshold(labeled, max_false_pass_rate=args.max_false_pass)
    print(f"\nrecommended threshold: {rec.threshold:.3f} — {rec.rationale}", flush=True)

    artifact = {
        "status": "calibrated",
        "threshold": round(rec.threshold, 4),
        "ragas_version": version("ragas"),
        "online_judge": settings.judge_model,
        "online_judge_model_env": "JUDGE_MODEL",
        "offline_judge_model_env": "OFFLINE_JUDGE_MODEL",
        "calibration_set": args.labeled,
        "n_labeled": len(labeled),
        "false_pass_rate": round(rec.confusion.false_pass_rate, 4),
        "false_abstain_rate": round(rec.confusion.false_abstain_rate, 4),
        "max_false_pass_rate_budget": args.max_false_pass,
        "youden_j": round(rec.confusion.youden_j, 4),
        "rationale": rec.rationale,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(
        "\nNote: set FAITHFULNESS_THRESHOLD to this value in .env so the live gate "
        "runs at the calibrated point.",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
