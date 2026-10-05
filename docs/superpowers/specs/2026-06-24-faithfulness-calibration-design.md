# Calibrate + pin + watch the faithfulness gate

**Date:** 2026-06-24
**Status:** Approved
**Origin:** Issue #10 (pipeline + literature review). Hardens the online faithfulness
gate established in ADR-0009 / ADR-0011.

## Problem

The online guardrail accepts an answer when its RAGAS-faithfulness score clears a
**fixed 0.7 threshold** (`guardrail.decide_next`, `PipelineDeps.threshold`). Three
gaps make that scalar fragile:

1. **Uncalibrated, single-error threshold.** 0.7 was inherited from RAGAS
   conventions, never fit to human grounding labels. The gate makes two opposite
   errors — false-pass (an unfaithful answer shown to the user) and false-abstain
   (a faithful answer suppressed) — and a single scalar hides that trade-off. The
   LLM-judge literature shows zero-shot faithfulness judges are uncalibrated and
   near ~50% on hard cases, so a fixed point mis-gates both ways.
2. **No drift watch.** The Claude online gate and DeepSeek offline judge are read
   as a mutual consistency check (ADR-0009) but were never calibrated to each
   other, and nothing notices when either re-anchors (Azure model-build roll,
   deployment repoint, RAGAS prompt change).
3. **Nothing pinned / decomposed.** `ragas` was unpinned (scores aren't comparable
   across versions); only the scalar was logged, never the per-claim verdicts it
   is computed from.

Faithfulness also measures **grounding, not correctness** — a faithful summary of
wrong-but-retrieved context passes — and that was undocumented.

## Goals

- Pin `ragas` exactly so the gate's operating point is meaningful.
- Provide pure, tested calibration machinery that tracks **false-pass and
  false-abstain separately** and recommends a threshold (optionally under a
  false-pass budget), plus a live script that fits it from human labels and writes
  a pinned artifact.
- Provide a **frozen drift canary** re-scored by both judges on a schedule that
  fails closed on label drift, cross-family divergence, or a judge outage, and
  logs per-claim verdicts.
- Document faithfulness as a grounding heuristic, not a correctness oracle.

## Non-goals

- Changing the live gate decision (`decide_next` stays a scalar comparison).
- Feeding per-claim verdicts back into the gate (logged for diagnosis only).
- Swapping the primary gate to a non-LLM classifier (HHEM) — recorded as the
  intended secondary check, not adopted here.
- Producing real human labels (the example calibration set is a schema seed).

## Design

Two pure modules carry the logic; live wiring stays `# pragma: no cover`.

- `ragpipe.calibration` — `LabeledScore`, `confusion_at` (splits the two error
  directions), `sweep_thresholds`, `recommend_threshold` (max Youden's J, or
  strictest gate within a `max_false_pass_rate` budget).
- `ragpipe.canary` — `CanaryItem` / `load_canary_items`, `evaluate_canary`
  (fail-closed drift verdict from online vs. offline scores against known labels),
  `parse_claim_verdicts` + `score_with_claims` (reproduce RAGAS's two-call
  faithfulness pass, keep the verdicts).

Frozen artifacts: `data/faithfulness_canary.jsonl` (8 obvious items),
`data/faithfulness_calibration.json` (pinned versions/threshold; ships
`uncalibrated-default`), `data/faithfulness_calibration_set.example.jsonl`.

Live: `scripts/calibrate_threshold.py`, `scripts/faithfulness_canary.py`, and a
scheduled `.github/workflows/faithfulness-canary.yml` gated behind `CANARY_ENABLED`.

**Spec → decision:** `docs/adr/0018-calibrated-faithfulness-gate.md`.
