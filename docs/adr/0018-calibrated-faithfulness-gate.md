# 0018 — Calibrated faithfulness gate, pinned judge versions, drift canary

**Status:** Accepted (2026-06-24)

## Context

The online guardrail (ADR-0009) accepts an answer when its RAGAS-faithfulness
score clears a **single fixed threshold of 0.7** (`PipelineDeps.threshold`;
`guardrail.decide_next`). That number was inherited from RAGAS conventions, not
fit to human judgements of which answers are actually grounded, and it sits at the
centre of three unaddressed weaknesses:

1. **Uncalibrated threshold, one scalar for two different errors.** The gate makes
   two opposite mistakes — letting an *unfaithful* answer through (false-pass, the
   dangerous one) and suppressing a *faithful* answer behind the directive
   abstention (false-abstain, the strictness cost). A single accuracy number hides
   that trade-off, and nothing in the repo fit the threshold to labels. LLM-judge
   faithfulness scores are uncalibrated and only weakly human-correlated, and hard
   hallucination cases sit near ~50% detector accuracy, so a fixed point mis-gates
   in both directions.
2. **Cross-family scores were read as a "consistency check" but never calibrated
   to each other.** The Claude online gate and the DeepSeek offline judge are
   deliberately different families (ADR-0009), but a 0.8 from one is not a 0.8
   from the other, and either can silently re-anchor when Azure rolls a model
   build, a deployment id is repointed, or a RAGAS metric prompt changes — with no
   signal.
3. **Nothing pinned RAGAS or logged the decomposition.** `ragas` was an unpinned
   dependency, and the gate logged only the scalar — not the decomposed claims and
   per-claim verdicts the score is computed from — so a drift could not be read at
   the level of *which claim flipped*.

Faithfulness also measures **grounding, not correctness**: a faithful summary of
wrong-but-retrieved context passes the gate. On GraphRAG, where the retrieved
context is measurably worse, faithfulness is 0.744 — high grounding over weak
context. The gate is a grounding heuristic, not a correctness oracle, and was not
documented as such.

## Decision

1. **Pin RAGAS exactly (`ragas==0.4.3`).** Judge scores are not comparable across
   RAGAS versions (metric prompts change; legacy metrics are slated for removal by
   v1.0), so the gate's operating point is only meaningful against a pinned metric.
   Bumping RAGAS is now a deliberate act that requires recalibrating the threshold
   and re-baselining the canary in the same change.

2. **Calibration machinery that tracks the two errors separately**
   (`ragpipe.calibration`, pure/deterministic). Given a human-labeled set of
   `(score, label)` pairs it sweeps every reachable threshold and reports
   **false-pass rate and false-abstain rate independently**, plus Youden's J
   (class-balance-insensitive). `recommend_threshold` either maximises separation
   or, given a `--max-false-pass` budget, picks the strictest gate within it. The
   live half (`scripts/calibrate_threshold.py`) scores a labeled set with the
   online judge and writes the **pinned artifact**
   `data/faithfulness_calibration.json` (RAGAS version, judge id, threshold, both
   error rates, date) so the operating point is reproducible. The committed
   artifact ships as `status: uncalibrated-default` (threshold 0.7) until a real
   human-labeled Azure-docs set replaces
   `data/faithfulness_calibration_set.example.jsonl`.

3. **A frozen drift canary** (`ragpipe.canary` + `data/faithfulness_canary.jsonl`
   + `scripts/faithfulness_canary.py` + a scheduled workflow). A small set of
   obviously-faithful and obviously-unfaithful triples with known labels is
   re-scored by **both** judges on a weekly schedule. `evaluate_canary` declares
   drift — fail-closed — when any judge mislabels a known-obvious item (regression
   vs. the calibration), when the two families' scores diverge past a tolerance
   (the consistency-check assumption has decayed), or when a judge fails to score
   an item at all. The run exits non-zero on drift so the schedule turns red.

4. **Log decomposed claims + per-claim verdicts.** `canary.score_with_claims`
   reproduces RAGAS's own two-call faithfulness pass (decompose answer into
   statements → NLI each against context) but keeps the intermediate verdicts, so
   each canary item logs *which claim was found grounded or not*, not just the
   scalar. `parse_claim_verdicts` is the version-independent boundary that
   normalises RAGAS's statement-verdict objects. Touching RAGAS internals is made
   safe by the exact version pin (decision 1).

5. **Document the grounding-vs-correctness caveat** in the README and `.env.example`:
   faithfulness gates grounding in the retrieved context, not factual correctness.

The live gate (`decide_next`) is unchanged — it still compares the scalar to the
threshold; this ADR is about *how that threshold is chosen, pinned, and watched*,
not about changing the hot path.

## Alternatives rejected

- **Keep the fixed 0.7.** Simplest, but uncalibrated and unmonitored — the exact
  gap the LLM-judge literature warns about.
- **Replace RAGAS faithfulness with a non-LLM classifier (Vectara HHEM) as the
  primary gate.** HHEM is faster and steadier and RAGAS even ships
  `FaithfulnesswithHHEM`, but swapping the primary gate re-anchors every existing
  faithfulness number and adds a transformer/model download to the hot path.
  Recorded instead as the intended **secondary check for high-risk answers** and a
  candidate canary cross-check; not adopted as the gate in this change.
- **Calibrate to a single accuracy/F1 scalar.** Collapses the false-pass vs
  false-abstain trade-off operators actually tune; we keep them separate and let a
  false-pass budget drive the choice.
- **Feed per-claim verdicts back into `decide_next`** (e.g. abstain on any
  unsupported claim). A real option, but it changes gate behaviour and belongs
  behind its own measured decision; here verdicts are logged for diagnosis only.
- **Daily canary.** Weekly catches a model-build roll without burning
  marketplace-billed judge tokens every day; `workflow_dispatch` covers ad-hoc
  checks.

## Consequences

- The gate's operating point is now a reproducible, version-pinned artifact rather
  than a magic constant; bumping RAGAS or a judge is a deliberate recalibrate +
  re-baseline step.
- A judge that drifts surfaces as a red scheduled run with a per-claim report,
  instead of silently re-anchoring faithfulness everywhere.
- Until a human-labeled Azure-docs calibration set exists, the threshold stays at
  the documented uncalibrated default (0.7) — the machinery is shipped, the labels
  are the remaining manual step.
- The canary adds a weekly paid judge run (8 items × 2 judges); it is gated behind
  `CANARY_ENABLED` so it is a no-op until configured.
- New runtime dep surface is unchanged (RAGAS already shipped scikit-learn etc.);
  only the version constraint tightened.

## Sources

- LLM-judge position/verbosity/self-preference bias and prompt sensitivity:
  Zheng et al., *Judging LLM-as-a-Judge (MT-Bench / Chatbot Arena)*, NeurIPS 2023 —
  https://arxiv.org/abs/2306.05685
- Self-preference bias specifically: Panickssery, Bowman & Feng —
  https://arxiv.org/abs/2404.13076 ; Wataoka et al. — https://arxiv.org/abs/2410.21819
- Faithfulness/hallucination detectors near ~50% on hard cases (a fixed threshold
  mis-gates): Bao et al., *FaithBench*, NAACL 2025 —
  https://aclanthology.org/2025.naacl-short.38/ ; Tamber et al., *FaithJudge*,
  EMNLP Industry 2025 — https://arxiv.org/abs/2505.04847
- RAGAS-style zero-shot judging is uncalibrated / weakly human-correlated vs.
  trained judges: Saad-Falcon et al., *ARES* — https://arxiv.org/abs/2311.09476 ;
  Ru et al., *RAGChecker* — https://arxiv.org/abs/2408.08067 ; Friel et al.,
  *RAGBench / TRACe* — https://arxiv.org/abs/2407.11005
- Non-LLM faithfulness classifier as a faster secondary check / canary: Vectara
  HHEM hallucination leaderboard — https://github.com/vectara/hallucination-leaderboard
- RAGAS metrics/APIs change across versions (legacy metrics slated for removal by
  v1.0) — pin them: https://docs.ragas.io/en/stable/references/metrics/
- Youden's J as a threshold-selection statistic: Youden, *Index for rating
  diagnostic tests*, Cancer 1950 — https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3
