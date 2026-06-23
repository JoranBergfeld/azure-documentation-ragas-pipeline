# 0018 — Eval confidence intervals and significance

**Status:** Accepted (2026-06-23)

## Context

ADR-0016 made per-mode means the primary cross-mode comparison signal, led by
LLM-free URL-match retrieval metrics. That signal was too easy to over-read: the
current committed test set has 33 items, and tag cohorts are much smaller, so many
mode-to-mode differences are within sampling noise. The issue-9 scope is the
statistics and reporting methodology applied to the current test set only. Growing
or diversifying `data/testset.jsonl` is deliberately left to companion issue #6 so
two changes do not conflict on the same test-set file.

## Decision

Report uncertainty with the existing means instead of replacing the eval axis.
Each per-mode metric now includes a seeded percentile bootstrap confidence interval
for its finite per-item scores under `means_ci`. The combined `eval_results.json`
also includes `comparisons`: paired bootstrap mean-difference intervals and a
two-sided bootstrap p-value for each treatment mode versus `baseline`, paired by
record index because every mode runs the same test set in the same order.

A cross-mode difference whose paired confidence interval overlaps zero is reported
as **no measurable difference**. Per-mode intervals should be read the same way:
overlapping intervals mean the apparent mean separation is not strong evidence of a
mode difference on the current test set. This amends ADR-0016's “means are the
primary cross-mode signal” by attaching uncertainty to that signal.

## Consequences

- `eval_results_<mode>.json` grows with a `means_ci` object per metric; the derived
  combined `eval_results.json` grows with top-level `comparisons`.
- The dashboard can surface practical verdicts instead of inviting readers to rank
  nine modes by small mean differences alone.
- Results remain deterministic and hermetic: bootstrap resampling uses fixed seeds
  and only finite metric values, matching the harness aggregation semantics.
- The current n=33 test set still limits power. Non-significant results should not
  be treated as proof of equality; they mean this test set cannot measure a clear
  difference. Test-set growth remains issue #6.

## Sources

- Saad-Falcon et al., ARES, NAACL 2024 — https://arxiv.org/abs/2311.09476
- Smucker, Allan & Carterette, *A Comparison of Statistical Significance Tests for
  IR Evaluation*, CIKM 2007 — DOI 10.1145/1321440.1321528
- Thakur et al., BEIR — https://arxiv.org/abs/2104.08663
- Friel et al., RAGBench — https://arxiv.org/abs/2407.11005
- Ru et al., RAGChecker — https://arxiv.org/abs/2408.08067
- ADR-0016 — Multi-mode evaluation axis
