# 0018 — Bootstrap confidence intervals and paired significance for the mode comparison

**Status:** Accepted (2026-06-24)

## Context

The multi-mode evaluation axis (ADR-0016) replays the same test set through up to
**9 retrieval modes** and, until now, reported only `means`, `means_by_tag`, and
`coverage`. It designates the per-mode means the "primary cross-mode signal" used to
rank modes against a frozen baseline. But the test set is small —
`data/testset.jsonl` is **33 items**, with per-tag cohorts of **7 or fewer** (a per-tag
rate like `hit_rate@reranked = 0.286` is literally 2 of 7) — and there is **no
confidence interval or significance test anywhere** in `src/ragpipe/eval/`.

At this scale the per-item variance of both the LLM-judged RAGAS metrics and the
deterministic URL-match metrics swamps most mode-to-mode mean differences, and baseline
retrieval already saturates (`hit_rate@dense` / `@fused` = 1.0), compressing the dynamic
range further. A bare difference of means is therefore not evidence of a real effect: we
were ranking 9 modes on noise.

## Decision

Report dispersion and significance alongside the means, so overlapping intervals are
read as "no measurable difference" rather than a ranking.

1. **Percentile bootstrap confidence intervals on every per-mode metric mean.** A new
   pure module `ragpipe.eval.stats` resamples each metric's per-item scores (default
   10,000 resamples, fixed seed, 95%) and reports `{mean, low, high, n}`. `run.py`
   attaches this as a `ci` map to each mode's result (and its standalone
   `eval_results_<mode>.json`). Calibrated estimation *with intervals* is the recommended
   fix for LLM-judge noise at small n (ARES).

2. **Paired randomization (sign-flip) test of each mode against the baseline.** Because
   every mode is replayed over the *same* test set in the same order, record *i* is the
   same item everywhere, so per-item metric rows line up for a paired test. For each
   metric we compute per-item differences `treatment_i - baseline_i` over jointly-valid
   pairs, estimate a two-sided p-value by randomly flipping each difference's sign (the
   exchangeable-under-H0 permutation for paired data), and report a bootstrap CI on the
   mean difference (the effect size). This is the standard significance test for IR
   evaluation. The combined `eval_results.json` gains a top-level `baseline_mode` and a
   `paired_vs_baseline` block (`{mode: {metric: {n, mean_diff, ci_low, ci_high, p_value,
   significant}}}`).

3. **Surface it where modes are compared.** The dashboard's mode-comparison view shows
   the overlapping-interval caveat, the per-mode CIs (in the drill-in table), and a
   paired-significance table vs. the baseline. `GET /eval` returns `ci`, `baselineMode`,
   and `pairedVsBaseline`.

4. **Determinism.** Seeded RNG and add-one-smoothed randomization p-values make every
   reported interval and p-value reproducible across runs, matching ADR-0016's
   reproducibility goal. The whole module is network-free and unit-tested directly.

5. **Test-set growth is the complementary live step, not part of this change.** Larger,
   heterogeneous evaluation sets are the other half of the fix (BEIR; RAGBench;
   RAGChecker). The synthetic generation path already exists (ADR-0010 /
   `TESTSET_MODE=synthetic`); growing `data/testset.jsonl` is left to that path so the
   statistics machinery — the gate this decision is about — lands first and independently
   reviewable.

## Alternatives rejected

- **Keep reporting bare means.** Cheapest, but it ranks 9 modes on differences that the
  intervals show are within noise — the exact failure this ADR fixes.
- **Unpaired bootstrap comparison between two modes' means.** Throws away the pairing.
  Since both modes answer the *same* items, a paired test removes per-item difficulty as a
  variance source and is strictly more powerful at this n.
- **Parametric paired t-test.** Assumes approximately normal per-item differences;
  bounded, saturating, often-degenerate metric differences (many exact zeros) violate that.
  The randomization test makes no distributional assumption and is the IR-standard choice.
- **Normal-approximation / Wald intervals on the mean.** Same normality problem near the
  0/1 boundaries where these metrics live (baseline hit-rate already saturates at 1.0).
  The percentile bootstrap is distribution-free.
- **Grow the test set in the same change.** Larger n is necessary but orthogonal; doing it
  here would couple a live, corpus-dependent data task to the pure statistics code and
  delay both. Tracked as the follow-up live step (point 5).

## Consequences

- Each `eval_results_<mode>.json` and the combined `eval_results.json` grow by a small
  `ci` / `paired_vs_baseline` block; consumers that read only `means` are unaffected
  (the additions are additive keys).
- Reviewers and the dashboard can now see when a mode's apparent edge is within noise —
  overlapping CIs or `p_value >= 0.05` mean "no measurable difference", which should be
  said explicitly in any README/dashboard ranking claim.
- The bootstrap/randomization passes add a sub-second, CPU-only step at the end of a run
  that otherwise spends hours in live pipelines and judges; negligible cost.
- Significance is only as meaningful as the sample: with n in the low tens many real
  differences will be **undetectable** (wide CIs). That is the honest message, and it is
  the motivation to grow the test set next (point 5).
- Older committed per-mode files written before this change simply lack the `ci` key; the
  harness and app treat a missing `ci`/`paired_vs_baseline` as empty.

## Sources

- Saad-Falcon, Khattab, Potts & Zaharia, *ARES: An Automated Evaluation Framework for
  Retrieval-Augmented Generation Systems*, NAACL 2024 — https://arxiv.org/abs/2311.09476
  (prediction-powered inference / calibrated estimation with confidence intervals for
  LLM-judged metrics; also cited in ADR-0002).
- Smucker, Allan & Carterette, *A Comparison of Statistical Significance Tests for
  Information Retrieval Evaluation*, CIKM 2007 — DOI 10.1145/1321440.1321528 (paired
  randomization / bootstrap as the IR-standard significance tests).
- Thakur, Reimers, Rücklé, Srivastava & Gurevych, *BEIR: A Heterogeneous Benchmark for
  Zero-shot Evaluation of Information Retrieval Models* — https://arxiv.org/abs/2104.08663
  (adequately-sized, heterogeneous evaluation sets).
- Friel, Belyi & Sanyal, *RAGBench* — https://arxiv.org/abs/2407.11005 ; Ru et al.,
  *RAGChecker* — https://arxiv.org/abs/2408.08067 (RAG-eval scale/reproducibility motivating
  larger labeled sets).
- Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 (percentile bootstrap
  confidence intervals).
- Nygard, *Documenting Architecture Decisions* —
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
