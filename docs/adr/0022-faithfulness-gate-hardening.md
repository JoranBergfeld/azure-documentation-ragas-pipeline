# 0022 — Faithfulness gate hardening

**Status:** Accepted (2026-06-23)

## Context

The online guardrail uses a fixed RAGAS faithfulness threshold of `0.7`. That threshold is a
heuristic, not a value calibrated against human labels for this corpus, generator, retrieved-context
shape, or judge deployment. Scores from the Claude online gate and DeepSeek offline RAGAS judge are
also not cross-calibrated: ADR-0009's three-family split reduces self-preference bias, but it does
not make scores interchangeable.

RAGAS faithfulness measures whether the answer is grounded in the retrieved context. It is not a
correctness oracle: a faithful summary of wrong, stale, or irrelevant retrieved context can still pass.
The retry loop can spend up to three generate+judge cycles per query, so mis-gating is both a quality
and cost problem.

The gate previously had no in-repo calibration set, no frozen drift canary, and no explicit
fingerprint for the RAGAS version, private faithfulness prompts, generator model, online judge, or
offline judge. That is risky because LLM judges are sensitive to prompt wording and exhibit
position, verbosity, and self-preference biases; hard faithfulness cases remain difficult for both
LLM and non-LLM hallucination detectors; and RAGAS APIs/prompts are changing as legacy metrics move
toward removal before v1.0.

## Decision

Ship hardening machinery without changing the production threshold:

- Add structured `FaithfulnessResult` / `ClaimVerdict` types and normalize scalar scorers into the
  structured form so existing tests and fakes remain compatible.
- Drive the installed RAGAS `Faithfulness` metric's real 0.4.3 private decomposition seam
  (`_create_statements` → `_create_verdicts` → `_compute_score`) in live detailed scoring. The
  pipeline logs per-claim verdicts in both trace data and progress events while preserving existing
  scalar score and `NaN` semantics.
- Keep the existing scalar builder intact, but wire live `build_pipeline_fn` through the detailed
  builder so operators can inspect claim-level grounding decisions.
- Pin `ragas==0.4.3` and add `ragpipe.eval.judge_fingerprint`, which records the installed and
  expected RAGAS version, whether the pin matches, online/offline/generator model names, and a
  16-character SHA-256 signature of the RAGAS faithfulness prompt instructions.
- Add a frozen labeled canary loader and drift harness over `data/faithfulness_canary.jsonl`. The
  live command `uv run python -m ragpipe.eval.canary` writes `eval_results_canary.json`, returns
  nonzero on drift or an unpinned RAGAS version, and keeps that paid live artifact gitignored.
- Add a seam-guard test that imports the installed RAGAS classes and asserts the private methods and
  verdict fields still exist. This deliberately fails loudly if a future RAGAS upgrade changes the
  private seam instead of silently mis-scoring.

## Consequences

The default faithfulness threshold intentionally remains `0.7` until operators run a separate human
label calibration exercise and choose a threshold from measured precision/recall/cost tradeoffs. This
ADR documents that calibration as future operational work, not as part of this machinery change.

Running the drift canary against Azure judges is also future operational work because it makes paid
LLM calls. Its output (`eval_results_canary.json`) must not be committed. A scheduled job can run the
command and fail on drift or a RAGAS version mismatch.

An optional HHEM-style non-LLM hallucination classifier remains future work as a faster secondary
check or canary signal. It should be evaluated against the same human-labeled calibration set before
it affects the online gate.

The detailed scorer is coupled to RAGAS private APIs by design. The explicit dependency pin,
fingerprint, canary, and seam-guard test make that coupling visible and operationally detectable.

## Sources

- Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” NeurIPS 2023 — https://arxiv.org/abs/2306.05685
- Panickssery, Bowman, and Feng, self-preference in LLM judges — https://arxiv.org/abs/2404.13076
- Wataoka et al., self-preference in LLM-as-a-judge — https://arxiv.org/abs/2410.21819
- Bao et al., FaithBench, NAACL 2025 — https://aclanthology.org/2025.naacl-short.38/
- Tamber et al., FaithJudge, EMNLP Industry 2025 — https://arxiv.org/abs/2505.04847
- Saad-Falcon et al., ARES — https://arxiv.org/abs/2311.09476
- Ru et al., RAGChecker — https://arxiv.org/abs/2408.08067
- Friel et al., RAGBench/TRACe — https://arxiv.org/abs/2407.11005
- Vectara HHEM leaderboard — https://github.com/vectara/hallucination-leaderboard
- RAGAS metrics reference — https://docs.ragas.io/en/stable/references/metrics/
