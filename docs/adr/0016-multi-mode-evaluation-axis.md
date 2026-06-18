# 0016 — Multi-mode evaluation axis

**Status:** Accepted (2026-06-16)

## Context

The eval harness (ADR-0006) runs one retrieval mode against a test set and produces
`means`, `means_by_tag`, and `records`. The tag axis compares cohorts within a mode
(e.g., questions about section X vs. section Y). There is no axis that compares
retrieval modes against each other.

With 8 modes now in the picture, we need a way to run the same test set through each
mode and put the results side by side. We also need to pick a primary comparison
signal. RAGAS faithfulness and answer relevancy are useful, but they are LLM-judged
and therefore harder to reproduce exactly and potentially confounded by model variance.
The deterministic metrics (ADR-0002) are exact, reproducible, and LLM-free.

There's also a fairness question: if the existing `top_k`, `rrf_k`, `candidate_pool`,
and guardrail threshold knobs are set differently per mode, the comparison is not
meaningful. They should be shared.

## Decision

1. **New mode axis alongside the tag axis.** `run.py` takes a set of modes (CLI arg,
   default all 8 or a configured subset), runs the same `testset.jsonl` through each
   mode's `pipeline_fn`, and aggregates `means_by_mode` keyed by mode name.
   `eval_results.json` becomes keyed by mode at the top level, each holding the
   existing `means`, `means_by_tag`, `coverage`, `records` structure unchanged.

2. **Deterministic URL-match metrics as the primary cross-mode signal.** `hit_rate@stage`
   and `mrr@stage` (ADR-0002) are exact, reproducible, and LLM-free. They are the
   numbers we lead with when comparing modes. RAGAS metrics (faithfulness,
   answer_relevancy, context precision/recall) stay as a complement, per mode, for
   the richer picture.

3. **Dynamic stage reading.** `RETRIEVAL_STAGES` stops being a fixed tuple. The harness
   reads stage names from each record's `state.stages` keys (ADR-0012). Each mode's
   stages are whatever that substrate named them; the harness reports stats for each.
   The `reranked` stage is the stable well-known name that the final per-mode summary
   always includes.

4. **Shared knobs for fair comparison.** `top_k`, `rrf_k`, `candidate_pool`, and the
   faithfulness gate threshold are shared across modes. A mode that bumps `top_k` to
   get a better score is cheating. The config makes this explicit: per-substrate index
   names are allowed, per-substrate retrieval budget knobs are not.

5. **Per-mode standalone artifacts, written incrementally.** Besides the combined
   roll-up, `run.py` writes one self-contained `eval_results_<mode>.json` (suffixed
   with the mode value, e.g. `eval_results_graphrag.json`) the moment each mode
   finishes. Each holds that mode's `means`, `means_by_tag`, `coverage`, and `records`
   plus a top-level `mode` key, so it stands alone and is committed to the repo as the
   reference result for that substrate. Because a full multi-mode run over the corpus is
   agentic and takes hours, these files double as checkpoints: a re-run reuses any mode
   whose file already exists and resumes at the next unfinished mode. The combined
   `eval_results.json` is rebuilt from the per-mode aggregates at the end and stays
   gitignored (a derived roll-up the dashboard/API read locally).

## Alternatives rejected

- **Separate eval runs with separate config files, stitched by hand.** Reproducible only
  if the human always runs exactly the same flags. The mode axis in one run guarantees
  same test set, same knobs, same judge, same commit, and produces one
  `eval_results.json` that captures the full comparison without manual stitching.
- **RAGAS faithfulness as the primary cross-mode signal.** More holistic, but LLM-judged
  numbers re-anchor whenever the judge model changes (see ADR-0009's work-machine
  protocol), and they're slower and costlier to compute at scale. Deterministic metrics
  are the right primary signal; RAGAS is the complement.
- **Tag axis extended with a mode dimension (mode x tag grid).** Covers more combinations
  but the result is harder to read and the per-cell sample sizes shrink fast. The simpler
  cross-product (mode axis + tag axis within each mode) is enough for the research story.

## Consequences

- Running all 8 modes on a full test set multiplies eval wall-clock time by up to 8x
  (modes run sequentially by default). A `--modes` flag lets you run a subset when
  iterating. Phase 1 starts with just 2 modes (baseline + contextual), so the cost is
  manageable now.
- `eval_results.json` grows: it now has a top-level key per mode, each containing the
  full existing structure. Anything that reads the old flat structure breaks. Dashboard
  and any downstream scripts need to handle the new shape.
- Shared knobs mean adding a substrate that genuinely needs a different `top_k` to
  work reasonably can't just tune it in isolation. That's the price of a fair
  comparison; document it if a substrate needs a note.
- The deterministic metrics require the source URLs to be present in the test set
  (ADR-0002). Test items without ground-truth URLs contribute to RAGAS metrics only.
- The committed per-mode files (a few MB each, ~2–6 MB) make each substrate's results
  reviewable in version control without re-running the judges; the combined
  `eval_results.json` stays gitignored and derived.
- The offline RAGAS judge is a reasoning model, so it is slow per job. The harness gives
  it a dedicated `RunConfig` (longer per-job timeout, fewer concurrent workers than the
  RAGAS defaults) — without it, faithfulness and context_precision time out to NaN under
  the default 16-way concurrency. This trades wall-clock time for reliable coverage; a
  full 5-mode run takes several hours.

## Sources

- *Document-Level Retrieval Mismatch* (motivation for richer, multi-strategy retrieval
  evaluation) — https://arxiv.org/abs/2510.06999
- Nygard, *Documenting Architecture Decisions* —
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §8
  (eval harness mode axis), §7 (shared config knobs)
