# 0002 — Deterministic URL-match retrieval metrics as primary signal

**Status:** Accepted (2026-06-10)
**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md`

## Context

Retrieval quality was measured only by LLM-judged RAGAS metrics
(`context_precision`/`context_recall`): noisy per-item, non-reproducible across
runs, and costly (one judge pass per stage in the per-stage sweep). Meanwhile every
test item already carries a ground-truth source URL (`ground_truth_context`) and
every indexed chunk carries its `url` — an exact relevance judgment we weren't using.

## Decision

Add LLM-free metrics computed from URL membership, per retrieval stage
(`dense`, `bm25`, `fused`, `reranked`):

- `hit_rate`: any chunk in the stage's top-k matches the ground-truth URL.
- `mrr`: reciprocal rank of the first matching chunk.

Keys follow the existing `metric@stage` convention (`hit_rate@dense`,
`mrr@reranked`). These are the **primary regression signal** for retrieval changes;
RAGAS context metrics remain as a complement (they capture partial/graded relevance
that binary URL match cannot).

## Alternatives rejected

- **LLM-judged metrics only** (status quo): per-item judge variance swamps real
  differences at the current test-set size, and scores aren't reproducible
  run-to-run. Exact judgments are the standard for IR evaluation (BEIR).
- **nDCG / graded relevance**: requires graded judgments we don't have; with one
  gold URL per question, hit-rate and MRR carry the same information.

## Consequences

- Retrieval metrics become free (microseconds, no tokens) and exactly reproducible,
  so they can run on every eval without the `PER_STAGE_METRICS` cost gate.
- Page-level granularity: a hit means the right *page*, not necessarily the right
  *chunk* of it. Acceptable: DRM (wrong document) is the failure mode under attack.
- Stages with no canonical source URL, such as GraphRAG community/global LLM-summary
  stages, are excluded from URL-match `hit_rate`/`mrr`; interpreting a structural
  0.0 there as retrieval failure was a measurement artifact. Use graded RAGAS
  `context_precision`/`context_recall` for those stages instead.
- Questions whose answer genuinely spans multiple pages need a single canonical URL
  in the test set (or the loader must accept a list later).

## Sources

- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models* — https://arxiv.org/abs/2104.08663
- Saad-Falcon et al., *ARES: An Automated Evaluation Framework for RAG* (LLM-judge
  noise and the need for calibrated/statistical treatment) —
  https://arxiv.org/abs/2311.09476
- Microsoft Research, *GraphRAG: A modular graph-based RAG system* (community reports
  synthesize over many source chunks) — https://arxiv.org/abs/2404.16130
