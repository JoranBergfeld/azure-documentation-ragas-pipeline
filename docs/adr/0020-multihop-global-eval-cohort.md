# 0020 — Multi-hop and global eval cohort

**Status:** Accepted (2026-06-23)

## Context

The hand-authored test set had 33 single-hop factoid items, each with one gold URL. That
shape cannot measure multi-hop questions whose answer requires more than one source page,
or global/sensemaking questions with no single gold page, which is GraphRAG's intended
regime. The deterministic URL-match metric also assumed exactly one gold URL.

## Decision

`TestItem` supports multiple gold URLs for multi-hop questions. Deterministic
`hit_rate`/`mrr` use any-match semantics across those URLs, so retrieving any gold page is
a URL-match hit. Items with no gold URL are global sensemaking items: the harness skips
URL-match metrics per item and scores them only with graded RAGAS metrics.

The hand-authored test set now includes `multihop` and `global` tags plus a small, real,
hand-verified seed cohort anchored on corpus pages: 3 multi-hop items and 8 global items.
Expanding this cohort is an ongoing operations step. Evaluation labels are human-verified
work: author new items by hand, verify them against live Microsoft Learn pages, and do not
auto-generate labels.

## Consequences

- GraphRAG global routing (#8) now has items that exercise the global route.
- `aggregate_by_tag` surfaces `multihop` and `global` cohorts automatically through the
  existing tag aggregation path.
- Multi-hop `hit_rate` is any-gold, not full gold-coverage recall. Measuring whether all
  required gold pages were retrieved is a future metric.
- Re-running the live eval to refresh `eval_results_*.json` checkpoints with the new
  items is an operations step, not part of this PR.
- This change has resolvable textual overlap with #7, which also edits
  `retrieval_metrics.py`, and #9, which also edits `harness.py`.

## Sources

- GraphRAG, *From Local to Global* — https://arxiv.org/abs/2404.16130
- HotpotQA (multi-hop QA) — https://arxiv.org/abs/1809.09600
- MuSiQue (multi-hop) — https://arxiv.org/abs/2108.00573
- RAGAS — https://arxiv.org/abs/2309.15217
