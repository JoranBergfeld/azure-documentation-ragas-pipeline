# Eval history

The committed `eval_results_<mode>.json` files only ever hold the latest run. This is the
log of what they said before, so a re-run doesn't quietly erase the last baseline.

Newest first. Means over `data/testset.jsonl`. With n in the tens, treat anything inside
the confidence intervals as "no measurable difference" (ADR-0018).

## 2026-10-25: retrieval on pgvector (ADR-0021, #37)

First run after retrieval moved from Azure AI Search to pgvector in homelab's Postgres.
Same 43 questions, same generator, same offline judge (`DeepSeek-V4-Pro`), ragas 0.4.3,
Python 3.11. The "before" numbers are the Azure-era files from f87ec34 (2026-09-13).

Only the six modes that can run without a reranker. `contextual`, `baseline` and
`raptor_sac` need the semantic ranker, which left with Azure. Their Azure-era files were
dropped in 72869fa, and they get re-evaluated when the Spark serves `bge-reranker-v2-m3`.

Azure → pgvector:

| mode | hit_rate@fused | mrr@fused | faithfulness | answer_relevancy | context_precision | context_recall | abstained |
|---|---|---|---|---|---|---|---|
| graphrag | 0.951 → 0.956 | 0.576 → 0.580 | 0.806 → 0.818 | 0.781 → 0.743 | 0.450 → 0.463 | 0.393 → 0.321 | 0.349 → 0.279 |
| combined | 0.960 → 0.984 | 0.754 → 0.769 | 0.876 → 0.892 | 0.866 → 0.895 | 0.584 → 0.603 | 0.729 → 0.672 | 0.140 → 0.140 |
| baseline_agentic | 0.977 → 0.972 | 0.825 → 0.826 | 0.875 → 0.850 | 0.854 → 0.864 | 0.568 → 0.538 | 0.770 → 0.717 | 0.116 → 0.093 |
| raptor_sac_agentic | 0.974 → 0.967 | 0.857 → 0.860 | 0.896 → 0.878 | 0.862 → 0.865 | 0.571 → 0.552 | 0.742 → 0.759 | 0.047 → 0.047 |
| graphrag_agentic | 0.980 → 0.974 | 0.612 → 0.610 | 0.821 → 0.804 | 0.796 → 0.821 | 0.498 → 0.453 | 0.414 → 0.381 | 0.349 → 0.302 |
| combined_agentic | 0.985 → 0.988 | 0.769 → 0.791 | 0.859 → 0.888 | 0.830 → 0.877 | 0.578 → 0.572 | 0.591 → 0.654 | 0.163 → 0.047 |

95% bootstrap intervals on the new run:

| mode | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| graphrag | 0.74–0.89 | 0.62–0.85 | 0.35–0.57 | 0.19–0.45 |
| combined | 0.84–0.94 | 0.85–0.93 | 0.49–0.71 | 0.55–0.79 |
| baseline_agentic | 0.78–0.91 | 0.82–0.91 | 0.42–0.65 | 0.59–0.83 |
| raptor_sac_agentic | 0.84–0.92 | 0.80–0.92 | 0.43–0.67 | 0.64–0.87 |
| graphrag_agentic | 0.72–0.88 | 0.72–0.90 | 0.34–0.56 | 0.26–0.52 |
| combined_agentic | 0.86–0.92 | 0.82–0.92 | 0.46–0.68 | 0.52–0.78 |

What it says, as far as 43 questions can say anything:

- Retrieval held up. hit_rate and mrr moved by about 0.02 at most, in both directions.
- Every Azure-era judge-metric mean sits inside the new run's interval. No measurable
  regression. The only means outside it are combined_agentic's abstain rate (below) and
  combined's raptor_sac sub-leg hit rate, which went up (0.91 → 0.97).
- context_recall dipped on graphrag, combined and baseline_agentic. Best guess is the
  re-embed at import shifting which chunks come back. Not checked.
- combined_agentic abstains a lot less (16% → 5%, outside the 0.00–0.12 interval) and its
  judge scores went up anyway. The one change here that looks real.

Coverage: faithfulness and answer_relevancy are only scored on answered questions, so their
`valid` counts follow the abstain rate. Six judge jobs died across the run (three 600s
timeouts, three 429s from the DeepSeek deployment), costing at most two context_precision
scores per mode.

Run notes: on Python 3.14 the RAGAS step deadlocked at 0 jobs done (twice, hours each), so
this ran on 3.11 to match the image. Hence `.python-version`.
