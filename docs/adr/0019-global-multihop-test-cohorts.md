# 0019 — Global-sensemaking and multi-hop test cohorts (multi-gold URLs)

**Status:** Accepted (2026-06-24)

## Context

The multi-mode comparison (ADR-0016) is run entirely on **single-hop factoid**
questions: `data/testset.jsonl` was 33 items, each with exactly one
`ground_truth_context` URL. Two structural problems follow.

1. **No headroom.** Baseline `hit_rate@dense` = `hit_rate@fused` = 1.000
   (`eval_results_baseline.json`): retrieval is already saturated, so the
   deterministic URL-match metrics that ADR-0016/ADR-0002 designate the *primary*
   cross-mode signal have almost nothing left to discriminate substrates with.
2. **Out-of-envelope evaluation.** The substrates whose value is corpus-level
   synthesis are judged only on factoids, the regime where they are *expected* to
   lose. GraphRAG is a query-focused **summarization** system for global
   "sensemaking" questions (Edge et al. 2024); RAPTOR's gains are on
   higher-level-synthesis benchmarks (Sarthi et al. 2024); structure-augmented RAG
   can underperform strong embedding RAG on plain QA (HippoRAG 2, Gutiérrez et al.
   2025); multi-hop is where graph / PPR methods show their advantage (HippoRAG,
   Gutiérrez et al. 2024). A factoid-only set structurally cannot reward them.

ADR-0002 already anticipated the fix: "Questions whose answer genuinely spans
multiple pages need a single canonical URL in the test set (**or the loader must
accept a list later**)."

## Decision

1. **Gold labels may be a list of URLs.** `TestItem.ground_truth_context` is now
   `str | tuple[str, ...]`: a bare string for single-hop factoids (unchanged), a
   list for items whose answer spans several pages. The loader normalizes a JSON
   list to a tuple; a bare string stays a string, so the existing 33 items and
   their committed scores are byte-for-byte unchanged.

2. **`hit_rate` becomes recall over the gold set.** It is the fraction of gold URLs
   present in a stage's results. For a single gold URL this is the original binary
   1.0/0.0 (backward compatible); for several it gives partial credit — the
   headroom a binary "any of N pages" hit-rate lacks. `mrr` is the reciprocal rank
   of the first chunk matching **any** gold URL.

3. **Two new tagged cohorts** in `data/testset.jsonl` (ADR-0006 tag axis):
   - `multihop` (5 items): a question that genuinely requires ≥2 corpus pages,
     gold = the set of those pages. Recall measures how many of the required pages
     a substrate retrieves.
   - `global` (5 items): a corpus-level / sensemaking question over a theme, gold =
     a curated set of the pages a comprehensive answer should draw on. Recall over
     that set is a **coverage proxy** — exactly what GraphRAG global search and
     RAPTOR summary nodes are meant to win on — without yet introducing a new
     LLM-judged comprehensiveness metric.

4. **Gold-less items are RAGAS-only.** If `ground_truth_context` is empty,
   `stage_retrieval_metrics` emits no deterministic keys at all; the item still
   contributes to the RAGAS suite (ADR-0016 already states this). Every committed
   item today carries gold, but the path is supported for open-ended sensemaking
   items a coverage set cannot fairly enumerate.

## Alternatives rejected

- **Keep a single canonical URL per multi-page question.** Loses information: a
  multi-hop answer is not "on" one page, and binary hit-rate on it saturates just
  like the factoid set. A gold *set* with recall is strictly more discriminating.
- **Only re-scope the README/ADR-0016 claims to "single-hop factoid".** Honest, but
  it abandons the comparison the project exists to make. Adding the cohorts is the
  substantive fix; the scoping note is folded in alongside (README).
- **A new LLM-judged summarization/comprehensiveness metric for global items now.**
  The right long-term metric for open sensemaking, but ill-defined, costly, and
  judge-variance-prone (ADR-0009). Recall over a curated coverage set reuses the
  exact, free, reproducible deterministic machinery and lands the cohorts today; a
  comprehensiveness metric can come later as a complement.

## Consequences

- `hit_rate` is now "recall over gold" everywhere. Single-gold values are
  unchanged, so committed per-mode results stay valid and comparable; multi-gold
  items report graded recall (0, 1/N … 1).
- The deterministic primary signal regains headroom on the `multihop`/`global`
  cohorts even while `original` stays saturated — the comparison can finally move
  on the substrates ADR-0016 added.
- Anything constructing or reading `ground_truth_context` must handle both shapes;
  `retrieval_metrics.gold_set()` is the single normalizer. The synthetic generator
  path still emits single-URL items (unchanged).
- Per-tag aggregation (ADR-0006) now reports `multihop` and `global` groups for
  free. Sample sizes are small (5 each); treat them as directional until grown
  toward the ADR-0010 size targets.

## Sources

- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization*, 2024 — https://arxiv.org/abs/2404.16130
- Gutiérrez et al., *From RAG to Memory: Non-Parametric Continual Learning for LLMs*
  (HippoRAG 2), ICML 2025 — https://arxiv.org/abs/2502.14802
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized
  Retrieval*, ICLR 2024 — https://arxiv.org/abs/2401.18059
- Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for
  LLMs*, NeurIPS 2024 — https://arxiv.org/abs/2405.14831
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models*, 2021 — https://arxiv.org/abs/2104.08663
- ADR-0002 (deterministic URL-match metrics), ADR-0006 (tagged subsets), ADR-0016
  (multi-mode evaluation axis), ADR-0009 (three-family judge split / LLM-judge
  noise).