# 0018 — Route GraphRAG's global leg by query class instead of always RRF-fusing it

**Status:** Accepted (2026-06-24)

## Context

ADR-0014 built GraphRAG as a local leg (entity match + 1-hop relationship expansion)
and a global leg (community-summary report ranking), and fused both with RRF on every
query before the shared rerank tail.

On the evaluated factoid workload the global leg is a **net negative**. The measured
evidence in this repo:

- `eval_results_graphrag.json` `means_by_tag`: the local leg retrieves the correct chunk
  **100% of the time** (`hit_rate@local` = 1.0 for every tag), but after fusing global
  and semantic-reranking, `hit_rate@reranked` falls to 0.938 (`original`), 0.571
  (`lookalike`) and **0.286** (`paraphrase`). Overall GraphRAG `hit_rate@reranked` =
  **0.727** vs the contextual baseline's **0.970**.
- `eval_results_combined.json`: fusing the GraphRAG leg drags `answer_relevancy` to 0.757
  and `context_precision` to 0.707 — **below** pure `contextual` (0.856 / 0.860). The
  weaker leg dilutes the stronger one.

The mechanism: community summaries are topically attractive to the reranker but carry an
empty URL and no leaf-level precision. RRF-fusing them and then reranking pushes the
correct leaf chunk below the top-k cut. RRF itself is sound (it is how Azure AI Search
fuses full-text and vector results — Cormack et al. 2009), but RRF over graph-derived
*summaries* is not GraphRAG's native map-reduce global search: flattening breadth-oriented
summaries into a top-k factoid ranking collapses their intended benefit and evicts precise
local evidence.

## Decision

**Route the global leg by query class; do not always fuse it.**

1. A deterministic, network-free classifier (`retrieval/query_class.py`,
   `classify_query`) labels each query `LOCAL` or `GLOBAL` by matching a conservative,
   word-boundaried list of breadth/sensemaking markers (`compare`, `overview`,
   `summarize`, `themes`, `relationship between`, `pros and cons`, `types of`, `across`,
   …). The default is **LOCAL**: the evaluated workload is factoid-heavy and that is
   exactly where the global leg hurts, so only an explicit sensemaking marker promotes a
   query to GLOBAL.
2. `GraphRAGSubstrate.retrieve` calls the community search and RRF-fuses local ⊕ global
   **only for GLOBAL queries**. For LOCAL queries it skips the community call entirely and
   returns the local ranking alone (saving a search round-trip as well). The `stages`
   dict still always carries `local`, `global` (empty when skipped) and `fused`, so the
   dashboard/eval contract is unchanged.
3. A `routing` flag (substrate ctor) plumbed from `Settings.graph_query_routing`
   (`GRAPH_QUERY_ROUTING`, default `true`) gates the behavior. Setting it `false` restores
   the legacy always-fuse path for A/B evaluation. `classify_fn` is injectable for tests
   and future smarter routers (e.g. reusing the agentic planner `ctx.plan`).

Chosen because it is the smallest change that removes the measured regression while
keeping global search available where it is designed to help. It lives entirely behind the
retrieval substrate seam (ADR-0012) — nothing downstream of retrieval changes.

## Alternatives rejected

- **Weight the legs in RRF / rerank** so empty-URL summaries cannot outrank leaf chunks.
  A continuous knob, but it needs tuning per workload and still feeds irrelevant summaries
  into the reranker on factoid queries. Routing is a cleaner on/off boundary for a workload
  that is overwhelmingly factoid.
- **Drop the global leg outright.** Simplest, but throws away GraphRAG's actual purpose;
  global community search is the right tool for sensemaking/breadth queries and the planned
  global/multi-hop test items will need it.
- **LLM-classify every query.** More flexible, but adds a per-query network call and
  non-determinism to a hot path; the heuristic is cheap, reproducible, and unit-testable,
  and the seam lets us swap in an LLM/planner router later without touching the substrate.

## Consequences

- GraphRAG (and the GraphRAG leg of `combined`, which builds the same substrate) no longer
  dilutes factoid retrieval; the local leg's correct chunk survives the rerank cut.
- The benefit on sensemaking queries can only be *measured* once global/multi-hop test
  items exist (the routing pairs naturally with the `@global` URL issue and the
  "add global/multi-hop test items" issue). Until then the router's GLOBAL branch is
  exercised by unit tests, not by the eval set.
- The classifier is a heuristic: an unmarked sensemaking query stays LOCAL and a factoid
  query containing a marker word is promoted to GLOBAL. The marker list is intentionally
  conservative and easy to extend; misroutes degrade gracefully (the faithfulness gate and
  rerank tail remain the final arbiters).

## Sources

- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*,
  2024 — https://arxiv.org/abs/2404.16130
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual
  Rank Learning Methods*, SIGIR 2009; Azure AI Search hybrid ranking —
  https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- Gutiérrez et al., *From RAG to Memory: Non-Parametric Continual Learning for LLMs*
  (HippoRAG 2 — structure-augmented retrieval underperforms strong embedding RAG outside
  its regime), ICML 2025 — https://arxiv.org/abs/2502.14802
- Li et al., *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and
  Hybrid Approach* (self-reflection routing), EMNLP Industry 2024 —
  https://aclanthology.org/2024.emnlp-industry.66/
- Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation* (dual-level
  low-/high-level retrieval) — https://arxiv.org/abs/2410.05779
- Issue #8; ADR-0012 (retrieval substrate seam); ADR-0014 (flat GraphRAG on Azure AI Search)
