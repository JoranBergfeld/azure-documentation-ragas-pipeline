# 0019 — GraphRAG query-class routing

**Status:** Accepted (2026-06-23)

## Context

GraphRAG local retrieval is precise on the current factoid workload, but global
community summaries are broad, topically attractive, and have empty URLs
(`graph_substrate.py` builds community `Chunk` objects with `url=""`). When global
summaries are always RRF-fused with local chunks and then semantic-reranked, they can
evict the precise local leaf chunk below the top-k cut.

The observed eval evidence shows the failure mode: GraphRAG `hit_rate@local` is 1.0 on
every tag, while `hit_rate@reranked` falls to 0.938 on original questions, 0.571 on
lookalikes, and 0.286 on paraphrases. Overall GraphRAG `hit_rate@reranked` is 0.727
versus baseline at 0.970. On this factoid workload, the global leg is a net negative.

## Decision

Route GraphRAG by query class. The default `classify_query` heuristic sends
factoid/local queries to local-only retrieval and engages global community search only
when the query contains sensemaking or breadth cues such as overview, summary,
comparison, across, themes, broad categories, or overall/landscape wording.

The default remains conservative: local retrieval is used unless a breadth cue is
present. An LLM- or `ctx.plan`-based router is a future seam if the heuristic becomes
too brittle. Weighted RRF was considered as an alternative, but routing directly removes
the harmful global candidate set for factoid questions instead of merely down-weighting
it.

## Consequences

- Factoid queries no longer lose precise local chunks to empty-URL community summaries.
- Global sensemaking remains available when the query explicitly asks for breadth.
- Combined retrieval inherits this behavior because it consumes GraphRAG's routed
  candidates and stages.
- Measuring the benefit on global/multi-hop queries requires the new global/multi-hop
  test items from issue #6. Without them, the win shows only as "no regression" on the
  current factoid testset.

## Sources

- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and
  individual Rank Learning Methods* (SIGIR 2009) —
  https://doi.org/10.1145/1571941.1572114
- Microsoft Learn, *Hybrid search ranking* —
  https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization* — https://arxiv.org/abs/2404.16130
- Gutiérrez et al., *HippoRAG 2: Neurally Biologically Inspired Long-Term Memory for
  Large Language Models* — https://arxiv.org/abs/2502.14802
- Li et al., *Retrieval-Augmented Generation or Long-Context LLMs? A Comprehensive
  Study and Hybrid Approach* — https://aclanthology.org/2024.emnlp-industry.66/
- Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation* —
  https://arxiv.org/abs/2410.05779
