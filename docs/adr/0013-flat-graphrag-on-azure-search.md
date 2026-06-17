# 0013 — Flat GraphRAG on Azure AI Search (no graph database)

**Status:** Accepted (2026-06-16)

## Context

GraphRAG (Microsoft Research, 2024) structures a corpus as a graph of entities,
relationships, and communities, then answers queries using either entity-level local
search or community-report-level global search. It's designed for queries that require
reasoning across the full corpus rather than retrieving the one nearest passage.

The question is where to store the graph. GraphRAG's reference implementation uses a
graph database or parquet files. This project stores everything in Azure AI Search and
has no other persistence layer.

There's also the question of how to build the graph. The Microsoft `graphrag` package
does the full pipeline, but it brings its own storage, config, and orchestration that
won't slot into the RAGAS harness without a substantial adapter.

## Decision

1. **Three flat Azure AI Search indexes.** The graph is materialized at build time into:
   - `graph-entities`: one row per entity (name, type, description, description
     embedding, community id, source chunk ids).
   - `graph-relationships`: one row per edge (source entity, target entity, description,
     weight, source chunk ids).
   - `graph-communities`: one row per community (level, title, LLM summary report,
     report embedding).

   Azure AI Search stores and searches the output. Python is the graph engine at build
   time. This is the same shape as Microsoft's GraphRAG solution accelerator.

2. **In-memory adjacency for local search.** At startup, the substrate reads
   `graph-relationships` once and builds an in-memory adjacency map. Local search then:
   hybrid-searches `graph-entities` to find seed entities, expands 1 to 2 hops via the
   in-memory map, gathers the source chunks of seed + neighbor entities and their
   connecting edges. No graph database, no query-time traversal over a remote store.

3. **Global search via community reports.** Global search hybrid-searches
   `graph-communities` for the most relevant community summary reports, then
   map-reduces the top results into candidate context. Zero traversal at query time.

4. **Substrate fuses local + global with RRF.** Both modes run per query; the substrate
   merges the candidate lists and returns one ranked list to the common rerank tail.

5. **Hand-rolled build path.** `build_graph` does: per-chunk entity + relationship
   extraction, entity deduplication, networkx graph construction, Leiden community
   detection (graspologic), LLM community report generation, upload to the three indexes.
   Each step logs failures per chunk/community and skips rather than aborting.

## Alternatives rejected

- **Cosmos DB (Gremlin) for graph storage.** A real graph DB with native traversal, but
  it is a new Azure resource, new cost, a new auth path, and overkill for 1-2 hop
  expansion. An in-memory adjacency map over flat rows is fast enough for a 584-page
  corpus and keeps the stack Azure AI Search only.
- **Microsoft `graphrag` package.** Less extraction code, but the package brings
  parquet/LanceDB storage and its own orchestration. Getting it to emit into Azure AI
  Search and be scored by the RAGAS harness would require more adapter code than
  hand-rolling the extraction in the first place.
- **Deep online traversal (3+ hops, shortest path).** More expressive graph queries, but
  the adjacency map grows quadratically past 2 hops and the latency budget for a live
  `/query` endpoint doesn't support it. The whole point of the build-time materialization
  is to push that cost offline.
- **One shared index with a `strategy` discriminator.** Entities and community reports
  have fundamentally different shapes; forcing them into one schema creates sparse rows
  and awkward query logic. Three indexes with clean schemas are simpler.

## Consequences

- Three new Azure AI Search indexes. Storage and indexing cost scale with extraction
  quality (more entities per chunk = more rows). On a 584-page corpus this is
  manageable; revisit if the corpus grows significantly.
- Build time is dominated by LLM extraction per chunk (one call per chunk for entity
  extraction, one call per community for community reports). The ADR-0005 cache helps
  on rebuilds.
- The in-memory adjacency map is loaded once at startup. On a large corpus this could
  be a memory concern; for now the 584-page target is fine.
- Local search quality depends on extraction quality. A chunk with no entities extracted
  contributes nothing to graph traversal. The build logs these as skipped chunks, and
  the global search path still covers them via community reports.
- Community detection uses Leiden (graspologic), which requires an extra dependency.
  The fallback on an empty or tiny graph is one trivial community covering all entities.

## Sources

- Microsoft Research, *GraphRAG: A modular graph-based RAG system* —
  https://arxiv.org/abs/2404.16130
- Microsoft, *GraphRAG solution accelerator* (flat graph in Azure AI Search) —
  https://github.com/Azure-Samples/graphrag-accelerator
- Traag, Waltman & van Eck, *From Louvain to Leiden: guaranteeing well-connected
  communities* — https://arxiv.org/abs/1810.08473
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §2
  (GraphRAG para), §3 (`build_graph`)
