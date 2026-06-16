# Multi-substrate retrieval: 8 comparable retrieval modes over one corpus

**Date:** 2026-06-16
**Status:** Approved
**Origin:** Expansion beyond the original single-pipeline scope. The foundational design
(`2026-05-29-ragas-infused-pipeline-design.md` §2 Non-goals) deliberately dropped a
second pipeline, and the preprocessing spec (`2026-06-10`) explicitly rejected RAPTOR and
graph approaches as out of scope. This spec reverses both, on purpose: the project is now
a research demonstrator whose whole point is comparing retrieval strategies head to head,
live on a website and in the RAGAS harness.

## Problem

The pipeline is wired around one Azure AI Search index and one retrieval topology
(dense + BM25 → RRF → semantic rerank → generate → faithfulness gate). That is fine for
serving, but it can't answer the question the project now exists to answer: which
retrieval decoration actually helps, and by how much, on this corpus?

To demonstrate the research we need several retrieval strategies that share the same
corpus, the same generator, the same judges, and the same test set, so the only thing
that varies is retrieval. Today there is no seam to plug a strategy into, no place to
store more than one index family, and no eval axis that compares strategies (the existing
tag axis compares test-subset cohorts, not retrieval modes).

## Goals

- Introduce a retrieval-strategy seam so multiple substrates plug into the existing
  generate/gate tail without touching it.
- Ship 8 retrieval modes built from 4 substrates crossed with an agentic on/off wrapper.
- Keep storage Azure-native (Azure AI Search only, no new backend) and rebuildable per
  tier.
- Compare modes in the RAGAS harness on a new mode axis, with the deterministic URL-match
  metrics (ADR-0002) as the primary cross-mode signal.
- Surface the comparison live: a `mode` param on `/query` and a `/compare` endpoint that
  runs a query through several modes side by side, for the website.
- Record load-bearing decisions as ADRs with sources.

## Non-goals

- No change to the generate → faithfulness gate → retry tail, the judge families
  (ADR-0009), or the generator model.
- No new storage backend. GraphRAG is materialized as flat rows in Azure AI Search; no
  graph database (Cosmos Gremlin) and no parquet/LanceDB. See Rejected alternatives.
- No external RAG frameworks. RAPTOR, GraphRAG, and the agentic loop are hand-rolled on
  the existing Azure stack. See Rejected alternatives.
- No deep online graph traversal (5+ hops, shortest-path, query-time analytics). The
  graph is consumed at build time and flattened; query-time expansion is 1 to 2 hops.

## The shape: 8 modes = 4 substrates × agentic toggle

| Substrate | non-agentic | + agentic | reads store(s) |
|---|---|---|---|
| Baseline (plain chunks) | mode 1 | mode 2 | `baseline` index |
| SAC + RAPTOR | mode 3 | mode 4 | `raptor-sac` index |
| GraphRAG | mode 5 | mode 6 | `graph-entities`, `graph-relationships`, `graph-communities` |
| SAC+RAPTOR + Graph (combined) | mode 7 | mode 8 | `raptor-sac` + the three graph indexes |

Agentic adds no store. The combined substrate adds no store (it fuses two existing
substrates). So 8 modes are built from 4 substrates + 1 wrapper over 5 Azure AI Search
indexes (plus the existing contextual index, see §2).

## Design

### 1. The retrieval seam (`ragpipe/retrieval/substrate.py`, `ragpipe/app_wiring.py`)

One interface every substrate implements:

```python
class RetrievalSubstrate(Protocol):
    name: str
    async def retrieve(self, query: str, k: int) -> list[Chunk]: ...
```

`retrieve` returns ranked candidates. Everything downstream (rerank → generate →
faithfulness gate → retry loop) stays exactly as it is today. The substrate is the only
thing that varies across modes.

`app_wiring.py` stops hard-wiring `DenseRetriever + BM25Retriever + SemanticReranker +
RRF`. Instead `build_pipeline_fn(settings, mode)` selects a substrate from a registry
keyed by mode, then builds the same `PipelineDeps` around it. `make_deps` stays the clean
injection point; only the `dense`/`bm25` pair behind it changes shape.

A substrate may run its own internal fusion (Baseline still does dense+BM25→RRF inside
itself; GraphRAG does local+global inside itself). The pipeline no longer owns RRF; the
substrate does. The reranker still runs over whatever candidates the substrate returns.

### 2. The four substrates

**Baseline (`substrates/baseline.py`).** Plain heading-aware chunks, no SAC decoration.
Hybrid dense + BM25 → RRF, same machinery as today minus the contextual decoration. New
Azure AI Search index `baseline`: same schema as the contextual index but the embedding
input and the BM25 fields are the raw chunk text (no `context` field contribution).

**SAC + RAPTOR (`substrates/raptor_sac.py`).** Level-0 leaves are the existing
SAC-decorated chunks. RAPTOR adds summary nodes at levels 1+ built at ingest by recursive
cluster-then-summarize (see §3). Retrieval is collapsed-tree: a single hybrid search
across all levels at once (no tree walk at query time), which RAPTOR's paper reports as
matching or beating tree traversal. Stored in `raptor-sac` (see the index-reuse note
below).

**GraphRAG (`substrates/graphrag.py`).** A flat materialized graph in three Azure AI
Search indexes built at ingest (see §3):
- `graph-entities`: one row per entity (name, type, description, description embedding,
  community id, source chunk ids).
- `graph-relationships`: one row per edge (source, target, description, weight, source
  chunk ids).
- `graph-communities`: one row per community (level, title, LLM summary report, report
  embedding).

Two query modes, both served from Azure AI Search:
- *Global search*: hybrid search over `graph-communities` reports, map-reduce the top
  community summaries into candidate context. Zero traversal.
- *Local search*: hybrid search over `graph-entities` to find seed entities, expand 1 to 2
  hops via an in-memory adjacency map built once at startup from `graph-relationships`,
  then gather the source chunks of the seed + neighbor entities and their connecting
  edges.

The substrate runs both and fuses (RRF) into one candidate list. The expensive graph work
(extraction, Leiden community detection, community reports) happens once at build time in
Python; Azure AI Search only stores the materialized output. This is the same shape as
Microsoft's GraphRAG solution accelerator: the database is not a graph DB, Python is the
graph engine at build time.

**Combined (`substrates/combined.py`).** Runs the RAPTOR+SAC substrate and the GraphRAG
substrate, fuses their candidate lists with RRF. No new store, no new build path.

### 3. Ingestion (`ragpipe/ingest.py` + per-substrate builders)

A common driver dispatches to independent, separately runnable build paths so any one
tier rebuilds without touching the others:

- `build_baseline`: plain chunks → `baseline` index. Reuses the heading-aware chunker,
  skips contextual decoration.
- `build_raptor`: take the existing SAC leaves, embed, recursively cluster (GMM/UMAP per
  the RAPTOR paper, or a simpler agglomerative fallback), LLM-summarize each cluster into
  a parent node, repeat until one root or a level cap. Summary generation reuses the
  deterministic-cache discipline from ADR-0005 (content-addressed, prompt-versioned).
  Upload summary nodes with a `level` field.
- `build_graph`: per chunk, LLM extraction of entities + relationships (typed, with
  descriptions); merge duplicate entities; build a networkx graph; Leiden community
  detection (graspologic) at one or more levels; LLM community reports per community;
  upload to the three graph indexes.

Each builder logs counts and failures and never blocks the whole ingest on one LLM
failure (mirrors the existing `skip_reasons`/fallback pattern). Builders are idempotent
and use in-place upsert + prune (ADR-0007).

### 4. Index reuse and the Foundry binding

The existing contextual index already holds SAC-decorated leaves and is bound to a Foundry
knowledge source (ADR-0007, can't be freely deleted). Decision: the RAPTOR+SAC substrate
gets its own dedicated `raptor-sac` index rather than mutating the Foundry-bound index, so
RAPTOR summary nodes never leak into the live generator's knowledge source. The SAC leaves
are re-uploaded into `raptor-sac` (cheap, decoration is cache-hit), and RAPTOR summaries
go on top with a `level` field. The Foundry-bound contextual index stays exactly as it is.
This trades a little storage duplication for a clean boundary, which matters more for a
demo than the storage cost on a 584-page corpus.

### 5. Generalized pipeline state (`ragpipe/models.py`, `ragpipe/workflow.py`)

`PipelineState`'s fixed `dense`/`bm25`/`fused`/`reranked` fields don't fit RAPTOR levels,
graph local/global, or agentic iterations. Replace them with an ordered
`stages: dict[str, StageResult]`, where `StageResult` carries the captured chunks/urls for
that stage. Each substrate names its own stages (e.g. `dense`, `bm25`, `fused` for
Baseline; `local`, `global`, `fused` for GraphRAG; `iter_0`, `iter_1`, ... for agentic).
The reranked/final stage keeps a stable well-known name (`reranked`) so the gate, harness,
and dashboard always find the final set.

The eval harness, the viz workflow (`build_viz_workflow`), and the dashboard read stage
names dynamically from `state.stages` instead of hard-coded attributes. This is the
largest blast-radius change; it lands in Phase 1 before any new substrate so later phases
just register stages.

### 6. The agentic wrapper (`ragpipe/retrieval/agentic.py`)

`AgenticSubstrate(inner: RetrievalSubstrate)` implements `RetrievalSubstrate`, so it
composes over any of the four. Built on the Microsoft Agent Framework already in the stack
(`agent-framework`, `FoundryAgent`). The agent:

1. Plans: decompose the query into sub-queries if it looks multi-part.
2. Loops: for each sub-query (and any follow-up it decides on), call `inner.retrieve` as a
   tool, accumulate candidates, reflect on whether coverage is sufficient.
3. Stops: on a sufficiency judgement, or at `agentic_max_iterations` (config, default 3),
   whichever first. Bounded so it can never loop forever.

Returns the accumulated, de-duplicated candidate list to the normal rerank → generate →
gate tail. The faithfulness gate stays the final arbiter; the agentic loop is purely a
retrieval-side amplifier. Each iteration records a stage (`iter_N`) for visibility.

### 7. Configuration (`ragpipe/config.py`)

- `RetrievalMode` enum with 8 values (or a `Substrate` enum + `agentic: bool`; the spec
  uses the latter internally and exposes a flat 8-value name on the API for clarity).
- Per-substrate index names: `baseline_index`, `raptor_sac_index`,
  `graph_entities_index`, `graph_relationships_index`, `graph_communities_index`.
- `agentic_max_iterations` (default 3), `raptor_max_levels` (default 3),
  `graph_community_levels` (default 1).
- Existing knobs (`top_k`, `rrf_k`, `candidate_pool`, guardrail threshold) unchanged and
  shared across modes so the comparison is fair.

### 8. Eval harness: the mode axis (`ragpipe/eval/harness.py`, `eval/run.py`)

New axis alongside the existing tag axis (ADR-0006). `run.py` takes a set of modes (CLI
arg, default all 8 or a configured subset), runs the same `testset.jsonl` through each
mode's `pipeline_fn`, and aggregates `means_by_mode` keyed by mode name. `eval_results.json`
becomes keyed by mode at the top level, each holding the existing `means`, `means_by_tag`,
`coverage`, `records` structure.

`RETRIEVAL_STAGES` stops being a fixed tuple; the harness reads stages from each record's
`state.stages`. The deterministic `hit_rate@stage` / `mrr@stage` metrics (ADR-0002) are
the primary cross-mode comparison signal because they are exact, reproducible, and
LLM-free. RAGAS metrics (faithfulness, answer_relevancy, context precision/recall) stay as
a complement, per mode.

### 9. Surfaces (`app/api.py`, `app/dashboard.py`)

- `/query` gains a `mode` param (one of the 8 names); defaults to the current contextual
  mode for backward compatibility.
- New `/compare`: takes a query + list of modes, runs each, returns per-mode answer,
  per-stage contexts, and scores, side by side. This is the endpoint the website calls for
  the research story.
- Dashboard: a mode selector and a comparison view reading `means_by_mode`.
- The single cached `_pipeline_fn` becomes a small per-mode cache (build once per mode,
  reuse).

### 10. Error handling

- Substrate retrieve failure: logged, returns empty candidates for that stage; the gate's
  abstention path (ADR-0009) already handles no-context answers.
- Agentic loop: any tool/iteration error is caught, the loop stops early with whatever it
  has, and the failure is counted; never blocks the response.
- Graph build: extraction failures per chunk are counted and skipped; a chunk with no
  entities just contributes none. Community detection on an empty/tiny graph degrades to
  one trivial community.
- RAPTOR build: a cluster whose summary generation fails falls back to concatenated child
  titles, counted and logged.
- Missing index for a requested mode: the API returns a clear error naming the mode and
  the missing index, rather than a generic 500.

### 11. Testing

- Seam contract test: every registered substrate runs the same contract test against a
  fake Azure AI Search client (returns ranked candidates, respects `k`, handles empty).
- Baseline: dense/BM25/RRF wiring over a fake store (mirrors `tests/retrieval/`).
- RAPTOR: clustering + summary tree build over a stubbed LLM and embedder; level tagging;
  collapsed-tree query returns nodes across levels.
- GraphRAG: extraction parsing, entity merge, community detection on a known small graph,
  local-search 1-2 hop expansion via in-memory adjacency, global-search community ranking.
- Combined: fuses two stubbed substrates, RRF order correct.
- Agentic: planner/loop over a fake inner substrate, iteration cap enforced, early stop on
  sufficiency, error path.
- Harness: same testset through two fake modes → `means_by_mode` keyed correctly; dynamic
  stage reading.
- One live smoke per substrate before any full build: tiny corpus subset end to end.

### 12. Phasing (drives the implementation plan)

- **Phase 1 — seam + Baseline.** `RetrievalSubstrate`, generalized `PipelineState`/stages,
  `RetrievalMode` config, registry, refactor the current pipeline behind the seam, build
  the Baseline substrate, teach the harness + dashboard the mode axis, add `/query?mode`
  and `/compare`. End state: baseline vs the existing contextual mode are comparable.
- **Phase 2 — SAC + RAPTOR.** `build_raptor`, `raptor-sac` index, collapsed-tree
  substrate, register modes 3 and 4's substrate (non-agentic first).
- **Phase 3 — GraphRAG.** `build_graph`, the three graph indexes, local/global substrate,
  register mode 5's substrate.
- **Phase 4 — Combined + Agentic.** Combined substrate (modes 7), then the agentic wrapper
  applied across all four substrates (modes 2, 4, 6, 8).

Surfaces and eval get the mode axis in Phase 1; each later phase just registers its modes.

### 13. ADRs (`docs/adr/`)

Written as the first step of Phase 1, before any substrate code, so the decision log
exists before implementation:

- `0011-retrieval-substrate-seam.md` — the `RetrievalSubstrate` interface, substrate-owned
  fusion, generalized `PipelineState` stages, mode registry.
- `0012-raptor-collapsed-tree-on-azure-search.md` — RAPTOR summary nodes with a `level`
  field, collapsed-tree retrieval, dedicated `raptor-sac` index (Foundry-binding rationale).
- `0013-flat-graphrag-on-azure-search.md` — materialized graph as three flat indexes,
  in-memory adjacency, local/global search, no graph DB.
- `0014-agentic-retrieval-wrapper.md` — agentic loop on Agent Framework over a common
  substrate interface, bounded iterations, gate stays final arbiter.
- `0015-multi-mode-evaluation-axis.md` — mode axis in the harness, deterministic URL-match
  metrics as primary cross-mode signal, fair-comparison shared knobs.

## Rejected alternatives

- **One shared index with a `strategy` discriminator field.** Cheapest infra, but RAPTOR
  levels and (especially) graph entities/edges don't model cleanly in one flat index, and
  rebuilding one tier risks the others. Per-substrate indexes isolate failure and rebuild.
- **Cosmos DB (Gremlin) for the graph.** A real graph DB with native traversal, but it is
  a new Azure resource, new cost, new auth path, and overkill: GraphRAG consumes the graph
  at build time and only needs 1-2 hop expansion at query time, which an in-memory
  adjacency over flat rows serves fine. Revisit only if deep online traversal is needed.
- **Microsoft `graphrag` package / a RAPTOR reference lib.** Less code, but each brings its
  own pipeline, config, and storage (parquet/LanceDB) that won't slot into the Azure AI
  Search + RAGAS harness without an adapter layer, and diverges from the hand-rolled,
  Azure-native style of the repo. Hand-rolling keeps every tier inside one testable shape
  the eval harness already understands.
- **Azure AI Search built-in agentic retrieval (knowledge agent).** Server-side query
  planning with less code, but it only works over Azure AI Search indexes, so it can't
  uniformly wrap the GraphRAG substrate. That would split the agentic abstraction into two
  implementations and break the "one wrapper over a common `retrieve`" property that keeps
  the 8-mode matrix cheap.
- **Baking agentic into a tier instead of a wrapper.** Would force a separate
  implementation per substrate; the orthogonal wrapper is what makes 8 modes cost 4
  substrates + 1 wrapper.

## Sources

- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* —
  https://arxiv.org/abs/2401.18059
- Microsoft Research, *GraphRAG: A modular graph-based RAG system* —
  https://arxiv.org/abs/2404.16130
- Microsoft, *GraphRAG solution accelerator* (flat graph stored in Azure AI Search) —
  https://github.com/Azure-Samples/graphrag-accelerator
- Anthropic, *Introducing Contextual Retrieval* (SAC / contextual embeddings) —
  https://www.anthropic.com/news/contextual-retrieval
- Traag, Waltman & van Eck, *From Louvain to Leiden: guaranteeing well-connected
  communities* — https://arxiv.org/abs/1810.08473
- *Document-Level Retrieval Mismatch* (motivation for richer retrieval) —
  https://arxiv.org/abs/2510.06999
- Nygard, *Documenting Architecture Decisions* —
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
