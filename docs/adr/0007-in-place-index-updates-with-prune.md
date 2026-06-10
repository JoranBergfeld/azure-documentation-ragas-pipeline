# 0007 — In-place index updates with stale-document pruning

**Status:** Accepted (backfilled; decided 2026-05-30, recorded 2026-06-10)

## Context

A Foundry knowledge source (`ks-searchindex-311`) binds to the Azure AI Search
index, so `delete_index` returns `400 CannotDeleteIndex` — the common
"drop and rebuild" re-indexing pattern is unavailable. (An earlier implementation
swallowed that error with a bare `except`, which silently left the index stale
across several ingests.) Uploads are upserts keyed on `id`, so re-ingesting a
smaller or re-chunked corpus leaves orphaned chunks behind, inflating the document
count and surfacing dead content in retrieval.

## Decision

- Index schema changes go through `create_or_update_index` (idempotent, in-place,
  additive).
- After every upload, `prune_stale_documents` enumerates indexed ids and deletes
  those absent from the freshly-uploaded set.

Verified live 2026-05-30: pruned 106 stale chunks; index count matched the fresh
corpus exactly (2723).

## Alternatives rejected

- **Delete and recreate the index**: blocked by the knowledge-source binding;
  unbinding/rebinding the Foundry knowledge source on every ingest would couple
  corpus refreshes to agent-service configuration.
- **Versioned indexes with alias swap** (blue/green): the clean general solution,
  but the knowledge source pins a concrete index name, and a second index doubles
  storage; revisit if zero-downtime schema-breaking changes are ever needed.

## Consequences

- Schema evolution must stay additive (new fields only) — fits this spec's new
  `context` field. Breaking field changes (type/analyzer/vector dimensions) would
  force the blue/green alternative.
- Re-chunking changes (ADR-0004) that shift every chunk id are safe: the first
  re-ingest uploads the new set and prunes the entire old one.
- Prune is not atomic with upload; a failed run can briefly leave both sets — the
  next successful ingest converges.

## Sources

- Azure AI Search, *Update or rebuild an index* (which changes require rebuild;
  drop-and-rebuild pattern) —
  https://learn.microsoft.com/azure/search/search-howto-reindex
- Project verification: live ingest 2026-05-30 ("pruned 106 stale chunks",
  count 2829 → 2723), recorded in `src/ragpipe/ingest.py` /
  `src/ragpipe/search_index.py` docstrings.
