# 0021 — Retrieval on pgvector in homelab's Postgres

**Status:** Accepted (2026-10-24)

## Context

`ragpipe-search` is an Azure AI Search S1 at about €210 a month, for a demo that barely
runs. personal-website ADR 0002 moves retrieval onto pgvector in homelab's shared Postgres 16
(the `rag` database, homelab #186). The six indexes were exported on 2026-10-24, but without
vectors: every vector field was `retrievable: false`. Issue #37 is the port.

Three things Azure did for us need a replacement:

1. **k-NN on the chunk vectors.** Straightforward in pgvector.
2. **BM25.** Postgres's own `ts_rank` has no IDF: its docs say the ranking functions "do not
   use any global information". The `pg_search`/BM25 extensions want Postgres 17, and the
   shared instance is 16.
3. **The semantic ranker** on `contextual`, `baseline` and `raptor_sac`. It goes with the
   service. Its replacement, `bge-reranker-v2-m3` on vLLM on the DGX Spark, isn't up yet.

## Decision

**One table per former index**, in a `rag` schema, named after the index (`ms_docs`,
`baseline`, `raptor_sac`, `graph_entities`, `graph_communities`, `graph_relationships`).
Same columns as the export, plus `embedding vector(1536)`, a generated `tsv tsvector` over the
fields Azure marked searchable, and `doc_len`. The table names are fixed in
`ragpipe/pgstore.py`; the `*_INDEX` env vars and `SEARCH_ENDPOINT` are gone, replaced by one
`RAG_DATABASE_URL`.

**Dense:** HNSW with `vector_cosine_ops`, approximate. RAG has no clearance filter, so there's
no recall cliff from post-filtering an approximate scan.

**BM25, hand-rolled in SQL.** Okapi BM25 with Azure's defaults (k1 = 1.2, b = 0.75) and
Lucene's IDF, `ln(1 + (N - df + 0.5) / (df + 0.5))`. Query terms are the query's lexemes,
OR'ed, like Azure's `searchMode=any`. Document frequency is counted live from the GIN index,
one count per query term, so there's no stats table to go stale. The `english` text search
config stems, Azure's standard analyzer didn't, and fields are concatenated rather than scored
per field. So it's BM25, but not Azure's BM25 byte for byte. The re-eval is what says whether
that matters.

The first version of the query took ~850 ms when a term is in nearly every chunk ("azure"):
the planner guessed 40 matches instead of 7,195 and spilled a hash join to disk with every row's
content in tow. The shipped query cuts each matching tsvector down to the query's lexemes
(`setweight` + `ts_filter`) before unnesting, sorts only ids and scores, and fetches full rows
for the top k. `idf` is `MATERIALIZED`, because Postgres inlines a CTE used once and would redo
the df counts per row. Same rankings on every test query, ~25 ms.

**Graph hybrid search** (entities, communities) runs both legs and fuses with the existing
RRF, which is what Azure did internally for a query with `search_text` and a vector.
`graph_relationships` gets no vectors: `build_adjacency` only scans the text, so embedding
36k rows nobody queries would be money for nothing.

**The rerank modes are offline, not degraded.** `contextual`, `baseline` and `raptor_sac`
report `available: false, reason: "reranker-offline"` on `/modes`, `/run` refuses them with a
503, and the eval skips them. Running them on the passthrough reranker would work, but their
numbers would then describe a different pipeline under the same name. When the Spark serves the
reranker they come back and get re-evaluated.

**The service connects read-only**: every pooled session starts with
`default_transaction_read_only=on`, like the spatial layer's store, and the login is
`rag_service` (member of `rag_reader`). Only `python -m ragpipe.pg_import` writes, from the
laptop, as `rag`, from `RAG_IMPORT_DATABASE_URL`.

**The import** verifies every export file against `manifest.json` (sha256, count, complete,
unique ids) before it touches the database, then loads table by table in committed batches.
Rows already present are kept and not re-embedded, rows the export lacks are deleted. A rerun
converges on the export, never doubles rows, and a crash halfway resumes without paying twice.
`--replace` drops and reloads, for an export whose rows changed under the same ids.

### Rejected

- **One `chunks` table with an `index` column.** Reads fine, but BM25's N, avgdl and df would
  need a `WHERE index = …` everywhere, and the graph tables have different columns anyway.
- **A precomputed term-statistics table.** Faster to read, but it has to be rebuilt on every
  load and can silently drift. Live GIN counts are a few ms per term at this size.
- **`ts_rank`/`ts_rank_cd`.** No IDF, so "azure" would count as much as "durable".
- **Passthrough rerank for the three rerank modes in the meantime.** See above: it changes what
  their eval numbers mean. If that's wanted it should be its own decision, with its own label.

## Consequences

- Nothing in the running pipeline talks to `*.search.windows.net`. The `DefaultAzureCredential`
  service principal is still needed for Foundry (generation, embeddings, judges).
- Every eval figure from before this describes Azure's ranking. The six runnable modes are
  re-evaluated against Postgres and their `eval_results_<mode>.json` replaced. The three rerank
  modes' Azure-era files are removed rather than left on the card next to Postgres numbers;
  they're in git history.
- `paired_vs_baseline` is empty until `baseline` can run again, since that's the reference mode.
- Ingest still writes Azure AI Search (now with its own `SEARCH_ENDPOINT`). Until it learns to
  write Postgres, a fresh corpus means ingest → export → import. That's a follow-up, and it
  matters once the search service is deleted.

## Sources

- personal-website ADR 0002, *Retrieval on Postgres, not Azure Search*:
  https://github.com/JoranBergfeld/personal-website/blob/main/docs/adr/0002-retrieval-on-postgres-not-azure-search.md
- pgvector README, HNSW and `vector_cosine_ops`: https://github.com/pgvector/pgvector#hnsw
- PostgreSQL 16 docs, *Ranking Search Results* (ranking ignores global information):
  https://www.postgresql.org/docs/16/textsearch-controls.html#TEXTSEARCH-RANKING
- PostgreSQL 16 docs, `MATERIALIZED` CTEs: https://www.postgresql.org/docs/16/queries-with.html#QUERIES-WITH-CTE-MATERIALIZATION
- PostgreSQL 16 docs, `setweight` / `ts_filter`: https://www.postgresql.org/docs/16/functions-textsearch.html
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Lucene `BM25Similarity` (IDF form, k1/b defaults):
  https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
- Azure AI Search, *Configure BM25 relevance scoring* (k1 = 1.2, b = 0.75):
  https://learn.microsoft.com/en-us/azure/search/index-ranking-similarity
- Azure AI Search, *Hybrid search scoring (RRF)*:
  https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
