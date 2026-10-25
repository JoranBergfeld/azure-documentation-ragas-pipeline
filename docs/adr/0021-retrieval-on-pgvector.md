# 0021 — Retrieval on pgvector in homelab's Postgres

**Status:** Accepted (2026-10-24), amended 2026-10-25 for the reranker (#40)

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
   service. Its replacement, a cross-encoder on vLLM on the DGX Spark, wasn't up yet when
   this was written. See the amendment below for what's serving.

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

## Amendment (2026-10-25): the Spark reranker (#40)

The reranker is up, and it isn't the `bge-reranker-v2-m3` named above. Homelab serves
`nvidia/llama-nemotron-rerank-1b-v2` (1B cross-encoder, 8192-token context, commercial-OK
licence) on NVIDIA's `nvcr.io/nvidia/vllm:26.02-py3` image (vLLM 0.15.1). The reason is
homelab's NVIDIA-first rule for NVIDIA hardware, not a bake-off. Nobody benchmarked the two
on this corpus. The re-eval of the three modes is the first real check.

**The three modes come back on it.** `retrieval/cross_encoder.py` posts to `/v1/rerank`
with the fused candidates and keeps vLLM's order and `relevance_score`. Not bare `/rerank`:
vLLM answers there too but skips the API key, and only the firewall keeps it shut. Document text
copies the old semantic configs so the swap is like for like: title + `content` + `context`
for `ms_docs` and `raptor_sac`, title + `content` for `baseline`. The `question:` /
`passage:` template is applied server side, so we send plain text. `Chunk` grew a `context`
field for this. It's retrieval-only, the generator still never sees it (ADR-0003).

**Available means reachable.** `unavailable_reason()` no longer hard-codes
`reranker-offline`. The rerank modes are available when `RERANKER_URL` is set and `/health`
answers 200. The probe has a 2 s timeout and its answer is cached for 60 s, so `/modes` stays
cheap and greys the modes out within a minute of the Spark going down.

**A mid-query failure is `ModeUnavailable`, not passthrough.** Calls get a 15 s timeout and
two retries with backoff on timeouts, connection errors, 429 and 5xx. Anything else, or
running out of retries, raises `ModeUnavailable("reranker-offline")`: `/run` and `/compare`
answer the same 503 as the up-front check, `/run/stream` sends an error frame with the
reason, and the eval drops the mode instead of writing a half-scored file. The failure also
marks the reranker down in the health cache. Falling back to score-sorting would quietly run
a different pipeline under the same name, the argument this ADR already made.

**No usage entry.** The Spark is homelab hardware and nobody bills it, so a rerank call adds
nothing to the run's usage (ADR-0020), same as the passthrough reranker.

**Their re-evaluated numbers won't compare like for like** with the Azure-era files:
different reranker, different first-stage BM25. The `docs/eval-history.md` entry for that run
has to say so.

### Rejected (amendment)

- **`bge-reranker-v2-m3`**, as planned. Fine model, just not what homelab serves on NVIDIA
  hardware.
- **Probing `/health` on every `/modes` call.** One more network hop per page load for a
  status that changes on the scale of minutes.
- **Retrying forever / long timeouts.** A request stuck on a dead Spark holds a worker and
  the user's spinner. Fail fast and say why.

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
- NVIDIA, `llama-nemotron-rerank-1b-v2` model card (vLLM serving, question/passage template):
  https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2
- vLLM docs, Cohere-compatible rerank API (`/rerank`, `/v1/rerank`):
  https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#cohere-rerank-api
- Azure AI Search, *Configure semantic ranker* (title, content and keyword fields, in
  priority order): https://learn.microsoft.com/en-us/azure/search/semantic-how-to-configure
