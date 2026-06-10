# Preprocessing: contextual chunk decoration + deterministic retrieval metrics

**Date:** 2026-06-10
**Status:** Approved
**Origin:** First of a series of specs derived from the 2026-06-10 design/implementation
review. This spec covers the preprocessing track: structure-preserving extraction,
contextual chunk decoration, deterministic retrieval metrics, and test-set expansion.
Out of scope here (later specs): the hybrid-rerank candidate-drop bug, refusal/judge-error
routing in the guardrail loop, judge-model independence, and candidate-pool widening.

## Problem

Three findings from the review motivate this work:

1. **Ingest destroys document structure.** `html_to_text` whitespace-flattens entire
   MS Learn pages (`" ".join(get_text().split())`), so code blocks, tables, and the
   heading hierarchy are gone before chunking starts. For Azure docs — where the payload
   is often a CLI snippet or a limits table — this is the dominant content-quality defect.
2. **Chunks lose their document identity.** The corpus is ~584 structurally similar pages
   across ~199 services ("What is X", "Quickstart X"). Once chunked, fragments from
   different services look nearly identical. This is the *Document-Level Retrieval
   Mismatch* (DRM) failure mode described in arXiv:2510.06999: the retriever returns a
   plausible chunk from the wrong document.
3. **Retrieval quality is measured only by LLM judges.** `TestItem.ground_truth_context`
   (a URL) is loaded and never used, while context metrics are LLM-judged (noisy,
   non-reproducible, costly). There is no deterministic measurement, and the 16-item
   test set shows recall=1.0 at every stage — no headroom to detect improvement.

## Goals

- Preserve document structure (markdown, code fences, tables, headings) through ingest.
- Decorate every chunk with document-level context so retrieval can distinguish
  lookalike chunks, without contaminating what the generator and faithfulness judge see.
- Add deterministic (LLM-free, exact, reproducible) retrieval metrics per stage.
- Expand the test set so preprocessing gains are measurable, with a baseline-before/
  after protocol.
- Record load-bearing decisions as ADRs with sources (`docs/adr/`).

## Non-goals

- No changes to retrieval topology, rerank query, guardrail loop, judge model, or
  `top_k` (separate specs).
- No RAPTOR-style hierarchical summarization or cross-document synthesis
  (YAGNI for single-page Q&A; see Rejected alternatives).

## Design

### 1. Structure-preserving extraction (`ragpipe/ingest.py`)

Replace `html_to_text` with extraction that:

- Locates the MS Learn main-content container via BeautifulSoup.
- Converts it to markdown with `markdownify`, preserving fenced code blocks, tables,
  and `#`-level headings.

Pure function; unit-tested against saved HTML fixtures (a real MS Learn page snapshot
committed under `tests/fixtures/`).

### 2. Heading-aware chunking (`ragpipe/chunking.py`)

- Split the page markdown on heading boundaries first.
- Size-bound long sections with the existing character-window splitter
  (2000 chars / 200 overlap unchanged).
- Each chunk carries a **breadcrumb**: `page title > H2 > H3` path metadata.

Deterministic, no LLM. The breadcrumb is the guaranteed decoration floor.

### 3. Contextual decoration (new `ragpipe/context_gen.py`)

Per chunk, one chat-model call (gpt-4o, `temperature=0`) with the full page markdown +
the chunk, using Anthropic's published situating prompt, producing 1–2 sentences that
situate the chunk within its document.

Deterministic controls around the one non-deterministic step:

- **Content-addressed cache**: keyed on
  `sha256(page_markdown + chunk_text + prompt_version)`, persisted to a local JSON file
  (gitignored). Unchanged chunks cost zero LLM calls on re-ingest; a full unchanged
  corpus re-ingest is LLM-free.
- **Fallback floor**: if generation fails after bounded retries, the chunk is decorated
  with breadcrumb only, and the failure is counted and logged. Ingest never blocks on
  the LLM.
- **Cost note**: ~2,700 calls on first full ingest; Azure OpenAI prompt caching applies
  because the (large) page markdown is the shared prompt prefix across all chunks of a
  page.

### 4. Index schema (`ragpipe/search_index.py`)

- New **searchable `context` field**: `breadcrumb + "\n" + generated_context`.
- `context` joins the BM25-searchable fields (contextual BM25) and the semantic
  configuration's prioritized content fields.
- **Embedding input** becomes `context + "\n\n" + content` (contextual embeddings).
- `content` stays clean: the generator prompt and the RAGAS faithfulness judge keep
  seeing undecorated chunk text. Rationale: repeating near-identical decoration across
  the 5 retrieved chunks wastes generator context, and the judge must score claims
  against chunk bodies, not against summary text (see ADR-0003).

Schema change is additive — `create_or_update_index` handles it in place (the index
cannot be deleted; a Foundry knowledge source binds to it). Chunk ids shift because
chunk boundaries change; the existing `prune_stale_documents` removes the stale ones.

### 5. Deterministic retrieval metrics (new `ragpipe/eval/retrieval_metrics.py`)

Pure functions, zero LLM calls:

- `hit_rate` per stage: 1 if any chunk in the stage's captured list (its top-k,
  k = `settings.top_k`) has `url == ground_truth_context`, else 0.
- `mrr` per stage: reciprocal rank of the first matching URL in that list.
- Stages: `dense`, `bm25`, `fused`, `reranked`. Metric names contain no `@`; they are
  reported via the existing `metric@stage` key convention (e.g. `hit_rate@dense`,
  `mrr@reranked`) so `aggregate()`, coverage, the dashboard, and `/eval` pick them up
  unchanged.

`EvalRecord` additionally captures per-stage chunk URLs (alongside the existing
`stage_contexts`). RAGAS metrics remain as a complement; the deterministic metrics are
the primary regression signal.

### 6. Test-set expansion (`data/testset.jsonl`)

- Grow 16 → ~40–50 items. `TestItem` gains an optional `tags: list[str]`
  (backward compatible; absent tag implies `original`).
- The existing 16 keep tag `original`.
- ~10 new hand-authored **hard** items, tagged:
  - `paraphrase`: low lexical overlap with the target page's vocabulary.
  - `lookalike`: answer lives on one of several near-identical service pages (the DRM
    case, e.g. consistency levels across Cosmos DB vs other stores).
- Remainder generated with the existing synthetic mode, manually screened, then
  committed (tag `synthetic`).
- Aggregation reports means per tag group as well as overall.

### 7. Measurement protocol (baseline before treatment)

1. Land metrics (§5) + test set (§6) with no pipeline changes.
2. **Baseline run** against the current index → committed as `eval_baseline.json`.
3. Land extraction (§1), chunking (§2), decoration (§3), schema (§4); re-ingest.
4. Post-change run → compare.

**Success criteria:** `hit_rate` and `mrr` (per stage) improve on the `paraphrase` and
`lookalike` subsets; no regression on `original`; RAGAS faithfulness mean does not
degrade. If decoration shows no gain on the hard subsets, ADR-0001 is revisited
(SAC and breadcrumb-only are the documented cheaper fallbacks).

### 8. Error handling

- Context generation: bounded retries → breadcrumb-only fallback, counted and logged
  (a high fallback rate must be visible in ingest output, mirroring the existing
  `skip_reasons` pattern).
- Extraction: a page whose main-content container is not found falls back to the
  current whole-page text path, counted and logged.
- Cache file corrupt/missing: treated as empty cache; never fails ingest.

### 9. Testing

- Extraction: HTML fixtures → expected markdown (code fences, tables, headings intact).
- Chunking: heading-boundary splits, breadcrumb assembly, size bounds, overlap.
- Decoration: `context_gen` with a stubbed LLM callable — cache hit/miss, fallback path,
  prompt-version invalidation.
- Metrics: synthetic ranked lists with known URLs → exact `hit_rate@k`/`MRR` values.
- Test-set loader: `tags` parsing, backward compatibility with untagged rows.
- One live smoke before full re-ingest: 3-page subset end-to-end (ingest → query →
  decorated `context` visible in the index, `content` clean).

### 10. ADRs (`docs/adr/`)

- `README.md` — convention: Nygard format (Context / Decision / Consequences) plus a
  mandatory **Sources** section; sequential numbering.
- `0001-contextual-chunk-decoration.md` — per-chunk contextual retrieval over SAC and
  breadcrumb-only; expected gains, costs, fallbacks.
- `0002-deterministic-retrieval-metrics.md` — URL-match hit-rate/MRR as primary signal
  over LLM-judged context metrics.
- `0003-decoration-isolated-from-generator-context.md` — decorations are visible to
  retrieval (BM25, embeddings, semantic ranker) but not to the generator or the
  faithfulness judge.

## Rejected alternatives

- **SAC — per-document summary prepended to all chunks**
  ([arXiv:2510.06999](https://arxiv.org/abs/2510.06999)): ~5× cheaper (584 vs ~2,700
  calls) and directly targets DRM, but per-chunk context subsumes it and has the
  strongest published evidence: Anthropic reports a 35% reduction in top-20 retrieval
  failure from contextual embeddings alone and 49% combined with contextual BM25
  ([Anthropic, *Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval)).
  The scale/cost concerns raised by
  [*Reconstructing Context* (arXiv:2504.19754)](https://arxiv.org/abs/2504.19754)
  do not bite at 584 pages with a content-addressed cache. SAC remains the documented
  fallback if measured gains don't justify cost.
- **Breadcrumb-only (no LLM)**: kept as the deterministic floor and failure fallback,
  but a breadcrumb states *where* a chunk lives, not *what the page is about*, so it is
  not expected to resolve lookalike confusion on its own.
- **Late chunking** ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)): requires
  token-level embeddings, which `text-embedding-3-small` on Azure OpenAI does not
  expose. Not feasible on this stack.
- **RAPTOR** ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)): hierarchical
  summary trees help cross-document synthesis questions; this corpus's Q&A is
  single-page. Disproportionate architectural change.
- **Semantic chunking**: fixed-size/heading-bounded chunking retained —
  [Qu, Tu & Bao (arXiv:2410.13070)](https://arxiv.org/abs/2410.13070) find semantic
  chunking's gains don't justify its cost on real-world documents.

## Sources

- Anthropic, *Introducing Contextual Retrieval* (2024) —
  https://www.anthropic.com/news/contextual-retrieval
- *Towards Reliable Retrieval in RAG Systems for Large Legal Datasets* (SAC / DRM) —
  https://arxiv.org/abs/2510.06999
- *Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG* —
  https://arxiv.org/abs/2504.19754
- *Late Chunking: Contextual Chunk Embeddings* — https://arxiv.org/abs/2409.04701
- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* —
  https://arxiv.org/abs/2401.18059
- *Is Semantic Chunking Worth the Computational Cost?* —
  https://arxiv.org/abs/2410.13070
- Thakur et al., *BEIR* (exact-judgment IR evaluation precedent) —
  https://arxiv.org/abs/2104.08663
- Nygard, *Documenting Architecture Decisions* —
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
