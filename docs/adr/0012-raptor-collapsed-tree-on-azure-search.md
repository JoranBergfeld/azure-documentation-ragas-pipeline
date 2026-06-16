# 0012 — RAPTOR: collapsed-tree retrieval on Azure AI Search

**Status:** Accepted (2026-06-16)

## Context

The existing pipeline stores SAC-decorated chunks in the contextual index and searches
them with a flat hybrid query. That works, but it misses structure that exists in the
corpus: a long policy document has sections that are meaningfully summarizable, and a
search over raw chunks will miss queries whose intent matches a high-level section rather
than any individual passage.

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) addresses this
by building a tree of LLM summaries over clusters of chunks, then searching across
all tree levels at once at query time. The paper reports collapsed-tree retrieval as
matching or beating tree-traversal on most benchmarks, and it's much simpler to
implement on a flat search index.

The RAPTOR+SAC substrate needs an index. The existing contextual index is already bound
to a Foundry knowledge source (ADR-0007) and cannot be freely altered without affecting
the live generator's knowledge source.

## Decision

1. **Summary nodes with a `level` field.** Level 0 leaves are the existing SAC-decorated
   chunks. RAPTOR adds levels 1+ built at ingest by recursive cluster-then-summarize:
   embed leaves, cluster with GMM/UMAP (or agglomerative fallback), LLM-summarize each
   cluster into a parent node, repeat until one root or a level cap. Each node carries a
   `level` field so query-time and eval can distinguish leaves from summaries.

2. **Collapsed-tree retrieval.** At query time, run one hybrid search (dense + BM25)
   across all levels in one shot. No tree traversal. This is what the RAPTOR paper calls
   "collapsed tree" and it's the approach the paper recommends for its simplicity and
   competitive performance. The semantic reranker then runs over the mixed-level candidate
   list as normal.

3. **Dedicated `raptor-sac` index.** The RAPTOR+SAC substrate gets its own Azure AI
   Search index rather than mutating the Foundry-bound contextual index. SAC leaves are
   re-uploaded into `raptor-sac` (cheap: decoration is a cache-hit per ADR-0005), and
   RAPTOR summary nodes go on top. The Foundry-bound contextual index stays exactly as
   it is. This trades a small amount of storage duplication for a clean boundary, which
   matters more for a demo than the storage cost on a 584-page corpus.

4. **Build path is idempotent and independent.** `build_raptor` runs separately from
   other substrate builders. Summary generation uses the content-addressed,
   prompt-versioned cache discipline from ADR-0005. Failures per cluster fall back to
   concatenated child titles and are counted; the build never blocks on a single LLM
   failure.

## Alternatives rejected

- **Mutate the existing contextual index with a `level` field.** Avoids a new index,
  but RAPTOR summaries would then appear in the Foundry knowledge source, leaking
  synthetic text into the live generator's retrieval pool. The Foundry binding is the
  hard constraint here.
- **Microsoft `graphrag` or a RAPTOR reference library.** Less code, but both bring
  their own pipeline, config, and storage (parquet/LanceDB) that won't slot into the
  Azure AI Search + RAGAS harness without an adapter layer. Hand-rolling keeps every
  tier inside one testable shape the harness already understands.
- **Tree-traversal retrieval (top-down or bottom-up).** More faithful to the RAPTOR
  paper's original approach, but more complex to implement over a flat search index and
  the paper itself reports collapsed-tree as competitive. Simpler wins when the goal is
  a fair comparison demo.

## Consequences

- One extra Azure AI Search index (`raptor-sac`). Storage cost on a 584-page corpus
  is low; build time is the bigger concern (one LLM call per cluster per level).
- The `level` field in the index schema needs to propagate to the `Chunk` model so the
  harness can report per-level retrieval stats if we want them later.
- Summary generation reuses the ADR-0005 cache, so repeated builds after the first are
  fast. First build on a new corpus will be slow (one summarization call per cluster).
- Collapsed-tree means level-0 chunks and level-2 summaries can appear side by side in
  the candidate list. The reranker handles mixed granularity fine in practice, but it is
  worth watching in eval whether summary nodes crowd out leaf-level evidence.

## Sources

- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* —
  https://arxiv.org/abs/2401.18059
- Anthropic, *Introducing Contextual Retrieval* (SAC / contextual embeddings) —
  https://www.anthropic.com/news/contextual-retrieval
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §2
  (SAC+RAPTOR para), §3 (`build_raptor`), §4 (index reuse and Foundry binding)
