# 0004 — Heading-aware, fixed-size chunking over semantic chunking

**Status:** Accepted (2026-06-10)
**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md`

## Context

Ingest previously flattened whole pages to whitespace-normalized text before
slicing fixed 2000-char windows: code blocks, tables, and headings were destroyed,
and chunk boundaries ignored document structure. A replacement chunking strategy
was needed, and "semantic chunking" (embedding-similarity-driven boundaries) is the
fashionable candidate.

## Decision

1. Extract MS Learn main content to **markdown** (BeautifulSoup + markdownify),
   preserving fenced code, tables, and the heading hierarchy.
2. Split on **heading boundaries** first; size-bound long sections with the
   existing character-window splitter (2000 chars / 200 overlap unchanged).
3. Attach each chunk's heading breadcrumb (`page title > H2 > H3`) as metadata.

Fully deterministic; no LLM and no embedding calls in the chunker.

## Alternatives rejected

- **Semantic chunking**: Qu, Tu & Bao find its gains over fixed-size chunking are
  inconsistent on real-world documents and don't justify the computational cost.
  Heading boundaries in curated documentation already encode the topic shifts
  semantic chunking tries to detect statistically.
- **Keep flat-text fixed windows** (status quo): destroys exactly the content
  (CLI snippets, limits tables) that answers most Azure-docs questions.
- **Token-based windows**: more faithful to embedding-model limits, but adds a
  tokenizer dependency for no observed need; 2000 chars ≈ 500 tokens is well under
  `text-embedding-3-small`'s limit. Revisit only if chunks start truncating.

## Consequences

- Chunk ids shift (boundaries change), so the first re-ingest after this change
  prunes the entire old chunk set via `prune_stale_documents` (ADR-0007).
- Chunks align with sections, so breadcrumbs (ADR-0001's deterministic floor) are
  meaningful per chunk.
- Code-heavy chunks embed and retrieve as code, which should specifically improve
  how-to/CLI questions.

## Sources

- Qu, Tu & Bao, *Is Semantic Chunking Worth the Computational Cost?* —
  https://arxiv.org/abs/2410.13070
- *Reconstructing Context* (chunking-strategy comparison incl. structure-aware
  baselines) — https://arxiv.org/abs/2504.19754
