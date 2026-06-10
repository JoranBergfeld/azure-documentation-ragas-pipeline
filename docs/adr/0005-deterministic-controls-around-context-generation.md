# 0005 — Deterministic controls around LLM context generation

**Status:** Accepted (2026-06-10)
**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md`

## Context

Per-chunk context generation (ADR-0001) introduces a non-deterministic, paid,
fallible step into ingest — previously a fully deterministic pipeline. The project
principle is to keep behavior deterministic and reproducible wherever possible.

## Decision

Bound the non-determinism on every side:

- **Content-addressed cache**: generated context is cached keyed on
  `sha256(page_markdown + chunk_text + prompt_version)`, persisted locally
  (gitignored). Unchanged chunks never re-call the LLM; an unchanged corpus
  re-ingests with zero LLM calls. Bumping `prompt_version` deliberately invalidates
  everything.
- **`temperature=0`** for generation: not a determinism guarantee, but minimizes
  drift between cache rebuilds.
- **Fallback floor**: after bounded retries, a failed chunk is decorated with its
  deterministic breadcrumb only; the fallback count is logged in ingest output
  (mirroring the existing `skip_reasons` pattern). Ingest never blocks on the LLM.
- **Corrupt/missing cache** is treated as empty, never as an error.

## Alternatives rejected

- **No cache, regenerate every ingest**: ~2,700 calls per re-ingest, and chunk
  decorations would drift between ingests, making before/after comparisons
  (ADR-0006) muddier.
- **Committing the cache to git**: reproducible for others, but couples the repo to
  a large generated artifact; revisit if collaboration needs it.
- **Hard-fail ingest on generation errors**: turns a transient 429 into a blocked
  corpus refresh; the breadcrumb floor is strictly better.

## Consequences

- Ingest cost after the first run is proportional to corpus *change*, not corpus
  *size*.
- A visible fallback-rate counter makes silent quality degradation (many chunks
  breadcrumb-only) diagnosable.
- The cache file is environment-local; a fresh clone pays one full generation pass.

## Sources

- Anthropic, *Introducing Contextual Retrieval* (cost note: prompt caching across
  per-chunk calls sharing the document prefix) —
  https://www.anthropic.com/news/contextual-retrieval
- Azure OpenAI prompt caching —
  https://learn.microsoft.com/azure/ai-services/openai/how-to/prompt-caching
