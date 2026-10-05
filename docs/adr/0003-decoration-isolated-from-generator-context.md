# 0003 — Decoration visible to retrieval, hidden from generator and judge

**Status:** Accepted (2026-06-10)
**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md`

## Context

Chunk decoration (breadcrumb + generated context, ADR-0001) must influence
retrieval. But the same text, if injected into the generator prompt and the RAGAS
faithfulness judge's contexts, has side effects: the top-5 retrieved chunks often
come from the same page, so the prompt would repeat near-identical decoration five
times (wasted context window), and the faithfulness judge would score answer claims
as "supported" by summary text rather than by actual document content — quietly
inflating the guardrail metric the pipeline loops on.

## Decision

Store decoration in a separate searchable `context` index field:

- **Visible to retrieval:** `context` joins the BM25-searchable fields (contextual
  BM25), the semantic ranker configuration, and the embedding input
  (`context + "\n\n" + content`).
- **Hidden from generation and judging:** `content` stays undecorated; the
  generator prompt and the faithfulness scorer receive `content` only.

## Alternatives rejected

- **Prepend decoration into `content`** (what per-document SAC and Anthropic's
  published recipe do): simpler (no schema change), but pays the duplication and
  judge-contamination costs above. The papers evaluate retrieval, not a
  faithfulness-guardrail loop, so they never face this interaction.
- **Embedding-only decoration** (decorate the vector, not BM25): loses the
  contextual-BM25 half of the published gains (35% → 49% failure reduction came
  from adding it).

## Consequences

- Index schema gains one field (additive change; see ADR-0007 for why in-place).
- Faithfulness scores before/after the decoration change remain comparable — the
  judge's input distribution is unchanged.
- The generator cannot cite the generated context text; citations keep pointing at
  real document content.

## Sources

- Anthropic, *Introducing Contextual Retrieval* (contextual BM25 + embeddings
  ablation) — https://www.anthropic.com/news/contextual-retrieval
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
  (faithfulness = claims supported by retrieved contexts; hence judge-input
  contamination matters) — https://arxiv.org/abs/2309.15217
