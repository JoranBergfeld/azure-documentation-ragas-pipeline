# 0012 — Retrieval substrate seam

**Status:** Accepted (2026-06-16)

## Context

The pipeline is wired around one retrieval topology: dense + BM25 via RRF, then semantic
rerank. That topology is embedded in `app_wiring.py` and `PipelineState` by name
(`dense`, `bm25`, `fused`, `reranked` as fixed fields). There is no place to plug in a
different retrieval strategy without touching the generate/gate tail. And the
`PipelineState` model silently breaks for anything that doesn't produce those exact four
stages, which RAPTOR levels, local/global graph search, and agentic iterations definitely
won't.

The project now exists to compare retrieval strategies head to head, so we need a seam
between retrieval and everything downstream, and a generalized way to track what happened
at each stage so the eval harness can read it.

## Decision

1. **One interface.** Every retrieval strategy implements `RetrievalSubstrate`:
   ```python
   class RetrievalSubstrate(Protocol):
       name: str
       async def retrieve(self, query: str, k: int) -> list[Chunk]: ...
   ```
   Everything downstream (rerank, generate, faithfulness gate, retry loop) stays
   exactly as it is. The substrate is the only thing that varies.

2. **Substrate-owned fusion.** The pipeline no longer owns RRF. If a substrate does
   dense + BM25 fusion internally, that's its business. The Baseline substrate does
   exactly that; GraphRAG does local + global fusion inside itself. The reranker still
   runs over whatever candidates the substrate returns.

3. **Generalized `PipelineState` stages.** Replace the fixed `dense`/`bm25`/`fused`/
   `reranked` fields with `stages: dict[str, StageResult]`. Each substrate names its own
   stages. The final stage keeps the stable name `reranked` so the gate, harness, and
   dashboard always find the final set without knowing which substrate ran. The eval
   harness and viz workflow read stage names dynamically instead of hard-coding them.

4. **Mode registry.** `build_pipeline_fn(settings, mode)` selects a substrate from a
   registry keyed by mode name, then builds the same `PipelineDeps` around it. Adding a
   new substrate is: implement the protocol, register the name, done.

5. **8 modes = 4 substrates x agentic toggle.** The four substrates are Baseline,
   RAPTOR over Anthropic-contextual leaves, GraphRAG, and Combined. The agentic wrapper
   composes over any of them as a fifth orthogonal concern. Configuration exposes a flat
   8-value mode name on the API for clarity; internally it is `Substrate` enum +
   `agentic: bool`.

## Alternatives rejected

- **Keep fixed fields, add substrate-specific ones with Optional.** Avoids the model
  change, but every consumer that reads stages would need `getattr` guards, and the
  harness would still need special-casing per substrate. The dynamic dict is cleaner.
- **One monolithic retriever with a strategy enum.** Less infra, but the stages,
  indexes, and internal logic are different enough that a single class becomes a
  switch statement, not an abstraction.
- **One shared index with a `strategy` discriminator field.** Cheapest storage, but
  RAPTOR levels and graph entities don't model cleanly in one flat index, and rebuilding
  one tier risks corrupting the others. Per-substrate indexes isolate failure and rebuild.

## Consequences

- The generalized `PipelineState` is the largest blast-radius change in Phase 1. It
  lands before any new substrate so later phases only register new modes and don't touch
  the model.
- Any code that read `state.dense`, `state.bm25` etc. by attribute breaks. The
  migration is mechanical but has to happen in one shot.
- Adding a new substrate later costs: implement the protocol (roughly one file), add
  index config, register. No changes to the generate/gate tail.
- The mode registry makes the 8-mode matrix cheap: 4 substrates + 1 wrapper, not 8
  independent implementations.

## Sources

- *Document-Level Retrieval Mismatch* (motivation for richer retrieval) —
  https://arxiv.org/abs/2510.06999
- Nygard, *Documenting Architecture Decisions* —
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §1, §5
