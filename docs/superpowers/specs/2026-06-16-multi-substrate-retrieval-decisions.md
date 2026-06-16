# Autonomous build decision log: multi-substrate retrieval

Joran went to bed and asked me to proceed without involving them, recording decisions for
morning review. This log captures every non-trivial call I made that wasn't explicitly
decided during the brainstorm. Newest entries at the bottom of each section.

## Decided together during the brainstorm (for reference)

- Purpose: research demonstrator, both benchmarking and live serving on the website.
- SAC = the existing contextual decoration; Baseline is plainer than today (no decoration).
- 8 modes = 4 substrates (Baseline, SAC+RAPTOR, GraphRAG, Combined) × agentic on/off.
- Agentic is an orthogonal wrapper over a common `retrieve` interface.
- Hand-roll RAPTOR and GraphRAG, Azure-native, no external RAG frameworks.
- GraphRAG graph stored as flat rows in Azure AI Search (no graph DB).
- Agentic loop built on Microsoft Agent Framework, bounded iterations.
- One spec covering the whole expansion, phased build.
- API: `/query?mode=` plus a `/compare` endpoint for side-by-side.

## Decisions I made autonomously while building

(Filled in as I go. Each entry: what I chose, why, and what the alternative was so you can
overrule it cheaply.)

- **`.superpowers/` gitignore:** already present, no action needed.
- **RAPTOR+SAC gets its own `raptor-sac` index** rather than mutating the Foundry-bound
  contextual index. Why: keeps RAPTOR summary nodes out of the live generator's knowledge
  source. Cost: re-uploads SAC leaves (cheap, decoration is cache-hit). Alternative: add a
  `level` filter on the knowledge source and reuse the existing index.
- **Phasing: one plan per phase, not one giant plan.** The spec is one document, but the
  implementation is split into Phase 1..4 plans. Phase 1 (`docs/superpowers/plans/
  2026-06-16-multi-substrate-retrieval-phase1.md`) is the seam + Baseline and is the
  working/testable foundation. Phases 2-4 (RAPTOR, GraphRAG, Combined+Agentic) get their
  own plans written just before their execution so they reference the real seam code, not
  guesses. Why: a single bite-sized plan for all 8 modes would be enormous and would bake
  in guesses about code that doesn't exist yet.
- **A `contextual` mode is kept as the default**, alongside the 8 spec modes. It's the
  current decorated-index pipeline (SAC without RAPTOR) and stays the API default for
  backward compatibility. It is not one of the 8 headline modes; it's the legacy/default.
  `baseline` and `contextual` share one `HybridSubstrate` class, differing only by index.
- **`PipelineState` keeps `reranked` as a real field** (not just a stages entry) because
  it changes every retry and the gate/generator consume it directly; it's also mirrored
  into `stages["reranked"]` so dynamic readers see it. `dense/bm25/fused` become entries
  in the `stages` dict. Alternative was a pure dict with no named fields (more churn in
  the gate loop for no benefit).
- **`eval_results.json` shape changes** to `{means_by_mode, modes: {<mode>: {...}}}`.
  The old top-level `means/means_by_tag/coverage/records` now live under each mode. The
  `/eval` endpoint falls back to the old shape if it sees a stale file, so nothing crashes
  before a re-run. Morning check: re-run eval to regenerate in the new shape.
- **All Phase 1 work is unit-tested with fakes; no live Azure calls overnight.** Building
  the `baseline` index (ingest) and running a live multi-mode eval cost money and need
  creds, so I am NOT running them autonomously. They're left as a morning checklist item.

## Morning checklist (live steps I did not run)

- `uv run python -m ragpipe.ingest` equivalent for baseline: build the `baseline` index
  (the `build_baseline` driver). Needs Azure creds + embedding/LLM budget.
- `uv run python -m ragpipe.eval.run --modes contextual,baseline` to regenerate
  `eval_results.json` in the new mode-keyed shape and confirm the two modes are comparable.
- Eyeball `/compare` against the running API once the baseline index exists.

## Open questions for you (morning)

- (Filled in if I hit anything I'd genuinely want your call on but had to pick a default
  for.)
