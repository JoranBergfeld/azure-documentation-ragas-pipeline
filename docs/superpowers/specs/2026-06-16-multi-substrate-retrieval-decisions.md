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

## Phase 1 execution outcome (overnight)

All 12 build tasks + final review done on branch `feat/multi-substrate-retrieval`. Full
suite green: **138 passed, ruff clean.** Built via subagent-driven development (fresh
implementer per task, reviews after).

What got built (the seam + Baseline):
- `RetrievalSubstrate` seam (`retrieval/substrate.py`): substrates return
  `RetrievalResult(candidates, stages)`; `HybridSubstrate` owns dense+bm25+rrf.
- `PipelineState` generalized to a `stages` dict (+ `candidates`); `reranked` kept and
  mirrored into `stages["reranked"]`.
- Mode registry (`retrieval/registry.py`) with `contextual` (default) + `baseline`;
  `build_pipeline_fn(settings, mode)` is mode-aware.
- `RetrievalMode` enum + per-substrate index names in config.
- Harness reads stages dynamically + `aggregate_by_mode`; `run.py --modes` writes
  `eval_results.json` keyed by mode; dashboard + API (`/run?mode`, `/compare`, dual-shape
  `/eval`) updated. Baseline ingest path + `build_index(include_context=False)`.
- ADRs 0012-0016.

Decisions / deviations made during execution (your call to overrule any):
- **Review approach:** I used full reviewer subagents for the substantive integration
  tasks (the workflow refactor, plus a final holistic branch review) and verified the
  small verbatim tasks myself (ran tests + ruff + read the diff). This deviates from the
  skill's strict per-task two-reviewer split, traded for tractability on a 12-task
  unattended run. Every task still passed tests + lint, and the final review caught the
  one real gap below.
- **Removed `rrf_k` from `PipelineDeps`** (dead after the substrate took over fusion;
  flagged by review). It now lives only on `Settings` and `HybridSubstrate`.
- **Stale test fixtures:** three test files set old-style `state.dense/bm25/fused`
  directly; migrated them to `set_stage`/`set_reranked`. One (`test_harness_metrics.py`)
  was a real regression a scoped per-task test run missed — caught when a later task ran
  the fuller suite. Lesson logged: run the whole suite between coupled refactors.
- **Final-review fix:** the Streamlit dashboard eval tab only understood the old
  single-run `eval_results.json`; I added a per-mode comparison chart + mode drill-down
  for the new shape. The API `/eval` already handled both shapes.

Known limitations left for later phases (not bugs):
- `build_viz_workflow` still draws the static dense/bm25/rrf graph. It's label-only and
  the contextual/baseline modes still produce those stages, so it's fine for Phase 1;
  RAPTOR/graph stages won't appear in the static diagram until we generalize it.
- `RETRIEVAL_STAGES` (the per-stage sweep default) is still the hybrid tuple; later
  substrates name different stages and the sweep default will need widening.

## Morning checklist (live steps I did not run)

- `uv run python -m ragpipe.ingest` equivalent for baseline: build the `baseline` index
  (the `build_baseline` driver). Needs Azure creds + embedding/LLM budget.
- `uv run python -m ragpipe.eval.run --modes contextual,baseline` to regenerate
  `eval_results.json` in the new mode-keyed shape and confirm the two modes are comparable.
- Eyeball `/compare` against the running API once the baseline index exists.

## Open questions for you (morning)

- (Filled in if I hit anything I'd genuinely want your call on but had to pick a default
  for.)
