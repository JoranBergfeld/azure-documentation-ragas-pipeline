# README + diagrams refresh for the multi-substrate structure

**Date:** 2026-06-19
**Status:** Approved
**Origin:** Follow-on to the multi-substrate work
(`2026-06-16-multi-substrate-retrieval-design.md`) and the per-mode eval artifacts
(ADR-0016). The retrieval-substrate seam, the 9 live modes, the extra Azure AI Search
indexes, and the committed per-mode eval results now exist, but the README narrative
and the architecture diagrams still describe the original contextual-only
`dense → BM25 → RRF → rerank` topology.

## Problem

The top-of-funnel documentation is stale relative to the code:

- `README.md`'s opening narrative describes only the contextual hybrid flow
  ("hybrid retrieval (dense + BM25) → RRF fusion → rerank"), and still references the
  removed fixed `stages.{dense,bm25,fused,reranked}` shape and `dense/bm25/fused/reranked`
  per-stage metrics. The substrate seam (ADR-0012), RAPTOR (ADR-0013), GraphRAG
  (ADR-0014), the combined substrate, and the agentic wrapper (ADR-0015) appear only
  obliquely (in the API mode list), not in the architecture story.
- The hand-authored SVG diagrams (`docs/pipeline.svg`, `docs/pipeline-query.svg`) draw the
  fixed `dense + bm25 → rrf` topology over a single index. There is no diagram for the
  multiple-index substrate seam or for the GraphRAG graph.
- There is no visualization of the eval results, even though five modes have committed
  `eval_results_<mode>.json` files with the full RAGAS suite.
- The dashboard's Architecture tab renders `docs/pipeline.mmd`, generated from
  `build_viz_workflow()` — both still hardcode `dense → bm25 → rrf`.

## Goals

- The README narrative and diagrams reflect the current architecture: a pluggable
  retrieval substrate (9 modes), multiple Azure AI Search indexes, and the three-family
  judge split.
- A reproducible, data-backed eval-results chart is embedded in the README.
- New diagrams cover the two concepts that have no visual today: the multi-index
  substrate seam and the GraphRAG graph.
- The dashboard workflow diagram matches the substrate-seam topology.

## Non-goals

- No changes to `why-rag-evaluation.svg` / `ragas-ares-gaps.svg` (not referenced by the
  README or app; out of scope).
- No PNG raster companions for the diagrams — the README references SVG only, and no
  SVG→PNG renderer is installed. (Decision confirmed with the user.)
- No new ADRs — this change documents decisions already recorded in ADR-0012–0016.
- No re-run of the eval harness; the chart is built from the already-committed per-mode
  result files.

## Design

### 1. Eval-results chart (new, data-backed)

- **`scripts/plot_eval_results.py`** (matplotlib). Reads each committed
  `eval_results_<mode>.json` (top-level `means`) for the five evaluated modes
  (`contextual`, `baseline`, `raptor_sac`, `graphrag`, `combined`). Renders one grouped
  bar chart: x-axis = mode, grouped bars = the **4 core RAGAS metrics**
  (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), 0–1 axis,
  legend + per-bar value labels, title. Writes **`docs/eval-results.svg`**
  (`savefig(..., format="svg")` — matplotlib emits SVG directly, so no rasterizer is
  needed). Missing files/metrics are skipped gracefully rather than crashing.
  Runnable as `uv run python scripts/plot_eval_results.py`.
- The agentic modes are retrieval wrappers with no eval files; they are intentionally
  excluded and called out in the README caption.
- **`matplotlib`** is added to the dev dependency group in `pyproject.toml`, and
  `uv.lock` is refreshed (`uv lock` / `uv sync`).

### 2. New hand-authored SVG diagrams (match existing `docs/pipeline*.svg` style)

- **`docs/retrieval-substrates.svg`** — the multiple-indexes / substrate-seam view
  (ADR-0012): the 5 base substrates with their Azure AI Search indexes — `contextual`
  (SAC, Foundry-bound), `baseline` (plain chunks), `raptor-sac` (SAC leaves + RAPTOR
  summary levels), and the 3 graph indexes (`graph-entities`, `graph-relationships`,
  `graph-communities`); the **agentic** wrapper composing over any base substrate
  (bounded plan→retrieve loop); the **combined** substrate fusing RAPTOR + GraphRAG via
  RRF; all converging on the single shared tail `rerank → generate → faithfulness gate`.
- **`docs/graphrag.svg`** — the graph part (ADR-0014). Build time: entity / relationship /
  community extraction → 3 flat indexes. Query time: **local** (hybrid-search
  `graph-entities` → in-memory adjacency 1–2 hop expansion → gather source chunks) +
  **global** (hybrid-search `graph-communities` report summaries), fused via RRF into the
  reranked set.

### 3. Updated SVG diagrams

- **`docs/pipeline.svg`** (overview) and **`docs/pipeline-query.svg`**: replace the fixed
  `dense → bm25 → rrf` retrieval depiction with the pluggable **retrieval substrate**
  (a single `retrieve()` call producing dynamically named stages, the final set always
  mirrored as `reranked`), feeding the unchanged `rerank → generate → faithfulness gate`
  tail. Refresh subtitles to name the substrate seam. The overview's ingest band notes
  that ingest now also builds the raptor/graph indexes.

### 4. README.md edits

- Rewrite the opening narrative paragraph: contextual decoration at ingest → a
  **pluggable retrieval substrate** (9 modes: `contextual`, `baseline`, `raptor_sac`,
  `graphrag`, `combined`, and the 4 agentic variants) over **multiple Azure AI Search
  indexes** → shared rerank → Foundry generator → directive RAGAS faithfulness guardrail
  (three-family judge split) → multi-mode evaluation.
- Fix stale references: `stages.{dense,bm25,fused,reranked}` → substrate-named stages
  with the final set mirrored as `reranked`; update the `PER_STAGE_METRICS` line to
  describe per-substrate stages rather than `dense/bm25/fused/reranked`.
- Add the two new diagrams as inline images / deep-dive links alongside the existing
  per-phase links.
- Add an **"Evaluation results"** subsection that embeds `docs/eval-results.svg`, gives a
  1–2 line takeaway, names the 5 modes and 4 metrics shown, and documents the regenerate
  command (`uv run python scripts/plot_eval_results.py`).

### 5. Dashboard workflow viz

- **`build_viz_workflow()`** (`src/ragpipe/workflow.py`): new topology mirroring
  `run_pipeline` — `start → retrieve → rerank → generate → faithfulness → answer`, with
  the conditional retry edge `faithfulness → rerank` (retrieval runs once; retries
  re-enter at rerank). Drops the `dense` / `bm25` / `rrf` nodes.
- Regenerate **`docs/pipeline.mmd`** from the updated workflow.
- Update **`tests/test_workflow_viz.py`** to assert the new node set
  (`retrieve`, `rerank`, `generate`, `faithfulness`) instead of `dense`/`bm25`/`rrf`.

## Validation

- `uv run python scripts/plot_eval_results.py` writes `docs/eval-results.svg` from the
  committed per-mode files without network access.
- `uv run pytest -q` is green (in particular the updated `test_workflow_viz`).
- `uv run ruff check .` is clean.
- The four SVGs and the eval chart render correctly (visual check); the README links and
  embeds resolve.
