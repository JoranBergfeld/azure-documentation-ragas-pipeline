# README + diagrams refresh for the multi-substrate structure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the README narrative, refresh and add architecture SVGs, add a data-backed eval-results chart, and realign the dashboard workflow diagram so all top-level docs reflect the retrieval-substrate seam (9 modes, multiple indexes).

**Architecture:** Five deliverables — (1) a regenerated workflow-viz topology + `pipeline.mmd`; (2) a committed matplotlib script that renders `docs/eval-results.svg` from the per-mode eval files; (3) two new hand-authored SVGs (multi-index substrate seam + GraphRAG); (4) refreshed `pipeline.svg`/`pipeline-query.svg`; (5) README edits. Code deliverables (1–2) are TDD; SVG/README deliverables are content-specified + visually verified.

**Tech Stack:** Python 3.11, `uv`, pytest, matplotlib (new dev dep), Microsoft Agent Framework `WorkflowViz`, hand-authored SVG.

**Spec:** `docs/superpowers/specs/2026-06-19-readme-diagrams-refresh-design.md`

**Shared SVG conventions** (copy from existing `docs/pipeline*.svg`; all new/edited SVGs must match):
- Canvas: `<rect ... rx="14" fill="#FFFFFF"/>`; root `font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"`.
- Title `font-size="22" font-weight="700" fill="#1B2733"` at `x=32 y=34`; subtitle `font-size="13" fill="#5B6B7B"` at `x=32 y=52`.
- Marker defs (reuse verbatim): `arrow` (#6B7A88), `arrowAz` (#0078D4), `arrowTeal` (#2E9E4B).
- Palette: Azure service `fill="#E3F0FB" stroke="#0078D4"` (light leg stroke `#5AA0DC`); Foundry agent `fill="#F0EAFA" stroke="#8661C5"`; RAGAS/eval `fill="#E5F5E9" stroke="#2E9E4B"` (answer `stroke="#107C10"`); neutral `fill="#EEF2F6" stroke="#9AA8B5"`; artifact/yaml/json `fill="#FFF6E6" stroke="#E3B341"`; warn/abstain `fill="#FDF2F2" stroke="#C56666"`; band fills `#F5F8FB`/`#FAF7FD`/`#F3FAF4` with strokes `#D6E4F0`/`#E2D6F0`/`#CDE9D2`.
- Box: `rx="8"`, `stroke-width="1.5"`; bold label `font-size="12.5" font-weight="600" fill="#1B2733"`, sub-label `font-size="10.5" fill="#5B6B7B"`, both `text-anchor="middle"`.

**Index/substrate facts to encode (from code, do not invent):**
- Indexes: `contextual` (= `settings.search_index`, SAC-decorated, Foundry-bound, ADR-0007), `baseline` (plain chunks, no SAC), `raptor-sac` (SAC level-0 leaves + RAPTOR summary nodes level≥1, ADR-0013), `graph-entities` / `graph-relationships` / `graph-communities` (ADR-0014).
- Substrates → modes (`registry.py`): `contextual`/`baseline`/`raptor_sac` = `HybridSubstrate` (dense+BM25→RRF); `graphrag` = local+global fused; `combined` = RAPTOR⊕GraphRAG RRF-fused; agentic wrapper over baseline/raptor_sac/graphrag/combined (no `contextual_agentic`). 9 modes total.
- raptor_sac retrieval = **collapsed tree**: one hybrid query over leaves+summaries in the single `raptor-sac` index; `level` is filterable but never filtered at query time.
- Pipeline tail is shared & unchanged: substrate `retrieve()` once → `rerank` (window widens each retry, `k = top_k + rerank_widen_step*attempt`) → Foundry generator → RAGAS faithfulness gate (Claude) → PASS / RETRY(rerank) / EXHAUSTED→abstain.

---

### Task 1: Realign dashboard workflow viz to the substrate-seam topology

**Files:**
- Modify: `tests/test_workflow_viz.py`
- Modify: `src/ragpipe/workflow.py:103-155` (`build_viz_workflow`)
- Regenerate: `docs/pipeline.mmd`

- [ ] **Step 1: Update the test to assert the new node set**

Replace the whole body of `tests/test_workflow_viz.py` with:

```python
from ragpipe.workflow import build_viz_workflow


def test_build_viz_workflow_has_substrate_seam_nodes():
    wf = build_viz_workflow()
    from agent_framework import WorkflowViz

    diagram = WorkflowViz(wf).to_mermaid()
    for stage in ["retrieve", "rerank", "generate", "faithfulness", "answer"]:
        assert stage in diagram


def test_build_viz_workflow_drops_legacy_retrieval_nodes():
    wf = build_viz_workflow()
    from agent_framework import WorkflowViz

    diagram = WorkflowViz(wf).to_mermaid()
    # The fixed dense/bm25/rrf topology is gone — retrieval is one substrate call now.
    for legacy in ["dense", "bm25", "rrf"]:
        assert legacy not in diagram
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_workflow_viz.py -q`
Expected: FAIL — current diagram still contains `dense`/`bm25`/`rrf` and has no `retrieve`/`answer` (well, `answer` exists; the `retrieve` assertion and the legacy-drop assertions fail).

- [ ] **Step 3: Rewrite `build_viz_workflow` to the substrate-seam topology**

In `src/ragpipe/workflow.py`, replace the node/edge section of `build_viz_workflow` (the block from `start = _Stage(id="start")` through `return builder.build()`) with:

```python
    # One retrieval node now stands in for the whole substrate (dense+BM25, RAPTOR,
    # local/global graph, combined, agentic) — run_pipeline calls retrieve() once,
    # then loops rerank -> generate -> faithfulness (ADR-0012).
    start = _Stage(id="start")
    retrieve = _Stage(id="retrieve")
    rerank = _Stage(id="rerank")
    generate = _Stage(id="generate")
    faithfulness = _Stage(id="faithfulness")
    answer = _Stage(id="answer")

    def low_faithfulness(_msg: str) -> bool:
        return True  # label-only; real decision is in run_pipeline()

    builder = WorkflowBuilder(start_executor=start)
    builder.add_edge(start, retrieve)
    builder.add_edge(retrieve, rerank)
    builder.add_edge(rerank, generate)
    builder.add_edge(generate, faithfulness)
    # Retries re-enter at rerank (widened window over the fixed candidate set),
    # never re-running retrieval — retrieval is one substrate call per query (ADR-0009/0012).
    builder.add_edge(faithfulness, rerank, condition=low_faithfulness)
    builder.add_edge(faithfulness, answer)
    return builder.build()
```

Also update the docstring's parenthetical `(incl. the conditional loop edge faithfulness->rrf)` to `(incl. the conditional loop edge faithfulness->rerank)` and the dispatch comment that mentions "BOTH dense and bm25" — replace that comment with `# A single retrieval node stands in for the pluggable substrate (ADR-0012).`

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_workflow_viz.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regenerate `docs/pipeline.mmd`**

Run:
```bash
uv run python -c "from ragpipe.workflow import build_viz_workflow; from agent_framework import WorkflowViz; open('docs/pipeline.mmd','w').write(WorkflowViz(build_viz_workflow()).to_mermaid())"
```
Then `cat docs/pipeline.mmd` and confirm it contains `retrieve`, `rerank`, `generate`, `faithfulness`, `answer`, the edge `faithfulness -. conditional .-> rerank`, and **no** `dense`/`bm25`/`rrf`.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/workflow.py tests/test_workflow_viz.py docs/pipeline.mmd
git commit -m "feat(viz): substrate-seam workflow diagram (retrieve replaces dense/bm25/rrf)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Eval-results chart — matplotlib dev dep + committed plot script

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups] dev`)
- Modify: `uv.lock` (via `uv lock`)
- Create: `scripts/plot_eval_results.py`
- Create: `tests/test_plot_eval_results.py`
- Generate: `docs/eval-results.svg`

- [ ] **Step 1: Add matplotlib to the dev group and lock**

Edit `pyproject.toml`:
```toml
[dependency-groups]
dev = ["pytest", "pytest-asyncio", "pytest-mock", "ruff", "matplotlib"]
```
Run: `uv lock && uv sync`
Expected: lock resolves (prerelease allowed), `.venv` gains matplotlib. Verify: `uv run python -c "import matplotlib; print(matplotlib.__version__)"` prints a version.

- [ ] **Step 2: Write the failing unit test for the pure data loader**

Create `tests/test_plot_eval_results.py`:
```python
from __future__ import annotations

import json

from scripts.plot_eval_results import RAGAS_METRICS, load_means


def test_load_means_selects_only_ragas_metrics(tmp_path):
    (tmp_path / "eval_results_contextual.json").write_text(
        json.dumps(
            {
                "mode": "contextual",
                "means": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.86,
                    "context_recall": 0.81,
                    "mrr@fused": 0.5,
                    "abstained": 0.03,
                },
            }
        )
    )
    means = load_means(["contextual", "baseline"], tmp_path)

    # baseline file is absent -> skipped, not an error
    assert set(means) == {"contextual"}
    # only the 4 core RAGAS metrics are kept, retrieval/abstention dropped
    assert set(means["contextual"]) == set(RAGAS_METRICS)
    assert means["contextual"]["faithfulness"] == 0.9


def test_load_means_tolerates_missing_metric(tmp_path):
    (tmp_path / "eval_results_graphrag.json").write_text(
        json.dumps({"mode": "graphrag", "means": {"faithfulness": 0.74}})
    )
    means = load_means(["graphrag"], tmp_path)
    assert means["graphrag"]["faithfulness"] == 0.74
    assert means["graphrag"]["context_recall"] is None
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_plot_eval_results.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.plot_eval_results'`.

- [ ] **Step 4: Implement `scripts/plot_eval_results.py`**

Create `scripts/plot_eval_results.py`:
```python
"""Render docs/eval-results.svg from the committed per-mode eval files.

Reads each eval_results_<mode>.json (the standalone, committed reference result
for that substrate; ADR-0016) and draws one grouped bar chart of the four core
RAGAS metrics across the evaluated modes. matplotlib emits SVG directly, so no
rasterizer is needed. Agentic modes are retrieval wrappers with no eval files
and are therefore not shown.

Run: uv run python scripts/plot_eval_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
EVAL_MODES = ["contextual", "baseline", "raptor_sac", "graphrag", "combined"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
METRIC_COLORS = {
    "faithfulness": "#2E9E4B",
    "answer_relevancy": "#0078D4",
    "context_precision": "#8661C5",
    "context_recall": "#E3B341",
}


def load_means(modes: list[str], results_dir: Path) -> dict[str, dict[str, float | None]]:
    """For each mode with an eval_results_<mode>.json, return its 4 core RAGAS means.

    Modes whose file is absent are skipped. A metric missing from a file's `means`
    maps to None so the caller can decide how to render it.
    """
    out: dict[str, dict[str, float | None]] = {}
    for mode in modes:
        path = results_dir / f"eval_results_{mode}.json"
        if not path.exists():
            continue
        means = json.loads(path.read_text()).get("means", {})
        out[mode] = {m: means.get(m) for m in RAGAS_METRICS}
    return out


def build_figure(means_by_mode: dict[str, dict[str, float | None]]):  # pragma: no cover - rendering
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in EVAL_MODES if m in means_by_mode]
    width = 0.8 / len(RAGAS_METRICS)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, metric in enumerate(RAGAS_METRICS):
        offsets = [xi - 0.4 + width * (i + 0.5) for xi in range(len(modes))]
        values = [means_by_mode[m].get(metric) or 0.0 for m in modes]
        bars = ax.bar(
            offsets, values, width=width, label=METRIC_LABELS[metric], color=METRIC_COLORS[metric]
        )
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1B2733",
            )

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title(
        "RAGAS evaluation across retrieval modes", fontweight="bold", color="#1B2733"
    )
    ax.yaxis.grid(True, color="#E3E9EF")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def main() -> None:  # pragma: no cover - file IO entrypoint
    root = Path(__file__).resolve().parent.parent
    means = load_means(EVAL_MODES, root)
    if not means:
        raise SystemExit("No eval_results_<mode>.json files found; run the eval harness first.")
    out = root / "docs" / "eval-results.svg"
    build_figure(means).savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out} for modes: {', '.join(means)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_plot_eval_results.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Generate the chart and verify the artifact**

Run: `uv run python scripts/plot_eval_results.py`
Expected: prints `wrote .../docs/eval-results.svg for modes: contextual, baseline, raptor_sac, graphrag, combined`.
Verify: `head -1 docs/eval-results.svg` starts with `<?xml` or `<svg`, and `grep -c "<rect" docs/eval-results.svg` is > 0.

- [ ] **Step 7: Lint**

Run: `uv run ruff check scripts/plot_eval_results.py tests/test_plot_eval_results.py`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock scripts/plot_eval_results.py tests/test_plot_eval_results.py docs/eval-results.svg
git commit -m "feat(eval): plot_eval_results.py renders docs/eval-results.svg from per-mode files

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Refresh `docs/pipeline.svg` (overview) retrieval band → substrate seam

**Files:**
- Modify: `docs/pipeline.svg`

- [ ] **Step 1: Update the subtitle**

Change the line-19 subtitle text to:
`Contextual decoration at ingest → pluggable retrieval substrate (9 modes / multiple indexes) → rerank → Foundry agent → faithfulness guardrail; multi-mode RAGAS evaluation`

- [ ] **Step 2: Add a one-line note to the ingest band**

After the existing two `x="460"` notes (lines 66–67), the ingest story still indexes contextual chunks; append one note centered under the band:
`<text x="460" y="184" ...>` is outside the band — instead change line-66 note end to mention the extra indexes. Replace the line-66 text content with:
`context = breadcrumb + LLM situating sentence (ADR-0001/0005) · ingest also builds the baseline / raptor-sac / graph-* indexes (ADR-0013/0014)`
(Keep the line-67 prune note unchanged.)

- [ ] **Step 3: Replace the Band B retrieval cluster (the Dense / BM25 / RRF boxes) with a substrate-seam depiction**

In Band B (`y` ≈ 252–344), replace the three boxes **Dense retrieval** (lines 82–86), **BM25 retrieval** (lines 87–91), and **RRF fusion** (lines 92–96), plus their inter-box arrows (lines 123–127) and the two dashed index-feed paths (lines 74–75) and label (line 76), with:

- One **Retrieval substrate** box (Azure-blue) where the two retrievers were, e.g. `rect x="210" y="300" width="170" height="44"`:
  - bold label `Retrieval substrate`
  - sub-label `retrieve() once · dense+BM25 / RAPTOR / graph / combined`
- A small stacked **index legend** to its left/above feeding it via one dashed `arrowAz` path from the ingest "Azure AI Search index" box, labelled:
  - `contextual · baseline · raptor-sac · graph-* (mode picks the index)`
- An **agentic** annotation under the substrate box (purple `#8661C5`, font-size 10.5): `+ optional agentic wrapper (bounded plan→retrieve loop, ADR-0015)`.
- Keep the existing **Azure semantic reranker** box (lines 97–101), **Foundry generator agent** (102–106), **Answer** (107–111), **RAGAS faithfulness guardrail** (112–116) and all their arrows/labels (128–135) unchanged, but reconnect the substrate box → reranker with a single `url(#arrow)` path replacing the old RRF→reranker arrow (line 127).
- Update the retry label (line 135) target wording from "retry from RRF" semantics to: keep arrow but ensure it points back to the **reranker** box (retries re-enter at rerank), label `unfaithful < θ — widen window & retry (max N)`.

- [ ] **Step 4: Visually verify**

Open `docs/pipeline.svg` in a browser/preview. Confirm: no Dense/BM25/RRF boxes remain; one Retrieval-substrate box feeds the reranker; index names + agentic note are legible; retry arrow lands on the reranker; nothing overlaps. Run `grep -c "Dense retrieval\|BM25 retrieval\|RRF fusion" docs/pipeline.svg` → expect `0`.

- [ ] **Step 5: Commit**

```bash
git add docs/pipeline.svg
git commit -m "docs(svg): overview retrieval band now shows the substrate seam

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Refresh `docs/pipeline-query.svg` Stage 1 → retrieval substrate

**Files:**
- Modify: `docs/pipeline-query.svg`

- [ ] **Step 1: Update the Stage-1 band title (line 17)**

Replace `1 · Hybrid retrieval — two legs over the same index` with
`1 · Retrieval substrate — one retrieve(query, k) call (ADR-0012); stages are substrate-named, final set mirrored as "reranked"`.

- [ ] **Step 2: Redraw the Stage-1 interior**

Replace the **DenseRetriever** (lines 27–31), **BM25Retriever** (32–36), **RRF fusion** (37–41) boxes and their arrows (47–52) with:
- Keep **Question** (18–21) and **embed query** (22–26).
- Add a single wide **Retrieval substrate** box (Azure-blue, `rect x="420" y="138" width="230" height="52"`), bold `Retrieval substrate`, sub-label `HybridSubstrate · RaptorSAC · GraphRAG · Combined (± agentic)`.
- Below it a neutral caption box or text listing the per-substrate stage names it can emit: `stages: dense·bm25·fused | local·global·fused | raptor_sac:*·graphrag:* | iter_N`.
- Keep **SemanticReranker** (42–46) but move/redraw a single `url(#arrow)` from the substrate box → reranker. Update the reranker sub-label to `over substrate candidates · search.in(id) filter · top k (widens on retry)`.
- Keep the line-53 note `chunks selected as id · title · url · content — the context field stays in the index (ADR-0003)`.

- [ ] **Step 3: Correct the now-stale retry/judge notes (Stage 3) to match current behavior**

These lines describe pre-substrate behavior and are now wrong:
- Line 107 text: replace `regenerates with the SAME contexts (attempt + 1)` with `re-reranks a widened window then regenerates (attempt + 1)`.
- Line 108 (`known quirks` text): replace its content with `retry widens the rerank window over the same retrieved candidate set (no re-retrieval); the faithfulness judge is claude-sonnet-4-6 (online gate, ADR-0009/0011), a different family from the gpt-5.4 generator`.

- [ ] **Step 4: Visually verify**

Open in preview. Confirm Stage 1 shows one substrate box (no Dense/BM25/RRF leg boxes), arrows are clean, Stage-3 notes read correctly. `grep -c "DenseRetriever\|BM25Retriever\|RRF fusion" docs/pipeline-query.svg` → `0`.

- [ ] **Step 5: Commit**

```bash
git add docs/pipeline-query.svg
git commit -m "docs(svg): query-detail Stage 1 generalized to the retrieval substrate

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: New `docs/retrieval-substrates.svg` (multiple-index substrate seam)

**Files:**
- Create: `docs/retrieval-substrates.svg`

- [ ] **Step 1: Author the SVG**

Create `docs/retrieval-substrates.svg` using the shared conventions. `viewBox="0 0 1180 720"`. Reuse the `arrow`/`arrowAz`/`arrowTeal` marker defs and the white rounded canvas. Layout, left→right:

- **Title** `Retrieval substrate seam` ; **subtitle** `One RetrievalSubstrate interface (name + async retrieve) · 9 modes over multiple Azure AI Search indexes (ADR-0012)`.
- **Legend** (top-right, same style as pipeline.svg): Azure AI service / Foundry agent / RAGAS-eval swatches.
- **Column 1 — Indexes** (Azure-blue boxes, stacked), each box = index name (bold) + one-line note:
  - `contextual` — `SAC-decorated chunks · Foundry-bound (ADR-0007)`
  - `baseline` — `plain chunks · no SAC`
  - `raptor-sac` — `SAC leaves (level 0) + RAPTOR summaries (level ≥1) · ADR-0013`
  - `graph-entities` / `graph-relationships` / `graph-communities` — group these three under a brace labelled `GraphRAG graph (ADR-0014)`.
- **Column 2 — Substrates** (neutral boxes), one per base substrate, each wired from its index(es):
  - `Contextual` ← contextual ; `Baseline` ← baseline ; `RAPTOR+SAC` ← raptor-sac (note: `collapsed tree: one hybrid over all levels`) — these three tagged `HybridSubstrate (dense+BM25 → RRF)`.
  - `GraphRAG` ← the three graph indexes (note: `local (entities+adjacency) ⊕ global (community reports) → RRF`).
  - `Combined` ← fed by RAPTOR+SAC **and** GraphRAG boxes (note: `RRF-fuse both substrates' candidates`).
- **Column 3 — Agentic wrapper** (Foundry-purple box) spanning Baseline/RAPTOR/GraphRAG/Combined: `AgenticSubstrate(inner) · bounded plan→retrieve loop · iter_N stages (ADR-0015) — no contextual_agentic`. Draw dashed purple `arrowAz`-style edges (use `arrow`/purple stroke) from those four substrates into it.
- **Column 4 — Shared tail** (single funnel all substrates converge into): `rerank (widens on retry)` → `Foundry generator (gpt-5.4)` → `RAGAS faithfulness gate (claude-sonnet-4-6)` → `Answer / abstain`. Use teal arrows for the gate edges, matching pipeline.svg.
- **Footer note** (`font-size 10.5 fill #5B6B7B`): `PipelineState.stages is a dynamic dict[str, list[Chunk]] — each substrate names its own stages; the final set is always mirrored as "reranked".`

Keep boxes non-overlapping; mirror spacing/visual rhythm of `pipeline.svg`. Every arrow uses a defined marker.

- [ ] **Step 2: Visually verify**

Open in preview. Confirm: 6 index boxes, 5 substrate boxes, the agentic wrapper spanning 4, combined fed by two substrates, one shared tail, readable labels, no overlaps, all 9 modes inferable (5 base + 4 agentic). Confirm well-formed XML: `uv run python -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('docs/retrieval-substrates.svg'); print('well-formed')"`.

- [ ] **Step 3: Commit**

```bash
git add docs/retrieval-substrates.svg
git commit -m "docs(svg): new retrieval-substrates diagram (multi-index seam)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: New `docs/graphrag.svg` (the graph part)

**Files:**
- Create: `docs/graphrag.svg`

- [ ] **Step 1: Author the SVG**

Create `docs/graphrag.svg`, shared conventions, `viewBox="0 0 1180 640"`. Two bands.

- **Title** `GraphRAG substrate — flat graph on Azure AI Search` ; **subtitle** `No graph DB: the graph is materialized into three search indexes; query = local ⊕ global, RRF-fused (ADR-0014)`.
- **Band 1 — Build time** (`fill #F5F8FB stroke #D6E4F0`), title `Build time — ragpipe.ingest graph builder`:
  - `chunks` (neutral) → `LLM entity/relationship extraction` (Azure-blue, note `cached · .graph_cache.json, ADR resume`) → `community detection + report summaries` (Azure-blue) → three index boxes (Azure-blue): `graph-entities` (`name·type·description+emb·community·source chunks`), `graph-relationships` (`source·target·description·weight·source chunks`), `graph-communities` (`level·title·report summary+emb`). Arrows left→right.
- **Band 2 — Query time** (`fill #FAF7FD stroke #E2D6F0`), title `Query time — GraphRAGSubstrate.retrieve(query, k)`:
  - **Local search** path: `hybrid-search graph-entities` → `expand 1–2 hops (in-memory adjacency)` → `gather source chunks of seeds + neighbors` (stage `local`).
  - **Global search** path: `hybrid-search graph-communities` → `top community report summaries` (stage `global`).
  - Both → **RRF fuse** box (stage `fused`) → arrow out to `rerank → generate → faithfulness` (draw a compact shared-tail stub with a teal arrow, matching pipeline.svg; label `shared pipeline tail`).
  - Note (`10.5 #5B6B7B`): `adjacency map is read once from graph-relationships at startup; no query-time traversal over a remote store.`
- **Band note** about combined (optional, small): `the combined mode RRF-fuses this substrate's candidates with RAPTOR+SAC.`

- [ ] **Step 2: Visually verify**

Open in preview. Confirm: build band shows 3 indexes produced; query band shows local + global → RRF → tail; labels legible; no overlaps. Well-formed check: `uv run python -c "import xml.dom.minidom; xml.dom.minidom.parse('docs/graphrag.svg'); print('well-formed')"`.

- [ ] **Step 3: Commit**

```bash
git add docs/graphrag.svg
git commit -m "docs(svg): new GraphRAG diagram (build-time indexes + local/global query)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: README.md — narrative, stale fixes, new diagram links, eval-results section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite the top narrative paragraph (lines 8–19)**

Replace the paragraph that begins `The whole flow: **① Ingest** ...` through `... comparing against a frozen baseline.` with:

```markdown
The whole flow: **① Ingest** crawls Microsoft Learn, extracts main content as
markdown (code/tables preserved), splits on headings, decorates every chunk with a
breadcrumb + cached LLM situating context (SAC — visible to retrieval only, see
`docs/adr/0001`), and indexes it across the substrate indexes in Azure AI Search
(`contextual`, `baseline`, `raptor-sac`, and the three `graph-*` indexes);
**② Query pipeline** runs a **pluggable retrieval substrate** (ADR-0012) — one of
**9 modes**: `contextual`, `baseline`, `raptor_sac` (RAPTOR collapsed-tree over SAC
leaves, ADR-0013), `graphrag` (flat local+global graph, ADR-0014), `combined`
(RAPTOR ⊕ GraphRAG, RRF-fused), and the four `*_agentic` wrappers (bounded
plan→retrieve loop, ADR-0015) — then a shared tail: Azure semantic rerank → Foundry
generator agent, with a directive RAGAS faithfulness guardrail judged by Claude
(ADR-0009) that widens the rerank window and regenerates on weak grounding, and
abstains when retries exhaust; **③ Evaluation** replays every mode over a tagged test
set and scores deterministic per-stage retrieval metrics (hit rate / MRR) plus the
RAGAS suite, comparing modes head-to-head against a frozen baseline (ADR-0016).
```

- [ ] **Step 2: Add the two new diagrams + deep-dive links (after line 24)**

After the `Per-phase deep dives: ...` paragraph (ends line 24), insert:

```markdown
Substrate deep dives: [Retrieval substrate seam (multiple indexes)](docs/retrieval-substrates.svg) ·
[GraphRAG (graph build + local/global query)](docs/graphrag.svg).

![Retrieval substrate seam — multiple indexes](docs/retrieval-substrates.svg)
```

- [ ] **Step 3: Fix the stale `/run` stage shape (lines 75–76)**

Replace `lowConfidence, abstained, and per-stage chunk tables (\`stages.{dense,bm25,fused,reranked}\`).` with:
`lowConfidence, abstained, and per-stage chunk tables (\`stages\` is a dynamic map — each substrate names its own stages; the final set is always mirrored under \`reranked\`).`

- [ ] **Step 4: Fix the stale per-stage metrics line (lines 98–99)**

Replace `# Also score context_precision/recall at each retrieval stage (dense/bm25/fused/` and the continuation `# reranked) — heavier (one judge pass per stage), shown as a grouped chart:` with:
```
# Also score context_precision/recall at each retrieval stage (the substrate's own
# stage names, final mirrored as reranked) — heavier (one judge pass per stage):
```

- [ ] **Step 5: Add an "Evaluation results" subsection (immediately after the `## Evaluate` block, before `## Test` at line 124)**

Insert:

```markdown
### Evaluation results

The committed per-mode `eval_results_<mode>.json` files render to a single comparison
chart of the four core RAGAS metrics across the evaluated substrates (the `*_agentic`
modes are retrieval wrappers with no standalone eval files, so they're not shown):

![RAGAS evaluation across retrieval modes](docs/eval-results.svg)

Regenerate it after a new eval run with:

```bash
uv run python scripts/plot_eval_results.py   # reads eval_results_<mode>.json → docs/eval-results.svg
```
```

- [ ] **Step 6: Verify the README references resolve**

Run:
```bash
grep -n "docs/retrieval-substrates.svg\|docs/graphrag.svg\|docs/eval-results.svg" README.md
for f in docs/retrieval-substrates.svg docs/graphrag.svg docs/eval-results.svg; do test -f "$f" && echo "OK $f" || echo "MISSING $f"; done
```
Expected: every referenced SVG exists. Also confirm `grep -n "dense,bm25,fused,reranked" README.md` returns nothing.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(readme): reflect substrate seam, multi-index ingest, eval-results chart

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Full validation

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass (prior ~201 + the 2 new plot tests; workflow-viz updated). No failures.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Diagram well-formedness + staleness sweep**

Run:
```bash
for f in docs/pipeline.svg docs/pipeline-query.svg docs/retrieval-substrates.svg docs/graphrag.svg docs/eval-results.svg; do
  uv run python -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('$f'); print('well-formed', '$f')"
done
grep -rn "dense,bm25,fused,reranked\|two legs over the same index" README.md docs/pipeline*.svg || echo "no stale retrieval phrasing"
```
Expected: all SVGs well-formed; no stale phrasing remains.

- [ ] **Step 4: Final review**

Open the four hand-authored SVGs + `docs/eval-results.svg` in a preview and skim the rendered README. Confirm diagrams are legible, non-overlapping, and accurate; the eval chart shows 5 modes × 4 metrics with value labels.

---

## Self-Review

**Spec coverage:**
- Eval chart (script + matplotlib dep + SVG + README embed + regen command) → Task 2 + Task 7 Step 5. ✓
- New `retrieval-substrates.svg` (multi-index) → Task 5. ✓
- New `graphrag.svg` → Task 6. ✓
- Update `pipeline.svg` + `pipeline-query.svg` → Tasks 3, 4. ✓
- README narrative + stale `stages.{...}` + `PER_STAGE_METRICS` fixes + diagram links → Task 7. ✓
- Workflow viz `build_viz_workflow` + `pipeline.mmd` + test → Task 1. ✓
- Validation (pytest/ruff/render) → Task 8. ✓
- Non-goals respected: no PNGs, no edits to `why-rag-evaluation.svg`/`ragas-ares-gaps.svg`, no new ADRs, no eval re-run.

**Placeholder scan:** Code tasks (1, 2) contain complete code. SVG/README tasks specify exact content (labels, palette, edits). No TBD/TODO.

**Type/name consistency:** `load_means(modes, results_dir)`, `RAGAS_METRICS`, `EVAL_MODES`, `METRIC_LABELS`, `METRIC_COLORS`, `build_figure`, `main` used identically across the script and `tests/test_plot_eval_results.py`. Stage node ids (`retrieve`/`rerank`/`generate`/`faithfulness`/`answer`) match between `build_viz_workflow` and `test_workflow_viz.py`. SVG filenames match between the diagram tasks and the README references.
