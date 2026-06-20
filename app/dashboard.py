from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragpipe.models import PipelineState
from ragpipe.retrieval.registry import registered_modes

EVAL_RESULTS_PATH = "eval_results.json"
PIPELINE_DIAGRAM_PATH = "docs/pipeline.mmd"

# Hand-authored architecture SVGs surfaced in the dashboard's Architecture tab,
# alongside the Mermaid pipeline, so the multi-index substrate seam and the
# agentic wrapper are actually drawn (the Mermaid shows `retrieve` as one node).
ARCHITECTURE_DIAGRAMS = (
    ("Retrieval substrate seam — one index per mode, plus the agentic wrapper", "docs/retrieval-substrates.svg"),
    ("GraphRAG substrate — entity/community indexes, local + global query", "docs/graphrag.svg"),
)


def mode_options() -> list[str]:
    """Runnable retrieval modes in registry order, for the Run-tab selector."""
    return [m.value for m in registered_modes()]


def is_agentic_mode(mode: str) -> bool:
    """True for the ``*_agentic`` wrapper modes (multi-round planner retrieval)."""
    return mode.endswith("_agentic")


def stage_expanded(label: str, mode: str) -> bool:
    """Which per-stage trace expanders open by default. Always ``reranked`` (the
    final set fed to the generator); plus ``iter_0`` for agentic modes so the
    planner's first sub-query round is visible without an extra click."""
    if label == "reranked":
        return True
    return is_agentic_mode(mode) and label == "iter_0"


def available_architecture_diagrams() -> list[tuple[str, str]]:
    """(caption, path) for each architecture SVG present on disk, so the
    Architecture tab can show the multi-index substrate seam and the agentic
    wrapper next to the Mermaid pipeline. Missing files are skipped."""
    return [(caption, path) for caption, path in ARCHITECTURE_DIAGRAMS if Path(path).exists()]


def chunk_label(chunk: Any) -> str:
    """Human-readable name for a retrieved chunk.

    The document id is base64(url)_index_hash — unreadable in a trace — so prefer
    the page title, then the URL, and only fall back to the id if both are empty.
    """
    if chunk.title:
        return chunk.title
    if chunk.url:
        return chunk.url
    return chunk.id


def stage_chunk_tables(state: PipelineState) -> dict[str, list[dict[str, Any]]]:
    """Per retrieval stage, a ranked table of readable chunk rows for the Run tab.

    {stage: [{rank, title, score, url}, ...]} in pipeline order. Shows the title
    (not the opaque id) so you can see *which documents* each stage surfaced and
    how reranking reorders them.
    """
    tables: dict[str, list[dict[str, Any]]] = {}
    for label, chunks in state.stages.items():
        tables[label] = [
            {
                "rank": rank,
                "title": chunk_label(c),
                "score": round(c.score, 3),
                "url": c.url,
            }
            for rank, c in enumerate(chunks, 1)
        ]
    return tables


def stage_rows(state: PipelineState) -> list[dict[str, Any]]:
    """One summary row per stage: chunk count + readable titles (not raw ids)."""
    rows: list[dict[str, Any]] = []
    for label, chunks in state.stages.items():
        titles = ", ".join(f"{chunk_label(c)} ({c.score:.2f})" for c in chunks)
        rows.append({"stage": label, "count": len(chunks), "detail": titles})
    rows.append({"stage": "answer", "count": "", "detail": state.answer})
    rows.append(
        {
            "stage": "faithfulness",
            "count": "",
            "detail": "n/a" if state.faithfulness is None else f"{state.faithfulness:.2f}",
        }
    )
    return rows


# Canonical retrieval order for the per-stage chart. Raw retrieval stages come
# first — agentic `iter_N` rounds in numeric order, then the known hybrid/graph
# stage names — followed by the substrate's `fused` merge and finally `reranked`.
_LEAD_STAGE_ORDER = ("dense", "bm25", "local", "global")


def _stage_sort_key(stage: str, position: int) -> tuple[int, int]:
    """Sort raw stages first, then `fused`, then `reranked` last. `iter_N` order
    numerically; other unknown stages (e.g. combined's `<sub>:<stage>`) keep
    their first-seen order via `position`."""
    if stage == "reranked":
        return (4, 0)
    if stage == "fused":
        return (3, 0)
    if stage.startswith("iter_"):
        suffix = stage[len("iter_"):]
        return (0, int(suffix)) if suffix.isdigit() else (2, position)
    if stage in _LEAD_STAGE_ORDER:
        return (1, _LEAD_STAGE_ORDER.index(stage))
    return (2, position)


def eval_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the overall (non-per-stage) metrics into rows (mean + coverage).

    Per-stage keys ('<metric>@<stage>') are excluded here and shown separately by
    per_stage_chart_data. Coverage (valid/total) makes visible when a metric's mean
    is over fewer items than the full set, e.g. when RAGAS returned NaN for an item.
    """
    means = results.get("means", {})
    cov = results.get("coverage", {})
    rows: list[dict[str, Any]] = []
    for k, v in sorted(means.items()):
        if "@" in k:  # per-stage metric, handled elsewhere
            continue
        row = {"metric": k, "mean_score": round(v, 4)}
        if k in cov:
            row["coverage"] = f"{cov[k]['valid']}/{cov[k]['total']}"
        rows.append(row)
    return rows


def per_stage_chart_data(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Pivot '<metric>@<stage>' means into {stage: {metric: score}} for a grouped chart.

    Returns stages in retrieval order — raw stages (agentic `iter_0..iter_N`, or
    `dense`/`bm25`, or graph `local`/`global`) → `fused` → `reranked` — so the
    chart reads as the flow. Empty dict if the per-stage sweep wasn't run.
    """
    means = results.get("means", {})
    by_stage: dict[str, dict[str, float]] = {}
    for key, value in means.items():
        if "@" not in key:
            continue
        metric, stage = key.split("@", 1)
        by_stage.setdefault(stage, {})[metric] = round(value, 4)
    positions = {stage: i for i, stage in enumerate(by_stage)}
    return {
        stage: by_stage[stage]
        for stage in sorted(by_stage, key=lambda s: _stage_sort_key(s, positions[s]))
    }


def _render_mermaid(mermaid_src: str) -> None:  # pragma: no cover - UI rendering
    """Render a mermaid diagram via the mermaid.js CDN inside an HTML component."""
    import streamlit.components.v1 as components

    html = f"""
    <div class="mermaid">{mermaid_src}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    components.html(html, height=560, scrolling=True)


def _render_svg(svg_src: str) -> None:  # pragma: no cover - UI rendering
    """Embed a hand-authored SVG diagram inline at full vector fidelity, scaled
    to the container width. Uses st.html (st.image would rasterize, and
    components.html is deprecated); a width style is injected because the SVGs
    carry only a viewBox, not explicit width/height."""
    import streamlit as st

    responsive = svg_src.replace('<svg ', '<svg style="width:100%;height:auto" ', 1)
    st.html(responsive)


def main() -> None:  # pragma: no cover - UI entry point
    import asyncio

    import streamlit as st

    from ragpipe.config import Settings
    from ragpipe.guardrail import prewarm_ragas_imports

    # Streamlit reruns corrupt the *first* langchain_openai import (pydantic
    # ValidationError building RunnablePassthrough); build the judge models now,
    # at load time, before any rerun can trigger the Run-tab pipeline build.
    prewarm_ragas_imports()

    st.set_page_config(page_title="RAGAS-infused pipeline", layout="wide")
    st.title("RAGAS-infused pipeline")
    tab_run, tab_eval, tab_arch = st.tabs(["Run", "Evaluation", "Architecture"])

    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        mode = st.selectbox(
            "Retrieval index / mode",
            mode_options(),
            help="Which substrate/index answers the query. *_agentic modes run "
            "multiple retrieval rounds and are slower.",
        )
        if st.button("Run", key="run_query") and query:
            from ragpipe.app_wiring import build_pipeline_fn

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings, mode=mode)
            with st.spinner("Running pipeline (retrieve → rerank → generate → faithfulness)…"):
                state = asyncio.run(pipeline_fn(query))
            st.subheader("Answer")
            st.write(state.answer)
            score = "n/a" if state.faithfulness is None else f"{state.faithfulness:.2f}"
            st.metric("Faithfulness", score, help="RAGAS faithfulness of the answer vs. retrieved context")
            if state.low_confidence:
                st.warning(
                    f"Low confidence: faithfulness below threshold after {state.attempt} retries."
                )
            st.subheader("Per-stage trace")
            st.caption(
                "Documents surfaced at each retrieval stage (title + score), in rank "
                "order — watch fusion merge the two lists and rerank reorder them."
            )
            if is_agentic_mode(mode):
                st.caption(
                    "Agentic mode: each `iter_N` block is one planner sub-query round; "
                    "`fused` is their de-duplicated merge (ADR-0015)."
                )
            tables = stage_chunk_tables(state)
            for label, rows in tables.items():
                with st.expander(f"{label} — {len(rows)} chunk(s)", expanded=stage_expanded(label, mode)):
                    if rows:
                        st.table(rows)
                    else:
                        st.write("_no chunks_")

    with tab_eval:
        st.caption(
            "Offline RAGAS metrics over the test set "
            "(`data/testset.jsonl`), averaged across items."
        )
        results_path = Path(EVAL_RESULTS_PATH)
        if results_path.exists():
            results = json.loads(results_path.read_text())
            # New multi-mode shape: {"means_by_mode": {...}, "modes": {<mode>: {...}}}.
            # Show a per-mode comparison up front, then let the user drill into one
            # mode using the same single-run rendering below.
            if "means_by_mode" in results:
                means_by_mode = results["means_by_mode"]
                st.subheader("Mode comparison")
                st.caption("Mean of each metric per retrieval mode — the head-to-head view.")
                metric_names = sorted({m for v in means_by_mode.values() for m in v})
                import pandas as pd

                chart = {
                    metric: [means_by_mode[mode].get(metric) for mode in means_by_mode]
                    for metric in metric_names
                }
                st.bar_chart(pd.DataFrame(chart, index=list(means_by_mode.keys())))
                mode_results = results.get("modes", {})
                if mode_results:
                    selected = st.selectbox("Drill into mode", list(mode_results.keys()))
                    results = mode_results[selected]
                else:
                    results = {}
            rows = eval_rows(results)
            if rows:
                st.subheader("Overall metrics")
                st.bar_chart({r["metric"]: r["mean_score"] for r in rows})
                st.table(rows)

            stage_data = per_stage_chart_data(results)
            if stage_data:
                st.subheader("Per-stage retrieval quality")
                st.caption(
                    "context_precision / context_recall at each retrieval stage — "
                    "watch recall hold through fusion and precision rise after rerank."
                )
                # rows = stages (pipeline order), columns = metrics → grouped bars
                metrics = sorted({m for s in stage_data.values() for m in s})
                chart = {
                    m: [stage_data[s].get(m) for s in stage_data] for m in metrics
                }
                import pandas as pd

                st.bar_chart(pd.DataFrame(chart, index=list(stage_data.keys())))
                st.table(
                    [{"stage": s, **vals} for s, vals in stage_data.items()]
                )

            n = len(results.get("records", []))
            st.caption(f"From {n} evaluated item(s) in `{EVAL_RESULTS_PATH}`.")
            with st.expander("Per-item records"):
                st.json(results.get("records", []))
        else:
            st.info(
                f"No `{EVAL_RESULTS_PATH}` yet. Generate it by running the offline "
                "harness:\n\n```bash\nuv run python -m ragpipe.eval.run\n```\n\n"
                "Then reload this tab."
            )

    with tab_arch:
        diagram = Path(PIPELINE_DIAGRAM_PATH)
        if diagram.exists():
            _render_mermaid(diagram.read_text())
            with st.expander("Mermaid source"):
                st.code(diagram.read_text(), language="text")
        else:
            st.info(
                f"No `{PIPELINE_DIAGRAM_PATH}` yet. Generate it with the WorkflowViz export."
            )
        for caption, path in available_architecture_diagrams():
            st.caption(caption)
            _render_svg(Path(path).read_text())


if __name__ == "__main__":  # pragma: no cover
    main()
