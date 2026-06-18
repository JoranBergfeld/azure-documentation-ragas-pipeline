from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragpipe.models import PipelineState
from ragpipe.retrieval.registry import registered_modes

EVAL_RESULTS_PATH = "eval_results.json"
PIPELINE_DIAGRAM_PATH = "docs/pipeline.mmd"


def mode_options() -> list[str]:
    """Runnable retrieval modes in registry order, for the Run-tab selector."""
    return [m.value for m in registered_modes()]


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


_STAGE_ORDER = ("dense", "bm25", "fused", "reranked")


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

    Returns stages in pipeline order (dense → bm25 → fused → reranked) so the chart
    reads as the retrieval flow. Empty dict if the per-stage sweep wasn't run.
    """
    means = results.get("means", {})
    by_stage: dict[str, dict[str, float]] = {}
    for key, value in means.items():
        if "@" not in key:
            continue
        metric, stage = key.split("@", 1)
        by_stage.setdefault(stage, {})[metric] = round(value, 4)
    ordered = {s: by_stage[s] for s in _STAGE_ORDER if s in by_stage}
    # include any unexpected stages after the known ones, for forward-compatibility
    for s in by_stage:
        if s not in ordered:
            ordered[s] = by_stage[s]
    return ordered


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


def main() -> None:  # pragma: no cover - UI entry point
    import asyncio

    import streamlit as st

    from ragpipe.config import Settings

    st.set_page_config(page_title="RAGAS-infused pipeline", layout="wide")
    st.title("RAGAS-infused pipeline")
    tab_run, tab_eval, tab_arch = st.tabs(["Run", "Evaluation", "Architecture"])

    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        if st.button("Run", key="run_query") and query:
            from ragpipe.app_wiring import build_pipeline_fn

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings)
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
            tables = stage_chunk_tables(state)
            for label, rows in tables.items():
                with st.expander(f"{label} — {len(rows)} chunk(s)", expanded=label == "reranked"):
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


if __name__ == "__main__":  # pragma: no cover
    main()
