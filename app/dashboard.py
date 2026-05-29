from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragpipe.models import PipelineState

EVAL_RESULTS_PATH = "eval_results.json"
PIPELINE_DIAGRAM_PATH = "docs/pipeline.mmd"


def stage_rows(state: PipelineState) -> list[dict[str, Any]]:
    """Flatten a PipelineState into table rows for the Run tab."""
    rows: list[dict[str, Any]] = []
    for label, chunks in [
        ("dense", state.dense),
        ("bm25", state.bm25),
        ("fused", state.fused),
        ("reranked", state.reranked),
    ]:
        rows.append(
            {
                "stage": label,
                "detail": ", ".join(f"{c.id}({c.score:.2f})" for c in chunks),
            }
        )
    rows.append({"stage": "answer", "detail": state.answer})
    rows.append(
        {
            "stage": "faithfulness",
            "detail": "n/a" if state.faithfulness is None else f"{state.faithfulness:.2f}",
        }
    )
    return rows


def eval_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an eval_results.json into per-metric rows (mean + coverage).

    Coverage (valid/total) makes visible when a metric's mean is over fewer items
    than the full set, e.g. when RAGAS returned NaN for an item.
    """
    means = results.get("means", {})
    cov = results.get("coverage", {})
    rows: list[dict[str, Any]] = []
    for k, v in sorted(means.items()):
        row = {"metric": k, "mean_score": round(v, 4)}
        if k in cov:
            row["coverage"] = f"{cov[k]['valid']}/{cov[k]['total']}"
        rows.append(row)
    return rows


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
            st.table(stage_rows(state))

    with tab_eval:
        st.caption(
            "Offline RAGAS metrics over the test set "
            "(`data/testset.jsonl`), averaged across items."
        )
        results_path = Path(EVAL_RESULTS_PATH)
        if results_path.exists():
            results = json.loads(results_path.read_text())
            rows = eval_rows(results)
            if rows:
                st.bar_chart({r["metric"]: r["mean_score"] for r in rows})
                st.table(rows)
            n = len(results.get("records", []))
            st.caption(f"From {n} evaluated item(s) in `{EVAL_RESULTS_PATH}`.")
            with st.expander("Per-item records"):
                st.json(results.get("records", []))
        else:
            st.info(
                f"No `{EVAL_RESULTS_PATH}` yet. Generate it by running the offline "
                "harness:\n\n```bash\npython -m ragpipe.eval.run\n```\n\n"
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
