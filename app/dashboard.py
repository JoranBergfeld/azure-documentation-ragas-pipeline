from __future__ import annotations

from typing import Any

from ragpipe.models import PipelineState


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


def main() -> None:  # pragma: no cover - UI entry point
    import asyncio

    import streamlit as st

    from ragpipe.config import Settings

    st.title("RAGAS-infused pipeline")
    tab_run, tab_eval, tab_arch = st.tabs(["Run", "Evaluation", "Architecture"])

    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        if st.button("Run") and query:
            from ragpipe.app_wiring import build_pipeline_fn  # lazy: defined in Task 15b

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings)
            state = asyncio.run(pipeline_fn(query))
            st.table(stage_rows(state))
            if state.low_confidence:
                st.warning("Low confidence: faithfulness threshold not met after retries.")

    with tab_arch:
        try:
            with open("docs/pipeline.mmd") as f:
                st.code(f.read(), language="mermaid")
        except FileNotFoundError:
            st.info("Run the WorkflowViz export to generate docs/pipeline.mmd.")


if __name__ == "__main__":  # pragma: no cover
    main()
