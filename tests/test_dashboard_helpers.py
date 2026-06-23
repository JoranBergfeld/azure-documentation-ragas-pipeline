from __future__ import annotations

from pathlib import Path

from ragpipe.config import RetrievalMode
from ragpipe.models import Chunk, PipelineState
from app.dashboard import (
    available_architecture_diagrams,
    chunk_label,
    eval_rows,
    is_agentic_mode,
    mode_label,
    per_stage_chart_data,
    progress_step_view,
    stage_chunk_tables,
    stage_expanded,
    stage_rows,
)


def _state() -> PipelineState:
    state = PipelineState(query="q")
    state.set_stage("dense", [Chunk(id="d1", title="Dense Doc", url="http://d", content="x", score=0.5)])
    state.set_stage("bm25", [Chunk(id="b1", title="BM25 Doc", url="http://b", content="y", score=0.8)])
    state.set_stage("fused", [Chunk(id="f1", title="Fused Doc", url="http://f", content="z", score=0.7)])
    state.set_reranked([Chunk(id="r1", title="Reranked Doc", url="http://r", content="w", score=3.2)])
    state.answer = "final"
    state.faithfulness = 0.81
    return state


def test_stage_rows_summarizes_each_stage_with_readable_titles():
    rows = stage_rows(_state())

    labels = [r["stage"] for r in rows]
    assert labels == ["dense", "bm25", "fused", "reranked", "answer", "faithfulness"]
    faith = next(r for r in rows if r["stage"] == "faithfulness")
    assert faith["detail"] == "0.81"
    red = next(r for r in rows if r["stage"] == "reranked")
    # readable title + score, NOT the opaque base64 id
    assert "Reranked Doc (3.20)" in red["detail"]
    assert "r1" not in red["detail"]
    assert red["count"] == 1


def test_chunk_label_prefers_title_then_url_then_id():
    assert chunk_label(Chunk(id="i", title="T", url="u", content="")) == "T"
    assert chunk_label(Chunk(id="i", title="", url="u", content="")) == "u"
    assert chunk_label(Chunk(id="i", title="", url="", content="")) == "i"


def test_stage_chunk_tables_are_ranked_and_readable():
    tables = stage_chunk_tables(_state())
    assert set(tables) == {"dense", "bm25", "fused", "reranked"}
    row = tables["reranked"][0]
    assert row == {"rank": 1, "title": "Reranked Doc", "score": 3.2, "url": "http://r"}


def test_eval_rows_flattens_means_sorted_with_coverage():
    results = {
        "means": {"faithfulness": 0.9, "answer_relevancy": 0.812345},
        "coverage": {
            "faithfulness": {"valid": 2, "total": 3},
            "answer_relevancy": {"valid": 3, "total": 3},
        },
        "records": [{"question": "q1"}],
    }
    rows = eval_rows(results)
    assert rows == [
        {"metric": "answer_relevancy", "mean_score": 0.8123, "coverage": "3/3"},
        {"metric": "faithfulness", "mean_score": 0.9, "coverage": "2/3"},
    ]


def test_eval_rows_without_coverage_omits_field():
    rows = eval_rows({"means": {"faithfulness": 0.5}})
    assert rows == [{"metric": "faithfulness", "mean_score": 0.5}]


def test_eval_rows_empty_when_no_means():
    assert eval_rows({}) == []


def test_eval_rows_excludes_per_stage_keys():
    results = {"means": {"faithfulness": 0.9, "context_recall@dense": 0.5}}
    rows = eval_rows(results)
    assert rows == [{"metric": "faithfulness", "mean_score": 0.9}]


def test_per_stage_chart_data_pivots_in_pipeline_order():
    results = {
        "means": {
            "context_recall@reranked": 1.0,
            "context_recall@dense": 0.5,
            "context_precision@dense": 0.4,
            "context_precision@reranked": 0.95,
            "faithfulness": 0.9,  # plain key ignored
        }
    }
    data = per_stage_chart_data(results)
    # dense before reranked (pipeline order), faithfulness excluded
    assert list(data.keys()) == ["dense", "reranked"]
    assert data["dense"] == {"context_recall": 0.5, "context_precision": 0.4}
    assert data["reranked"] == {"context_recall": 1.0, "context_precision": 0.95}


def test_per_stage_chart_data_empty_without_stage_keys():
    assert per_stage_chart_data({"means": {"faithfulness": 0.9}}) == {}


def test_is_agentic_mode_detects_agentic_suffix():
    assert is_agentic_mode("baseline_agentic") is True
    assert is_agentic_mode("combined_agentic") is True
    assert is_agentic_mode("contextual") is False
    assert is_agentic_mode("graphrag") is False


def test_mode_label_marks_experimental_modes_only():
    assert mode_label("baseline_agentic") == "baseline_agentic — experimental (unevaluated)"
    assert mode_label(RetrievalMode.COMBINED_AGENTIC) == "combined_agentic — experimental (unevaluated)"
    assert mode_label("contextual") == "contextual"
    assert mode_label(RetrievalMode.BASELINE) == "baseline"


def test_stage_expanded_opens_reranked_always_and_iter0_only_for_agentic():
    # reranked is the final set: always open, regardless of mode
    assert stage_expanded("reranked", "contextual") is True
    assert stage_expanded("reranked", "baseline_agentic") is True
    # iter_0 (the first planner sub-query round) opens only for agentic modes
    assert stage_expanded("iter_0", "baseline_agentic") is True
    assert stage_expanded("iter_0", "contextual") is False
    # other stages stay collapsed
    assert stage_expanded("iter_1", "baseline_agentic") is False
    assert stage_expanded("dense", "contextual") is False


def test_per_stage_chart_data_orders_agentic_iters_before_fused_then_reranked():
    # iter_N are the agentic sub-query rounds; they must read iter_0..iter_N ->
    # fused -> reranked even when the means dict lists them scrambled.
    results = {
        "means": {
            "context_recall@reranked": 0.9,
            "context_recall@fused": 0.7,
            "context_recall@iter_1": 0.6,
            "context_recall@iter_0": 0.5,
        }
    }
    data = per_stage_chart_data(results)
    assert list(data.keys()) == ["iter_0", "iter_1", "fused", "reranked"]


def test_per_stage_chart_data_orders_iters_numerically_not_lexically():
    results = {
        "means": {
            "context_recall@iter_2": 0.2,
            "context_recall@iter_10": 0.1,
            "context_recall@reranked": 0.9,
        }
    }
    data = per_stage_chart_data(results)
    # iter_10 must come after iter_2 (numeric), reranked last
    assert list(data.keys()) == ["iter_2", "iter_10", "reranked"]


def test_per_stage_chart_data_keeps_hybrid_dense_bm25_order():
    # HybridSubstrate (contextual/baseline/raptor_sac) still emits dense/bm25/fused;
    # they must stay in retrieval order ahead of fused -> reranked.
    results = {
        "means": {
            "context_recall@reranked": 0.9,
            "context_recall@fused": 0.7,
            "context_recall@bm25": 0.6,
            "context_recall@dense": 0.5,
        }
    }
    data = per_stage_chart_data(results)
    assert list(data.keys()) == ["dense", "bm25", "fused", "reranked"]


def test_per_stage_chart_data_orders_graphrag_local_global_before_fused():
    results = {
        "means": {
            "context_recall@reranked": 0.9,
            "context_recall@fused": 0.7,
            "context_recall@global": 0.6,
            "context_recall@local": 0.5,
        }
    }
    data = per_stage_chart_data(results)
    assert list(data.keys()) == ["local", "global", "fused", "reranked"]


def test_available_architecture_diagrams_returns_committed_substrate_svgs():
    diagrams = available_architecture_diagrams()
    paths = [path for _caption, path in diagrams]
    assert "docs/retrieval-substrates.svg" in paths
    assert "docs/graphrag.svg" in paths
    # every returned diagram exists on disk and carries a human caption
    for caption, path in diagrams:
        assert caption and Path(path).exists()


def test_progress_step_view_icons_by_status():
    from ragpipe.progress import ProgressEvent

    start = ProgressEvent(phase="generate", status="start", message="Generating answer (attempt 1)")
    done = ProgressEvent(phase="faithfulness", status="complete", message="Faithfulness 0.82")
    err = ProgressEvent(phase="faithfulness", status="error", message="Faithfulness judge failed")

    assert progress_step_view(start) == ("⏳", "Generating answer (attempt 1)")
    assert progress_step_view(done) == ("✅", "Faithfulness 0.82")
    assert progress_step_view(err) == ("⚠️", "Faithfulness judge failed")


def test_progress_step_view_falls_back_to_phase_when_no_message():
    from ragpipe.progress import ProgressEvent

    ev = ProgressEvent(phase="retrieve.fuse", status="complete")
    assert progress_step_view(ev) == ("✅", "retrieve.fuse")
