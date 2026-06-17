from ragpipe.models import Chunk, PipelineState
from app.dashboard import (
    chunk_label,
    eval_rows,
    per_stage_chart_data,
    stage_chunk_tables,
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
