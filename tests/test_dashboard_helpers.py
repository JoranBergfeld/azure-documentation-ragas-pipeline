from ragpipe.models import Chunk, PipelineState
from app.dashboard import eval_rows, stage_rows


def test_stage_rows_summarizes_each_stage():
    state = PipelineState(query="q")
    state.dense = [Chunk(id="a", title="t", url="u", content="x", score=0.5)]
    state.reranked = [Chunk(id="a", title="t", url="u", content="x", score=3.2)]
    state.answer = "final"
    state.faithfulness = 0.81

    rows = stage_rows(state)

    labels = [r["stage"] for r in rows]
    assert "dense" in labels
    assert "reranked" in labels
    faith = next(r for r in rows if r["stage"] == "faithfulness")
    assert faith["detail"] == "0.81"


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
