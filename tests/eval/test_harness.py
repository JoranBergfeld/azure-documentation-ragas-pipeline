import pytest

from ragpipe.eval.harness import (
    EvalRecord,
    aggregate,
    parse_stage_metric,
    run_harness,
    stage_metric_key,
)
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


def test_aggregate_means_per_metric():
    records = [
        EvalRecord(question="q1", answer="a1", contexts=["c"], ground_truth="g1",
                   metrics={"faithfulness": 0.8, "answer_relevancy": 0.6}),
        EvalRecord(question="q2", answer="a2", contexts=["c"], ground_truth="g2",
                   metrics={"faithfulness": 0.6, "answer_relevancy": 1.0}),
    ]
    means = aggregate(records)
    assert means["faithfulness"] == pytest.approx(0.7)
    assert means["answer_relevancy"] == pytest.approx(0.8)


def test_stage_metric_key_roundtrip():
    key = stage_metric_key("context_recall", "dense")
    assert key == "context_recall@dense"
    assert parse_stage_metric(key) == ("context_recall", "dense")


def test_parse_stage_metric_returns_none_for_plain_key():
    assert parse_stage_metric("faithfulness") is None


def test_aggregate_handles_per_stage_keys():
    records = [
        EvalRecord(question="q", answer="a", contexts=["c"], ground_truth="g",
                   metrics={"context_recall@dense": 0.5, "context_recall@reranked": 1.0}),
        EvalRecord(question="q", answer="a", contexts=["c"], ground_truth="g",
                   metrics={"context_recall@dense": 0.7, "context_recall@reranked": 1.0}),
    ]
    means = aggregate(records)
    assert means["context_recall@dense"] == pytest.approx(0.6)
    assert means["context_recall@reranked"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_harness_builds_records_from_pipeline_and_evaluator():
    items = [TestItem(question="q1", ground_truth="g1", ground_truth_context="ctx")]

    async def fake_pipeline(q):
        s = PipelineState(query=q)
        s.answer = "a1"
        s.dense = [Chunk(id="d", title="t", url="u", content="dense-ctx")]
        s.bm25 = [Chunk(id="b", title="t", url="u", content="bm25-ctx")]
        s.fused = [Chunk(id="f", title="t", url="u", content="fused-ctx")]
        s.reranked = [Chunk(id="c", title="t", url="u", content="ctx-content")]
        return s

    async def fake_evaluator(records):
        for r in records:
            r.metrics = {"faithfulness": 0.9}
        return records

    records = await run_harness(items, pipeline_fn=fake_pipeline, evaluator_fn=fake_evaluator)

    assert records[0].answer == "a1"
    assert records[0].contexts == ["ctx-content"]
    assert records[0].metrics["faithfulness"] == 0.9
    # per-stage contexts are captured for the sweep (cheap; always populated)
    assert records[0].stage_contexts["dense"] == ["dense-ctx"]
    assert records[0].stage_contexts["bm25"] == ["bm25-ctx"]
    assert records[0].stage_contexts["fused"] == ["fused-ctx"]
    assert records[0].stage_contexts["reranked"] == ["ctx-content"]
