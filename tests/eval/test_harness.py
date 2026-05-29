import pytest

from ragpipe.eval.harness import EvalRecord, aggregate, run_harness
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


@pytest.mark.asyncio
async def test_run_harness_builds_records_from_pipeline_and_evaluator():
    items = [TestItem(question="q1", ground_truth="g1", ground_truth_context="ctx")]

    async def fake_pipeline(q):
        s = PipelineState(query=q)
        s.answer = "a1"
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
