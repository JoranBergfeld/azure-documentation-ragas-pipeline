import asyncio

from ragpipe.eval.harness import EvalRecord, aggregate, aggregate_by_tag, run_harness
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


def _chunk(cid: str, url: str) -> Chunk:
    return Chunk(id=cid, title=cid, url=url, content=f"content {cid}")


def _state(query: str) -> PipelineState:
    s = PipelineState(query=query)
    s.dense = [_chunk("d1", "https://learn.microsoft.com/en-us/azure/right")]
    s.bm25 = [_chunk("b1", "https://learn.microsoft.com/azure/wrong")]
    s.fused = s.dense + s.bm25
    s.reranked = [s.bm25[0], s.dense[0]]  # right page at rank 2
    s.answer = "an answer"
    return s


def test_run_harness_computes_deterministic_metrics_without_evaluator():
    items = [
        TestItem(
            question="q",
            ground_truth="a",
            ground_truth_context="https://learn.microsoft.com/azure/right",
            tags=("lookalike",),
        )
    ]

    async def pipeline_fn(q):
        return _state(q)

    async def evaluator_fn(records):
        return records  # no LLM metrics

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
    r = records[0]
    assert r.tags == ("lookalike",)
    assert r.metrics["hit_rate@dense"] == 1.0
    assert r.metrics["hit_rate@bm25"] == 0.0
    assert r.metrics["hit_rate@reranked"] == 1.0
    assert r.metrics["mrr@reranked"] == 0.5
    assert r.metrics["hit_rate@fused"] == 1.0


def test_aggregate_by_tag_groups_untagged_as_original():
    r1 = EvalRecord(question="q1", answer="a", contexts=[], ground_truth="g",
                    metrics={"m": 1.0}, tags=("paraphrase",))
    r2 = EvalRecord(question="q2", answer="a", contexts=[], ground_truth="g",
                    metrics={"m": 0.0})
    by_tag = aggregate_by_tag([r1, r2])
    assert by_tag["paraphrase"]["m"] == 1.0
    assert by_tag["original"]["m"] == 0.0


def test_aggregate_unchanged_for_plain_metrics():
    r = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g",
                   metrics={"m": 0.5})
    assert aggregate([r]) == {"m": 0.5}
