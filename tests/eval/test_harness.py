import pytest

from ragpipe.eval.harness import (
    EvalRecord,
    aggregate,
    aggregate_by_mode,
    parse_stage_metric,
    run_harness,
    stage_metric_key,
)
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


def test_offline_judge_requires_offline_judge_model():
    from ragpipe.eval.harness import _build_ragas_clients

    class _Settings:
        foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
        foundry_chat_model = "gpt-5.4"
        foundry_embedding_model = "text-embedding-3-small"
        offline_judge_model = None

    with pytest.raises(ValueError, match="OFFLINE_JUDGE_MODEL"):
        _build_ragas_clients(_Settings())


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
        s.set_stage("dense", [Chunk(id="d", title="t", url="u", content="dense-ctx")])
        s.set_stage("bm25", [Chunk(id="b", title="t", url="u", content="bm25-ctx")])
        s.set_stage("fused", [Chunk(id="f", title="t", url="u", content="fused-ctx")])
        s.set_reranked([Chunk(id="c", title="t", url="u", content="ctx-content")])
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


@pytest.mark.asyncio
async def test_harness_records_abstention_metric():
    async def pipeline_fn(q):
        s = PipelineState(query=q, answer="abstained")
        s.abstained = True
        return s

    async def evaluator_fn(records):
        return records

    items = [TestItem(question="q", ground_truth="g", ground_truth_context="http://u")]
    records = await run_harness(items, pipeline_fn, evaluator_fn)
    assert records[0].abstained is True
    assert records[0].metrics["abstained"] == 1.0


def test_aggregate_reports_abstention_rate():
    r1 = EvalRecord(question="a", answer="x", contexts=[], ground_truth="g",
                    metrics={"abstained": 1.0})
    r2 = EvalRecord(question="b", answer="y", contexts=[], ground_truth="g",
                    metrics={"abstained": 0.0})
    assert aggregate([r1, r2])["abstained"] == 0.5


@pytest.mark.asyncio
async def test_run_harness_reads_dynamic_stages():
    async def pipeline_fn(q):
        s = PipelineState(query=q)
        s.set_stage("local", [Chunk(id="1", title="", url="http://x", content="c")])
        s.set_reranked([Chunk(id="1", title="", url="http://x", content="c")])
        s.answer = "a"
        return s
    async def evaluator_fn(records):
        return records
    items = [TestItem(question="q", ground_truth="g", ground_truth_context="http://x")]
    recs = await run_harness(items, pipeline_fn, evaluator_fn)
    assert "local" in recs[0].stage_urls
    assert "reranked" in recs[0].stage_urls


def test_aggregate_by_mode():
    a = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g")
    a.metrics["hit_rate@reranked"] = 1.0
    b = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g")
    b.metrics["hit_rate@reranked"] = 0.0
    out = aggregate_by_mode({"baseline": [a], "contextual": [b]})
    assert out["baseline"]["hit_rate@reranked"] == 1.0
    assert out["contextual"]["hit_rate@reranked"] == 0.0


def test_ragas_run_config_overrides_starving_defaults():
    # The reasoning offline judge timed out under RAGAS defaults (timeout=180s,
    # max_workers=16): every faithfulness/context_precision job hit the wall and
    # dropped to NaN. The run config must give a longer per-job budget and fewer
    # concurrent jobs so those metrics actually compute.
    from ragpipe.eval.harness import (
        RAGAS_JOB_TIMEOUT,
        RAGAS_MAX_WORKERS,
        _ragas_run_config,
    )
    from ragpipe.guardrail import _ensure_ragas_importable

    _ensure_ragas_importable()
    rc = _ragas_run_config()

    assert rc.timeout == RAGAS_JOB_TIMEOUT
    assert rc.max_workers == RAGAS_MAX_WORKERS
    # Strictly more generous than the defaults that starved the reasoning judge.
    assert rc.timeout > 180
    assert rc.max_workers < 16
