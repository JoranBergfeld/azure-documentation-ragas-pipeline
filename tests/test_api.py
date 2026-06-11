from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragpipe.models import Chunk, PipelineState

import app.api as api


@pytest.fixture
def client():
    return TestClient(api.app)


def _state() -> PipelineState:
    def c(cid, score):
        return Chunk(id=cid, title=f"Doc {cid}", url=f"http://{cid}", content="x", score=score)

    s = PipelineState(query="what is RRF?")
    s.dense = [c("a", 0.8)]
    s.bm25 = [c("b", 0.7)]
    s.fused = [c("a", 0.5), c("b", 0.4)]
    s.reranked = [c("a", 0.99)]
    s.answer = "RRF merges ranked lists."
    s.faithfulness = 0.98
    s.attempt = 1
    s.low_confidence = False
    return s


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_run_returns_answer_and_stages(client):
    async def fake_pipeline(query: str) -> PipelineState:
        return _state()

    api.app.dependency_overrides[api.get_pipeline_fn] = lambda: fake_pipeline
    try:
        res = client.post("/run", json={"query": "what is RRF?"})
    finally:
        api.app.dependency_overrides.clear()

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "RRF merges ranked lists."
    assert body["faithfulness"] == 0.98
    assert body["lowConfidence"] is False
    assert body["attempt"] == 1
    assert [r["title"] for r in body["stages"]["reranked"]] == ["Doc a"]
    assert body["stages"]["fused"][0]["rank"] == 1


def test_eval_reads_results_file(client, tmp_path, monkeypatch):
    results = {
        "means": {"faithfulness": 1.0, "context_precision@dense": 0.54},
        "coverage": {"faithfulness": {"valid": 3, "total": 3}},
        "records": [{"question": "q1"}],
    }
    f = tmp_path / "eval_results.json"
    f.write_text(json.dumps(results))
    monkeypatch.setattr(api, "EVAL_RESULTS_PATH", str(f))

    res = client.get("/eval")
    assert res.status_code == 200
    body = res.json()
    assert body["nRecords"] == 1
    assert {"metric": "faithfulness", "meanScore": 1.0, "coverage": "3/3"} in body["overall"]
    assert body["perStage"]["dense"]["context_precision"] == 0.54


def test_run_reports_abstention():
    from fastapi.testclient import TestClient

    from app import api
    from ragpipe.models import PipelineState

    async def fake_pipeline(q):
        return PipelineState(query=q, answer="abstained text", abstained=True, low_confidence=True)

    api.app.dependency_overrides[api.get_pipeline_fn] = lambda: fake_pipeline
    try:
        resp = TestClient(api.app).post("/run", json={"query": "x"})
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True
    finally:
        api.app.dependency_overrides.clear()


def test_eval_missing_file_returns_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EVAL_RESULTS_PATH", str(tmp_path / "nope.json"))
    res = client.get("/eval")
    assert res.status_code == 200
    assert res.json() == {"overall": [], "perStage": {}, "nRecords": 0}
