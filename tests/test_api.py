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
    s.set_stage("dense", [c("a", 0.8)])
    s.set_stage("bm25", [c("b", 0.7)])
    s.set_stage("fused", [c("a", 0.5), c("b", 0.4)])
    s.set_reranked([c("a", 0.99)])
    s.answer = "RRF merges ranked lists."
    s.faithfulness = 0.98
    s.attempt = 1
    s.low_confidence = False
    return s


def _make_factory(pipeline_fn):
    """Return a factory override that ignores mode and always uses pipeline_fn."""

    async def factory(mode: str):
        return pipeline_fn

    def get_factory():
        return factory

    return get_factory


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_run_returns_answer_and_stages(client):
    async def fake_pipeline(query: str) -> PipelineState:
        return _state()

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        res = client.post("/run", json={"query": "what is RRF?", "mode": "contextual"})
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

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        resp = TestClient(api.app).post("/run", json={"query": "x", "mode": "contextual"})
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True
    finally:
        api.app.dependency_overrides.clear()


def test_eval_missing_file_returns_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EVAL_RESULTS_PATH", str(tmp_path / "nope.json"))
    res = client.get("/eval")
    assert res.status_code == 200
    assert res.json() == {"overall": [], "perStage": {}, "nRecords": 0}


# --- new tests for mode param and /compare ---


def _fake_state(mode: str) -> PipelineState:
    s = PipelineState(query="q")
    s.set_stage("fused", [Chunk(id="1", title="t", url="u", content="c")])
    s.set_reranked([Chunk(id="1", title="t", url="u", content="c")])
    s.answer = f"answer-{mode}"
    s.faithfulness = 0.9
    return s


def test_run_with_explicit_mode():
    async def fake_pipeline(q: str) -> PipelineState:
        return _fake_state("baseline")

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        resp = TestClient(api.app).post("/run", json={"query": "q", "mode": "baseline"})
    finally:
        api.app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "baseline"
    assert "answer" in body


def test_run_requires_mode(client):
    res = client.post("/run", json={"query": "x"})
    assert res.status_code == 422


def test_run_rejects_invalid_mode(client):
    res = client.post("/run", json={"query": "x", "mode": "not-a-mode"})
    assert res.status_code == 422


def test_compare_runs_multiple_modes():
    async def fake_factory(mode: str):
        async def fn(q: str) -> PipelineState:
            return _fake_state(mode)

        return fn

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = lambda: fake_factory
    try:
        resp = TestClient(api.app).post(
            "/compare", json={"query": "q", "modes": ["baseline", "contextual"]}
        )
    finally:
        api.app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert {r["mode"] for r in body["results"]} == {"baseline", "contextual"}
    assert all("answer" in r for r in body["results"])


def test_eval_new_shape(client, tmp_path, monkeypatch):
    results = {
        "means_by_mode": {
            "baseline": {"faithfulness": 0.8},
            "contextual": {"faithfulness": 0.9},
        },
        "modes": {
            "baseline": {},
            "contextual": {},
        },
    }
    f = tmp_path / "eval_results.json"
    f.write_text(json.dumps(results))
    monkeypatch.setattr(api, "EVAL_RESULTS_PATH", str(f))

    res = client.get("/eval")
    assert res.status_code == 200
    body = res.json()
    assert "meansByMode" in body
    assert set(body["modes"]) == {"baseline", "contextual"}


def test_run_stream_emits_progress_and_result():
    from ragpipe.progress import ProgressEvent

    async def fake_pipeline(query, *, on_event=None):
        on_event(ProgressEvent(phase="retrieve", status="start", message="Retrieving"))
        on_event(ProgressEvent(phase="generate", status="complete", attempt=0, message="Answer generated"))
        return _state()

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        res = TestClient(api.app).post("/run/stream", json={"query": "what is RRF?", "mode": "contextual"})
    finally:
        api.app.dependency_overrides.clear()

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    body = res.text
    assert "event: progress" in body
    assert "event: result" in body
    assert '"phase": "retrieve"' in body
    assert "RRF merges ranked lists." in body  # serialized final state


def test_run_stream_emits_error_frame_on_failure():
    async def boom_pipeline(query, *, on_event=None):
        raise RuntimeError("kaboom")

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(boom_pipeline)
    try:
        res = TestClient(api.app).post("/run/stream", json={"query": "x", "mode": "contextual"})
    finally:
        api.app.dependency_overrides.clear()

    assert res.status_code == 200
    assert "event: error" in res.text
    assert "RuntimeError" in res.text


# --- issue #11: experimental flag for the unevaluated *_agentic modes ---


def test_run_payload_marks_evaluated_mode_not_experimental():
    async def fake_pipeline(q: str) -> PipelineState:
        return _fake_state("contextual")

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        resp = TestClient(api.app).post("/run", json={"query": "q", "mode": "contextual"})
    finally:
        api.app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["experimental"] is False


def test_run_payload_marks_agentic_mode_experimental():
    async def fake_pipeline(q: str) -> PipelineState:
        return _fake_state("combined_agentic")

    api.app.dependency_overrides[api.get_pipeline_fn_for_mode] = _make_factory(fake_pipeline)
    try:
        resp = TestClient(api.app).post("/run", json={"query": "q", "mode": "combined_agentic"})
    finally:
        api.app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["experimental"] is True


def test_modes_lists_all_modes_with_experimental_flags(client):
    res = client.get("/modes")
    assert res.status_code == 200
    body = res.json()
    by_mode = {item["mode"]: item["experimental"] for item in body["modes"]}
    # all 9 registry modes are present
    assert len(by_mode) == 9
    assert by_mode["contextual"] is False
    assert by_mode["combined_agentic"] is True
    # the experimental list names exactly the four agentic wrappers
    assert set(body["experimental"]) == {
        "baseline_agentic",
        "raptor_sac_agentic",
        "graphrag_agentic",
        "combined_agentic",
    }
