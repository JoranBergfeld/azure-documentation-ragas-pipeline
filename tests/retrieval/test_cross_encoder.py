from __future__ import annotations

import json

import httpx
import pytest

from ragpipe.config import ModeUnavailable, RetrievalMode
from ragpipe.models import Chunk
from ragpipe.retrieval import cross_encoder
from ragpipe.retrieval.cross_encoder import (
    CrossEncoderReranker,
    RerankerConfig,
    document_text,
    health,
    reranker_reachable,
)

CONFIG = RerankerConfig(url="http://spark:8100", api_key="k")


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="a", title="A", url="u/a", content="alpha", context="ctx a", score=0.03),
        Chunk(id="b", title="B", url="u/b", content="beta", context="ctx b", score=0.02),
        Chunk(id="c", title="C", url="u/c", content="gamma", context="ctx c", score=0.01),
    ]


def _reranker(handler, *, mode=RetrievalMode.CONTEXTUAL, with_context=True, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return CrossEncoderReranker(
        CONFIG, mode, top_k=2, with_context=with_context, client=client, backoff=0, **kw
    )


def test_rerank_posts_documents_and_orders_by_relevance():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.40},
        ], "usage": {"total_tokens": 42}})

    out = _reranker(handler).rerank("which?", _chunks(), top_k=2)

    assert [c.id for c in out] == ["c", "a"]
    assert [c.score for c in out] == [0.99, 0.40]
    assert out[0].context == "ctx c"
    req = seen[0]
    assert req.url == "http://spark:8100/v1/rerank"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == "nvidia/llama-nemotron-rerank-1b-v2"
    assert body["query"] == "which?"
    assert body["top_n"] == 2
    assert body["documents"][0] == "A\n\nalpha\n\nctx a"


def test_baseline_leaves_the_context_out():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    _reranker(handler, mode=RetrievalMode.BASELINE, with_context=False).rerank("q", _chunks(), 1)
    assert seen[0]["documents"] == ["A\n\nalpha", "B\n\nbeta", "C\n\ngamma"]


def test_document_text_skips_empty_fields():
    chunk = Chunk(id="x", title="", url="", content="body", context="")
    assert document_text(chunk, with_context=True) == "body"


def test_top_n_never_exceeds_the_candidates():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"index": i, "relevance_score": 1.0 - i / 10}
                                                     for i in range(3)]})

    out = _reranker(handler).rerank("q", _chunks(), top_k=11)
    assert seen[0]["top_n"] == 3
    assert len(out) == 3


def test_no_candidates_means_no_call():
    def handler(request):
        raise AssertionError("should not be called")

    assert _reranker(handler).rerank("q", [], top_k=5) == []


def test_a_5xx_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.8}]})

    out = _reranker(handler).rerank("q", _chunks(), top_k=1)
    assert [c.id for c in out] == ["b"]
    assert len(calls) == 2


def test_an_outage_is_mode_unavailable_and_greys_the_modes_out(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ModeUnavailable) as err:
        _reranker(handler, max_retries=2).rerank("q", _chunks(), top_k=2)
    assert err.value.reason == "reranker-offline"
    assert err.value.mode == "contextual"
    assert len(calls) == 3
    # The failure is cached as down, so /modes doesn't wait for the next probe.
    monkeypatch.setattr(cross_encoder, "_probe", lambda *a: pytest.fail("probed"))
    assert health.reachable(CONFIG) is False


def test_a_timeout_is_retried_then_gives_up():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModeUnavailable):
        _reranker(handler, max_retries=1).rerank("q", _chunks(), top_k=2)
    assert len(calls) == 2


def test_a_bad_key_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": "Unauthorized"})

    with pytest.raises(ModeUnavailable):
        _reranker(handler).rerank("q", _chunks(), top_k=2)
    assert len(calls) == 1


def test_a_malformed_answer_is_mode_unavailable():
    def handler(request):
        return httpx.Response(200, json={"results": [{"index": 9, "relevance_score": 1.0}]})

    with pytest.raises(ModeUnavailable):
        _reranker(handler).rerank("q", _chunks(), top_k=2)


def test_config_from_env(monkeypatch):
    assert RerankerConfig.from_env() is None
    monkeypatch.setenv("RERANKER_URL", "http://spark:8100/")
    monkeypatch.setenv("RERANKER_API_KEY", "secret")
    config = RerankerConfig.from_env()
    assert config == RerankerConfig(url="http://spark:8100", api_key="secret")
    monkeypatch.setenv("RERANKER_MODEL", "other/model")
    assert RerankerConfig.from_env().model == "other/model"


def test_no_key_sends_no_authorization_header():
    assert RerankerConfig(url="http://x").headers() == {}


def test_health_probe_is_cached(monkeypatch):
    probes = []

    def handler(request):
        probes.append(request)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert health.reachable(CONFIG, client) is True
    assert health.reachable(CONFIG, client) is True
    assert len(probes) == 1
    assert probes[0].url == "http://spark:8100/health"

    # Past the TTL it asks again.
    now = cross_encoder.time.monotonic()
    monkeypatch.setattr(
        cross_encoder.time, "monotonic", lambda: now + cross_encoder.HEALTH_TTL_SECONDS + 1
    )
    assert health.reachable(CONFIG, client) is True
    assert len(probes) == 2


def test_health_probe_failure_is_unreachable():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert health.reachable(CONFIG, client) is False


def test_reachable_needs_a_url(monkeypatch):
    monkeypatch.setattr(cross_encoder, "_probe", lambda *a: True)
    assert reranker_reachable() is False
    monkeypatch.setenv("RERANKER_URL", "http://spark:8100")
    assert reranker_reachable() is True
