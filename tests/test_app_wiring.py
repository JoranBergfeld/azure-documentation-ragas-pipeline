from __future__ import annotations

from ragpipe.app_wiring import make_deps
from ragpipe.config import Settings, TestsetMode
from ragpipe.faithfulness_gate import ScoredClaims
from ragpipe.retrieval.substrate import RetrievalResult


def _settings():
    return Settings(
        foundry_project_endpoint="https://p.services.ai.azure.com",
        foundry_chat_model="gpt-4o",
        foundry_embedding_model="emb",
        generator_agent_name="gen",
        testset_mode=TestsetMode.HANDAUTHORED,
    )


def test_make_deps_wires_callables_from_injected_components():
    async def fake_retrieve(q, k):
        return RetrievalResult(candidates=[], stages={})

    class FakeReranker:
        def rerank(self, q, fused, top_k=None):
            return []

    class FakeGen:
        async def generate(self, q, chunks, previous_answer=None):
            return "ans"

    class FakeGate:
        async def score(self, q, a, contexts):
            return ScoredClaims(score=0.9, claims=[])

    deps = make_deps(
        _settings(),
        retrieve=fake_retrieve,
        reranker=FakeReranker(),
        generator=FakeGen(),
        gate=FakeGate(),
    )

    assert deps.threshold == 0.7
    assert deps.max_retries == 2
    assert callable(deps.retrieve)


def test_make_deps_threads_top_k_and_new_signatures():
    class _S:
        faithfulness_threshold = 0.7
        max_retries = 2
        top_k = 4
        candidate_pool = 15

    class _Rerank:
        def __init__(self):
            self.k = None

        def rerank(self, q, fused, top_k=None):
            self.k = top_k
            return fused

    class _Gen:
        def __init__(self):
            self.prev = "sentinel"

        async def generate(self, q, chunks, previous_answer=None):
            self.prev = previous_answer
            return "a"

    class _Gate:
        def score(self, q, a, contexts):
            return ScoredClaims(score=1.0, claims=[])

    async def fake_retrieve(q, k):
        return RetrievalResult(candidates=[], stages={})

    rr, gen = _Rerank(), _Gen()
    deps = make_deps(_S(), retrieve=fake_retrieve, reranker=rr, generator=gen, gate=_Gate())
    assert deps.top_k == 4
    deps.rerank("q", [], 9)
    assert rr.k == 9


def test_build_pipeline_fn_requires_keyword_only_mode():
    import inspect

    from ragpipe.app_wiring import build_pipeline_fn

    params = inspect.signature(build_pipeline_fn).parameters
    assert "mode" in params
    mode = params["mode"]
    assert mode.default is inspect.Parameter.empty
    assert mode.kind is inspect.Parameter.KEYWORD_ONLY


def test_make_deps_judges_chunk_content_only():
    """The judge reads what the generator read, never the situating context."""
    from ragpipe.models import Chunk

    class _S:
        faithfulness_threshold = 0.7
        max_retries = 2
        top_k = 4
        candidate_pool = 15

    seen = {}
    scored = ScoredClaims(score=0.4, claims=[])

    class _Gate:
        def score(self, q, a, contexts):
            seen["contexts"] = contexts
            return scored

    chunk = Chunk(id="a", title="T", url="u", content="body", context="situating note")
    deps = make_deps(_S(), retrieve=None, reranker=None, generator=None, gate=_Gate())

    assert deps.score("q", "a", [chunk]) is scored
    assert seen["contexts"] == ["body"]


def test_build_reranker_is_passthrough_outside_the_rerank_modes():
    from ragpipe.app_wiring import build_reranker
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.passthrough import PassthroughReranker

    for mode in (RetrievalMode.GRAPHRAG, RetrievalMode.COMBINED, RetrievalMode.BASELINE_AGENTIC):
        assert isinstance(build_reranker(_settings(), mode), PassthroughReranker)


def test_build_reranker_uses_the_cross_encoder_with_each_modes_fields(monkeypatch):
    from ragpipe.app_wiring import build_reranker
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.cross_encoder import CrossEncoderReranker

    monkeypatch.setenv("RERANKER_URL", "http://spark:8100")
    expected = {
        RetrievalMode.CONTEXTUAL: True,
        RetrievalMode.BASELINE: False,
        RetrievalMode.RAPTOR_SAC: True,
    }
    for mode, with_context in expected.items():
        reranker = build_reranker(_settings(), mode)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker._with_context is with_context
        assert reranker._mode is mode


def test_build_reranker_refuses_a_rerank_mode_without_a_url():
    import pytest

    from ragpipe.app_wiring import build_reranker
    from ragpipe.config import ModeUnavailable, RetrievalMode

    with pytest.raises(ModeUnavailable) as err:
        build_reranker(_settings(), RetrievalMode.BASELINE)
    assert err.value.reason == "reranker-offline"
