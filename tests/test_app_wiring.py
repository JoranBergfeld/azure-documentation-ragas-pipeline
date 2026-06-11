from ragpipe.app_wiring import make_deps
from ragpipe.config import Settings, TestsetMode


def _settings():
    return Settings(
        foundry_project_endpoint="https://p.services.ai.azure.com",
        foundry_chat_model="gpt-4o",
        foundry_embedding_model="emb",
        search_endpoint="https://s.search.windows.net",
        search_index="idx",
        generator_agent_name="gen",
        testset_mode=TestsetMode.HANDAUTHORED,
    )


def test_make_deps_wires_callables_from_injected_components():
    class FakeDense:
        def retrieve(self, q):
            return []

    class FakeBm25:
        def retrieve(self, q):
            return []

    class FakeReranker:
        def rerank(self, q, fused):
            return []

    class FakeGen:
        async def generate(self, q, chunks):
            return "ans"

    class FakeScorer:
        async def score(self, q, a, c):
            return 0.9

    deps = make_deps(
        _settings(),
        dense=FakeDense(),
        bm25=FakeBm25(),
        reranker=FakeReranker(),
        generator=FakeGen(),
        scorer=FakeScorer(),
    )

    assert deps.threshold == 0.7
    assert deps.max_retries == 2
    assert callable(deps.dense)
    assert deps.dense("q") == []


def test_make_deps_threads_top_k_and_new_signatures():
    from ragpipe.app_wiring import make_deps

    class _S:
        faithfulness_threshold = 0.7
        max_retries = 2
        rrf_k = 60
        top_k = 4

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

    class _Id:
        def retrieve(self, q):
            return []

        def score(self, q, a, c):
            return 1.0

    rr, gen = _Rerank(), _Gen()
    deps = make_deps(_S(), dense=_Id(), bm25=_Id(), reranker=rr, generator=gen, scorer=_Id())
    assert deps.top_k == 4
    deps.rerank("q", [], 9)
    assert rr.k == 9
