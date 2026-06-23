from __future__ import annotations

from ragpipe.config import RetrievalMode
from ragpipe.retrieval.registry import (
    build_substrate,
    experimental_modes,
    is_experimental,
    registered_modes,
)
from ragpipe.retrieval.substrate import RetrievalSubstrate


def test_phase1_modes_registered():
    modes = registered_modes()
    assert RetrievalMode.CONTEXTUAL in modes
    assert RetrievalMode.BASELINE in modes


class _FakeSettings:
    search_index = "ms-docs"
    baseline_index = "baseline"
    candidate_pool = 15
    rrf_k = 60
    top_k = 5


def test_build_substrate_returns_substrate():
    class _Ctx:
        def search_client(self, index):
            return object()

        def embed(self, text):
            return [0.0]

    sub = build_substrate(RetrievalMode.BASELINE, settings=_FakeSettings(), ctx=_Ctx())
    assert isinstance(sub, RetrievalSubstrate)
    assert sub.name == "baseline"


def test_raptor_sac_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.RAPTOR_SAC in registered_modes()


def test_build_raptor_sac_substrate_uses_raptor_index():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        search_index = "ms-docs"
        baseline_index = "baseline"
        raptor_sac_index = "raptor-sac"
        candidate_pool = 15
        rrf_k = 60
        top_k = 5

    class _Ctx:
        def search_client(self, index): return object()
        def embed(self, text): return [0.0]

    sub = build_substrate(RetrievalMode.RAPTOR_SAC, settings=_Settings(), ctx=_Ctx())
    assert sub.name == "raptor_sac"


def test_graphrag_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.GRAPHRAG in registered_modes()


def test_combined_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.COMBINED in registered_modes()


def test_all_modes_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert set(registered_modes()) == set(RetrievalMode)


def test_experimental_modes_are_exactly_unevaluated_agentic_wrappers():
    expected = [
        RetrievalMode.BASELINE_AGENTIC,
        RetrievalMode.RAPTOR_SAC_AGENTIC,
        RetrievalMode.GRAPHRAG_AGENTIC,
        RetrievalMode.COMBINED_AGENTIC,
    ]

    assert experimental_modes() == expected
    assert set(experimental_modes()).issubset(registered_modes())


def test_is_experimental_accepts_modes_and_values():
    experimental = {
        RetrievalMode.BASELINE_AGENTIC,
        RetrievalMode.RAPTOR_SAC_AGENTIC,
        RetrievalMode.GRAPHRAG_AGENTIC,
        RetrievalMode.COMBINED_AGENTIC,
    }
    evaluated = {
        RetrievalMode.CONTEXTUAL,
        RetrievalMode.BASELINE,
        RetrievalMode.RAPTOR_SAC,
        RetrievalMode.GRAPHRAG,
        RetrievalMode.COMBINED,
    }

    for mode in experimental:
        assert is_experimental(mode) is True
        assert is_experimental(mode.value) is True
    for mode in evaluated:
        assert is_experimental(mode) is False
        assert is_experimental(mode.value) is False


def test_every_mode_builds_with_fake_ctx():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        search_index = "ms-docs"
        baseline_index = "baseline"
        raptor_sac_index = "raptor-sac"
        graph_entities_index = "ge"
        graph_relationships_index = "gr"
        graph_communities_index = "gc"
        candidate_pool = 15
        rrf_k = 60
        top_k = 5
        agentic_max_iterations = 3

    class _Search:
        def search(self, *a, **k):
            return []

    class _Ctx:
        def search_client(self, index):
            return _Search()

        def embed(self, text):
            return [0.0]

        def plan(self, query):
            return [query]

    for mode in RetrievalMode:
        sub = build_substrate(mode, _Settings(), _Ctx())
        assert sub.name
