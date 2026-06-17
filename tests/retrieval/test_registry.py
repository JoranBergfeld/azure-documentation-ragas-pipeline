from __future__ import annotations

from ragpipe.config import RetrievalMode
from ragpipe.retrieval.registry import registered_modes, build_substrate
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
