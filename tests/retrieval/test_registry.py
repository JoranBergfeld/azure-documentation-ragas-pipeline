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
