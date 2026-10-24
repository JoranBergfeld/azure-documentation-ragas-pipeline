from __future__ import annotations

from ragpipe.config import RetrievalMode
from ragpipe.retrieval.registry import registered_modes, build_substrate
from ragpipe.retrieval.substrate import RetrievalSubstrate


def test_phase1_modes_registered():
    modes = registered_modes()
    assert RetrievalMode.CONTEXTUAL in modes
    assert RetrievalMode.BASELINE in modes


class _FakeSettings:
    candidate_pool = 15
    rrf_k = 60
    top_k = 5


def test_build_substrate_returns_substrate():
    class _Ctx:
        def table(self, spec):
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


def test_build_raptor_sac_substrate_uses_raptor_table():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        candidate_pool = 15
        rrf_k = 60
        top_k = 5

    class _Ctx:
        seen: list[str] = []
        def table(self, spec):
            _Ctx.seen.append(spec.table)
            return object()
        def embed(self, text): return [0.0]

    sub = build_substrate(RetrievalMode.RAPTOR_SAC, settings=_Settings(), ctx=_Ctx())
    assert sub.name == "raptor_sac"
    assert _Ctx.seen == ["raptor_sac"]


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


def test_every_mode_builds_with_fake_ctx():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        candidate_pool = 15
        rrf_k = 60
        top_k = 5
        agentic_max_iterations = 3

    class _Table:
        def scan(self):
            return []

    class _Ctx:
        def table(self, spec):
            return _Table()

        def embed(self, text):
            return [0.0]

        def plan(self, query):
            return [query]

    for mode in RetrievalMode:
        sub = build_substrate(mode, _Settings(), _Ctx())
        assert sub.name
