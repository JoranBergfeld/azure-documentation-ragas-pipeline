from __future__ import annotations

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import RetrievalResult, RetrievalSubstrate


def test_retrieval_result_defaults_empty_stages():
    r = RetrievalResult(candidates=[Chunk(id="1", title="t", url="u", content="c")])
    assert r.stages == {}
    assert [c.id for c in r.candidates] == ["1"]


def test_protocol_is_runtime_checkable():
    class Dummy:
        name = "dummy"

        async def retrieve(self, query: str, k: int) -> RetrievalResult:
            return RetrievalResult(candidates=[])

    assert isinstance(Dummy(), RetrievalSubstrate)
