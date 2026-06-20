from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import HybridSubstrate, RetrievalResult, RetrievalSubstrate


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


class _FakeLeg:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query):  # sync, like DenseRetriever/BM25Retriever
        return self._chunks


@pytest.mark.asyncio
async def test_hybrid_substrate_fuses_and_records_stages():
    dense = _FakeLeg([Chunk(id="a", title="", url="", content="x")])
    bm25 = _FakeLeg([Chunk(id="b", title="", url="", content="y")])
    sub = HybridSubstrate(name="baseline", dense=dense, bm25=bm25, rrf_k=60)

    result = await sub.retrieve("q", k=10)

    assert sub.name == "baseline"
    assert set(result.stages) == {"dense", "bm25", "fused"}
    assert [c.id for c in result.stages["dense"]] == ["a"]
    assert {c.id for c in result.candidates} == {"a", "b"}  # fused union
    assert result.candidates is result.stages["fused"]


@pytest.mark.asyncio
async def test_hybrid_accepts_and_ignores_on_event():
    dense = _FakeLeg([Chunk(id="a", title="", url="", content="x")])
    bm25 = _FakeLeg([Chunk(id="b", title="", url="", content="y")])
    sub = HybridSubstrate(name="baseline", dense=dense, bm25=bm25, rrf_k=60)

    seen: list = []
    result = await sub.retrieve("q", k=10, on_event=seen.append)

    assert set(result.stages) == {"dense", "bm25", "fused"}
    assert seen == []  # hybrid does not emit; it just accepts the contract kwarg
