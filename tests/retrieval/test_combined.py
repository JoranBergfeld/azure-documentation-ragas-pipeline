from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.combined import CombinedSubstrate
from ragpipe.retrieval.substrate import RetrievalResult


class _FakeSub:
    def __init__(self, name, chunks):
        self.name = name
        self._chunks = chunks

    async def retrieve(self, query, k):
        return RetrievalResult(candidates=self._chunks, stages={"fused": self._chunks})


@pytest.mark.asyncio
async def test_combined_fuses_both_substrates():
    a = _FakeSub("raptor_sac", [Chunk(id="r1", title="", url="", content="", score=0.9)])
    b = _FakeSub("graphrag", [Chunk(id="g1", title="", url="", content="", score=0.8)])
    sub = CombinedSubstrate(name="combined", substrates=[a, b])
    result = await sub.retrieve("q", k=10)
    assert sub.name == "combined"
    ids = {c.id for c in result.candidates}
    assert ids == {"r1", "g1"}
    assert "raptor_sac:fused" in result.stages
    assert "graphrag:fused" in result.stages
    assert "fused" in result.stages
