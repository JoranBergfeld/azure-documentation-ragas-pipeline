from __future__ import annotations

from ragpipe.models import Chunk
from ragpipe.retrieval.passthrough import PassthroughReranker


def test_passthrough_sorts_by_score_and_truncates():
    chunks = [Chunk(id="a", title="", url="", content="", score=0.2),
              Chunk(id="b", title="", url="", content="", score=0.9),
              Chunk(id="c", title="", url="", content="", score=0.5)]
    out = PassthroughReranker(top_k=2).rerank("q", chunks, top_k=2)
    assert [c.id for c in out] == ["b", "c"]
