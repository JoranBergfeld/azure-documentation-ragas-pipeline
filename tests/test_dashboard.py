from __future__ import annotations

from app.dashboard import stage_chunk_tables
from ragpipe.models import Chunk, PipelineState


def test_stage_chunk_tables_uses_dynamic_stages():
    s = PipelineState(query="q")
    s.set_stage("local", [Chunk(id="1", title="t", url="u", content="hello")])
    s.set_reranked([Chunk(id="1", title="t", url="u", content="hello")])
    tables = stage_chunk_tables(s)
    assert "local" in tables
    assert "reranked" in tables
