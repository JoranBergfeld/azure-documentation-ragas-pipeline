from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.graph_substrate import GraphRAGSubstrate


class _FakeEntitySearch:
    def search_entities(self, query, k):
        return [Chunk(id="entity-0", title="AZURE FUNCTIONS", url="u1",
                      content="AZURE FUNCTIONS (service): Serverless", score=0.9)]


class _FakeCommunitySearch:
    def search_communities(self, query, k):
        return [Chunk(id="community-1", title="Compute", url="",
                      content="Compute services summary", score=0.8)]


@pytest.mark.asyncio
async def test_graph_substrate_fuses_local_and_global():
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=_FakeCommunitySearch(),
        adjacency={"AZURE FUNCTIONS": [Chunk(id="rel-0", title="AZURE FUNCTIONS — BLOB STORAGE",
                                             url="u1", content="triggered by blob", score=0.5)]},
    )
    result = await sub.retrieve("how do functions read blobs", k=10)
    assert sub.name == "graphrag"
    assert set(result.stages) == {"local", "global", "fused"}
    ids = {c.id for c in result.candidates}
    assert "entity-0" in ids and "community-1" in ids
    assert "rel-0" in ids
