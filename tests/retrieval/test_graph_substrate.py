from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.graph_substrate import GraphRAGSubstrate
from ragpipe.retrieval.query_class import QueryClass


class _FakeEntitySearch:
    def search_entities(self, query, k):
        return [Chunk(id="entity-0", title="AZURE FUNCTIONS", url="u1",
                      content="AZURE FUNCTIONS (service): Serverless", score=0.9)]


class _FakeCommunitySearch:
    def __init__(self):
        self.calls = 0

    def search_communities(self, query, k):
        self.calls += 1
        return [Chunk(id="community-1", title="Compute", url="",
                      content="Compute services summary", score=0.8)]


def _adjacency():
    return {"AZURE FUNCTIONS": [Chunk(id="rel-0", title="AZURE FUNCTIONS — BLOB STORAGE",
                                      url="u1", content="triggered by blob", score=0.5)]}


@pytest.mark.asyncio
async def test_graph_substrate_local_query_skips_global_leg():
    # A factoid/LOCAL query must not engage the global community leg (issue #8):
    # the empty-URL summary would otherwise evict the precise leaf chunk.
    com = _FakeCommunitySearch()
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=com,
        adjacency=_adjacency(),
    )
    result = await sub.retrieve("how do functions read blobs", k=10)
    assert sub.name == "graphrag"
    assert set(result.stages) == {"local", "global", "fused"}
    assert result.stages["global"] == []
    assert com.calls == 0
    ids = {c.id for c in result.candidates}
    assert ids == {"entity-0", "rel-0"}
    assert "community-1" not in ids


@pytest.mark.asyncio
async def test_graph_substrate_global_query_fuses_local_and_global():
    com = _FakeCommunitySearch()
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=com,
        adjacency=_adjacency(),
    )
    result = await sub.retrieve("compare Azure Functions and Logic Apps", k=10)
    assert com.calls == 1
    ids = {c.id for c in result.candidates}
    assert "entity-0" in ids and "rel-0" in ids and "community-1" in ids


@pytest.mark.asyncio
async def test_graph_substrate_routing_disabled_always_fuses():
    com = _FakeCommunitySearch()
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=com,
        adjacency=_adjacency(),
        routing=False,
    )
    result = await sub.retrieve("how do functions read blobs", k=10)
    assert com.calls == 1
    ids = {c.id for c in result.candidates}
    assert "community-1" in ids


@pytest.mark.asyncio
async def test_graph_substrate_accepts_injected_classifier():
    com = _FakeCommunitySearch()
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=com,
        adjacency=_adjacency(),
        classify_fn=lambda q: QueryClass.GLOBAL,
    )
    result = await sub.retrieve("anything", k=10)
    assert com.calls == 1
    assert "community-1" in {c.id for c in result.candidates}


class _FakeSearchClient:
    def __init__(self, hits):
        self._hits = hits

    def search(self, *args, **kwargs):
        self._select = kwargs.get("select")
        return iter(self._hits)


def test_community_search_populates_url_from_source_urls():
    from ragpipe.retrieval.graph_substrate import build_community_search

    client = _FakeSearchClient([
        {"id": "community-0", "title": "Compute", "summary": "Compute services",
         "source_urls": ["https://learn.microsoft.com/azure/functions", "https://learn.microsoft.com/azure/blob"],
         "@search.score": 0.7},
        {"id": "community-1", "title": "Empty", "summary": "No sources", "@search.score": 0.3},
    ])
    com = build_community_search(client, embed_fn=lambda q: [0.1, 0.2], k=5)
    chunks = com.search_communities("query", 5)
    assert "source_urls" in client._select
    # First source URL is surfaced as the chunk URL so @global is URL-scorable.
    assert chunks[0].url == "https://learn.microsoft.com/azure/functions"
    # A community with no source URLs degrades gracefully to an empty URL.
    assert chunks[1].url == ""
