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


class _FakeTable:
    def __init__(self, dense, lexical=(), scan=()):
        self._dense = list(dense)
        self._lexical = list(lexical)
        self._scan = list(scan)

    def nearest(self, vector, k):
        return self._dense[:k]

    def bm25(self, query, k):
        return self._lexical[:k]

    def scan(self):
        return self._scan


def test_community_search_populates_url_from_source_urls():
    from ragpipe.retrieval.graph_substrate import build_community_search

    table = _FakeTable([
        {"id": "community-0", "title": "Compute", "summary": "Compute services",
         "source_urls": ["https://learn.microsoft.com/azure/functions", "https://learn.microsoft.com/azure/blob"],
         "score": 0.7},
        {"id": "community-1", "title": "Empty", "summary": "No sources", "source_urls": [], "score": 0.3},
    ])
    com = build_community_search(table, embed_fn=lambda q: [0.1, 0.2], k=5)
    chunks = com.search_communities("query", 5)
    # First source URL is surfaced as the chunk URL so @global is URL-scorable.
    assert chunks[0].url == "https://learn.microsoft.com/azure/functions"
    # A community with no source URLs degrades gracefully to an empty URL.
    assert chunks[1].url == ""


def test_entity_search_fuses_vector_and_lexical_hits():
    from ragpipe.retrieval.graph_substrate import build_entity_search

    def ent(i):
        return {"id": f"entity-{i}", "name": f"E{i}", "type": "service",
                "description": f"d{i}", "source_urls": [f"https://x/{i}"], "score": 1.0}

    # entity-2 is second on both legs, so RRF puts it ahead of either leg's winner.
    table = _FakeTable(dense=[ent(1), ent(2)], lexical=[ent(3), ent(2)])
    ents = build_entity_search(table, embed_fn=lambda q: [0.0], k=2)
    chunks = ents.search_entities("q", 2)
    assert [c.id for c in chunks] == ["entity-2", "entity-1"]
    assert chunks[0].title == "E2"
    assert chunks[0].content == "E2 (service): d2"


def test_build_adjacency_registers_each_edge_under_both_ends():
    from ragpipe.retrieval.graph_substrate import build_adjacency

    table = _FakeTable([], scan=[
        {"id": "rel-1", "source": "Azure Functions", "target": "Durable Task",
         "description": "uses", "weight": 9.0, "source_urls": ["https://x/f"]},
    ])
    adjacency = build_adjacency(table)
    assert set(adjacency) == {"AZURE FUNCTIONS", "DURABLE TASK"}
    edge = adjacency["DURABLE TASK"][0]
    assert (edge.id, edge.score, edge.url) == ("rel-1", 9.0, "https://x/f")
