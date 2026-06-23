from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.graph_substrate import GraphRAGSubstrate, classify_query


class _FakeEntitySearch:
    def search_entities(self, query, k):
        return [
            Chunk(
                id="entity-0",
                title="AZURE FUNCTIONS",
                url="u1",
                content="AZURE FUNCTIONS (service): Serverless",
                score=0.9,
            )
        ]


class _FakeCommunitySearch:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def search_communities(self, query, k):
        self.calls.append((query, k))
        return [
            Chunk(
                id="community-1",
                title="Compute",
                url="",
                content="Compute services summary",
                score=0.8,
            )
        ]


class _FailingCommunitySearch:
    def search_communities(self, query, k):
        raise AssertionError("community search should not be consulted for local route")


@pytest.mark.asyncio
async def test_graph_substrate_fuses_local_and_global_for_global_route():
    communities = _FakeCommunitySearch()
    events = []
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=communities,
        adjacency={
            "AZURE FUNCTIONS": [
                Chunk(
                    id="rel-0",
                    title="AZURE FUNCTIONS — BLOB STORAGE",
                    url="u1",
                    content="triggered by blob",
                    score=0.5,
                )
            ]
        },
        route_fn=lambda _query: "global",
    )
    result = await sub.retrieve(
        "give an overview of azure compute services",
        k=10,
        on_event=events.append,
    )
    assert sub.name == "graphrag"
    assert set(result.stages) == {"local", "global", "fused"}
    ids = {c.id for c in result.candidates}
    assert "entity-0" in ids and "community-1" in ids
    assert "rel-0" in ids
    assert communities.calls == [("give an overview of azure compute services", 10)]
    assert events[-1].phase == "retrieve.route"
    assert events[-1].message == "global"
    assert events[-1].detail == {"route": "global"}


@pytest.mark.asyncio
async def test_graph_substrate_uses_local_only_for_local_route():
    events = []
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=_FailingCommunitySearch(),
        adjacency={
            "AZURE FUNCTIONS": [
                Chunk(
                    id="rel-0",
                    title="AZURE FUNCTIONS — BLOB STORAGE",
                    url="u1",
                    content="triggered by blob",
                    score=0.5,
                )
            ]
        },
        route_fn=lambda _query: "local",
    )
    result = await sub.retrieve("how do functions read blobs", k=10, on_event=events.append)
    assert set(result.stages) == {"local", "fused"}
    assert "global" not in result.stages
    assert result.stages["fused"] == result.stages["local"]
    assert [c.id for c in result.candidates] == ["entity-0", "rel-0"]
    assert events[-1].phase == "retrieve.route"
    assert events[-1].message == "local"
    assert events[-1].detail == {"route": "local"}


@pytest.mark.parametrize(
    "query",
    [
        "how do functions read blobs",
        "what is the max message size for azure service bus",
        "where is diagnostic logging configured",
        "which setting enables managed identity",
    ],
)
def test_classify_query_defaults_factoid_queries_to_local(query):
    assert classify_query(query) == "local"


@pytest.mark.parametrize(
    "query",
    [
        "give an overview of azure compute services",
        "summarize the main storage options",
        "compare authentication approaches across Azure services",
        "what are the different types of networking services",
        "how do these categories relate overall",
    ],
)
def test_classify_query_routes_sensemaking_queries_to_global(query):
    assert classify_query(query) == "global"
