from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


class _EntitySearch:  # pragma: no cover
    """Hybrid (BM25 + vector) search over the entities index."""

    def __init__(self, client, embed_fn: Callable[[str], list[float]], k: int) -> None:
        self._client = client
        self._embed = embed_fn
        self._k = k

    def search_entities(self, query: str, k: int) -> list[Chunk]:
        from azure.search.documents.models import VectorizedQuery

        vector = self._embed(query)
        vq = VectorizedQuery(vector=vector, k=k, fields="description_vector")
        results = self._client.search(
            search_text=query,
            vector_queries=[vq],
            top=k,
            select=["id", "name", "type", "description", "source_urls"],
        )
        chunks = []
        for hit in results:
            chunks.append(
                Chunk(
                    id=hit["id"],
                    title=hit["name"],
                    url=(hit.get("source_urls") or [""])[0],
                    content=f'{hit["name"]} ({hit.get("type", "")}): {hit.get("description", "")}',
                    score=float(hit.get("@search.score", 0.0)),
                )
            )
        return chunks


class _CommunitySearch:  # pragma: no cover
    """Hybrid (BM25 + vector) search over the communities index."""

    def __init__(self, client, embed_fn: Callable[[str], list[float]], k: int) -> None:
        self._client = client
        self._embed = embed_fn
        self._k = k

    def search_communities(self, query: str, k: int) -> list[Chunk]:
        from azure.search.documents.models import VectorizedQuery

        vector = self._embed(query)
        vq = VectorizedQuery(vector=vector, k=k, fields="summary_vector")
        results = self._client.search(
            search_text=query,
            vector_queries=[vq],
            top=k,
            select=["id", "title", "summary"],
        )
        chunks = []
        for hit in results:
            chunks.append(
                Chunk(
                    id=hit["id"],
                    title=hit.get("title", ""),
                    url="",
                    content=hit.get("summary", ""),
                    score=float(hit.get("@search.score", 0.0)),
                )
            )
        return chunks


def build_entity_search(client, embed_fn: Callable[[str], list[float]], k: int) -> _EntitySearch:  # pragma: no cover
    """Build a live entity-search helper backed by an Azure Search client."""
    return _EntitySearch(client, embed_fn, k)


def build_community_search(client, embed_fn: Callable[[str], list[float]], k: int) -> _CommunitySearch:  # pragma: no cover
    """Build a live community-search helper backed by an Azure Search client."""
    return _CommunitySearch(client, embed_fn, k)


def build_adjacency(client) -> dict[str, list[Chunk]]:  # pragma: no cover
    """Read ALL relationship docs and build a name -> [Chunk] adjacency map.

    Keys are upper-cased entity names (matching entity .title). Each relationship
    doc is registered under both its source and target so a seed entity can find
    all edges incident to it.
    """
    adjacency: dict[str, list[Chunk]] = {}
    # No `top=`: Azure Search caps a single page and would silently drop edges past
    # it. The SearchItemPaged iterator pages through the full result set, so the
    # whole relationships index is loaded into the adjacency map.
    results = client.search(
        search_text="*",
        select=["id", "source", "target", "description", "weight", "source_urls"],
    )
    for hit in results:
        rel_chunk = Chunk(
            id=hit["id"],
            title=f'{hit["source"]} — {hit["target"]}',
            url=(hit.get("source_urls") or [""])[0],
            content=hit.get("description", ""),
            score=float(hit.get("weight", 0.0)),
        )
        source_key = hit["source"].upper()
        target_key = hit["target"].upper()
        adjacency.setdefault(source_key, []).append(rel_chunk)
        adjacency.setdefault(target_key, []).append(rel_chunk)
    return adjacency


class GraphRAGSubstrate:
    """Local (entity match + 1-hop relationship expansion) + global (community
    report ranking) search, fused by RRF. entity_search/community_search expose
    search_entities(query,k)/search_communities(query,k) -> list[Chunk]. adjacency
    maps an entity title -> its relationship Chunks (built once from the
    relationships index at wiring time)."""

    def __init__(self, *, name, entity_search, community_search, adjacency, rrf_k=60):
        self.name = name
        self._entities = entity_search
        self._communities = community_search
        self._adjacency = adjacency
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int, on_event=None) -> RetrievalResult:
        seeds = self._entities.search_entities(query, k)
        expanded: list[Chunk] = list(seeds)
        seen = {c.id for c in seeds}
        for seed in seeds:
            for rel in self._adjacency.get(seed.title, []):
                if rel.id not in seen:
                    expanded.append(rel)
                    seen.add(rel.id)
        glob = self._communities.search_communities(query, k)
        fused = reciprocal_rank_fusion(expanded, glob, k=self._rrf_k)
        return RetrievalResult(candidates=fused, stages={"local": expanded, "global": glob, "fused": fused})
