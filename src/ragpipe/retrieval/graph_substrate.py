from __future__ import annotations

import re
from typing import Callable

from ragpipe.models import Chunk
from ragpipe.progress import emit
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


def classify_query(query: str) -> str:
    """Route GraphRAG queries with a conservative breadth-cue heuristic.

    Factoid single-page lookups default to local retrieval. Global/community
    retrieval is reserved for explicit sensemaking cues; an LLM or ``ctx.plan``
    router is a documented future seam.
    """
    normalized = query.lower()
    global_cues = (
        "overview",
        "summarize",
        "summary",
        "compare",
        "comparison",
        "across",
        "theme",
        "themes",
        "trends",
        "in general",
        "landscape",
        "broadly",
        "overall",
    )
    if any(cue in normalized for cue in global_cues):
        return "global"
    if re.search(r"\bmain\b.*\b(services|types|categories|options)\b", normalized):
        return "global"
    if re.search(r"\bwhat are the\b.*\b(types|categories|kinds)\b", normalized):
        return "global"
    if re.search(r"\bhow do\b.*\brelate\b", normalized):
        return "global"
    return "local"


class GraphRAGSubstrate:
    """Routed GraphRAG retrieval over local graph neighborhoods and communities.

    Local search uses entity match + 1-hop relationship expansion. Global search
    ranks community reports and is RRF-fused with local results only when the
    route asks for breadth. ``entity_search``/``community_search`` expose
    ``search_entities(query,k)``/``search_communities(query,k)``; ``adjacency``
    maps an entity title to relationship chunks loaded at wiring time.
    """

    def __init__(
        self,
        *,
        name,
        entity_search,
        community_search,
        adjacency,
        rrf_k=60,
        route_fn: Callable[[str], str] | None = None,
    ):
        self.name = name
        self._entities = entity_search
        self._communities = community_search
        self._adjacency = adjacency
        self._rrf_k = rrf_k
        self._route_fn = route_fn or classify_query

    async def retrieve(self, query: str, k: int, on_event=None) -> RetrievalResult:
        route = self._route_fn(query)
        emit(on_event, "retrieve.route", "complete", message=route, route=route)
        seeds = self._entities.search_entities(query, k)
        expanded: list[Chunk] = list(seeds)
        seen = {c.id for c in seeds}
        for seed in seeds:
            for rel in self._adjacency.get(seed.title, []):
                if rel.id not in seen:
                    expanded.append(rel)
                    seen.add(rel.id)
        if route == "global":
            glob = self._communities.search_communities(query, k)
            fused = reciprocal_rank_fusion(expanded, glob, k=self._rrf_k)
            stages = {"local": expanded, "global": glob, "fused": fused}
            return RetrievalResult(candidates=fused, stages=stages)
        stages = {"local": expanded, "fused": expanded}
        return RetrievalResult(candidates=expanded, stages=stages)
