from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval.query_class import QueryClass, classify_query
from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


def _hybrid(table, embed_fn, query: str, k: int, to_chunk: Callable[[dict], Chunk]) -> list[Chunk]:
    """BM25 + vector over one table, RRF-fused: what Azure did for a query that
    carried both search_text and a vector query."""
    dense = [to_chunk(r) for r in table.nearest(embed_fn(query), k)]
    lexical = [to_chunk(r) for r in table.bm25(query, k)]
    return reciprocal_rank_fusion(dense, lexical)[:k]


def _entity_chunk(hit: dict) -> Chunk:
    return Chunk(
        id=hit["id"],
        title=hit["name"],
        url=(hit.get("source_urls") or [""])[0],
        content=f'{hit["name"]} ({hit.get("type", "")}): {hit.get("description", "")}',
        score=float(hit.get("score", 0.0)),
    )


def _community_chunk(hit: dict) -> Chunk:
    return Chunk(
        id=hit["id"],
        title=hit.get("title", ""),
        url=(hit.get("source_urls") or [""])[0],
        content=hit.get("summary", ""),
        score=float(hit.get("score", 0.0)),
    )


class _EntitySearch:
    """Hybrid (BM25 + vector) search over the entities table."""

    def __init__(self, table, embed_fn: Callable[[str], list[float]], k: int) -> None:
        self._table = table
        self._embed = embed_fn
        self._k = k

    def search_entities(self, query: str, k: int) -> list[Chunk]:
        return _hybrid(self._table, self._embed, query, k, _entity_chunk)


class _CommunitySearch:
    """Hybrid (BM25 + vector) search over the communities table."""

    def __init__(self, table, embed_fn: Callable[[str], list[float]], k: int) -> None:
        self._table = table
        self._embed = embed_fn
        self._k = k

    def search_communities(self, query: str, k: int) -> list[Chunk]:
        return _hybrid(self._table, self._embed, query, k, _community_chunk)


def build_entity_search(table, embed_fn: Callable[[str], list[float]], k: int) -> _EntitySearch:
    """Build a live entity-search helper backed by the entities table."""
    return _EntitySearch(table, embed_fn, k)


def build_community_search(table, embed_fn: Callable[[str], list[float]], k: int) -> _CommunitySearch:
    """Build a live community-search helper backed by the communities table."""
    return _CommunitySearch(table, embed_fn, k)


def build_adjacency(table) -> dict[str, list[Chunk]]:
    """Read ALL relationship rows and build a name -> [Chunk] adjacency map.

    Keys are upper-cased entity names (matching entity .title). Each relationship
    row is registered under both its source and target so a seed entity can find
    all edges incident to it.
    """
    adjacency: dict[str, list[Chunk]] = {}
    for hit in table.scan():
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
    report ranking) search. entity_search/community_search expose
    search_entities(query,k)/search_communities(query,k) -> list[Chunk]. adjacency
    maps an entity title -> its relationship Chunks (built once from the
    relationships table at wiring time).

    The global leg is *routed*, not always fused (issue #8, ADR-0018): a
    deterministic query classifier engages community summaries only for
    sensemaking/breadth (GLOBAL) queries. Factoid/local queries retrieve from the
    local leg alone, so the empty-URL community summaries can no longer outrank
    and evict the precise leaf chunk under the reranker. Set ``routing=False`` to
    restore the legacy always-fuse behavior (e.g. for A/B evaluation)."""

    def __init__(
        self,
        *,
        name,
        entity_search,
        community_search,
        adjacency,
        rrf_k=60,
        routing=True,
        classify_fn: Callable[[str], QueryClass] | None = None,
    ):
        self.name = name
        self._entities = entity_search
        self._communities = community_search
        self._adjacency = adjacency
        self._rrf_k = rrf_k
        self._routing = routing
        self._classify = classify_fn or classify_query

    async def retrieve(self, query: str, k: int, on_event=None) -> RetrievalResult:
        seeds = self._entities.search_entities(query, k)
        expanded: list[Chunk] = list(seeds)
        seen = {c.id for c in seeds}
        for seed in seeds:
            for rel in self._adjacency.get(seed.title, []):
                if rel.id not in seen:
                    expanded.append(rel)
                    seen.add(rel.id)
        if not self._routing or self._classify(query) is QueryClass.GLOBAL:
            glob = self._communities.search_communities(query, k)
        else:
            # LOCAL query: skip the global community leg entirely so its
            # empty-URL summaries cannot evict the precise leaf chunk (issue #8).
            glob = []
        fused = reciprocal_rank_fusion(expanded, glob, k=self._rrf_k)
        return RetrievalResult(candidates=fused, stages={"local": expanded, "global": glob, "fused": fused})
