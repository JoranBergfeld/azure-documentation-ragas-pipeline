from __future__ import annotations

from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


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

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
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
