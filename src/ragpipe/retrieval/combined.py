from __future__ import annotations

from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


class CombinedSubstrate:
    """Fuse the candidate lists of several inner substrates by RRF. Inner stages
    are namespaced (`<inner.name>:<stage>`) and merged so the dashboard/eval can
    still see each leg."""

    def __init__(self, *, name, substrates, rrf_k=60):
        self.name = name
        self._substrates = substrates
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        stages: dict = {}
        candidate_lists = []
        for sub in self._substrates:
            result = await sub.retrieve(query, k)
            candidate_lists.append(result.candidates)
            for name, chunks in result.stages.items():
                stages[f"{sub.name}:{name}"] = chunks
        fused = candidate_lists[0] if candidate_lists else []
        for nxt in candidate_lists[1:]:
            fused = reciprocal_rank_fusion(fused, nxt, k=self._rrf_k)
        stages["fused"] = fused
        return RetrievalResult(candidates=fused, stages=stages)
