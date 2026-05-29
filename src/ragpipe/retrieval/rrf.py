from __future__ import annotations

from dataclasses import replace

from ragpipe.models import Chunk


def reciprocal_rank_fusion(
    dense: list[Chunk], bm25: list[Chunk], k: int = 60
) -> list[Chunk]:
    """Merge two ranked lists by Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank), rank is 0-based.
    Returns a new list of Chunks sorted by fused score descending; the
    fused score is written to each returned Chunk's `score`.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked in (dense, bm25):
        for rank, chunk in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(chunk.id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(by_id[cid], score=scores[cid]) for cid in ordered_ids]
