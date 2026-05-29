from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=cid)


def test_rrf_merges_and_dedupes_by_id():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    bm25 = [_chunk("b"), _chunk("d")]

    fused = reciprocal_rank_fusion(dense, bm25, k=60)

    ids = [c.id for c in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    # 'b' appears in both lists near the top -> highest fused score -> first
    assert ids[0] == "b"


def test_rrf_score_formula():
    dense = [_chunk("a")]  # rank 0 -> 1/(60+1)
    bm25 = [_chunk("a")]   # rank 0 -> 1/(60+1)
    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    assert fused[0].id == "a"
    assert abs(fused[0].score - (2 / 61)) < 1e-9


def test_rrf_empty_inputs_returns_empty():
    assert reciprocal_rank_fusion([], [], k=60) == []
