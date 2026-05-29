from ragpipe.models import Chunk
from ragpipe.retrieval.rerank import SemanticReranker


class FakeSearchClient:
    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def search(self, search_text=None, **kwargs):
        self.last_kwargs = {"search_text": search_text, **kwargs}
        return iter(self._results)


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=cid, score=0.1)


def _doc(cid, reranker_score):
    return {
        "id": cid,
        "title": cid,
        "url": f"http://{cid}",
        "content": cid,
        "@search.rerankerScore": reranker_score,
    }


def test_reranker_filters_to_fused_ids_and_orders_by_reranker_score():
    fused = [_chunk("a"), _chunk("b"), _chunk("c")]
    client = FakeSearchClient([_doc("b", 3.5), _doc("a", 2.0), _doc("c", 1.0)])
    reranker = SemanticReranker(client, semantic_config="default-semantic", top_k=3)

    out = reranker.rerank("query", fused)

    assert [c.id for c in out] == ["b", "a", "c"]
    assert out[0].score == 3.5
    # filter restricts to the fused IDs
    flt = client.last_kwargs["filter"]
    assert "a" in flt and "b" in flt and "c" in flt
    assert client.last_kwargs["query_type"] == "semantic"


def test_reranker_empty_input_returns_empty():
    client = FakeSearchClient([])
    reranker = SemanticReranker(client, semantic_config="default-semantic", top_k=3)
    assert reranker.rerank("q", []) == []
