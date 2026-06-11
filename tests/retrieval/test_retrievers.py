from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever


class FakeSearchClient:
    """Mimics azure.search.documents.SearchClient.search()."""

    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def search(self, search_text=None, **kwargs):
        self.last_kwargs = {"search_text": search_text, **kwargs}
        return iter(self._results)


def _doc(cid, score):
    return {
        "id": cid,
        "title": f"title-{cid}",
        "url": f"http://{cid}",
        "content": f"content-{cid}",
        "@search.score": score,
    }


def test_bm25_retriever_uses_full_text_only_and_maps_chunks():
    client = FakeSearchClient([_doc("a", 3.0), _doc("b", 2.0)])
    retriever = BM25Retriever(client, top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a", "b"]
    assert chunks[0].score == 3.0
    # BM25 = full-text query, no vector_queries
    assert client.last_kwargs["search_text"] == "hybrid search"
    assert "vector_queries" not in client.last_kwargs


def test_dense_retriever_issues_vector_query():
    client = FakeSearchClient([_doc("a", 0.9)])
    embed = lambda text: [0.1, 0.2, 0.3]  # noqa: E731
    retriever = DenseRetriever(client, embed_fn=embed, top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a"]
    # dense = vector query, no full-text search_text
    assert client.last_kwargs["search_text"] is None
    assert client.last_kwargs["vector_queries"]  # non-empty
    # Regression: k kwarg (not k_nearest_neighbors) must be passed to VectorizedQuery
    assert client.last_kwargs["vector_queries"][0].k == 5
