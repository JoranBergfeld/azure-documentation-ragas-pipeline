from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever


class FakeTable:
    """Mimics ragpipe.pgstore.PgTable."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def nearest(self, vector, k):
        self.calls.append(("nearest", vector, k))
        return self._rows

    def bm25(self, query, k):
        self.calls.append(("bm25", query, k))
        return self._rows


def _row(cid, score):
    return {
        "id": cid,
        "title": f"title-{cid}",
        "url": f"http://{cid}",
        "content": f"content-{cid}",
        "score": score,
    }


def test_bm25_retriever_runs_a_lexical_query_and_maps_chunks():
    table = FakeTable([_row("a", 3.0), _row("b", 2.0)])
    retriever = BM25Retriever(table, top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a", "b"]
    assert chunks[0].score == 3.0
    assert chunks[0].url == "http://a"
    assert table.calls == [("bm25", "hybrid search", 5)]


def test_dense_retriever_runs_a_vector_query():
    table = FakeTable([_row("a", 0.9)])
    retriever = DenseRetriever(table, embed_fn=lambda text: [0.1, 0.2, 0.3], top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a"]
    assert table.calls == [("nearest", [0.1, 0.2, 0.3], 5)]


def test_retrievers_carry_the_context_for_the_reranker():
    row = {**_row("a", 1.0), "context": "situating text"}
    chunk = BM25Retriever(FakeTable([row]), top_k=1).retrieve("q")[0]
    assert chunk.context == "situating text"
    # Tables without one (graph rows) leave it empty.
    assert BM25Retriever(FakeTable([_row("b", 1.0)]), top_k=1).retrieve("q")[0].context == ""
