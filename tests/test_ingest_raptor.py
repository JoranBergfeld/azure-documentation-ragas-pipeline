from __future__ import annotations

from ragpipe.ingest import raptor_summary_documents
from ragpipe.raptor import RaptorNode


def test_summary_documents_carry_level_and_embedding():
    nodes = [RaptorNode(id="raptor-L1-0", text="a summary", level=1, embedding=[0.5])]
    docs = raptor_summary_documents(nodes)
    d = docs[0]
    assert d["id"] == "raptor-L1-0"
    assert d["level"] == 1
    assert d["content"] == "a summary"
    assert d["content_vector"] == [0.5]
    assert d.get("context", "") == ""
    assert d["url"] == ""
