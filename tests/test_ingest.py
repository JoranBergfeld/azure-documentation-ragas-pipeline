import re

from ragpipe.ingest import build_documents


def test_build_documents_chunks_and_embeds():
    pages = [{"url": "http://x", "title": "T", "text": "word " * 600}]
    embed = lambda text: [0.0, 0.1]  # noqa: E731

    docs = build_documents(pages, embed_fn=embed, max_chars=1000, overlap=100)

    assert len(docs) >= 2
    first = docs[0]
    # id must be a valid Azure AI Search key: ^[A-Za-z0-9_\-=]+$
    assert re.fullmatch(r"[A-Za-z0-9_\-=]+", first["id"])
    # deterministic: same url+index always yields the same id
    again = build_documents(pages, embed_fn=embed, max_chars=1000, overlap=100)
    assert again[0]["id"] == first["id"]
    # distinct chunks get distinct ids
    assert docs[0]["id"] != docs[1]["id"]
    assert first["title"] == "T"
    assert first["url"] == "http://x"
    assert first["content_vector"] == [0.0, 0.1]
    assert "content" in first
