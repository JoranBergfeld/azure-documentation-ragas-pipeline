import re

from ragpipe.ingest import build_documents, html_to_text


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><script>x=1</script></head><body><h1>Hi</h1><p>Body</p></body></html>"
    text = html_to_text(html)
    assert "Hi" in text
    assert "Body" in text
    assert "x=1" not in text


def _fake_batch_embed(texts):
    # one deterministic 2-d vector per input, in order
    return [[float(len(t) % 7), 0.1] for t in texts]


def test_build_documents_chunks_and_embeds():
    pages = [{"url": "http://x", "title": "T", "text": "word " * 600}]

    docs = build_documents(
        pages, embed_batch_fn=_fake_batch_embed, max_chars=1000, overlap=100, batch_size=2
    )

    assert len(docs) >= 2
    first = docs[0]
    # id must be a valid Azure AI Search key: ^[A-Za-z0-9_\-=]+$
    assert re.fullmatch(r"[A-Za-z0-9_\-=]+", first["id"])
    # deterministic: same url+index always yields the same id
    again = build_documents(
        pages, embed_batch_fn=_fake_batch_embed, max_chars=1000, overlap=100, batch_size=2
    )
    assert again[0]["id"] == first["id"]
    # distinct chunks get distinct ids
    assert docs[0]["id"] != docs[1]["id"]
    assert first["title"] == "T"
    assert first["url"] == "http://x"
    assert "content" in first
    assert len(first["content_vector"]) == 2


def test_build_documents_preserves_chunk_order_across_batches():
    # 5 chunks with batch_size 2 spans 3 batches; vectors must line up with chunks.
    pages = [{"url": "http://x", "title": "T", "text": "word " * 1500}]
    seen = []

    def batch(texts):
        seen.extend(texts)
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    docs = build_documents(
        pages, embed_batch_fn=batch, max_chars=500, overlap=50, batch_size=2
    )
    # every chunk's content was embedded exactly once, in order
    assert [d["content"] for d in docs] == seen
    assert len(docs) >= 3
