from ragpipe.ingest import build_documents, html_to_text


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><script>x=1</script></head><body><h1>Hi</h1><p>Body</p></body></html>"
    text = html_to_text(html)
    assert "Hi" in text
    assert "Body" in text
    assert "x=1" not in text


def test_build_documents_chunks_and_embeds():
    pages = [{"url": "http://x", "title": "T", "text": "word " * 600}]
    embed = lambda text: [0.0, 0.1]  # noqa: E731

    docs = build_documents(pages, embed_fn=embed, max_chars=1000, overlap=100)

    assert len(docs) >= 2
    first = docs[0]
    assert first["id"].startswith("http://x")  # url + chunk index
    assert first["title"] == "T"
    assert first["url"] == "http://x"
    assert first["content_vector"] == [0.0, 0.1]
    assert "content" in first
