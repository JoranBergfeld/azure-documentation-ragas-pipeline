from __future__ import annotations

from ragpipe.ingest import build_baseline_documents


def test_baseline_documents_have_no_context_and_embed_raw_content():
    pages = [{"title": "T", "url": "http://x", "markdown": "# H\n\nbody text here"}]
    docs = build_baseline_documents(pages, embed_fn=lambda s: [float(len(s))])
    assert docs, "expected at least one chunk"
    d = docs[0]
    assert d.get("context", "") == ""              # no decoration
    assert d["content_vector"] == [float(len(d["content"]))]  # embeds raw content
    assert d["url"] == "http://x"
