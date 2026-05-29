from ragpipe.chunking import chunk_text


def test_chunk_respects_max_chars_with_overlap():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, max_chars=1000, overlap=100)

    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    # consecutive chunks overlap by ~100 chars
    assert chunks[0][-50:] in chunks[1]


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text("short", max_chars=1000, overlap=100)
    assert chunks == ["short"]


def test_chunk_empty_text_returns_empty():
    assert chunk_text("", max_chars=1000, overlap=100) == []
