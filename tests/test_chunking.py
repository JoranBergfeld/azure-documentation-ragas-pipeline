from ragpipe.chunking import chunk_text
from ragpipe.chunking import chunk_markdown


MD = """# Consistency levels

Intro paragraph.

## Configure

Some text about configuring.

```
# this hash is code, not a heading
az cosmosdb update
```

## Overview

| Level | Staleness |
| --- | --- |
| Strong | None |
"""


def test_chunk_markdown_splits_on_headings_with_breadcrumbs():
    chunks = chunk_markdown(MD, page_title="Cosmos DB docs", max_chars=2000, overlap=200)
    crumbs = [c.breadcrumb for c in chunks]
    assert "Cosmos DB docs > Consistency levels" in crumbs
    assert "Cosmos DB docs > Consistency levels > Configure" in crumbs
    assert "Cosmos DB docs > Consistency levels > Overview" in crumbs


def test_chunk_markdown_ignores_headings_inside_code_fences():
    chunks = chunk_markdown(MD, page_title="T", max_chars=2000, overlap=200)
    configure = next(c for c in chunks if c.breadcrumb.endswith("Configure"))
    assert "az cosmosdb update" in configure.text
    assert not any("this hash is code" in c.breadcrumb for c in chunks)


def test_chunk_markdown_size_bounds_long_sections():
    md = "# Title\n\n" + ("word " * 1000)
    chunks = chunk_markdown(md, page_title="P", max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.breadcrumb == "P > Title" for c in chunks)


def test_chunk_markdown_empty():
    assert chunk_markdown("", page_title="P") == []


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
