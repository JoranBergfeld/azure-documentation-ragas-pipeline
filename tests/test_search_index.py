from ragpipe.search_index import build_index


def test_build_index_has_expected_fields():
    index = build_index("docs", vector_dimensions=1536)
    names = {f.name for f in index.fields}
    assert names == {"id", "title", "url", "chunk_id", "content", "content_vector"}


def test_build_index_key_field_is_filterable():
    # The semantic reranker restricts to the RRF candidate set via search.in(id, ...).
    index = build_index("docs", vector_dimensions=1536)
    key = next(f for f in index.fields if f.name == "id")
    assert key.key and key.filterable
