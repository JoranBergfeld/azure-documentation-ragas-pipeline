from ragpipe.search_index import build_index


def test_build_index_has_expected_fields():
    index = build_index("docs", vector_dimensions=1536)
    names = {f.name for f in index.fields}
    assert names == {"id", "title", "url", "chunk_id", "content", "content_vector", "context"}


def test_build_index_key_field_is_filterable():
    # The semantic reranker restricts to the RRF candidate set via search.in(id, ...).
    index = build_index("docs", vector_dimensions=1536)
    key = next(f for f in index.fields if f.name == "id")
    assert key.key and key.filterable


def test_index_has_searchable_context_field():
    index = build_index("idx", vector_dimensions=2)
    ctx = next(f for f in index.fields if f.name == "context")
    assert ctx.searchable is True


def test_semantic_config_includes_context_after_content():
    index = build_index("idx", vector_dimensions=2)
    config = index.semantic_search.configurations[0]
    content_fields = [f.field_name for f in config.prioritized_fields.content_fields]
    assert content_fields == ["content", "context"]
