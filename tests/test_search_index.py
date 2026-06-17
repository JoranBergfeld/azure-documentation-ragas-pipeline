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


def test_include_context_false_omits_context_from_bm25_and_semantic():
    index = build_index("baseline", vector_dimensions=2, include_context=False)
    # context field is still present so docs with empty context upload without error
    names = {f.name for f in index.fields}
    assert "context" in names
    # but it must NOT be searchable (i.e. not wired into BM25)
    ctx_field = next(f for f in index.fields if f.name == "context")
    assert not getattr(ctx_field, "searchable", False)
    # and must NOT appear in the semantic config's content fields
    config = index.semantic_search.configurations[0]
    sem_content_names = [f.field_name for f in config.prioritized_fields.content_fields]
    assert "context" not in sem_content_names
    assert "content" in sem_content_names


def test_include_level_adds_filterable_level_field():
    from ragpipe.search_index import build_index
    idx = build_index("raptor-sac", 1536, include_context=True, include_level=True)
    fields = {f.name: f for f in idx.fields}
    assert "level" in fields
    assert fields["level"].filterable is True
    idx2 = build_index("baseline", 1536)
    assert "level" not in {f.name for f in idx2.fields}


def test_graph_index_builders_have_expected_fields():
    from ragpipe.search_index import (
        build_entities_index, build_relationships_index, build_communities_index,
    )
    ent = {f.name for f in build_entities_index("graph-entities", 1536).fields}
    assert {"id", "name", "type", "description", "description_vector", "community_id"} <= ent
    rel = {f.name for f in build_relationships_index("graph-relationships", 1536).fields}
    assert {"id", "source", "target", "description", "weight"} <= rel
    com = {f.name for f in build_communities_index("graph-communities", 1536).fields}
    assert {"id", "level", "title", "summary", "summary_vector"} <= com
