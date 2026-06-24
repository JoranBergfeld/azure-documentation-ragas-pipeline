from __future__ import annotations

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

SEMANTIC_CONFIG_NAME = "default-semantic"
VECTOR_PROFILE_NAME = "default-vector"


def build_index(name: str, vector_dimensions: int, include_context: bool = True, include_level: bool = False) -> SearchIndex:
    """Build an Azure AI Search index schema.

    When ``include_context`` is True (default), the ``context`` field is
    searchable for BM25 and included in the semantic config's prioritized
    content fields -- matching the contextual-decoration pipeline (ADR-0003).
    When False, ``context`` is still present in the schema (so documents with
    an empty context field don't error on upload) but is NOT wired into BM25
    or semantic config -- the baseline plain-chunk index.
    """
    fields = [
        # filterable=True is required: the semantic reranker restricts to the
        # RRF-fused candidate set via a search.in(id, ...) $filter expression.
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32),
        SearchableField(name="content", type=SearchFieldDataType.String),
        # Retrieval-only decoration: breadcrumb + generated situating context
        # (ADR-0003). When include_context=True, searchable for BM25 and in the
        # semantic config. When False, the field is still defined so docs with an
        # empty context field upload without error, but it is not searched.
        SearchableField(name="context", type=SearchFieldDataType.String)
        if include_context
        else SimpleField(name="context", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]
    if include_level:
        fields.append(
            SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True, facetable=True)
        )
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME, algorithm_configuration_name="hnsw"
            )
        ],
    )
    content_fields = [SemanticField(field_name="content")]
    if include_context:
        content_fields.append(SemanticField(field_name="context"))
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=content_fields,
                ),
            )
        ]
    )
    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic,
    )


def _vector_search_config() -> VectorSearch:
    """Return a VectorSearch config with the shared HNSW profile.

    Each SearchIndex object needs its own instance (the SDK owns the object
    after you hand it over), but the profile name constant is reused across
    all indexes -- that's fine because profile names are scoped per-index.
    """
    return VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME, algorithm_configuration_name="hnsw"
            )
        ],
    )


def _vector_field(name: str, vector_dimensions: int) -> SearchField:
    return SearchField(
        name=name,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=vector_dimensions,
        vector_search_profile_name=VECTOR_PROFILE_NAME,
    )


def build_entities_index(name: str, vector_dimensions: int) -> SearchIndex:
    """Build the GraphRAG entities index schema."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="name", type=SearchFieldDataType.String),
        SimpleField(name="type", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="description", type=SearchFieldDataType.String),
        _vector_field("description_vector", vector_dimensions),
        SimpleField(name="community_id", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(
            name="source_urls",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        ),
    ]
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="name"),
                    content_fields=[SemanticField(field_name="description")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=semantic,
    )


def build_relationships_index(name: str, vector_dimensions: int) -> SearchIndex:
    """Build the GraphRAG relationships index schema."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="target", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="description", type=SearchFieldDataType.String),
        _vector_field("description_vector", vector_dimensions),
        SimpleField(name="weight", type=SearchFieldDataType.Double),
        SimpleField(
            name="source_urls",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        ),
    ]
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="description")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=semantic,
    )


def build_communities_index(name: str, vector_dimensions: int) -> SearchIndex:
    """Build the GraphRAG communities index schema."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="summary", type=SearchFieldDataType.String),
        _vector_field("summary_vector", vector_dimensions),
        SimpleField(
            name="source_urls",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        ),
    ]
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="summary")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=_vector_search_config(),
        semantic_search=semantic,
    )


def create_index(client: SearchIndexClient, name: str, vector_dimensions: int) -> None:
    """Create the index, or update its schema in place if it already exists.

    Kept idempotent (create_or_update) rather than delete-then-create: a Foundry
    knowledge source binds to this index, so the index cannot be deleted while
    that binding exists. Stale documents from a previous, larger corpus are
    removed separately by prune_stale_documents after upload.
    """
    client.create_or_update_index(build_index(name, vector_dimensions))
