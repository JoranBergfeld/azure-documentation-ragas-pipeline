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


def build_index(name: str, vector_dimensions: int) -> SearchIndex:
    fields = [
        # filterable=True is required: the semantic reranker restricts to the
        # RRF-fused candidate set via a search.in(id, ...) $filter expression.
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME, algorithm_configuration_name="hnsw"
            )
        ],
    )
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
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


def create_index(client: SearchIndexClient, name: str, vector_dimensions: int) -> None:
    client.create_or_update_index(build_index(name, vector_dimensions))


def recreate_index(
    client: SearchIndexClient, name: str, vector_dimensions: int
) -> None:  # pragma: no cover - live Azure call
    """Delete the index if it exists, then create it fresh.

    create_or_update_index keeps existing documents, so re-ingesting a changed
    corpus would leave stale chunks behind (and double-count). A full reingest
    should start from an empty index, so we delete first.
    """
    try:
        client.delete_index(name)
    except Exception:  # noqa: BLE001 - absent index is fine; nothing to delete
        pass
    client.create_index(build_index(name, vector_dimensions))
