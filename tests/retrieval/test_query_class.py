from __future__ import annotations

import pytest

from ragpipe.retrieval.query_class import QueryClass, classify_query


@pytest.mark.parametrize(
    "query",
    [
        "how do functions read blobs",
        "what is the default timeout for an Azure Function",
        "where do I set the connection string",
        "which SKU supports availability zones",
        "create a storage account with the CLI",
    ],
)
def test_factoid_queries_classify_local(query):
    assert classify_query(query) is QueryClass.LOCAL


@pytest.mark.parametrize(
    "query",
    [
        "compare Azure Functions and Logic Apps",
        "what are the main differences between blob and file storage",
        "give an overview of Azure compute services",
        "summarize the networking options",
        "how do these services relate to each other",
        "what are the key themes across the storage docs",
        "Functions vs App Service",
        "pros and cons of serverless",
        "what types of triggers exist",
    ],
)
def test_sensemaking_queries_classify_global(query):
    assert classify_query(query) is QueryClass.GLOBAL


def test_classifier_is_case_insensitive():
    assert classify_query("COMPARE these two") is QueryClass.GLOBAL
    assert classify_query("Overview Of Compute") is QueryClass.GLOBAL


def test_empty_query_is_local():
    assert classify_query("") is QueryClass.LOCAL


def test_marker_requires_word_boundary():
    # "vs" must not match inside "vsphere"; this stays a factoid lookup.
    assert classify_query("how do I configure vsphere networking") is QueryClass.LOCAL
