"""Retrieval SQL against a real Postgres with pgvector.

Skipped unless RAG_TEST_DATABASE_URL points at a throwaway database the test may
create and drop the ``rag`` schema in, e.g.:

    docker run -d --name ragpipe-pg-test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=rag \\
        -p 55432:5432 pgvector/pgvector:pg16
    RAG_TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/rag uv run pytest
"""
from __future__ import annotations

import os

import pytest

from ragpipe.pg_import import import_table
from ragpipe.pgstore import (
    EMBEDDING_DIMENSIONS,
    GRAPH_RELATIONSHIPS,
    MS_DOCS,
    SCHEMA,
    PgTable,
    create_table_sql,
)

URL = os.environ.get("RAG_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="RAG_TEST_DATABASE_URL not set")


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIMENSIONS
    v[i] = 1.0
    return v


def _chunk(i: int, content: str, url: str = "https://x/page") -> dict:
    return {"id": f"c{i}", "title": f"T{i}", "url": url, "chunk_id": i,
            "content": content, "context": ""}


DOCS = [
    _chunk(0, "Azure Functions runs event driven code on a consumption plan."),
    _chunk(1, "Durable Functions orchestrate stateful workflows in Azure Functions."),
    _chunk(2, "Cosmos DB is a globally distributed database. Cosmos DB scales."),
    _chunk(3, "Blob storage keeps unstructured data.", url="https://x/other"),
]


def _embed_by_index(texts: list[str]) -> list[list[float]]:
    # Each chunk's vector is the unit vector of its position in DOCS, found by
    # its content (MS_DOCS embeds "context\n\ncontent").
    by_content = {d["content"]: i for i, d in enumerate(DOCS)}
    return [_unit(by_content[t.split("\n\n", 1)[1]]) for t in texts]


@pytest.fixture
def conn():
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"CREATE SCHEMA {SCHEMA}")
        yield c
        c.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


@pytest.fixture
def pool(conn):
    from ragpipe.pgstore import open_pool

    p = open_pool(URL)
    yield p
    p.close()


def _quiet(_msg: str) -> None:
    pass


def test_import_is_rerunnable_and_never_doubles_rows(conn):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return _embed_by_index(texts)

    first = import_table(conn, MS_DOCS, DOCS, embed, log=_quiet)
    assert (first.rows, first.inserted, first.deleted) == (4, 4, 0)

    # Same export again: nothing loaded, nothing embedded.
    second = import_table(conn, MS_DOCS, DOCS, embed, log=_quiet)
    assert (second.rows, second.inserted, second.deleted) == (4, 0, 0)
    assert calls == [4]

    # A row the export no longer has goes; the table converges on the export.
    third = import_table(conn, MS_DOCS, DOCS[:3], embed, log=_quiet)
    assert (third.rows, third.inserted, third.deleted) == (3, 0, 1)


def test_replace_reloads_everything(conn):
    import_table(conn, MS_DOCS, DOCS, _embed_by_index, log=_quiet)
    again = import_table(conn, MS_DOCS, DOCS, _embed_by_index, replace=True, log=_quiet)
    assert (again.rows, again.inserted) == (4, 4)


def test_ddl_is_idempotent(conn):
    for _ in range(2):
        for stmt in create_table_sql(MS_DOCS):
            conn.execute(stmt)


def test_nearest_orders_by_cosine_similarity(conn, pool):
    import_table(conn, MS_DOCS, DOCS, _embed_by_index, log=_quiet)
    rows = PgTable(pool, MS_DOCS).nearest(_unit(2), k=2)
    assert rows[0]["id"] == "c2"
    assert rows[0]["score"] == pytest.approx(1.0)
    assert set(rows[0]) >= {"id", "title", "url", "content", "chunk_id"}


def test_bm25_prefers_the_rarer_term_and_skips_non_matches(conn, pool):
    import_table(conn, MS_DOCS, DOCS, _embed_by_index, log=_quiet)
    table = PgTable(pool, MS_DOCS)

    # "durable" is in one chunk, "functions" in two: the rare term decides.
    rows = table.bm25("durable functions", k=10)
    assert [r["id"] for r in rows][:2] == ["c1", "c0"]
    assert "c2" not in {r["id"] for r in rows}
    assert rows[0]["score"] > rows[1]["score"] > 0

    # Term frequency counts: Cosmos DB twice beats once (only one doc has it).
    assert [r["id"] for r in table.bm25("cosmos", k=10)] == ["c2"]
    # Stop words alone match nothing rather than everything.
    assert table.bm25("the and of", k=10) == []


def test_page_and_sample(conn, pool):
    import_table(conn, MS_DOCS, DOCS, _embed_by_index, log=_quiet)
    table = PgTable(pool, MS_DOCS)
    assert [r["chunk_id"] for r in table.page("https://x/page")] == [0, 1, 2]
    assert [r["id"] for r in table.sample(2)] == ["c0", "c1"]


def test_relationships_load_without_vectors_and_scan(conn, pool):
    rels = [{"id": "rel-1", "source": "A", "target": "B", "description": "a uses b",
             "weight": 9.0, "source_urls": ["https://x/a"]}]

    def no_embedding(texts):
        raise AssertionError("relationships aren't embedded")

    import_table(conn, GRAPH_RELATIONSHIPS, rels, no_embedding, log=_quiet)
    assert PgTable(pool, GRAPH_RELATIONSHIPS).scan() == [
        {"id": "rel-1", "source": "A", "target": "B", "description": "a uses b",
         "weight": 9.0, "source_urls": ["https://x/a"]}
    ]


def test_the_service_pool_cannot_write(conn, pool):
    import psycopg

    import_table(conn, MS_DOCS, DOCS, _embed_by_index, log=_quiet)
    with pool.connection() as c:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            c.execute(f"DELETE FROM {MS_DOCS.qualified}")
