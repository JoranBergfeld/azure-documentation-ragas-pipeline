"""Retrieval on pgvector in homelab's Postgres (issue #37, ADR-0021).

One table per former Azure AI Search index, in the ``rag`` schema. Each table
keeps the index's fields as columns, plus:

- ``embedding vector(1536)``: what the index's ``*_vector`` field held, cosine
  k-NN through an HNSW index. ``graph_relationships`` has none: nothing queries
  its vectors, ``build_adjacency`` only reads the text.
- ``tsv``: a generated ``tsvector`` over the fields Azure marked searchable, and
  ``doc_len``, its total term count. Together they feed a hand-rolled BM25.
  ``ts_rank`` has no IDF, and the BM25 extension wants Postgres 17.

The service connects read-only (``default_transaction_read_only=on``) as a
reader login; only the import (``ragpipe.pg_import``) writes, as the owner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

SCHEMA = "rag"
EMBEDDING_DIMENSIONS = 1536
TEXT_SEARCH_CONFIG = "english"
# Azure AI Search's BM25 defaults, so the lexical leg is tuned like the old one.
BM25_K1 = 1.2
BM25_B = 0.75


@dataclass(frozen=True)
class TableSpec:
    """One former Azure index. ``columns`` follows the export's field order."""

    index: str
    table: str
    columns: dict[str, str]
    searchable: tuple[str, ...] = ()
    # Builds the embedding input from an exported row, the way ingest built it.
    # None: the table has no vector column.
    embed_input: Callable[[dict[str, Any]], str] | None = None

    @property
    def qualified(self) -> str:
        return f"{SCHEMA}.{self.table}"


_CHUNK_COLUMNS = {
    "id": "text PRIMARY KEY",
    "title": "text NOT NULL",
    "url": "text NOT NULL",
    "chunk_id": "integer NOT NULL",
    "content": "text NOT NULL",
    "context": "text NOT NULL",
}

MS_DOCS = TableSpec(
    index="ms-docs",
    table="ms_docs",
    columns=_CHUNK_COLUMNS,
    searchable=("title", "content", "context"),
    # ingest.py embed_inputs: the situating context, then the chunk.
    embed_input=lambda row: f"{row['context']}\n\n{row['content']}",
)
BASELINE = TableSpec(
    index="baseline",
    table="baseline",
    columns=_CHUNK_COLUMNS,
    searchable=("title", "content"),
    embed_input=lambda row: row["content"],
)
RAPTOR_SAC = TableSpec(
    index="raptor-sac",
    table="raptor_sac",
    columns={**_CHUNK_COLUMNS, "level": "integer NOT NULL"},
    searchable=("title", "content", "context"),
    embed_input=lambda row: row["content"],
)
GRAPH_ENTITIES = TableSpec(
    index="graph-entities",
    table="graph_entities",
    columns={
        "id": "text PRIMARY KEY",
        "name": "text NOT NULL",
        "type": "text NOT NULL",
        "description": "text NOT NULL",
        "community_id": "integer NOT NULL",
        "source_urls": "text[] NOT NULL",
    },
    searchable=("name", "description"),
    embed_input=lambda row: row["description"],
)
GRAPH_COMMUNITIES = TableSpec(
    index="graph-communities",
    table="graph_communities",
    columns={
        "id": "text PRIMARY KEY",
        "level": "integer NOT NULL",
        "title": "text NOT NULL",
        "summary": "text NOT NULL",
        "source_urls": "text[] NOT NULL",
    },
    searchable=("title", "summary"),
    embed_input=lambda row: row["summary"],
)
GRAPH_RELATIONSHIPS = TableSpec(
    index="graph-relationships",
    table="graph_relationships",
    columns={
        "id": "text PRIMARY KEY",
        "source": "text NOT NULL",
        "target": "text NOT NULL",
        "description": "text NOT NULL",
        "weight": "double precision NOT NULL",
        "source_urls": "text[] NOT NULL",
    },
)

TABLES: tuple[TableSpec, ...] = (
    MS_DOCS,
    BASELINE,
    RAPTOR_SAC,
    GRAPH_ENTITIES,
    GRAPH_COMMUNITIES,
    GRAPH_RELATIONSHIPS,
)


def create_table_sql(spec: TableSpec) -> list[str]:
    """DDL for one table and its indexes. Idempotent, safe to run on every import.

    The HNSW index goes on before the load: building it after is faster, but a
    resumed import would then pay for it twice. At 23k rows it doesn't matter.
    """
    cols = [f"{name} {sql_type}" for name, sql_type in spec.columns.items()]
    if spec.embed_input is not None:
        cols.append(f"embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL")
    if spec.searchable:
        joined = " || ' ' || ".join(f"coalesce({c}, '')" for c in spec.searchable)
        cols.append(
            f"tsv tsvector GENERATED ALWAYS AS "
            f"(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, {joined})) STORED"
        )
        # Filled after each load batch, see fill_doc_len_sql.
        cols.append("doc_len integer")
    stmts = [f"CREATE TABLE IF NOT EXISTS {spec.qualified} (\n  " + ",\n  ".join(cols) + "\n)"]
    if spec.embed_input is not None:
        stmts.append(
            f"CREATE INDEX IF NOT EXISTS {spec.table}_embedding_hnsw "
            f"ON {spec.qualified} USING hnsw (embedding vector_cosine_ops)"
        )
    if spec.searchable:
        stmts.append(
            f"CREATE INDEX IF NOT EXISTS {spec.table}_tsv_gin ON {spec.qualified} USING gin (tsv)"
        )
    return stmts


def fill_doc_len_sql(spec: TableSpec) -> str:
    """Set each row's BM25 document length: every position of every lexeme."""
    return (
        f"UPDATE {spec.qualified} d SET doc_len = "
        f"(SELECT coalesce(sum(cardinality(u.positions)), 0) FROM unnest(d.tsv) u) "
        f"WHERE d.doc_len IS NULL"
    )


def vector_literal(vector: list[float]) -> str:
    """pgvector's text form. Sent as text and cast, so no adapter is needed."""
    return json.dumps([float(x) for x in vector], separators=(",", ":"))


def _select_list(spec: TableSpec, alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{c}" for c in spec.columns)


def nearest_sql(spec: TableSpec) -> str:
    """k-NN by cosine distance. Score is cosine similarity, higher is better."""
    return (
        f"SELECT {_select_list(spec)}, 1 - (embedding <=> %(vector)s::vector) AS score "
        f"FROM {spec.qualified} "
        f"ORDER BY embedding <=> %(vector)s::vector "
        f"LIMIT %(k)s"
    )


def bm25_sql(spec: TableSpec) -> str:
    """Okapi BM25 over the table's tsvector, IDF over the whole table.

    Query terms are the query's own lexemes under the same text search config,
    OR'ed (Azure's simple query syntax with searchMode=any). Per-term document
    frequency comes from the GIN index, one count per term, so there is no
    statistics table to drift out of date. IDF is Lucene's
    ln(1 + (N - df + 0.5) / (df + 0.5)). A query with no lexemes (all stop
    words) matches nothing.

    Shaped for speed: each matching row's tsvector is cut down to the query's
    lexemes (setweight + ts_filter, both C) before unnesting, only ids and
    scores go through the sort, and the full rows are fetched for the top k
    only. A term in nearly every chunk ("azure") otherwise costs ~1s. ``idf`` is
    MATERIALIZED because Postgres inlines a CTE used once, which would rerun the
    document-frequency counts for every row.
    """
    t = spec.qualified
    return f"""
WITH terms AS (
  SELECT DISTINCT lexeme FROM unnest(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, %(query)s))
),
q AS (
  SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq,
         array_agg(lexeme) AS lexemes
  FROM terms
),
corpus AS (
  SELECT count(*)::float8 AS n, avg(doc_len)::float8 AS avgdl FROM {t}
),
idf AS MATERIALIZED (
  SELECT terms.lexeme,
         ln(1 + (corpus.n - df.df + 0.5) / (df.df + 0.5)) AS idf
  FROM terms
  CROSS JOIN corpus
  CROSS JOIN LATERAL (
    SELECT count(*)::float8 AS df FROM {t} x WHERE x.tsv @@ quote_literal(terms.lexeme)::tsquery
  ) df
),
scored AS (
  SELECT d.id,
         (SELECT sum(
                   idf.idf * (cardinality(u.positions) * ({BM25_K1} + 1))
                   / (cardinality(u.positions)
                      + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * d.doc_len / nullif(corpus.avgdl, 0)))
                 )
          FROM unnest(ts_filter(setweight(d.tsv, 'A', q.lexemes), '{{a}}')) u
          JOIN idf ON idf.lexeme = u.lexeme) AS score
  FROM {t} d
  CROSS JOIN q
  CROSS JOIN corpus
  WHERE d.tsv @@ q.tsq
  ORDER BY score DESC, d.id
  LIMIT %(k)s
)
SELECT {_select_list(spec, "d")}, scored.score
FROM scored
JOIN {t} d ON d.id = scored.id
ORDER BY scored.score DESC, d.id
"""


class PgTable:
    """Read access to one table through a connection pool."""

    def __init__(self, pool, spec: TableSpec) -> None:
        self._pool = pool
        self.spec = spec

    def _fetch(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                names = [d.name for d in cur.description]
                return [dict(zip(names, row)) for row in cur.fetchall()]

    def nearest(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        return self._fetch(nearest_sql(self.spec), {"vector": vector_literal(vector), "k": k})

    def bm25(self, query: str, k: int) -> list[dict[str, Any]]:
        return self._fetch(bm25_sql(self.spec), {"query": query, "k": k})

    def scan(self) -> list[dict[str, Any]]:
        return self._fetch(f"SELECT {_select_list(self.spec)} FROM {self.spec.qualified}")

    def sample(self, limit: int) -> list[dict[str, Any]]:
        """The first ``limit`` rows by id: a stable sample, not a random one."""
        return self._fetch(
            f"SELECT {_select_list(self.spec)} FROM {self.spec.qualified} ORDER BY id LIMIT %(k)s",
            {"k": limit},
        )

    def page(self, url: str) -> list[dict[str, Any]]:
        """Every chunk of one page, in reading order."""
        return self._fetch(
            f"SELECT {_select_list(self.spec)} FROM {self.spec.qualified} "
            f"WHERE url = %(url)s ORDER BY chunk_id",
            {"url": url},
        )


def open_pool(database_url: str):  # pragma: no cover - needs a database
    """A small read-only pool. Every session starts with
    default_transaction_read_only=on, like the spatial layer's store, so even a
    login that could write can't from here."""
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=4,
        kwargs={"options": "-c default_transaction_read_only=on"},
        open=True,
    )
