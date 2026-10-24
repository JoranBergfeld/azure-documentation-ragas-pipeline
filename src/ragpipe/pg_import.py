"""Load a ragpipe-search export into Postgres (issue #37, ADR-0021).

    uv run python -m ragpipe.pg_import ~/ingest-artifacts/rag/2026-10-24

Runs from the laptop as the database owner (``RAG_IMPORT_DATABASE_URL``, the
``rag`` login), never from the container. It checks every export file against
``manifest.json`` before touching the database, re-embeds what the export left
out (vectors weren't retrievable), and loads all six tables.

Re-runnable: rows already in a table are kept and not re-embedded, rows the
export doesn't have are deleted, so a second run converges on the export and
never doubles anything. A run that dies halfway resumes where it stopped, having
paid for nothing twice. ``--replace`` drops the tables first, for an export that
changed rows under the same ids.

The export directory is read-only input. Nothing is written back into it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ragpipe.pgstore import TABLES, TableSpec, create_table_sql, fill_doc_len_sql, vector_literal

# text-embedding-3-small, USD per 1M tokens (Azure OpenAI, 2026-10). Only for the
# spend line at the end; the bill is what counts.
EMBEDDING_USD_PER_MTOK = 0.02


class ExportError(Exception):
    """The export doesn't match its manifest. Nothing has been written."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_export(export_dir: Path, specs: tuple[TableSpec, ...] = TABLES) -> dict[str, list[dict]]:
    """Every table's rows, after checking each file against the manifest.

    Refuses (ExportError) on a missing index or file, an incomplete export, a
    sha256 mismatch, a row count other than the manifest's, a duplicate id, or a
    row missing one of the table's columns.
    """
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ExportError(f"no manifest.json in {export_dir}")
    indexes = json.loads(manifest_path.read_text())["indexes"]

    out: dict[str, list[dict]] = {}
    for spec in specs:
        entry = indexes.get(spec.index)
        if entry is None:
            raise ExportError(f"{spec.index}: not in the manifest")
        if not entry.get("complete") or entry["exported"] != entry["expected"]:
            raise ExportError(
                f"{spec.index}: export incomplete "
                f"({entry['exported']} of {entry['expected']} documents)"
            )
        path = export_dir / entry["file"]
        if not path.is_file():
            raise ExportError(f"{spec.index}: {entry['file']} is missing")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ExportError(
                f"{spec.index}: sha256 mismatch on {entry['file']} "
                f"(manifest {entry['sha256'][:12]}…, file {actual[:12]}…)"
            )
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) != entry["expected"]:
            raise ExportError(f"{spec.index}: {len(rows)} rows, manifest says {entry['expected']}")
        if len({r["id"] for r in rows}) != len(rows):
            raise ExportError(f"{spec.index}: duplicate ids")
        missing = [c for c in spec.columns if any(c not in r for r in rows)]
        if missing:
            raise ExportError(f"{spec.index}: rows without {', '.join(missing)}")
        out[spec.index] = rows
    return out


@dataclass
class TableResult:
    table: str
    rows: int
    inserted: int
    deleted: int


def import_table(
    conn,
    spec: TableSpec,
    rows: list[dict[str, Any]],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    replace: bool = False,
    batch_size: int = 512,
    log: Callable[[str], None] = print,
) -> TableResult:
    """Make ``spec``'s table hold exactly ``rows``. ``conn`` is an autocommit
    psycopg connection; each batch commits on its own, so an interrupted import
    keeps (and doesn't re-embed) what it already loaded."""
    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {spec.qualified}")
    for stmt in create_table_sql(spec):
        conn.execute(stmt)

    wanted = {r["id"] for r in rows}
    present = {r[0] for r in conn.execute(f"SELECT id FROM {spec.qualified}")}
    stale = sorted(present - wanted)
    if stale:
        conn.execute(f"DELETE FROM {spec.qualified} WHERE id = ANY(%s)", (stale,))
    todo = [r for r in rows if r["id"] not in present]
    log(f"{spec.qualified}: {len(present & wanted)} already loaded, {len(todo)} to load, "
        f"{len(stale)} stale removed")

    columns = list(spec.columns)
    if spec.embed_input is not None:
        columns.append("embedding")
    copy_sql = f"COPY {spec.qualified} ({', '.join(columns)}) FROM STDIN"
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = (
            embed_batch([spec.embed_input(r) for r in batch])
            if spec.embed_input is not None
            else None
        )
        with conn.transaction():
            with conn.cursor().copy(copy_sql) as copy:
                for i, row in enumerate(batch):
                    values = [row[c] for c in spec.columns]
                    if vectors is not None:
                        values.append(vector_literal(vectors[i]))
                    copy.write_row(values)
        log(f"  {spec.qualified}: {min(start + batch_size, len(todo))}/{len(todo)}")

    if spec.searchable:
        conn.execute(fill_doc_len_sql(spec))
    conn.execute(f"ANALYZE {spec.qualified}")
    count = conn.execute(f"SELECT count(*) FROM {spec.qualified}").fetchone()[0]
    if count != len(rows):
        raise RuntimeError(f"{spec.qualified}: {count} rows after import, expected {len(rows)}")
    return TableResult(spec.table, count, len(todo), len(stale))


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - live DB + Azure
    import psycopg
    from dotenv import load_dotenv

    from ragpipe.config import Settings
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.pgstore import SCHEMA

    parser = argparse.ArgumentParser(prog="python -m ragpipe.pg_import")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument(
        "--replace", action="store_true", help="drop and reload every table"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    url = os.environ.get("RAG_IMPORT_DATABASE_URL")
    if not url:
        raise SystemExit("set RAG_IMPORT_DATABASE_URL to the owner login (rag)")

    try:
        export = read_export(args.export_dir.expanduser())
    except ExportError as err:
        raise SystemExit(f"refusing to import: {err}")
    print(f"export verified: {', '.join(f'{k} ({len(v)})' for k, v in export.items())}")

    tokens = 0

    def count(n: int) -> None:
        nonlocal tokens
        tokens += n

    embed_batch = build_batch_embed_fn(Settings.from_env(load=False), on_tokens=count)
    results = []
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        for spec in TABLES:
            results.append(
                import_table(conn, spec, export[spec.index], embed_batch, replace=args.replace)
            )

    for r in results:
        print(f"{r.table}: {r.rows} rows ({r.inserted} loaded, {r.deleted} removed)")
    usd = tokens / 1_000_000 * EMBEDDING_USD_PER_MTOK
    print(f"embedding spend: {tokens:,} tokens, about ${usd:.4f} at ${EMBEDDING_USD_PER_MTOK}/1M")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
