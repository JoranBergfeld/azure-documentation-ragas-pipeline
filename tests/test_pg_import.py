from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ragpipe.pg_import import ExportError, read_export
from ragpipe.pgstore import GRAPH_COMMUNITIES, GRAPH_RELATIONSHIPS


def _rel(i: int) -> dict:
    return {"id": f"rel-{i}", "source": "A", "target": "B", "description": "d",
            "weight": 1.0, "source_urls": []}


def _write_export(tmp_path: Path, rows: list[dict], **entry) -> Path:
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (tmp_path / "graph-relationships.jsonl").write_bytes(data)
    manifest = {"indexes": {"graph-relationships": {
        "expected": len(rows), "exported": len(rows), "complete": True,
        "file": "graph-relationships.jsonl", "sha256": hashlib.sha256(data).hexdigest(),
        **entry,
    }}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_reads_a_matching_export(tmp_path):
    export = _write_export(tmp_path, [_rel(1), _rel(2)])
    rows = read_export(export, (GRAPH_RELATIONSHIPS,))
    assert [r["id"] for r in rows["graph-relationships"]] == ["rel-1", "rel-2"]


def test_refuses_a_sha256_mismatch(tmp_path):
    export = _write_export(tmp_path, [_rel(1)], sha256="0" * 64)
    with pytest.raises(ExportError, match="sha256 mismatch"):
        read_export(export, (GRAPH_RELATIONSHIPS,))


def test_refuses_a_short_count(tmp_path):
    export = _write_export(tmp_path, [_rel(1)], expected=2, exported=2)
    with pytest.raises(ExportError, match="1 rows, manifest says 2"):
        read_export(export, (GRAPH_RELATIONSHIPS,))


def test_refuses_an_incomplete_export(tmp_path):
    export = _write_export(tmp_path, [_rel(1)], complete=False)
    with pytest.raises(ExportError, match="incomplete"):
        read_export(export, (GRAPH_RELATIONSHIPS,))


def test_refuses_duplicate_ids(tmp_path):
    export = _write_export(tmp_path, [_rel(1), _rel(1)])
    with pytest.raises(ExportError, match="duplicate ids"):
        read_export(export, (GRAPH_RELATIONSHIPS,))


def test_refuses_an_index_missing_from_the_manifest(tmp_path):
    export = _write_export(tmp_path, [_rel(1)])
    with pytest.raises(ExportError, match="graph-communities: not in the manifest"):
        read_export(export, (GRAPH_COMMUNITIES,))


def test_refuses_rows_missing_a_column(tmp_path):
    row = _rel(1)
    del row["weight"]
    export = _write_export(tmp_path, [row])
    with pytest.raises(ExportError, match="without weight"):
        read_export(export, (GRAPH_RELATIONSHIPS,))
