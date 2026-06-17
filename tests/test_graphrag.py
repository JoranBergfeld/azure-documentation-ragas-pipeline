from __future__ import annotations

from ragpipe.graphrag import parse_extraction


def test_parse_extraction_reads_entities_and_relationships():
    raw = (
        '("entity"<|>AZURE FUNCTIONS<|>service<|>Serverless compute)##'
        '("entity"<|>BLOB STORAGE<|>service<|>Object storage)##'
        '("relationship"<|>AZURE FUNCTIONS<|>BLOB STORAGE<|>Functions can be triggered by Blob events<|>8)'
    )
    entities, rels = parse_extraction(raw, source_chunk_id="c1", source_url="http://x")
    names = {e.name for e in entities}
    assert names == {"AZURE FUNCTIONS", "BLOB STORAGE"}
    assert entities[0].type == "service"
    assert entities[0].source_chunk_ids == ["c1"]
    assert entities[0].source_urls == ["http://x"]
    assert len(rels) == 1
    assert rels[0].source == "AZURE FUNCTIONS"
    assert rels[0].target == "BLOB STORAGE"
    assert rels[0].weight == 8.0


def test_parse_extraction_tolerates_garbage_records():
    entities, rels = parse_extraction("not a record##()##", source_chunk_id="c", source_url="u")
    assert entities == []
    assert rels == []
