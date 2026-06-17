from __future__ import annotations

from ragpipe.graphrag import Entity, Relationship, detect_communities, merge_entities, parse_extraction


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


def test_merge_entities_unions_sources_and_descriptions():
    a = Entity("AZURE FUNCTIONS", "service", "Serverless compute", ["c1"], ["u1"])
    b = Entity("AZURE FUNCTIONS", "service", "Event-driven", ["c2"], ["u2"])
    c = Entity("BLOB STORAGE", "service", "Object storage", ["c1"], ["u1"])
    merged = merge_entities([a, b, c])
    by_name = {e.name: e for e in merged}
    assert set(by_name) == {"AZURE FUNCTIONS", "BLOB STORAGE"}
    fn = by_name["AZURE FUNCTIONS"]
    assert set(fn.source_chunk_ids) == {"c1", "c2"}
    assert "Serverless compute" in fn.description and "Event-driven" in fn.description


def test_detect_communities_groups_connected_entities():
    rels = [Relationship("A", "B", "", 1.0), Relationship("B", "C", "", 1.0), Relationship("X", "Y", "", 1.0)]
    comm = detect_communities(["A", "B", "C", "X", "Y"], rels, seed=0)
    assert comm["A"] == comm["B"] == comm["C"]
    assert comm["X"] == comm["Y"]
    assert comm["A"] != comm["X"]
