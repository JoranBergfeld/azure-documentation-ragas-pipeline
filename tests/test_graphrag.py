from __future__ import annotations

from ragpipe.graphrag import Entity, Relationship, detect_communities, merge_entities, parse_extraction
from ragpipe.graphrag import (
    Community, community_documents, entity_documents, relationship_documents,
)


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


def test_merge_relationships_dedupes_by_pair():
    from ragpipe.graphrag import merge_relationships
    a = Relationship("A", "B", "triggers", 5.0, ["c1"], ["u1"])
    b = Relationship("A", "B", "also via events", 8.0, ["c2"], ["u2"])
    c = Relationship("A", "C", "reads", 3.0, ["c1"], ["u1"])
    merged = merge_relationships([a, b, c])
    pairs = {(r.source, r.target): r for r in merged}
    assert set(pairs) == {("A", "B"), ("A", "C")}
    ab = pairs[("A", "B")]
    assert ab.weight == 8.0  # max
    assert set(ab.source_chunk_ids) == {"c1", "c2"}
    assert "triggers" in ab.description and "also via events" in ab.description


def test_detect_communities_groups_connected_entities():
    rels = [Relationship("A", "B", "", 1.0), Relationship("B", "C", "", 1.0), Relationship("X", "Y", "", 1.0)]
    comm = detect_communities(["A", "B", "C", "X", "Y"], rels, seed=0)
    assert comm["A"] == comm["B"] == comm["C"]
    assert comm["X"] == comm["Y"]
    assert comm["A"] != comm["X"]


def test_doc_shapers():
    from ragpipe.graphrag import Entity, Relationship
    e = Entity("AZURE FUNCTIONS", "service", "Serverless", ["c1"], ["u1"])
    edocs = entity_documents([e], community={"AZURE FUNCTIONS": 3}, embed_batch_fn=lambda t: [[0.1]] * len(t))
    assert edocs[0]["name"] == "AZURE FUNCTIONS"
    assert edocs[0]["community_id"] == 3
    assert edocs[0]["description_vector"] == [0.1]
    assert edocs[0]["id"] == "entity-0"

    r = Relationship("A", "B", "rel desc", 5.0, ["c1"], ["u1"])
    rdocs = relationship_documents([r])
    assert rdocs[0]["source"] == "A" and rdocs[0]["weight"] == 5.0
    assert rdocs[0]["id"] == "rel-0"

    c = Community(id=2, level=0, title="Compute", summary="A summary")
    cdocs = community_documents([c], embed_batch_fn=lambda t: [[0.2]] * len(t))
    assert cdocs[0]["id"] == "community-2"
    assert cdocs[0]["summary_vector"] == [0.2]
