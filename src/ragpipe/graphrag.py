from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    source: str
    target: str
    description: str
    weight: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)


def _fields(record: str) -> list[str]:
    inner = record.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return [f.strip().strip('"') for f in inner.split("<|>")]


def parse_extraction(raw: str, *, source_chunk_id: str, source_url: str) -> tuple[list[Entity], list[Relationship]]:
    """Parse delimited LLM extraction output into entities + relationships.
    Records separated by '##', fields by '<|>'. Malformed records skipped."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    for record in raw.split("##"):
        parts = _fields(record)
        if len(parts) < 4:
            continue
        kind = parts[0].lower()
        if kind == "entity":
            name, etype, desc = parts[1], parts[2], parts[3]
            if not name:
                continue
            entities.append(Entity(name=name.upper(), type=etype, description=desc,
                                   source_chunk_ids=[source_chunk_id], source_urls=[source_url]))
        elif kind == "relationship" and len(parts) >= 5:
            src, tgt, desc, weight = parts[1], parts[2], parts[3], parts[4]
            try:
                w = float(weight)
            except ValueError:
                w = 1.0
            relationships.append(Relationship(source=src.upper(), target=tgt.upper(), description=desc,
                                              weight=w, source_chunk_ids=[source_chunk_id], source_urls=[source_url]))
    return entities, relationships


def merge_entities(entities: list[Entity]) -> list[Entity]:
    by_name: dict[str, Entity] = {}
    for e in entities:
        cur = by_name.get(e.name)
        if cur is None:
            by_name[e.name] = Entity(name=e.name, type=e.type, description=e.description,
                                     source_chunk_ids=list(e.source_chunk_ids), source_urls=list(e.source_urls))
            continue
        descs = cur.description.split("\n")
        if e.description and e.description not in descs:
            cur.description = (cur.description + "\n" + e.description).strip()
        for cid in e.source_chunk_ids:
            if cid not in cur.source_chunk_ids:
                cur.source_chunk_ids.append(cid)
        for u in e.source_urls:
            if u not in cur.source_urls:
                cur.source_urls.append(u)
    return list(by_name.values())


