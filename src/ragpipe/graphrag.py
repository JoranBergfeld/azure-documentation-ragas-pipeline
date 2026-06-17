from __future__ import annotations

import networkx as nx
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


@dataclass
class Community:
    id: int
    level: int
    title: str
    summary: str


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


def detect_communities(entity_names: list[str], relationships: list[Relationship], *, seed: int = 0) -> dict[str, int]:
    """Assign each entity a community id via networkx Louvain over the relationship
    graph. Isolated entities each get their own community."""
    g = nx.Graph()
    g.add_nodes_from(entity_names)
    for r in relationships:
        if r.source in g and r.target in g:
            g.add_edge(r.source, r.target, weight=r.weight)
    communities = nx.community.louvain_communities(g, seed=seed, weight="weight")
    mapping: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for name in members:
            mapping[name] = cid
    return mapping


def entity_documents(
    entities: list[Entity],
    *,
    community: dict[str, int],
    embed_batch_fn: object,
) -> list[dict]:
    """Shape Entity records into Azure AI Search doc dicts."""
    vecs = embed_batch_fn([e.description for e in entities])  # type: ignore[operator]
    return [
        {
            "id": f"entity-{i}",
            "name": e.name,
            "type": e.type,
            "description": e.description,
            "description_vector": vecs[i],
            "community_id": community.get(e.name, -1),
            "source_urls": e.source_urls,
        }
        for i, e in enumerate(entities)
    ]


def relationship_documents(
    relationships: list[Relationship],
    *,
    embed_batch_fn: object = None,
) -> list[dict]:
    """Shape Relationship records into Azure AI Search doc dicts."""
    vecs = embed_batch_fn([r.description for r in relationships]) if embed_batch_fn is not None else None  # type: ignore[operator]
    docs = []
    for i, r in enumerate(relationships):
        doc: dict = {
            "id": f"rel-{i}",
            "source": r.source,
            "target": r.target,
            "description": r.description,
            "weight": r.weight,
            "source_urls": r.source_urls,
        }
        if vecs is not None:
            doc["description_vector"] = vecs[i]
        docs.append(doc)
    return docs


def community_documents(
    communities: list[Community],
    *,
    embed_batch_fn: object,
) -> list[dict]:
    """Shape Community records into Azure AI Search doc dicts."""
    vecs = embed_batch_fn([c.summary for c in communities])  # type: ignore[operator]
    return [
        {
            "id": f"community-{c.id}",
            "level": c.level,
            "title": c.title,
            "summary": c.summary,
            "summary_vector": vecs[i],
        }
        for i, c in enumerate(communities)
    ]


