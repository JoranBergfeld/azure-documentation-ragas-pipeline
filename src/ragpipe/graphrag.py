from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx


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


def merge_relationships(relationships: list[Relationship]) -> list[Relationship]:
    """Merge relationships sharing a (source, target) pair: concat unique
    descriptions, union source ids/urls, keep the max weight. The LLM extracts
    the same edge from many chunks, so without this the relationships index fills
    with near-duplicate edges that bloat embeddings and skew RRF/expansion."""
    by_pair: dict[tuple[str, str], Relationship] = {}
    for r in relationships:
        key = (r.source, r.target)
        cur = by_pair.get(key)
        if cur is None:
            by_pair[key] = Relationship(
                source=r.source, target=r.target, description=r.description, weight=r.weight,
                source_chunk_ids=list(r.source_chunk_ids), source_urls=list(r.source_urls),
            )
            continue
        descs = cur.description.split("\n")
        if r.description and r.description not in descs:
            cur.description = (cur.description + "\n" + r.description).strip()
        cur.weight = max(cur.weight, r.weight)
        for cid in r.source_chunk_ids:
            if cid not in cur.source_chunk_ids:
                cur.source_chunk_ids.append(cid)
        for u in r.source_urls:
            if u not in cur.source_urls:
                cur.source_urls.append(u)
    return list(by_pair.values())


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


# Persisted graph-state version. Bump to invalidate caches whose shape changed.
GRAPH_STATE_VERSION = "v1"


def save_graph_state(
    path: str | Path,
    *,
    limit: int | None,
    entities: list[Entity],
    relationships: list[Relationship],
    communities: list[Community],
    community_map: dict[str, int],
) -> None:
    """Persist the expensive LLM-derived graph state (entities, relationships,
    communities, entity->community map) so a re-run can skip the ~hour-long
    extraction + report generation and resume at the embed/upload tail.

    Written atomically via a uniquely-named temp file + os.replace (mirrors
    context_gen._save_cache) so a crash mid-write can't leave a partial cache.
    Keyed on `limit` so a `--limit N` smoke run never satisfies a full re-run.
    """
    payload = {
        "version": GRAPH_STATE_VERSION,
        "limit": limit,
        "entities": [asdict(e) for e in entities],
        "relationships": [asdict(r) for r in relationships],
        "communities": [asdict(c) for c in communities],
        "community_map": community_map,
    }
    path = Path(path)
    directory = path.parent if str(path.parent) else Path(".")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_graph_state(
    path: str | Path, *, limit: int | None
) -> tuple[list[Entity], list[Relationship], list[Community], dict[str, int]] | None:
    """Reload state written by save_graph_state. Returns None (never raises) when
    the cache is missing, corrupt, a different version, or built for a different
    `limit`, so the caller transparently falls back to a full rebuild."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != GRAPH_STATE_VERSION or payload.get("limit") != limit:
        return None
    try:
        entities = [Entity(**e) for e in payload["entities"]]
        relationships = [Relationship(**r) for r in payload["relationships"]]
        communities = [Community(**c) for c in payload["communities"]]
        community_map = {str(k): int(v) for k, v in payload["community_map"].items()}
    except (KeyError, TypeError, ValueError):
        return None
    return entities, relationships, communities, community_map


