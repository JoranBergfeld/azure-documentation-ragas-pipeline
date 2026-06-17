# Multi-substrate retrieval — Phase 3 (GraphRAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** Add the GraphRAG retrieval mode: a flat materialized graph (entities, relationships, community reports) built at ingest into three Azure AI Search indexes, retrieved by combining local search (entity match + 1-hop expansion) and global search (community-report ranking).

**Architecture:** The expensive graph work happens once at build time in Python (extraction, community detection, community reports). At query time the `GraphRAGSubstrate` does two AAS searches — global (rank community reports) and local (match entities, expand 1 hop via an in-memory adjacency built from the relationships index, gather entity/relationship descriptions) — and fuses them with RRF into `Chunk` candidates. Graph candidates span three indexes and are not rerankable by Azure's id-filtered semantic ranker, so graph mode uses a `PassthroughReranker` (sort by score, take top-k) instead of `SemanticReranker`.

**Tech Stack:** Python 3.12, `uv`, pytest/pytest-asyncio, networkx (already installed, used for community detection via `louvain_communities`), Azure AI Search. The seam, registry, config index names (`graph_entities_index`, `graph_relationships_index`, `graph_communities_index`), and ingest helpers exist on `main`.

**Conventions:** `uv run`; ruff clean; `from __future__ import annotations`; commit per task; branch `feat/multi-substrate-phase3`. No live Azure/LLM in tests — inject fakes. Live drivers `# pragma: no cover`.

**Data model (Chunk representation of graph artifacts):**
- Entity → `Chunk(id=f"entity:{name}", title=name, url=<first source url>, content=f"{name} ({type}): {description}")`.
- Relationship → `Chunk(id=f"rel:{source}->{target}", title=f"{source} — {target}", url=<first source url>, content=description)`.
- Community report → `Chunk(id=f"community:{cid}", title=title, url="", content=summary)`.
Giving entities/relationships a source `url` keeps the deterministic URL-match metric (ADR-0002) partially meaningful for graph mode.

---

## File structure (Phase 3)

- Create `src/ragpipe/graphrag.py` — extraction parsing, entity merge, community detection, in-memory adjacency + local/global search helpers (all pure / dependency-injected).
- Create `tests/test_graphrag.py`.
- Create `src/ragpipe/retrieval/graph_substrate.py` — `GraphRAGSubstrate`.
- Create `tests/retrieval/test_graph_substrate.py`.
- Modify `src/ragpipe/retrieval/rerank.py` (or a new `passthrough.py`) — `PassthroughReranker`.
- Modify `src/ragpipe/search_index.py` — `build_entities_index`, `build_relationships_index`, `build_communities_index`.
- Modify `src/ragpipe/ingest.py` — `build_graph(settings, limit)` live driver + pure doc shapers.
- Modify `src/ragpipe/retrieval/registry.py` + `src/ragpipe/app_wiring.py` — register GRAPHRAG, select PassthroughReranker for it.
- Modify tests for registry/search_index.

---

## Task 1: extraction parsing (`graphrag.py`)

The LLM extracts entities and relationships per chunk in a delimited format (GraphRAG-style). This task parses that text into typed records. Parsing is pure and fully testable; the LLM call itself is in the live driver.

**Files:** Create `src/ragpipe/graphrag.py`, `tests/test_graphrag.py`.

- [ ] **Step 1: failing test**

```python
# tests/test_graphrag.py
from __future__ import annotations

from ragpipe.graphrag import Entity, Relationship, parse_extraction


def test_parse_extraction_reads_entities_and_relationships():
    # delimited format: records separated by ##, fields by <|>
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
```

- [ ] **Step 2: confirm fail** — `uv run pytest tests/test_graphrag.py -v`.

- [ ] **Step 3: implement** in `src/ragpipe/graphrag.py`:

```python
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


def parse_extraction(
    raw: str, *, source_chunk_id: str, source_url: str
) -> tuple[list[Entity], list[Relationship]]:
    """Parse delimited LLM extraction output into entities + relationships.
    Records are separated by '##', fields by '<|>'. Malformed records are
    skipped (the LLM occasionally emits noise)."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    for record in raw.split("##"):
        parts = _fields(record)
        if len(parts) < 4:
            continue
        kind = parts[0].lower()
        if kind == "entity":
            _, name, etype, desc = parts[0], parts[1], parts[2], parts[3]
            if not name:
                continue
            entities.append(
                Entity(
                    name=name.upper(),
                    type=etype,
                    description=desc,
                    source_chunk_ids=[source_chunk_id],
                    source_urls=[source_url],
                )
            )
        elif kind == "relationship" and len(parts) >= 5:
            _, src, tgt, desc, weight = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                w = float(weight)
            except ValueError:
                w = 1.0
            relationships.append(
                Relationship(
                    source=src.upper(),
                    target=tgt.upper(),
                    description=desc,
                    weight=w,
                    source_chunk_ids=[source_chunk_id],
                    source_urls=[source_url],
                )
            )
    return entities, relationships
```

- [ ] **Step 4: confirm pass.** ruff clean.
- [ ] **Step 5: commit** `feat(graphrag): parse delimited entity/relationship extraction`.

---

## Task 2: entity merge (`graphrag.py`)

Duplicate entities (same normalized name across chunks) merge into one, unioning descriptions and source refs.

- [ ] **Step 1: failing test** (append)

```python
from ragpipe.graphrag import merge_entities


def test_merge_entities_unions_sources_and_descriptions():
    from ragpipe.graphrag import Entity
    a = Entity("AZURE FUNCTIONS", "service", "Serverless compute", ["c1"], ["u1"])
    b = Entity("AZURE FUNCTIONS", "service", "Event-driven", ["c2"], ["u2"])
    c = Entity("BLOB STORAGE", "service", "Object storage", ["c1"], ["u1"])
    merged = merge_entities([a, b, c])
    by_name = {e.name: e for e in merged}
    assert set(by_name) == {"AZURE FUNCTIONS", "BLOB STORAGE"}
    fn = by_name["AZURE FUNCTIONS"]
    assert set(fn.source_chunk_ids) == {"c1", "c2"}
    assert "Serverless compute" in fn.description and "Event-driven" in fn.description
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement**

```python
def merge_entities(entities: list[Entity]) -> list[Entity]:
    """Merge entities sharing a name (already upper-cased by parse). Descriptions
    are concatenated (deduped, newline-joined); source ids/urls unioned; type taken
    from the first occurrence."""
    by_name: dict[str, Entity] = {}
    for e in entities:
        cur = by_name.get(e.name)
        if cur is None:
            by_name[e.name] = Entity(
                name=e.name, type=e.type, description=e.description,
                source_chunk_ids=list(e.source_chunk_ids), source_urls=list(e.source_urls),
            )
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
```

- [ ] **Step 4: pass + ruff.** **Step 5: commit** `feat(graphrag): merge duplicate entities`.

---

## Task 3: community detection (`graphrag.py`)

Wrap networkx Louvain to assign each entity a community id from the relationship edges.

- [ ] **Step 1: failing test** (append)

```python
from ragpipe.graphrag import detect_communities


def test_detect_communities_groups_connected_entities():
    from ragpipe.graphrag import Relationship
    rels = [
        Relationship("A", "B", "", 1.0), Relationship("B", "C", "", 1.0),
        Relationship("X", "Y", "", 1.0),
    ]
    comm = detect_communities(["A", "B", "C", "X", "Y"], rels, seed=0)
    # A,B,C in one community; X,Y in another; the two differ
    assert comm["A"] == comm["B"] == comm["C"]
    assert comm["X"] == comm["Y"]
    assert comm["A"] != comm["X"]
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement**

```python
import networkx as nx


def detect_communities(
    entity_names: list[str], relationships: list[Relationship], *, seed: int = 0
) -> dict[str, int]:
    """Assign each entity a community id via networkx Louvain over the relationship
    graph. Isolated entities (no edges) each get their own community."""
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
```

- [ ] **Step 4: pass + ruff.** (If Louvain ordering is nondeterministic across runs even with seed, the test only checks co-membership and inequality, which is stable.) **Step 5: commit** `feat(graphrag): Louvain community detection`.

---

## Task 4: graph index schemas (`search_index.py`)

- [ ] **Step 1: failing test** (append to tests/test_search_index.py)

```python
def test_graph_index_builders_have_expected_fields():
    from ragpipe.search_index import (
        build_entities_index, build_relationships_index, build_communities_index,
    )
    ent = {f.name for f in build_entities_index("graph-entities", 1536).fields}
    assert {"id", "name", "type", "description", "description_vector", "community_id"} <= ent
    rel = {f.name for f in build_relationships_index("graph-relationships", 1536).fields}
    assert {"id", "source", "target", "description", "weight"} <= rel
    com = {f.name for f in build_communities_index("graph-communities", 1536).fields}
    assert {"id", "level", "title", "summary", "summary_vector"} <= com
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement** three builders in search_index.py mirroring the existing `build_index` structure (same HNSW vector profile constant `VECTOR_PROFILE_NAME`, a vector field per index, a semantic config so hybrid search works). Use the field classes already imported (`SimpleField`, `SearchableField`, `SearchField`, `SearchFieldDataType`). Entities: id(key), name(searchable), type(filterable), description(searchable), description_vector(vector dims), community_id(Int32 filterable), source_urls(Collection(String)). Relationships: id(key), source(searchable/filterable), target(searchable/filterable), description(searchable), weight(Double), source_urls(Collection(String)). Communities: id(key), level(Int32 filterable), title(searchable), summary(searchable), summary_vector(vector dims). Give each a semantic config (reuse `SEMANTIC_CONFIG_NAME`) prioritizing the description/summary field.
- [ ] **Step 4: pass + ruff.** **Step 5: commit** `feat(index): entities/relationships/communities index schemas`.

---

## Task 5: graph doc shapers (`graphrag.py`)

Pure functions mapping Entity/Relationship/community records to index doc dicts. Embeddings injected.

- [ ] **Step 1: failing test** (append to tests/test_graphrag.py)

```python
from ragpipe.graphrag import entity_documents, relationship_documents, community_documents, Community


def test_doc_shapers():
    from ragpipe.graphrag import Entity, Relationship
    e = Entity("AZURE FUNCTIONS", "service", "Serverless", ["c1"], ["u1"])
    edocs = entity_documents([e], community={"AZURE FUNCTIONS": 3}, embed_batch_fn=lambda t: [[0.1]] * len(t))
    assert edocs[0]["name"] == "AZURE FUNCTIONS"
    assert edocs[0]["community_id"] == 3
    assert edocs[0]["description_vector"] == [0.1]

    r = Relationship("A", "B", "rel desc", 5.0, ["c1"], ["u1"])
    rdocs = relationship_documents([r])
    assert rdocs[0]["source"] == "A" and rdocs[0]["weight"] == 5.0

    c = Community(id=2, level=0, title="Compute", summary="A summary")
    cdocs = community_documents([c], embed_batch_fn=lambda t: [[0.2]] * len(t))
    assert cdocs[0]["id"] == "community-2"
    assert cdocs[0]["summary_vector"] == [0.2]
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement** a `@dataclass Community(id:int, level:int, title:str, summary:str)` plus the three shapers. Entity doc id must be an AAS-safe key (e.g. `f"entity-{i}"` or a slug of the name — names contain spaces, which are NOT valid in AAS keys, so slug: keep only alnum/dash/underscore, or just index). Use `f"entity-{i}"` with `name` stored separately. Relationship id `f"rel-{i}"`. Community id `f"community-{c.id}"`. Embed entity descriptions and community summaries via `embed_batch_fn`. Store `source_urls` as a list.
- [ ] **Step 4: pass + ruff.** **Step 5: commit** `feat(graphrag): entity/relationship/community doc shapers`.

---

## Task 6: PassthroughReranker (`retrieval/passthrough.py`)

- [ ] **Step 1: failing test** — create tests/retrieval/test_passthrough.py

```python
from __future__ import annotations

from ragpipe.models import Chunk
from ragpipe.retrieval.passthrough import PassthroughReranker


def test_passthrough_sorts_by_score_and_truncates():
    chunks = [Chunk(id="a", title="", url="", content="", score=0.2),
              Chunk(id="b", title="", url="", content="", score=0.9),
              Chunk(id="c", title="", url="", content="", score=0.5)]
    out = PassthroughReranker(top_k=2).rerank("q", chunks, top_k=2)
    assert [c.id for c in out] == ["b", "c"]
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement**

```python
from __future__ import annotations

from ragpipe.models import Chunk


class PassthroughReranker:
    """Rerank by existing candidate score, no external call. Used for substrates
    whose candidates span multiple indexes and can't use Azure's id-filtered
    semantic ranker (e.g. GraphRAG)."""

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def rerank(self, query: str, fused: list[Chunk], top_k: int | None = None) -> list[Chunk]:
        k = top_k or self._top_k
        return sorted(fused, key=lambda c: c.score, reverse=True)[:k]
```

- [ ] **Step 4: pass + ruff.** **Step 5: commit** `feat(retrieval): PassthroughReranker for multi-index substrates`.

---

## Task 7: GraphRAGSubstrate (`retrieval/graph_substrate.py`)

Local + global search fused into Chunk candidates. Depends on injected search clients (entities/communities) and an adjacency map.

- [ ] **Step 1: failing test** — create tests/retrieval/test_graph_substrate.py

```python
from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.graph_substrate import GraphRAGSubstrate


class _FakeEntitySearch:
    def search_entities(self, query, k):
        return [Chunk(id="entity-0", title="AZURE FUNCTIONS", url="u1",
                      content="AZURE FUNCTIONS (service): Serverless", score=0.9)]


class _FakeCommunitySearch:
    def search_communities(self, query, k):
        return [Chunk(id="community-1", title="Compute", url="",
                      content="Compute services summary", score=0.8)]


@pytest.mark.asyncio
async def test_graph_substrate_fuses_local_and_global():
    sub = GraphRAGSubstrate(
        name="graphrag",
        entity_search=_FakeEntitySearch(),
        community_search=_FakeCommunitySearch(),
        adjacency={"AZURE FUNCTIONS": [Chunk(id="rel-0", title="AZURE FUNCTIONS — BLOB STORAGE",
                                             url="u1", content="triggered by blob", score=0.5)]},
    )
    result = await sub.retrieve("how do functions read blobs", k=10)
    assert sub.name == "graphrag"
    assert set(result.stages) == {"local", "global", "fused"}
    ids = {c.id for c in result.candidates}
    assert "entity-0" in ids and "community-1" in ids
    # 1-hop expansion pulled the relationship of the matched entity
    assert "rel-0" in ids
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement** `src/ragpipe/retrieval/graph_substrate.py`:

```python
from __future__ import annotations

from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


class GraphRAGSubstrate:
    """Local (entity match + 1-hop relationship expansion) + global (community
    report ranking) search, fused by RRF. entity_search/community_search expose
    search_entities(query,k)/search_communities(query,k) -> list[Chunk]. adjacency
    maps an entity title -> its relationship Chunks (built once from the
    relationships index at wiring time)."""

    def __init__(self, *, name, entity_search, community_search, adjacency, rrf_k=60):
        self.name = name
        self._entities = entity_search
        self._communities = community_search
        self._adjacency = adjacency
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        seeds = self._entities.search_entities(query, k)
        expanded: list[Chunk] = list(seeds)
        seen = {c.id for c in seeds}
        for seed in seeds:
            for rel in self._adjacency.get(seed.title, []):
                if rel.id not in seen:
                    expanded.append(rel)
                    seen.add(rel.id)
        glob = self._communities.search_communities(query, k)
        fused = reciprocal_rank_fusion(expanded, glob, k=self._rrf_k)
        return RetrievalResult(
            candidates=fused,
            stages={"local": expanded, "global": glob, "fused": fused},
        )
```

- [ ] **Step 4: pass + ruff.** **Step 5: commit** `feat(retrieval): GraphRAGSubstrate (local+global fused)`.

---

## Task 8: build_graph driver + register GRAPHRAG mode

**Files:** Modify `src/ragpipe/ingest.py`, `src/ragpipe/retrieval/registry.py`, `src/ragpipe/app_wiring.py`, `tests/retrieval/test_registry.py`.

- [ ] **Step 1: failing test** (append to test_registry.py) — registry must build a GraphRAGSubstrate:

```python
def test_graphrag_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.GRAPHRAG in registered_modes()
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3a: registry.** GraphRAG needs a different factory than `_hybrid` (it builds a GraphRAGSubstrate from the entity+community clients and an adjacency map). Add a `_graphrag` factory:

```python
def _graphrag(settings, ctx):
    from ragpipe.retrieval.graph_substrate import GraphRAGSubstrate, build_entity_search, build_community_search, build_adjacency
    ent = build_entity_search(ctx.search_client(settings.graph_entities_index), ctx.embed, settings.candidate_pool)
    com = build_community_search(ctx.search_client(settings.graph_communities_index), ctx.embed, settings.candidate_pool)
    adjacency = build_adjacency(ctx.search_client(settings.graph_relationships_index))
    return GraphRAGSubstrate(name="graphrag", entity_search=ent, community_search=com, adjacency=adjacency, rrf_k=settings.rrf_k)

_REGISTRY[RetrievalMode.GRAPHRAG] = _graphrag
```

Implement `build_entity_search`/`build_community_search` (small adapter objects doing hybrid vector+text search over the entity/community index, returning Chunks via a `_to_chunk`-style mapper that maps name/title→title, description/summary→content, source_urls[0]→url) and `build_adjacency` (read all relationships from the index, group relationship Chunks by both endpoint titles into `{title: [Chunk,...]}`) in `graph_substrate.py`. These are live (`# pragma: no cover`) since they hit Azure, EXCEPT keep the pure GraphRAGSubstrate testable as in Task 7.

- [ ] **Step 3b: app_wiring reranker selection.** In `build_pipeline_fn`, after building the substrate, choose the reranker by substrate name:

```python
    if substrate.name == "graphrag":
        from ragpipe.retrieval.passthrough import PassthroughReranker
        reranker = PassthroughReranker(settings.top_k)
    else:
        # existing SemanticReranker path (contextual/baseline/raptor_sac)
        ...
```

Keep the existing `rerank_index_attr` map for the non-graph substrates.

- [ ] **Step 3c: build_graph driver** (`# pragma: no cover`) in ingest.py: fetch pages → build SAC leaf docs (or read leaves) → for each chunk, LLM extraction (reuse the chat client; prompt asks for the delimited entity/relationship format parse_extraction expects) → `parse_extraction` per chunk → concat → `merge_entities` → `detect_communities` → community reports (LLM summary per community over its entities/relationships) → `entity_documents`/`relationship_documents`/`community_documents` (embed via build_batch_embed_fn) → create the three indexes (build_entities_index/build_relationships_index/build_communities_index) → upload each. Mirror build_baseline's fetch/create/upload structure. Note any shortcut as DONE_WITH_CONCERNS.

- [ ] **Step 4: full suite** `uv run pytest tests/ -q` green; `uv run ruff check .` clean. `uv run python -c "import ragpipe.ingest, ragpipe.retrieval.registry, ragpipe.retrieval.graph_substrate"`.
- [ ] **Step 5: commit** `feat(graphrag): build_graph driver + register GRAPHRAG mode`.

---

## Phase 3 done-when
- `uv run pytest tests/ -q` green; ruff clean.
- `RetrievalMode.GRAPHRAG` resolves to a GraphRAGSubstrate; graph mode uses PassthroughReranker.
- `build_graph` exists (live, untested) to populate the three graph indexes.
- Live graph ingest is a morning/Azure step.

## Self-review notes
- Spec coverage: design §2 (GraphRAG) → T1-T3,T7; §3 build_graph → T5,T8; graph indexes → T4; local/global → T7; reranker incompatibility handled by PassthroughReranker (T6) — a design addition not in the spec, recorded in the decision log.
- Type consistency: `Entity`, `Relationship`, `Community`, `parse_extraction(raw, source_chunk_id, source_url)`, `merge_entities`, `detect_communities(names, rels, seed)`, `entity_documents(entities, community, embed_batch_fn)`, `GraphRAGSubstrate(name, entity_search, community_search, adjacency, rrf_k)` consistent across tasks.
