# Multi-substrate retrieval — Phase 2 (SAC + RAPTOR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the SAC+RAPTOR retrieval mode: a RAPTOR summary tree built at ingest over the existing SAC (contextual) leaves, stored with the leaves in a `raptor-sac` index, retrieved as a collapsed tree (flat hybrid search across all levels).

**Architecture:** RAPTOR's retrieval is collapsed-tree, i.e. all nodes (leaf chunks + summary nodes) live in one index and are searched together. So the substrate is just `HybridSubstrate` pointed at `raptor_sac_index` — Phase 1's seam already provides it via the registry. The new work is the build: cluster leaf embeddings, LLM-summarize each cluster into a parent node, re-embed, recurse until one root or a level cap; tag every node with a `level`; upload leaves + summaries to `raptor_sac_index`.

**Tech Stack:** Python 3.12, `uv`, pytest/pytest-asyncio, numpy + scikit-learn (new dep: GaussianMixture clustering with BIC), Azure AI Search. The seam (`RetrievalSubstrate`, `HybridSubstrate`, registry, `RetrievalMode`), config index names, and ingest helpers (`build_documents`, `fetch_pages`, `_upload_in_batches`, `prune_stale_documents`, `build_batch_embed_fn`) already exist on `main`.

**Conventions:** `uv run` for everything; `uv run ruff check .` clean; `from __future__ import annotations` at top of every module; commit after each task with the shown message; branch `feat/multi-substrate-phase2`. No live Azure/LLM in tests — inject fakes. Live drivers stay `# pragma: no cover`.

---

## File structure (Phase 2)

- Create `src/ragpipe/raptor.py` — clustering + tree build (pure, dependency-injected LLM/embed).
- Create `tests/test_raptor.py`.
- Modify `src/ragpipe/search_index.py` — `include_level: bool = False` param (adds a filterable `level` field).
- Modify `src/ragpipe/ingest.py` — `build_raptor(settings, limit)` live driver + a pure `raptor_summary_documents(...)` shaper.
- Modify `src/ragpipe/retrieval/registry.py` — register `RetrievalMode.RAPTOR_SAC`.
- Modify `src/ragpipe/app_wiring.py` — add `raptor_sac → raptor_sac_index` to the rerank-index map.
- Modify `pyproject.toml` — add `scikit-learn` (and `numpy` if not already present transitively).
- Modify `tests/test_search_index.py`, `tests/retrieval/test_registry.py` — cover the new field + mode.

---

## Task 1: add scikit-learn dependency

**Files:** Modify `pyproject.toml`.

- [ ] **Step 1:** Add `scikit-learn>=1.5` to `[project] dependencies` (and `numpy>=1.26` if it is not already a direct or transitive pin you can rely on — check `uv pip list | grep -i numpy` first; numpy ships with scikit-learn, so an explicit pin is optional). Run `uv sync` (or `uv lock && uv sync`) to update `uv.lock`.
- [ ] **Step 2:** Verify import: `uv run python -c "import sklearn, numpy; print(sklearn.__version__)"`.
- [ ] **Step 3:** Commit:
```bash
git add pyproject.toml uv.lock
git commit -m "build: add scikit-learn for RAPTOR clustering"
```

---

## Task 2: RAPTOR clustering (`raptor.py`)

A pure function that clusters embedding vectors into groups using GaussianMixture with BIC-selected component count, capped.

**Files:** Create `src/ragpipe/raptor.py`; Test `tests/test_raptor.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_raptor.py
from __future__ import annotations

from ragpipe.raptor import cluster_embeddings


def test_cluster_separates_two_obvious_groups():
    # two tight, well-separated blobs in 2D
    va = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    vb = [[10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]
    labels = cluster_embeddings(va + vb, max_clusters=4, random_state=0)
    # the three points of each blob share a label; the two blobs differ
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_cluster_handles_tiny_input():
    labels = cluster_embeddings([[1.0, 2.0]], max_clusters=4, random_state=0)
    assert labels == [0]
```

- [ ] **Step 2: Run, confirm fail** — `uv run pytest tests/test_raptor.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** in `src/ragpipe/raptor.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def cluster_embeddings(
    vectors: list[list[float]], *, max_clusters: int = 50, random_state: int = 0
) -> list[int]:
    """Hard-assign each vector to a cluster. Component count chosen by BIC over
    1..min(max_clusters, n-1) GaussianMixtures (RAPTOR uses GMM soft clustering;
    we take the argmax responsibility as a hard label). Degenerate inputs (<=2
    vectors) return a single cluster."""
    n = len(vectors)
    if n <= 2:
        return [0] * n
    x = np.asarray(vectors, dtype=float)
    upper = min(max_clusters, n - 1)
    best_bic = float("inf")
    best_labels = [0] * n
    for k in range(1, upper + 1):
        gm = GaussianMixture(n_components=k, random_state=random_state)
        gm.fit(x)
        bic = gm.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_labels = gm.predict(x).tolist()
    return best_labels
```

- [ ] **Step 4: Run, confirm pass** — `uv run pytest tests/test_raptor.py -v`. `uv run ruff check src/ragpipe/raptor.py tests/test_raptor.py`.

- [ ] **Step 5: Commit**
```bash
git add src/ragpipe/raptor.py tests/test_raptor.py
git commit -m "feat(raptor): BIC-selected GMM clustering of embeddings"
```

---

## Task 3: RAPTOR tree build (`raptor.py`)

Recursively cluster → summarize → re-embed → repeat, producing summary nodes with levels. LLM + embedder are injected for testability.

**Files:** Modify `src/ragpipe/raptor.py`; Test `tests/test_raptor.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
from ragpipe.raptor import RaptorNode, build_raptor_tree


def test_build_tree_produces_higher_level_nodes_and_terminates():
    # 6 leaves in two clusters; fake summarize concatenates ids; fake embed maps
    # text length to a 1-D vector so re-clustering still works.
    leaves = [
        RaptorNode(id=f"L{i}", text=t, level=0, embedding=e)
        for i, (t, e) in enumerate([
            ("alpha", [0.0, 0.0]), ("alpha2", [0.1, 0.0]), ("alpha3", [0.0, 0.1]),
            ("beta", [9.0, 9.0]), ("beta2", [9.1, 9.0]), ("beta3", [9.0, 9.1]),
        ])
    ]

    def fake_summarize(texts: list[str]) -> str:
        return "summary:" + "|".join(texts)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]

    summaries = build_raptor_tree(
        leaves, summarize_fn=fake_summarize, embed_batch_fn=fake_embed,
        max_levels=3, max_clusters=4, random_state=0,
    )
    assert summaries, "expected at least one summary node"
    assert all(s.level >= 1 for s in summaries)
    # terminates: at most a handful of nodes, never explodes
    assert len(summaries) < len(leaves) * 3
    # summary text came from the injected summarizer
    assert any(s.text.startswith("summary:") for s in summaries)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** (append to `raptor.py`):

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class RaptorNode:
    id: str
    text: str
    level: int
    embedding: list[float]


def build_raptor_tree(
    leaves: list[RaptorNode],
    *,
    summarize_fn: Callable[[list[str]], str],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    max_levels: int = 3,
    max_clusters: int = 50,
    random_state: int = 0,
) -> list[RaptorNode]:
    """Build RAPTOR summary nodes above the leaves. Returns ONLY the summary
    nodes (levels >= 1); the caller already has the leaves. Stops when a level
    yields a single cluster or max_levels is reached."""
    summaries: list[RaptorNode] = []
    current = leaves
    level = 1
    while level <= max_levels and len(current) > 1:
        labels = cluster_embeddings(
            [n.embedding for n in current], max_clusters=max_clusters, random_state=random_state
        )
        groups: dict[int, list[RaptorNode]] = {}
        for node, lab in zip(current, labels):
            groups.setdefault(lab, []).append(node)
        if len(groups) <= 1 and level > 1:
            break  # converged: one cluster covers everything
        texts = [summarize_fn([n.text for n in group]) for group in groups.values()]
        embeddings = embed_batch_fn(texts)
        new_nodes = [
            RaptorNode(id=f"raptor-L{level}-{i}", text=t, level=level, embedding=e)
            for i, (t, e) in enumerate(zip(texts, embeddings))
        ]
        summaries.extend(new_nodes)
        current = new_nodes
        level += 1
    return summaries
```

- [ ] **Step 4: Run, confirm pass.** `uv run ruff check` clean.

- [ ] **Step 5: Commit**
```bash
git add src/ragpipe/raptor.py tests/test_raptor.py
git commit -m "feat(raptor): recursive cluster-summarize tree build"
```

---

## Task 4: `level` field in the index schema

**Files:** Modify `src/ragpipe/search_index.py`; Test `tests/test_search_index.py`.

- [ ] **Step 1: Write the failing test** (append to tests/test_search_index.py)

```python
def test_include_level_adds_filterable_level_field():
    from ragpipe.search_index import build_index
    idx = build_index("raptor-sac", 1536, include_context=True, include_level=True)
    fields = {f.name: f for f in idx.fields}
    assert "level" in fields
    assert fields["level"].filterable is True
    # default keeps it out
    idx2 = build_index("baseline", 1536)
    assert "level" not in {f.name for f in idx2.fields}
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** Add `include_level: bool = False` param to `build_index`. When True, append a field: `SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True, facetable=True)` (import whatever `SimpleField`/`SearchFieldDataType` the module already uses). Do not add it to BM25/semantic config. Keep default behavior identical.

- [ ] **Step 4: Run, confirm pass.** `uv run ruff check` clean.

- [ ] **Step 5: Commit**
```bash
git add src/ragpipe/search_index.py tests/test_search_index.py
git commit -m "feat(index): optional filterable level field for RAPTOR"
```

---

## Task 5: RAPTOR document shaping + live build driver (`ingest.py`)

**Files:** Modify `src/ragpipe/ingest.py`; Test `tests/test_ingest_raptor.py`.

- [ ] **Step 1: Write the failing test** — pure shaper only (no Azure/LLM):

```python
# tests/test_ingest_raptor.py
from __future__ import annotations

from ragpipe.ingest import raptor_summary_documents
from ragpipe.raptor import RaptorNode


def test_summary_documents_carry_level_and_embedding():
    nodes = [RaptorNode(id="raptor-L1-0", text="a summary", level=1, embedding=[0.5])]
    docs = raptor_summary_documents(nodes)
    d = docs[0]
    assert d["id"] == "raptor-L1-0"
    assert d["level"] == 1
    assert d["content"] == "a summary"
    assert d["content_vector"] == [0.5]
    assert d.get("context", "") == ""   # summaries carry no SAC context
    assert d["url"] == ""               # summary nodes are not a single page
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `raptor_summary_documents(nodes)`** in ingest.py: map each `RaptorNode` to a doc dict matching the index schema fields used elsewhere (`id`, `title`, `url`, `chunk_id`, `content`, `context`, `content_vector`) plus `level`. For summaries: `title=f"RAPTOR summary L{level}"`, `url=""`, `chunk_id=node.id`, `content=node.text`, `context=""`, `content_vector=node.embedding`, `level=node.level`. Leaf docs (level 0) will come from the existing `build_documents` (Step 4 sets their `level=0`).

- [ ] **Step 4: Implement `build_raptor(settings, limit=None)`** live driver (`# pragma: no cover`), mirroring `build_baseline`:
  - `fetch_pages` → `build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)` to get SAC leaf docs. Add `"level": 0` to each leaf doc (set it on the dicts returned by build_documents before upload).
  - Build `RaptorNode` leaves from those leaf docs (`id`, `text=content`, `level=0`, `embedding=content_vector`).
  - `summaries = build_raptor_tree(leaves, summarize_fn=<LLM summarizer>, embed_batch_fn=embed_batch, max_levels=settings.raptor_max_levels)`. The summarizer is an LLM call over the joined member texts; reuse the existing chat-completion client used by `context_gen`/the generator (build a small `summarize_fn(texts)` that prompts "Summarize the following related documentation passages into a concise abstractive summary" and returns text). Bound + cache it the way `context_gen` does (ADR-0005 / ADR-0011 timeouts) if straightforward; if not, at minimum bound with the existing timeout/retry helper. Note any shortcut as DONE_WITH_CONCERNS.
  - `create_or_update_index` via `build_index(settings.raptor_sac_index, dims, include_context=True, include_level=True)`.
  - `docs = leaf_docs + raptor_summary_documents(summaries)`; `_upload_in_batches`; `prune_stale_documents`.

- [ ] **Step 5: Run, confirm pass** (`tests/test_ingest_raptor.py`). `uv run python -c "import ragpipe.ingest"`. `uv run ruff check` clean.

- [ ] **Step 6: Commit**
```bash
git add src/ragpipe/ingest.py tests/test_ingest_raptor.py
git commit -m "feat(ingest): RAPTOR build driver + summary-node shaping"
```

---

## Task 6: register the RAPTOR_SAC mode

**Files:** Modify `src/ragpipe/retrieval/registry.py`, `src/ragpipe/app_wiring.py`; Test `tests/retrieval/test_registry.py`.

- [ ] **Step 1: Write the failing test** (append to test_registry.py)

```python
def test_raptor_sac_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.RAPTOR_SAC in registered_modes()


def test_build_raptor_sac_substrate_uses_raptor_index():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        search_index = "ms-docs"
        baseline_index = "baseline"
        raptor_sac_index = "raptor-sac"
        candidate_pool = 15
        rrf_k = 60
        top_k = 5

    class _Ctx:
        def search_client(self, index): return object()
        def embed(self, text): return [0.0]

    sub = build_substrate(RetrievalMode.RAPTOR_SAC, settings=_Settings(), ctx=_Ctx())
    assert sub.name == "raptor_sac"
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** In `registry.py`, add to `_REGISTRY`:
```python
    RetrievalMode.RAPTOR_SAC: _hybrid("raptor_sac_index", "raptor_sac"),
```
In `app_wiring.py`'s `rerank_index_attr` map, add `"raptor_sac": "raptor_sac_index"`.

- [ ] **Step 4: Run, confirm pass.** Then run the WHOLE suite `uv run pytest tests/ -q` and `uv run ruff check .` — both green.

- [ ] **Step 5: Commit**
```bash
git add src/ragpipe/retrieval/registry.py src/ragpipe/app_wiring.py tests/retrieval/test_registry.py
git commit -m "feat(retrieval): register RAPTOR_SAC mode (collapsed-tree over raptor-sac index)"
```

---

## Phase 2 done-when
- `uv run pytest tests/ -q` green; `uv run ruff check .` clean.
- `RetrievalMode.RAPTOR_SAC` resolves to a HybridSubstrate over `raptor_sac_index`.
- `build_raptor` exists (live, untested) to populate that index with leaves (level 0) + summary nodes (level >= 1).
- Live ingest of `raptor-sac` is a morning/Azure step, not run here.

## Self-review notes
- Spec coverage: design §2 (SAC+RAPTOR) → T2/T3/T5/T6; §3 build_raptor → T5; §4 raptor-sac index → T4/T5. Retrieval reuses the Phase 1 seam (no new substrate class needed).
- Type consistency: `cluster_embeddings(vectors, max_clusters, random_state)`, `RaptorNode(id,text,level,embedding)`, `build_raptor_tree(leaves, summarize_fn, embed_batch_fn, max_levels, max_clusters, random_state)`, `raptor_summary_documents(nodes)`, registry `_hybrid("raptor_sac_index","raptor_sac")` used consistently.
