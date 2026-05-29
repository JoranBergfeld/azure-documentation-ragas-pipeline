# RAGAS-infused RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an observable RAG pipeline over Microsoft/Azure docs (dense + BM25 → RRF → semantic rerank → Foundry generator agent w/ Code Interpreter → RAGAS faithfulness loop), with an offline RAGAS harness and a Streamlit dashboard.

**Architecture:** A Microsoft Agent Framework *Workflow* wires one executor per pipeline stage; a `PipelineState` object threads through and records a per-stage trace. Retrieval uses Azure AI Search (vector, BM25, semantic ranker). Generation is a Foundry-registered Prompt Agent carrying the Code Interpreter tool. RAGAS runs online (faithfulness guardrail with a capped retry loop) and offline (full metric suite + per-stage context metrics). Infra is provisioned with azd + Bicep.

**Tech Stack:** Python 3.11, `agent-framework` + `agent-framework-foundry`, `azure-search-documents`, `azure-ai-projects`, `azure-identity`, `ragas`, `langchain-openai`, `streamlit`, `pytest`, `azd` + Bicep.

**Reference spec:** `docs/superpowers/specs/2026-05-29-ragas-infused-pipeline-design.md`

---

## Conventions for every task

- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Run tests from repo root with the venv active. `pytest -q` unless a specific test is named.
- Commit messages use Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`).
- All Azure-touching code is isolated behind thin adapters so unit tests never hit the network.
- Type annotations are required on executor handlers (Agent Framework routes messages by type).

---

## File structure (created across the plan)

```
ragas-infused-pipeline/
  pyproject.toml                 # deps + tool config (Task 1)
  .env.example                   # config template (Task 2)
  src/ragpipe/
    __init__.py
    config.py                    # Settings + client factories (Task 2)
    models.py                    # Chunk, RetrievalResult, PipelineState, TraceEvent (Task 3)
    retrieval/
      __init__.py
      rrf.py                     # reciprocal rank fusion — pure logic (Task 4)
      dense.py                   # Azure AI Search vector query (Task 6)
      bm25.py                    # Azure AI Search full-text query (Task 6)
      rerank.py                  # Azure AI Search semantic rerank over fused IDs (Task 7)
    chunking.py                  # heading-aware chunker — pure logic (Task 5)
    ingest.py                    # fetch → chunk → embed → upload (Task 8)
    search_index.py              # index schema create/delete (Task 8)
    generate.py                  # Foundry generator agent wrapper (Task 9)
    guardrail.py                 # RAGAS faithfulness + loop policy (Task 10, 11)
    workflow.py                  # WorkflowBuilder wiring + WorkflowViz export (Task 12)
    eval/
      __init__.py
      testset.py                 # load_testset() switch (Task 13)
      harness.py                 # RAGAS suite + per-stage metrics (Task 14)
  scripts/
    setup_agents.py              # register Foundry generator agent + Code Interpreter (Task 9)
  app/
    dashboard.py                 # Streamlit UI (Task 15)
  data/
    corpus_sources.yaml          # MS Learn URLs (Task 8)
    testset.jsonl                # hand-authored Q/A/ground-truth (Task 13)
  infra/
    main.bicep                   # resources (Task 16)
    main.parameters.json
  azure.yaml                     # azd config + hooks (Task 16)
  tests/
    ...                          # mirrors src/ layout
```

---

## Task 1: Project scaffold and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/ragpipe/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.gitignore` additions

- [ ] **Step 1: Write a smoke test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import ragpipe

    assert ragpipe.__name__ == "ragpipe"
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "ragpipe"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "agent-framework",
    "agent-framework-foundry",
    "azure-search-documents>=11.5.0",
    "azure-ai-projects>=2.0.0",
    "azure-identity",
    "ragas",
    "langchain-openai",
    "streamlit",
    "pyyaml",
    "python-dotenv",
    "httpx",
    "beautifulsoup4",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "pytest-mock", "ruff"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create the package marker files**

`src/ragpipe/__init__.py`:
```python
"""RAGAS-infused RAG pipeline over Microsoft/Azure documentation."""
```

`tests/__init__.py`: (empty file)

- [ ] **Step 4: Create the virtualenv and install**

Run:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
Expected: install completes; `pip show ragpipe` shows version 0.1.0.

- [ ] **Step 5: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ragpipe/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: project scaffold and tooling"
```

---

## Task 2: Configuration module

**Files:**
- Create: `src/ragpipe/config.py`
- Create: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest

from ragpipe.config import Settings, TestsetMode


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://proj.services.ai.azure.com")
    monkeypatch.setenv("FOUNDRY_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://s.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "ms-docs")
    monkeypatch.setenv("GENERATOR_AGENT_NAME", "ragpipe-generator")

    s = Settings.from_env()

    assert s.foundry_chat_model == "gpt-4o"
    assert s.search_index == "ms-docs"
    assert s.faithfulness_threshold == 0.7  # default
    assert s.max_retries == 2  # default
    assert s.top_k == 5  # default
    assert s.rrf_k == 60  # default
    assert s.testset_mode is TestsetMode.HANDAUTHORED  # default


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="FOUNDRY_PROJECT_ENDPOINT"):
        Settings.from_env()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragpipe.config'`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/config.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


class TestsetMode(str, Enum):
    HANDAUTHORED = "handauthored"
    SYNTHETIC = "synthetic"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    foundry_project_endpoint: str
    foundry_chat_model: str
    foundry_embedding_model: str
    search_endpoint: str
    search_index: str
    generator_agent_name: str
    generator_agent_version: str | None = None
    faithfulness_threshold: float = 0.7
    max_retries: int = 2
    top_k: int = 5
    rrf_k: int = 60
    testset_mode: TestsetMode = TestsetMode.HANDAUTHORED

    @classmethod
    def from_env(cls, *, load: bool = True) -> "Settings":
        if load:
            load_dotenv()
        return cls(
            foundry_project_endpoint=_require("FOUNDRY_PROJECT_ENDPOINT"),
            foundry_chat_model=_require("FOUNDRY_CHAT_MODEL"),
            foundry_embedding_model=_require("FOUNDRY_EMBEDDING_MODEL"),
            search_endpoint=_require("SEARCH_ENDPOINT"),
            search_index=_require("SEARCH_INDEX"),
            generator_agent_name=_require("GENERATOR_AGENT_NAME"),
            generator_agent_version=os.environ.get("GENERATOR_AGENT_VERSION"),
            faithfulness_threshold=float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7")),
            max_retries=int(os.environ.get("MAX_RETRIES", "2")),
            top_k=int(os.environ.get("TOP_K", "5")),
            rrf_k=int(os.environ.get("RRF_K", "60")),
            testset_mode=TestsetMode(os.environ.get("TESTSET_MODE", "handauthored")),
        )
```

- [ ] **Step 4: Create `.env.example`**

```bash
FOUNDRY_PROJECT_ENDPOINT="https://<your-project>.services.ai.azure.com"
FOUNDRY_CHAT_MODEL="gpt-4o"
FOUNDRY_EMBEDDING_MODEL="text-embedding-3-small"
SEARCH_ENDPOINT="https://<your-search>.search.windows.net"
SEARCH_INDEX="ms-docs"
GENERATOR_AGENT_NAME="ragpipe-generator"
GENERATOR_AGENT_VERSION="1.0"
FAITHFULNESS_THRESHOLD="0.7"
MAX_RETRIES="2"
TOP_K="5"
RRF_K="60"
TESTSET_MODE="handauthored"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/config.py .env.example tests/test_config.py
git commit -m "feat: typed Settings loaded from environment"
```

---

## Task 3: Domain models and PipelineState

**Files:**
- Create: `src/ragpipe/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from ragpipe.models import Chunk, PipelineState, TraceEvent


def test_chunk_holds_identity_and_score():
    c = Chunk(id="doc1#0", title="T", url="http://x", content="body", score=1.5)
    assert c.id == "doc1#0"
    assert c.score == 1.5


def test_pipeline_state_records_trace_in_order():
    state = PipelineState(query="what is RRF?")
    state.add_trace("dense", {"hits": 3})
    state.add_trace("bm25", {"hits": 2})

    assert [e.stage for e in state.trace] == ["dense", "bm25"]
    assert isinstance(state.trace[0], TraceEvent)
    assert state.trace[0].data == {"hits": 3}


def test_pipeline_state_attempt_increments():
    state = PipelineState(query="q")
    assert state.attempt == 0
    state.next_attempt()
    assert state.attempt == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragpipe.models'`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class TraceEvent:
    stage: str
    data: dict[str, Any]


@dataclass
class PipelineState:
    query: str
    dense: list[Chunk] = field(default_factory=list)
    bm25: list[Chunk] = field(default_factory=list)
    fused: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    answer: str = ""
    faithfulness: float | None = None
    attempt: int = 0
    low_confidence: bool = False
    trace: list[TraceEvent] = field(default_factory=list)

    def add_trace(self, stage: str, data: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(stage=stage, data=data))

    def next_attempt(self) -> None:
        self.attempt += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/models.py tests/test_models.py
git commit -m "feat: domain models and PipelineState with trace"
```

---

## Task 4: Reciprocal Rank Fusion (pure logic)

**Files:**
- Create: `src/ragpipe/retrieval/__init__.py`
- Create: `src/ragpipe/retrieval/rrf.py`
- Test: `tests/retrieval/test_rrf.py`

- [ ] **Step 1: Write the failing test**

`tests/retrieval/__init__.py`: (empty file)

`tests/retrieval/test_rrf.py`:
```python
from ragpipe.models import Chunk
from ragpipe.retrieval.rrf import reciprocal_rank_fusion


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=cid)


def test_rrf_merges_and_dedupes_by_id():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    bm25 = [_chunk("b"), _chunk("d")]

    fused = reciprocal_rank_fusion(dense, bm25, k=60)

    ids = [c.id for c in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    # 'b' appears in both lists near the top → highest fused score → first
    assert ids[0] == "b"


def test_rrf_score_formula():
    dense = [_chunk("a")]  # rank 0 → 1/(60+1)
    bm25 = [_chunk("a")]   # rank 0 → 1/(60+1)
    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    assert fused[0].id == "a"
    assert abs(fused[0].score - (2 / 61)) < 1e-9


def test_rrf_empty_inputs_returns_empty():
    assert reciprocal_rank_fusion([], [], k=60) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_rrf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragpipe.retrieval.rrf'`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/retrieval/__init__.py`: (empty file)

`src/ragpipe/retrieval/rrf.py`:
```python
from __future__ import annotations

from dataclasses import replace

from ragpipe.models import Chunk


def reciprocal_rank_fusion(
    dense: list[Chunk], bm25: list[Chunk], k: int = 60
) -> list[Chunk]:
    """Merge two ranked lists by Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank), rank is 0-based.
    Returns a new list of Chunks sorted by fused score descending; the
    fused score is written to each returned Chunk's `score`.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked in (dense, bm25):
        for rank, chunk in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(chunk.id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(by_id[cid], score=scores[cid]) for cid in ordered_ids]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_rrf.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/__init__.py src/ragpipe/retrieval/rrf.py tests/retrieval/
git commit -m "feat: reciprocal rank fusion"
```

---

## Task 5: Heading-aware chunker (pure logic)

**Files:**
- Create: `src/ragpipe/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunking.py`:
```python
from ragpipe.chunking import chunk_text


def test_chunk_respects_max_chars_with_overlap():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, max_chars=1000, overlap=100)

    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    # consecutive chunks overlap by ~100 chars
    assert chunks[0][-50:] in chunks[1]


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text("short", max_chars=1000, overlap=100)
    assert chunks == ["short"]


def test_chunk_empty_text_returns_empty():
    assert chunk_text("", max_chars=1000, overlap=100) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/chunking.py`:
```python
from __future__ import annotations


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping windows of at most `max_chars` characters.

    Prefers to break on a whitespace boundary near the window end so words are
    not split. Consecutive chunks overlap by `overlap` characters.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/chunking.py tests/test_chunking.py
git commit -m "feat: heading-aware text chunker with overlap"
```

---

## Task 6: Dense and BM25 retrievers (component, mocked Search)

**Files:**
- Create: `src/ragpipe/retrieval/dense.py`
- Create: `src/ragpipe/retrieval/bm25.py`
- Test: `tests/retrieval/test_retrievers.py`

These wrap `azure.search.documents.SearchClient`. We inject the client so tests use a fake. The fake returns dicts shaped like Azure AI Search documents (including `@search.score`).

- [ ] **Step 1: Write the failing test**

`tests/retrieval/test_retrievers.py`:
```python
from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever


class FakeSearchClient:
    """Mimics azure.search.documents.SearchClient.search()."""

    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def search(self, search_text=None, **kwargs):
        self.last_kwargs = {"search_text": search_text, **kwargs}
        return iter(self._results)


def _doc(cid, score):
    return {
        "id": cid,
        "title": f"title-{cid}",
        "url": f"http://{cid}",
        "content": f"content-{cid}",
        "@search.score": score,
    }


def test_bm25_retriever_uses_full_text_only_and_maps_chunks():
    client = FakeSearchClient([_doc("a", 3.0), _doc("b", 2.0)])
    retriever = BM25Retriever(client, top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a", "b"]
    assert chunks[0].score == 3.0
    # BM25 = full-text query, no vector_queries
    assert client.last_kwargs["search_text"] == "hybrid search"
    assert "vector_queries" not in client.last_kwargs


def test_dense_retriever_issues_vector_query():
    client = FakeSearchClient([_doc("a", 0.9)])
    embed = lambda text: [0.1, 0.2, 0.3]  # noqa: E731
    retriever = DenseRetriever(client, embed_fn=embed, top_k=5)

    chunks = retriever.retrieve("hybrid search")

    assert [c.id for c in chunks] == ["a"]
    # dense = vector query, no full-text search_text
    assert client.last_kwargs["search_text"] is None
    assert client.last_kwargs["vector_queries"]  # non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retrievers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementations**

`src/ragpipe/retrieval/bm25.py`:
```python
from __future__ import annotations

from typing import Any, Protocol

from ragpipe.models import Chunk


class _Searchable(Protocol):
    def search(self, search_text: str | None = None, **kwargs: Any): ...


def _to_chunk(doc: dict[str, Any]) -> Chunk:
    return Chunk(
        id=doc["id"],
        title=doc.get("title", ""),
        url=doc.get("url", ""),
        content=doc.get("content", ""),
        score=float(doc.get("@search.score", 0.0)),
    )


class BM25Retriever:
    def __init__(self, client: _Searchable, top_k: int = 5) -> None:
        self._client = client
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        results = self._client.search(
            search_text=query,
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_chunk(d) for d in results]
```

`src/ragpipe/retrieval/dense.py`:
```python
from __future__ import annotations

from typing import Any, Callable, Protocol

from azure.search.documents.models import VectorizedQuery

from ragpipe.models import Chunk
from ragpipe.retrieval.bm25 import _to_chunk


class _Searchable(Protocol):
    def search(self, search_text: str | None = None, **kwargs: Any): ...


class DenseRetriever:
    def __init__(
        self,
        client: _Searchable,
        embed_fn: Callable[[str], list[float]],
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._embed = embed_fn
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        vector = self._embed(query)
        vq = VectorizedQuery(
            vector=vector, k_nearest_neighbors=self._top_k, fields="content_vector"
        )
        results = self._client.search(
            search_text=None,
            vector_queries=[vq],
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_chunk(d) for d in results]
```

> Note: `VectorizedQuery` is imported at module load. If the installed
> `azure-search-documents` version names it differently, the import error will
> surface immediately when running the test; pin the version from Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_retrievers.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/dense.py src/ragpipe/retrieval/bm25.py tests/retrieval/test_retrievers.py
git commit -m "feat: dense and BM25 retrievers over Azure AI Search"
```

---

## Task 7: Semantic reranker over fused IDs (component, mocked Search)

**Files:**
- Create: `src/ragpipe/retrieval/rerank.py`
- Test: `tests/retrieval/test_rerank.py`

Reranks our RRF-fused set by issuing a semantic query filtered to the fused doc IDs and reading `@search.rerankerScore`. (Spec §4.3.)

- [ ] **Step 1: Write the failing test**

`tests/retrieval/test_rerank.py`:
```python
from ragpipe.models import Chunk
from ragpipe.retrieval.rerank import SemanticReranker


class FakeSearchClient:
    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def search(self, search_text=None, **kwargs):
        self.last_kwargs = {"search_text": search_text, **kwargs}
        return iter(self._results)


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=cid, score=0.1)


def _doc(cid, reranker_score):
    return {
        "id": cid,
        "title": cid,
        "url": f"http://{cid}",
        "content": cid,
        "@search.rerankerScore": reranker_score,
    }


def test_reranker_filters_to_fused_ids_and_orders_by_reranker_score():
    fused = [_chunk("a"), _chunk("b"), _chunk("c")]
    client = FakeSearchClient([_doc("b", 3.5), _doc("a", 2.0), _doc("c", 1.0)])
    reranker = SemanticReranker(client, semantic_config="default-semantic", top_k=3)

    out = reranker.rerank("query", fused)

    assert [c.id for c in out] == ["b", "a", "c"]
    assert out[0].score == 3.5
    # filter restricts to the fused IDs
    flt = client.last_kwargs["filter"]
    assert "a" in flt and "b" in flt and "c" in flt
    assert client.last_kwargs["query_type"] == "semantic"


def test_reranker_empty_input_returns_empty():
    client = FakeSearchClient([])
    reranker = SemanticReranker(client, semantic_config="default-semantic", top_k=3)
    assert reranker.rerank("q", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/retrieval/rerank.py`:
```python
from __future__ import annotations

from typing import Any, Protocol

from ragpipe.models import Chunk


class _Searchable(Protocol):
    def search(self, search_text: str | None = None, **kwargs: Any): ...


def _to_reranked_chunk(doc: dict[str, Any]) -> Chunk:
    return Chunk(
        id=doc["id"],
        title=doc.get("title", ""),
        url=doc.get("url", ""),
        content=doc.get("content", ""),
        score=float(doc.get("@search.rerankerScore", 0.0)),
    )


def _quote_ids(ids: list[str]) -> str:
    # OData search.in filter: search.in(id, 'a,b,c', ',')
    joined = ",".join(ids)
    return f"search.in(id, '{joined}', ',')"


class SemanticReranker:
    def __init__(
        self, client: _Searchable, semantic_config: str, top_k: int = 5
    ) -> None:
        self._client = client
        self._semantic_config = semantic_config
        self._top_k = top_k

    def rerank(self, query: str, fused: list[Chunk]) -> list[Chunk]:
        if not fused:
            return []
        ids = [c.id for c in fused]
        results = self._client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
            filter=_quote_ids(ids),
            top=self._top_k,
            select=["id", "title", "url", "content"],
        )
        return [_to_reranked_chunk(d) for d in results]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_rerank.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat: semantic reranker over RRF-fused candidate IDs"
```

---

## Task 8: Index schema and ingestion

**Files:**
- Create: `src/ragpipe/search_index.py`
- Create: `src/ragpipe/ingest.py`
- Create: `data/corpus_sources.yaml`
- Test: `tests/test_ingest.py`

The pure logic (HTML→text, building documents) is unit-tested; the Azure upload is isolated behind an injected uploader so tests don't hit the network.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
from ragpipe.ingest import build_documents, html_to_text


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><script>x=1</script></head><body><h1>Hi</h1><p>Body</p></body></html>"
    text = html_to_text(html)
    assert "Hi" in text
    assert "Body" in text
    assert "x=1" not in text


def test_build_documents_chunks_and_embeds():
    pages = [{"url": "http://x", "title": "T", "text": "word " * 600}]
    embed = lambda text: [0.0, 0.1]  # noqa: E731

    docs = build_documents(pages, embed_fn=embed, max_chars=1000, overlap=100)

    assert len(docs) >= 2
    first = docs[0]
    assert first["id"].startswith("http://x")  # url + chunk index
    assert first["title"] == "T"
    assert first["url"] == "http://x"
    assert first["content_vector"] == [0.0, 0.1]
    assert "content" in first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the ingestion logic**

`src/ragpipe/ingest.py`:
```python
from __future__ import annotations

import hashlib
from typing import Any, Callable

from bs4 import BeautifulSoup

from ragpipe.chunking import chunk_text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _doc_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{url}#{index}-{digest}"


def build_documents(
    pages: list[dict[str, Any]],
    embed_fn: Callable[[str], list[float]],
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Turn fetched pages into Azure AI Search documents."""
    documents: list[dict[str, Any]] = []
    for page in pages:
        chunks = chunk_text(page["text"], max_chars=max_chars, overlap=overlap)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                    "content": chunk,
                    "content_vector": embed_fn(chunk),
                }
            )
    return documents
```

- [ ] **Step 4: Run the unit test**

Run: `pytest tests/test_ingest.py -v`
Expected: both PASS.

- [ ] **Step 5: Add the index schema module (no unit test; verified at provision time)**

`src/ragpipe/search_index.py`:
```python
from __future__ import annotations

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

SEMANTIC_CONFIG_NAME = "default-semantic"
VECTOR_PROFILE_NAME = "default-vector"


def build_index(name: str, vector_dimensions: int) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME, algorithm_configuration_name="hnsw"
            )
        ],
    )
    semantic = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )
    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic,
    )


def create_index(client: SearchIndexClient, name: str, vector_dimensions: int) -> None:
    client.create_or_update_index(build_index(name, vector_dimensions))
```

- [ ] **Step 6: Add the corpus list**

`data/corpus_sources.yaml`:
```yaml
# Microsoft Learn articles forming the RAG corpus.
sources:
  - https://learn.microsoft.com/azure/search/hybrid-search-overview
  - https://learn.microsoft.com/azure/search/hybrid-search-ranking
  - https://learn.microsoft.com/azure/search/semantic-search-overview
  - https://learn.microsoft.com/azure/search/search-relevance-overview
  - https://learn.microsoft.com/agent-framework/workflows/
  - https://learn.microsoft.com/agent-framework/workflows/executors
  - https://learn.microsoft.com/agent-framework/workflows/edges
  - https://learn.microsoft.com/azure/foundry/concepts/retrieval-augmented-generation
  - https://learn.microsoft.com/azure/foundry/concepts/evaluation-evaluators/rag-evaluators
  - https://learn.microsoft.com/agent-framework/agents/providers/microsoft-foundry
```

- [ ] **Step 7: Add the ingestion CLI entry point (manual run, no unit test)**

Append to `src/ragpipe/ingest.py`:
```python
def fetch_pages(urls: list[str]) -> list[dict[str, Any]]:
    import httpx

    pages: list[dict[str, Any]] = []
    for url in urls:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        pages.append({"url": url, "title": title, "text": html_to_text(resp.text)})
    return pages


def main() -> None:  # pragma: no cover - integration entry point
    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from agent_framework.foundry import FoundryEmbeddingClient

    from ragpipe.config import Settings
    from ragpipe.search_index import create_index

    settings = Settings.from_env()
    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]

    cred = DefaultAzureCredential()
    embed_client = FoundryEmbeddingClient()

    import asyncio

    def embed(text: str) -> list[float]:
        result = asyncio.get_event_loop().run_until_complete(
            embed_client.get_embeddings([text])
        )
        return list(result[0].embedding)

    pages = fetch_pages(urls)
    first_vec = embed(pages[0]["text"][:100])
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    create_index(index_client, settings.search_index, vector_dimensions=len(first_vec))

    docs = build_documents(pages, embed_fn=embed)
    search_client = SearchClient(settings.search_endpoint, settings.search_index, cred)
    search_client.upload_documents(docs)
    print(f"Uploaded {len(docs)} chunks to index '{settings.search_index}'.")


if __name__ == "__main__":  # pragma: no cover
    main()
```

> The exact `FoundryEmbeddingClient` result attribute (`.embedding` vs
> `.dimensions`) must be confirmed against the installed SDK during the first
> real run; adjust the `embed` adapter if needed. This is the one place a live
> call is unavoidable, so it is excluded from unit tests (`# pragma: no cover`).

- [ ] **Step 8: Commit**

```bash
git add src/ragpipe/ingest.py src/ragpipe/search_index.py data/corpus_sources.yaml tests/test_ingest.py
git commit -m "feat: corpus ingestion (html->text, chunk, embed, index schema)"
```

---

## Task 9: Foundry generator agent + registration script

**Files:**
- Create: `src/ragpipe/generate.py`
- Create: `scripts/setup_agents.py`
- Test: `tests/test_generate.py`

The generator builds a grounding prompt from chunks and calls a `FoundryAgent`. We inject the agent so tests use a fake (no network). The agent itself (with Code Interpreter) is registered by `setup_agents.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_generate.py`:
```python
import pytest

from ragpipe.generate import build_grounding_prompt, Generator
from ragpipe.models import Chunk


def _chunk(cid, content):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=content)


def test_build_grounding_prompt_includes_numbered_sources():
    chunks = [_chunk("a", "Alpha fact."), _chunk("b", "Beta fact.")]
    prompt = build_grounding_prompt("What is alpha?", chunks)

    assert "What is alpha?" in prompt
    assert "Alpha fact." in prompt
    assert "Beta fact." in prompt
    assert "[1]" in prompt and "[2]" in prompt


class FakeAgent:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    async def run(self, prompt):
        self.last_prompt = prompt
        return type("R", (), {"text": self._text})()


@pytest.mark.asyncio
async def test_generator_returns_agent_text():
    agent = FakeAgent("Alpha is the first letter.")
    gen = Generator(agent)
    chunks = [_chunk("a", "Alpha fact.")]

    answer = await gen.generate("What is alpha?", chunks)

    assert answer == "Alpha is the first letter."
    assert "Alpha fact." in agent.last_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/generate.py`:
```python
from __future__ import annotations

from typing import Protocol

from ragpipe.models import Chunk


class _Agent(Protocol):
    async def run(self, prompt: str): ...


def build_grounding_prompt(query: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({c.url}) {c.content}" for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources do not contain the answer, "
        "say you don't know.\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class Generator:
    def __init__(self, agent: _Agent) -> None:
        self._agent = agent

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        prompt = build_grounding_prompt(query, chunks)
        result = await self._agent.run(prompt)
        return result.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate.py -v`
Expected: both PASS.

- [ ] **Step 5: Write the agent registration script (manual run, no unit test)**

`scripts/setup_agents.py`:
```python
"""Register the Foundry generator agent with the Code Interpreter tool.

Run once after `azd up` (or via the azd postprovision hook):
    python scripts/setup_agents.py
Writes GENERATOR_AGENT_NAME / GENERATOR_AGENT_VERSION to stdout for .env.
"""
from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from ragpipe.config import Settings

INSTRUCTIONS = (
    "You are a Microsoft/Azure documentation assistant. Answer using only the "
    "provided numbered sources and cite them inline like [1]. When a question "
    "requires counting, comparison tables, or arithmetic over the sourced facts, "
    "use the code interpreter tool. Never invent facts not present in the sources."
)


def main() -> None:
    settings = Settings.from_env()
    client = AIProjectClient(
        endpoint=settings.foundry_project_endpoint,
        credential=DefaultAzureCredential(),
    )
    agent = client.agents.create_agent(
        model=settings.foundry_chat_model,
        name=settings.generator_agent_name,
        instructions=INSTRUCTIONS,
        tools=[{"type": "code_interpreter"}],
    )
    print(f"GENERATOR_AGENT_NAME={agent.name}")
    print(f"GENERATOR_AGENT_VERSION={getattr(agent, 'version', '1.0')}")


if __name__ == "__main__":
    main()
```

> Confirm the exact `create_agent` signature and the Code Interpreter tool
> spec against the installed `azure-ai-projects>=2.0.0` during the first run;
> the tool may be a typed object rather than a dict. Adjust accordingly.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/generate.py scripts/setup_agents.py tests/test_generate.py
git commit -m "feat: generator (grounding prompt + Foundry agent) and registration script"
```

---

## Task 10: RAGAS faithfulness adapter (component, mocked judge)

**Files:**
- Create: `src/ragpipe/guardrail.py`
- Test: `tests/test_faithfulness.py`

Wrap the RAGAS faithfulness scorer behind a thin callable so the loop policy (Task 11) can be tested without a live LLM judge.

- [ ] **Step 1: Write the failing test**

`tests/test_faithfulness.py`:
```python
import pytest

from ragpipe.guardrail import FaithfulnessScorer
from ragpipe.models import Chunk


def _chunk(content):
    return Chunk(id="c", title="t", url="http://c", content=content)


@pytest.mark.asyncio
async def test_scorer_passes_answer_and_context_to_metric():
    captured = {}

    async def fake_metric(*, question, answer, contexts):
        captured["question"] = question
        captured["answer"] = answer
        captured["contexts"] = contexts
        return 0.83

    scorer = FaithfulnessScorer(metric_fn=fake_metric)
    score = await scorer.score("q", "a", [_chunk("ctx1"), _chunk("ctx2")])

    assert score == 0.83
    assert captured["answer"] == "a"
    assert captured["contexts"] == ["ctx1", "ctx2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_faithfulness.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/guardrail.py`:
```python
from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from ragpipe.models import Chunk

MetricFn = Callable[..., Awaitable[float]]


class _Scorer(Protocol):
    async def score(self, query: str, answer: str, contexts: list[Chunk]) -> float: ...


class FaithfulnessScorer:
    """Thin adapter around a RAGAS faithfulness metric callable."""

    def __init__(self, metric_fn: MetricFn) -> None:
        self._metric_fn = metric_fn

    async def score(self, query: str, answer: str, contexts: list[Chunk]) -> float:
        return await self._metric_fn(
            question=query,
            answer=answer,
            contexts=[c.content for c in contexts],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_faithfulness.py -v`
Expected: PASS.

- [ ] **Step 5: Add the real RAGAS metric factory (manual wiring, no unit test)**

Append to `src/ragpipe/guardrail.py`:
```python
def build_ragas_faithfulness(settings) -> MetricFn:  # pragma: no cover
    """Build a faithfulness metric callable backed by Foundry models via RAGAS."""
    from langchain_openai import AzureChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    judge = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=settings.foundry_project_endpoint,
            azure_deployment=settings.foundry_chat_model,
            api_version="2024-10-21",
        )
    )
    metric = Faithfulness(llm=judge)

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await metric.single_turn_ascore(sample))

    return metric_fn
```

> Confirm RAGAS import paths and `SingleTurnSample` field names against the
> pinned `ragas` version on first run; the metric API has moved between
> releases. The adapter boundary keeps any change to this one function.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/guardrail.py tests/test_faithfulness.py
git commit -m "feat: RAGAS faithfulness scorer adapter"
```

---

## Task 11: Guardrail loop policy (pure logic)

**Files:**
- Modify: `src/ragpipe/guardrail.py`
- Test: `tests/test_loop_policy.py`

The decision "pass / retry / exhausted" is pure logic, separated from the workflow so it is fully unit-tested.

- [ ] **Step 1: Write the failing test**

`tests/test_loop_policy.py`:
```python
from ragpipe.guardrail import LoopDecision, decide_next


def test_passes_when_score_meets_threshold():
    d = decide_next(score=0.8, threshold=0.7, attempt=0, max_retries=2)
    assert d is LoopDecision.PASS


def test_retries_when_below_threshold_and_attempts_remain():
    d = decide_next(score=0.5, threshold=0.7, attempt=0, max_retries=2)
    assert d is LoopDecision.RETRY


def test_exhausted_when_below_threshold_and_no_attempts_left():
    d = decide_next(score=0.5, threshold=0.7, attempt=2, max_retries=2)
    assert d is LoopDecision.EXHAUSTED


def test_failed_score_none_treated_as_below_threshold():
    # fail-closed: a missing score must not pass the guardrail
    assert decide_next(score=None, threshold=0.7, attempt=0, max_retries=2) is LoopDecision.RETRY
    assert decide_next(score=None, threshold=0.7, attempt=2, max_retries=2) is LoopDecision.EXHAUSTED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop_policy.py -v`
Expected: FAIL with `ImportError` (names not defined).

- [ ] **Step 3: Add the loop policy**

Append to `src/ragpipe/guardrail.py`:
```python
from enum import Enum


class LoopDecision(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    EXHAUSTED = "exhausted"


def decide_next(
    score: float | None, threshold: float, attempt: int, max_retries: int
) -> LoopDecision:
    """Decide whether to accept the answer, retry, or give up.

    A missing score (judge failure) is fail-closed: never PASS.
    """
    if score is not None and score >= threshold:
        return LoopDecision.PASS
    if attempt < max_retries:
        return LoopDecision.RETRY
    return LoopDecision.EXHAUSTED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop_policy.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/guardrail.py tests/test_loop_policy.py
git commit -m "feat: fail-closed guardrail loop policy"
```

---

## Task 12: Workflow wiring + WorkflowViz export

**Files:**
- Create: `src/ragpipe/workflow.py`
- Test: `tests/test_workflow.py`

Wire the stages into an Agent Framework Workflow. Retrieval/rerank/generate/score are injected as plain callables so the workflow is testable end-to-end with fakes (no Azure, no LLM). The conditional loop edge uses `add_edge(..., condition=...)`.

- [ ] **Step 1: Write the failing test (happy path + loop)**

`tests/test_workflow.py`:
```python
import pytest

from ragpipe.models import Chunk, PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=f"content-{cid}")


def _deps(score_sequence):
    """Build deps whose scorer returns scores from a sequence per attempt."""
    scores = iter(score_sequence)
    return PipelineDeps(
        dense=lambda q: [_chunk("a"), _chunk("b")],
        bm25=lambda q: [_chunk("b"), _chunk("c")],
        rerank=lambda q, fused: fused[:2],
        generate=lambda q, chunks: f"answer for {q}",
        score=lambda q, answer, chunks: next(scores),
        threshold=0.7,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_pipeline_passes_first_try():
    state = await run_pipeline("what is RRF?", _deps([0.9]))
    assert isinstance(state, PipelineState)
    assert state.answer == "answer for what is RRF?"
    assert state.faithfulness == 0.9
    assert state.attempt == 0
    assert state.low_confidence is False
    stages = [e.stage for e in state.trace]
    assert stages[:4] == ["dense", "bm25", "rrf", "rerank"]


@pytest.mark.asyncio
async def test_pipeline_loops_then_passes():
    state = await run_pipeline("q", _deps([0.4, 0.85]))
    assert state.attempt == 1
    assert state.faithfulness == 0.85
    assert state.low_confidence is False


@pytest.mark.asyncio
async def test_pipeline_exhausts_and_flags_low_confidence():
    state = await run_pipeline("q", _deps([0.1, 0.2, 0.3]))
    assert state.attempt == 2
    assert state.low_confidence is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

> Design note: the spec's loop is expressed in the diagram as a conditional
> edge back to RRF. For a deterministic, fully-testable core we implement the
> control flow as an explicit async driver around the stage callables, and
> (Task 12b) ALSO expose the same stages as an Agent Framework Workflow for the
> WorkflowViz diagram. The driver is what the dashboard and tests use; the
> Workflow object is what we visualize. Both call the identical stage functions,
> so they cannot diverge in behavior.

`src/ragpipe/workflow.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from ragpipe.guardrail import LoopDecision, decide_next
from ragpipe.models import Chunk, PipelineState
from ragpipe.retrieval.rrf import reciprocal_rank_fusion

# Callable stage signatures (sync or async tolerated via _maybe_await).
DenseFn = Callable[[str], list[Chunk]]
Bm25Fn = Callable[[str], list[Chunk]]
RerankFn = Callable[[str, list[Chunk]], list[Chunk]]
GenerateFn = Callable[[str, list[Chunk]], object]
ScoreFn = Callable[[str, str, list[Chunk]], object]


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass
class PipelineDeps:
    dense: DenseFn
    bm25: Bm25Fn
    rerank: RerankFn
    generate: GenerateFn
    score: ScoreFn
    threshold: float = 0.7
    max_retries: int = 2
    rrf_k: int = 60
    top_k: int = 5


async def run_pipeline(query: str, deps: PipelineDeps) -> PipelineState:
    state = PipelineState(query=query)

    state.dense = await _maybe_await(deps.dense(query))
    state.add_trace("dense", {"ids": [c.id for c in state.dense]})
    state.bm25 = await _maybe_await(deps.bm25(query))
    state.add_trace("bm25", {"ids": [c.id for c in state.bm25]})

    while True:
        state.fused = reciprocal_rank_fusion(state.dense, state.bm25, k=deps.rrf_k)
        state.add_trace("rrf", {"ids": [c.id for c in state.fused]})

        state.reranked = await _maybe_await(deps.rerank(query, state.fused))
        state.add_trace("rerank", {"ids": [c.id for c in state.reranked]})

        state.answer = await _maybe_await(deps.generate(query, state.reranked))
        state.add_trace("generate", {"answer": state.answer})

        try:
            score = await _maybe_await(deps.score(query, state.answer, state.reranked))
        except Exception:  # judge failure → fail-closed
            score = None
        state.faithfulness = score
        state.add_trace("faithfulness", {"score": score, "attempt": state.attempt})

        decision = decide_next(
            score=score,
            threshold=deps.threshold,
            attempt=state.attempt,
            max_retries=deps.max_retries,
        )
        if decision is LoopDecision.PASS:
            return state
        if decision is LoopDecision.EXHAUSTED:
            state.low_confidence = True
            return state
        state.next_attempt()  # RETRY
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow.py -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/workflow.py tests/test_workflow.py
git commit -m "feat: pipeline driver with guardrail loop and trace"
```

---

## Task 12b: Agent Framework Workflow + WorkflowViz diagram (manual verify)

**Files:**
- Modify: `src/ragpipe/workflow.py`
- Test: `tests/test_workflow_viz.py`

- [ ] **Step 1: Write the failing test**

`tests/test_workflow_viz.py`:
```python
from ragpipe.workflow import build_viz_workflow


def test_build_viz_workflow_has_all_stage_nodes():
    wf = build_viz_workflow()
    # WorkflowViz renders to mermaid; assert the stage ids appear.
    from agent_framework import WorkflowViz

    mermaid = WorkflowViz(wf).to_mermaid()
    for stage in ["dense", "bm25", "rrf", "rerank", "generate", "faithfulness"]:
        assert stage in mermaid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_viz.py -v`
Expected: FAIL with `ImportError` (`build_viz_workflow` not defined).

- [ ] **Step 3: Implement the visualization-only workflow**

Append to `src/ragpipe/workflow.py`:
```python
def build_viz_workflow():
    """Build an Agent Framework Workflow purely for WorkflowViz diagram export.

    The executors are no-op passthroughs whose only purpose is to make the graph
    topology (incl. the conditional loop edge) renderable. Runtime behavior lives
    in run_pipeline().
    """
    from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

    class _Stage(Executor):
        @handler
        async def go(self, msg: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(msg)

    dense = _Stage(id="dense")
    bm25 = _Stage(id="bm25")
    rrf = _Stage(id="rrf")
    rerank = _Stage(id="rerank")
    generate = _Stage(id="generate")
    faithfulness = _Stage(id="faithfulness")
    answer = _Stage(id="answer")

    def low_faithfulness(_msg: str) -> bool:
        return True  # label-only; real decision is in decide_next()

    return (
        WorkflowBuilder(start_executor=dense)
        .add_edge(dense, rrf)
        .add_edge(bm25, rrf)
        .add_edge(rrf, rerank)
        .add_edge(rerank, generate)
        .add_edge(generate, faithfulness)
        .add_edge(faithfulness, rrf, condition=low_faithfulness)
        .add_edge(faithfulness, answer)
        .build()
    )
```

> Confirm `WorkflowViz().to_mermaid()` is the current method name against the
> installed `agent-framework`; the docs show `WorkflowViz(workflow)` with
> mermaid/DOT export. If the start executor must fan to both dense and bm25,
> add a tiny dispatch executor — adjust here only.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_viz.py -v`
Expected: PASS. If the method name differs, fix the one call and re-run.

- [ ] **Step 5: Export the diagram to docs**

Run:
```bash
python -c "from ragpipe.workflow import build_viz_workflow; from agent_framework import WorkflowViz; open('docs/pipeline.mmd','w').write(WorkflowViz(build_viz_workflow()).to_mermaid())"
```
Expected: `docs/pipeline.mmd` created with a mermaid graph.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/workflow.py tests/test_workflow_viz.py docs/pipeline.mmd
git commit -m "feat: WorkflowViz diagram export for the pipeline graph"
```

---

## Task 13: Test set loader with config switch

**Files:**
- Create: `src/ragpipe/eval/__init__.py`
- Create: `src/ragpipe/eval/testset.py`
- Create: `data/testset.jsonl`
- Test: `tests/eval/test_testset.py`

- [ ] **Step 1: Write the failing test**

`tests/eval/__init__.py`: (empty file)

`tests/eval/test_testset.py`:
```python
import json

import pytest

from ragpipe.config import TestsetMode
from ragpipe.eval.testset import TestItem, load_testset


def test_load_handauthored_reads_jsonl(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        json.dumps(
            {"question": "q1", "ground_truth": "a1", "ground_truth_context": "c1"}
        )
        + "\n"
    )

    items = load_testset(TestsetMode.HANDAUTHORED, handauthored_path=str(p))

    assert items == [TestItem(question="q1", ground_truth="a1", ground_truth_context="c1")]


def test_load_synthetic_calls_generator(tmp_path):
    sentinel = [TestItem(question="gen-q", ground_truth="gen-a", ground_truth_context="gen-c")]

    items = load_testset(
        TestsetMode.SYNTHETIC,
        handauthored_path="unused",
        synthetic_fn=lambda: sentinel,
    )

    assert items == sentinel


def test_load_synthetic_without_generator_raises():
    with pytest.raises(ValueError, match="synthetic"):
        load_testset(TestsetMode.SYNTHETIC, handauthored_path="x", synthetic_fn=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_testset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/eval/__init__.py`: (empty file)

`src/ragpipe/eval/testset.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ragpipe.config import TestsetMode


@dataclass(frozen=True)
class TestItem:
    question: str
    ground_truth: str
    ground_truth_context: str


def _load_jsonl(path: str) -> list[TestItem]:
    items: list[TestItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                TestItem(
                    question=row["question"],
                    ground_truth=row["ground_truth"],
                    ground_truth_context=row["ground_truth_context"],
                )
            )
    return items


def load_testset(
    mode: TestsetMode,
    handauthored_path: str = "data/testset.jsonl",
    synthetic_fn: Callable[[], list[TestItem]] | None = None,
) -> list[TestItem]:
    if mode is TestsetMode.HANDAUTHORED:
        return _load_jsonl(handauthored_path)
    if synthetic_fn is None:
        raise ValueError("synthetic mode requires a synthetic_fn generator")
    return synthetic_fn()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_testset.py -v`
Expected: all PASS.

- [ ] **Step 5: Seed the hand-authored test set**

`data/testset.jsonl` (add ~15-25 lines; here are 3 to start — extend over the corpus):
```jsonl
{"question": "What algorithm does Azure AI Search use to merge full-text and vector results in hybrid search?", "ground_truth": "Reciprocal Rank Fusion (RRF).", "ground_truth_context": "https://learn.microsoft.com/azure/search/hybrid-search-overview"}
{"question": "What score does the Azure AI Search semantic ranker produce and what is its range?", "ground_truth": "It produces @search.rerankerScore, ranging from 0.00 to 4.00.", "ground_truth_context": "https://learn.microsoft.com/azure/search/hybrid-search-ranking"}
{"question": "In Microsoft Agent Framework, what decorator marks an executor's message handler in Python?", "ground_truth": "The @handler decorator.", "ground_truth_context": "https://learn.microsoft.com/agent-framework/workflows/executors"}
```

- [ ] **Step 6: Add the synthetic generator factory (manual wiring, no unit test)**

Append to `src/ragpipe/eval/testset.py`:
```python
def build_synthetic_generator(settings, corpus_docs):  # pragma: no cover
    """Return a synthetic_fn that builds a test set from corpus docs via RAGAS."""
    def synthetic_fn() -> list[TestItem]:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_core.documents import Document

        llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_chat_model,
                api_version="2024-10-21",
            )
        )
        emb = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_embedding_model,
                api_version="2024-10-21",
            )
        )
        docs = [Document(page_content=d["content"], metadata={"url": d["url"]}) for d in corpus_docs]
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        dataset = generator.generate_with_langchain_docs(docs, testset_size=15)
        items: list[TestItem] = []
        for row in dataset.to_list():
            items.append(
                TestItem(
                    question=row["user_input"],
                    ground_truth=row.get("reference", ""),
                    ground_truth_context=(row.get("reference_contexts") or [""])[0],
                )
            )
        return items

    return synthetic_fn
```

> Confirm the RAGAS `TestsetGenerator` API and dataset row keys against the
> pinned version on first run; isolate any change to this one function.

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/eval/__init__.py src/ragpipe/eval/testset.py data/testset.jsonl tests/eval/
git commit -m "feat: test set loader with handauthored/synthetic config switch"
```

---

## Task 14: Offline RAGAS harness + per-stage metrics

**Files:**
- Create: `src/ragpipe/eval/harness.py`
- Test: `tests/eval/test_harness.py`

The harness logic — assembling per-item records and aggregating scores — is pure and unit-tested with a fake evaluator; the live RAGAS `evaluate()` call is isolated.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_harness.py`:
```python
import pytest

from ragpipe.eval.harness import EvalRecord, aggregate, run_harness
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


def test_aggregate_means_per_metric():
    records = [
        EvalRecord(question="q1", answer="a1", contexts=["c"], ground_truth="g1",
                   metrics={"faithfulness": 0.8, "answer_relevancy": 0.6}),
        EvalRecord(question="q2", answer="a2", contexts=["c"], ground_truth="g2",
                   metrics={"faithfulness": 0.6, "answer_relevancy": 1.0}),
    ]
    means = aggregate(records)
    assert means["faithfulness"] == pytest.approx(0.7)
    assert means["answer_relevancy"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_run_harness_builds_records_from_pipeline_and_evaluator():
    items = [TestItem(question="q1", ground_truth="g1", ground_truth_context="ctx")]

    async def fake_pipeline(q):
        s = PipelineState(query=q)
        s.answer = "a1"
        s.reranked = [Chunk(id="c", title="t", url="u", content="ctx-content")]
        return s

    async def fake_evaluator(records):
        for r in records:
            r.metrics = {"faithfulness": 0.9}
        return records

    records = await run_harness(items, pipeline_fn=fake_pipeline, evaluator_fn=fake_evaluator)

    assert records[0].answer == "a1"
    assert records[0].contexts == ["ctx-content"]
    assert records[0].metrics["faithfulness"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/eval/harness.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable

from ragpipe.eval.testset import TestItem
from ragpipe.models import PipelineState


@dataclass
class EvalRecord:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    metrics: dict[str, float] = field(default_factory=dict)


PipelineFn = Callable[[str], Awaitable[PipelineState]]
EvaluatorFn = Callable[[list[EvalRecord]], Awaitable[list[EvalRecord]]]


async def run_harness(
    items: list[TestItem],
    pipeline_fn: PipelineFn,
    evaluator_fn: EvaluatorFn,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for item in items:
        state = await pipeline_fn(item.question)
        records.append(
            EvalRecord(
                question=item.question,
                answer=state.answer,
                contexts=[c.content for c in state.reranked],
                ground_truth=item.ground_truth,
            )
        )
    return await evaluator_fn(records)


def aggregate(records: list[EvalRecord]) -> dict[str, float]:
    keys = {k for r in records for k in r.metrics}
    return {
        k: mean([r.metrics[k] for r in records if k in r.metrics]) for k in keys
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_harness.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the real RAGAS evaluator + per-stage variant (manual, no unit test)**

Append to `src/ragpipe/eval/harness.py`:
```python
def build_ragas_evaluator(settings):  # pragma: no cover
    """Return an evaluator_fn that scores records with the full RAGAS suite."""
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from datasets import Dataset
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ds = Dataset.from_list(
            [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts,
                    "ground_truth": r.ground_truth,
                }
                for r in records
            ]
        )
        llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_chat_model,
                api_version="2024-10-21",
            )
        )
        emb = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_embedding_model,
                api_version="2024-10-21",
            )
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=emb,
        )
        df = result.to_pandas()
        for i, r in enumerate(records):
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in df.columns:
                    r.metrics[metric] = float(df.iloc[i][metric])
        return records

    return evaluator_fn
```

> The per-stage context metrics (context precision after RRF vs after rerank,
> spec §5) reuse this evaluator: run it once with `contexts` = fused chunks and
> once with `contexts` = reranked chunks, then compare the `context_precision`
> aggregates. Implement that as a thin wrapper when wiring the dashboard
> (Task 15) — it needs no new logic, only different `contexts` inputs.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/eval/harness.py tests/eval/test_harness.py
git commit -m "feat: offline RAGAS harness with per-metric aggregation"
```

---

## Task 15: Streamlit dashboard

**Files:**
- Create: `app/dashboard.py`
- Test: `tests/test_dashboard_helpers.py`

UI rendering isn't unit-tested, but the data-shaping helpers are. Keep all non-trivial logic in importable helper functions.

- [ ] **Step 1: Write the failing test**

`tests/test_dashboard_helpers.py`:
```python
from ragpipe.models import Chunk, PipelineState
from app.dashboard import stage_rows


def test_stage_rows_summarizes_each_stage():
    state = PipelineState(query="q")
    state.dense = [Chunk(id="a", title="t", url="u", content="x", score=0.5)]
    state.reranked = [Chunk(id="a", title="t", url="u", content="x", score=3.2)]
    state.answer = "final"
    state.faithfulness = 0.81

    rows = stage_rows(state)

    labels = [r["stage"] for r in rows]
    assert "dense" in labels
    assert "reranked" in labels
    faith = next(r for r in rows if r["stage"] == "faithfulness")
    assert faith["detail"] == "0.81"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Write the helper + minimal app**

`app/__init__.py`: (empty file)

`app/dashboard.py`:
```python
from __future__ import annotations

from typing import Any

from ragpipe.models import PipelineState


def stage_rows(state: PipelineState) -> list[dict[str, Any]]:
    """Flatten a PipelineState into table rows for the Run tab."""
    rows: list[dict[str, Any]] = []
    for label, chunks in [
        ("dense", state.dense),
        ("bm25", state.bm25),
        ("fused", state.fused),
        ("reranked", state.reranked),
    ]:
        rows.append(
            {
                "stage": label,
                "detail": ", ".join(f"{c.id}({c.score:.2f})" for c in chunks),
            }
        )
    rows.append({"stage": "answer", "detail": state.answer})
    rows.append(
        {
            "stage": "faithfulness",
            "detail": "n/a" if state.faithfulness is None else f"{state.faithfulness:.2f}",
        }
    )
    return rows


def main() -> None:  # pragma: no cover - UI entry point
    import asyncio

    import streamlit as st

    from ragpipe.config import Settings

    st.title("RAGAS-infused pipeline")
    tab_run, tab_eval, tab_arch = st.tabs(["Run", "Evaluation", "Architecture"])

    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        if st.button("Run") and query:
            from ragpipe.app_wiring import build_pipeline_fn  # Task 15b

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings)
            state = asyncio.run(pipeline_fn(query))
            st.table(stage_rows(state))
            if state.low_confidence:
                st.warning("Low confidence: faithfulness threshold not met after retries.")

    with tab_arch:
        try:
            with open("docs/pipeline.mmd") as f:
                st.code(f.read(), language="mermaid")
        except FileNotFoundError:
            st.info("Run the WorkflowViz export to generate docs/pipeline.mmd.")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/dashboard.py tests/test_dashboard_helpers.py
git commit -m "feat: Streamlit dashboard with stage trace table"
```

---

## Task 15b: Live wiring module

**Files:**
- Create: `src/ragpipe/app_wiring.py`
- Test: `tests/test_app_wiring.py`

Assembles real clients into a `PipelineDeps`/`pipeline_fn`. Construction is testable (it builds deps); the network calls only happen when the returned callable runs against Azure.

- [ ] **Step 1: Write the failing test**

`tests/test_app_wiring.py`:
```python
from ragpipe.app_wiring import make_deps
from ragpipe.config import Settings, TestsetMode


def _settings():
    return Settings(
        foundry_project_endpoint="https://p.services.ai.azure.com",
        foundry_chat_model="gpt-4o",
        foundry_embedding_model="emb",
        search_endpoint="https://s.search.windows.net",
        search_index="idx",
        generator_agent_name="gen",
        testset_mode=TestsetMode.HANDAUTHORED,
    )


def test_make_deps_wires_callables_from_injected_components():
    class FakeDense:
        def retrieve(self, q):
            return []

    class FakeBm25:
        def retrieve(self, q):
            return []

    class FakeReranker:
        def rerank(self, q, fused):
            return []

    class FakeGen:
        async def generate(self, q, chunks):
            return "ans"

    class FakeScorer:
        async def score(self, q, a, c):
            return 0.9

    deps = make_deps(
        _settings(),
        dense=FakeDense(),
        bm25=FakeBm25(),
        reranker=FakeReranker(),
        generator=FakeGen(),
        scorer=FakeScorer(),
    )

    assert deps.threshold == 0.7
    assert deps.max_retries == 2
    assert callable(deps.dense)
    assert deps.dense("q") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_wiring.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/ragpipe/app_wiring.py`:
```python
from __future__ import annotations

from typing import Any, Awaitable, Callable

from ragpipe.config import Settings
from ragpipe.models import PipelineState
from ragpipe.workflow import PipelineDeps, run_pipeline


def make_deps(
    settings: Settings,
    dense: Any,
    bm25: Any,
    reranker: Any,
    generator: Any,
    scorer: Any,
) -> PipelineDeps:
    return PipelineDeps(
        dense=lambda q: dense.retrieve(q),
        bm25=lambda q: bm25.retrieve(q),
        rerank=lambda q, fused: reranker.rerank(q, fused),
        generate=lambda q, chunks: generator.generate(q, chunks),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        rrf_k=settings.rrf_k,
        top_k=settings.top_k,
    )


def build_pipeline_fn(
    settings: Settings,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from agent_framework.foundry import FoundryAgent, FoundryEmbeddingClient

    from ragpipe.generate import Generator
    from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
    from ragpipe.retrieval.bm25 import BM25Retriever
    from ragpipe.retrieval.dense import DenseRetriever
    from ragpipe.retrieval.rerank import SemanticReranker
    from ragpipe.search_index import SEMANTIC_CONFIG_NAME

    cred = DefaultAzureCredential()
    search = SearchClient(settings.search_endpoint, settings.search_index, cred)
    embed_client = FoundryEmbeddingClient()

    import asyncio

    def embed(text: str) -> list[float]:
        result = asyncio.get_event_loop().run_until_complete(
            embed_client.get_embeddings([text])
        )
        return list(result[0].embedding)

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        dense=DenseRetriever(search, embed, settings.top_k),
        bm25=BM25Retriever(search, settings.top_k),
        reranker=SemanticReranker(search, SEMANTIC_CONFIG_NAME, settings.top_k),
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str) -> PipelineState:
        return await run_pipeline(query, deps)

    return pipeline_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/app_wiring.py tests/test_app_wiring.py
git commit -m "feat: live wiring of real Azure/Foundry components into the pipeline"
```

---

## Task 16: Infrastructure (azd + Bicep)

**Files:**
- Create: `azure.yaml`
- Create: `infra/main.bicep`
- Create: `infra/main.parameters.json`

No unit tests; verified by `azd provision` / lint.

- [ ] **Step 1: Write `azure.yaml` with provisioning hooks**

`azure.yaml`:
```yaml
name: ragas-infused-pipeline
metadata:
  template: ragas-infused-pipeline@0.1.0
hooks:
  postprovision:
    shell: sh
    run: |
      python -m ragpipe.ingest
      python scripts/setup_agents.py
```

- [ ] **Step 2: Write the Bicep**

`infra/main.bicep`:
```bicep
@description('Primary location for all resources')
param location string = resourceGroup().location

@description('Base name for resources')
param baseName string = 'ragpipe'

@description('Chat model deployment name')
param chatModel string = 'gpt-4o'

@description('Embedding model deployment name')
param embeddingModel string = 'text-embedding-3-small'

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: '${baseName}-search'
  location: location
  sku: { name: 'standard' } // semantic ranker requires Basic+; Standard recommended
  properties: {
    semanticSearch: 'standard'
    replicaCount: 1
    partitionCount: 1
  }
}

resource foundry 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: '${baseName}-foundry'
  location: location
  kind: 'AIServices'
  sku: { name: 'S0' }
  properties: {
    allowProjectManagement: true
    customSubDomainName: '${baseName}-foundry'
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: foundry
  name: '${baseName}-project'
  location: location
  properties: {}
}

resource chat 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: chatModel }
  }
}

resource embedding 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: embeddingModel
  dependsOn: [chat]
  sku: { name: 'Standard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModel }
  }
}

output FOUNDRY_PROJECT_ENDPOINT string = 'https://${foundry.name}.services.ai.azure.com/api/projects/${project.name}'
output SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output FOUNDRY_CHAT_MODEL string = chatModel
output FOUNDRY_EMBEDDING_MODEL string = embeddingModel
```

- [ ] **Step 3: Write the parameters file**

`infra/main.parameters.json`:
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "baseName": { "value": "ragpipe" },
    "chatModel": { "value": "gpt-4o" },
    "embeddingModel": { "value": "text-embedding-3-small" }
  }
}
```

- [ ] **Step 4: Lint the Bicep**

Run: `az bicep build --file infra/main.bicep`
Expected: compiles to ARM JSON with no errors (warnings about preview API versions are acceptable). Fix any hard errors (resource API mismatches) before continuing.

> Resource API versions and the exact Foundry/project/deployment resource shapes
> evolve quickly. If `az bicep build` reports an unknown type/property, check the
> current Foundry "Set up resources" docs and adjust — these three resource
> blocks are the only place to change.

- [ ] **Step 5: Commit**

```bash
git add azure.yaml infra/main.bicep infra/main.parameters.json
git commit -m "feat: azd + Bicep infrastructure with postprovision hooks"
```

---

## Task 17: README and end-to-end runbook

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# RAGAS-infused RAG pipeline

Observable hybrid-retrieval RAG over Microsoft/Azure docs, built on Microsoft
Agent Framework + Azure AI Foundry + Azure AI Search, evaluated with RAGAS.

See the design spec in `docs/superpowers/specs/` and the diagram in `docs/pipeline.mmd`.

## Prerequisites
- Python 3.11, Azure CLI (`az login`), Azure Developer CLI (`azd`), an Azure subscription.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
azd up                       # provisions Foundry, Search, model deployments; runs ingestion + agent registration
cp .env.example .env         # then fill from azd outputs
```

## Run
```bash
streamlit run app/dashboard.py     # Run / Evaluation / Architecture tabs
```

## Evaluate
```bash
python -m ragpipe.eval.run         # offline RAGAS harness over data/testset.jsonl
```
Set `TESTSET_MODE=synthetic` in `.env` to generate the test set from the corpus instead.

## Test
```bash
pytest -q
```
```

- [ ] **Step 2: Add the eval CLI entry point referenced above**

`src/ragpipe/eval/run.py`:
```python
"""Offline evaluation entry point: python -m ragpipe.eval.run"""
from __future__ import annotations

import asyncio
import json

from ragpipe.app_wiring import build_pipeline_fn
from ragpipe.config import Settings
from ragpipe.eval.harness import aggregate, build_ragas_evaluator, run_harness
from ragpipe.eval.testset import load_testset


def main() -> None:  # pragma: no cover - integration entry point
    settings = Settings.from_env()
    items = load_testset(settings.testset_mode)
    pipeline_fn = build_pipeline_fn(settings)
    evaluator_fn = build_ragas_evaluator(settings)

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
    means = aggregate(records)
    with open("eval_results.json", "w") as f:
        json.dump(
            {"means": means, "records": [r.__dict__ for r in records]}, f, indent=2
        )
    print(json.dumps(means, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 3: Run the full unit suite one final time**

Run: `pytest -q`
Expected: all unit tests PASS (everything except the `# pragma: no cover` live entry points).

- [ ] **Step 4: Commit**

```bash
git add README.md src/ragpipe/eval/run.py
git commit -m "docs: README runbook and offline eval entry point"
```

---

## Coverage check (spec → tasks)

- Hybrid retrieval (dense, BM25): Tasks 6 · RRF: Task 4 · semantic rerank over fused IDs (§4.3): Task 7
- Agent Framework Workflow + WorkflowViz (§4.1, §7): Tasks 12, 12b
- Foundry generator agent + Code Interpreter tool (§4.2): Task 9
- RAGAS online faithfulness guardrail + capped loop, fail-closed (§5, §10): Tasks 10, 11, 12
- RAGAS offline suite + per-stage metrics (§5): Task 14
- Test set config switch handauthored|synthetic (§6): Task 13
- Streamlit dashboard Run/Eval/Architecture (§7): Task 15
- Config module + explicit load_dotenv (§9): Task 2
- Error handling: empty retrieval, capped loop, fail-closed judge (§10): Tasks 11, 12 (empty-retrieval graceful answer is handled by the generator's "say you don't know" instruction + reranker empty-input guard in Task 7)
- azd + Bicep provisioning + agent registration (§8): Tasks 16, 9
- Testing strategy: unit (pure logic), component (mocked Azure/Foundry), live entry points excluded from coverage (§11): every task
```
