# Multi-substrate retrieval — Phase 1 (seam + Baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the `RetrievalSubstrate` seam, generalize `PipelineState` to named stages, and land the Baseline substrate so the existing contextual mode and a new plain Baseline mode are comparable in the harness, dashboard, and API.

**Architecture:** A substrate owns its own retrieval + fusion and returns a `RetrievalResult` (final candidates + named intermediate stages). `run_pipeline` calls one substrate, then runs the unchanged rerank → generate → faithfulness gate → retry tail. Modes are selected from a registry keyed by mode name. Today's dense+BM25+RRF becomes `HybridSubstrate`, registered as both `contextual` (existing decorated index, default) and `baseline` (new plain index).

**Tech Stack:** Python 3.12, `uv`, `pytest`/`pytest-asyncio`, dataclasses, Azure AI Search (`azure-search-documents`), FastAPI, Streamlit. Run everything via `uv run`.

**Conventions for this plan:**
- Every test command is `uv run pytest ...`. Lint with `uv run ruff check .`.
- This repo uses `from __future__ import annotations` at the top of every module — keep it.
- Commit after each task with the message shown. We are on branch `feat/multi-substrate-retrieval`.
- No live Azure calls in tests. All Azure clients are faked. Live wiring stays under `# pragma: no cover`.

---

## File structure (created/modified in Phase 1)

- Create `src/ragpipe/retrieval/substrate.py` — `RetrievalResult`, `RetrievalSubstrate` Protocol, `HybridSubstrate`.
- Create `src/ragpipe/retrieval/registry.py` — mode enum + substrate registry/factory.
- Create `tests/retrieval/test_substrate.py`, `tests/retrieval/test_registry.py`.
- Modify `src/ragpipe/models.py` — generalized `PipelineState` (stages dict + candidates).
- Modify `src/ragpipe/workflow.py` — `PipelineDeps.retrieve` replaces dense/bm25; `run_pipeline` consumes a substrate result.
- Modify `src/ragpipe/app_wiring.py` — `make_deps`/`build_pipeline_fn` take a mode.
- Modify `src/ragpipe/config.py` — mode + per-substrate index names.
- Modify `src/ragpipe/eval/harness.py` — read stages dynamically; `aggregate_by_mode`.
- Modify `src/ragpipe/eval/run.py` — run the testset through each mode; `eval_results.json` keyed by mode.
- Modify `app/dashboard.py` — read stages dynamically.
- Modify `app/api.py` — `mode` on `/run`; new `/compare`.
- Modify `src/ragpipe/ingest.py` + `src/ragpipe/search_index.py` — `build_baseline` path + plain-text index variant.
- Create ADRs `docs/adr/0011`..`0015`.

---

## Task 1: ADRs for the five load-bearing decisions

**Files:**
- Create: `docs/adr/0011-retrieval-substrate-seam.md`
- Create: `docs/adr/0012-raptor-collapsed-tree-on-azure-search.md`
- Create: `docs/adr/0013-flat-graphrag-on-azure-search.md`
- Create: `docs/adr/0014-agentic-retrieval-wrapper.md`
- Create: `docs/adr/0015-multi-mode-evaluation-axis.md`

- [ ] **Step 1: Write the five ADRs** using the repo's Nygard + mandatory Sources format (see `docs/adr/README.md` and any existing ADR for the exact section layout: Status / Context / Decision / Consequences / Sources). Content for each is the corresponding numbered subsection of `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` (§13 lists them; the design body §1–§9 and Rejected alternatives supply Context/Decision/Consequences; the spec's Sources section supplies citations). Mark all five `Status: Accepted`.

- [ ] **Step 2: Commit**

```bash
git add docs/adr/0011-retrieval-substrate-seam.md docs/adr/0012-raptor-collapsed-tree-on-azure-search.md docs/adr/0013-flat-graphrag-on-azure-search.md docs/adr/0014-agentic-retrieval-wrapper.md docs/adr/0015-multi-mode-evaluation-axis.md
git commit -m "docs(adr): 0011-0015 for multi-substrate retrieval"
```

---

## Task 2: `RetrievalResult` + `RetrievalSubstrate` protocol

**Files:**
- Create: `src/ragpipe/retrieval/substrate.py`
- Test: `tests/retrieval/test_substrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_substrate.py
from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import RetrievalResult, RetrievalSubstrate


def test_retrieval_result_defaults_empty_stages():
    r = RetrievalResult(candidates=[Chunk(id="1", title="t", url="u", content="c")])
    assert r.stages == {}
    assert [c.id for c in r.candidates] == ["1"]


def test_protocol_is_runtime_checkable():
    class Dummy:
        name = "dummy"

        async def retrieve(self, query: str, k: int) -> RetrievalResult:
            return RetrievalResult(candidates=[])

    assert isinstance(Dummy(), RetrievalSubstrate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_substrate.py -v`
Expected: FAIL with `ModuleNotFoundError: ragpipe.retrieval.substrate`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragpipe/retrieval/substrate.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragpipe.models import Chunk


@dataclass
class RetrievalResult:
    """What a substrate returns: the final candidate list fed to rerank, plus
    named intermediate stages captured for the dashboard and eval (e.g. dense,
    bm25, fused). The substrate owns its own fusion; the pipeline does not."""

    candidates: list[Chunk]
    stages: dict[str, list[Chunk]] = field(default_factory=dict)


@runtime_checkable
class RetrievalSubstrate(Protocol):
    name: str

    async def retrieve(self, query: str, k: int) -> RetrievalResult: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_substrate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/substrate.py tests/retrieval/test_substrate.py
git commit -m "feat(retrieval): RetrievalSubstrate seam (result + protocol)"
```

---

## Task 3: Generalize `PipelineState` to named stages

`PipelineState` currently hard-codes `dense/bm25/fused/reranked`. Replace the substrate-produced legs with an ordered `stages` dict and a `candidates` field (the substrate's final output fed to rerank). Keep `reranked` as a real field (it changes every retry and the gate/generator consume it) and mirror it into `stages["reranked"]`.

**Files:**
- Modify: `src/ragpipe/models.py`
- Test: `tests/test_models.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from __future__ import annotations

from ragpipe.models import Chunk, PipelineState


def test_set_stage_records_and_orders():
    s = PipelineState(query="q")
    s.set_stage("dense", [Chunk(id="1", title="", url="", content="")])
    s.set_stage("bm25", [Chunk(id="2", title="", url="", content="")])
    assert list(s.stages.keys()) == ["dense", "bm25"]
    assert [c.id for c in s.stages["dense"]] == ["1"]


def test_set_reranked_mirrors_into_stages():
    s = PipelineState(query="q")
    s.set_reranked([Chunk(id="9", title="", url="", content="")])
    assert [c.id for c in s.reranked] == ["9"]
    assert [c.id for c in s.stages["reranked"]] == ["9"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `AttributeError: 'PipelineState' object has no attribute 'set_stage'`.

- [ ] **Step 3: Write minimal implementation** — replace the `PipelineState` dataclass body (keep `Chunk`, `TraceEvent` unchanged):

```python
@dataclass
class PipelineState:
    query: str
    # Named intermediate retrieval stages captured for the dashboard and eval.
    # Substrates fill these (e.g. dense/bm25/fused, or local/global/fused). The
    # final reranked set is mirrored in here under "reranked".
    stages: dict[str, list[Chunk]] = field(default_factory=dict)
    # The substrate's final candidate list, fed to the reranker each attempt.
    candidates: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    answer: str = ""
    faithfulness: float | None = None
    attempt: int = 0
    low_confidence: bool = False
    # Directive guardrail (ADR-0009): when retries exhaust, the answer is
    # replaced with a fixed abstention and this flag is set. The suppressed
    # answer survives in the trace only.
    abstained: bool = False
    trace: list[TraceEvent] = field(default_factory=list)

    def set_stage(self, name: str, chunks: list[Chunk]) -> None:
        self.stages[name] = chunks

    def set_reranked(self, chunks: list[Chunk]) -> None:
        self.reranked = chunks
        self.stages["reranked"] = chunks

    def add_trace(self, stage: str, data: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(stage=stage, data=data))

    def next_attempt(self) -> None:
        self.attempt += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (2 passed). Other tests will break until Task 6 — that is expected; do not fix them here.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/models.py tests/test_models.py
git commit -m "feat(models): PipelineState named stages + candidates"
```

---

## Task 4: `HybridSubstrate` (dense + BM25 + RRF, substrate-owned)

Wrap the existing `DenseRetriever`/`BM25Retriever`/`reciprocal_rank_fusion` behind the substrate interface. The retrievers stay sync; the substrate exposes the async `retrieve`. It records `dense`, `bm25`, `fused` stages and returns `fused` as candidates.

**Files:**
- Modify: `src/ragpipe/retrieval/substrate.py`
- Test: `tests/retrieval/test_substrate.py`

- [ ] **Step 1: Write the failing test** (append to the file)

```python
from ragpipe.retrieval.substrate import HybridSubstrate


class _FakeLeg:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query):  # sync, like DenseRetriever/BM25Retriever
        return self._chunks


@pytest.mark.asyncio
async def test_hybrid_substrate_fuses_and_records_stages():
    dense = _FakeLeg([Chunk(id="a", title="", url="", content="x")])
    bm25 = _FakeLeg([Chunk(id="b", title="", url="", content="y")])
    sub = HybridSubstrate(name="baseline", dense=dense, bm25=bm25, rrf_k=60)

    result = await sub.retrieve("q", k=10)

    assert sub.name == "baseline"
    assert set(result.stages) == {"dense", "bm25", "fused"}
    assert [c.id for c in result.stages["dense"]] == ["a"]
    assert {c.id for c in result.candidates} == {"a", "b"}  # fused union
    assert result.candidates is result.stages["fused"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_substrate.py -v`
Expected: FAIL with `ImportError: cannot import name 'HybridSubstrate'`.

- [ ] **Step 3: Write minimal implementation** (append to `substrate.py`)

```python
from typing import Callable

from ragpipe.retrieval.rrf import reciprocal_rank_fusion


class HybridSubstrate:
    """Dense + BM25 hybrid with RRF fusion — the original pipeline topology,
    now owned by the substrate. `dense` and `bm25` are objects with a sync
    `.retrieve(query) -> list[Chunk]` (DenseRetriever / BM25Retriever)."""

    def __init__(self, *, name: str, dense, bm25, rrf_k: int = 60) -> None:
        self.name = name
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> "RetrievalResult":
        dense = self._dense.retrieve(query)
        bm25 = self._bm25.retrieve(query)
        fused = reciprocal_rank_fusion(dense, bm25, k=self._rrf_k)
        return RetrievalResult(
            candidates=fused,
            stages={"dense": dense, "bm25": bm25, "fused": fused},
        )
```

Note: `k` is accepted for interface uniformity; the candidate pool is set by the leg objects' own `top_k` (constructed in wiring), matching today's behavior. Do not slice here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_substrate.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/substrate.py tests/retrieval/test_substrate.py
git commit -m "feat(retrieval): HybridSubstrate (dense+bm25+rrf behind the seam)"
```

---

## Task 5: Config — mode + per-substrate index names

Add the mode selector and per-substrate index names. Phase 1 only needs `contextual` (default, the existing `search_index`) and `baseline`. Later phases add the RAPTOR/graph index names; declare them now with sensible defaults so config is stable.

**Files:**
- Modify: `src/ragpipe/config.py`
- Test: `tests/test_config.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from __future__ import annotations

from ragpipe.config import RetrievalMode


def test_retrieval_mode_values():
    assert RetrievalMode.CONTEXTUAL.value == "contextual"
    assert RetrievalMode.BASELINE.value == "baseline"
    # the 8 spec modes exist as names even if not all wired yet
    assert {"baseline", "baseline_agentic", "raptor_sac", "raptor_sac_agentic",
            "graphrag", "graphrag_agentic", "combined", "combined_agentic"} <= {
        m.value for m in RetrievalMode}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'RetrievalMode'`.

- [ ] **Step 3: Write minimal implementation** — add the enum near `TestsetMode`:

```python
class RetrievalMode(str, Enum):
    # Phase 1: the existing decorated index (default) and the new plain index.
    CONTEXTUAL = "contextual"
    BASELINE = "baseline"
    BASELINE_AGENTIC = "baseline_agentic"
    # Later phases register substrates for these; declared up front for a stable
    # API surface and config.
    RAPTOR_SAC = "raptor_sac"
    RAPTOR_SAC_AGENTIC = "raptor_sac_agentic"
    GRAPHRAG = "graphrag"
    GRAPHRAG_AGENTIC = "graphrag_agentic"
    COMBINED = "combined"
    COMBINED_AGENTIC = "combined_agentic"
```

Then add these fields to `Settings` (after `candidate_pool`), with defaults:

```python
    # Per-substrate index names. `search_index` stays the default contextual index.
    baseline_index: str = "baseline"
    raptor_sac_index: str = "raptor-sac"
    graph_entities_index: str = "graph-entities"
    graph_relationships_index: str = "graph-relationships"
    graph_communities_index: str = "graph-communities"
    # Default mode for surfaces that don't specify one.
    default_mode: RetrievalMode = RetrievalMode.CONTEXTUAL
    # Bounds for later phases (declared now so config is stable).
    agentic_max_iterations: int = 3
    raptor_max_levels: int = 3
    graph_community_levels: int = 1
```

And in `from_env`, append (before the closing `)`):

```python
            baseline_index=os.environ.get("BASELINE_INDEX", "baseline"),
            raptor_sac_index=os.environ.get("RAPTOR_SAC_INDEX", "raptor-sac"),
            graph_entities_index=os.environ.get("GRAPH_ENTITIES_INDEX", "graph-entities"),
            graph_relationships_index=os.environ.get("GRAPH_RELATIONSHIPS_INDEX", "graph-relationships"),
            graph_communities_index=os.environ.get("GRAPH_COMMUNITIES_INDEX", "graph-communities"),
            default_mode=RetrievalMode(os.environ.get("DEFAULT_MODE", "contextual")),
            agentic_max_iterations=int(os.environ.get("AGENTIC_MAX_ITERATIONS", "3")),
            raptor_max_levels=int(os.environ.get("RAPTOR_MAX_LEVELS", "3")),
            graph_community_levels=int(os.environ.get("GRAPH_COMMUNITY_LEVELS", "1")),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/config.py tests/test_config.py
git commit -m "feat(config): RetrievalMode + per-substrate index names"
```

---

## Task 6: Refactor `run_pipeline` to consume a substrate

Replace `PipelineDeps.dense`/`bm25` with a single `retrieve` coroutine returning `RetrievalResult`. The fixed RRF call in `run_pipeline` goes away (substrate owns fusion). The rerank → generate → gate loop is unchanged except it reranks `state.candidates` and uses `state.set_reranked(...)`.

**Files:**
- Modify: `src/ragpipe/workflow.py`
- Test: `tests/test_workflow.py` (exists — update the deps construction)

- [ ] **Step 1: Update/inspect the existing workflow test.** Read `tests/test_workflow.py`. It builds `PipelineDeps(dense=..., bm25=..., rerank=..., generate=..., score=...)`. Rewrite the fixture so deps use a single async `retrieve`:

```python
from ragpipe.retrieval.substrate import RetrievalResult

def _fake_retrieve(chunks):
    async def retrieve(query, k):
        return RetrievalResult(candidates=chunks,
                               stages={"dense": chunks, "bm25": [], "fused": chunks})
    return retrieve
```

and pass `retrieve=_fake_retrieve([...])` to `PipelineDeps`, dropping `dense`/`bm25`. Keep every existing assertion about answer/faithfulness/retry/abstain; add one asserting `state.stages["fused"]` is populated and `state.reranked == state.stages["reranked"]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: FAIL (`PipelineDeps` has no `retrieve`).

- [ ] **Step 3: Implement.** In `workflow.py`:
  - Replace the `DenseFn`/`Bm25Fn` aliases with `RetrieveFn = Callable[[str, int], object]` (returns an awaitable `RetrievalResult`).
  - In `PipelineDeps`, replace `dense`/`bm25` with `retrieve: RetrieveFn`. Add `candidate_pool: int = 15`.
  - Rewrite the retrieval section of `run_pipeline`:

```python
    from ragpipe.retrieval.substrate import RetrievalResult  # local import avoids cycle

    result: RetrievalResult = await _maybe_await(deps.retrieve(query, deps.candidate_pool))
    for name, chunks in result.stages.items():
        state.set_stage(name, chunks)
        state.add_trace(name, {"ids": [c.id for c in chunks]})
    state.candidates = result.candidates
```

  - In the loop, replace `state.reranked = await _maybe_await(deps.rerank(query, state.fused, k))` with:

```python
        reranked = await _maybe_await(deps.rerank(query, state.candidates, k))
        state.set_reranked(reranked)
```

  Leave the rest (generate, score, decide_next, abstain) untouched. Keep `build_viz_workflow` as is for now (it is label-only; stage names still apply for the default hybrid mode).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/workflow.py tests/test_workflow.py
git commit -m "refactor(workflow): run_pipeline consumes a substrate RetrievalResult"
```

---

## Task 7: Substrate registry + mode-aware wiring

A registry maps a `RetrievalMode` to a factory that builds a `RetrievalSubstrate` from `Settings` + shared clients. Phase 1 registers `contextual` and `baseline` (both `HybridSubstrate`, different index). `make_deps`/`build_pipeline_fn` take a `mode`.

**Files:**
- Create: `src/ragpipe/retrieval/registry.py`
- Modify: `src/ragpipe/app_wiring.py`
- Test: `tests/retrieval/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_registry.py
from __future__ import annotations

import pytest

from ragpipe.config import RetrievalMode
from ragpipe.retrieval.registry import registered_modes, build_substrate
from ragpipe.retrieval.substrate import RetrievalSubstrate


def test_phase1_modes_registered():
    modes = registered_modes()
    assert RetrievalMode.CONTEXTUAL in modes
    assert RetrievalMode.BASELINE in modes


def test_build_substrate_returns_substrate(monkeypatch):
    # build_substrate needs a search client + embed fn; pass fakes via ctx.
    class _Ctx:
        def search_client(self, index): return object()
        def embed(self, text): return [0.0]
    sub = build_substrate(RetrievalMode.BASELINE, settings=_FakeSettings(), ctx=_Ctx())
    assert isinstance(sub, RetrievalSubstrate)
    assert sub.name == "baseline"


class _FakeSettings:
    search_index = "ms-docs"
    baseline_index = "baseline"
    candidate_pool = 15
    rrf_k = 60
    top_k = 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: ragpipe.retrieval.registry`).

- [ ] **Step 3: Implement registry.** The registry decouples "which index/legs" from live Azure construction via a small `ctx` providing `search_client(index)` and `embed(text)`:

```python
# src/ragpipe/retrieval/registry.py
from __future__ import annotations

from typing import Callable, Protocol

from ragpipe.config import RetrievalMode, Settings
from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever
from ragpipe.retrieval.substrate import HybridSubstrate, RetrievalSubstrate


class SubstrateCtx(Protocol):
    def search_client(self, index: str): ...
    def embed(self, text: str) -> list[float]: ...


def _hybrid(index_attr: str, name: str):
    def factory(settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
        index = getattr(settings, index_attr)
        client = ctx.search_client(index)
        return HybridSubstrate(
            name=name,
            dense=DenseRetriever(client, ctx.embed, settings.candidate_pool),
            bm25=BM25Retriever(client, settings.candidate_pool),
            rrf_k=settings.rrf_k,
        )
    return factory


_REGISTRY: dict[RetrievalMode, Callable[[Settings, SubstrateCtx], RetrievalSubstrate]] = {
    RetrievalMode.CONTEXTUAL: _hybrid("search_index", "contextual"),
    RetrievalMode.BASELINE: _hybrid("baseline_index", "baseline"),
}


def registered_modes() -> list[RetrievalMode]:
    return list(_REGISTRY)


def build_substrate(mode: RetrievalMode, settings: Settings, ctx: SubstrateCtx) -> RetrievalSubstrate:
    if mode not in _REGISTRY:
        raise ValueError(f"mode {mode.value!r} is not registered yet (phase not built)")
    return _REGISTRY[mode](settings, ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `app_wiring.py`.** Change `make_deps` to take `retrieve` instead of `dense`/`bm25`:

```python
def make_deps(settings, retrieve, reranker, generator, scorer) -> PipelineDeps:
    return PipelineDeps(
        retrieve=retrieve,
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        rrf_k=settings.rrf_k,
        top_k=settings.top_k,
        candidate_pool=settings.candidate_pool,
    )
```

Change `build_pipeline_fn(settings, mode=None)` to build the ctx + substrate (live wiring, keep `# pragma: no cover`):

```python
def build_pipeline_fn(settings, mode=None):  # pragma: no cover - live wiring
    from functools import lru_cache
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from agent_framework.foundry import FoundryAgent

    from ragpipe.config import RetrievalMode
    from ragpipe.embeddings import build_embed_fn
    from ragpipe.generate import Generator
    from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
    from ragpipe.retrieval.registry import build_substrate
    from ragpipe.retrieval.rerank import SemanticReranker
    from ragpipe.search_index import SEMANTIC_CONFIG_NAME

    mode = mode or settings.default_mode
    cred = DefaultAzureCredential()
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    _clients: dict[str, SearchClient] = {}

    class _Ctx:
        def search_client(self, index: str):
            if index not in _clients:
                _clients[index] = SearchClient(settings.search_endpoint, index, cred)
            return _clients[index]
        def embed(self, text: str):
            return embed(text)

    ctx = _Ctx()
    substrate = build_substrate(RetrievalMode(mode) if isinstance(mode, str) else mode, settings, ctx)

    # The reranker reranks within whatever the substrate returns; it uses the
    # substrate's own index for the hybrid re-score stage.
    rerank_index = getattr(settings, {
        "contextual": "search_index", "baseline": "baseline_index",
    }.get(substrate.name, "search_index"))
    reranker = SemanticReranker(
        ctx.search_client(rerank_index), SEMANTIC_CONFIG_NAME, settings.top_k, embed_fn=embed
    )

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        retrieve=substrate.retrieve,
        reranker=reranker,
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )

    async def pipeline_fn(query: str) -> PipelineState:
        return await run_pipeline(query, deps)

    return pipeline_fn
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: PASS (any tests that constructed `make_deps` with `dense=`/`bm25=` must be updated to `retrieve=`; fix them now).

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/retrieval/registry.py tests/retrieval/test_registry.py src/ragpipe/app_wiring.py tests/
git commit -m "feat(retrieval): mode registry + mode-aware wiring"
```

---

## Task 8: Harness reads stages dynamically + `aggregate_by_mode`

**Files:**
- Modify: `src/ragpipe/eval/harness.py`
- Test: `tests/eval/test_harness.py` (exists — extend)

- [ ] **Step 1: Write the failing test** (append)

```python
import pytest
from ragpipe.eval.harness import run_harness, aggregate_by_mode, EvalRecord
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


@pytest.mark.asyncio
async def test_run_harness_reads_dynamic_stages():
    async def pipeline_fn(q):
        s = PipelineState(query=q)
        s.set_stage("local", [Chunk(id="1", title="", url="http://x", content="c")])
        s.set_reranked([Chunk(id="1", title="", url="http://x", content="c")])
        s.answer = "a"
        return s
    async def evaluator_fn(records):
        return records
    items = [TestItem(question="q", ground_truth="g", ground_truth_context="http://x")]
    recs = await run_harness(items, pipeline_fn, evaluator_fn)
    assert "local" in recs[0].stage_urls
    assert "reranked" in recs[0].stage_urls


def test_aggregate_by_mode():
    a = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g")
    a.metrics["hit_rate@reranked"] = 1.0
    b = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g")
    b.metrics["hit_rate@reranked"] = 0.0
    out = aggregate_by_mode({"baseline": [a], "contextual": [b]})
    assert out["baseline"]["hit_rate@reranked"] == 1.0
    assert out["contextual"]["hit_rate@reranked"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: FAIL (`run_harness` reads `state.dense` → AttributeError; `aggregate_by_mode` undefined).

- [ ] **Step 3: Implement.** In `run_harness`, replace the hard-coded `by_stage` dict with the dynamic stages (reranked already mirrored into `state.stages` by `set_reranked`):

```python
        by_stage = state.stages
```

Delete the `RETRIEVAL_STAGES`-based assumption from `run_harness` (the constant stays for the optional per-stage sweep default). Add:

```python
def aggregate_by_mode(records_by_mode: dict[str, list[EvalRecord]]) -> dict[str, dict[str, float]]:
    """aggregate() per mode. Keys are mode names; values are the per-mode means."""
    return {mode: aggregate(recs) for mode, recs in records_by_mode.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/harness.py tests/eval/test_harness.py
git commit -m "feat(eval): dynamic stage capture + aggregate_by_mode"
```

---

## Task 9: `run.py` runs the testset through each mode

**Files:**
- Modify: `src/ragpipe/eval/run.py`

- [ ] **Step 1: Implement** (integration entry, `# pragma: no cover`). Add a `--modes` CLI arg (comma-separated mode names; default `contextual,baseline`). Loop modes, build a `pipeline_fn` per mode, run the harness per mode, and key the output by mode:

```python
import argparse
from ragpipe.config import RetrievalMode
from ragpipe.eval.harness import aggregate_by_mode

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="contextual,baseline")
    args = parser.parse_args()
    modes = [RetrievalMode(m.strip()) for m in args.modes.split(",") if m.strip()]

    settings = Settings.from_env()
    items = _load_items(settings)  # existing load logic factored into a helper
    evaluator_fn = build_ragas_evaluator(settings)

    results_by_mode: dict[str, dict] = {}
    records_by_mode: dict[str, list] = {}
    for mode in modes:
        print(f"=== mode: {mode.value} ===")
        pipeline_fn = build_pipeline_fn(settings, mode=mode)
        records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
        if settings.per_stage_metrics:
            records = asyncio.run(build_per_stage_context_evaluator(settings)(records))
        records_by_mode[mode.value] = records
        results_by_mode[mode.value] = {
            "means": aggregate(records),
            "means_by_tag": aggregate_by_tag(records),
            "coverage": {k: {"valid": v, "total": t} for k, (v, t) in coverage(records).items()},
            "records": [r.__dict__ for r in records],
        }

    payload = _clean({
        "means_by_mode": aggregate_by_mode(records_by_mode),
        "modes": results_by_mode,
    })
    with open("eval_results.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps({"means_by_mode": payload["means_by_mode"]}, indent=2))
```

Factor the existing testset-loading block (synthetic vs handauthored) into `_load_items(settings)`.

- [ ] **Step 2: Smoke the arg parsing offline**

Run: `uv run python -c "import ragpipe.eval.run"` (import must succeed; no Azure).
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/ragpipe/eval/run.py
git commit -m "feat(eval): run.py iterates modes, eval_results keyed by mode"
```

Note for morning: `eval_results.json` shape changed (now `means_by_mode` + `modes`). `/eval` and the dashboard's `eval_rows`/`per_stage_chart_data` read the old shape; Task 11 updates the API read path. Record this in the decision log.

---

## Task 10: Dashboard reads stages dynamically

**Files:**
- Modify: `app/dashboard.py`
- Test: `tests/test_dashboard.py` (create if absent; if dashboard has no unit tests, add a minimal one)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py
from __future__ import annotations

from app.dashboard import stage_chunk_tables
from ragpipe.models import Chunk, PipelineState


def test_stage_chunk_tables_uses_dynamic_stages():
    s = PipelineState(query="q")
    s.set_stage("local", [Chunk(id="1", title="t", url="u", content="hello")])
    s.set_reranked([Chunk(id="1", title="t", url="u", content="hello")])
    tables = stage_chunk_tables(s)
    assert "local" in tables
    assert "reranked" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: FAIL (`stage_chunk_tables` reads `state.dense`).

- [ ] **Step 3: Implement.** Replace the hard-coded `_RETRIEVAL_STAGES` mapping in `stage_chunk_tables` (lines ~29-43) and the per-stage chart helper (lines ~60-63) to iterate `state.stages.items()` instead of the four named attributes. Keep the row-building/formatting logic unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): render stages dynamically"
```

---

## Task 11: API — `mode` on `/run`, new `/compare`, mode-keyed `/eval`

**Files:**
- Modify: `app/api.py`
- Test: `tests/test_api.py` (exists — extend; uses FastAPI dependency override)

- [ ] **Step 1: Write the failing test** (append). Use the existing dependency-override pattern in the file:

```python
from fastapi.testclient import TestClient
from app.api import app, get_pipeline_fn_for_mode  # new factory dependency
from ragpipe.models import Chunk, PipelineState


def _fake_state(mode):
    s = PipelineState(query="q")
    s.set_stage("fused", [Chunk(id="1", title="t", url="u", content="c")])
    s.set_reranked([Chunk(id="1", title="t", url="u", content="c")])
    s.answer = f"answer-{mode}"
    s.faithfulness = 0.9
    return s


def test_compare_runs_multiple_modes():
    async def fake_factory(mode):
        async def fn(q):
            return _fake_state(mode)
        return fn
    app.dependency_overrides[get_pipeline_fn_for_mode] = lambda: fake_factory
    client = TestClient(app)
    resp = client.post("/compare", json={"query": "q", "modes": ["baseline", "contextual"]})
    app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert {r["mode"] for r in body["results"]} == {"baseline", "contextual"}
    assert all("answer" in r for r in body["results"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL (`get_pipeline_fn_for_mode` / `/compare` missing).

- [ ] **Step 3: Implement.** Replace the single cached `_pipeline_fn` with a per-mode cache and a factory dependency, add `mode` to `RunRequest`, and add `/compare`:

```python
from ragpipe.config import RetrievalMode, Settings

_pipeline_fns: dict[str, Callable[[str], Awaitable[PipelineState]]] = {}


def get_pipeline_fn_for_mode():
    async def factory(mode: str):
        if mode not in _pipeline_fns:
            from ragpipe.app_wiring import build_pipeline_fn
            _pipeline_fns[mode] = build_pipeline_fn(Settings.from_env(), mode=RetrievalMode(mode))
        return _pipeline_fns[mode]
    return factory


class RunRequest(BaseModel):
    query: str
    mode: str = "contextual"


class CompareRequest(BaseModel):
    query: str
    modes: list[str]


@app.post("/run")
async def run(req: RunRequest, factory=Depends(get_pipeline_fn_for_mode)) -> dict[str, Any]:
    pipeline_fn = await factory(req.mode)
    state = await pipeline_fn(req.query)
    return _state_payload(req.mode, state)


@app.post("/compare")
async def compare(req: CompareRequest, factory=Depends(get_pipeline_fn_for_mode)) -> dict[str, Any]:
    results = []
    for mode in req.modes:
        pipeline_fn = await factory(mode)
        results.append(_state_payload(mode, await pipeline_fn(req.query)))
    return {"query": req.query, "results": results}


def _state_payload(mode: str, state: PipelineState) -> dict[str, Any]:
    return {
        "mode": mode,
        "query": state.query,
        "answer": state.answer,
        "faithfulness": state.faithfulness,
        "attempt": state.attempt,
        "lowConfidence": state.low_confidence,
        "abstained": state.abstained,
        "stages": stage_chunk_tables(state),
    }
```

Update `/eval` to read the new mode-keyed `eval_results.json`: if `means_by_mode` is present, return `{"meansByMode": results["means_by_mode"], "modes": list(results.get("modes", {}))}`; otherwise fall back to the old shape so a stale file still renders.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "feat(api): mode on /run, /compare endpoint, mode-keyed /eval"
```

---

## Task 12: Baseline ingest path + plain-text index

The Baseline substrate needs a `baseline` index of plain chunks (no contextual decoration). Reuse the chunker; skip context generation; embed raw `content`.

**Files:**
- Modify: `src/ragpipe/search_index.py` (a builder that omits the `context` field contribution)
- Modify: `src/ragpipe/ingest.py` (`build_baseline` build path)
- Test: `tests/test_ingest_baseline.py`

- [ ] **Step 1: Read `ingest.py` and `search_index.py`** to learn `build_documents`, `chunk_markdown`, `build_index`, and the upload/prune helpers. The baseline path mirrors `build_documents` but: no `ContextGenerator` call, `context` field empty/absent, embedding input is `content` only.

- [ ] **Step 2: Write the failing test** (pure-function level, fake embedder, no Azure):

```python
# tests/test_ingest_baseline.py
from __future__ import annotations

from ragpipe.ingest import build_baseline_documents


def test_baseline_documents_have_no_context_and_embed_raw_content():
    pages = [{"title": "T", "url": "http://x", "markdown": "# H\n\nbody text here"}]
    docs = build_baseline_documents(pages, embed_fn=lambda s: [float(len(s))])
    assert docs, "expected at least one chunk"
    d = docs[0]
    assert d.get("context", "") == ""              # no decoration
    assert d["content_vector"] == [float(len(d["content"]))]  # embeds raw content
    assert d["url"] == "http://x"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest_baseline.py -v`
Expected: FAIL (`build_baseline_documents` undefined).

- [ ] **Step 4: Implement `build_baseline_documents`** in `ingest.py`, reusing the existing chunker and the existing doc-shaping helper minus decoration. Match the field names the existing `build_documents` emits (`id`, `title`, `url`, `chunk_id`, `content`, `context`, `content_vector`), with `context=""` and `content_vector = embed_fn(content)`. Add a `build_baseline(settings, limit=None)` live driver (`# pragma: no cover`) mirroring `main()` but targeting `settings.baseline_index` via a `build_index` variant that does not add `context` to the semantic/BM25 config (add a `include_context: bool = True` param to the index builder; baseline passes `False`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest_baseline.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + lint**

Run: `uv run pytest tests/ -q && uv run ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/ingest.py src/ragpipe/search_index.py tests/test_ingest_baseline.py
git commit -m "feat(ingest): baseline (plain-chunk) build path + index variant"
```

---

## Phase 1 done-when

- `uv run pytest tests/ -q` green; `uv run ruff check .` clean.
- `contextual` and `baseline` modes both resolve through the registry and run end to end against fakes.
- Harness, dashboard, and API all read stages dynamically; `/compare` returns per-mode results.
- ADRs 0011-0015 committed.
- Live ingest of the `baseline` index and a live multi-mode eval are NOT run here (need Azure creds + cost); they are a morning checklist item. Document in the decision log.

## Self-review notes (for the author before execution)

- Spec coverage: §1 seam → T2/T6/T7; §2 Baseline substrate → T4/T7/T12; §5 generalized state → T3/T6; §7 config → T5; §8 eval mode axis → T8/T9; §9 surfaces → T10/T11; §13 ADRs → T1. RAPTOR/GraphRAG/Combined/Agentic (§2-§6 remainder) are Phases 2-4, separate plans.
- Type consistency: `RetrievalResult(candidates, stages)`, `RetrievalSubstrate.retrieve(query, k)`, `PipelineDeps.retrieve`, `state.set_stage`/`set_reranked`, `build_substrate(mode, settings, ctx)`, `aggregate_by_mode(dict)` used consistently across tasks.
