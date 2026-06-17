# Multi-substrate retrieval — Phase 4 (Combined + Agentic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Light up the last modes. `CombinedSubstrate` fuses the RAPTOR_SAC and GraphRAG substrates. `AgenticSubstrate` wraps any substrate with a bounded plan→retrieve loop. Together with Phase 1-3 this completes all 8 modes (4 substrates × agentic toggle), plus the legacy `contextual` default.

**Architecture:** Both new substrates are pure composition over existing substrates and the seam. `CombinedSubstrate` calls `build_substrate(RAPTOR_SAC)` + `build_substrate(GRAPHRAG)` and RRF-fuses their candidates. `AgenticSubstrate` takes an `inner` substrate + a `plan_fn(query)->list[str]`, runs `inner.retrieve` over the planned sub-queries (bounded by `agentic_max_iterations`), accumulates+dedupes candidates by id (keeping max score), and records each iteration as a stage. Candidates from combined/graphrag/agentic modes span multiple indexes, so those modes use `PassthroughReranker`.

**Tech Stack:** as Phase 1-3. The seam, registry (`_hybrid`, `_graphrag`, `build_substrate`, `registered_modes`), `RetrievalResult`, `reciprocal_rank_fusion`, `PassthroughReranker`, and `RetrievalMode` (all 9 values incl. the 4 `*_agentic`) already exist on `main`.

**Conventions:** `uv run`; ruff clean; `from __future__ import annotations`; commit per task; branch `feat/multi-substrate-phase4`. No live Azure/LLM in tests — inject fakes. Live wiring `# pragma: no cover`.

---

## File structure (Phase 4)

- Create `src/ragpipe/retrieval/combined.py` — `CombinedSubstrate`.
- Create `src/ragpipe/retrieval/agentic.py` — `AgenticSubstrate`.
- Create `tests/retrieval/test_combined.py`, `tests/retrieval/test_agentic.py`.
- Modify `src/ragpipe/retrieval/registry.py` — `_combined` + agentic factories; register COMBINED + 4 agentic modes.
- Modify `src/ragpipe/app_wiring.py` — generalize reranker selection; add `ctx.complete` planner for agentic modes.
- Modify `tests/retrieval/test_registry.py` — assert all 9 modes registered.

---

## Task 1: CombinedSubstrate + register COMBINED

**Files:** Create `src/ragpipe/retrieval/combined.py`, `tests/retrieval/test_combined.py`; modify `registry.py`, `tests/retrieval/test_registry.py`.

- [ ] **Step 1: failing test** — `tests/retrieval/test_combined.py`

```python
from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.combined import CombinedSubstrate
from ragpipe.retrieval.substrate import RetrievalResult


class _FakeSub:
    def __init__(self, name, chunks):
        self.name = name
        self._chunks = chunks

    async def retrieve(self, query, k):
        return RetrievalResult(candidates=self._chunks, stages={"fused": self._chunks})


@pytest.mark.asyncio
async def test_combined_fuses_both_substrates():
    a = _FakeSub("raptor_sac", [Chunk(id="r1", title="", url="", content="", score=0.9)])
    b = _FakeSub("graphrag", [Chunk(id="g1", title="", url="", content="", score=0.8)])
    sub = CombinedSubstrate(name="combined", substrates=[a, b])
    result = await sub.retrieve("q", k=10)
    assert sub.name == "combined"
    ids = {c.id for c in result.candidates}
    assert ids == {"r1", "g1"}
    # each inner substrate's stages are namespaced and the fused stage exists
    assert "raptor_sac:fused" in result.stages
    assert "graphrag:fused" in result.stages
    assert "fused" in result.stages
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement** `src/ragpipe/retrieval/combined.py`:

```python
from __future__ import annotations

from ragpipe.retrieval.rrf import reciprocal_rank_fusion
from ragpipe.retrieval.substrate import RetrievalResult


class CombinedSubstrate:
    """Fuse the candidate lists of several inner substrates by RRF. Inner stages
    are namespaced (`<inner.name>:<stage>`) and merged so the dashboard/eval can
    still see each leg."""

    def __init__(self, *, name, substrates, rrf_k=60):
        self.name = name
        self._substrates = substrates
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        stages: dict = {}
        candidate_lists = []
        for sub in self._substrates:
            result = await sub.retrieve(query, k)
            candidate_lists.append(result.candidates)
            for name, chunks in result.stages.items():
                stages[f"{sub.name}:{name}"] = chunks
        fused = candidate_lists[0] if candidate_lists else []
        for nxt in candidate_lists[1:]:
            fused = reciprocal_rank_fusion(fused, nxt, k=self._rrf_k)
        stages["fused"] = fused
        return RetrievalResult(candidates=fused, stages=stages)
```

- [ ] **Step 4: confirm pass.** ruff clean.
- [ ] **Step 5: register.** In `registry.py` add a `_combined` factory and register it:

```python
def _combined(settings, ctx):
    raptor = build_substrate(RetrievalMode.RAPTOR_SAC, settings, ctx)
    graph = build_substrate(RetrievalMode.GRAPHRAG, settings, ctx)
    from ragpipe.retrieval.combined import CombinedSubstrate
    return CombinedSubstrate(name="combined", substrates=[raptor, graph], rrf_k=settings.rrf_k)

_REGISTRY[RetrievalMode.COMBINED] = _combined
```

(`build_substrate` is defined in the same module — reference it directly; if it's defined below `_combined`, move `_combined` after it or call via the module global. Keep `_combined` lazy-importing CombinedSubstrate so the module stays import-safe.) Append to `tests/retrieval/test_registry.py`:

```python
def test_combined_mode_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    assert RetrievalMode.COMBINED in registered_modes()
```

- [ ] **Step 6: pass + ruff.** Commit `feat(retrieval): CombinedSubstrate (RAPTOR+Graph fused) + register COMBINED`.

---

## Task 2: AgenticSubstrate

**Files:** Create `src/ragpipe/retrieval/agentic.py`, `tests/retrieval/test_agentic.py`.

- [ ] **Step 1: failing test**

```python
from __future__ import annotations

import pytest

from ragpipe.models import Chunk
from ragpipe.retrieval.agentic import AgenticSubstrate
from ragpipe.retrieval.substrate import RetrievalResult


class _RecordingSub:
    name = "baseline"

    def __init__(self):
        self.queries = []

    async def retrieve(self, query, k):
        self.queries.append(query)
        # return a distinct chunk per sub-query
        return RetrievalResult(
            candidates=[Chunk(id=query, title="", url="", content="", score=1.0)],
            stages={"fused": []},
        )


@pytest.mark.asyncio
async def test_agentic_loops_planned_subqueries_bounded_and_dedupes():
    inner = _RecordingSub()
    sub = AgenticSubstrate(
        name="baseline_agentic", inner=inner,
        plan_fn=lambda q: ["sub a", "sub b", "sub c", "sub d"],
        max_iterations=2,
    )
    result = await sub.retrieve("original", k=5)
    assert sub.name == "baseline_agentic"
    # bounded: only the first 2 planned sub-queries ran
    assert inner.queries == ["sub a", "sub b"]
    ids = {c.id for c in result.candidates}
    assert ids == {"sub a", "sub b"}
    assert "iter_0" in result.stages and "iter_1" in result.stages
    assert "fused" in result.stages


@pytest.mark.asyncio
async def test_agentic_falls_back_to_original_query_when_plan_empty():
    inner = _RecordingSub()
    sub = AgenticSubstrate(name="x_agentic", inner=inner, plan_fn=lambda q: [], max_iterations=3)
    await sub.retrieve("original", k=5)
    assert inner.queries == ["original"]
```

- [ ] **Step 2: confirm fail.**
- [ ] **Step 3: implement** `src/ragpipe/retrieval/agentic.py`:

```python
from __future__ import annotations

from typing import Callable

from ragpipe.models import Chunk
from ragpipe.retrieval.substrate import RetrievalResult


class AgenticSubstrate:
    """Wrap any substrate with a bounded plan->retrieve loop. `plan_fn` decomposes
    the query into sub-queries; we run inner.retrieve over the first
    `max_iterations` of them, accumulate+dedupe candidates by id (keeping the max
    score), and record each iteration as a stage. The faithfulness gate downstream
    stays the final arbiter; this only amplifies retrieval."""

    def __init__(self, *, name, inner, plan_fn: Callable[[str], list[str]], max_iterations: int = 3):
        self.name = name
        self._inner = inner
        self._plan_fn = plan_fn
        self._max_iterations = max_iterations

    async def retrieve(self, query: str, k: int) -> RetrievalResult:
        subqueries = self._plan_fn(query)[: self._max_iterations] or [query]
        accumulated: dict[str, Chunk] = {}
        stages: dict = {}
        for i, sq in enumerate(subqueries):
            result = await self._inner.retrieve(sq, k)
            stages[f"iter_{i}"] = result.candidates
            for c in result.candidates:
                prev = accumulated.get(c.id)
                if prev is None or c.score > prev.score:
                    accumulated[c.id] = c
        candidates = sorted(accumulated.values(), key=lambda c: c.score, reverse=True)
        stages["fused"] = candidates
        return RetrievalResult(candidates=candidates, stages=stages)
```

- [ ] **Step 4: pass + ruff.** Commit `feat(retrieval): AgenticSubstrate (bounded plan->retrieve loop)`.

---

## Task 3: register agentic modes + planner + reranker generalization

**Files:** Modify `src/ragpipe/retrieval/registry.py`, `src/ragpipe/app_wiring.py`, `tests/retrieval/test_registry.py`.

- [ ] **Step 1: failing test** (append to test_registry.py)

```python
def test_all_eight_modes_plus_contextual_registered():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import registered_modes
    modes = set(registered_modes())
    assert modes == set(RetrievalMode)  # every declared mode is registered
```

- [ ] **Step 2: confirm fail** (the 4 `*_agentic` modes aren't registered yet).

- [ ] **Step 3a: registry agentic factories.** The agentic wrapper needs a planner LLM. Expose it via `ctx` (live wiring provides `ctx.complete(prompt) -> str`; see Step 3c). Add a factory builder:

```python
def _agentic(base_mode: RetrievalMode, name: str):
    def factory(settings, ctx):
        inner = build_substrate(base_mode, settings, ctx)
        from ragpipe.retrieval.agentic import AgenticSubstrate
        def plan_fn(query: str) -> list[str]:
            return ctx.plan(query)  # live planner; see app_wiring
        return AgenticSubstrate(name=name, inner=inner, plan_fn=plan_fn,
                                max_iterations=settings.agentic_max_iterations)
    return factory

_REGISTRY[RetrievalMode.BASELINE_AGENTIC] = _agentic(RetrievalMode.BASELINE, "baseline_agentic")
_REGISTRY[RetrievalMode.RAPTOR_SAC_AGENTIC] = _agentic(RetrievalMode.RAPTOR_SAC, "raptor_sac_agentic")
_REGISTRY[RetrievalMode.GRAPHRAG_AGENTIC] = _agentic(RetrievalMode.GRAPHRAG, "graphrag_agentic")
_REGISTRY[RetrievalMode.COMBINED_AGENTIC] = _agentic(RetrievalMode.COMBINED, "combined_agentic")
```

Add `plan(query)` to the `SubstrateCtx` Protocol (returns `list[str]`). `registered_modes()` still must not construct substrates — only `build_substrate` does — so membership test passes without Azure.

- [ ] **Step 3b: app_wiring reranker generalization.** Replace the `if substrate.name == "graphrag"` check with a helper that also covers combined and every agentic mode (their candidates span multiple indexes / accumulate across sub-queries):

```python
    def _uses_passthrough(name: str) -> bool:
        return name in {"graphrag", "combined"} or name.endswith("_agentic")

    if _uses_passthrough(substrate.name):
        from ragpipe.retrieval.passthrough import PassthroughReranker
        reranker = PassthroughReranker(settings.top_k)
    else:
        <existing SemanticReranker block unchanged>
```

- [ ] **Step 3c: app_wiring planner.** Add a `plan(query)` method to the live `_Ctx` class (`# pragma: no cover`). It calls the chat model (reuse the same completion client `build_graph`/`build_raptor` use — read those for the pattern) with a prompt that decomposes a question into 2-4 focused sub-queries, one per line, and returns them as a list (split on newlines, strip, drop empties). Bound it with the existing timeout/retry helper. If the model returns nothing usable, return `[]` (AgenticSubstrate falls back to the original query).

- [ ] **Step 4: full suite + verify all modes build.** Run `uv run pytest tests/ -q` and `uv run ruff check .` — green. Add one more test to test_registry.py that every mode can be *constructed* with fakes that satisfy the ctx (search_client/embed/plan), to catch wiring typos without Azure:

```python
def test_every_mode_builds_with_fake_ctx():
    from ragpipe.config import RetrievalMode
    from ragpipe.retrieval.registry import build_substrate

    class _Settings:
        search_index = "ms-docs"; baseline_index = "baseline"; raptor_sac_index = "raptor-sac"
        graph_entities_index = "ge"; graph_relationships_index = "gr"; graph_communities_index = "gc"
        candidate_pool = 15; rrf_k = 60; top_k = 5; agentic_max_iterations = 3

    class _Ctx:
        def search_client(self, index): return object()
        def embed(self, text): return [0.0]
        def plan(self, query): return [query]

    for mode in RetrievalMode:
        sub = build_substrate(mode, _Settings(), _Ctx())
        assert sub.name
```

(Note: `_graphrag`/combined build adjacency via `build_adjacency(client)` which calls `client.search(...)`. `object()` has no `.search`, so this test would fail for graph modes. To keep the test useful without Azure, give the fake `search_client` a stub returning an object whose `.search(**kwargs)` returns `[]`. Adjust the fake accordingly so construction doesn't hit a real network — the point is to catch factory wiring typos, and an empty-search stub achieves that.)

- [ ] **Step 5: commit** `feat(retrieval): register combined+agentic modes; generalize reranker selection`.

---

## Phase 4 done-when
- `uv run pytest tests/ -q` green; ruff clean.
- `registered_modes()` == every `RetrievalMode` (all 8 modes + contextual).
- `build_substrate` constructs every mode with a fake ctx (no Azure).
- combined/graphrag/`*_agentic` use PassthroughReranker; contextual/baseline/raptor_sac use SemanticReranker.

## Self-review notes
- Spec coverage: design §2 (combined) → T1; §6 (agentic wrapper) → T2/T3; reranker selection generalized in T3. The live planner (ctx.plan) is the only untested live addition.
- Type consistency: `CombinedSubstrate(name, substrates, rrf_k)`, `AgenticSubstrate(name, inner, plan_fn, max_iterations)`, `_agentic(base_mode, name)`, `ctx.plan(query)->list[str]`, `_uses_passthrough(name)` consistent across tasks.
- Decision to log: agentic "sufficiency" is implicit (process planned sub-queries up to the iteration cap) rather than an LLM reflect-and-stop loop — simpler and deterministic for a demo; revisit if needed.
