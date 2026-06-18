# Caller-selected retrieval mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the retrieval mode (and thus the Azure AI Search index) a required, caller-supplied input on every live surface — required+validated in the HTTP API, a dropdown in the Streamlit Run tab — and delete the server-side `default_mode`/`DEFAULT_MODE` fallback.

**Architecture:** Three live surfaces select a substrate via `RetrievalMode`. The API already routes by mode but allows omission (defaults to `"contextual"`); the Run tab passes no mode and relies on `Settings.default_mode`. This plan removes the server default, types the API mode fields as the `RetrievalMode` enum (FastAPI → 422 on missing/invalid), adds a registry-sourced `mode_options()` selector to the Run tab, and makes `build_pipeline_fn(settings, *, mode)` require its mode.

**Tech Stack:** Python 3.11, FastAPI + Pydantic v2, Streamlit, pytest, `uv` runner, ruff. `RetrievalMode(str, Enum)` from `src/ragpipe/config.py`.

---

## File Structure

- `app/api.py` — type `RunRequest.mode` / `CompareRequest.modes` as `RetrievalMode` (required, validated).
- `app/dashboard.py` — add pure `mode_options()` helper; wire a Run-tab `selectbox` that passes `mode=` to `build_pipeline_fn`.
- `src/ragpipe/config.py` — delete the `default_mode` field and its `DEFAULT_MODE` parse in `from_env()`.
- `src/ragpipe/app_wiring.py` — `build_pipeline_fn(settings, *, mode)` (required, keyword-only); drop the `mode or settings.default_mode` fallback.
- `tests/test_api.py` — send `mode` in the two `/run` tests that omit it; add a 422 test.
- `tests/test_dashboard.py` — unit-test `mode_options()`.
- `tests/test_app_wiring.py` — add a signature-contract test that `mode` is required + keyword-only.
- `tests/test_config.py` (or nearest config test module) — assert `Settings` has no `default_mode` field.
- `README.md` — API section shows `mode` as required.

Ordering keeps every commit shippable: Task 1 (API) and Task 2 (`mode_options`) are independent additions; Task 3 lands all the coupled live-wiring removals together (config field + wiring signature + dashboard caller) so no intermediate commit leaves the dashboard broken.

---

### Task 1: Require and validate `mode` in the HTTP API

**Files:**
- Modify: `app/api.py:36-43` (request models)
- Test: `tests/test_api.py:52-68`, `tests/test_api.py:90-105`, plus a new test

- [ ] **Step 1: Update the two `/run` tests that omit `mode`, and add a 422 test**

In `tests/test_api.py`, change the request body on line 58 from:

```python
        res = client.post("/run", json={"query": "what is RRF?"})
```

to:

```python
        res = client.post("/run", json={"query": "what is RRF?", "mode": "contextual"})
```

Change the request body on line 101 from:

```python
        resp = TestClient(api.app).post("/run", json={"query": "x"})
```

to:

```python
        resp = TestClient(api.app).post("/run", json={"query": "x", "mode": "contextual"})
```

Add this new test directly after `test_run_with_explicit_mode` (after line 140):

```python
def test_run_requires_mode(client):
    res = client.post("/run", json={"query": "x"})
    assert res.status_code == 422


def test_run_rejects_invalid_mode(client):
    res = client.post("/run", json={"query": "x", "mode": "not-a-mode"})
    assert res.status_code == 422
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `uv run pytest tests/test_api.py::test_run_requires_mode tests/test_api.py::test_run_rejects_invalid_mode -v`
Expected: FAIL — currently `RunRequest.mode` defaults to `"contextual"` so a missing mode returns 200, and an invalid mode raises `ValueError` deep in the factory → 500 (not 422).

- [ ] **Step 3: Type the API mode fields as the `RetrievalMode` enum**

In `app/api.py`, `RetrievalMode` is already imported (line 16: `from ragpipe.config import RetrievalMode, Settings`). Change lines 36-43 from:

```python
class RunRequest(BaseModel):
    query: str
    mode: str = "contextual"


class CompareRequest(BaseModel):
    query: str
    modes: list[str]
```

to:

```python
class RunRequest(BaseModel):
    query: str
    mode: RetrievalMode


class CompareRequest(BaseModel):
    query: str
    modes: list[RetrievalMode]
```

No other change is needed: `_state_payload(req.mode, state)` and `factory(req.mode)` keep working because `RetrievalMode` is a `str` subclass — it serializes to its value (`"baseline"`) in the JSON response and behaves as a `str` cache key.

- [ ] **Step 4: Run the full API test module to verify everything passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS — including `test_run_requires_mode` (422), `test_run_rejects_invalid_mode` (422), `test_run_with_explicit_mode` (`body["mode"] == "baseline"`), and `test_compare_runs_multiple_modes` (`{"baseline","contextual"}`).

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "feat(api): require and validate retrieval mode on /run and /compare

Type RunRequest.mode and CompareRequest.modes as RetrievalMode so FastAPI
returns 422 for a missing or invalid mode instead of silently defaulting to
contextual (missing) or 500 (invalid).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Add the `mode_options()` selector source

**Files:**
- Modify: `app/dashboard.py:1-7` (imports), add helper near the other pure helpers
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dashboard.py`:

```python
def test_mode_options_lists_all_registered_modes_in_registry_order():
    from app.dashboard import mode_options
    from ragpipe.retrieval.registry import registered_modes

    assert mode_options() == [m.value for m in registered_modes()]
    assert len(mode_options()) == 9
    assert mode_options()[0] == "contextual"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py::test_mode_options_lists_all_registered_modes_in_registry_order -v`
Expected: FAIL with `ImportError: cannot import name 'mode_options' from 'app.dashboard'`.

- [ ] **Step 3: Implement the helper**

In `app/dashboard.py`, add the registry import to the top-level imports (after line 7 `from ragpipe.models import PipelineState`):

```python
from ragpipe.retrieval.registry import registered_modes
```

Then add this helper immediately after the `EVAL_RESULTS_PATH` / `PIPELINE_DIAGRAM_PATH` constants (after line 10), before `chunk_label`:

```python
def mode_options() -> list[str]:
    """Runnable retrieval modes in registry order, for the Run-tab selector."""
    return [m.value for m in registered_modes()]
```

(`registered_modes()` returns every registered `RetrievalMode`; the import is side-effect-free — the registry only builds factory callables, with Azure clients constructed lazily when a factory is invoked.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): add mode_options() registry-sourced selector list

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Remove the server default and wire the Run-tab selector

This task lands the three coupled live-wiring changes in one commit so no intermediate state breaks the dashboard: delete `Settings.default_mode`, make `build_pipeline_fn` require a keyword-only `mode`, and update the Run tab to pass the selected mode.

**Files:**
- Modify: `src/ragpipe/config.py:69-70`, `src/ragpipe/config.py:117`
- Modify: `src/ragpipe/app_wiring.py:23-26`, `src/ragpipe/app_wiring.py:41`
- Modify: `app/dashboard.py:133-141` (Run-tab handler)
- Test: `tests/test_config.py` (contract), `tests/test_app_wiring.py` (contract)

- [ ] **Step 1: Write the failing contract tests**

Add to `tests/test_config.py`:

```python
def test_settings_has_no_default_mode_field():
    import dataclasses

    from ragpipe.config import Settings

    names = {f.name for f in dataclasses.fields(Settings)}
    assert "default_mode" not in names
```

Add to `tests/test_app_wiring.py`:

```python
def test_build_pipeline_fn_requires_keyword_only_mode():
    import inspect

    from ragpipe.app_wiring import build_pipeline_fn

    params = inspect.signature(build_pipeline_fn).parameters
    assert "mode" in params
    mode = params["mode"]
    assert mode.default is inspect.Parameter.empty
    assert mode.kind is inspect.Parameter.KEYWORD_ONLY
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `uv run pytest tests/test_config.py::test_settings_has_no_default_mode_field "tests/test_app_wiring.py::test_build_pipeline_fn_requires_keyword_only_mode" -v`
Expected: FAIL — `default_mode` still exists on `Settings`, and `build_pipeline_fn`'s `mode` is currently `mode=None` (positional-or-keyword with a default).

- [ ] **Step 3: Remove `default_mode` from `Settings`**

In `src/ragpipe/config.py`, delete lines 69-70:

```python
    # Default mode for surfaces that don't specify one.
    default_mode: RetrievalMode = RetrievalMode.CONTEXTUAL
```

and delete line 117 inside `from_env()`:

```python
            default_mode=RetrievalMode(os.environ.get("DEFAULT_MODE", "contextual")),
```

`RetrievalMode` stays — it is defined in this module (class at line 15) and imported by other modules.

- [ ] **Step 4: Make `mode` required and keyword-only in `build_pipeline_fn`**

In `src/ragpipe/app_wiring.py`, change the signature (lines 23-26) from:

```python
def build_pipeline_fn(
    settings: Settings,
    mode=None,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
```

to:

```python
def build_pipeline_fn(
    settings: Settings,
    *,
    mode,
) -> Callable[[str], Awaitable[PipelineState]]:  # pragma: no cover - live wiring
```

Then change line 41 from:

```python
    mode = mode or settings.default_mode
    if isinstance(mode, str):
        mode = RetrievalMode(mode)
```

to:

```python
    if isinstance(mode, str):
        mode = RetrievalMode(mode)
```

(Both existing call sites already pass `mode=` keyword: `app/api.py:30` and `src/ragpipe/eval/run.py:99`.)

- [ ] **Step 5: Wire the Run-tab selector**

In `app/dashboard.py`, change the Run-tab handler (lines 133-141) from:

```python
    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        if st.button("Run", key="run_query") and query:
            from ragpipe.app_wiring import build_pipeline_fn

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings)
```

to:

```python
    with tab_run:
        query = st.text_input("Ask a Microsoft/Azure docs question")
        mode = st.selectbox(
            "Retrieval index / mode",
            mode_options(),
            help="Which substrate/index answers the query. *_agentic modes run "
            "multiple retrieval rounds and are slower.",
        )
        if st.button("Run", key="run_query") and query:
            from ragpipe.app_wiring import build_pipeline_fn

            settings = Settings.from_env()
            pipeline_fn = build_pipeline_fn(settings, mode=mode)
```

- [ ] **Step 6: Run the contract tests, then the full suite + ruff**

Run: `uv run pytest tests/test_config.py::test_settings_has_no_default_mode_field "tests/test_app_wiring.py::test_build_pipeline_fn_requires_keyword_only_mode" -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS (whole suite green; previously 196 passed → now with the added tests).

Run: `uv run ruff check .`
Expected: no errors.

- [ ] **Step 7: Smoke-check imports (live-wiring seams are `# pragma: no cover`)**

Run: `uv run python -c "import app.api, app.dashboard; from ragpipe.config import Settings; import dataclasses; print('default_mode' in {f.name for f in dataclasses.fields(Settings)})"`
Expected: prints `False` and imports cleanly (no `AttributeError` from the removed field).

- [ ] **Step 8: Commit**

```bash
git add src/ragpipe/config.py src/ragpipe/app_wiring.py app/dashboard.py tests/test_config.py tests/test_app_wiring.py
git commit -m "feat: remove server default mode; require caller-supplied mode

Delete Settings.default_mode/DEFAULT_MODE, make build_pipeline_fn(settings, *,
mode) require a keyword-only mode, and add a Run-tab selectbox sourced from
mode_options() so the dashboard picks the index per query.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Update README API docs

**Files:**
- Modify: `README.md` (API/run section)

- [ ] **Step 1: Edit the API endpoint list**

In `README.md`, replace the `POST /run` bullet (lines 75-76):

```markdown
- `POST /run` `{"query": "..."}` → answer, faithfulness, attempt, lowConfidence,
  abstained, and per-stage chunk tables (`stages.{dense,bm25,fused,reranked}`).
```

with:

```markdown
- `POST /run` `{"query": "...", "mode": "contextual"}` → answer, faithfulness, attempt,
  lowConfidence, abstained, and per-stage chunk tables (`stages.{dense,bm25,fused,reranked}`).
  `mode` is **required**; an omitted or unknown mode returns 422. Valid modes: `contextual`,
  `baseline`, `raptor_sac`, `graphrag`, `combined`, and their `*_agentic` variants.
- `POST /compare` `{"query": "...", "modes": ["contextual", "baseline"]}` → the same payload
  per mode under `results`.
```

(There is no `DEFAULT_MODE` reference in `README.md` to remove; `grep -n DEFAULT_MODE README.md` returns nothing.)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mode is now required on POST /run

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Notes for the implementer

- Use the `uv run` prefix for all Python/pytest/ruff commands (repo convention).
- `RetrievalMode` is `class RetrievalMode(str, Enum)`, so enum members compare/hash equal to their `str` values and FastAPI serializes them to `.value`. This is why typing the API fields as the enum keeps `body["mode"] == "baseline"` assertions valid and dict-cache lookups working.
- `main()` in `app/dashboard.py` and `build_pipeline_fn` in `app_wiring.py` are `# pragma: no cover` (live wiring) — they are verified via the suite staying green + the import smoke check, not new execution tests. The pure/tested seams are `mode_options()` (Task 2) and the two contract tests (Task 3).
- Do not re-introduce a default anywhere: the selectbox's first entry (`contextual`) is a UI presentation default only, not a server config default.
