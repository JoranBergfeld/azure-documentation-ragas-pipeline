# Caller-selected retrieval mode (Run-tab selector + required API mode)

**Date:** 2026-06-18
**Status:** Approved
**Origin:** Follow-on to the multi-substrate work (`2026-06-16-multi-substrate-retrieval-design.md`).
The substrates and per-mode eval results now exist; the live surfaces should let the
*caller* pick the retrieval substrate (and thus the Azure AI Search index) per request,
with no server-side default to fall back on.

## Problem

The retrieval mode is selectable in two of three surfaces but not consistently:

- The HTTP API already takes a mode (`POST /run {"query","mode"}`, plus `/compare`), but
  `RunRequest.mode` defaults to `"contextual"`, so a caller can omit it and silently get
  one substrate.
- `Settings.default_mode` (from `DEFAULT_MODE` in `.env`) is a server-side default that
  `build_pipeline_fn` falls back to when no mode is passed.
- The Streamlit Run tab passes no mode at all, so it's pinned to that env default and
  requires an `.env` edit + restart to switch index.

We want the index choice to be an explicit input from the caller everywhere — the
demonstrator's whole point is comparing substrates — and we want no hidden server default
that masks a missing choice.

## Goals

- Retrieval mode is a required input on every live surface; there is no server-side
  default mode anywhere.
- The Run tab picks the mode/index per query from a dropdown, no restart.
- The dropdown offers every runnable mode, sourced from the registry so it can't drift.
- A `/run` request that omits `mode` is rejected (422), not silently defaulted.

## Non-goals

- No change to the Evaluation tab (it already drills into modes from `eval_results.json`).
- No per-mode index-name display in the UI (possible fast-follow).
- No new global/sidebar state shared across tabs.

## Design

**1. Remove the server-side default (`config.py`).** Delete the `default_mode` field and its
`DEFAULT_MODE` parsing in `Settings.from_env()`. Nothing else reads it (`grep` confirms only
`app_wiring.py:41`). `RetrievalMode` stays — it's defined in this module.

**2. `build_pipeline_fn` requires a mode (`app_wiring.py`).** Change the signature to
`build_pipeline_fn(settings, *, mode)` and drop `mode = mode or settings.default_mode`. Keep
the `str -> RetrievalMode` coercion. All three callers already pass a mode (`api.py`,
`eval/run.py`, and the new dashboard selector), so this is a no-op for them and a clear
`TypeError` for any future caller that forgets.

**3. API mode is required and validated (`api.py`).** Type the request fields as the enum so
FastAPI returns a clean 422 for both missing and invalid values, instead of a 500 from
`RetrievalMode(bad)` deeper in:
- `RunRequest.mode: RetrievalMode` (no default).
- `CompareRequest.modes: list[RetrievalMode]` (already required; typed for consistent
  validation).
The factory and `_state_payload` keep working unchanged — `RetrievalMode` is a `str` enum, so
it serializes to its value (`"baseline"`) in the JSON response and behaves as a `str` cache key.

**4. Run-tab selector (`app/dashboard.py`).** Add a pure, testable helper:
```python
def mode_options() -> list[str]:
    """Runnable retrieval modes in registry order, for the Run-tab selector."""
    return [m.value for m in registered_modes()]
```
In the Run tab, render `st.selectbox("Retrieval index / mode", mode_options(), help=...)` with
a help note that `*_agentic` modes run multiple retrieval rounds and are slower, then call
`build_pipeline_fn(settings, mode=selected)` on **Run**. The selectbox's initial value is the
first option (`contextual`) — a UI presentation default, not a server config default;
`build_pipeline_fn`'s context wires a planner so agentic modes run unchanged
(`app_wiring.py:67-93`). `main()` stays `# pragma: no cover`; the only new logic is
`mode_options`.

## Testing

- `mode_options`: unit-test that it returns all 9 registered modes in registry order (the
  selector can't silently drop a substrate).
- `api.py`: update the two `/run` tests that omit `mode` (`test_run_returns_answer_and_stages`,
  `test_run_reports_abstention`) to send a valid mode; add `test_run_requires_mode` asserting a
  modeless `POST /run` returns 422. Existing `test_run_with_explicit_mode` and
  `test_compare_runs_multiple_modes` already pass modes and stay green.
- No UI/Streamlit test (consistent with `main()` being uncovered and pure helpers being the
  tested seam). No config test references `default_mode`, so its removal needs no test change.

## Consequences

- **Breaking API change.** `POST /run` now requires `mode`; callers that relied on the
  `"contextual"` default (e.g. the website's Spring backend) must send one. This is intended:
  the index is a deliberate choice, not an inherited default. README's API section is updated
  to show `mode` as required.
- `DEFAULT_MODE` is no longer read; it can be dropped from any `.env` (it is not in
  `.env.example`). The earlier Run-tab spec's dependence on it is removed.
- Anyone can A/B retrieval strategies live from one screen, and every API call records exactly
  which substrate produced an answer.
- The Run tab couples to `registered_modes()` as the single source of truth; a newly registered
  substrate appears in the dropdown automatically.
