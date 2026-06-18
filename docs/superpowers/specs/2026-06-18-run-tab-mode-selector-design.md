# Run-tab retrieval-mode selector

**Date:** 2026-06-18
**Status:** Approved
**Origin:** Follow-on to the multi-substrate work (`2026-06-16-multi-substrate-retrieval-design.md`).
The substrates and per-mode eval results now exist; this makes the live Run tab able to
switch between them without an env edit + restart.

## Problem

The Streamlit dashboard's Run tab calls `build_pipeline_fn(settings)` with no mode, so the
retrieval substrate (and thus the Azure AI Search index it hits) is fixed by `DEFAULT_MODE`
in `.env`. To try a different index live you must edit `.env` and restart Streamlit. That
makes interactive index comparison clumsy — the one thing this research demonstrator exists
to make easy.

The harness already proves the seam works per mode (ADR-0016, `run.py --modes`); the live UI
just hasn't been given the same knob.

## Goals

- Pick the retrieval mode/index per query from the Run tab, no restart.
- Offer every runnable mode, sourced from the registry so the list can't drift.
- Leave on-load behavior identical to today (default to `DEFAULT_MODE`).

## Non-goals

- No change to the Evaluation tab (it already drills into modes from `eval_results.json`).
- No per-mode index-name display in the UI (possible fast-follow; the mode→index mapping is
  split across hybrid vs. graphrag/combined and not worth centralizing yet).
- No new global/sidebar state shared across tabs.

## Design

One file, `app/dashboard.py`.

1. **Pure helper (testable):**
   ```python
   def mode_options(settings) -> tuple[list[str], int]:
       """Dropdown values (all registered modes, registry order) + the index of the
       env default, falling back to 0 if it isn't registered."""
   ```
   Built from `ragpipe.retrieval.registry.registered_modes()`, so the 5 core substrates
   come first and the 4 `*_agentic` variants follow, automatically tracking the registry.

2. **Run tab wiring:** load `settings` at the top of the tab, render
   `st.selectbox("Retrieval index / mode", values, index=default_idx, help=...)` with a
   help note that `*_agentic` modes run multiple retrieval rounds and are slower, then on
   **Run** call `build_pipeline_fn(settings, mode=selected)`. `build_pipeline_fn` already
   accepts a mode string (`app_wiring.py:42-43`) and its context wires a planner, so the
   agentic modes run unchanged (`app_wiring.py:67-93`).

`main()` stays `# pragma: no cover`; the only new logic lives in `mode_options`.

## Testing

Unit-test `mode_options` with a lightweight stub exposing `default_mode` (a real
`RetrievalMode`), asserting: all 9 registered values are present in registry order; the
env default is the returned selected index. The fallback-to-0 branch is defensive only
(every `RetrievalMode` is currently registered, so it can't trigger via a real `Settings`)
— cover it with a stub whose `default_mode.value` isn't in the list. No UI/Streamlit test
(consistent with `main()` being uncovered and pure helpers like `stage_chunk_tables` being
the tested seam).

## Consequences

- Anyone can A/B retrieval strategies live from one screen — the demonstrator's headline
  interaction.
- `*_agentic` modes are reachable from the UI; they are correct but slower (multiple
  planning + retrieval rounds), flagged via the selectbox help text.
- The Run tab now reads the registry, coupling the UI to `registered_modes()`. That is the
  intended single source of truth; a newly registered substrate appears automatically.
