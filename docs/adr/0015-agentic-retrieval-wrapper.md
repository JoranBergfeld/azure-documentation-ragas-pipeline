# 0015 — Agentic retrieval as a composable wrapper

**Status:** Accepted (2026-06-16); implementation reconciled (2026-06-19)

## Context

Multi-part queries ("what are the rules on X and how does that interact with Y?") don't
decompose well into a single `retrieve(query, k)` call. An agentic loop could help: plan
sub-queries, retrieve for each, accumulate candidates, and ideally stop once coverage looks
sufficient.

The question is where to build it. Options are: bake it into each substrate, use Azure
AI Search's built-in agentic retrieval (server-side query planning), or treat it as an
orthogonal wrapper over the `RetrievalSubstrate` interface.

## Decision

1. **Wrapper, not per-substrate.** `AgenticSubstrate(*, name, inner, plan_fn, max_iterations=3)`
   implements `RetrievalSubstrate`. It composes over any base substrate without each one
   needing its own agentic variant. This is what makes the four agentic modes
   (`baseline_agentic`, `raptor_sac_agentic`, `graphrag_agentic`, `combined_agentic`) cost
   four reused substrates + one wrapper rather than four bespoke implementations.
   Contextual is deliberately left unwrapped, so there is no `contextual_agentic`.

2. **Single-shot LLM planner, not a FoundryAgent tool-loop.** A planner (`plan()` in
   `app_wiring.py`) issues one LLM chat-completion — via `build_context_complete_fn`, the
   same `foundry_chat_model` generator family used for per-chunk situating context — that
   decomposes the query into 2-4 focused sub-queries (`PLAN_PROMPT`, one per line). The
   wrapper then runs `inner.retrieve` over those sub-queries in a plain Python loop. There
   is no `FoundryAgent`, no retrieval tool, and no agent-driven plan/act/observe cycle:
   planning happens once, up front, and retrieval fans out over its output. See the
   implementation note below for why this diverged from the original Agent Framework design.

3. **Bounded, fixed iterations.** The wrapper runs exactly the first
   `agentic_max_iterations` (config, default 3) planned sub-queries — it slices
   `plan_fn(query)[:max_iterations]` and falls back to `[query]` if planning returns
   nothing. There is no result-inspecting early stop; the bound is purely the iteration
   cap, so it can never loop forever. Each iteration records a stage (`iter_N`) in
   `PipelineState.stages` and the merged set is mirrored under `fused`, so the harness can
   see how many sub-queries actually ran.

4. **Gate stays the final arbiter.** The agentic loop is purely a retrieval-side
   amplifier. It returns an accumulated, de-duplicated candidate list to the normal
   rerank, generate, and faithfulness gate tail (ADR-0009). The gate decides whether the
   answer is faithful; the loop has no say in that.

5. **Planning is fail-soft; retrieval iterations are not wrapped.** The planner retries
   `max_retries + 1` times and returns `[]` on persistent failure, so the wrapper falls
   back to a single retrieve over the original query — a planner outage never blanks the
   response. Note the divergence from the original additive-error intent: per-iteration
   `inner.retrieve` calls are *not* individually caught, so a substrate failure inside the
   loop propagates rather than stopping early with partial candidates. Tightening this is
   a known follow-up.

## Alternatives rejected

- **Baking agentic into each substrate.** Would force a separate implementation per
  substrate. That's 4x the code for the loop logic, with no benefit over the wrapper
  for this project's scope.
- **Azure AI Search built-in agentic retrieval (knowledge agent).** Server-side query
  planning with less code on our side. Rejected because it only works over Azure AI
  Search indexes, so it can't uniformly wrap the GraphRAG substrate (which also uses
  Azure AI Search indexes, but via a different multi-index local/global query pattern).
  Splitting the agentic abstraction into two implementations breaks the "one wrapper
  over a common interface" property. May be worth revisiting if Microsoft's knowledge
  agent evolves to handle multi-index patterns natively.
- **LangGraph or a third-party agent loop.** Another framework dependency, more config.
  The Agent Framework is already in the stack; adding a second orchestrator for a 3-
  iteration loop is not worth it.

## Consequences

- The wrapper is straightforward, but the per-iteration overhead (one `inner.retrieve`
  call per sub-query) multiplies the latency. At `agentic_max_iterations=3` the worst
  case is roughly 3x retrieval latency before the reranker even runs. Keep the default
  low and measure before raising it.
- Each iteration's candidates are de-duplicated before passing to the reranker, so the
  candidate list doesn't balloon. The reranker still runs once over the merged set.
- The `iter_N` stages in `PipelineState.stages` give the harness visibility into how many
  sub-queries the planner produced (capped at `agentic_max_iterations`). Since there is no
  early stop, fewer than `max_iterations` stages means the planner returned fewer
  sub-queries or fell back to the original query — useful for spotting decomposition that
  isn't firing.
- The wrapper composes with Combined too, so the most expensive mode is `combined_agentic`:
  two substrates per iteration, up to 3 iterations.

## Implementation note (2026-06-19)

The original decision (2026-06-16, decision 2) specified a Microsoft Agent Framework
`FoundryAgent` tool-loop with a sufficiency-based early stop (spec §6). The shipped
implementation is intentionally leaner: a single up-front LLM query-decomposition feeds a
fixed, bounded fan-out (`agentic_max_iterations`), with no agent tool-loop and no
result-inspecting early termination. For a 2-4 sub-query plan -> retrieve -> merge, the
agent machinery added orchestration and latency without changing outcomes — the
faithfulness gate (decision 4) stays the real arbiter either way. Decisions 2, 3, and 5
above have been rewritten to match the code; the Agent Framework route is recorded here as
the original intent and a possible future direction if planning ever needs to react to
results.

## Sources

- Implementation: `src/ragpipe/retrieval/agentic.py` (the wrapper + plan -> retrieve loop),
  `src/ragpipe/app_wiring.py` (`plan()` planner + `PLAN_PROMPT`),
  `src/ragpipe/retrieval/registry.py` (`_agentic` factory + the four `*_agentic` registrations)
- Config: `agentic_max_iterations` / `AGENTIC_MAX_ITERATIONS` (default 3) and
  `max_retries` / `MAX_RETRIES` (default 2) in `src/ragpipe/config.py`
- Microsoft Agent Framework / FoundryAgent (original intent, not used by the shipped loop)
  — cited via the spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §6
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §6
  (agentic wrapper), §12 Phase 4 (agentic rolled out across the four wrappable substrates)
