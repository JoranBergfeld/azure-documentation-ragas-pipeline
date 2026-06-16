# 0014 — Agentic retrieval as a composable wrapper

**Status:** Accepted (2026-06-16)

## Context

Multi-part queries ("what are the rules on X and how does that interact with Y?") don't
decompose well into a single `retrieve(query, k)` call. An agentic loop could help: plan
sub-queries, retrieve for each, accumulate candidates, stop when coverage looks sufficient.

The question is where to build it. Options are: bake it into each substrate, use Azure
AI Search's built-in agentic retrieval (server-side query planning), or treat it as an
orthogonal wrapper over the `RetrievalSubstrate` interface.

## Decision

1. **Wrapper, not per-substrate.** `AgenticSubstrate(inner: RetrievalSubstrate)`
   implements `RetrievalSubstrate`. It composes over any of the four substrates without
   each one needing its own agentic variant. This is what makes 8 modes cost 4
   substrates + 1 wrapper rather than 8 independent implementations.

2. **Built on Microsoft Agent Framework.** Uses `FoundryAgent` from the `agent-framework`
   dependency already in the stack. The agent's tool is a thin wrapper around
   `inner.retrieve`. The framework handles the plan/act/observe loop; the substrate
   handles retrieval.

3. **Bounded iterations.** The loop stops at `agentic_max_iterations` (config, default 3)
   or on a sufficiency judgment from the agent, whichever comes first. It can never loop
   forever. Each iteration records a stage (`iter_N`) in `PipelineState.stages` so the
   harness can see how many iterations ran.

4. **Gate stays the final arbiter.** The agentic loop is purely a retrieval-side
   amplifier. It returns an accumulated, de-duplicated candidate list to the normal
   rerank, generate, and faithfulness gate tail (ADR-0009). The gate decides whether the
   answer is faithful; the loop has no say in that.

5. **Error handling is additive.** Any iteration or tool failure is caught; the loop
   stops early with whatever it has accumulated so far. A failed iteration doesn't blank
   the response; it just means fewer candidates than the max.

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
- The `iter_N` stages in `PipelineState.stages` give the harness visibility into how
  often the loop actually terminates early vs. hitting the cap. That's useful for
  diagnosing whether the sufficiency judgment is doing anything.
- The wrapper composes with Combined too (modes 7 and 8), so the most expensive mode
  is Combined + agentic: two substrates per iteration, up to 3 iterations.

## Sources

- Microsoft Agent Framework / FoundryAgent (already in the project stack) — cited via
  the spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §6
- Spec: `docs/superpowers/specs/2026-06-16-multi-substrate-retrieval-design.md` §6
  (agentic wrapper), §12 Phase 4 (agentic rolled out across all four substrates)
