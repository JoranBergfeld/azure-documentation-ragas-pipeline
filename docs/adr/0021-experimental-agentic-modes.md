# 0021 — Experimental designation for unevaluated agentic modes

**Status:** Accepted (2026-06-23)

## Context

The retrieval registry exposes nine modes, and the API/dashboard validate all of them as
first-class options. Four modes — `baseline_agentic`, `raptor_sac_agentic`,
`graphrag_agentic`, and `combined_agentic` — have no committed
`eval_results_*_agentic.json` files, so they lack the same evidence attached to the
evaluated modes.

ADR-0015 also records that the shipped agentic substrate is a single-shot LLM planner plus
fixed bounded fan-out, not an agent tool-loop with iterative observe/act behavior. The
`agentic` label therefore oversells the mechanism unless the UI/API also state that these
modes are unevaluated. Iterative or planned retrieval can provide real value, especially
for multi-hop workloads, but that value is workload-dependent and must be measured rather
than assumed.

## Decision

Keep all four agentic modes runnable, but surface them as experimental / unevaluated. The
single source of truth is `registry.experimental_modes()` and `registry.is_experimental()`.
API run payloads include an `experimental` boolean, `GET /modes` lists every registered mode
with that flag, and the dashboard labels experimental options and shows a warning when one
is selected.

We are not running or committing paid eval outputs as part of this decision. Evaluating the
agentic wrappers, ideally on the #6 multi-hop/global cohort, remains a future operational
step.

## Consequences

- The exposed surface now matches the available evidence.
- Retrieval behavior does not change; the modes remain selectable and runnable.
- When committed eval coverage exists for a mode, remove it from `_EXPERIMENTAL_MODES`.
- Downstream clients can discover and filter experimental modes through `GET /modes`.

## Sources

- Azure AI Search agentic retrieval:
  https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview
- HippoRAG (multi-hop-specific gains): https://arxiv.org/abs/2405.14831
- LaRA: Benchmarking RAG and Long-Context LLMs:
  https://proceedings.mlr.press/v267/li25dv.html
- RAG-or-Long-Context routing: https://aclanthology.org/2024.emnlp-industry.66/
