# 0020 — Per-run usage reporting

**Status:** Accepted (2026-10-05)

## Context

The pipeline sits behind the website's Spring backend, which is moving from run counts to one
money budget across every paid call the site makes (personal-website issue #23, "The ledger").
The backend prices a call as reported usage times a dated rate keyed by deployment. It can only
see spend that happens in this service if the service reports it, and until now a run recorded
no usage at all.

## Decision

Every run reports one `UsageEntry` per metered call, in call order, on `PipelineState.usage` and
as `usage` in the `/run`, `/run/stream` (result frame) and `/compare` payloads.

- Token calls (query embedding, agentic plan, generation, each faithfulness judge LLM call) carry
  `inputTokens`, `cachedInputTokens` and `outputTokens`, read from the provider's response.
  `inputTokens` is the total, cached part included.
- Per-request billing (semantic ranker query, code interpreter session) carries `requestUnits`.
  These have no model deployment, so they report the fixed names
  `azure-ai-search-semantic-ranker` and `foundry-code-interpreter`.
- Nothing is estimated. A call with no reported usage, or a generation or judge call that timed
  out, is recorded with empty counts, `usageMissing: true`, and a line on stderr.
- No prices. The rate table lives with the caller.
- A call that never left the process reports nothing: an embedding served from the `lru_cache`,
  the passthrough reranker, a semantic rerank over zero candidates. A run with no paid call
  reports `[]`.

Entries are recorded at the point the provider response is in hand (`ragpipe.usage.record_*`)
into a list bound per run by a `ContextVar` in `run_pipeline`.

ADR-0017 rejected a `contextvars` sink for progress events in favour of an explicit callback.
Usage is the other way round on purpose. Progress is emitted by the few layers that own a phase.
Metered calls happen at the leaves (an embed function behind a cache, inside a retriever, inside
a substrate, inside an agentic or combined wrapper, and inside RAGAS). Threading a sink through
all of that changes the substrate protocol and every retriever, and a new substrate that forgets
to pass it under-reports spend without failing anything. Recording at the client call cannot be
forgotten by the layers above it.

Rejected: summing to one total per stage (retries and model splits would be lost, and the
caller prices per deployment); estimating tokens with a tokenizer when usage is missing (the
ledger would hold numbers no provider reported).

## Consequences

- Additive. Existing fields and `/eval` are unchanged; a consumer that ignores `usage` keeps
  working.
- A run that raises returns no payload, so its usage is not reported. That is spend the caller
  cannot see (e.g. generation timing out on every attempt).
- The generation deployment is `FOUNDRY_CHAT_MODEL`, the model `scripts/setup_agents.py`
  registers the agent with. An agent version registered against another model would be
  reported under the wrong name.
- The code interpreter entry is one session per generation call whose response contains a code
  interpreter tool call. The provider does not return a session count.
- Cache-write tokens are not split out. No call here sets up a prompt cache; if one ever does,
  add a field rather than folding it into `inputTokens`.
- Ingest and scripts share the embedding and completion clients but run outside a pipeline run,
  where recording is a no-op.
- Standard search queries (dense, BM25, graph lookups) are covered by the fixed Search tier and
  are not reported.

## Sources

- personal-website issue #23, resolution section "The ledger"
- `src/ragpipe/usage.py`, `src/ragpipe/workflow.py`, `app/api.py`
- `src/ragpipe/embeddings.py`, `src/ragpipe/retrieval/rerank.py`, `src/ragpipe/generate.py`,
  `src/ragpipe/guardrail.py`, `src/ragpipe/context_gen.py`
- ADR-0009 (guardrail retries), ADR-0015 (agentic plan call), ADR-0017 (explicit progress sink)
