# Provider-aware online judge (Kimi-K2.5 gate)

**Date:** 2026-06-11
**Status:** Approved

## Problem

The online faithfulness gate (ADR-0009) is hardwired to the Anthropic Messages
route (`<account>.services.ai.azure.com/anthropic`) via `langchain_anthropic.ChatAnthropic`
(RAGAS metric) and the `anthropic.Anthropic` SDK (raw completion). The chosen
gate model, `claude-sonnet-4-6`, has **zero deployable quota** on this
subscription (GlobalStandard = 0 in both swedencentral and eastus2), so the
gate cannot run.

A `Kimi-K2.5` deployment (MoonshotAI) is available and verified working, but it
is served on the **OpenAI-compatible route** (`/openai/deployments/<name>` on
the services host), *not* the `/anthropic` route. So `JUDGE_MODEL=Kimi-K2.5`
fails against the current Anthropic-only wiring.

ADR-0009's requirement is that the gate be a **different model family** than the
gpt generator (to avoid self-preference bias) and independent of the offline
judge (`DeepSeek-V4-Pro`). Kimi-K2.5 (MoonshotAI) satisfies both — it is a third
distinct family. The model choice changes; the independence principle holds.

## Goal

Route the online judge by provider so `Kimi-K2.5` works **now** via the
OpenAI-compatible route, while the Claude/Anthropic path is preserved for when
quota becomes available. Apply the same routing to the gate's RAGAS metric and
to the shared raw-completion function.

Non-goal: changing the offline judge (stays `DeepSeek-V4-Pro`) or the generator.

## Design

### Provider detection

A single helper is the source of truth:

```python
def _judge_provider(model: str) -> str:
    """Anthropic (Claude) deployments use the /anthropic Messages route;
    every other family (Kimi, DeepSeek, ...) uses the OpenAI-compatible route."""
    return "anthropic" if "claude" in model.lower() else "openai"
```

Name-based, zero-config. Claude is the only Anthropic-route family on this
account, so this is unambiguous today. No new env var (YAGNI); an explicit
override can be added later if a non-"claude" Anthropic model ever appears.

### Faithfulness gate — `src/ragpipe/guardrail.py`

`build_ragas_faithfulness(settings)` keeps its empty-`JUDGE_MODEL` guard
(raises `ValueError`), then dispatches on `_judge_provider(settings.judge_model)`:

- `anthropic` -> `_build_claude_faithfulness(settings)` (unchanged).
- `openai` -> **new** `_build_openai_faithfulness(settings)`:
  `LangchainLLMWrapper(AzureChatOpenAI(azure_endpoint=services_endpoint_from_project(...),
  azure_deployment=settings.judge_model, api_version="2024-10-21",
  azure_ad_token_provider=<cognitiveservices scope>))` wrapped in RAGAS
  `Faithfulness`. This mirrors `eval/harness.py::_build_ragas_clients_live`
  (the proven DeepSeek pattern). The `azure_ad_token_provider` auto-refreshes,
  so the per-call rebuild used by the Anthropic path (fixed headers) is not
  needed; build once.

`temperature`: the OpenAI-route judge omits an explicit temperature, matching
the DeepSeek pattern (reasoning deployments may reject sampling overrides).
Verified during implementation; if Kimi accepts `temperature=0`, prefer it for
determinism.

### Raw completion — `src/ragpipe/foundry_judge.py` (renamed from `foundry_claude.py`)

`git mv foundry_claude.py foundry_judge.py` (no longer Claude-only). Rename
`build_claude_complete_fn` -> `build_judge_complete_fn`, provider-aware:

- `anthropic` -> existing `anthropic.Anthropic` Messages path (unchanged).
- `openai` -> `openai.AzureOpenAI(azure_endpoint=services_endpoint_from_project(...),
  azure_ad_token_provider=<cognitiveservices scope>, api_version=...)` then
  `client.chat.completions.create(model=settings.judge_model, ...)`.

Keeps the empty-`JUDGE_MODEL` guard. Internal branches are split into
`_anthropic_complete_fn` / `_openai_complete_fn` so the dispatch is unit-testable
without live calls.

`AI_FOUNDRY_SCOPE` stays in this module; `guardrail.py` updates its import path.

### Callers to update

- `scripts/verify_judges.py` — import `build_judge_complete_fn`.
- `scripts/generate_synthetic_testset.py` — import `build_judge_complete_fn`.

### Config / env

- Set `JUDGE_MODEL=Kimi-K2.5` in the azd env (`azd env set`) and repo `.env`.
- `.env.example`: document that `JUDGE_MODEL` is provider-routed
  (claude* -> Anthropic route; otherwise -> OpenAI-compatible route).

## Components & data flow

```
build_pipeline_fn / verify_judges / synthetic-testset
        |                          |
        v                          v
build_ragas_faithfulness     build_judge_complete_fn
        |  _judge_provider()        |  _judge_provider()
   anthropic / openai          anthropic / openai
        |        \                  |        \
ChatAnthropic   AzureChatOpenAI  Anthropic   AzureOpenAI.chat
(/anthropic)    (services /openai) SDK        (services /openai)
```

Unchanged: `FaithfulnessScorer` adapter, the workflow's gate semantics
(threshold, retries, directive abstention), and the offline DeepSeek judge.

## Error handling

- Empty `JUDGE_MODEL` still raises `ValueError` in both entry points (no silent
  fallback to the generator — ADR-0009).
- A judge infrastructure failure at runtime keeps the existing fail-closed
  behaviour in `workflow.py` (abstain on scorer exception).

## Testing

TDD, no live Azure calls in unit tests:

- `_judge_provider`: `claude-sonnet-4-6` -> `anthropic`; `Kimi-K2.5` -> `openai`;
  `DeepSeek-V4-Pro` -> `openai`; case-insensitive.
- `build_ragas_faithfulness` dispatch: monkeypatch `_build_claude_faithfulness`
  and `_build_openai_faithfulness`; assert the right one is selected per
  `judge_model`, and that empty `judge_model` still raises.
- `build_judge_complete_fn` dispatch: monkeypatch `_anthropic_complete_fn` and
  `_openai_complete_fn`; assert selection per `judge_model` and empty-guard.
- Keep `tests/test_faithfulness.py::test_gate_requires_judge_model`.

Live verification (after ingest): `uv run python scripts/verify_judges.py` —
all four checks expected to PASS with `JUDGE_MODEL=Kimi-K2.5` (gate, offline
DeepSeek, gpt decoration, raw judge completion).

## Docs

New **ADR-0011 — Provider-aware online judge**: records the name-based routing,
Kimi-K2.5 as the active gate (Claude quota unavailable), and that the ADR-0009
independence principle is preserved (third distinct family). References ADR-0009.

## Risks

- Kimi via RAGAS `Faithfulness` uses prompt-based JSON parsing — the same path
  that works for DeepSeek, but Kimi's output format is unverified until the live
  `verify_judges.py` run.
- `temperature` override acceptance on Kimi (mitigation above).
