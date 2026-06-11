# 0011 — Provider-aware online judge

**Status:** Accepted (2026-06-11)

## Context

ADR-0009 set the online faithfulness gate to `claude-sonnet-4-6` on the Foundry
account's `/anthropic` route. That model has **zero deployable quota** on this
subscription (GlobalStandard = 0 in both swedencentral and eastus2, the only
Claude regions), so the gate could not run. A `Kimi-K2.5` deployment
(MoonshotAI) is available and is served on the **OpenAI-compatible route**
(`<account>.services.ai.azure.com/openai/deployments/<name>`), not `/anthropic`,
so `JUDGE_MODEL=Kimi-K2.5` failed against the Anthropic-only wiring.

## Decision

Route the online judge by provider. `judge_provider(model)` returns `anthropic`
for Claude deployments (name contains "claude") and `openai` for every other
family. Two dispatch points honour it: the RAGAS faithfulness metric
(`guardrail.build_ragas_faithfulness`) and the raw completion fn
(`foundry_judge.build_judge_complete_fn`, renamed from `build_claude_complete_fn`;
module renamed `foundry_claude.py` -> `foundry_judge.py`). The OpenAI branch uses
`AzureChatOpenAI` / `AzureOpenAI` on the services host with the
`https://cognitiveservices.azure.com/.default` scope — the same wiring as the
offline DeepSeek judge. The Anthropic branch is unchanged.

`JUDGE_MODEL` is set to `Kimi-K2.5`. ADR-0009's independence principle holds:
Kimi-K2.5 (MoonshotAI) is a third family, distinct from the gpt generator and
the DeepSeek offline judge, so the gate stays free of self-preference bias.

## Alternatives rejected

- **Replace the Anthropic path entirely:** less code, but discards the Claude
  gate for when quota returns and supersedes ADR-0009's model choice outright.
  Provider routing keeps both paths at the cost of one small dispatch.
- **Explicit `JUDGE_PROVIDER` env var:** unnecessary today — Claude is the only
  Anthropic-route family on this account, so the name heuristic is unambiguous
  (YAGNI). Add it if a non-"claude" Anthropic model ever appears.

## Consequences

- The gate runs on Kimi-K2.5 now; all four `scripts/verify_judges.py` checks
  pass. All LLM-judged numbers re-anchor (the gate model changed).
- Kimi-K2.5 has no verified `temperature=0` support on this route, so the
  OpenAI judge omits sampling overrides (like DeepSeek); gate determinism is
  best-effort, consistent with the offline judge.
- Restoring Claude is config-only: set `JUDGE_MODEL=claude-sonnet-4-6` once
  quota is granted — no code change.

## Sources

- ADR-0009 (three-family judge split): the independence principle this decision
  preserves while changing the gate model.
- Self-preference bias (why the gate must be a non-generator family): Panickssery,
  Bowman & Feng — https://arxiv.org/abs/2404.13076
- Models sold directly by Azure, served on the OpenAI-compatible route (covers
  Kimi/MoonshotAI and DeepSeek):
  https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
- OpenAI-compatible endpoint on `<account>.services.ai.azure.com` for non-OpenAI
  models: https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints
- Measurement: `Kimi-K2.5` returns a chat completion via `AzureOpenAI` against the
  services host (verified this change set); `scripts/verify_judges.py` is the
  standing live check that all four routes work.
