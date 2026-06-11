# 0009 — Three-family judge split and directive guardrail

**Status:** Accepted (2026-06-11)

## Context

After ADR-0008, one model (gpt-5.4) still generated answers, judged them online,
judged them offline, and authored synthetic test items — self-preference bias in
every LLM-judged number, and a circular loop: the online guardrail retried
answers until they passed the same metric/model that offline eval then scored,
so offline faithfulness was saturated by construction. The guardrail was also
advisory: an exhausted retry loop returned the unfaithful answer with only a
lowConfidence flag, and each retry re-ran an identical prompt over an identical
candidate set ("regenerate and hope").

## Decision

1. **Three families.** Generator: `gpt-5.4` (OpenAI). Online faithfulness gate:
   `claude-sonnet-4-6` (Anthropic, Messages API on the account's `/anthropic`
   route, Entra bearer with scope `https://ai.azure.com/.default`; the SDK's
   `X-Api-Key` header is suppressed so only the bearer reaches the gateway).
   Offline RAGAS suite: `DeepSeek-V4-Pro` (DeepSeek; sold directly by Azure —
   Azure-direct licensing, GlobalStandard in all regions, served on the
   OpenAI-compatible route of `<account>.services.ai.azure.com` — the existing
   `AzureChatOpenAI` pattern, no marketplace acceptance). The gate gets the
   strongest judge because it decides what users see; the offline judge's
   independence from BOTH other families keeps offline scores un-self-judged
   and un-gate-saturated. Caveats: DeepSeek-V4-Pro is preview, has no tool
   calling (RAGAS uses prompt-based JSON parsing), and is a reasoning model
   (returns `reasoning_content`; no sampling overrides sent).
   `JUDGE_MODEL` / `OFFLINE_JUDGE_MODEL` are required where used — no silent
   fallback to the generator.
2. **Directive guardrail.** On exhaustion the answer is REPLACED with a fixed
   abstention; the suppressed answer survives only in the trace. Judge
   infrastructure failure abstains immediately (fail-closed and fail-fast —
   regeneration cannot fix a judge outage, so retries are not burned on it).
3. **Retries change something.** Retrieval legs fetch `CANDIDATE_POOL` (15)
   candidates once; each retry widens the rerank window (`top_k + 3·attempt`)
   and feeds the rejected answer back via a corrective instruction. The rerank
   query itself became hybrid (text + vector with an id filter) in the same
   change set, so dense-only candidates survive stage-1 retrieval.
4. **Abstention rate is a first-class metric** (`abstained` in every eval
   report, overall and per tag). Offline faithfulness is scored on answered
   items only and read as a consistency check near the gate threshold — the
   discriminating offline metrics are the deterministic ones (ADR-0002) plus
   abstention rate.

## Alternatives rejected

- **Claude judges both online and offline:** one fewer deployment, but offline
  faithfulness becomes fully saturated (only answers that passed Claude's bar
  get scored by Claude) and self-preference returns at the gate boundary's
  mirror. A third family costs one Bicep resource.
- **GPT gate + Claude offline:** preserves a discriminating offline
  faithfulness, but leaves a self-lenient gate in production — backwards once
  the gate is directive.
- **Mistral-Large-3 as third family:** also sold directly by Azure in all
  regions and tool-calling capable, but DeepSeek-V4-Pro was preferred for its
  fully Azure-direct licensing model. Documented fallback if the DeepSeek
  preview misbehaves as a RAGAS judge (e.g. JSON-parsing failures in
  `scripts/verify_judges.py`): swap `OFFLINE_JUDGE_MODEL` and the bicep
  `format` to `'Mistral AI'` — no code changes needed.
- **Grok/Llama as third family:** grok is unavailable in swedencentral;
  Llama 3.3 70B requires marketplace serverless plumbing.
- **Retry with query rewriting:** the right feature in the wrong place; it
  belongs at the front of the pipeline for all queries, measured by eval.

## Consequences

- Cost/latency: Claude marketplace-billed tokens in the hot path (× retries);
  RAGAS faithfulness is multi-call. Keep `MAX_RETRIES` low; measure gate
  latency once live.
- Comparability: deterministic metrics remain comparable across this change;
  all LLM-judged numbers re-anchor (judge changed). The work-machine protocol:
  run `scripts/verify_judges.py`, then a baseline eval on `main`, commit it as
  `eval_baseline.json` (ADR-0006), then run this branch.
- Per-stage hit_rate is computed over CANDIDATE_POOL-deep lists for dense/bm25
  (deeper lists → higher hit_rate by construction); MRR of top ranks is
  unaffected. Stage metrics are comparable to each other within a run, and to
  prior runs only via the reranked stage.

## Sources

- Self-preference bias: Panickssery, Bowman & Feng — https://arxiv.org/abs/2404.13076
- Claude on Foundry (routes, Entra scope, regions):
  https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-claude
- DeepSeek-V4-Pro sold by Azure (preview, GlobalStandard all regions, JSON
  response format, no tool calling, reasoning content):
  https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
- OpenAI-compatible route serving non-OpenAI sold-by-Azure models:
  https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints
- Azure semantic ranker is a second-stage re-scorer (motivates the hybrid
  rerank fix shipped alongside):
  https://learn.microsoft.com/azure/search/semantic-search-overview
