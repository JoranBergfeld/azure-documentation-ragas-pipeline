# 0008 — Sweden Central region; gpt-5.4 generator; claude-sonnet-4-6 provisioned as judge

**Status:** Accepted (2026-06-10)

## Context

The project is moving to a new subscription (re-provisioning from scratch), and we
want (a) Claude Sonnet 4.6 available as an independent RAGAS judge — the 2026-06-10
review's top finding was that one model (gpt-4o) generates, judges online, judges
offline, and authored synthetic test items, importing self-preference bias — and
(b) gpt-5.4 as the generator. Claude models in Microsoft Foundry deploy **only**
when the Foundry resource is in **East US 2 or Sweden Central**, which conflicts
with the previous default region (switzerlandnorth, chosen 2026-05 because
`text-embedding-3-small` had no regional-Standard availability elsewhere in
Europe).

## Decision

- **Region: swedencentral** (EU; Claude-capable). The old 3-small blocker applies
  only to regional Standard deployments — 3-small and gpt-5.4 are both available
  in swedencentral via **GlobalStandard/DataZone** deployments, so the embedding
  deployment switches from `Standard` to `GlobalStandard`.
- **Generator: `gpt-5.4`** (version `2026-03-05`, format OpenAI, GlobalStandard).
  It is also the RAGAS judge until the judge split lands, so the in-flight
  decoration A/B (ADR-0006) stays internally consistent: baseline and post-change
  both run generator=judge=gpt-5.4 on the new index.
- **Judge: `claude-sonnet-4-6`** (preview, format Anthropic, GlobalStandard,
  marketplace-billed) is **deployed now but not wired** — client wiring, a
  separate `JUDGE_MODEL` setting, and a re-anchored RAGAS baseline are the next
  spec. Bicep gates it behind `judgeModel` (empty string skips).

## Alternatives rejected

- **eastus2 (everything co-located):** also Claude-capable with full model lineup,
  but the user prefers an EU region; eastus2 also showed chronic Foundry capacity
  errors on the previous subscription.
- **Stay in switzerlandnorth + second Foundry account in a Claude region:** no
  migration, but two endpoints/two RBAC setups permanently, for no benefit once a
  full re-provision is happening anyway.
- **text-embedding-3-large in swedencentral:** the 2026-05 workaround for the
  regional-Standard gap; unnecessary now that GlobalStandard hosts 3-small, and it
  would change vector dimensions for no measured gain.
- **Swap the generator after the decoration experiment:** rejected by the user;
  running gpt-5.4 from the start keeps every measurement on the final model at the
  cost of comparability with the (discarded) old-subscription anecdotes.

## Consequences

- GlobalStandard/marketplace caveats: Claude requires Azure Marketplace access and
  pay-as-you-go billing; the first deployment may need a one-time offer acceptance
  in the portal (Bicep-only provisioning may fail until then).
- GlobalStandard processes traffic outside the resource region; if strict EU data
  residency becomes a requirement, chat/embedding can move to `DataZoneStandard`
  (both models are DataZone-EU-available in swedencentral) — Claude has no
  DataZone option.
- The generator agent registration and `.env` (`FOUNDRY_CHAT_MODEL=gpt-5.4`) must
  match the deployment name; `JUDGE_MODEL` is exported by Bicep but unused by code
  until the judge spec.

## Sources

- Claude models in Foundry — regions, Global Standard, marketplace requirements:
  https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-claude
- Claude deployment names (`claude-sonnet-4-6` etc.):
  https://learn.microsoft.com/azure/foundry/foundry-models/how-to/configure-claude-code
- GPT-5.4 GA in Microsoft Foundry (March 2026):
  https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-5-4-in-microsoft-foundry/4499785
- Region availability (gpt-5.4 2026-03-05 and text-embedding-3-small in
  swedencentral via DataZone/GlobalStandard; 3-small regional-Standard EU gap):
  https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure-region-availability
  https://learn.microsoft.com/azure/ai-foundry/openai/concepts/models
- Self-preference bias motivation (judge independence): Panickssery, Bowman & Feng,
  https://arxiv.org/abs/2404.13076
