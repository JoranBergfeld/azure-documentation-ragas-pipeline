# 0022 — Model roles on either inference host

**Status:** Accepted (2026-10-31). Amends ADR-0009 and ADR-0011.

## Context

Every model call here used to be wired straight to the Foundry account, and the
recipe was copy-pasted per call site: `DefaultAzureCredential()` in twelve places,
`api_version="2024-10-21"` in seven, timeout and retry budgets defined in four modules,
endpoint derivation borrowed from `embeddings.py` by the judges and the harness. The
ingest client recorded its usage as `plan` because the agentic planner happened to
borrow it. `tests/test_client_timeouts.py` existed only to monkeypatch fourteen SDK
symbols, because there was no one place to test.

The direction changed too. The DGX Spark already serves the reranker (ADR-0021, #40),
and the plan now is to lean on it as much as is feasible: the generator, the planner,
the ingest writer and the embedder can all run as vLLM behind an OpenAI-compatible
`/v1`, and the Spark is meant to be shared with the other demos (homelab#220). One-off
Azure costs for re-running evals are fine. So "which host serves this call" has to be
a config change, not a code change, and it has to happen one role at a time, because
each switch re-anchors every eval number and needs its own 9-mode rerun.

That also reopens ADR-0009's model choice. It picked Claude for the online gate, and
ADR-0011 had to route around missing Claude quota. Neither ADR states the real rule
precisely enough to survive a move to open weights.

## Decision

1. **Model roles.** Every model call fills exactly one role: generator, planner, ingest
   writer (situating context, RAPTOR summaries, graph extraction and community reports),
   embedder, online judge, offline judge, reranker. `ragpipe.model_roles` owns them.

2. **One seam, two adapters.** Each role is served by the Foundry account, or by an
   OpenAI-compatible URL. Config follows the reranker's existing pattern: setting
   `<ROLE>_URL` (server root, no `/v1`) moves the role, `<ROLE>_MODEL` names the served
   model and is required once a URL is set, and `<ROLE>_API_KEY` is sent if present.
   Unset means Foundry, under the deployment names config already had. The judges and
   the reranker keep their env prefixes (`JUDGE_*`, `OFFLINE_JUDGE_*`, `RERANKER_*`).
   Foundry has no reranker, so a reranker without a URL is simply not served.

3. **Callers ask for a role, not a client recipe.** `openai_client`, `complete_fn`,
   `judge_chat_factory`, `langchain_embeddings` and `generator_agent` take settings and a
   role and hand back a bounded client. Credentials, Foundry route selection (the
   `/openai` host, the services host, the `/anthropic` route), API version and the
   per-role timeout and retry budget stay inside. Usage is recorded per role (the
   planner as `plan`, the online judge as `faithfulness_judge`), not per client.

4. **The generator off Foundry is a local Agent Framework agent** on
   `OpenAIChatCompletionClient` with the same instructions and no tools. Code Interpreter
   only exists in the Foundry Agent Service, and none of the 43 testset questions needs
   computation, so it's dropped when the generator moves rather than replaced.

5. **The judge rule is about model families, not vendors.** The generator, the online
   judge and the offline judge must come from three different model families. A family
   is a base model and everything derived from it: a fine-tune or distillation belongs to
   its base's family whatever brand it ships under, so a Llama-based Nemotron is Llama
   family. Which three families they are doesn't matter; Anthropic isn't required.
   Picking a Spark model for any of these roles means checking its lineage first.

## Alternatives rejected

- **Keep a Foundry-only factory module.** Fixes the copy-paste but not the direction:
  the second adapter is the reason the seam exists (one adapter would be a hypothetical
  seam).
- **One `<ROLE>_PROVIDER=foundry|spark` switch plus URLs.** Two settings that can
  disagree; the URL being present already says it.
- **Move every role at once.** One eval rerun instead of several, but no way to tell
  which move changed which number.
- **Fall back to Foundry when the Spark is down.** The numbers would describe a
  different pipeline under the same mode name, the same reason ADR-0021 refuses a
  passthrough fallback for the reranker. A role that can't answer makes its modes
  unavailable instead (wired into `/modes` in the next change).
- **Keep "Claude for the gate" as the rule.** It was always a means to family
  independence, and it's the one family the Spark can't host.

## Consequences

- Moving a role is env only. The first move after this change is planner and ingest
  writer, then generator, then embedder (re-embed, new `vector(n)`, re-import). Each is
  followed by a full eval rerun.
- Nothing changes on Foundry today: same routes, same deployments, same budgets, same
  agent instructions.
- Spark deployments show up in usage under their model names, so the site needs dated
  zero-rate rows for them (personal-website#113).
- Once the generator or the embedder lives on the Spark, every mode depends on it, not
  just the rerank modes. Health per role and the per-role offline reasons are the next
  change (the mode catalogue).
- The client-kwargs tests sit on one module now (`tests/test_model_roles.py`), Spark
  adapters included. `foundry_judge.py`, `context_gen.build_context_complete_fn`, the
  endpoint helpers in `embeddings.py` and `tests/test_client_timeouts.py` are gone.

## Sources

- Self-preference: Panickssery, Bowman & Feng (2024), *LLM Evaluators Recognize and
  Favor Their Own Generations*. https://arxiv.org/abs/2404.13076
- Family and inheritance bias: Li et al. (2025), *Preference Leakage: A Contamination
  Problem in LLM-as-a-judge*, which defines relatedness as same model, inheritance
  relationship, or same model family, and finds judges favor all three.
  https://arxiv.org/abs/2502.01534
- Measured same-family lift: Awuni et al. (2026), *Who Judges Matters: Measuring
  Family-Conditioned Preference in LLM-as-Judge Panels*, a positive lift of 3.4 to 8.4
  points for all four open-weight families studied. https://arxiv.org/abs/2609.17857
- vLLM's OpenAI-compatible server (`/v1/chat/completions`, `/v1/embeddings`, `--api-key`
  guarding `/v1/*` only): homelab `docs/services/spark.md` and the `spark_reranker` role.
- Agent Framework's OpenAI-compatible chat client (`agent_framework.openai.
  OpenAIChatCompletionClient(model, base_url, api_key)`), verified against the installed
  `agent-framework-openai` 1.7.0.
