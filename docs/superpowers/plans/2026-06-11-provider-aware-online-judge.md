# Provider-aware Online Judge (Kimi-K2.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the online faithfulness judge by provider so `Kimi-K2.5` (OpenAI-compatible route) can serve the gate now, while preserving the Claude `/anthropic` path for when quota returns.

**Architecture:** A single `judge_provider(model)` helper (name-based: `claude*` → anthropic, else → openai) drives two dispatch points — the RAGAS faithfulness metric (`guardrail.build_ragas_faithfulness`) and the raw completion fn (`foundry_judge.build_judge_complete_fn`). The OpenAI branch mirrors the proven offline DeepSeek wiring (`AzureChatOpenAI` on the services host, cognitiveservices scope). The Anthropic branch is unchanged.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `ruff`, RAGAS, `langchain-openai`/`langchain-anthropic`, `openai`/`anthropic` SDKs, Azure AI Foundry (Entra auth).

---

## File Structure

- `src/ragpipe/foundry_judge.py` — **renamed** from `foundry_claude.py`. Hosts `AI_FOUNDRY_SCOPE`, `judge_provider`, provider-aware `build_judge_complete_fn`, and the two private branch builders.
- `src/ragpipe/guardrail.py` — `build_ragas_faithfulness` dispatches by provider; adds `_build_openai_faithfulness`; updates the `AI_FOUNDRY_SCOPE` import path.
- `scripts/verify_judges.py` — import the renamed fn; make check labels provider-neutral.
- `scripts/generate_synthetic_testset.py` — import the renamed fn.
- `tests/test_judge_routing.py` — **new**; unit tests for `judge_provider` and both dispatchers (no live calls).
- `.env.example` — document provider routing on `JUDGE_MODEL`.
- `docs/adr/0011-provider-aware-online-judge.md` — **new** ADR.
- azd env + repo `.env` — set `JUDGE_MODEL=Kimi-K2.5`.

---

## Task 1: Rename module + add `judge_provider` + provider-aware raw completion

**Files:**
- Create (via rename): `src/ragpipe/foundry_judge.py` (from `src/ragpipe/foundry_claude.py`)
- Modify: `src/ragpipe/guardrail.py:83` (import path only, this task)
- Modify: `scripts/verify_judges.py:46-49`
- Modify: `scripts/generate_synthetic_testset.py:18,31`
- Test: `tests/test_judge_routing.py`

- [ ] **Step 1: Write the failing tests for `judge_provider` and raw-completion dispatch**

Create `tests/test_judge_routing.py`:

```python
import pytest

from ragpipe import foundry_judge
from ragpipe.foundry_judge import build_judge_complete_fn, judge_provider


class _S:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"

    def __init__(self, judge_model):
        self.judge_model = judge_model


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-sonnet-4-6", "anthropic"),
        ("Claude-Sonnet-4-6", "anthropic"),
        ("Kimi-K2.5", "openai"),
        ("DeepSeek-V4-Pro", "openai"),
        ("gpt-5.4", "openai"),
    ],
)
def test_judge_provider(model, provider):
    assert judge_provider(model) == provider


def test_complete_fn_dispatches_anthropic_for_claude(monkeypatch):
    calls = []
    monkeypatch.setattr(
        foundry_judge, "_anthropic_complete_fn",
        lambda s, m: calls.append("anthropic") or (lambda p: "A"),
    )
    monkeypatch.setattr(
        foundry_judge, "_openai_complete_fn",
        lambda s, m: calls.append("openai") or (lambda p: "O"),
    )
    fn = build_judge_complete_fn(_S("claude-sonnet-4-6"))
    assert fn("x") == "A"
    assert calls == ["anthropic"]


def test_complete_fn_dispatches_openai_for_kimi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        foundry_judge, "_anthropic_complete_fn",
        lambda s, m: calls.append("anthropic") or (lambda p: "A"),
    )
    monkeypatch.setattr(
        foundry_judge, "_openai_complete_fn",
        lambda s, m: calls.append("openai") or (lambda p: "O"),
    )
    fn = build_judge_complete_fn(_S("Kimi-K2.5"))
    assert fn("x") == "O"
    assert calls == ["openai"]


def test_complete_fn_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_judge_complete_fn(_S(None))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_judge_routing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragpipe.foundry_judge'`.

- [ ] **Step 3: Rename the module (preserve history)**

Run: `git mv src/ragpipe/foundry_claude.py src/ragpipe/foundry_judge.py`

- [ ] **Step 4: Replace the module contents with the provider-aware implementation**

Overwrite `src/ragpipe/foundry_judge.py` with:

```python
"""Judge model transport on the Foundry account, routed by provider (ADR-0011).

The online faithfulness gate (ADR-0009) and synthetic test-item authoring
(ADR-0010) call a judge model through this module. Routing is by provider:
Anthropic (Claude) deployments use the Messages API on the account's
``/anthropic`` route (Entra scope ``https://ai.azure.com/.default``); every
other family (Kimi, DeepSeek, ...) is served on the OpenAI-compatible route of
``<account>.services.ai.azure.com`` (Entra scope
``https://cognitiveservices.azure.com/.default``) — the same route the offline
DeepSeek judge uses.
"""
from __future__ import annotations

from typing import Callable

AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def judge_provider(model: str) -> str:
    """Transport provider for a judge deployment name.

    Claude deployments are served on the ``/anthropic`` Messages route; every
    other family (Kimi, DeepSeek, ...) is served on the OpenAI-compatible route.
    The name is the only signal available without a live deployment lookup, and
    Claude is the only Anthropic-route family on this account (ADR-0011).
    """
    return "anthropic" if "claude" in model.lower() else "openai"


def build_judge_complete_fn(settings, max_tokens: int = 2048) -> Callable[[str], str]:
    """`complete(prompt) -> str` against the judge deployment, routed by provider."""
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required to call the judge (ADR-0009); set it in .env"
        )
    if judge_provider(settings.judge_model) == "anthropic":
        return _anthropic_complete_fn(settings, max_tokens)
    return _openai_complete_fn(settings, max_tokens)


def _anthropic_complete_fn(settings, max_tokens: int) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """Claude via the Anthropic Messages API on the account's /anthropic route.

    The client is rebuilt per call so the (cached, auto-refreshed) Entra bearer
    token from azure-identity is always current — client construction is cheap
    next to the model call.
    """
    from anthropic import Anthropic
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    from ragpipe.embeddings import anthropic_endpoint_from_project

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    def complete(prompt: str) -> str:
        client = Anthropic(base_url=base_url, auth_token=token_provider())
        resp = client.messages.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return complete


def _openai_complete_fn(settings, max_tokens: int) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """Non-Anthropic judge (Kimi, DeepSeek, ...) via the OpenAI-compatible route.

    Targets ``<account>.services.ai.azure.com/openai/deployments/<model>`` — the
    same host the offline DeepSeek judge uses. No explicit temperature: reasoning
    deployments on this route may reject sampling overrides.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    from ragpipe.embeddings import (
        COGNITIVE_SERVICES_SCOPE,
        services_endpoint_from_project,
    )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    azure_endpoint = services_endpoint_from_project(settings.foundry_project_endpoint)

    def complete(prompt: str) -> str:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-10-21",
        )
        resp = client.chat.completions.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
```

- [ ] **Step 5: Update the `AI_FOUNDRY_SCOPE` import in `guardrail.py`**

In `src/ragpipe/guardrail.py`, inside `_build_claude_faithfulness` (currently line 83), change:

```python
    from ragpipe.foundry_claude import AI_FOUNDRY_SCOPE
```
to:
```python
    from ragpipe.foundry_judge import AI_FOUNDRY_SCOPE
```

- [ ] **Step 6: Update the two script callers**

In `scripts/verify_judges.py`, replace lines 46-49:

```python
    def claude_raw():
        from ragpipe.foundry_claude import build_claude_complete_fn

        return build_claude_complete_fn(settings)("Reply with exactly: OK")
```
with:
```python
    def judge_raw():
        from ragpipe.foundry_judge import build_judge_complete_fn

        return build_judge_complete_fn(settings)("Reply with exactly: OK")
```

And update the two affected labels + docstring:
- Line 2-3 docstring: change `raw Claude completion` → `raw judge completion`, and `Claude gate scoring` → `online judge gate scoring`.
- Line 51: `check("claude gate (RAGAS faithfulness)", gate)` → `check("online judge gate (RAGAS faithfulness)", gate)`
- Line 54: `check("claude raw completion", claude_raw)` → `check("judge raw completion", judge_raw)`

In `scripts/generate_synthetic_testset.py`:
- Line 18: `from ragpipe.foundry_claude import build_claude_complete_fn` → `from ragpipe.foundry_judge import build_judge_complete_fn`
- Line 31: `complete = build_claude_complete_fn(settings)` → `complete = build_judge_complete_fn(settings)`

- [ ] **Step 7: Run the routing tests + full suite**

Run: `uv run pytest tests/test_judge_routing.py tests/test_faithfulness.py -q`
Expected: PASS (routing dispatch + existing `test_gate_requires_judge_model`).

Run: `uv run pytest -q`
Expected: PASS (no import breakage from the rename).

- [ ] **Step 8: Commit**

```bash
git add src/ragpipe/foundry_judge.py src/ragpipe/guardrail.py \
        scripts/verify_judges.py scripts/generate_synthetic_testset.py \
        tests/test_judge_routing.py
git commit -m "feat: provider-aware judge transport (rename foundry_claude -> foundry_judge)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Provider-aware faithfulness gate

**Files:**
- Modify: `src/ragpipe/guardrail.py:56-71` (`build_ragas_faithfulness`) and add `_build_openai_faithfulness`
- Test: `tests/test_judge_routing.py` (extend)

- [ ] **Step 1: Write the failing gate-dispatch tests**

Append to `tests/test_judge_routing.py`:

```python
from ragpipe import guardrail
from ragpipe.guardrail import build_ragas_faithfulness


def test_gate_dispatches_anthropic_for_claude(monkeypatch):
    calls = []
    monkeypatch.setattr(
        guardrail, "_build_claude_faithfulness",
        lambda s: calls.append("anthropic") or "A",
    )
    monkeypatch.setattr(
        guardrail, "_build_openai_faithfulness",
        lambda s: calls.append("openai") or "O",
    )
    assert build_ragas_faithfulness(_S("claude-sonnet-4-6")) == "A"
    assert calls == ["anthropic"]


def test_gate_dispatches_openai_for_kimi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        guardrail, "_build_claude_faithfulness",
        lambda s: calls.append("anthropic") or "A",
    )
    monkeypatch.setattr(
        guardrail, "_build_openai_faithfulness",
        lambda s: calls.append("openai") or "O",
    )
    assert build_ragas_faithfulness(_S("Kimi-K2.5")) == "O"
    assert calls == ["openai"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_judge_routing.py -q -k gate_dispatches`
Expected: FAIL — `AttributeError: ... has no attribute '_build_openai_faithfulness'`.

- [ ] **Step 3: Add provider dispatch to `build_ragas_faithfulness`**

In `src/ragpipe/guardrail.py`, replace the body of `build_ragas_faithfulness` (the `return _build_claude_faithfulness(settings)` at line ~70) so the function reads:

```python
def build_ragas_faithfulness(settings) -> MetricFn:
    """Faithfulness gate judged by a non-generator family (ADR-0009).

    The gate decides what users see, so it must not share a family with the
    generator: a judge from the generator's own family is systematically
    lenient on its own outputs (self-preference bias). Routed by provider
    (ADR-0011): Claude on the /anthropic route, every other family on the
    OpenAI-compatible route. Raises if JUDGE_MODEL is unset — silently falling
    back to the generator would recreate the circular setup this replaces.
    """
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required: the faithfulness gate is judged by a "
            "non-generator family (ADR-0009); set it in .env"
        )
    from ragpipe.foundry_judge import judge_provider

    if judge_provider(settings.judge_model) == "anthropic":
        return _build_claude_faithfulness(settings)
    return _build_openai_faithfulness(settings)
```

- [ ] **Step 4: Add `_build_openai_faithfulness` after `_build_claude_faithfulness`**

In `src/ragpipe/guardrail.py`, add this function immediately after `_build_claude_faithfulness` (after its `return metric_fn`):

```python
def _build_openai_faithfulness(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    from ragpipe.embeddings import (
        COGNITIVE_SERVICES_SCOPE,
        services_endpoint_from_project,
    )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    # azure_ad_token_provider auto-refreshes, so the judge is built once (the
    # Anthropic path rebuilds per call because ChatAnthropic fixes headers at
    # construction). No explicit temperature: reasoning deployments on this
    # route may reject sampling overrides (matches the offline DeepSeek judge).
    judge_chat = AzureChatOpenAI(
        azure_endpoint=services_endpoint_from_project(settings.foundry_project_endpoint),
        azure_deployment=settings.judge_model,
        api_version="2024-10-21",
        azure_ad_token_provider=token_provider,
    )
    metric = Faithfulness(llm=LangchainLLMWrapper(judge_chat))

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await metric.single_turn_ascore(sample))

    return metric_fn
```

- [ ] **Step 5: Run the gate tests + full suite**

Run: `uv run pytest tests/test_judge_routing.py tests/test_faithfulness.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/guardrail.py tests/test_judge_routing.py
git commit -m "feat: route faithfulness gate by provider (OpenAI-compatible Kimi path)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Docs (ADR-0011) + `.env.example`

**Files:**
- Create: `docs/adr/0011-provider-aware-online-judge.md`
- Modify: `.env.example:4`

- [ ] **Step 1: Create the ADR**

Create `docs/adr/0011-provider-aware-online-judge.md`:

```markdown
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
module renamed `foundry_claude.py` → `foundry_judge.py`). The OpenAI branch uses
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
```

- [ ] **Step 2: Document routing on `JUDGE_MODEL` in `.env.example`**

In `.env.example`, replace line 4:

```
JUDGE_MODEL="claude-sonnet-4-6"
```
with:
```
# Online faithfulness gate, routed by provider (ADR-0011): a name containing
# "claude" uses the Anthropic /anthropic route; any other model (e.g. Kimi-K2.5)
# uses the OpenAI-compatible route. Must be a non-generator family (ADR-0009).
JUDGE_MODEL="Kimi-K2.5"
```

- [ ] **Step 3: Commit**

```bash
git add docs/adr/0011-provider-aware-online-judge.md .env.example
git commit -m "docs: ADR-0011 provider-aware online judge; document JUDGE_MODEL routing

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Set `JUDGE_MODEL=Kimi-K2.5` and live-verify

**Files:** none (operational); uses azd env + repo `.env`.

- [ ] **Step 1: Set the judge model in both env stores**

Run:
```bash
azd env set JUDGE_MODEL Kimi-K2.5
sed -i 's/^JUDGE_MODEL=.*/JUDGE_MODEL="Kimi-K2.5"/' .env
grep JUDGE_MODEL .env
```
Expected: `JUDGE_MODEL="Kimi-K2.5"`.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/ragpipe/foundry_judge.py src/ragpipe/guardrail.py scripts/verify_judges.py scripts/generate_synthetic_testset.py tests/test_judge_routing.py`
Expected: no errors. (Run `uv run ruff format` on the same paths if the repo formats with ruff.)

- [ ] **Step 3: Full unit suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Live judge smoke (needs Azure access; index not required)**

Run: `uv run python scripts/verify_judges.py`
Expected: all four checks `PASS` — `online judge gate (RAGAS faithfulness)`, `deepseek offline judge`, `gpt decoration completion`, `judge raw completion`.

Failure handling:
- If `judge raw completion` fails with an error about `max_tokens`, switch the
  `chat.completions.create(...)` call in `_openai_complete_fn` to use
  `max_completion_tokens=max_tokens` instead of `max_tokens` and re-run.
- If the gate fails on RAGAS JSON parsing, capture the error; Kimi's output
  format may differ from DeepSeek's. Record it for follow-up (do not silently
  fall back).

- [ ] **Step 5: Commit any fix from Step 4 (only if code changed)**

```bash
git add -A
git commit -m "fix: adjust OpenAI judge call for Kimi-K2.5 per live verify

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Self-Review

**Spec coverage:** provider detection (Task 1) ✓; gate dispatch + OpenAI faithfulness (Task 2) ✓; raw-completion rename + dispatch + callers (Task 1) ✓; module rename (Task 1) ✓; config/env (Task 4) ✓; `.env.example` (Task 3) ✓; tests (Tasks 1-2) ✓; ADR-0011 (Task 3) ✓; verify (Task 4) ✓. No gaps.

**Placeholder scan:** every code/edit step shows full content; no TBD/TODO. ✓

**Type consistency:** `judge_provider`, `build_judge_complete_fn`, `_anthropic_complete_fn`, `_openai_complete_fn`, `_build_openai_faithfulness`, `_build_claude_faithfulness` are referenced with consistent names and signatures across tasks. `services_endpoint_from_project` and `COGNITIVE_SERVICES_SCOPE` exist in `embeddings.py`. ✓
