# Three-Family Judge Split + Directive Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the evaluation credible (three model families: gpt-5.4 generates, claude-sonnet-4-6 gates online, DeepSeek-V4-Pro judges offline), make the guardrail directive (abstain instead of flagging), make retries actually change something (widened rerank window + corrective prompt), fix the broken `azd up`, fix the reranker dropping dense-only candidates, and establish a Claude-authored synthetic-testset pipeline with provenance gold URLs.

**Architecture:** The pipeline keeps its decomposed client-side stages. Retrieval legs fetch a wider candidate pool once; the semantic rerank becomes a hybrid (text+vector) query so dense-only candidates survive; the retry loop widens the rerank window per attempt and feeds the failed answer back to the generator; on exhaustion the answer is replaced with a fixed abstention. Judges: the online gate keeps RAGAS Faithfulness but backed by Claude via the Foundry `/anthropic` route (Entra bearer); offline RAGAS metrics move to DeepSeek-V4-Pro via the OpenAI-compatible route (existing `AzureChatOpenAI` pattern, new endpoint). Synthetic test items are authored by Claude from explicitly-named corpus pages, so the gold URL is provenance, not recovered by substring match.

**Tech Stack:** Python 3.11 / uv, Azure AI Search, Azure AI Foundry (gpt-5.4, claude-sonnet-4-6, DeepSeek-V4-Pro), RAGAS 0.4.3, langchain-anthropic + anthropic (new deps), Bicep/azd, pytest.

**Out of scope (tracked separately, from the 2026-06-10 review):** API auth header, async/threadpool fix for blocking I/O, non-root Dockerfile user, ingest prune guard + upload-result checking, testset growth itself (needs human authoring time; ADR-0010 defines the policy).

**Verification reality check:** unit tests run locally; anything touching Azure (judge calls, decoration, candidate generation, `azd provision`) is exercised by `scripts/verify_judges.py` and the eval run on the work laptop. Tasks below mark live steps explicitly — do not block local execution on them.

---

## Phase A — Deploy & config plumbing

### Task 1: Fix `azd up` and provision the offline judge

**Files:**
- Modify: `infra/main.parameters.json`
- Modify: `infra/main.bicep` (after the `judge` resource, line 128; outputs at 165-169)
- Modify: `.env.example`

- [ ] **Step 1: Replace `infra/main.parameters.json` wholesale**

The current file pins `location=switzerlandnorth` and `chatModel=gpt-4o`, overriding the ADR-0008 bicep defaults (gpt-5.4 version `2026-03-05` is invalid for gpt-4o, and Claude cannot deploy in switzerlandnorth). Only pin what must differ from bicep defaults:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "location": { "value": "${AZURE_LOCATION=swedencentral}" },
    "baseName": { "value": "ragpipe" },
    "principalId": { "value": "${AZURE_PRINCIPAL_ID}" },
    "principalType": { "value": "${AZURE_PRINCIPAL_TYPE=User}" }
  }
}
```

- [ ] **Step 2: Add the offline judge deployment to `infra/main.bicep`**

Add a param after `judgeModel` (line 25):

```bicep
@description('Offline RAGAS judge deployment. DeepSeek-V4-Pro (preview) is sold directly by Azure: GlobalStandard in all regions (incl. swedencentral), served on the OpenAI-compatible route, Azure-direct licensing — no marketplace acceptance needed (unlike Claude). Third family besides the OpenAI generator and the Anthropic online gate (ADR-0009). Set to empty string to skip.')
param offlineJudgeModel string = 'DeepSeek-V4-Pro'
```

Add the deployment resource after the `judge` resource (after line 128):

```bicep
// DeepSeek model (sold directly by Azure) for the OFFLINE RAGAS judge (ADR-0009):
// a third family so offline scores are independent of both the generator (OpenAI)
// and the online gate (Anthropic). Sequential dependsOn — same
// one-deployment-at-a-time rule.
resource offlineJudge 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = if (!empty(offlineJudgeModel)) {
  parent: foundry
  name: offlineJudgeModel
  dependsOn: [judge]
  sku: { name: 'GlobalStandard', capacity: 50 }
  properties: {
    model: { format: 'DeepSeek', name: offlineJudgeModel }
  }
}
```

Add the output at the end (after line 169):

```bicep
output OFFLINE_JUDGE_MODEL string = offlineJudgeModel
```

- [ ] **Step 3: Verify the bicep compiles**

Run: `az bicep build --file infra/main.bicep --stdout > /dev/null && echo OK`
Expected: `OK` (warnings acceptable, no errors)

- [ ] **Step 4 (LIVE, work laptop, before first provision): verify the DeepSeek format string**

The `format: 'DeepSeek'` value is inferred by analogy with `'Anthropic'`/`'OpenAI'`. Confirm before provisioning:

Run: `az cognitiveservices account list-models -n ragpipe-foundry -g rg-ragas --query "[?contains(name, 'eepSeek')].{name:name, format:format}" -o table`

If the account doesn't exist yet, check the catalog instead: portal → Foundry → Model catalog → DeepSeek-V4-Pro → deployment JSON. Adjust `format` if it differs.

- [ ] **Step 5: Add the new env vars to `.env.example`**

Add after the `FOUNDRY_CHAT_MODEL` line:

```
JUDGE_MODEL=claude-sonnet-4-6
OFFLINE_JUDGE_MODEL=DeepSeek-V4-Pro
# Retrieval candidate pool per leg (dense/bm25) before fusion; rerank narrows to TOP_K.
CANDIDATE_POOL=15
```

- [ ] **Step 6: Commit**

```bash
git add infra/main.parameters.json infra/main.bicep .env.example
git commit -m "infra: fix azd params drift (swedencentral/gpt-5.4), add DeepSeek-V4-Pro offline judge deployment"
```

### Task 2: Settings — judge models, candidate pool, range validation

**Files:**
- Modify: `src/ragpipe/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_config.py`)

```python
import pytest

from ragpipe.config import Settings


def _base_env(monkeypatch):
    for k, v in {
        "FOUNDRY_PROJECT_ENDPOINT": "https://acct.services.ai.azure.com/api/projects/p",
        "FOUNDRY_CHAT_MODEL": "gpt-5.4",
        "FOUNDRY_EMBEDDING_MODEL": "text-embedding-3-small",
        "SEARCH_ENDPOINT": "https://s.search.windows.net",
        "SEARCH_INDEX": "idx",
        "GENERATOR_AGENT_NAME": "gen",
    }.items():
        monkeypatch.setenv(k, v)


def test_judge_models_parsed_from_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OFFLINE_JUDGE_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("CANDIDATE_POOL", "20")
    s = Settings.from_env(load=False)
    assert s.judge_model == "claude-sonnet-4-6"
    assert s.offline_judge_model == "DeepSeek-V4-Pro"
    assert s.candidate_pool == 20


def test_judge_models_default_to_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OFFLINE_JUDGE_MODEL", raising=False)
    s = Settings.from_env(load=False)
    assert s.judge_model is None
    assert s.offline_judge_model is None
    assert s.candidate_pool == 15


def test_threshold_out_of_range_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("FAITHFULNESS_THRESHOLD", "7")  # typo for 0.7
    with pytest.raises(ValueError, match="FAITHFULNESS_THRESHOLD"):
        Settings.from_env(load=False)


def test_candidate_pool_smaller_than_top_k_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("CANDIDATE_POOL", "3")
    monkeypatch.setenv("TOP_K", "5")
    with pytest.raises(ValueError, match="CANDIDATE_POOL"):
        Settings.from_env(load=False)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL (`judge_model` attribute / no ValueError)

- [ ] **Step 3: Implement in `src/ragpipe/config.py`**

Add fields to `Settings` (after `generator_agent_version`, line 30):

```python
    # Judge-model split (ADR-0009): the online faithfulness gate is judged by the
    # Anthropic deployment, the offline RAGAS suite by the DeepSeek deployment —
    # neither shares a family with the gpt generator. None = unset; the builders
    # that need them raise rather than silently falling back to the generator.
    judge_model: str | None = None
    offline_judge_model: str | None = None
```

Add after `rrf_k` (line 34):

```python
    # Candidate pool per retrieval leg (dense/bm25) before RRF fusion. Wider than
    # top_k so guardrail retries can widen the rerank window over real candidates.
    candidate_pool: int = 15
```

Add validation at the end of the class:

```python
    def __post_init__(self) -> None:
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError(
                f"FAITHFULNESS_THRESHOLD must be in [0, 1], got {self.faithfulness_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"MAX_RETRIES must be >= 0, got {self.max_retries}")
        if self.top_k < 1:
            raise ValueError(f"TOP_K must be >= 1, got {self.top_k}")
        if self.candidate_pool < self.top_k:
            raise ValueError(
                f"CANDIDATE_POOL ({self.candidate_pool}) must be >= TOP_K ({self.top_k})"
            )
```

Add to `from_env` (inside the `cls(...)` call):

```python
            judge_model=os.environ.get("JUDGE_MODEL") or None,
            offline_judge_model=os.environ.get("OFFLINE_JUDGE_MODEL") or None,
            candidate_pool=int(os.environ.get("CANDIDATE_POOL", "15")),
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS (all, including pre-existing)

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/config.py tests/test_config.py
git commit -m "feat(config): JUDGE_MODEL/OFFLINE_JUDGE_MODEL/CANDIDATE_POOL + range validation (ADR-0009)"
```

### Task 3: Endpoint helpers for the services / anthropic routes

**Files:**
- Modify: `src/ragpipe/embeddings.py` (after `openai_endpoint_from_project`, line 35)
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_embeddings.py`)

```python
from ragpipe.embeddings import (
    anthropic_endpoint_from_project,
    services_endpoint_from_project,
)


def test_services_endpoint_strips_project_path():
    ep = services_endpoint_from_project(
        "https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project"
    )
    assert ep == "https://ragpipe-foundry.services.ai.azure.com"


def test_anthropic_endpoint_appends_route():
    ep = anthropic_endpoint_from_project(
        "https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project"
    )
    assert ep == "https://ragpipe-foundry.services.ai.azure.com/anthropic"


def test_services_endpoint_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        services_endpoint_from_project("not-a-url")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_embeddings.py -q` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** (in `src/ragpipe/embeddings.py`, after `openai_endpoint_from_project`)

```python
def services_endpoint_from_project(project_endpoint: str) -> str:
    """Strip the project path: the account's services.ai.azure.com root.

    The OpenAI-compatible route on this host serves non-OpenAI models sold by
    Azure (e.g. Mistral-Large-3) at /openai/deployments/<name> — the plain
    <account>.openai.azure.com host serves only Azure OpenAI deployments.
    """
    host = urlparse(project_endpoint).netloc
    if not host:
        raise ValueError(f"Cannot derive host from endpoint: {project_endpoint!r}")
    return f"https://{host}"


def anthropic_endpoint_from_project(project_endpoint: str) -> str:
    """Base URL for the account's Anthropic Messages route (Claude deployments).

    The anthropic SDK appends /v1/messages, matching the documented target
    https://<account>.services.ai.azure.com/anthropic/v1/messages.
    """
    return f"{services_endpoint_from_project(project_endpoint)}/anthropic"
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_embeddings.py -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/embeddings.py tests/test_embeddings.py
git commit -m "feat: services/anthropic endpoint helpers for the judge split"
```

## Phase B — Judge wiring

### Task 4: Add the Anthropic dependencies

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via uv)

- [ ] **Step 1: Add deps**

Run: `uv add langchain-anthropic anthropic`
Expected: both resolve and install (note `[tool.uv] prerelease = "allow"` is global; check the resolved versions printed are stable releases).

- [ ] **Step 2: Sanity import**

Run: `uv run python -c "from langchain_anthropic import ChatAnthropic; from anthropic import Anthropic; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add langchain-anthropic + anthropic for the Claude gate"
```

### Task 5: Claude as the online faithfulness gate

**Files:**
- Create: `src/ragpipe/foundry_claude.py`
- Modify: `src/ragpipe/guardrail.py:56-87` (`build_ragas_faithfulness`)
- Test: `tests/test_faithfulness.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_faithfulness.py`)

```python
import pytest

from ragpipe.guardrail import build_ragas_faithfulness


class _Settings:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"
    judge_model = None


def test_gate_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_ragas_faithfulness(_Settings())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_faithfulness.py -q` — Expected: the new test FAILS (no ValueError raised — current code falls through to the gpt judge)

- [ ] **Step 3: Create `src/ragpipe/foundry_claude.py`**

```python
"""Claude on the Foundry account's /anthropic route (Anthropic Messages API).

Shared by the online faithfulness gate (ADR-0009) and synthetic test-item
authoring (ADR-0010). Entra-only: tokens use the AI Foundry scope, not the
Cognitive Services scope used by the /openai route.
"""
from __future__ import annotations

from typing import Callable

AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def build_claude_complete_fn(
    settings, max_tokens: int = 2048
) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """`complete(prompt) -> str` against the Claude judge deployment.

    The client is rebuilt per call so the (cached, auto-refreshed) Entra bearer
    token from azure-identity is always current — client construction is cheap
    next to the model call.
    """
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required to call Claude (ADR-0009); set it in .env"
        )
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
```

- [ ] **Step 4: Rewrite `build_ragas_faithfulness` in `src/ragpipe/guardrail.py`** (replace lines 56-87)

```python
def build_ragas_faithfulness(settings) -> MetricFn:
    """Faithfulness gate judged by the Claude deployment (ADR-0009).

    The gate decides what users see, so it must not share a family with the
    generator: a judge from the generator's own family is systematically
    lenient on its own outputs (self-preference bias). Raises if JUDGE_MODEL
    is unset — silently falling back to the generator would recreate the
    circular setup this replaces.
    """
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required: the faithfulness gate is judged by the "
            "Claude deployment (ADR-0009); set it in .env"
        )
    return _build_claude_faithfulness(settings)


def _build_claude_faithfulness(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_anthropic import ChatAnthropic
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    from ragpipe.embeddings import anthropic_endpoint_from_project
    from ragpipe.foundry_claude import AI_FOUNDRY_SCOPE

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    def _metric() -> Faithfulness:
        # Rebuilt per scoring call: Entra bearer tokens expire (~1h) and
        # ChatAnthropic fixes headers at construction. azure-identity caches the
        # token, so this is cheap until a refresh is actually due.
        judge = LangchainLLMWrapper(
            ChatAnthropic(
                model=settings.judge_model,
                base_url=base_url,
                api_key="unused-entra-bearer",  # real auth is the header below
                default_headers={"Authorization": f"Bearer {token_provider()}"},
                max_tokens=4096,
                temperature=0,
            )
        )
        return Faithfulness(llm=judge)

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await _metric().single_turn_ascore(sample))

    return metric_fn
```

Delete the now-unused `from ragpipe.embeddings import openai_endpoint_from_project` import inside the old function body (it moves out with the rewrite).

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_faithfulness.py -q` — Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/foundry_claude.py src/ragpipe/guardrail.py tests/test_faithfulness.py
git commit -m "feat(guardrail): Claude judges the online faithfulness gate (ADR-0009)"
```

**LIVE-VERIFY NOTE (work laptop):** if `scripts/verify_judges.py` (Task 15) gets a 401/403 from the gate, the likely cause is the `x-api-key`/`Authorization` header combination; fall back to `auth_token` plumbing via a custom `httpx` client, or as a last resort the resource API key (`api_key=<key>` — documented as supported for Claude deployments). Record the outcome in ADR-0009.

### Task 6: DeepSeek as the offline RAGAS judge

**Files:**
- Modify: `src/ragpipe/eval/harness.py:127-156` (`_build_ragas_clients`)
- Test: `tests/eval/test_harness.py`

- [ ] **Step 1: Write failing test** (append to `tests/eval/test_harness.py`)

```python
def test_offline_judge_requires_offline_judge_model():
    import pytest

    from ragpipe.eval.harness import _build_ragas_clients

    class _Settings:
        foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
        foundry_chat_model = "gpt-5.4"
        foundry_embedding_model = "text-embedding-3-small"
        offline_judge_model = None

    with pytest.raises(ValueError, match="OFFLINE_JUDGE_MODEL"):
        _build_ragas_clients(_Settings())
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/eval/test_harness.py -q` — Expected: new test FAILS

- [ ] **Step 3: Modify `_build_ragas_clients`** (replace lines 127-156)

```python
def _build_ragas_clients(settings):
    """(llm, embeddings) RAGAS wrappers: DeepSeek offline judge + OpenAI embeddings.

    The offline judge is the third family (ADR-0009) — independent of both the
    gpt generator and the Claude online gate, so offline scores are not
    self-judged and not gate-saturated by the same model instance. DeepSeek is
    sold directly by Azure and served on the OpenAI-compatible route of the
    services.ai.azure.com host, so the AzureChatOpenAI client works unchanged.
    Embeddings (answer_relevancy only) stay on text-embedding-3-small: they are
    a measurement primitive, not a judge.
    """
    if not settings.offline_judge_model:
        raise ValueError(
            "OFFLINE_JUDGE_MODEL is required: offline RAGAS metrics are judged "
            "by the DeepSeek deployment (ADR-0009); set it in .env"
        )
    return _build_ragas_clients_live(settings)


def _build_ragas_clients_live(settings):  # pragma: no cover - live Azure wiring
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from ragpipe.embeddings import (
        openai_endpoint_from_project,
        services_endpoint_from_project,
    )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    # No explicit temperature: DeepSeek-V4-Pro is a reasoning model and, like
    # other reasoning deployments on this route, may reject sampling overrides.
    llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=services_endpoint_from_project(settings.foundry_project_endpoint),
            azure_deployment=settings.offline_judge_model,
            api_version="2024-10-21",
            azure_ad_token_provider=token_provider,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_endpoint=openai_endpoint_from_project(settings.foundry_project_endpoint),
            azure_deployment=settings.foundry_embedding_model,
            api_version="2024-10-21",
            azure_ad_token_provider=token_provider,
        )
    )
    return llm, emb
```

Side effect to note in the commit: `build_synthetic_generator` (live `TESTSET_MODE=synthetic` path) calls `_build_ragas_clients`, so ad-hoc synthetic generation now uses the DeepSeek judge model as its author — still family-separated from the generator/embeddings; the *committed* testset path is Claude-authored (Task 13/14, ADR-0010).

**LIVE-VERIFY NOTE (work laptop):** DeepSeek-V4-Pro has no tool calling, so RAGAS must use its prompt-based JSON parsing path (the default for `LangchainLLMWrapper` in ragas 0.4.3); it also returns `reasoning_content` alongside `content`. `scripts/verify_judges.py` (Task 15) exercises this route — if structured-output parsing fails there, that is the first place to look.

- [ ] **Step 4: Run tests** — `uv run pytest tests/eval -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/harness.py tests/eval/test_harness.py
git commit -m "feat(eval): DeepSeek-V4-Pro judges the offline RAGAS suite (ADR-0009)"
```

## Phase C — Query-pipeline behavior

### Task 7: Hybrid semantic rerank (keep dense-only candidates) + top_k override

**Files:**
- Modify: `src/ragpipe/retrieval/rerank.py`
- Test: `tests/retrieval/test_rerank.py`

- [ ] **Step 1: Write failing tests** (append to `tests/retrieval/test_rerank.py`)

```python
def test_reranker_sends_hybrid_vector_query_when_embed_fn_present():
    fused = [_chunk("a"), _chunk("b")]
    client = FakeSearchClient([_doc("a", 2.0), _doc("b", 1.0)])
    reranker = SemanticReranker(
        client, semantic_config="default-semantic", top_k=2,
        embed_fn=lambda q: [0.1, 0.2],
    )

    reranker.rerank("query", fused)

    vqs = client.last_kwargs["vector_queries"]
    assert vqs is not None and len(vqs) == 1
    assert list(vqs[0].vector) == [0.1, 0.2]
    # stage-1 recall must cover every fused candidate, not just top_k
    assert vqs[0].k_nearest_neighbors == len(fused)
    # the lexical leg is still present (semantic reranker needs the text query)
    assert client.last_kwargs["search_text"] == "query"


def test_reranker_top_k_override_widens_window():
    fused = [_chunk("a"), _chunk("b"), _chunk("c")]
    client = FakeSearchClient([_doc("a", 3.0), _doc("b", 2.0), _doc("c", 1.0)])
    reranker = SemanticReranker(client, semantic_config="default-semantic", top_k=1)

    out = reranker.rerank("q", fused, top_k=3)

    assert client.last_kwargs["top"] == 3
    assert len(out) == 3
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/retrieval/test_rerank.py -q` — Expected: new tests FAIL (unexpected kwargs)

- [ ] **Step 3: Implement** (replace the `SemanticReranker` class in `src/ragpipe/retrieval/rerank.py`)

```python
from typing import Any, Callable

from azure.search.documents.models import VectorizedQuery
```

```python
class SemanticReranker:
    def __init__(
        self,
        client: Searchable,
        semantic_config: str,
        top_k: int = 5,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._client = client
        self._semantic_config = semantic_config
        self._top_k = top_k
        self._embed = embed_fn

    def rerank(
        self, query: str, fused: list[Chunk], top_k: int | None = None
    ) -> list[Chunk]:
        if not fused:
            return []
        k = top_k or self._top_k
        ids = [c.id for c in fused]
        # Semantic ranking is two-stage: stage 1 retrieves candidates, stage 2
        # re-scores them. With search_text alone, stage 1 is BM25 — a fused
        # candidate with zero lexical overlap (dense-only) never matches and is
        # silently dropped despite passing the id filter. Adding the vector leg
        # makes stage 1 hybrid, so every fused candidate is reachable.
        vector_queries = None
        if self._embed is not None:
            vector_queries = [
                VectorizedQuery(
                    vector=self._embed(query),
                    k_nearest_neighbors=len(ids),
                    fields="content_vector",
                )
            ]
        results = self._client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
            filter=_quote_ids(ids),
            vector_queries=vector_queries,
            top=k,
            select=["id", "title", "url", "content"],
        )
        return [_to_reranked_chunk(d) for d in results][:k]
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/retrieval -q` — Expected: PASS (old tests untouched: `embed_fn` defaults to None → `vector_queries=None` kwarg; the existing `FakeSearchClient` accepts arbitrary kwargs)

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "fix(rerank): hybrid stage-1 query so dense-only candidates survive; top_k override"
```

### Task 8: Judge outage abstains immediately (no retry burn)

**Files:**
- Modify: `src/ragpipe/guardrail.py:96-107` (`decide_next`)
- Test: `tests/test_loop_policy.py`

- [ ] **Step 1: Update the policy test** (replace `test_failed_score_none_treated_as_below_threshold` in `tests/test_loop_policy.py:19-22`)

```python
def test_judge_failure_exhausts_immediately():
    # fail-closed AND fail-fast: regeneration cannot fix judge infrastructure,
    # so a missing score abstains without burning generator/rerank retries.
    assert decide_next(score=None, threshold=0.7, attempt=0, max_retries=2) is LoopDecision.EXHAUSTED
    assert decide_next(score=None, threshold=0.7, attempt=2, max_retries=2) is LoopDecision.EXHAUSTED
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_loop_policy.py -q` — Expected: FAIL (None at attempt 0 currently RETRYs)

- [ ] **Step 3: Implement** (replace `decide_next` body, `src/ragpipe/guardrail.py:96-107`)

```python
def decide_next(
    score: float | None, threshold: float, attempt: int, max_retries: int
) -> LoopDecision:
    """Decide whether to accept the answer, retry, or give up.

    A missing score (judge failure) is fail-closed AND fail-fast: it can never
    PASS, and it goes straight to EXHAUSTED — retrying regenerates the answer,
    which cannot fix a judge infrastructure failure and only multiplies cost.
    """
    if score is None:
        return LoopDecision.EXHAUSTED
    if score >= threshold:
        return LoopDecision.PASS
    if attempt < max_retries:
        return LoopDecision.RETRY
    return LoopDecision.EXHAUSTED
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_loop_policy.py tests/test_workflow.py -q` — Expected: loop-policy PASS; if a workflow test fed `None` scores it is updated in Task 10

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/guardrail.py tests/test_loop_policy.py
git commit -m "feat(guardrail): judge outage abstains immediately instead of burning retries"
```

### Task 9: Corrective generation prompt

**Files:**
- Modify: `src/ragpipe/generate.py`
- Test: `tests/test_generate.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_generate.py`)

```python
def test_prompt_without_previous_answer_has_no_corrective_block():
    prompt = build_grounding_prompt("q", [_chunk("a", "Alpha.")])
    assert "previous_answer" not in prompt


def test_prompt_with_previous_answer_includes_corrective_block():
    prompt = build_grounding_prompt(
        "q", [_chunk("a", "Alpha.")], previous_answer="Bad claim."
    )
    assert "<previous_answer>" in prompt
    assert "Bad claim." in prompt
    assert "could not be verified" in prompt


@pytest.mark.asyncio
async def test_generator_threads_previous_answer_into_prompt():
    agent = FakeAgent("better answer")
    gen = Generator(agent)
    await gen.generate("q", [_chunk("a", "Alpha.")], previous_answer="old answer")
    assert "old answer" in agent.last_prompt
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_generate.py -q` — Expected: FAIL (unexpected kwarg)

- [ ] **Step 3: Implement** (replace `build_grounding_prompt` and `Generator.generate` in `src/ragpipe/generate.py`)

```python
CORRECTIVE_INSTRUCTION = (
    "Your previous answer (below) contained claims that could not be verified "
    "against the sources. Write a new answer using ONLY claims directly "
    "supported by the numbered sources. If the sources do not contain the "
    "answer, say you don't know.\n\n"
    "<previous_answer>\n{previous_answer}\n</previous_answer>\n\n"
)


def build_grounding_prompt(
    query: str, chunks: list[Chunk], previous_answer: str | None = None
) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({c.url}) {c.content}" for i, c in enumerate(chunks)
    )
    corrective = (
        CORRECTIVE_INSTRUCTION.format(previous_answer=previous_answer)
        if previous_answer
        else ""
    )
    return (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources do not contain the answer, "
        "say you don't know.\n\n"
        f"{corrective}"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
```

```python
    async def generate(
        self, query: str, chunks: list[Chunk], previous_answer: str | None = None
    ) -> str:
        prompt = build_grounding_prompt(query, chunks, previous_answer)
        result = await self._agent.run(prompt)
        return result.text
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_generate.py -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/generate.py tests/test_generate.py
git commit -m "feat(generate): corrective retry prompt carries the rejected answer"
```

### Task 10: Workflow — widened retries, threaded previous answer, directive abstention

**Files:**
- Modify: `src/ragpipe/models.py` (add `abstained` to `PipelineState`)
- Modify: `src/ragpipe/workflow.py:10-72`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Rewrite the workflow tests** (replace the contents of `tests/test_workflow.py` above `build_viz_workflow`-related tests — the whole file currently shown below is replaced)

```python
import pytest

from ragpipe.models import Chunk, PipelineState
from ragpipe.workflow import ABSTENTION_ANSWER, PipelineDeps, run_pipeline


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=f"content-{cid}")


def _deps(score_sequence, rerank_calls=None, generate_calls=None):
    """Deps whose scorer returns scores from a sequence per attempt; optionally
    records (top_k) per rerank call and (previous_answer) per generate call."""
    scores = iter(score_sequence)

    def rerank(q, fused, k):
        if rerank_calls is not None:
            rerank_calls.append(k)
        return fused[:k]

    def generate(q, chunks, previous_answer):
        if generate_calls is not None:
            generate_calls.append(previous_answer)
        return f"answer for {q} (attempt {len(generate_calls or [0])})"

    return PipelineDeps(
        dense=lambda q: [_chunk("a"), _chunk("b")],
        bm25=lambda q: [_chunk("b"), _chunk("c")],
        rerank=rerank,
        generate=generate,
        score=lambda q, answer, chunks: next(scores),
        threshold=0.7,
        max_retries=2,
        top_k=2,
        rerank_widen_step=3,
    )


@pytest.mark.asyncio
async def test_pipeline_passes_first_try():
    state = await run_pipeline("what is RRF?", _deps([0.9]))
    assert isinstance(state, PipelineState)
    assert state.faithfulness == 0.9
    assert state.attempt == 0
    assert state.low_confidence is False
    assert state.abstained is False
    stages = [e.stage for e in state.trace]
    assert stages[:4] == ["dense", "bm25", "rrf", "rerank"]


@pytest.mark.asyncio
async def test_pipeline_loops_then_passes():
    state = await run_pipeline("q", _deps([0.4, 0.85]))
    assert state.attempt == 1
    assert state.faithfulness == 0.85
    assert state.low_confidence is False
    assert state.abstained is False


@pytest.mark.asyncio
async def test_retry_widens_rerank_window():
    calls = []
    await run_pipeline("q", _deps([0.4, 0.85], rerank_calls=calls))
    assert calls == [2, 5]  # top_k + widen_step * attempt


@pytest.mark.asyncio
async def test_retry_threads_previous_answer():
    calls = []
    await run_pipeline("q", _deps([0.4, 0.85], generate_calls=calls))
    assert calls[0] is None
    assert calls[1] is not None and "answer for q" in calls[1]


@pytest.mark.asyncio
async def test_exhaustion_abstains_with_directive_answer():
    state = await run_pipeline("q", _deps([0.1, 0.2, 0.3]))
    assert state.attempt == 2
    assert state.low_confidence is True
    assert state.abstained is True
    assert state.answer == ABSTENTION_ANSWER
    # the suppressed answer is preserved in the trace for debugging
    abstain_events = [e for e in state.trace if e.stage == "abstain"]
    assert len(abstain_events) == 1
    assert "answer for q" in abstain_events[0].data["suppressed_answer"]


@pytest.mark.asyncio
async def test_judge_exception_abstains_without_retry():
    def boom(q, a, c):
        raise RuntimeError("judge down")

    deps = _deps([0.9])
    deps.score = boom
    state = await run_pipeline("q", deps)
    assert state.attempt == 0  # no retries burned
    assert state.abstained is True
    assert state.answer == ABSTENTION_ANSWER
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_workflow.py -q` — Expected: FAIL (PipelineDeps has no `top_k`, ABSTENTION_ANSWER missing)

- [ ] **Step 3: Add `abstained` to `PipelineState`** (`src/ragpipe/models.py`, after `low_confidence`, line 32)

```python
    # Directive guardrail (ADR-0009): when retries exhaust, the answer is
    # replaced with a fixed abstention and this flag is set. The suppressed
    # answer survives in the trace only.
    abstained: bool = False
```

- [ ] **Step 4: Rewrite the loop in `src/ragpipe/workflow.py`** (replace lines 10-72: signatures, deps, run_pipeline)

```python
# Callable stage signatures (sync or async tolerated via _maybe_await).
DenseFn = Callable[[str], list[Chunk]]
Bm25Fn = Callable[[str], list[Chunk]]
RerankFn = Callable[[str, list[Chunk], int], list[Chunk]]
GenerateFn = Callable[[str, list[Chunk], str | None], object]
ScoreFn = Callable[[str, str, list[Chunk]], object]

# Returned verbatim when the guardrail exhausts: the directive abstention
# (ADR-0009). Consumers get this text instead of the unfaithful answer.
ABSTENTION_ANSWER = (
    "I don't have enough grounded information in the indexed documentation "
    "to answer this question reliably."
)


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass
class PipelineDeps:
    dense: DenseFn
    bm25: Bm25Fn
    rerank: RerankFn
    generate: GenerateFn
    score: ScoreFn
    threshold: float = 0.7
    max_retries: int = 2
    rrf_k: int = 60
    top_k: int = 5
    # Each retry widens the rerank window by this many chunks: the most common
    # faithfulness failure is the needed chunk sitting just below the cut.
    rerank_widen_step: int = 3


async def run_pipeline(query: str, deps: PipelineDeps) -> PipelineState:
    state = PipelineState(query=query)

    state.dense = await _maybe_await(deps.dense(query))
    state.add_trace("dense", {"ids": [c.id for c in state.dense]})
    state.bm25 = await _maybe_await(deps.bm25(query))
    state.add_trace("bm25", {"ids": [c.id for c in state.bm25]})

    # Retrieval legs and fusion are fixed across attempts — compute once.
    state.fused = reciprocal_rank_fusion(state.dense, state.bm25, k=deps.rrf_k)
    state.add_trace("rrf", {"ids": [c.id for c in state.fused]})

    previous_answer: str | None = None
    while True:
        k = deps.top_k + deps.rerank_widen_step * state.attempt
        state.reranked = await _maybe_await(deps.rerank(query, state.fused, k))
        state.add_trace(
            "rerank", {"ids": [c.id for c in state.reranked], "top_k": k}
        )

        state.answer = await _maybe_await(
            deps.generate(query, state.reranked, previous_answer)
        )
        state.add_trace("generate", {"answer": state.answer})
        previous_answer = state.answer

        try:
            score = await _maybe_await(deps.score(query, state.answer, state.reranked))
        except Exception:  # judge failure -> fail-closed
            score = None
        state.faithfulness = score
        state.add_trace("faithfulness", {"score": score, "attempt": state.attempt})

        decision = decide_next(
            score=score,
            threshold=deps.threshold,
            attempt=state.attempt,
            max_retries=deps.max_retries,
        )
        if decision is LoopDecision.PASS:
            return state
        if decision is LoopDecision.EXHAUSTED:
            state.low_confidence = True
            state.abstained = True
            state.add_trace(
                "abstain", {"suppressed_answer": state.answer, "score": score}
            )
            state.answer = ABSTENTION_ANSWER
            return state
        state.next_attempt()  # RETRY
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_workflow.py tests/test_loop_policy.py -q` — Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/models.py src/ragpipe/workflow.py tests/test_workflow.py
git commit -m "feat(workflow): widened retries with corrective feedback; directive abstention on exhaustion"
```

### Task 11: Wire it — app_wiring, API, dashboard surface

**Files:**
- Modify: `src/ragpipe/app_wiring.py`
- Modify: `app/api.py:50-57`
- Test: `tests/test_app_wiring.py`, `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_app_wiring.py`:

```python
def test_make_deps_threads_top_k_and_new_signatures():
    from ragpipe.app_wiring import make_deps

    class _S:
        faithfulness_threshold = 0.7
        max_retries = 2
        rrf_k = 60
        top_k = 4

    class _Rerank:
        def __init__(self):
            self.k = None

        def rerank(self, q, fused, top_k=None):
            self.k = top_k
            return fused

    class _Gen:
        def __init__(self):
            self.prev = "sentinel"

        async def generate(self, q, chunks, previous_answer=None):
            self.prev = previous_answer
            return "a"

    class _Id:
        def retrieve(self, q):
            return []

        def score(self, q, a, c):
            return 1.0

    rr, gen = _Rerank(), _Gen()
    deps = make_deps(_S(), dense=_Id(), bm25=_Id(), reranker=rr, generator=gen, scorer=_Id())
    assert deps.top_k == 4
    deps.rerank("q", [], 9)
    assert rr.k == 9
```

Append to `tests/test_api.py`:

```python
def test_run_reports_abstention():
    from fastapi.testclient import TestClient

    from app import api
    from ragpipe.models import PipelineState

    async def fake_pipeline(q):
        return PipelineState(query=q, answer="abstained text", abstained=True, low_confidence=True)

    api.app.dependency_overrides[api.get_pipeline_fn] = lambda: fake_pipeline
    try:
        resp = TestClient(api.app).post("/run", json={"query": "x"})
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True
    finally:
        api.app.dependency_overrides.clear()
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_app_wiring.py tests/test_api.py -q` — Expected: FAIL

- [ ] **Step 3: Update `make_deps` and `build_pipeline_fn` in `src/ragpipe/app_wiring.py`**

Replace `make_deps` (lines 10-27):

```python
def make_deps(
    settings: Settings,
    dense: Any,
    bm25: Any,
    reranker: Any,
    generator: Any,
    scorer: Any,
) -> PipelineDeps:
    return PipelineDeps(
        dense=lambda q: dense.retrieve(q),
        bm25=lambda q: bm25.retrieve(q),
        rerank=lambda q, fused, k: reranker.rerank(q, fused, top_k=k),
        generate=lambda q, chunks, prev: generator.generate(q, chunks, prev),
        score=lambda q, a, c: scorer.score(q, a, c),
        threshold=settings.faithfulness_threshold,
        max_retries=settings.max_retries,
        rrf_k=settings.rrf_k,
        top_k=settings.top_k,
    )
```

In `build_pipeline_fn`, replace the embed + retriever construction (lines 50-62):

```python
    from functools import lru_cache

    # One embedding per query string, shared by the dense leg and the hybrid
    # rerank call (and across guardrail retries).
    embed_raw = build_embed_fn(settings)

    @lru_cache(maxsize=128)
    def _embed_cached(text: str) -> tuple[float, ...]:
        return tuple(embed_raw(text))

    def embed(text: str) -> list[float]:
        return list(_embed_cached(text))

    agent = FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=cred,
    )
    deps = make_deps(
        settings,
        # Legs fetch the wider candidate pool; rerank narrows to top_k (widened
        # per retry by the workflow).
        dense=DenseRetriever(search, embed, settings.candidate_pool),
        bm25=BM25Retriever(search, settings.candidate_pool),
        reranker=SemanticReranker(
            search, SEMANTIC_CONFIG_NAME, settings.top_k, embed_fn=embed
        ),
        generator=Generator(agent),
        scorer=FaithfulnessScorer(build_ragas_faithfulness(settings)),
    )
```

(The old `embed = build_embed_fn(settings)` line and its comment are replaced by the cached pair above.)

- [ ] **Step 4: Add `abstained` to the API response** (`app/api.py`, in the `/run` return dict after `"lowConfidence"`)

```python
        "abstained": state.abstained,
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_app_wiring.py tests/test_api.py tests/test_dashboard_helpers.py -q` — Expected: PASS (dashboard helpers consume `PipelineState` fields that still exist; if a helper test snapshots the `/run` payload keys, add `abstained`)

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/app_wiring.py app/api.py tests/test_app_wiring.py tests/test_api.py
git commit -m "feat: wire candidate pool, hybrid rerank embed, retry signatures; expose abstained in /run"
```

## Phase D — Eval & testset

### Task 12: Abstention as a first-class eval metric

**Files:**
- Modify: `src/ragpipe/eval/harness.py` (`EvalRecord`, `run_harness`, `build_ragas_evaluator`)
- Test: `tests/eval/test_harness.py`

- [ ] **Step 1: Write failing tests** (append to `tests/eval/test_harness.py`)

```python
@pytest.mark.asyncio
async def test_harness_records_abstention_metric():
    from ragpipe.eval.testset import TestItem

    async def pipeline_fn(q):
        s = PipelineState(query=q, answer="abstained")
        s.abstained = True
        return s

    async def evaluator_fn(records):
        return records

    items = [TestItem(question="q", ground_truth="g", ground_truth_context="http://u")]
    records = await run_harness(items, pipeline_fn, evaluator_fn)
    assert records[0].abstained is True
    assert records[0].metrics["abstained"] == 1.0


def test_aggregate_reports_abstention_rate():
    r1 = EvalRecord(question="a", answer="x", contexts=[], ground_truth="g",
                    metrics={"abstained": 1.0})
    r2 = EvalRecord(question="b", answer="y", contexts=[], ground_truth="g",
                    metrics={"abstained": 0.0})
    assert aggregate([r1, r2])["abstained"] == 0.5
```

(Match the existing import style at the top of the file — `EvalRecord`, `aggregate`, `run_harness`, `PipelineState`, `pytest` are already imported there; add any missing ones.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/eval/test_harness.py -q` — Expected: FAIL (`abstained` attribute)

- [ ] **Step 3: Implement in `src/ragpipe/eval/harness.py`**

Add to `EvalRecord` (after `tags`, line 33):

```python
    # Directive guardrail outcome (ADR-0009): True when the pipeline abstained.
    # Mirrored into metrics["abstained"] so aggregate()/aggregate_by_tag()
    # report the abstention rate alongside every other metric for free.
    abstained: bool = False
```

In `run_harness`, set the field and the metric (modify the `EvalRecord(...)` construction and the metrics update, lines 67-79):

```python
        record = EvalRecord(
            question=item.question,
            answer=state.answer,
            contexts=[c.content for c in state.reranked],
            ground_truth=item.ground_truth,
            stage_contexts={s: [c.content for c in cs] for s, cs in by_stage.items()},
            stage_urls={s: [c.url for c in cs] for s, cs in by_stage.items()},
            tags=item.tags,
            abstained=state.abstained,
        )
        # Deterministic metrics are free — always computed, no toggle.
        record.metrics["abstained"] = float(state.abstained)
        record.metrics.update(
            stage_retrieval_metrics(record.stage_urls, item.ground_truth_context)
        )
```

In `build_ragas_evaluator`, score answer-level metrics on answered records only (replace the body of `evaluator_fn` from the `ds = Dataset.from_list(...)` line through the metric write-back, lines 180-202):

```python
        llm, emb = _build_ragas_clients(settings)

        def _row(r: EvalRecord) -> dict:
            return {
                "question": r.question,
                "answer": r.answer,
                "contexts": r.contexts,
                "ground_truth": r.ground_truth,
            }

        # Answer-level metrics are meaningless on the fixed abstention text —
        # score them on answered records only. The abstention rate itself is
        # already in metrics["abstained"]. Context metrics depend on retrieval,
        # not the answer, so they are scored for every record.
        answered = [i for i, r in enumerate(records) if not r.abstained]
        if answered:
            ds = Dataset.from_list([_row(records[i]) for i in answered])
            result = evaluate(
                ds, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=emb
            )
            df = result.to_pandas()
            for j, i in enumerate(answered):
                for metric in ["faithfulness", "answer_relevancy"]:
                    if metric in df.columns:
                        records[i].metrics[metric] = float(df.iloc[j][metric])

        ds_all = Dataset.from_list([_row(r) for r in records])
        result = evaluate(
            ds_all, metrics=[context_precision, context_recall], llm=llm, embeddings=emb
        )
        df = result.to_pandas()
        for i, r in enumerate(records):
            for metric in ["context_precision", "context_recall"]:
                if metric in df.columns:
                    r.metrics[metric] = float(df.iloc[i][metric])
        return records
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/eval -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/harness.py tests/eval/test_harness.py
git commit -m "feat(eval): abstention rate as a first-class metric; answer metrics scored on answered items only"
```

### Task 13: Fix the silent all-zero metrics in live synthetic mode

**Files:**
- Modify: `src/ragpipe/eval/testset.py` (extract a pure `rows_to_items`, fix gold label)
- Test: `tests/eval/test_testset.py`

- [ ] **Step 1: Write failing tests** (append to `tests/eval/test_testset.py`)

```python
from ragpipe.eval.testset import rows_to_items


def test_rows_to_items_recovers_provenance_url_and_tags():
    docs = [
        {"content": "Azure AI Search supports hybrid retrieval over indexes.", "url": "http://learn/a"},
        {"content": "Cosmos DB bulk executor moves documents fast.", "url": "http://learn/b"},
    ]
    rows = [
        {
            "user_input": "How fast is bulk import?",
            "reference": "Fast.",
            "reference_contexts": ["Cosmos DB bulk executor moves documents fast."],
        }
    ]
    items = rows_to_items(rows, docs)
    assert len(items) == 1
    assert items[0].ground_truth_context == "http://learn/b"  # a URL, never chunk text
    assert items[0].tags == ("synthetic",)


def test_rows_to_items_drops_unrecoverable_rows():
    rows = [
        {"user_input": "q", "reference": "a", "reference_contexts": ["no such text"]},
        {"user_input": "q2", "reference": "", "reference_contexts": ["irrelevant"]},
    ]
    assert rows_to_items(rows, docs=[{"content": "other", "url": "http://x"}]) == []
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/eval/test_testset.py -q` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** (in `src/ragpipe/eval/testset.py`, add above `build_synthetic_generator`; then simplify the generator)

```python
def rows_to_items(rows: list[dict], docs: list[dict]) -> list[TestItem]:
    """Map RAGAS testset rows to TestItems with provenance gold URLs.

    The gold label MUST be a page URL (ADR-0002) — the harness passes it to
    URL-match metrics, so chunk text here silently scores hit_rate/mrr = 0.
    RAGAS only returns the source chunk text; recover the URL by matching the
    chunk back to the seed docs. Unrecoverable rows are dropped: a wrong gold
    label is worse than a smaller testset.
    """
    items: list[TestItem] = []
    for row in rows:
        probe = (row.get("reference_contexts") or [""])[0][:200]
        url = next((d["url"] for d in docs if probe and probe in d["content"]), "")
        if not url or not row.get("reference"):
            continue
        items.append(
            TestItem(
                question=row["user_input"],
                ground_truth=row["reference"],
                ground_truth_context=url,
                tags=("synthetic",),
            )
        )
    return items
```

In `build_synthetic_generator`'s `synthetic_fn`, replace the `items` loop (lines 76-85) with:

```python
        return rows_to_items(dataset.to_list(), corpus_docs)
```

Also update `scripts/generate_synthetic_testset.py` later (Task 14 rewrites it entirely).

- [ ] **Step 4: Run tests** — `uv run pytest tests/eval/test_testset.py -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/testset.py tests/eval/test_testset.py
git commit -m "fix(eval): synthetic mode gold labels are provenance URLs, not chunk text"
```

### Task 14: Claude-authored synthetic candidates from named pages

**Files:**
- Create: `src/ragpipe/eval/synthetic.py`
- Rewrite: `scripts/generate_synthetic_testset.py`
- Test: `tests/eval/test_synthetic.py` (new)

- [ ] **Step 1: Write failing tests** (create `tests/eval/test_synthetic.py`)

```python
from ragpipe.eval.synthetic import (
    content_word_overlap,
    make_candidates,
    parse_candidates,
)

DOC = (
    "Semantic ranking in Azure AI Search re-scores an initial result set "
    "using deep learning models to improve relevance of the top results."
)


def test_overlap_high_for_verbatim_question():
    q = "How does semantic ranking re-score the initial result set?"
    assert content_word_overlap(q, DOC) > 0.7


def test_overlap_low_for_user_phrased_question():
    q = "Can the engine make my best hits float upward automatically?"
    assert content_word_overlap(q, DOC) < 0.4


def test_parse_candidates_tolerates_fenced_json():
    raw = 'Here you go:\n```json\n[{"question": "q1", "ground_truth": "a1"}]\n```'
    assert parse_candidates(raw) == [{"question": "q1", "ground_truth": "a1"}]


def test_parse_candidates_drops_malformed_entries():
    raw = '[{"question": "q1"}, {"question": "q2", "ground_truth": "a2"}]'
    assert parse_candidates(raw) == [{"question": "q2", "ground_truth": "a2"}]


def test_make_candidates_screens_verbatim_and_stamps_provenance():
    def fake_complete(prompt):
        return (
            '[{"question": "How does semantic ranking re-score the initial '
            'result set using deep learning?", "ground_truth": "It re-scores."},'
            ' {"question": "Can my best hits float upward automatically?",'
            ' "ground_truth": "Yes, via semantic ranking."}]'
        )

    rows = make_candidates(fake_complete, url="http://learn/sem", document=DOC, n=2)
    # the verbatim-echo question is screened out; the user-phrased one survives
    assert len(rows) == 1
    assert rows[0]["question"].startswith("Can my best hits")
    assert rows[0]["ground_truth_context"] == "http://learn/sem"
    assert rows[0]["tags"] == ["synthetic"]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/eval/test_synthetic.py -q` — Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Create `src/ragpipe/eval/synthetic.py`**

```python
"""Claude-authored synthetic test-item candidates (ADR-0010).

Family separation: the question author (Claude) is a different model family
from the generator (gpt) and the embeddings (OpenAI), so the testset is not
phrased in the system-under-test's own idiom. Gold URLs are provenance — the
caller names the page — never recovered by substring matching. Candidates that
lexically echo the source document are screened out: verbatim questions are
trivially easy for BM25/embeddings and would understate exactly the
vocabulary-mismatch failures the hard subsets exist to measure.
"""
from __future__ import annotations

import json
import re
from typing import Callable

CANDIDATE_PROMPT = """You are writing evaluation questions for a documentation \
search system that indexes Microsoft Learn.

The document below is the page {url}.

<document>
{document}
</document>

Write {n} question-and-answer pairs about this document.
Rules:
- Phrase each question as a real user who has NOT read this page: everyday \
wording, no reuse of the page's phrasing, headings, or distinctive terms.
- Each answer must be fully supported by the document alone.
- Return ONLY a JSON array: [{{"question": "...", "ground_truth": "..."}}, ...]"""

# Small closed-class list; enough to keep overlap about content words.
_STOPWORDS = frozenset(
    "the a an and or but if then else when what which how why where who whom "
    "this that these those is are was were be been being have has had do does "
    "did can could should would may might must will shall with without within "
    "into onto from for of to in on at by as it its they them their there "
    "here you your our we us not no nor so than too very just about over "
    "under again further once more most other some such only own same".split()
)


def content_word_overlap(question: str, document: str) -> float:
    """Fraction of the question's content words that appear in the document.

    1.0 = every content word is lifted from the page (verbatim echo);
    low values = user phrasing. Words shorter than 4 chars are ignored.
    """
    words = {
        w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOPWORDS
    }
    if not words:
        return 1.0
    doc = document.lower()
    return sum(1 for w in words if w in doc) / len(words)


def parse_candidates(raw: str) -> list[dict]:
    """Extract the JSON array from a model response (tolerates code fences)."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except ValueError:
        return []
    return [
        {"question": d["question"], "ground_truth": d["ground_truth"]}
        for d in data
        if isinstance(d, dict) and d.get("question") and d.get("ground_truth")
    ]


def make_candidates(
    complete_fn: Callable[[str], str],
    url: str,
    document: str,
    n: int = 5,
    max_overlap: float = 0.6,
) -> list[dict]:
    """Screened candidate rows for one page, gold URL stamped by provenance."""
    raw = complete_fn(CANDIDATE_PROMPT.format(url=url, document=document[:30000], n=n))
    rows = []
    for cand in parse_candidates(raw):
        if content_word_overlap(cand["question"], document) > max_overlap:
            continue  # lexical echo — exactly what we must not reward
        rows.append(
            {
                "question": cand["question"],
                "ground_truth": cand["ground_truth"],
                "ground_truth_context": url,
                "tags": ["synthetic"],
            }
        )
    return rows


def page_text_from_index(search_client, url: str) -> str:  # pragma: no cover - live Azure
    """Reassemble a page's text from its indexed chunks (clean content only)."""
    safe = url.replace("'", "''")
    results = search_client.search(
        search_text="*",
        filter=f"url eq '{safe}'",
        select=["content", "chunk_id"],
        top=200,
    )
    chunks = sorted(results, key=lambda d: d.get("chunk_id", 0))
    return "\n\n".join(c["content"] for c in chunks)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/eval/test_synthetic.py -q` — Expected: PASS

- [ ] **Step 5: Rewrite `scripts/generate_synthetic_testset.py`**

```python
"""Generate synthetic test-item CANDIDATES with the Claude judge-family model.

Pages are named explicitly so gold URLs are provenance, not recovered (ADR-0010).
Prints screened candidate rows as JSON to stdout for manual review — nothing is
written to data/testset.jsonl by this script.

Usage:
    uv run python scripts/generate_synthetic_testset.py <url> [<url> ...] [--per-page N]
"""
import argparse
import json

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from ragpipe.config import Settings
from ragpipe.eval.synthetic import make_candidates, page_text_from_index
from ragpipe.foundry_claude import build_claude_complete_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+", help="corpus page URLs to author questions from")
    parser.add_argument("--per-page", type=int, default=5)
    args = parser.parse_args()

    settings = Settings.from_env()
    search = SearchClient(
        settings.search_endpoint, settings.search_index, DefaultAzureCredential()
    )
    complete = build_claude_complete_fn(settings)

    rows = []
    for url in args.urls:
        document = page_text_from_index(search, url)
        if not document:
            print(f"-- no indexed chunks for {url}; skipped", flush=True)
            continue
        rows.extend(make_candidates(complete, url=url, document=document, n=args.per_page))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full suite** — `uv run pytest -q` — Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/eval/synthetic.py scripts/generate_synthetic_testset.py tests/eval/test_synthetic.py
git commit -m "feat(eval): Claude-authored synthetic candidates from named pages with overlap screening (ADR-0010)"
```

### Task 15: Decoration hardening + live judge smoke script

**Files:**
- Modify: `src/ragpipe/context_gen.py`
- Modify: `src/ragpipe/ingest.py:191`
- Create: `scripts/verify_judges.py`
- Test: `tests/test_context_gen.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_context_gen.py`)

```python
def test_cache_key_includes_model(tmp_path):
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return "ctx"

    g1 = ContextGenerator(complete, cache_path=tmp_path / "c.json", model="gpt-4o")
    g1.generate("doc", "chunk")
    g2 = ContextGenerator(complete, cache_path=tmp_path / "c.json", model="gpt-5.4")
    g2.generate("doc", "chunk")
    # model change must miss the cache and re-call the LLM
    assert len(calls) == 2


def test_generation_failure_is_logged(tmp_path, capsys):
    def boom(prompt):
        raise RuntimeError("model rejected temperature")

    g = ContextGenerator(boom, cache_path=tmp_path / "c.json", max_retries=1)
    assert g.generate("doc", "chunk") == ""
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "temperature" in err
```

(Match the existing import style at the top of `tests/test_context_gen.py` — `ContextGenerator` is already imported there.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_context_gen.py -q` — Expected: FAIL

- [ ] **Step 3: Implement in `src/ragpipe/context_gen.py`**

Add `import sys` to the imports. Change `__init__` to accept the model (replace lines 34-45):

```python
    def __init__(
        self,
        complete_fn: Callable[[str], str],
        cache_path: str | Path = ".context_cache.json",
        max_retries: int = 2,
        model: str = "",
    ) -> None:
        self._complete = complete_fn
        self._cache_path = Path(cache_path)
        self._max_retries = max_retries
        self._model = model
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        self.fallback_count = 0
```

Make `_key` an instance method including the model (replace lines 58-61):

```python
    def _key(self, document: str, chunk: str) -> str:
        # Model is part of the key: a model swap (e.g. gpt-4o -> gpt-5.4) must
        # not serve contexts authored by the previous model (ADR-0005 covers
        # prompt changes via PROMPT_VERSION; this covers the silent case).
        payload = "\x00".join((PROMPT_VERSION, self._model, document, chunk))
        return hashlib.sha256(payload.encode()).hexdigest()
```

Log the swallowed exception (replace lines 70-74):

```python
        for _ in range(self._max_retries):
            try:
                context = self._complete(prompt).strip()
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort ingest
                print(
                    f"context_gen: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
```

Remove `temperature=0` from `build_context_complete_fn` (line 95) and update its docstring line to:

```python
    """`complete(prompt) -> str` over the account's /openai endpoint.

    No explicit temperature: gpt-5-family reasoning deployments reject
    non-default temperature values. Determinism comes from the content-addressed
    cache (ADR-0005), not from sampling parameters.
    """
```

- [ ] **Step 4: Pass the model at the call site** (`src/ragpipe/ingest.py:191`)

```python
    context_gen = ContextGenerator(
        build_context_complete_fn(settings), model=settings.foundry_chat_model
    )
```

- [ ] **Step 5: Create `scripts/verify_judges.py`**

```python
"""One-shot LIVE smoke for every model route (run on a machine with Azure access
BEFORE any ingest or eval run). Verifies: Claude gate scoring, Mistral offline
judge, gpt decoration call, raw Claude completion. Exits non-zero on failure."""
import asyncio
import sys

from ragpipe.config import Settings


def main() -> int:
    settings = Settings.from_env()
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            result = fn()
            print(f"PASS {name}: {str(result)[:120]}")
        except Exception as exc:  # noqa: BLE001 - report every route
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    def gate():
        from ragpipe.guardrail import build_ragas_faithfulness

        metric_fn = build_ragas_faithfulness(settings)
        return asyncio.run(
            metric_fn(
                question="What color is the sky?",
                answer="The sky is blue.",
                contexts=["The sky is blue during the day."],
            )
        )

    def offline_judge():
        from ragpipe.eval.harness import _build_ragas_clients

        llm, _ = _build_ragas_clients(settings)
        return llm.langchain_llm.invoke("Reply with exactly: OK").content

    def decoration():
        from ragpipe.context_gen import build_context_complete_fn

        return build_context_complete_fn(settings)("Reply with exactly: OK")

    def claude_raw():
        from ragpipe.foundry_claude import build_claude_complete_fn

        return build_claude_complete_fn(settings)("Reply with exactly: OK")

    check("claude gate (RAGAS faithfulness)", gate)
    check("deepseek offline judge", offline_judge)
    check("gpt decoration completion", decoration)
    check("claude raw completion", claude_raw)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run tests** — `uv run pytest tests/test_context_gen.py -q && uv run python -c "import scripts" 2>/dev/null; uv run python -m py_compile scripts/verify_judges.py && echo OK` — Expected: tests PASS, `OK`

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/context_gen.py src/ragpipe/ingest.py scripts/verify_judges.py tests/test_context_gen.py
git commit -m "fix(ingest): model-aware context cache, logged failures, no temperature; add live judge smoke script"
```

## Phase E — Documentation

### Task 16: ADR-0009, ADR-0010, ADR cross-refs, README

**Files:**
- Create: `docs/adr/0009-three-family-judge-split-directive-guardrail.md`
- Create: `docs/adr/0010-synthetic-test-data-policy.md`
- Modify: `docs/adr/0008-swedencentral-gpt54-claude-judge.md` (status note), `docs/adr/0006-baseline-before-treatment-evaluation.md` (addendum), `docs/adr/README.md` (index), `README.md`

- [ ] **Step 1: Write ADR-0009**

```markdown
# 0009 — Three-family judge split and directive guardrail

**Status:** Accepted (2026-06-10)

## Context

After ADR-0008, one model (gpt-5.4) still generated answers, judged them online,
judged them offline, and authored synthetic test items — self-preference bias in
every LLM-judged number, and a circular loop: the online guardrail retried
answers until they passed the same metric/model that offline eval then scored,
so offline faithfulness was saturated by construction. The guardrail was also
advisory: an exhausted retry loop returned the unfaithful answer with only a
lowConfidence flag.

## Decision

1. **Three families.** Generator: `gpt-5.4` (OpenAI). Online faithfulness gate:
   `claude-sonnet-4-6` (Anthropic, Messages API on the account's `/anthropic`
   route, Entra bearer with scope `https://ai.azure.com/.default`). Offline
   RAGAS suite: `DeepSeek-V4-Pro` (DeepSeek; sold directly by Azure —
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
   infrastructure failure abstains immediately (fail-closed and fail-fast).
3. **Retries change something.** Retrieval legs fetch CANDIDATE_POOL (15)
   candidates once; each retry widens the rerank window (top_k + 3·attempt) and
   feeds the rejected answer back via a corrective instruction.
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
  verify_judges.py): swap `OFFLINE_JUDGE_MODEL` and the bicep `format` to
  `'Mistral AI'` — no code changes needed.
- **Grok/Llama as third family:** grok is unavailable in swedencentral;
  Llama 3.3 70B requires marketplace serverless plumbing.
- **Retry with query rewriting:** the right feature in the wrong place; it
  belongs at the front of the pipeline for all queries, measured by eval.

## Consequences

- Cost/latency: Claude marketplace-billed tokens in the hot path (× retries);
  RAGAS faithfulness is multi-call. Keep MAX_RETRIES low; measure gate latency.
- Comparability: deterministic metrics remain comparable across this change;
  all LLM-judged numbers re-anchor (judge changed). The work-laptop protocol:
  run `scripts/verify_judges.py`, then a baseline eval on main, commit it as
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
```

- [ ] **Step 2: Write ADR-0010**

```markdown
# 0010 — Synthetic test data policy

**Status:** Accepted (2026-06-10)

## Context

The testset must grow (n=33 supports no statistically meaningful subset
comparison: binary hit_rate on a 7-item tag group has a 95% CI of roughly
±35pp), but synthetic generation has two failure modes: (1) distributional
bias — questions generated FROM chunks lexically echo them, trivially easy for
BM25/embeddings, understating exactly the vocabulary-mismatch failures the
decoration treatment targets; (2) family self-affinity — questions phrased in
the system-under-test's own idiom. Previous generation also recovered gold URLs
by fragile 200-char substring matching.

## Decision

1. **A different family authors questions:** Claude (`JUDGE_MODEL`) writes
   candidates; the generator (gpt) and embeddings (OpenAI) never author test
   items. Committed items come only from the curated path below.
2. **Provenance gold labels:** candidates are generated from explicitly named
   corpus pages (`scripts/generate_synthetic_testset.py <url> ...`), so the
   gold URL is known by construction, never recovered. (The legacy
   `TESTSET_MODE=synthetic` path inherits the DeepSeek offline-judge model as
   author — also family-separated.)
3. **Mechanical lexical screening:** candidates whose question shares >60% of
   content words with the source page are rejected (`content_word_overlap`),
   plus the existing manual screen before anything is committed.
4. **Human/synthetic monitoring:** items keep the `synthetic` tag; a
   persistent score gap between `synthetic` and `original` items IS the bias,
   made visible per release. Hard subsets (`paraphrase`, `lookalike`) remain
   preferentially hand-authored.
5. **Size target:** grow toward ~30 items per tag group (CI ≈ ±18pp), roughly
   40% human-written / 60% screened synthetic; report per-item paired flips
   (McNemar-style) when comparing runs, not just aggregate deltas.

## Alternatives rejected

- **Hand-authoring everything:** does not scale past ~50 items.
- **RAGAS TestsetGenerator over a corpus sample:** generator and gold-URL
  recovery are both biased/fragile (the previous approach; kept only as the
  legacy ad-hoc mode).
- **Embedding-similarity screening:** uses the system-under-test's own
  embedding space to judge distance — circular; word overlap is crude but
  independent and testable.

## Sources

- ARES (synthetic eval data + judge noise): https://arxiv.org/abs/2311.09476
- BEIR (lexical-overlap bias in IR benchmarks): https://arxiv.org/abs/2104.08663
- ADR-0002 (URL-match metrics), ADR-0006 (tagged subsets, baseline protocol)
```

- [ ] **Step 3: Cross-reference updates**

In `docs/adr/0008-swedencentral-gpt54-claude-judge.md`, change the status line to:

```markdown
**Status:** Accepted (2026-06-10) — judge wiring superseded by ADR-0009 (three-family split: the Claude deployment became the online gate; DeepSeek-V4-Pro was added as the offline judge)
```

In `docs/adr/0006-baseline-before-treatment-evaluation.md`, append at the end:

```markdown
## Addendum (2026-06-10)

ADR-0009 re-anchors all LLM-judged metrics (judge models changed) and adds
`abstained` to every report; ADR-0010 sets the synthetic-data policy and size
targets for the testset expansion in §1. The baseline protocol in §2 is
unchanged and still pending execution: `eval_baseline.json` must be produced
(on the work machine) from `main` *before* the ADR-0009 branch's first eval
run, judged with the same three-family configuration so the pair is comparable.
```

In `docs/adr/README.md`, add index entries for 0009 and 0010 following the existing list format.

- [ ] **Step 4: README updates** (`README.md`)

- In the architecture paragraph (lines 8-16): change "with a RAGAS faithfulness guardrail that retries on weak grounding" to "with a directive RAGAS faithfulness guardrail (judged by Claude) that widens retrieval and regenerates on weak grounding, and abstains when retries exhaust".
- In "Supported regions & models" (lines 99-115): update the `claude-sonnet-4-6` bullet from "provisioned for the upcoming judge-model split" to "judges the online faithfulness gate (`JUDGE_MODEL`)", and add a bullet: "`DeepSeek-V4-Pro` (preview, GlobalStandard, sold directly by Azure) — offline RAGAS judge (`OFFLINE_JUDGE_MODEL`); third family so offline scores are independent of both the generator and the gate (ADR-0009)".
- In the API section (line 72): add `abstained` to the listed `/run` response fields.
- In the Evaluate section (after line 91): add:

```markdown
Before the first eval run on a new machine: `uv run python scripts/verify_judges.py`
(smokes all three model routes + the decoration call).

Generate synthetic test-item candidates from pages you name (Claude-authored,
screened; see `docs/adr/0010`):

```bash
uv run python scripts/generate_synthetic_testset.py https://learn.microsoft.com/en-us/azure/search/semantic-search-overview --per-page 5
```
```

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest -q` — Expected: PASS

```bash
git add docs/adr/ README.md
git commit -m "docs: ADR-0009 (three-family judges, directive guardrail) + ADR-0010 (synthetic data policy)"
```

---

## Self-review checklist (done at planning time)

- Spec coverage: azd fix (T1), judge split incl. infra (T1/T2/T4/T5/T6), hybrid rerank (T7), retry widening + corrective prompt (T9/T10/T11), directive abstention (T8/T10/T11), abstention metric (T12), synthetic gold-URL bug (T13), Claude-authored synthetic from named pages (T14), decoration hardening + live verification (T15), ADRs/README (T16). ✓
- Known live-verification risks called out: DeepSeek bicep `format` string (T1 step 4), ChatAnthropic header auth (T5 note), DeepSeek prompt-based JSON parsing in RAGAS (T6 note), all four routes smoked by `scripts/verify_judges.py` (T15).
- Type consistency: `RerankFn(str, list[Chunk], int)` matches `make_deps` lambda and `SemanticReranker.rerank(query, fused, top_k)`; `GenerateFn(str, list[Chunk], str | None)` matches `Generator.generate`; `PipelineState.abstained` consumed by api/harness. ✓
