from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from ragpipe.models import Chunk


@dataclass(frozen=True)
class ClaimVerdict:
    statement: str
    verdict: bool
    reason: str


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float | None
    claims: tuple[ClaimVerdict, ...] = ()


MetricFn = Callable[..., Awaitable[float | None | FaithfulnessResult]]


def as_faithfulness_result(raw: float | None | FaithfulnessResult) -> FaithfulnessResult:
    """Normalize a scorer return (bare float/None or structured) to FaithfulnessResult."""
    if isinstance(raw, FaithfulnessResult):
        return raw
    return FaithfulnessResult(score=raw)


class FaithfulnessScorer:
    """Thin adapter around a RAGAS faithfulness metric callable."""

    def __init__(self, metric_fn: MetricFn) -> None:
        self._metric_fn = metric_fn

    async def score_detailed(
        self, query: str, answer: str, contexts: list[Chunk]
    ) -> FaithfulnessResult:
        raw = await self._metric_fn(
            question=query,
            answer=answer,
            contexts=[c.content for c in contexts],
        )
        return as_faithfulness_result(raw)

    async def score(self, query: str, answer: str, contexts: list[Chunk]) -> float | None:
        return (await self.score_detailed(query, answer, contexts)).score


def _ensure_ragas_importable() -> None:  # pragma: no cover
    """Work around two broken-import hazards before ``import ragas``.

    1. ``ragas.llms.base`` does ``from langchain_community.chat_models.vertexai
       import ChatVertexAI`` at module import time, but langchain-community 0.4.2
       (the version pinned in this project) removed that module path, so a plain
       ``import ragas`` raises ModuleNotFoundError before any of the symbols we
       use are reachable. We only ever drive RAGAS with an Azure judge, so we
       install a placeholder module that satisfies the import without pulling in
       Google Vertex. If a real Vertex integration is ever present, this is a
       no-op.

    2. ``import ragas`` pulls in ``langchain_core.runnables.passthrough``, which
       constructs ``RunnablePassthrough()`` at module-import time. Depending on
       the order langchain_core's pydantic models get built, that construction
       can raise ``ValidationError: name Field required`` and leave a
       half-imported ``langchain_openai`` cached in ``sys.modules`` that poisons
       every later import. Importing ``langchain_openai`` first builds those
       models in the order it expects, so the subsequent ragas import reuses the
       good cached classes instead of triggering the broken build order.
    """
    import importlib
    import sys
    import types

    importlib.import_module("langchain_openai")

    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
        return
    except ImportError:
        pass

    placeholder = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # minimal stand-in; only used to satisfy the import
        pass

    placeholder.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = placeholder


def prewarm_ragas_imports() -> None:
    """Build the langchain/RAGAS judge models once, at a clean import time.

    Streamlit re-executes the app script on every interaction ("reruns"). The
    faithfulness gate imports ``langchain_openai`` the first time a query runs,
    which lazily constructs langchain_core's ``RunnablePassthrough`` pydantic
    model. langchain_core defers annotation evaluation
    (``from __future__ import annotations``), and building that model for the
    *first* time during a rerun loses the ``name: str | None = None`` default and
    raises ``ValidationError: name Field required`` at
    ``runnables/passthrough.py`` import. Forcing the import once at module load —
    before Streamlit's first rerun — builds the model correctly and caches it for
    the process lifetime. Idempotent: the underlying imports are memoised by
    ``sys.modules``, so repeated calls are cheap no-ops.
    """
    _ensure_ragas_importable()


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


def build_ragas_faithfulness_detailed(settings) -> MetricFn:
    """Detailed RAGAS faithfulness gate with per-claim verdicts.

    This drives RAGAS 0.4.3's private Faithfulness seam so operators can inspect
    claim-level grounding decisions. The seam is intentionally version-coupled;
    the canary and seam-guard tests fail loudly on incompatible RAGAS upgrades.
    """
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required: the faithfulness gate is judged by a "
            "non-generator family (ADR-0009); set it in .env"
        )
    from ragpipe.foundry_judge import judge_provider

    if judge_provider(settings.judge_model) == "anthropic":
        return _build_claude_faithfulness_detailed(settings)
    return _build_openai_faithfulness_detailed(settings)


def _faithfulness_from_chat(judge_chat):  # pragma: no cover - live wiring
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    return Faithfulness(llm=LangchainLLMWrapper(judge_chat))


def _build_claude_judge_chat(settings, token_provider, base_url):  # pragma: no cover - live wiring
    from langchain_anthropic import ChatAnthropic

    from ragpipe.foundry_judge import JUDGE_MAX_RETRIES, JUDGE_TIMEOUT

    judge_chat = ChatAnthropic(
        model=settings.judge_model,
        base_url=base_url,
        api_key="placeholder",  # satisfies validation; nulled below
        default_headers={"Authorization": f"Bearer {token_provider()}"},
        max_tokens=4096,
        temperature=0,
        timeout=JUDGE_TIMEOUT,
        max_retries=JUDGE_MAX_RETRIES,
    )
    # The anthropic SDK sends X-Api-Key alongside any custom Authorization
    # header; a gateway that validates X-Api-Key first would 401. Clearing
    # the key on both underlying clients removes the header entirely
    # (verified against anthropic 0.109.1: absent from built requests).
    judge_chat._client.api_key = None
    judge_chat._async_client.api_key = None
    return judge_chat


def _build_openai_judge_chat(settings, token_provider):  # pragma: no cover - live wiring
    from langchain_openai import AzureChatOpenAI

    from ragpipe.embeddings import services_endpoint_from_project
    from ragpipe.foundry_judge import JUDGE_MAX_RETRIES, JUDGE_TIMEOUT

    # azure_ad_token_provider auto-refreshes, so the judge is built once (the
    # Anthropic path rebuilds per call because ChatAnthropic fixes headers at
    # construction). No explicit temperature: reasoning deployments on this
    # route may reject sampling overrides (matches the offline DeepSeek judge).
    # model= sets the request-body "model" field: sold-by-Azure servers (e.g.
    # DeepSeek's sglang) validate it and reject a null; Kimi tolerates either.
    return AzureChatOpenAI(
        azure_endpoint=services_endpoint_from_project(settings.foundry_project_endpoint),
        azure_deployment=settings.judge_model,
        model=settings.judge_model,
        api_version="2024-10-21",
        azure_ad_token_provider=token_provider,
        timeout=JUDGE_TIMEOUT,
        max_retries=JUDGE_MAX_RETRIES,
    )


async def _score_faithfulness_detailed(
    metric, *, question: str, answer: str, contexts: list[str]
) -> FaithfulnessResult:
    row = {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    statements = await metric._create_statements(row, None)
    if not statements.statements:
        return FaithfulnessResult(score=float("nan"), claims=())
    nli = await metric._create_verdicts(row, statements.statements, None)
    score = float(metric._compute_score(nli))
    claims = tuple(
        ClaimVerdict(
            statement=item.statement,
            verdict=bool(item.verdict),
            reason=item.reason,
        )
        for item in nli.statements
    )
    return FaithfulnessResult(score=score, claims=claims)


def _build_claude_faithfulness(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from ragas.dataset_schema import SingleTurnSample

    from ragpipe.embeddings import anthropic_endpoint_from_project
    from ragpipe.foundry_judge import AI_FOUNDRY_SCOPE

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    def _metric():
        # Rebuilt per scoring call: Entra bearer tokens expire (~1h) and
        # ChatAnthropic fixes headers at construction. azure-identity caches the
        # token, so this is cheap until a refresh is actually due.
        return _faithfulness_from_chat(
            _build_claude_judge_chat(settings, token_provider, base_url)
        )

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await _metric().single_turn_ascore(sample))

    return metric_fn


def _build_claude_faithfulness_detailed(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    from ragpipe.embeddings import anthropic_endpoint_from_project
    from ragpipe.foundry_judge import AI_FOUNDRY_SCOPE

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        metric = _faithfulness_from_chat(
            _build_claude_judge_chat(settings, token_provider, base_url)
        )
        return await _score_faithfulness_detailed(
            metric, question=question, answer=answer, contexts=contexts
        )

    return metric_fn


def _build_openai_faithfulness(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from ragas.dataset_schema import SingleTurnSample

    from ragpipe.embeddings import COGNITIVE_SERVICES_SCOPE

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    metric = _faithfulness_from_chat(_build_openai_judge_chat(settings, token_provider))

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await metric.single_turn_ascore(sample))

    return metric_fn


def _build_openai_faithfulness_detailed(settings) -> MetricFn:  # pragma: no cover - live wiring
    _ensure_ragas_importable()

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    from ragpipe.embeddings import COGNITIVE_SERVICES_SCOPE

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    metric = _faithfulness_from_chat(_build_openai_judge_chat(settings, token_provider))

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        return await _score_faithfulness_detailed(
            metric, question=question, answer=answer, contexts=contexts
        )

    return metric_fn


class LoopDecision(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    EXHAUSTED = "exhausted"


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
