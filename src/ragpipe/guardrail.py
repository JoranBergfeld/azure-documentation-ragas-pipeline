from __future__ import annotations

from enum import Enum
from typing import Awaitable, Callable

from ragpipe.models import Chunk

MetricFn = Callable[..., Awaitable[float]]


class FaithfulnessScorer:
    """Thin adapter around a RAGAS faithfulness metric callable."""

    def __init__(self, metric_fn: MetricFn) -> None:
        self._metric_fn = metric_fn

    async def score(self, query: str, answer: str, contexts: list[Chunk]) -> float:
        return await self._metric_fn(
            question=query,
            answer=answer,
            contexts=[c.content for c in contexts],
        )


def _ensure_ragas_importable() -> None:  # pragma: no cover
    """Work around a broken eager import in ragas 0.4.3.

    ``ragas.llms.base`` does ``from langchain_community.chat_models.vertexai
    import ChatVertexAI`` at module import time, but langchain-community 0.4.2
    (the version pinned in this project) removed that module path, so a plain
    ``import ragas`` raises ModuleNotFoundError before any of the symbols we
    use are reachable. We only ever drive RAGAS with an Azure judge, so we
    install a placeholder module that satisfies the import without pulling in
    Google Vertex. If a real Vertex integration is ever present, this is a
    no-op.
    """
    import importlib
    import sys
    import types

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
        judge_chat = ChatAnthropic(
            model=settings.judge_model,
            base_url=base_url,
            api_key="placeholder",  # satisfies validation; nulled below
            default_headers={"Authorization": f"Bearer {token_provider()}"},
            max_tokens=4096,
            temperature=0,
        )
        # The anthropic SDK sends X-Api-Key alongside any custom Authorization
        # header; a gateway that validates X-Api-Key first would 401. Clearing
        # the key on both underlying clients removes the header entirely
        # (verified against anthropic 0.109.1: absent from built requests).
        judge_chat._client.api_key = None
        judge_chat._async_client.api_key = None
        return Faithfulness(llm=LangchainLLMWrapper(judge_chat))

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await _metric().single_turn_ascore(sample))

    return metric_fn


class LoopDecision(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    EXHAUSTED = "exhausted"


def decide_next(
    score: float | None, threshold: float, attempt: int, max_retries: int
) -> LoopDecision:
    """Decide whether to accept the answer, retry, or give up.

    A missing score (judge failure) is fail-closed: never PASS.
    """
    if score is not None and score >= threshold:
        return LoopDecision.PASS
    if attempt < max_retries:
        return LoopDecision.RETRY
    return LoopDecision.EXHAUSTED
