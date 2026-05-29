from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from ragpipe.models import Chunk

MetricFn = Callable[..., Awaitable[float]]


class _Scorer(Protocol):
    async def score(self, query: str, answer: str, contexts: list[Chunk]) -> float: ...


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


def build_ragas_faithfulness(settings) -> MetricFn:  # pragma: no cover
    """Build a faithfulness metric callable backed by Foundry models via RAGAS."""
    _ensure_ragas_importable()

    from langchain_openai import AzureChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    judge = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=settings.foundry_project_endpoint,
            azure_deployment=settings.foundry_chat_model,
            api_version="2024-10-21",
        )
    )
    metric = Faithfulness(llm=judge)

    async def metric_fn(*, question: str, answer: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(await metric.single_turn_ascore(sample))

    return metric_fn


from enum import Enum


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
