from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable

from ragpipe.eval.testset import TestItem
from ragpipe.models import PipelineState


@dataclass
class EvalRecord:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    metrics: dict[str, float] = field(default_factory=dict)


PipelineFn = Callable[[str], Awaitable[PipelineState]]
EvaluatorFn = Callable[[list[EvalRecord]], Awaitable[list[EvalRecord]]]


async def run_harness(
    items: list[TestItem],
    pipeline_fn: PipelineFn,
    evaluator_fn: EvaluatorFn,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for item in items:
        state = await pipeline_fn(item.question)
        records.append(
            EvalRecord(
                question=item.question,
                answer=state.answer,
                contexts=[c.content for c in state.reranked],
                ground_truth=item.ground_truth,
            )
        )
    return await evaluator_fn(records)


def _is_valid(value: object) -> bool:
    """A usable metric score: a real number that is not NaN/inf."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def aggregate(records: list[EvalRecord]) -> dict[str, float]:
    """Mean of each metric across records, ignoring missing/NaN scores.

    RAGAS occasionally returns NaN for a single item (e.g. the LLM judge emits an
    unparseable response for one statement). Averaging only the valid scores keeps
    one flaky judge call from poisoning the whole metric's mean. Metrics with no
    valid scores at all are omitted.
    """
    keys = {k for r in records for k in r.metrics}
    means: dict[str, float] = {}
    for k in keys:
        valid = [r.metrics[k] for r in records if _is_valid(r.metrics.get(k))]
        if valid:
            means[k] = mean(valid)
    return means


def coverage(records: list[EvalRecord]) -> dict[str, tuple[int, int]]:
    """Per-metric (valid_count, total_count) so dropped/NaN scores are visible."""
    keys = {k for r in records for k in r.metrics}
    total = len(records)
    return {
        k: (sum(1 for r in records if _is_valid(r.metrics.get(k))), total) for k in keys
    }


def build_ragas_evaluator(settings):  # pragma: no cover
    """Return an evaluator_fn that scores records with the full RAGAS suite."""
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from datasets import Dataset
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from ragpipe.embeddings import openai_endpoint_from_project

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        openai_endpoint = openai_endpoint_from_project(settings.foundry_project_endpoint)

        ds = Dataset.from_list(
            [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts,
                    "ground_truth": r.ground_truth,
                }
                for r in records
            ]
        )
        llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                azure_endpoint=openai_endpoint,
                azure_deployment=settings.foundry_chat_model,
                api_version="2024-10-21",
                azure_ad_token_provider=token_provider,
            )
        )
        emb = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_endpoint=openai_endpoint,
                azure_deployment=settings.foundry_embedding_model,
                api_version="2024-10-21",
                azure_ad_token_provider=token_provider,
            )
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=emb,
        )
        df = result.to_pandas()
        for i, r in enumerate(records):
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in df.columns:
                    r.metrics[metric] = float(df.iloc[i][metric])
        return records

    return evaluator_fn
