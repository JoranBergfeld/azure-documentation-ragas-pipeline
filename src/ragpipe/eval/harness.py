from __future__ import annotations

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


def aggregate(records: list[EvalRecord]) -> dict[str, float]:
    keys = {k for r in records for k in r.metrics}
    return {
        k: mean([r.metrics[k] for r in records if k in r.metrics]) for k in keys
    }


def build_ragas_evaluator(settings):  # pragma: no cover
    """Return an evaluator_fn that scores records with the full RAGAS suite."""
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

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
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_chat_model,
                api_version="2024-10-21",
            )
        )
        emb = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_embedding_model,
                api_version="2024-10-21",
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
