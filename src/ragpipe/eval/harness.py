from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable

from ragpipe.eval.retrieval_metrics import stage_retrieval_metrics
from ragpipe.eval.testset import TestItem
from ragpipe.models import PipelineState


# Retrieval stages whose context sets can be scored independently. dense/bm25 run
# in parallel; fused is their RRF merge; reranked is the final precision pass. The
# answer-level metrics (faithfulness, relevancy) only exist after generation.
RETRIEVAL_STAGES = ("dense", "bm25", "fused", "reranked")


@dataclass
class EvalRecord:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-stage context text captured during the run (always populated, cheap).
    # Only *evaluated* when the per-stage sweep is enabled.
    stage_contexts: dict[str, list[str]] = field(default_factory=dict)
    # Per-stage page URLs captured during the run, for deterministic URL-match
    # metrics (ADR-0002).
    stage_urls: dict[str, list[str]] = field(default_factory=dict)
    # Test-item tags (ADR-0006); empty means 'original'.
    tags: tuple[str, ...] = ()
    # Directive guardrail outcome (ADR-0009): True when the pipeline abstained.
    # Mirrored into metrics["abstained"] so aggregate()/aggregate_by_tag()
    # report the abstention rate alongside every other metric for free.
    abstained: bool = False


PipelineFn = Callable[[str], Awaitable[PipelineState]]
EvaluatorFn = Callable[[list[EvalRecord]], Awaitable[list[EvalRecord]]]


def stage_metric_key(metric: str, stage: str) -> str:
    """Compose a per-stage metric key, e.g. ('context_recall', 'dense') -> 'context_recall@dense'."""
    return f"{metric}@{stage}"


def parse_stage_metric(key: str) -> tuple[str, str] | None:
    """Inverse of stage_metric_key; returns (metric, stage) or None for plain keys."""
    if "@" not in key:
        return None
    metric, stage = key.split("@", 1)
    return metric, stage


async def run_harness(
    items: list[TestItem],
    pipeline_fn: PipelineFn,
    evaluator_fn: EvaluatorFn,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for item in items:
        state = await pipeline_fn(item.question)
        by_stage = {
            "dense": state.dense,
            "bm25": state.bm25,
            "fused": state.fused,
            "reranked": state.reranked,
        }
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
        records.append(record)
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


def aggregate_by_tag(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    """aggregate() per tag group; records without tags count as 'original'.

    A record with several tags contributes to each of its groups.
    """
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        for tag in r.tags or ("original",):
            groups.setdefault(tag, []).append(r)
    return {tag: aggregate(rs) for tag, rs in sorted(groups.items())}


def coverage(records: list[EvalRecord]) -> dict[str, tuple[int, int]]:
    """Per-metric (valid_count, total_count) so dropped/NaN scores are visible."""
    keys = {k for r in records for k in r.metrics}
    total = len(records)
    return {
        k: (sum(1 for r in records if _is_valid(r.metrics.get(k))), total) for k in keys
    }


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
    """Build the (llm, embeddings) RAGAS wrappers backed by Foundry models via Entra."""
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


def build_ragas_evaluator(settings):  # pragma: no cover
    """Return an evaluator_fn that scores records with the full RAGAS suite.

    Scores the answer-level metrics (faithfulness, answer_relevancy) plus the
    context metrics on the *final* reranked context set.
    """
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

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

    return evaluator_fn


def build_per_stage_context_evaluator(settings, stages=RETRIEVAL_STAGES):  # pragma: no cover
    """Return an evaluator_fn that scores context_precision/recall at each retrieval stage.

    Runs the two context metrics once per stage over that stage's captured context
    set, writing keys like 'context_precision@dense'. This is the expensive sweep
    (one judge pass per stage) — gate it behind the PER_STAGE_METRICS toggle.
    """
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall

        llm, emb = _build_ragas_clients(settings)
        for stage in stages:
            ds = Dataset.from_list(
                [
                    {
                        "question": r.question,
                        # context metrics don't use the answer, but a value is required
                        "answer": r.answer,
                        "contexts": r.stage_contexts.get(stage, []),
                        "ground_truth": r.ground_truth,
                    }
                    for r in records
                ]
            )
            result = evaluate(
                ds,
                metrics=[context_precision, context_recall],
                llm=llm,
                embeddings=emb,
            )
            df = result.to_pandas()
            for i, r in enumerate(records):
                for metric in ["context_precision", "context_recall"]:
                    if metric in df.columns:
                        r.metrics[stage_metric_key(metric, stage)] = float(df.iloc[i][metric])
            print(f"  per-stage metrics done: {stage}", flush=True)
        return records

    return evaluator_fn
