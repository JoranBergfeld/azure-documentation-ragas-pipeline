from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable

from ragpipe.eval.retrieval_metrics import stage_retrieval_metrics
from ragpipe.eval.stats import bootstrap_ci_mean, paired_diff_test
from ragpipe.eval.testset import TestItem
from ragpipe.models import PipelineState


# Offline RAGAS judge budgets. The offline judge is a *reasoning* model
# (DeepSeek, ADR-0009): a single faithfulness/context_precision job is several
# sequential NLI calls over long context, each of which can run for minutes.
# RAGAS's default RunConfig (timeout=180s, max_workers=16) starves it — 16
# concurrent jobs queue on the one deployment and blow past 180s, so every heavy
# job raises TimeoutError and the metric collapses to NaN (observed: faithfulness
# 0/33, context_precision 1/33). A generous per-call timeout, a generous per-job
# timeout, and fewer concurrent jobs let the reasoning judge actually finish.
# These are separate from the online gate's JUDGE_TIMEOUT (foundry_judge) so
# raising the offline eval budget does not slow live faithfulness gating.
RAGAS_JUDGE_TIMEOUT = 300.0
RAGAS_JOB_TIMEOUT = 600.0
RAGAS_MAX_WORKERS = 8


def _ragas_run_config():
    """RunConfig that lets the reasoning offline judge finish its heavy metrics.

    Widens the per-job timeout and cuts concurrency vs. RAGAS's defaults so
    faithfulness/context_precision stop timing out (see the constants above).
    """
    from ragas.run_config import RunConfig

    return RunConfig(timeout=RAGAS_JOB_TIMEOUT, max_workers=RAGAS_MAX_WORKERS)


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


def stages_from_records(records: list[EvalRecord]) -> list[str]:
    """The substrate's own stage names across the records, in first-seen order.

    Dynamic stage reading (ADR-0016): the per-stage sweep scores whatever stages
    the active substrate produced — dense/bm25/fused (hybrid), local/global/fused
    (graph), iter_0..iter_N (agentic), always ending in the well-known `reranked`
    — instead of a hardcoded hybrid tuple. Records share the same stage set, so
    first-seen insertion order mirrors the pipeline order of `state.stages`.
    """
    ordered: dict[str, None] = {}
    for record in records:
        for stage in record.stage_contexts:
            ordered.setdefault(stage, None)
    return list(ordered)


async def run_harness(
    items: list[TestItem],
    pipeline_fn: PipelineFn,
    evaluator_fn: EvaluatorFn,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for item in items:
        state = await pipeline_fn(item.question)
        by_stage = state.stages
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


def aggregate_with_ci(
    records: list[EvalRecord],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> dict[str, dict]:
    """Mean and percentile bootstrap CI of each metric across finite item scores."""
    keys = {k for r in records for k in r.metrics}
    intervals: dict[str, dict] = {}
    for k in keys:
        valid = [r.metrics[k] for r in records if _is_valid(r.metrics.get(k))]
        if valid:
            intervals[k] = bootstrap_ci_mean(
                valid, confidence=confidence, n_resamples=n_resamples, seed=seed
            )
    return intervals


def aggregate_by_tag(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    """aggregate() per tag group; records without tags count as 'original'.

    A record with several tags contributes to each of its groups.
    """
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        for tag in r.tags or ("original",):
            groups.setdefault(tag, []).append(r)
    return {tag: aggregate(rs) for tag, rs in sorted(groups.items())}


def aggregate_by_mode(records_by_mode: dict[str, list[EvalRecord]]) -> dict[str, dict[str, float]]:
    """aggregate() per mode. Keys are mode names; values are the per-mode means."""
    return {mode: aggregate(recs) for mode, recs in records_by_mode.items()}


def compare_modes(
    records_by_mode: dict[str, list[EvalRecord]],
    baseline: str,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> dict[str, dict[str, dict]]:
    """Paired bootstrap diffs for each mode and metric against ``baseline``."""
    if baseline not in records_by_mode:
        return {}

    baseline_records = records_by_mode[baseline]
    baseline_metrics = {k for r in baseline_records for k in r.metrics}
    comparisons: dict[str, dict[str, dict]] = {}
    for mode, records in records_by_mode.items():
        if mode == baseline:
            continue
        treatment_metrics = {k for r in records for k in r.metrics}
        metric_results: dict[str, dict] = {}
        for metric in sorted(treatment_metrics & baseline_metrics):
            length = max(len(records), len(baseline_records))
            treatment_scores = [
                records[i].metrics.get(metric, float("nan")) if i < len(records) else float("nan")
                for i in range(length)
            ]
            baseline_scores = [
                baseline_records[i].metrics.get(metric, float("nan"))
                if i < len(baseline_records)
                else float("nan")
                for i in range(length)
            ]
            result = paired_diff_test(
                treatment_scores,
                baseline_scores,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            )
            if result["n"] > 0:
                metric_results[metric] = result
        comparisons[mode] = metric_results
    return comparisons


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
    from ragpipe.foundry_judge import JUDGE_MAX_RETRIES

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    # No explicit temperature: DeepSeek-V4-Pro is a reasoning model and, like
    # other reasoning deployments on this route, may reject sampling overrides.
    # model= sets the request-body "model" field: the sold-by-Azure DeepSeek
    # server (sglang) validates it and rejects a null, unlike Azure OpenAI which
    # takes the deployment from the URL and ignores the body field.
    # timeout + max_retries bound every judge/embedding call: without them a
    # stalled request blocks the eval forever (see embeddings._build_client). The
    # reasoning judge gets RAGAS_JUDGE_TIMEOUT (larger than the online gate's
    # JUDGE_TIMEOUT) because faithfulness/context_precision calls run for minutes.
    llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=services_endpoint_from_project(settings.foundry_project_endpoint),
            azure_deployment=settings.offline_judge_model,
            model=settings.offline_judge_model,
            api_version="2024-10-21",
            azure_ad_token_provider=token_provider,
            timeout=RAGAS_JUDGE_TIMEOUT,
            max_retries=JUDGE_MAX_RETRIES,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_endpoint=openai_endpoint_from_project(settings.foundry_project_endpoint),
            azure_deployment=settings.foundry_embedding_model,
            api_version="2024-10-21",
            azure_ad_token_provider=token_provider,
            timeout=RAGAS_JUDGE_TIMEOUT,
            max_retries=JUDGE_MAX_RETRIES,
        )
    )
    return llm, emb


def build_ragas_evaluator(settings):  # pragma: no cover
    """Return an evaluator_fn that scores records with the full RAGAS suite.

    Scores the answer-level metrics (faithfulness, answer_relevancy) plus the
    context metrics on the *final* reranked context set.
    """
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        if not records:
            return records

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
        run_config = _ragas_run_config()

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
                ds,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=emb,
                run_config=run_config,
            )
            df = result.to_pandas()
            for j, i in enumerate(answered):
                for metric in ["faithfulness", "answer_relevancy"]:
                    if metric in df.columns:
                        records[i].metrics[metric] = float(df.iloc[j][metric])

        ds_all = Dataset.from_list([_row(r) for r in records])
        result = evaluate(
            ds_all,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=emb,
            run_config=run_config,
        )
        df = result.to_pandas()
        for i, r in enumerate(records):
            for metric in ["context_precision", "context_recall"]:
                if metric in df.columns:
                    r.metrics[metric] = float(df.iloc[i][metric])
        return records

    return evaluator_fn


def build_per_stage_context_evaluator(settings, stages=None):  # pragma: no cover
    """Return an evaluator_fn that scores context_precision/recall at each retrieval stage.

    Runs the two context metrics once per stage over that stage's captured context
    set, writing keys like 'context_precision@dense'. This is the expensive sweep
    (one judge pass per stage) — gate it behind the PER_STAGE_METRICS toggle.

    Stages default to whatever the active substrate named (`stages_from_records`,
    ADR-0016); pass an explicit tuple to override.
    """
    async def evaluator_fn(records: list[EvalRecord]) -> list[EvalRecord]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall

        llm, emb = _build_ragas_clients(settings)
        run_config = _ragas_run_config()
        eval_stages = stages if stages is not None else stages_from_records(records)
        for stage in eval_stages:
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
                run_config=run_config,
            )
            df = result.to_pandas()
            for i, r in enumerate(records):
                for metric in ["context_precision", "context_recall"]:
                    if metric in df.columns:
                        r.metrics[stage_metric_key(metric, stage)] = float(df.iloc[i][metric])
            print(f"  per-stage metrics done: {stage}", flush=True)
        return records

    return evaluator_fn
