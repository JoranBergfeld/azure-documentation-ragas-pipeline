from __future__ import annotations

import pytest

from ragpipe.guardrail import (
    ClaimVerdict,
    FaithfulnessResult,
    FaithfulnessScorer,
    _score_faithfulness_detailed,
    as_faithfulness_result,
    build_ragas_faithfulness,
)
from ragpipe.models import Chunk


def _chunk(content):
    return Chunk(id="c", title="t", url="http://c", content=content)


@pytest.mark.asyncio
async def test_scorer_passes_answer_and_context_to_metric():
    captured = {}

    async def fake_metric(*, question, answer, contexts):
        captured["question"] = question
        captured["answer"] = answer
        captured["contexts"] = contexts
        return 0.83

    scorer = FaithfulnessScorer(metric_fn=fake_metric)
    score = await scorer.score("q", "a", [_chunk("ctx1"), _chunk("ctx2")])

    assert score == 0.83
    assert captured["answer"] == "a"
    assert captured["contexts"] == ["ctx1", "ctx2"]


@pytest.mark.asyncio
async def test_scorer_returns_structured_details_and_scalar_score():
    claim = ClaimVerdict(statement="s", verdict=True, reason="grounded")

    async def fake_metric(*, question, answer, contexts):
        return FaithfulnessResult(score=0.83, claims=(claim,))

    scorer = FaithfulnessScorer(metric_fn=fake_metric)

    detailed = await scorer.score_detailed("q", "a", [_chunk("ctx")])
    score = await scorer.score("q", "a", [_chunk("ctx")])

    assert detailed == FaithfulnessResult(score=0.83, claims=(claim,))
    assert score == 0.83


def test_as_faithfulness_result_accepts_float():
    assert as_faithfulness_result(0.42) == FaithfulnessResult(score=0.42)


def test_as_faithfulness_result_accepts_none():
    assert as_faithfulness_result(None) == FaithfulnessResult(score=None)


def test_as_faithfulness_result_passthrough():
    result = FaithfulnessResult(
        score=0.7,
        claims=(ClaimVerdict(statement="s", verdict=False, reason="missing"),),
    )
    assert as_faithfulness_result(result) is result


@pytest.mark.asyncio
async def test_detailed_scorer_passes_statement_list_to_ragas_verdict_step():
    class _Statements:
        statements = ["claim"]

    class _Answer:
        statement = "claim"
        verdict = 1
        reason = "grounded"

    class _Nli:
        statements = [_Answer()]

    class _Metric:
        async def _create_statements(self, row, callbacks):
            return _Statements()

        async def _create_verdicts(self, row, statements, callbacks):
            assert statements == ["claim"]
            return _Nli()

        def _compute_score(self, nli):
            return 1.0

    result = await _score_faithfulness_detailed(
        _Metric(), question="q", answer="a", contexts=["ctx"]
    )

    assert result == FaithfulnessResult(
        score=1.0,
        claims=(ClaimVerdict(statement="claim", verdict=True, reason="grounded"),),
    )


class _Settings:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"
    judge_model = None


def test_gate_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_ragas_faithfulness(_Settings())
