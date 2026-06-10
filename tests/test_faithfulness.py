import pytest

from ragpipe.guardrail import FaithfulnessScorer, build_ragas_faithfulness
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


class _Settings:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"
    judge_model = None


def test_gate_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_ragas_faithfulness(_Settings())
