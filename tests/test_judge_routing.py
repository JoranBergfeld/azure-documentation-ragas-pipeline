import pytest

from ragpipe import foundry_judge, guardrail
from ragpipe.foundry_judge import build_judge_complete_fn, judge_provider
from ragpipe.guardrail import build_ragas_faithfulness


class _S:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"

    def __init__(self, judge_model):
        self.judge_model = judge_model


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-sonnet-4-6", "anthropic"),
        ("Claude-Sonnet-4-6", "anthropic"),
        ("Kimi-K2.5", "openai"),
        ("DeepSeek-V4-Pro", "openai"),
        ("gpt-5.4", "openai"),
    ],
)
def test_judge_provider(model, provider):
    assert judge_provider(model) == provider


def test_complete_fn_dispatches_anthropic_for_claude(monkeypatch):
    calls = []
    monkeypatch.setattr(
        foundry_judge,
        "_anthropic_complete_fn",
        lambda s, m: calls.append("anthropic") or (lambda p: "A"),
    )
    monkeypatch.setattr(
        foundry_judge,
        "_openai_complete_fn",
        lambda s, m: calls.append("openai") or (lambda p: "O"),
    )
    fn = build_judge_complete_fn(_S("claude-sonnet-4-6"))
    assert fn("x") == "A"
    assert calls == ["anthropic"]


def test_complete_fn_dispatches_openai_for_kimi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        foundry_judge,
        "_anthropic_complete_fn",
        lambda s, m: calls.append("anthropic") or (lambda p: "A"),
    )
    monkeypatch.setattr(
        foundry_judge,
        "_openai_complete_fn",
        lambda s, m: calls.append("openai") or (lambda p: "O"),
    )
    fn = build_judge_complete_fn(_S("Kimi-K2.5"))
    assert fn("x") == "O"
    assert calls == ["openai"]


def test_complete_fn_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_judge_complete_fn(_S(None))


def test_gate_dispatches_anthropic_for_claude(monkeypatch):
    calls = []
    monkeypatch.setattr(
        guardrail,
        "_build_claude_faithfulness",
        lambda s: calls.append("anthropic") or "A",
    )
    monkeypatch.setattr(
        guardrail,
        "_build_openai_faithfulness",
        lambda s: calls.append("openai") or "O",
    )
    assert build_ragas_faithfulness(_S("claude-sonnet-4-6")) == "A"
    assert calls == ["anthropic"]


def test_gate_dispatches_openai_for_kimi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        guardrail,
        "_build_claude_faithfulness",
        lambda s: calls.append("anthropic") or "A",
    )
    monkeypatch.setattr(
        guardrail,
        "_build_openai_faithfulness",
        lambda s: calls.append("openai") or "O",
    )
    assert build_ragas_faithfulness(_S("Kimi-K2.5")) == "O"
    assert calls == ["openai"]
