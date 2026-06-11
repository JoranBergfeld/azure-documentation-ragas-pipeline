"""Every live LLM/judge client must be built with a request timeout and bounded
retries. Without them a stalled Azure call (request sent + ACKed, no response,
no keepalive) blocks its worker thread forever — the eval-harness hang we hit.
The reference convention is ``embeddings._build_client`` (timeout + max_retries).

These tests monkeypatch the SDK client classes to capture constructor kwargs, so
no live Azure call is made; only the kwargs handed to each client are asserted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class _S:
    foundry_project_endpoint = "https://acct.services.ai.azure.com/api/projects/p"
    foundry_chat_model = "gpt-5.4"
    foundry_embedding_model = "text-embedding-3-small"

    def __init__(self, judge_model="Kimi-K2.5", offline_judge_model="DeepSeek-V4-Pro"):
        self.judge_model = judge_model
        self.offline_judge_model = offline_judge_model


def _inert_identity(monkeypatch):
    """Stub azure-identity so no token is actually fetched."""
    import azure.identity as ident

    monkeypatch.setattr(ident, "DefaultAzureCredential", lambda *a, **k: object())
    monkeypatch.setattr(ident, "get_bearer_token_provider", lambda *a, **k: (lambda: "tok"))


def _openai_recorder(captured):
    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content="ok")
            choice = SimpleNamespace(message=msg)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[choice]))
            )

    return _Client


def _anthropic_recorder(captured):
    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            block = SimpleNamespace(type="text", text="ok")
            self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[block]))

    return _Client


def _kwarg_recorder(captured):
    """Minimal LangChain-chat stand-in that records constructor kwargs."""

    class _Chat:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            # The Claude builder nulls the SDK key on these after construction.
            self._client = SimpleNamespace(api_key="x")
            self._async_client = SimpleNamespace(api_key="x")

    return _Chat


class _FakeMetric:
    def __init__(self, llm=None):
        self._llm = llm

    async def single_turn_ascore(self, sample):
        return 1.0


def test_openai_judge_complete_passes_timeout_and_retries(monkeypatch):
    import openai

    from ragpipe import foundry_judge

    _inert_identity(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(openai, "AzureOpenAI", _openai_recorder(captured))

    complete = foundry_judge._openai_complete_fn(_S(), max_tokens=16)
    assert complete("prompt") == "ok"

    assert captured["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert captured["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES


def test_anthropic_judge_complete_passes_timeout_and_retries(monkeypatch):
    import anthropic

    from ragpipe import foundry_judge

    _inert_identity(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", _anthropic_recorder(captured))

    complete = foundry_judge._anthropic_complete_fn(_S(judge_model="claude-sonnet-4-6"), max_tokens=16)
    assert complete("prompt") == "ok"

    assert captured["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert captured["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES


def test_openai_faithfulness_passes_timeout_and_retries(monkeypatch):
    import langchain_openai

    from ragpipe import foundry_judge, guardrail

    guardrail._ensure_ragas_importable()
    import ragas.llms
    import ragas.metrics

    _inert_identity(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", _kwarg_recorder(captured))
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda x: x)
    monkeypatch.setattr(ragas.metrics, "Faithfulness", _FakeMetric)

    guardrail._build_openai_faithfulness(_S())

    assert captured["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert captured["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES


@pytest.mark.asyncio
async def test_claude_faithfulness_passes_timeout_and_retries(monkeypatch):
    import langchain_anthropic

    from ragpipe import foundry_judge, guardrail

    guardrail._ensure_ragas_importable()
    import ragas.llms
    import ragas.metrics

    _inert_identity(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", _kwarg_recorder(captured))
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda x: x)
    monkeypatch.setattr(ragas.metrics, "Faithfulness", _FakeMetric)

    metric_fn = guardrail._build_claude_faithfulness(_S(judge_model="claude-sonnet-4-6"))
    score = await metric_fn(question="q", answer="a", contexts=["c"])

    assert score == 1.0
    assert captured["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert captured["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES


def test_deepseek_offline_judge_passes_timeout_and_retries(monkeypatch):
    import langchain_openai

    from ragpipe import foundry_judge, guardrail
    from ragpipe.eval import harness

    guardrail._ensure_ragas_importable()
    import ragas.embeddings
    import ragas.llms

    _inert_identity(monkeypatch)
    chat_kwargs: dict = {}
    emb_kwargs: dict = {}
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", _kwarg_recorder(chat_kwargs))
    monkeypatch.setattr(langchain_openai, "AzureOpenAIEmbeddings", _kwarg_recorder(emb_kwargs))
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda x: x)
    monkeypatch.setattr(ragas.embeddings, "LangchainEmbeddingsWrapper", lambda x: x)

    harness._build_ragas_clients_live(_S())

    assert chat_kwargs["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert chat_kwargs["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES
    assert emb_kwargs["timeout"] == foundry_judge.JUDGE_TIMEOUT
    assert emb_kwargs["max_retries"] == foundry_judge.JUDGE_MAX_RETRIES
