"""Per-run usage reporting (ADR-0018). Model clients are stubbed; no live calls."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ragpipe import usage
from ragpipe.app_wiring import make_deps
from ragpipe.generate import Generator
from ragpipe.guardrail import FaithfulnessScorer
from ragpipe.models import Chunk
from ragpipe.retrieval.passthrough import PassthroughReranker
from ragpipe.retrieval.rerank import SemanticReranker
from ragpipe.retrieval.substrate import RetrievalResult
from ragpipe.usage import UsageEntry, collect_usage
from ragpipe.workflow import run_pipeline


def _chunk(cid):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=f"content-{cid}", score=0.5)


# --- recorders -------------------------------------------------------------


def test_recording_outside_a_run_is_a_noop():
    usage.record_requests("rerank", "x")
    usage.record_tokens("generation", "gpt", input_tokens=1, output_tokens=1)


def test_entry_serialises_every_field_camel_cased():
    entry = UsageEntry(stage="generation", deployment="gpt-5.4", input_tokens=10, output_tokens=3)
    assert entry.to_dict() == {
        "stage": "generation",
        "deployment": "gpt-5.4",
        "inputTokens": 10,
        "cachedInputTokens": None,
        "outputTokens": 3,
        "requestUnits": None,
        "usageMissing": False,
    }


def test_openai_embedding_records_prompt_tokens_only():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=9, total_tokens=9))
    with collect_usage([]) as entries:
        usage.record_openai_embedding("query_embedding", "text-embedding-3-small", resp)
    assert entries == [
        UsageEntry(stage="query_embedding", deployment="text-embedding-3-small", input_tokens=9)
    ]


def test_openai_chat_records_cached_tokens_when_reported():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        )
    )
    with collect_usage([]) as entries:
        usage.record_openai_chat("plan", "gpt-5.4", resp)
    assert entries == [
        UsageEntry(
            stage="plan",
            deployment="gpt-5.4",
            input_tokens=120,
            cached_input_tokens=64,
            output_tokens=30,
        )
    ]


def test_missing_usage_is_flagged_and_logged_not_invented(capsys):
    with collect_usage([]) as entries:
        usage.record_openai_chat("plan", "gpt-5.4", SimpleNamespace(usage=None))
    assert entries == [UsageEntry(stage="plan", deployment="gpt-5.4", usage_missing=True)]
    assert entries[0].input_tokens is None and entries[0].output_tokens is None
    assert "plan call to gpt-5.4" in capsys.readouterr().err


def test_agent_response_reads_usage_details_and_cached_key():
    resp = SimpleNamespace(
        usage_details={
            "input_token_count": 800,
            "output_token_count": 120,
            "openai.cached_input_tokens": 512,
        },
        messages=[],
    )
    with collect_usage([]) as entries:
        usage.record_agent_response("generation", "gpt-5.4", resp)
    assert entries == [
        UsageEntry(
            stage="generation",
            deployment="gpt-5.4",
            input_tokens=800,
            cached_input_tokens=512,
            output_tokens=120,
        )
    ]


def test_agent_response_with_code_interpreter_adds_one_session():
    call = SimpleNamespace(type="code_interpreter_tool_call")
    resp = SimpleNamespace(
        usage_details={"input_token_count": 5, "output_token_count": 2},
        messages=[SimpleNamespace(contents=[call, call, SimpleNamespace(type="text")])],
    )
    with collect_usage([]) as entries:
        usage.record_agent_response("generation", "gpt-5.4", resp)
    assert [e.stage for e in entries] == ["generation", "code_interpreter"]
    assert entries[1].request_units == 1
    assert entries[1].deployment == usage.CODE_INTERPRETER_DEPLOYMENT


async def test_langchain_callback_records_each_llm_call():
    from ragpipe.guardrail import _ensure_ragas_importable

    _ensure_ragas_importable()  # safe langchain import order, as the gate does
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def reply(n_in, n_out, cache_read=None):
        details = {} if cache_read is None else {"cache_read": cache_read}
        return AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": n_in,
                "output_tokens": n_out,
                "total_tokens": n_in + n_out,
                "input_token_details": details,
            },
        )

    model = GenericFakeChatModel(
        messages=iter([reply(100, 20), reply(140, 8, cache_read=96), AIMessage(content="bare")]),
        callbacks=[usage.build_langchain_usage_callback("faithfulness_judge", "claude-sonnet-4-6")],
    )
    with collect_usage([]) as entries:
        await model.ainvoke("statements")
        # RAGAS wraps scoring in wait_for, which runs it in a child task.
        await asyncio.wait_for(model.ainvoke("verdicts"), timeout=5)
        await model.ainvoke("no usage on this one")

    assert [e.to_dict() for e in entries] == [
        {
            "stage": "faithfulness_judge",
            "deployment": "claude-sonnet-4-6",
            "inputTokens": 100,
            "cachedInputTokens": None,
            "outputTokens": 20,
            "requestUnits": None,
            "usageMissing": False,
        },
        {
            "stage": "faithfulness_judge",
            "deployment": "claude-sonnet-4-6",
            "inputTokens": 140,
            "cachedInputTokens": 96,
            "outputTokens": 8,
            "requestUnits": None,
            "usageMissing": False,
        },
        {
            "stage": "faithfulness_judge",
            "deployment": "claude-sonnet-4-6",
            "inputTokens": None,
            "cachedInputTokens": None,
            "outputTokens": None,
            "requestUnits": None,
            "usageMissing": True,
        },
    ]


def test_langchain_callback_flags_a_timed_out_call_but_not_a_rejected_one(capsys):
    from ragpipe.guardrail import _ensure_ragas_importable

    _ensure_ragas_importable()
    callback = usage.build_langchain_usage_callback("faithfulness_judge", "claude-sonnet-4-6")

    class APITimeoutError(Exception):  # SDK timeouts do not subclass TimeoutError
        pass

    with collect_usage([]) as entries:
        callback.on_llm_error(TimeoutError())
        callback.on_llm_error(asyncio.CancelledError())
        callback.on_llm_error(APITimeoutError())
        callback.on_llm_error(PermissionError("401"))  # never metered

    assert entries == [
        UsageEntry(stage="faithfulness_judge", deployment="claude-sonnet-4-6", usage_missing=True)
    ] * 3
    assert "timed out" in capsys.readouterr().err


def test_embed_fn_records_query_embedding(monkeypatch):
    from ragpipe import embeddings

    class _Client:
        def __init__(self):
            self.embeddings = SimpleNamespace(create=self._create)

        def _create(self, *, model, input):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2])],
                usage=SimpleNamespace(prompt_tokens=7),
            )

    monkeypatch.setattr(embeddings, "_build_client", lambda *a, **k: _Client())
    embed = embeddings.build_embed_fn(
        SimpleNamespace(foundry_embedding_model="text-embedding-3-small")
    )
    with collect_usage([]) as entries:
        assert embed("what is RRF?") == [0.1, 0.2]
    assert entries == [
        UsageEntry(stage="query_embedding", deployment="text-embedding-3-small", input_tokens=7)
    ]


# --- whole runs, real stages over stubbed clients -------------------------------


class _S:
    faithfulness_threshold = 0.7
    max_retries = 2
    top_k = 2
    candidate_pool = 15


class _SearchClient:
    def search(self, search_text=None, **kwargs):
        return iter(
            {"id": c, "title": c, "url": c, "content": c, "@search.rerankerScore": 2.0}
            for c in ("a", "b")
        )


class _Agent:
    """Foundry agent stand-in: one response per call, with the given usage."""

    def __init__(self, usages):
        self._usages = iter(usages)

    async def run(self, prompt):
        return SimpleNamespace(text="an answer", usage_details=next(self._usages), messages=[])


def _judge(scores, calls_per_score=2):
    """Metric stand-in: each score makes `calls_per_score` judge LLM calls, as
    RAGAS faithfulness does (statements, then verdicts)."""
    scores = iter(scores)

    async def metric_fn(*, question, answer, contexts):
        for _ in range(calls_per_score):
            message = SimpleNamespace(usage_metadata={"input_tokens": 300, "output_tokens": 40})
            usage.record_langchain_result(
                "faithfulness_judge",
                "claude-sonnet-4-6",
                SimpleNamespace(generations=[[SimpleNamespace(message=message)]]),
            )
        return next(scores)

    return metric_fn


def _deps(*, reranker, agent, scores, embed=None):
    async def retrieve(query, k, on_event=None):
        if embed is not None:
            embed(query)
        chunks = [_chunk("a"), _chunk("b")]
        return RetrievalResult(candidates=chunks, stages={"fused": chunks})

    return make_deps(
        _S(),
        retrieve=retrieve,
        reranker=reranker,
        generator=Generator(agent, deployment="gpt-5.4"),
        scorer=FaithfulnessScorer(_judge(scores)),
    )


def _embed():
    def embed(text):
        usage.record_tokens("query_embedding", "text-embedding-3-small", input_tokens=6)
        return [0.0]

    return embed


def _gen_usage(n_in=900, n_out=150):
    return {"input_token_count": n_in, "output_token_count": n_out}


async def test_normal_run_reports_every_call_in_order():
    deps = _deps(
        reranker=SemanticReranker(_SearchClient(), "default-semantic", 2),
        agent=_Agent([_gen_usage()]),
        scores=[0.9],
        embed=_embed(),
    )
    state = await run_pipeline("what is RRF?", deps)

    assert [(e.stage, e.deployment) for e in state.usage] == [
        ("query_embedding", "text-embedding-3-small"),
        ("rerank", "azure-ai-search-semantic-ranker"),
        ("generation", "gpt-5.4"),
        ("faithfulness_judge", "claude-sonnet-4-6"),
        ("faithfulness_judge", "claude-sonnet-4-6"),
    ]
    rerank, generation = state.usage[1], state.usage[2]
    assert rerank.request_units == 1 and rerank.input_tokens is None
    assert (generation.input_tokens, generation.output_tokens) == (900, 150)
    assert not any(e.usage_missing for e in state.usage)


async def test_guardrail_retry_gets_its_own_entries():
    deps = _deps(
        reranker=SemanticReranker(_SearchClient(), "default-semantic", 2),
        agent=_Agent([_gen_usage(900, 150), _gen_usage(1400, 180)]),
        scores=[0.3, 0.9],
    )
    state = await run_pipeline("q", deps)

    assert state.attempt == 1
    assert [e.stage for e in state.usage] == [
        "rerank", "generation", "faithfulness_judge", "faithfulness_judge",
        "rerank", "generation", "faithfulness_judge", "faithfulness_judge",
    ]
    generations = [e for e in state.usage if e.stage == "generation"]
    assert [g.input_tokens for g in generations] == [900, 1400]


async def test_run_without_paid_calls_reports_empty_list():
    # Stub stages that touch no metered client: nothing to report.
    from ragpipe.workflow import PipelineDeps

    async def retrieve(query, k, on_event=None):
        return RetrievalResult(candidates=[_chunk("a")], stages={"fused": [_chunk("a")]})

    deps = PipelineDeps(
        retrieve=retrieve,
        rerank=lambda q, c, k: PassthroughReranker(2).rerank(q, c, k),
        generate=lambda q, chunks, prev: "cached answer",
        score=lambda q, a, c: 0.9,
    )
    state = await run_pipeline("q", deps)
    assert state.usage == []


async def test_passthrough_rerank_and_empty_candidates_are_not_billed():
    with collect_usage([]) as entries:
        PassthroughReranker(2).rerank("q", [_chunk("a")])
        SemanticReranker(_SearchClient(), "default-semantic", 2).rerank("q", [])
    assert entries == []


async def test_missing_provider_usage_is_flagged_in_the_run(capsys):
    deps = _deps(
        reranker=PassthroughReranker(2),
        agent=_Agent([None]),  # response carries no usage_details
        scores=[0.9],
    )
    state = await run_pipeline("q", deps)

    generation = state.usage[0]
    assert generation.stage == "generation" and generation.usage_missing is True
    assert generation.input_tokens is None and generation.output_tokens is None
    assert "generation call to gpt-5.4" in capsys.readouterr().err
    # The judge calls after it are still reported normally.
    assert [e.usage_missing for e in state.usage[1:]] == [False, False]


async def test_generation_timeout_records_a_call_without_counts():
    class _Sleepy:
        calls = 0

        async def run(self, prompt):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(10)
            return SimpleNamespace(text="ok", usage_details=_gen_usage(), messages=[])

    gen = Generator(_Sleepy(), timeout=0.05, max_retries=1, deployment="gpt-5.4")
    with collect_usage([]) as entries:
        await gen.generate("q", [_chunk("a")])
    assert [e.usage_missing for e in entries] == [True, False]


async def test_judge_failure_keeps_usage_of_calls_already_made():
    async def metric_fn(*, question, answer, contexts):
        usage.record_tokens(
            "faithfulness_judge", "claude-sonnet-4-6", input_tokens=300, output_tokens=40
        )
        raise RuntimeError("judge down")

    deps = make_deps(
        _S(),
        retrieve=_deps(reranker=None, agent=None, scores=[]).retrieve,
        reranker=PassthroughReranker(2),
        generator=Generator(_Agent([_gen_usage()]), deployment="gpt-5.4"),
        scorer=FaithfulnessScorer(metric_fn),
    )
    state = await run_pipeline("q", deps)

    assert state.abstained is True
    assert [e.stage for e in state.usage] == ["generation", "faithfulness_judge"]


async def test_concurrent_runs_do_not_share_usage():
    def deps(n_in):
        return _deps(
            reranker=PassthroughReranker(2), agent=_Agent([_gen_usage(n_in)]), scores=[0.9]
        )

    a, b = await asyncio.gather(run_pipeline("a", deps(111)), run_pipeline("b", deps(222)))
    assert [e.input_tokens for e in a.usage if e.stage == "generation"] == [111]
    assert [e.input_tokens for e in b.usage if e.stage == "generation"] == [222]
    assert len(a.usage) == len(b.usage) == 3


@pytest.mark.parametrize("stage", ["query_embedding", "rerank", "generation"])
def test_no_entry_carries_money(stage):
    keys = set(UsageEntry(stage=stage, deployment="d").to_dict())
    assert not {k for k in keys if "cost" in k.lower() or "price" in k.lower()}
