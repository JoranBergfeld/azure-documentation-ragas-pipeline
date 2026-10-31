from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ragpipe import faithfulness_gate
from ragpipe.faithfulness_gate import (
    ClaimVerdict,
    FaithfulnessGate,
    ScoredClaims,
    build_faithfulness_gate,
    score_with_claims,
)


class _Metric:
    """RAGAS ``Faithfulness`` stand-in: the two-call pass (decompose the answer into
    statements, then a verdict per statement) the gate drives to keep the claims."""

    def __init__(self, verdicts, *, score=0.5, statements=("s",), fail=None):
        self.verdicts = verdicts
        self.score = score
        self.statements = list(statements)
        self.fail = fail
        self.rows = []
        self.scalar_calls = 0

    async def _create_statements(self, row, callbacks):
        self.rows.append(row)
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(statements=self.statements)

    async def _create_verdicts(self, row, statements, callbacks):
        return SimpleNamespace(statements=self.verdicts)

    def _compute_score(self, verdicts):
        return self.score

    async def single_turn_ascore(self, sample):
        self.scalar_calls += 1
        return 0.9


def _verdict(statement, verdict, reason=""):
    return SimpleNamespace(statement=statement, verdict=verdict, reason=reason)


async def test_gate_returns_the_score_with_the_claims_behind_it():
    metric = _Metric(
        [_verdict("RRF fuses ranks", 1, "in ctx"), _verdict("RRF is from 2020", 0, "absent")]
    )
    scored = await FaithfulnessGate(lambda: metric).score("q", "a", ["ctx"])

    assert scored.score == 0.5
    assert scored.claims == [
        ClaimVerdict(claim="RRF fuses ranks", faithful=True, reason="in ctx"),
        ClaimVerdict(claim="RRF is from 2020", faithful=False, reason="absent"),
    ]


async def test_gate_hands_the_judge_the_query_answer_and_contexts():
    metric = _Metric([_verdict("s", 1)])
    await FaithfulnessGate(lambda: metric).score("the q", "the a", ["c1", "c2"])

    row = metric.rows[0]
    assert row["user_input"] == "the q"
    assert row["response"] == "the a"
    assert row["retrieved_contexts"] == ["c1", "c2"]


async def test_gate_builds_a_fresh_metric_per_score():
    # The Claude route fixes its bearer token at construction, so the gate asks
    # for a metric each time rather than holding one.
    built = []

    def new_metric():
        built.append(_Metric([_verdict("s", 1)]))
        return built[-1]

    gate = FaithfulnessGate(new_metric)
    await gate.score("q", "a", [])
    await gate.score("q", "a", [])
    assert len(built) == 2


async def test_gate_fails_closed_on_a_judge_error():
    metric = _Metric([], fail=RuntimeError("judge down"))
    with pytest.raises(RuntimeError, match="judge down"):
        await FaithfulnessGate(lambda: metric).score("q", "a", [])
    # No second, scalar-only scoring pass on the hot path.
    assert metric.scalar_calls == 0


async def test_gate_fails_closed_on_a_verdict_it_cannot_read():
    metric = _Metric([{"statement": "x", "verdict": "perhaps"}])
    with pytest.raises(ValueError, match="perhaps"):
        await FaithfulnessGate(lambda: metric).score("q", "a", [])


async def test_gate_reads_every_verdict_form_ragas_has_used():
    metric = _Metric(
        [
            {"claim": "a", "verdict": "yes"},
            {"statement": "b", "verdict": "0"},
            _verdict("c", True),
            _verdict("d", 1.0, None),
        ]
    )
    scored = await FaithfulnessGate(lambda: metric).score("q", "a", [])
    assert [(c.claim, c.faithful, c.reason) for c in scored.claims] == [
        ("a", True, ""),
        ("b", False, ""),
        ("c", True, ""),
        ("d", True, ""),
    ]


async def test_an_answer_with_no_claims_has_no_score():
    metric = _Metric([], statements=())
    scored = await FaithfulnessGate(lambda: metric).score("q", "", [])
    assert math.isnan(scored.score) and scored.claims == []


async def test_canary_scoring_falls_back_to_the_scalar_pass():
    # The canary's offline judge would rather have a score without claims than
    # no score at all; the live gate never takes this path.
    metric = _Metric([], fail=RuntimeError("seam moved"))
    sample = SimpleNamespace(to_dict=lambda: {})
    scored = await score_with_claims(metric, sample, fallback=True)
    assert scored == ScoredClaims(score=0.9, claims=[])


def _settings(judge_model):
    from ragpipe.config import Settings

    return Settings(
        foundry_project_endpoint="https://acct.services.ai.azure.com/api/projects/p",
        foundry_chat_model="gpt-5.4",
        foundry_embedding_model="text-embedding-3-small",
        generator_agent_name="gen",
        judge_model=judge_model,
    )


def test_gate_requires_judge_model():
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        build_faithfulness_gate(_settings(None))


async def test_the_gate_built_from_settings_keeps_the_claims(monkeypatch):
    """Regression: the live wiring once went through a float-only builder, so every
    production trace carried ``claims: []`` (ADR-0018). The gate built from settings,
    on whichever host serves the online judge, returns the verdicts."""
    from ragpipe.model_roles import ModelRole

    asked = []

    def judge_chat_factory(settings, role):
        asked.append(role)
        return lambda: "judge chat"

    monkeypatch.setattr(faithfulness_gate, "judge_chat_factory", judge_chat_factory)
    monkeypatch.setattr(
        faithfulness_gate,
        "_faithfulness_metric",
        lambda chat: _Metric([_verdict("s", 0, "absent")], score=0.0),
    )

    scored = await build_faithfulness_gate(_settings("Kimi-K2.5")).score("q", "a", ["c"])

    assert asked == [ModelRole.ONLINE_JUDGE]
    assert scored.claims == [ClaimVerdict(claim="s", faithful=False, reason="absent")]


def test_prewarm_ragas_imports_builds_runnable_passthrough_with_optional_name():
    """Guard the Streamlit-rerun workaround.

    The RAGAS faithfulness gate imports ``langchain_openai`` the first time a
    query runs. That import lazily constructs langchain_core's
    ``RunnablePassthrough`` pydantic model, whose ``name: str | None = None``
    field relies on deferred annotation evaluation
    (``from __future__ import annotations``). When the first import happens
    *during* a Streamlit rerun the default is lost and constructing
    ``RunnablePassthrough()`` raises ``ValidationError: name Field required``.
    ``prewarm_ragas_imports()`` forces that build once at a clean import time;
    assert it imports langchain_openai and that the model keeps its optional
    ``name`` default.
    """
    import sys

    from ragpipe.faithfulness_gate import prewarm_ragas_imports

    prewarm_ragas_imports()

    assert "langchain_openai" in sys.modules

    from langchain_core.runnables.passthrough import RunnablePassthrough

    assert RunnablePassthrough().name is None


def test_pinned_ragas_exposes_the_per_claim_seam():
    """The live gate keeps per-claim verdicts by driving RAGAS 0.4.3's private
    two-call faithfulness pass (``faithfulness_gate.score_with_claims``, ADR-0018). Fail
    loudly if a RAGAS bump removes or reshapes that seam."""
    import inspect

    from ragpipe.faithfulness_gate import ensure_ragas_importable

    ensure_ragas_importable()
    from ragas.metrics import Faithfulness
    from ragas.metrics._faithfulness import StatementFaithfulnessAnswer

    assert list(inspect.signature(Faithfulness._create_statements).parameters) == [
        "self", "row", "callbacks",
    ]
    assert list(inspect.signature(Faithfulness._create_verdicts).parameters) == [
        "self", "row", "statements", "callbacks",
    ]
    assert callable(Faithfulness._compute_score)
    assert {"statement", "verdict", "reason"} <= set(StatementFaithfulnessAnswer.model_fields)
