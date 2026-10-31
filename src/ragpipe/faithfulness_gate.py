"""The faithfulness gate: how well an answer is supported by its contexts, and why.

One module owns the whole path a score takes (ADR-0009, ADR-0011, ADR-0018): which
judge route to call, the RAGAS import hazard, and RAGAS's two-call faithfulness pass
(decompose the answer into claims, then a verdict per claim), kept whole so the score
always arrives with the claim verdicts behind it. The live pipeline, the drift canary,
threshold calibration and ``scripts/verify_judges.py`` all score through here, so they
measure the same thing.

The pass/retry/abstain decision built on the score is the guardrail's
(``guardrail.decide_next``), not this module's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ragpipe.model_roles import ModelRole, judge_chat_factory


@dataclass(frozen=True)
class ClaimVerdict:
    """One decomposed claim and whether the judge found it grounded in context."""

    claim: str
    faithful: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "faithful": self.faithful, "reason": self.reason}


@dataclass(frozen=True)
class ScoredClaims:
    """A faithfulness score plus the per-claim verdicts behind it."""

    score: float
    claims: list[ClaimVerdict]


def _coerce_verdict(value: Any) -> bool:
    """Normalise RAGAS's 0/1 verdict (or a stringy/boolean equivalent) to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"cannot interpret faithfulness verdict: {value!r}")
        return value >= 1
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "1.0", "true", "yes", "faithful", "supported"}:
            return True
        if token in {"0", "0.0", "false", "no", "unfaithful", "unsupported", ""}:
            return False
    raise ValueError(f"cannot interpret faithfulness verdict: {value!r}")


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute or mapping key from ``names``."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _parse_claim_verdicts(raw: Iterable[Any]) -> list[ClaimVerdict]:
    """Normalise RAGAS faithfulness statement-verdicts into ``ClaimVerdict``s.

    Accepts RAGAS ``StatementFaithfulnessAnswer`` objects (attributes
    ``statement`` / ``verdict`` / ``reason``) or plain dicts with the same keys
    (also tolerates ``claim`` for the statement text). This is the boundary that
    keeps per-claim logging independent of RAGAS's internal class names.
    """
    out: list[ClaimVerdict] = []
    for entry in raw:
        claim = _field(entry, "statement", "claim", default="")
        verdict = _field(entry, "verdict", "faithful")
        reason = _field(entry, "reason", default="") or ""
        out.append(
            ClaimVerdict(
                claim=str(claim),
                faithful=_coerce_verdict(verdict),
                reason=str(reason),
            )
        )
    return out


async def score_with_claims(metric, sample, *, fallback: bool = True) -> ScoredClaims:
    """Score a sample and capture the decomposed per-claim verdicts.

    Reproduces RAGAS 0.4.3 ``Faithfulness._ascore`` (decompose the answer into
    statements, NLI each against the context) but keeps the intermediate verdicts
    instead of discarding them — the same two judge calls, no extra cost. Relies
    on pinned ``ragas==0.4.3`` internals (``_create_statements`` /
    ``_create_verdicts`` / ``_compute_score``); the version pin (ADR-0018) is what
    makes that safe. On any failure it falls back to the public scalar path and
    returns empty claims rather than breaking the canary. The live gate passes
    ``fallback=False``: there a judge failure must propagate so the gate fails
    closed, instead of paying for a second scoring pass on the hot path.
    """
    row = sample.to_dict()
    try:
        statements = (await metric._create_statements(row, None)).statements
        if not statements:
            return ScoredClaims(score=math.nan, claims=[])
        verdicts = await metric._create_verdicts(row, statements, None)
        score = float(metric._compute_score(verdicts))
        return ScoredClaims(score=score, claims=_parse_claim_verdicts(verdicts.statements))
    except Exception:
        if not fallback:
            raise
        score = float(await metric.single_turn_ascore(sample))
        return ScoredClaims(score=score, claims=[])


class FaithfulnessGate:
    """Scores an answer against its contexts, verdicts included, failing closed.

    ``new_metric`` returns a RAGAS ``Faithfulness`` metric bound to the judge. It is
    called once per score: the Claude route has to rebuild its client to refresh the
    bearer token, the OpenAI route just hands back the one it built.
    """

    def __init__(self, new_metric: Callable[[], Any]) -> None:
        self._new_metric = new_metric

    async def score(self, query: str, answer: str, contexts: list[str]) -> ScoredClaims:
        ensure_ragas_importable()
        from ragas.dataset_schema import SingleTurnSample

        sample = SingleTurnSample(
            user_input=query, response=answer, retrieved_contexts=list(contexts)
        )
        return await score_with_claims(self._new_metric(), sample, fallback=False)


def build_faithfulness_gate(settings) -> FaithfulnessGate:
    """The gate judged by a non-generator family (ADR-0009, ADR-0022).

    The gate decides what users see, so it must not share a model family with the
    generator: a judge is systematically lenient on its own family's outputs. The
    online judge role says where it's served (``model_roles``). Raises if JUDGE_MODEL
    is unset: silently falling back to the generator would recreate the circular
    setup this replaces.
    """
    if not settings.endpoint(ModelRole.ONLINE_JUDGE).model:
        raise ValueError(
            "JUDGE_MODEL is required: the faithfulness gate is judged by a "
            "non-generator family (ADR-0009); set it in .env"
        )
    new_chat = judge_chat_factory(settings, ModelRole.ONLINE_JUDGE)
    return FaithfulnessGate(lambda: _faithfulness_metric(new_chat()))


def _faithfulness_metric(chat):  # pragma: no cover - builds the real RAGAS metric
    ensure_ragas_importable()
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    return Faithfulness(llm=LangchainLLMWrapper(chat))


def ensure_ragas_importable() -> None:  # pragma: no cover
    """Work around two broken-import hazards before ``import ragas``.

    1. ``ragas.llms.base`` does ``from langchain_community.chat_models.vertexai
       import ChatVertexAI`` at module import time, but langchain-community 0.4.2
       (the version pinned in this project) removed that module path, so a plain
       ``import ragas`` raises ModuleNotFoundError before any of the symbols we
       use are reachable. We only ever drive RAGAS with an Azure judge, so we
       install a placeholder module that satisfies the import without pulling in
       Google Vertex. If a real Vertex integration is ever present, this is a
       no-op.

    2. ``import ragas`` pulls in ``langchain_core.runnables.passthrough``, which
       constructs ``RunnablePassthrough()`` at module-import time. Depending on
       the order langchain_core's pydantic models get built, that construction
       can raise ``ValidationError: name Field required`` and leave a
       half-imported ``langchain_openai`` cached in ``sys.modules`` that poisons
       every later import. Importing ``langchain_openai`` first builds those
       models in the order it expects, so the subsequent ragas import reuses the
       good cached classes instead of triggering the broken build order.
    """
    import importlib
    import sys
    import types

    importlib.import_module("langchain_openai")

    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
        return
    except ImportError:
        pass

    placeholder = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # minimal stand-in; only used to satisfy the import
        pass

    placeholder.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = placeholder


def prewarm_ragas_imports() -> None:
    """Build the langchain/RAGAS judge models once, at a clean import time.

    Streamlit re-executes the app script on every interaction ("reruns"). The
    faithfulness gate imports ``langchain_openai`` the first time a query runs,
    which lazily constructs langchain_core's ``RunnablePassthrough`` pydantic
    model. langchain_core defers annotation evaluation
    (``from __future__ import annotations``), and building that model for the
    *first* time during a rerun loses the ``name: str | None = None`` default and
    raises ``ValidationError: name Field required`` at
    ``runnables/passthrough.py`` import. Forcing the import once at module load —
    before Streamlit's first rerun — builds the model correctly and caches it for
    the process lifetime. Idempotent: the underlying imports are memoised by
    ``sys.modules``, so repeated calls are cheap no-ops.
    """
    ensure_ragas_importable()
