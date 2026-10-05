from __future__ import annotations


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

    from ragpipe.guardrail import prewarm_ragas_imports

    prewarm_ragas_imports()

    assert "langchain_openai" in sys.modules

    from langchain_core.runnables.passthrough import RunnablePassthrough

    assert RunnablePassthrough().name is None


def test_pinned_ragas_exposes_the_per_claim_seam():
    """The live gate keeps per-claim verdicts by driving RAGAS 0.4.3's private
    two-call faithfulness pass (``canary.score_with_claims``, ADR-0018). Fail
    loudly if a RAGAS bump removes or reshapes that seam."""
    import inspect

    from ragpipe.guardrail import _ensure_ragas_importable

    _ensure_ragas_importable()
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
