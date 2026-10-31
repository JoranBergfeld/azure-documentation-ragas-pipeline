"""The guardrail: what happens after the faithfulness gate has scored an attempt.

Pass, retry with a wider rerank window, or give up and abstain (ADR-0009). The
score itself comes from ``faithfulness_gate``.
"""
from __future__ import annotations

from enum import Enum


class LoopDecision(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    EXHAUSTED = "exhausted"


def decide_next(
    score: float | None, threshold: float, attempt: int, max_retries: int
) -> LoopDecision:
    """Decide whether to accept the answer, retry, or give up.

    A missing score (judge failure) is fail-closed AND fail-fast: it can never
    PASS, and it goes straight to EXHAUSTED — retrying regenerates the answer,
    which cannot fix a judge infrastructure failure and only multiplies cost.
    """
    if score is None:
        return LoopDecision.EXHAUSTED
    if score >= threshold:
        return LoopDecision.PASS
    if attempt < max_retries:
        return LoopDecision.RETRY
    return LoopDecision.EXHAUSTED
