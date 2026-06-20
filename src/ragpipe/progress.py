from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    """A single pipeline progress event. Serializable (``to_dict``) so the same
    object drives the Streamlit checklist and the SSE payload.

    ``phase`` is one of retrieve | rerank | generate | faithfulness | decision |
    abstain, or a nested agentic sub-round: retrieve.plan | retrieve.iter |
    retrieve.fuse. ``status`` is "start" | "complete" | "error". ``detail`` carries
    phase-specific data (e.g. {"score": .82, "threshold": .7, "decision": "retry"})."""

    phase: str
    status: str
    attempt: int = 0
    message: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


ProgressSink = Callable[[ProgressEvent], None]


def emit(
    sink: ProgressSink | None,
    phase: str,
    status: str,
    *,
    attempt: int = 0,
    message: str = "",
    **detail,
) -> None:
    """None-tolerant emit: build a ProgressEvent and call ``sink`` if present."""
    if sink is None:
        return
    sink(
        ProgressEvent(
            phase=phase,
            status=status,
            attempt=attempt,
            message=message,
            detail=dict(detail),
        )
    )
