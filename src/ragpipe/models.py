from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class TraceEvent:
    stage: str
    data: dict[str, Any]


@dataclass
class PipelineState:
    query: str
    # Named intermediate retrieval stages captured for the dashboard and eval.
    # Substrates fill these (e.g. dense/bm25/fused, or local/global/fused). The
    # final reranked set is mirrored in here under "reranked".
    stages: dict[str, list[Chunk]] = field(default_factory=dict)
    # The substrate's final candidate list, fed to the reranker each attempt.
    candidates: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    answer: str = ""
    faithfulness: float | None = None
    attempt: int = 0
    low_confidence: bool = False
    # Directive guardrail (ADR-0009): when retries exhaust, the answer is
    # replaced with a fixed abstention and this flag is set. The suppressed
    # answer survives in the trace only.
    abstained: bool = False
    trace: list[TraceEvent] = field(default_factory=list)

    def set_stage(self, name: str, chunks: list[Chunk]) -> None:
        self.stages[name] = chunks

    def set_reranked(self, chunks: list[Chunk]) -> None:
        self.reranked = chunks
        self.stages["reranked"] = chunks

    def add_trace(self, stage: str, data: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(stage=stage, data=data))

    def next_attempt(self) -> None:
        self.attempt += 1
