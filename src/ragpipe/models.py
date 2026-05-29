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
    dense: list[Chunk] = field(default_factory=list)
    bm25: list[Chunk] = field(default_factory=list)
    fused: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    answer: str = ""
    faithfulness: float | None = None
    attempt: int = 0
    low_confidence: bool = False
    trace: list[TraceEvent] = field(default_factory=list)

    def add_trace(self, stage: str, data: dict[str, Any]) -> None:
        self.trace.append(TraceEvent(stage=stage, data=data))

    def next_attempt(self) -> None:
        self.attempt += 1
