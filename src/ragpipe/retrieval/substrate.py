from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragpipe.models import Chunk


@dataclass
class RetrievalResult:
    """What a substrate returns: the final candidate list fed to rerank, plus
    named intermediate stages captured for the dashboard and eval (e.g. dense,
    bm25, fused). The substrate owns its own fusion; the pipeline does not."""

    candidates: list[Chunk]
    stages: dict[str, list[Chunk]] = field(default_factory=dict)


@runtime_checkable
class RetrievalSubstrate(Protocol):
    name: str

    async def retrieve(self, query: str, k: int) -> RetrievalResult: ...
