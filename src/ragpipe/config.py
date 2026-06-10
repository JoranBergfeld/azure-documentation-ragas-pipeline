from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


class TestsetMode(str, Enum):
    HANDAUTHORED = "handauthored"
    SYNTHETIC = "synthetic"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    foundry_project_endpoint: str
    foundry_chat_model: str
    foundry_embedding_model: str
    search_endpoint: str
    search_index: str
    generator_agent_name: str
    generator_agent_version: str | None = None
    faithfulness_threshold: float = 0.7
    max_retries: int = 2
    top_k: int = 5
    rrf_k: int = 60
    testset_mode: TestsetMode = TestsetMode.HANDAUTHORED
    # When true, the offline eval also scores context_precision/recall at each
    # retrieval stage (dense/bm25/fused/reranked) — a heavier per-stage sweep.
    per_stage_metrics: bool = False
    # Judge-model split (ADR-0009): the online faithfulness gate is judged by the
    # Anthropic deployment, the offline RAGAS suite by the DeepSeek deployment —
    # neither shares a family with the gpt generator. None = unset; the builders
    # that need them raise rather than silently falling back to the generator.
    judge_model: str | None = None
    offline_judge_model: str | None = None
    # Candidate pool per retrieval leg (dense/bm25) before RRF fusion. Wider than
    # top_k so guardrail retries can widen the rerank window over real candidates.
    candidate_pool: int = 15

    def __post_init__(self) -> None:
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError(
                f"FAITHFULNESS_THRESHOLD must be in [0, 1], got {self.faithfulness_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"MAX_RETRIES must be >= 0, got {self.max_retries}")
        if self.top_k < 1:
            raise ValueError(f"TOP_K must be >= 1, got {self.top_k}")
        if self.candidate_pool < self.top_k:
            raise ValueError(
                f"CANDIDATE_POOL ({self.candidate_pool}) must be >= TOP_K ({self.top_k})"
            )

    @classmethod
    def from_env(cls, *, load: bool = True) -> "Settings":
        if load:
            load_dotenv()
        return cls(
            foundry_project_endpoint=_require("FOUNDRY_PROJECT_ENDPOINT"),
            foundry_chat_model=_require("FOUNDRY_CHAT_MODEL"),
            foundry_embedding_model=_require("FOUNDRY_EMBEDDING_MODEL"),
            search_endpoint=_require("SEARCH_ENDPOINT"),
            search_index=_require("SEARCH_INDEX"),
            generator_agent_name=_require("GENERATOR_AGENT_NAME"),
            generator_agent_version=os.environ.get("GENERATOR_AGENT_VERSION"),
            faithfulness_threshold=float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7")),
            max_retries=int(os.environ.get("MAX_RETRIES", "2")),
            top_k=int(os.environ.get("TOP_K", "5")),
            rrf_k=int(os.environ.get("RRF_K", "60")),
            testset_mode=TestsetMode(os.environ.get("TESTSET_MODE", "handauthored")),
            per_stage_metrics=os.environ.get("PER_STAGE_METRICS", "").lower()
            in ("1", "true", "yes"),
            judge_model=os.environ.get("JUDGE_MODEL") or None,
            offline_judge_model=os.environ.get("OFFLINE_JUDGE_MODEL") or None,
            candidate_pool=int(os.environ.get("CANDIDATE_POOL", "15")),
        )
