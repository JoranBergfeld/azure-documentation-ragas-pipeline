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
        )
