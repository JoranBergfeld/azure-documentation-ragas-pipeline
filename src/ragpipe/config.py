from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

from dotenv import load_dotenv

from ragpipe.model_roles import (
    DEFAULT_RERANKER_MODEL,
    ModelRole,
    RoleEndpoint,
    endpoints_from_env,
)


class TestsetMode(str, Enum):
    HANDAUTHORED = "handauthored"
    SYNTHETIC = "synthetic"


class RetrievalMode(str, Enum):
    # Phase 1: the existing decorated index (default) and the new plain index.
    CONTEXTUAL = "contextual"
    BASELINE = "baseline"
    BASELINE_AGENTIC = "baseline_agentic"
    # Later phases register substrates for these; declared up front for a stable
    # API surface and config.
    RAPTOR_SAC = "raptor_sac"
    RAPTOR_SAC_AGENTIC = "raptor_sac_agentic"
    GRAPHRAG = "graphrag"
    GRAPHRAG_AGENTIC = "graphrag_agentic"
    COMBINED = "combined"
    COMBINED_AGENTIC = "combined_agentic"


# The four agentic wrapper modes have no eval coverage (there is no committed
# eval_results_*_agentic.json) so they are exposed as experimental/unevaluated until an
# eval run scores them (issue #11, ADR-0018). The set is derived structurally -- every
# "*_agentic" wrapper -- so any future wrapper is experimental by default until it earns
# committed eval evidence.
EXPERIMENTAL_MODES: frozenset[RetrievalMode] = frozenset(
    m for m in RetrievalMode if m.value.endswith("_agentic")
)


def is_experimental_mode(mode: "RetrievalMode | str") -> bool:
    """True for retrieval modes exposed without eval coverage (the ``*_agentic``
    wrappers). Accepts a ``RetrievalMode`` or its string value."""
    value = mode.value if isinstance(mode, RetrievalMode) else str(mode)
    return any(value == m.value for m in EXPERIMENTAL_MODES)


# Modes whose last retrieval step is a cross-encoder rerank over one table. Azure's
# semantic ranker went with the search service (issue #37, ADR-0021); its
# replacement is a cross-encoder on vLLM on the DGX Spark (issue #40). These run
# only while it answers, and /modes says so. Running them without a reranker would
# change what their eval numbers describe, so they're offline rather than degraded.
RERANK_MODES: frozenset[RetrievalMode] = frozenset(
    {RetrievalMode.CONTEXTUAL, RetrievalMode.BASELINE, RetrievalMode.RAPTOR_SAC}
)
RERANKER_OFFLINE = "reranker-offline"


def unavailable_reason(mode: "RetrievalMode | str") -> str | None:
    """Why a mode can't run right now, or None when it can. The rerank modes need
    ``RERANKER_URL`` set and its ``/health`` answering (probed, cached a minute)."""
    if RetrievalMode(mode) in RERANK_MODES:
        from ragpipe.retrieval.cross_encoder import reranker_reachable

        if not reranker_reachable():
            return RERANKER_OFFLINE
    return None


class ModeUnavailable(RuntimeError):
    def __init__(self, mode: "RetrievalMode | str", reason: str) -> None:
        self.mode = RetrievalMode(mode).value
        self.reason = reason
        super().__init__(f"mode {self.mode!r} is unavailable: {reason}")


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
    # Retrieval's Postgres (ADR-0021). The service gets a read-only login; the
    # import runs as the owner. Optional so the canary and the judge scripts,
    # which never retrieve, don't need it. Wiring a pipeline refuses without it.
    rag_database_url: str | None = None
    # Route GraphRAG's global community-summary leg by query class instead of
    # always RRF-fusing it (issue #8, ADR-0018). When True, the global leg fires
    # only for sensemaking/breadth queries; factoid queries stay local-only. Set
    # False to restore the legacy always-fuse behavior for A/B evaluation.
    graph_query_routing: bool = True
    # Bounds for later phases (declared now so config is stable).
    agentic_max_iterations: int = 3
    raptor_max_levels: int = 3
    graph_community_levels: int = 1
    # Roles env moved or renamed explicitly (ADR-0022); every other role stays on the
    # Foundry account under the deployment names above. Read through endpoint().
    endpoints: Mapping[ModelRole, RoleEndpoint] = field(default_factory=dict)

    def endpoint(self, role: ModelRole) -> RoleEndpoint:
        """Where a model role is served and under which model name."""
        default_model = {
            ModelRole.GENERATOR: self.foundry_chat_model,
            ModelRole.PLANNER: self.foundry_chat_model,
            ModelRole.INGEST_WRITER: self.foundry_chat_model,
            ModelRole.EMBEDDER: self.foundry_embedding_model,
            ModelRole.ONLINE_JUDGE: self.judge_model,
            ModelRole.OFFLINE_JUDGE: self.offline_judge_model,
            ModelRole.RERANKER: DEFAULT_RERANKER_MODEL,
        }[role]
        configured = self.endpoints.get(role)
        if configured is None:
            return RoleEndpoint(role=role, model=default_model)
        return replace(configured, model=configured.model or default_model)

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
            rag_database_url=os.environ.get("RAG_DATABASE_URL") or None,
            graph_query_routing=os.environ.get("GRAPH_QUERY_ROUTING", "true").lower()
            in ("1", "true", "yes"),
            agentic_max_iterations=int(os.environ.get("AGENTIC_MAX_ITERATIONS", "3")),
            raptor_max_levels=int(os.environ.get("RAPTOR_MAX_LEVELS", "3")),
            graph_community_levels=int(os.environ.get("GRAPH_COMMUNITY_LEVELS", "1")),
            endpoints=endpoints_from_env(os.environ),
        )
