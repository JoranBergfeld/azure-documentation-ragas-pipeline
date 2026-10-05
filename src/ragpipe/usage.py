"""Per-run usage of every metered call (ADR-0018).

A run reports what it consumed so the caller can price it as usage times a
dated rate keyed by deployment. This module holds no prices.

Recording happens where the provider response is in hand (the embed call, the
semantic query, the agent run, the judge's LLM callback). The entries land in
the list that `collect_usage` binds for the current run; outside a run (ingest,
scripts) recording is a no-op.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

# Stage names reported per entry.
STAGE_QUERY_EMBEDDING = "query_embedding"
STAGE_PLAN = "plan"
STAGE_RERANK = "rerank"
STAGE_GENERATION = "generation"
STAGE_CODE_INTERPRETER = "code_interpreter"
STAGE_FAITHFULNESS_JUDGE = "faithfulness_judge"

# Billed per request, with no model deployment behind them. These fixed names
# are what the caller's rate table keys on.
SEMANTIC_RANKER_DEPLOYMENT = "azure-ai-search-semantic-ranker"
CODE_INTERPRETER_DEPLOYMENT = "foundry-code-interpreter"


@dataclass(frozen=True)
class UsageEntry:
    """One metered call. Token calls fill the token counts, per-request billing
    fills `request_units`. `input_tokens` is the total input, cached part
    included; `cached_input_tokens` is the part of it read from cache, or None
    when the provider did not report one. `usage_missing` marks a call whose
    counts the provider did not report."""

    stage: str
    deployment: str
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    request_units: int | None = None
    usage_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "deployment": self.deployment,
            "inputTokens": self.input_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "outputTokens": self.output_tokens,
            "requestUnits": self.request_units,
            "usageMissing": self.usage_missing,
        }


_current: ContextVar[list[UsageEntry] | None] = ContextVar("ragpipe_usage", default=None)


@contextmanager
def collect_usage(entries: list[UsageEntry]) -> Iterator[list[UsageEntry]]:
    """Bind `entries` as the destination for every `record_*` call in this run."""
    token = _current.set(entries)
    try:
        yield entries
    finally:
        _current.reset(token)


def _record(entry: UsageEntry) -> None:
    entries = _current.get()
    if entries is not None:
        entries.append(entry)


def _int_or_none(value: Any) -> int | None:
    # bool is an int subclass; a provider never reports usage as one.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def record_tokens(
    stage: str,
    deployment: str,
    *,
    input_tokens: Any = None,
    cached_input_tokens: Any = None,
    output_tokens: Any = None,
    reason: str = "provider reported no usage",
) -> None:
    """Record a token-billed call from the provider's reported counts.

    Never estimates: without a reported input count the entry is recorded with
    the counts absent, `usage_missing` set, and a line on stderr.
    """
    input_tokens = _int_or_none(input_tokens)
    if input_tokens is None:
        if _current.get() is not None:
            print(
                f"usage: {stage} call to {deployment}: {reason}; counts left empty",
                file=sys.stderr,
                flush=True,
            )
        _record(UsageEntry(stage=stage, deployment=deployment, usage_missing=True))
        return
    _record(
        UsageEntry(
            stage=stage,
            deployment=deployment,
            input_tokens=input_tokens,
            cached_input_tokens=_int_or_none(cached_input_tokens),
            output_tokens=_int_or_none(output_tokens),
        )
    )


def record_requests(stage: str, deployment: str, units: int = 1) -> None:
    """Record a call billed per request (e.g. the semantic ranker)."""
    _record(UsageEntry(stage=stage, deployment=deployment, request_units=units))


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def record_openai_embedding(stage: str, deployment: str, response: Any) -> None:
    """Embeddings response from the openai SDK: `usage.prompt_tokens`, no output."""
    record_tokens(
        stage, deployment, input_tokens=_get(_get(response, "usage"), "prompt_tokens")
    )


def record_openai_chat(stage: str, deployment: str, response: Any) -> None:
    """Chat completions response from the openai SDK."""
    usage = _get(response, "usage")
    record_tokens(
        stage,
        deployment,
        input_tokens=_get(usage, "prompt_tokens"),
        cached_input_tokens=_get(_get(usage, "prompt_tokens_details"), "cached_tokens"),
        output_tokens=_get(usage, "completion_tokens"),
    )


def record_agent_response(stage: str, deployment: str, response: Any) -> None:
    """Agent Framework `AgentResponse`: tokens from `usage_details`, plus one
    code interpreter session when the response shows the tool ran.

    The framework only sets a cached-token key when the count is non-zero, and
    names it per API ("openai.cached_input_tokens" on Responses,
    "prompt/cached_tokens" on Chat Completions).
    """
    usage = _get(response, "usage_details")
    cached = _get(usage, "openai.cached_input_tokens")
    if cached is None:
        cached = _get(usage, "prompt/cached_tokens")
    record_tokens(
        stage,
        deployment,
        input_tokens=_get(usage, "input_token_count"),
        cached_input_tokens=cached,
        output_tokens=_get(usage, "output_token_count"),
    )
    used_code_interpreter = any(
        _get(content, "type") == "code_interpreter_tool_call"
        for message in (_get(response, "messages") or [])
        for content in (_get(message, "contents") or [])
    )
    if used_code_interpreter:
        # Billed per session, not per tool call: one run opens one session.
        record_requests(STAGE_CODE_INTERPRETER, CODE_INTERPRETER_DEPLOYMENT)


def record_langchain_result(stage: str, deployment: str, result: Any) -> None:
    """LangChain `LLMResult` from one model call: reads `usage_metadata`, where
    `input_tokens` already includes cache reads on every provider."""
    for generations in _get(result, "generations") or []:
        first = generations[0] if generations else None
        usage = _get(_get(first, "message"), "usage_metadata")
        record_tokens(
            stage,
            deployment,
            input_tokens=_get(usage, "input_tokens"),
            cached_input_tokens=_get(_get(usage, "input_token_details"), "cache_read"),
            output_tokens=_get(usage, "output_tokens"),
        )


def _is_timeout(error: BaseException) -> bool:
    # The provider SDKs raise their own classes (APITimeoutError, ReadTimeout)
    # that do not subclass TimeoutError. A cancel is RAGAS's wait_for expiring.
    return isinstance(error, (TimeoutError, asyncio.CancelledError)) or (
        "Timeout" in type(error).__name__
    )


def build_langchain_usage_callback(stage: str, deployment: str):
    """Callback handler that records each completed LLM call of a LangChain
    chat model. Pass it in the model's `callbacks`; RAGAS makes several LLM
    calls per score and each one gets its own entry."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageCallback(BaseCallbackHandler):
        # Inline, so the handler runs in the calling task's context and sees
        # the run's entry list instead of hopping to an executor thread.
        run_inline = True

        def on_llm_end(self, response, **kwargs) -> None:
            record_langchain_result(stage, deployment, response)

        def on_llm_error(self, error, **kwargs) -> None:
            # Same rule as a generation timeout: the request went out and may
            # be billed, but no usage came back. Other errors (auth, 4xx) are
            # rejected before anything is metered, so they record nothing.
            if _is_timeout(error):
                record_tokens(stage, deployment, reason="call timed out before a response")

    return _UsageCallback()
