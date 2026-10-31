"""Model roles and where each one is served (ADR-0022).

Every model call this project makes fills one role: the generator, the agentic
planner, the ingest writer (situating context, RAPTOR summaries, graph extraction),
the embedder, the online judge, the offline judge, or the reranker. Each role is
served by one inference host at a time: the Foundry account, or an OpenAI-compatible
URL such as vLLM on the DGX Spark. Setting ``<ROLE>_URL`` (plus ``<ROLE>_MODEL`` and,
if the server wants one, ``<ROLE>_API_KEY``) moves that role; leaving it unset keeps
the role on Foundry with the deployment names config already has.

Callers ask for a role and get a client that's bounded (timeout + retries), points at
the right host, carries the right credential and, for roles metered inside a run,
records its usage (ADR-0020). Which Foundry route a model sits on, which Entra scope
it wants and which API version to send all stay in here.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from ragpipe.usage import STAGE_FAITHFULNESS_JUDGE, STAGE_PLAN


class ModelRole(str, Enum):
    GENERATOR = "generator"
    PLANNER = "planner"
    INGEST_WRITER = "ingest_writer"
    EMBEDDER = "embedder"
    ONLINE_JUDGE = "online_judge"
    OFFLINE_JUDGE = "offline_judge"
    RERANKER = "reranker"


class InferenceHost(str, Enum):
    FOUNDRY = "foundry"
    SPARK = "spark"


# The env prefix per role. The judges and the reranker keep the names they already had.
ENV_PREFIX: dict[ModelRole, str] = {
    ModelRole.GENERATOR: "GENERATOR",
    ModelRole.PLANNER: "PLANNER",
    ModelRole.INGEST_WRITER: "INGEST_WRITER",
    ModelRole.EMBEDDER: "EMBEDDER",
    ModelRole.ONLINE_JUDGE: "JUDGE",
    ModelRole.OFFLINE_JUDGE: "OFFLINE_JUDGE",
    ModelRole.RERANKER: "RERANKER",
}

DEFAULT_RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"


@dataclass(frozen=True)
class RoleEndpoint:
    role: ModelRole
    model: str | None
    # The server root, without /v1. None means the role stays on the Foundry account.
    url: str | None = None
    api_key: str | None = None

    @property
    def host(self) -> InferenceHost:
        return InferenceHost.SPARK if self.url else InferenceHost.FOUNDRY


def endpoints_from_env(env: Mapping[str, str]) -> dict[ModelRole, RoleEndpoint]:
    """The roles env configures explicitly. Roles it doesn't mention stay on Foundry."""
    out: dict[ModelRole, RoleEndpoint] = {}
    for role, prefix in ENV_PREFIX.items():
        url = (env.get(f"{prefix}_URL") or "").rstrip("/") or None
        model = env.get(f"{prefix}_MODEL") or None
        if url is None and model is None:
            continue
        if url and model is None and role is not ModelRole.RERANKER:
            # Foundry's deployment names mean nothing on another host.
            raise ValueError(f"{prefix}_MODEL is required when {prefix}_URL is set")
        out[role] = RoleEndpoint(
            role=role, model=model, url=url, api_key=env.get(f"{prefix}_API_KEY") or None
        )
    return out


# --- budgets ----------------------------------------------------------------
# Every live client gets a request timeout and bounded retries: without them a
# stalled call (sent, ACKed, no response, no keepalive) blocks its thread forever,
# which is the eval-harness hang we hit. Bounded retries back off on 429s.

JUDGE_TIMEOUT = 120.0
JUDGE_MAX_RETRIES = 4
# The offline judge is a reasoning model whose faithfulness/context_precision calls
# run for minutes, so it gets a bigger budget than the online gate.
OFFLINE_JUDGE_TIMEOUT = 300.0

_BUDGET: dict[ModelRole, tuple[float, int]] = {
    ModelRole.PLANNER: (60.0, 5),
    ModelRole.INGEST_WRITER: (60.0, 5),
    ModelRole.EMBEDDER: (30.0, 8),
    ModelRole.ONLINE_JUDGE: (JUDGE_TIMEOUT, JUDGE_MAX_RETRIES),
    ModelRole.OFFLINE_JUDGE: (OFFLINE_JUDGE_TIMEOUT, JUDGE_MAX_RETRIES),
}

# Roles whose calls are metered inside a pipeline run. The ingest writer and the
# offline judge only ever run outside one, where recording is a no-op anyway.
_USAGE_STAGE: dict[ModelRole, str] = {
    ModelRole.PLANNER: STAGE_PLAN,
    ModelRole.ONLINE_JUDGE: STAGE_FAITHFULNESS_JUDGE,
}

_JUDGES = (ModelRole.ONLINE_JUDGE, ModelRole.OFFLINE_JUDGE)


# --- the Foundry account ------------------------------------------------------

API_VERSION = "2024-10-21"
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def openai_endpoint_from_project(project_endpoint: str) -> str:
    """The account's Azure OpenAI endpoint, from a Foundry project endpoint.

    e.g. https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project
      -> https://ragpipe-foundry.openai.azure.com/
    """
    host = urlparse(project_endpoint).netloc
    account = host.split(".", 1)[0]
    if not account:
        raise ValueError(f"Cannot derive account name from endpoint: {project_endpoint!r}")
    return f"https://{account}.openai.azure.com/"


def services_endpoint_from_project(project_endpoint: str) -> str:
    """Strip the project path: the account's services.ai.azure.com root.

    The OpenAI-compatible route on this host serves non-OpenAI models sold by
    Azure (e.g. DeepSeek-V4-Pro) at /openai/deployments/<name>. The plain
    <account>.openai.azure.com host serves only Azure OpenAI deployments.
    """
    host = urlparse(project_endpoint).netloc
    if not host:
        raise ValueError(f"Cannot derive host from endpoint: {project_endpoint!r}")
    return f"https://{host}"


def anthropic_endpoint_from_project(project_endpoint: str) -> str:
    """Base URL for the account's Anthropic Messages route (Claude deployments).

    The anthropic SDK appends /v1/messages, matching the documented target
    https://<account>.services.ai.azure.com/anthropic/v1/messages.
    """
    return f"{services_endpoint_from_project(project_endpoint)}/anthropic"


def judge_provider(model: str) -> str:
    """Which Foundry route a judge deployment sits on (ADR-0011).

    Claude is served on the ``/anthropic`` Messages route, every other family on the
    OpenAI-compatible route. The name is the only signal available without a live
    deployment lookup. Off Foundry everything is OpenAI-compatible.
    """
    return "anthropic" if "claude" in model.lower() else "openai"


@functools.lru_cache(maxsize=1)
def _credential():  # one Entra credential per process; it caches its own tokens
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def _token_provider(scope: str):
    from azure.identity import get_bearer_token_provider

    return get_bearer_token_provider(_credential(), scope)


def _foundry_route(settings, role: ModelRole) -> str:
    # Azure OpenAI deployments live on <account>.openai.azure.com; the judges are
    # models sold by Azure, served on the services host's OpenAI-compatible route.
    if role in _JUDGES:
        return services_endpoint_from_project(settings.foundry_project_endpoint)
    return openai_endpoint_from_project(settings.foundry_project_endpoint)


def _spark_base_url(endpoint: RoleEndpoint) -> str:
    return f"{endpoint.url}/v1"


def _spark_key(endpoint: RoleEndpoint) -> str:
    # The SDKs refuse an empty key; vLLM started without --api-key ignores it.
    return endpoint.api_key or "unused"


def _require_model(settings, role: ModelRole) -> RoleEndpoint:
    endpoint = settings.endpoint(role)
    if not endpoint.model:
        raise ValueError(f"{ENV_PREFIX[role]}_MODEL is required for the {role.value} role")
    return endpoint


# --- clients --------------------------------------------------------------------


def openai_client(
    settings, role: ModelRole, *, timeout: float | None = None, max_retries: int | None = None
):
    """An openai-SDK client for the role's host. Both hosts speak the same SDK, so
    callers only ever touch ``.chat.completions`` / ``.embeddings``."""
    endpoint = settings.endpoint(role)
    default_timeout, default_retries = _BUDGET[role]
    timeout = default_timeout if timeout is None else timeout
    max_retries = default_retries if max_retries is None else max_retries
    if endpoint.host is InferenceHost.SPARK:
        from openai import OpenAI

        return OpenAI(
            base_url=_spark_base_url(endpoint),
            api_key=_spark_key(endpoint),
            timeout=timeout,
            max_retries=max_retries,
        )
    if role in _JUDGES and judge_provider(endpoint.model or "") == "anthropic":
        raise ValueError(f"{endpoint.model} is on Foundry's Anthropic route, not OpenAI's")
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=_foundry_route(settings, role),
        azure_ad_token_provider=_token_provider(COGNITIVE_SERVICES_SCOPE),
        api_version=API_VERSION,
        timeout=timeout,
        max_retries=max_retries,
    )


def complete_fn(settings, role: ModelRole, *, max_tokens: int | None = None) -> Callable[[str], str]:
    """``complete(prompt) -> str`` for a text role (planner, ingest writer, a judge).

    No explicit temperature: reasoning deployments reject sampling overrides, and the
    ingest writer gets its determinism from its content-addressed cache (ADR-0005).
    """
    endpoint = _require_model(settings, role)
    if max_tokens is None and role in _JUDGES:
        max_tokens = 2048
    if (
        endpoint.host is InferenceHost.FOUNDRY
        and role in _JUDGES
        and judge_provider(endpoint.model) == "anthropic"
    ):
        return _anthropic_complete_fn(settings, endpoint, max_tokens or 2048)

    client = openai_client(settings, role)
    stage = _USAGE_STAGE.get(role)

    def complete(prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        if stage is not None:
            from ragpipe.usage import record_openai_chat

            record_openai_chat(stage, endpoint.model, resp)
        return resp.choices[0].message.content or ""

    return complete


def _anthropic_complete_fn(settings, endpoint: RoleEndpoint, max_tokens: int):
    """Claude via the Messages API on the account's /anthropic route.

    The client is rebuilt per call so the (cached, auto-refreshed) Entra bearer token
    is always current; construction is cheap next to the model call.
    """
    from anthropic import Anthropic

    token_provider = _token_provider(AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)
    timeout, max_retries = _BUDGET[endpoint.role]

    def complete(prompt: str) -> str:
        client = Anthropic(
            base_url=base_url,
            auth_token=token_provider(),
            timeout=timeout,
            max_retries=max_retries,
        )
        resp = client.messages.create(
            model=endpoint.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return complete


def judge_chat_factory(settings, role: ModelRole) -> Callable[[], Any]:
    """A LangChain chat model for a judge role, for RAGAS to drive.

    Returns a factory rather than a model: on Foundry's Anthropic route the model has
    to be rebuilt for every score, because ChatAnthropic fixes its bearer header at
    construction and Entra tokens expire (~1h). Every other route builds one model and
    hands it back each time.
    """
    endpoint = _require_model(settings, role)
    timeout, max_retries = _BUDGET[role]
    extra: dict[str, Any] = {}
    if role in _USAGE_STAGE:
        from ragpipe.usage import build_langchain_usage_callback

        extra["callbacks"] = [build_langchain_usage_callback(_USAGE_STAGE[role], endpoint.model)]

    if endpoint.host is InferenceHost.SPARK:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            base_url=_spark_base_url(endpoint),
            api_key=_spark_key(endpoint),
            model=endpoint.model,
            timeout=timeout,
            max_retries=max_retries,
            **extra,
        )
        return lambda: chat

    if judge_provider(endpoint.model) == "anthropic":
        from langchain_anthropic import ChatAnthropic

        token_provider = _token_provider(AI_FOUNDRY_SCOPE)
        base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

        def new_chat():
            chat = ChatAnthropic(
                model=endpoint.model,
                base_url=base_url,
                api_key="placeholder",  # satisfies validation; nulled below
                default_headers={"Authorization": f"Bearer {token_provider()}"},
                max_tokens=4096,
                temperature=0,
                timeout=timeout,
                max_retries=max_retries,
                **extra,
            )
            # The anthropic SDK sends X-Api-Key alongside any custom Authorization
            # header; a gateway that validates X-Api-Key first would 401. Clearing the
            # key on both underlying clients removes the header entirely (verified
            # against anthropic 0.109.1: absent from built requests).
            chat._client.api_key = None
            chat._async_client.api_key = None
            return chat

        return new_chat

    from langchain_openai import AzureChatOpenAI

    # No explicit temperature: reasoning deployments on this route may reject
    # sampling overrides. model= sets the request-body "model" field: sold-by-Azure
    # servers (e.g. DeepSeek's sglang) validate it and reject a null.
    chat = AzureChatOpenAI(
        azure_endpoint=services_endpoint_from_project(settings.foundry_project_endpoint),
        azure_deployment=endpoint.model,
        model=endpoint.model,
        api_version=API_VERSION,
        azure_ad_token_provider=_token_provider(COGNITIVE_SERVICES_SCOPE),
        timeout=timeout,
        max_retries=max_retries,
        **extra,
    )
    return lambda: chat


def langchain_embeddings(settings):
    """The embedder role as LangChain embeddings, for the RAGAS suite during eval."""
    endpoint = _require_model(settings, ModelRole.EMBEDDER)
    # Used by the offline eval, so it shares the offline judge's budget.
    timeout, max_retries = _BUDGET[ModelRole.OFFLINE_JUDGE]
    if endpoint.host is InferenceHost.SPARK:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            base_url=_spark_base_url(endpoint),
            api_key=_spark_key(endpoint),
            model=endpoint.model,
            timeout=timeout,
            max_retries=max_retries,
            # Its tiktoken length check only knows OpenAI's tokenizers.
            check_embedding_ctx_length=False,
        )
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=openai_endpoint_from_project(settings.foundry_project_endpoint),
        azure_deployment=endpoint.model,
        api_version=API_VERSION,
        azure_ad_token_provider=_token_provider(COGNITIVE_SERVICES_SCOPE),
        timeout=timeout,
        max_retries=max_retries,
    )


def generator_agent(settings):
    """The generator as an Agent Framework agent with ``async run(prompt)``.

    On Foundry it's the registered agent (``scripts/setup_agents.py``), whose
    instructions and Code Interpreter tool live server side. Off Foundry it's a local
    agent on an OpenAI-compatible client with the same instructions and no tools.
    The per-attempt timeout lives in ``Generator``.
    """
    endpoint = _require_model(settings, ModelRole.GENERATOR)
    if endpoint.host is InferenceHost.SPARK:
        from agent_framework import Agent
        from agent_framework.openai import OpenAIChatCompletionClient

        from ragpipe.generate import GENERATOR_INSTRUCTIONS

        client = OpenAIChatCompletionClient(
            model=endpoint.model,
            base_url=_spark_base_url(endpoint),
            api_key=_spark_key(endpoint),
        )
        return Agent(client=client, instructions=GENERATOR_INSTRUCTIONS)

    from agent_framework.foundry import FoundryAgent

    return FoundryAgent(
        project_endpoint=settings.foundry_project_endpoint,
        agent_name=settings.generator_agent_name,
        agent_version=settings.generator_agent_version,
        credential=_credential(),
    )
