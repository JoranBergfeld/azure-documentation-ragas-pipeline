"""Embedding helper.

`FoundryEmbeddingClient` (agent_framework.foundry) targets the Azure AI *Inference*
(`/models`) API, which serves Models-as-a-Service models (Cohere, etc.). Our embedding
model, `text-embedding-3-small`, is an **Azure OpenAI** deployment — served only on the
`/openai` route, not `/models`. (Calling the `/models` route for it returns HTTP 200 with
an empty body and `grpc-message: Unrecognized endpoint`.) So for an Azure OpenAI embedding
model the correct client is the OpenAI-compatible one, not `FoundryEmbeddingClient`.

We reach the deployed model through the Foundry account's Azure OpenAI endpoint
(`<account>.openai.azure.com`) with a Microsoft Entra token — no API key. Generation still
uses `FoundryAgent` (the Foundry Agent Service); only embeddings use this OpenAI path,
which matches how the embedding model is actually served.
"""
from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from ragpipe.config import Settings

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def openai_endpoint_from_project(project_endpoint: str) -> str:
    """Derive the account's Azure OpenAI endpoint from a Foundry project endpoint.

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
    Azure (e.g. DeepSeek-V4-Pro) at /openai/deployments/<name> — the plain
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


def _build_client(settings: Settings, api_version: str, timeout: float, max_retries: int):  # pragma: no cover - live Azure
    """Build an AzureOpenAI client for the account's /openai endpoint with Entra auth.

    `timeout` + `max_retries` matter for bulk work: without a timeout a throttled
    call blocks its worker thread indefinitely; bounded retries back off on 429s.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    return AzureOpenAI(
        azure_endpoint=openai_endpoint_from_project(settings.foundry_project_endpoint),
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        timeout=timeout,
        max_retries=max_retries,
    )


def build_embed_fn(
    settings: Settings,
    api_version: str = "2024-10-21",
    timeout: float = 30.0,
    max_retries: int = 5,
) -> Callable[[str], list[float]]:  # pragma: no cover - live Azure call
    """Return a synchronous `embed(text) -> list[float]` over Azure OpenAI + Entra auth.

    The underlying `openai` client is synchronous, so the returned callable is safe to
    invoke from inside a running event loop (the retrievers call it that way).

    `timeout` and `max_retries` are essential for bulk ingestion: without a request
    timeout a throttled/stalled call blocks its worker thread indefinitely (the
    openai client otherwise waits on the socket forever). With them, 429s back off
    and retry a bounded number of times, then raise instead of hanging.
    """
    client = _build_client(settings, api_version, timeout, max_retries)

    def embed(text: str) -> list[float]:
        result = client.embeddings.create(
            model=settings.foundry_embedding_model, input=[text]
        )
        return list(result.data[0].embedding)

    return embed


def build_batch_embed_fn(
    settings: Settings,
    api_version: str = "2024-10-21",
    timeout: float = 60.0,
    max_retries: int = 6,
) -> Callable[[list[str]], list[list[float]]]:  # pragma: no cover - live Azure call
    """Return `embed_batch(texts) -> list[vector]` sending many inputs per request.

    The embeddings API accepts an array of inputs in a single request, so a batch
    of N chunks costs one request instead of N. This is essential for bulk ingest:
    it keeps total requests far under the deployment's per-10s request limit (the
    one-request-per-chunk approach thrashed on 429s). Results are returned in input
    order (the API guarantees response `index` ordering, which we sort on).
    """
    client = _build_client(settings, api_version, timeout, max_retries)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        result = client.embeddings.create(
            model=settings.foundry_embedding_model, input=texts
        )
        ordered = sorted(result.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]

    return embed_batch
