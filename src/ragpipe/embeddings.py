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


def build_embed_fn(
    settings: Settings, api_version: str = "2024-10-21"
) -> Callable[[str], list[float]]:  # pragma: no cover - live Azure call
    """Return a synchronous `embed(text) -> list[float]` over Azure OpenAI + Entra auth.

    The underlying `openai` client is synchronous, so the returned callable is safe to
    invoke from inside a running event loop (the retrievers call it that way).
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    client = AzureOpenAI(
        azure_endpoint=openai_endpoint_from_project(settings.foundry_project_endpoint),
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    def embed(text: str) -> list[float]:
        result = client.embeddings.create(
            model=settings.foundry_embedding_model, input=[text]
        )
        return list(result.data[0].embedding)

    return embed
