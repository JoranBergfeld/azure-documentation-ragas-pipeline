"""Judge model transport on the Foundry account, routed by provider (ADR-0011).

The online faithfulness gate (ADR-0009) and synthetic test-item authoring
(ADR-0010) call a judge model through this module. Routing is by provider:
Anthropic (Claude) deployments use the Messages API on the account's
``/anthropic`` route (Entra scope ``https://ai.azure.com/.default``); every
other family (Kimi, DeepSeek, ...) is served on the OpenAI-compatible route of
``<account>.services.ai.azure.com`` (Entra scope
``https://cognitiveservices.azure.com/.default``) — the same route the offline
DeepSeek judge uses.
"""
from __future__ import annotations

from typing import Callable

AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"

# Judge HTTP calls must be bounded (see embeddings._build_client): without a
# request timeout a stalled call blocks its worker thread forever, and bounded
# retries back off on transient 429s/stalls instead of failing the whole eval.
JUDGE_TIMEOUT = 120.0
JUDGE_MAX_RETRIES = 4


def judge_provider(model: str) -> str:
    """Transport provider for a judge deployment name.

    Claude deployments are served on the ``/anthropic`` Messages route; every
    other family (Kimi, DeepSeek, ...) is served on the OpenAI-compatible route.
    The name is the only signal available without a live deployment lookup, and
    Claude is the only Anthropic-route family on this account (ADR-0011).
    """
    return "anthropic" if "claude" in model.lower() else "openai"


def build_judge_complete_fn(settings, max_tokens: int = 2048) -> Callable[[str], str]:
    """`complete(prompt) -> str` against the judge deployment, routed by provider."""
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required to call the judge (ADR-0009); set it in .env"
        )
    if judge_provider(settings.judge_model) == "anthropic":
        return _anthropic_complete_fn(settings, max_tokens)
    return _openai_complete_fn(settings, max_tokens)


def _anthropic_complete_fn(settings, max_tokens: int) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """Claude via the Anthropic Messages API on the account's /anthropic route.

    The client is rebuilt per call so the (cached, auto-refreshed) Entra bearer
    token from azure-identity is always current — client construction is cheap
    next to the model call.
    """
    from anthropic import Anthropic
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    from ragpipe.embeddings import anthropic_endpoint_from_project

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    def complete(prompt: str) -> str:
        client = Anthropic(
            base_url=base_url,
            auth_token=token_provider(),
            timeout=JUDGE_TIMEOUT,
            max_retries=JUDGE_MAX_RETRIES,
        )
        resp = client.messages.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return complete


def _openai_complete_fn(settings, max_tokens: int) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """Non-Anthropic judge (Kimi, DeepSeek, ...) via the OpenAI-compatible route.

    Targets ``<account>.services.ai.azure.com/openai/deployments/<model>`` — the
    same host the offline DeepSeek judge uses. No explicit temperature: reasoning
    deployments on this route may reject sampling overrides.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    from ragpipe.embeddings import (
        COGNITIVE_SERVICES_SCOPE,
        services_endpoint_from_project,
    )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE
    )
    azure_endpoint = services_endpoint_from_project(settings.foundry_project_endpoint)

    def complete(prompt: str) -> str:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-10-21",
            timeout=JUDGE_TIMEOUT,
            max_retries=JUDGE_MAX_RETRIES,
        )
        resp = client.chat.completions.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
