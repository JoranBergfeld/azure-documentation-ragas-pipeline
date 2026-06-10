"""Claude on the Foundry account's /anthropic route (Anthropic Messages API).

Shared by the online faithfulness gate (ADR-0009) and synthetic test-item
authoring (ADR-0010). Entra-only: tokens use the AI Foundry scope, not the
Cognitive Services scope used by the /openai route.
"""
from __future__ import annotations

from typing import Callable

AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def build_claude_complete_fn(
    settings, max_tokens: int = 2048
) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """`complete(prompt) -> str` against the Claude judge deployment.

    The client is rebuilt per call so the (cached, auto-refreshed) Entra bearer
    token from azure-identity is always current — client construction is cheap
    next to the model call.
    """
    if not settings.judge_model:
        raise ValueError(
            "JUDGE_MODEL is required to call Claude (ADR-0009); set it in .env"
        )
    from anthropic import Anthropic
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    from ragpipe.embeddings import anthropic_endpoint_from_project

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_FOUNDRY_SCOPE)
    base_url = anthropic_endpoint_from_project(settings.foundry_project_endpoint)

    def complete(prompt: str) -> str:
        client = Anthropic(base_url=base_url, auth_token=token_provider())
        resp = client.messages.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return complete
