"""Register the Foundry generator agent with the Code Interpreter tool.

Run once after `azd up` (or via the azd postprovision hook):
    python scripts/setup_agents.py
Writes GENERATOR_AGENT_NAME / GENERATOR_AGENT_VERSION to stdout for .env.
"""
from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from ragpipe.config import Settings

INSTRUCTIONS = (
    "You are a Microsoft/Azure documentation assistant. Answer using only the "
    "provided numbered sources and cite them inline like [1]. When a question "
    "requires counting, comparison tables, or arithmetic over the sourced facts, "
    "use the code interpreter tool. Never invent facts not present in the sources."
)


def main() -> None:  # pragma: no cover
    settings = Settings.from_env()
    client = AIProjectClient(
        endpoint=settings.foundry_project_endpoint,
        credential=DefaultAzureCredential(),
    )
    agent = client.agents.create_agent(
        model=settings.foundry_chat_model,
        name=settings.generator_agent_name,
        instructions=INSTRUCTIONS,
        tools=[{"type": "code_interpreter"}],
    )
    print(f"GENERATOR_AGENT_NAME={agent.name}")
    print(f"GENERATOR_AGENT_VERSION={getattr(agent, 'version', '1.0')}")


if __name__ == "__main__":  # pragma: no cover
    main()
