#!/usr/bin/env python3
"""Register the Foundry generator agent (with the Code Interpreter tool).

Run after `azd up` (or via the azd postprovision hook):
    python scripts/setup_agents.py

Uses the Foundry projects (new) API: an agent is a named, versioned asset created
with ``project.agents.create_version(agent_name=..., definition=PromptAgentDefinition(...))``.
"""
from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

from ragpipe.config import Settings
from ragpipe.generate import CODE_INTERPRETER_HINT, GENERATOR_INSTRUCTIONS


def main() -> None:
    settings = Settings.from_env()
    client = AIProjectClient(
        endpoint=settings.foundry_project_endpoint,
        credential=DefaultAzureCredential(),
    )
    agent = client.agents.create_version(
        agent_name=settings.generator_agent_name,
        definition=PromptAgentDefinition(
            model=settings.foundry_chat_model,
            instructions=GENERATOR_INSTRUCTIONS + CODE_INTERPRETER_HINT,
            tools=[CodeInterpreterTool()],
        ),
    )
    print(f"Created agent '{agent.name}' version {agent.version}")


if __name__ == "__main__":
    main()
