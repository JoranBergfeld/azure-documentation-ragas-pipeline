from __future__ import annotations

from typing import Protocol

from ragpipe.models import Chunk


class _Agent(Protocol):
    async def run(self, prompt: str): ...


def build_grounding_prompt(query: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({c.url}) {c.content}" for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources do not contain the answer, "
        "say you don't know.\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class Generator:
    def __init__(self, agent: _Agent) -> None:
        self._agent = agent

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        prompt = build_grounding_prompt(query, chunks)
        result = await self._agent.run(prompt)
        return result.text
