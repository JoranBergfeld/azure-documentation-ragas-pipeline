from __future__ import annotations

from typing import Protocol

from ragpipe.models import Chunk


class _Agent(Protocol):
    async def run(self, prompt: str): ...


CORRECTIVE_INSTRUCTION = (
    "Your previous answer (below) contained claims that could not be verified "
    "against the sources. Write a new answer using ONLY claims directly "
    "supported by the numbered sources. If the sources do not contain the "
    "answer, say you don't know.\n\n"
    "<previous_answer>\n{previous_answer}\n</previous_answer>\n\n"
)


def build_grounding_prompt(
    query: str, chunks: list[Chunk], previous_answer: str | None = None
) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({c.url}) {c.content}" for i, c in enumerate(chunks)
    )
    corrective = (
        CORRECTIVE_INSTRUCTION.format(previous_answer=previous_answer)
        if previous_answer
        else ""
    )
    return (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources do not contain the answer, "
        "say you don't know.\n\n"
        f"{corrective}"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class Generator:
    def __init__(self, agent: _Agent) -> None:
        self._agent = agent

    async def generate(
        self, query: str, chunks: list[Chunk], previous_answer: str | None = None
    ) -> str:
        prompt = build_grounding_prompt(query, chunks, previous_answer)
        result = await self._agent.run(prompt)
        return result.text
