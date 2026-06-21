from __future__ import annotations

import asyncio
from typing import Protocol

from ragpipe.models import Chunk

# Without an application-level timeout, a single stalled FoundryAgent call
# (request sent, ACKed, but no response — no keepalive on the socket) blocks the
# whole pipeline indefinitely; this is the eval-harness hang we observed. The
# generator legitimately takes minutes (reasoning + Code Interpreter), so the
# per-attempt budget is generous; a bounded retry recovers from a transient
# stall without unbounded waiting.
DEFAULT_GENERATE_TIMEOUT = 300.0
DEFAULT_GENERATE_MAX_RETRIES = 1


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
    def __init__(
        self,
        agent: _Agent,
        timeout: float = DEFAULT_GENERATE_TIMEOUT,
        max_retries: int = DEFAULT_GENERATE_MAX_RETRIES,
    ) -> None:
        self._agent = agent
        self._timeout = timeout
        self._max_retries = max_retries

    async def generate(
        self, query: str, chunks: list[Chunk], previous_answer: str | None = None
    ) -> str:
        prompt = build_grounding_prompt(query, chunks, previous_answer)
        last_exc: asyncio.TimeoutError | None = None
        for _ in range(self._max_retries + 1):
            try:
                # Use asyncio.timeout (a context manager) rather than
                # asyncio.wait_for. On Python 3.11 wait_for runs the awaited
                # coroutine as a child Task in a *copied* context, which breaks
                # agent-framework's eager ContextVar set/reset pair ("Token was
                # created in a different Context"; issue #5). asyncio.timeout
                # does not spawn a child Task, so set/reset stay balanced; the
                # TimeoutError semantics are identical.
                async with asyncio.timeout(self._timeout):
                    result = await self._agent.run(prompt)
                return result.text
            except asyncio.TimeoutError as exc:
                last_exc = exc
        raise last_exc
