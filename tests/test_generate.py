import asyncio
import contextvars

import pytest

from ragpipe.generate import build_grounding_prompt, Generator
from ragpipe.models import Chunk


def _chunk(cid, content):
    return Chunk(id=cid, title=cid, url=f"http://{cid}", content=content)


def test_build_grounding_prompt_includes_numbered_sources():
    chunks = [_chunk("a", "Alpha fact."), _chunk("b", "Beta fact.")]
    prompt = build_grounding_prompt("What is alpha?", chunks)

    assert "What is alpha?" in prompt
    assert "Alpha fact." in prompt
    assert "Beta fact." in prompt
    assert "[1]" in prompt and "[2]" in prompt


class FakeAgent:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    async def run(self, prompt):
        self.last_prompt = prompt
        return type("R", (), {"text": self._text})()


@pytest.mark.asyncio
async def test_generator_returns_agent_text():
    agent = FakeAgent("Alpha is the first letter.")
    gen = Generator(agent)
    chunks = [_chunk("a", "Alpha fact.")]

    answer = await gen.generate("What is alpha?", chunks)

    assert answer == "Alpha is the first letter."
    assert "Alpha fact." in agent.last_prompt


def test_prompt_without_previous_answer_has_no_corrective_block():
    prompt = build_grounding_prompt("q", [_chunk("a", "Alpha.")])
    assert "previous_answer" not in prompt


def test_prompt_with_previous_answer_includes_corrective_block():
    prompt = build_grounding_prompt(
        "q", [_chunk("a", "Alpha.")], previous_answer="Bad claim."
    )
    assert "<previous_answer>" in prompt
    assert "Bad claim." in prompt
    assert "could not be verified" in prompt


@pytest.mark.asyncio
async def test_generator_threads_previous_answer_into_prompt():
    agent = FakeAgent("better answer")
    gen = Generator(agent)
    await gen.generate("q", [_chunk("a", "Alpha.")], previous_answer="old answer")
    assert "old answer" in agent.last_prompt


class SleepyAgent:
    """Agent whose ``run`` sleeps; optionally only on the first N calls."""

    def __init__(self, text="recovered", sleep_until_call=None, sleep_for=10.0):
        self._text = text
        self._sleep_until_call = sleep_until_call
        self._sleep_for = sleep_for
        self.calls = 0

    async def run(self, prompt):
        self.calls += 1
        if self._sleep_until_call is None or self.calls <= self._sleep_until_call:
            await asyncio.sleep(self._sleep_for)
        return type("R", (), {"text": self._text})()


@pytest.mark.asyncio
async def test_generator_times_out_when_agent_exceeds_timeout():
    agent = SleepyAgent(sleep_for=10.0)
    gen = Generator(agent, timeout=0.01, max_retries=0)
    with pytest.raises(asyncio.TimeoutError):
        await gen.generate("q", [_chunk("a", "Alpha.")])


@pytest.mark.asyncio
async def test_generator_retries_after_timeout_then_succeeds():
    agent = SleepyAgent(text="recovered", sleep_until_call=1, sleep_for=10.0)
    gen = Generator(agent, timeout=0.05, max_retries=1)

    answer = await gen.generate("q", [_chunk("a", "Alpha.")])

    assert answer == "recovered"
    assert agent.calls == 2


class ContextVarAgent:
    """Mimics agent-framework's ContextVar handling: eagerly ``.set()`` a
    ContextVar at call time (in the caller's context), then ``.reset()`` the
    token inside the returned coroutine's ``finally``. ``asyncio.wait_for``
    runs the coroutine in a *copied* context, so the reset blows up with
    "Token was created in a different Context" (issue #5). ``run`` is a plain
    (non-async) method on purpose so the ``.set()`` happens eagerly, exactly
    like agent-framework 1.7.0's ``agent.run()``.
    """

    _CV = contextvars.ContextVar("ctxvar_agent")

    def __init__(self, text="grounded"):
        self._text = text

    def run(self, prompt):
        token = self._CV.set(123)

        async def _run():
            try:
                await asyncio.sleep(0)
                return type("R", (), {"text": self._text})()
            finally:
                self._CV.reset(token)

        return _run()


@pytest.mark.asyncio
async def test_generator_survives_eager_contextvar_set_reset():
    agent = ContextVarAgent("grounded")
    gen = Generator(agent, timeout=5.0, max_retries=0)

    answer = await gen.generate("q", [_chunk("a", "Alpha.")])

    assert answer == "grounded"


@pytest.mark.asyncio
async def test_generator_raises_after_exhausting_retries():
    agent = SleepyAgent(sleep_for=10.0)
    gen = Generator(agent, timeout=0.01, max_retries=2)

    with pytest.raises(asyncio.TimeoutError):
        await gen.generate("q", [_chunk("a", "Alpha.")])

    assert agent.calls == 3
