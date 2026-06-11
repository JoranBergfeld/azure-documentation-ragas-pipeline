import asyncio

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


@pytest.mark.asyncio
async def test_generator_raises_after_exhausting_retries():
    agent = SleepyAgent(sleep_for=10.0)
    gen = Generator(agent, timeout=0.01, max_retries=2)

    with pytest.raises(asyncio.TimeoutError):
        await gen.generate("q", [_chunk("a", "Alpha.")])

    assert agent.calls == 3
