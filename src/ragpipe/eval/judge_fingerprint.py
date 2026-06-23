from __future__ import annotations

import hashlib
from importlib.metadata import version

from ragpipe.guardrail import _ensure_ragas_importable

EXPECTED_RAGAS_VERSION = "0.4.3"


def installed_ragas_version() -> str:
    return version("ragas")


def version_is_pinned(installed: str, expected: str) -> bool:
    return installed == expected


def prompt_signature(*instructions: str) -> str:
    joined = "\0".join(instructions)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def faithfulness_prompt_instructions() -> tuple[str, str]:
    _ensure_ragas_importable()
    from ragas.metrics._faithfulness import NLIStatementPrompt, StatementGeneratorPrompt

    return (NLIStatementPrompt.instruction, StatementGeneratorPrompt.instruction)


def judge_fingerprint(
    settings,
    *,
    ragas_version: str | None = None,
    prompt_instructions: tuple[str, str] | None = None,
) -> dict:
    if ragas_version is None:
        ragas_version = installed_ragas_version()
    if prompt_instructions is None:
        prompt_instructions = faithfulness_prompt_instructions()
    return {
        "ragas_version": ragas_version,
        "expected_ragas_version": EXPECTED_RAGAS_VERSION,
        "ragas_version_pinned": version_is_pinned(ragas_version, EXPECTED_RAGAS_VERSION),
        "online_judge_model": settings.judge_model or "",
        "offline_judge_model": settings.offline_judge_model or "",
        "generator_model": settings.foundry_chat_model,
        "faithfulness_prompt_signature": prompt_signature(*prompt_instructions),
    }
