from __future__ import annotations

from ragpipe.eval.judge_fingerprint import (
    EXPECTED_RAGAS_VERSION,
    faithfulness_prompt_instructions,
    installed_ragas_version,
    judge_fingerprint,
    prompt_signature,
    version_is_pinned,
)


class _S:
    judge_model = "claude-sonnet-4-6"
    offline_judge_model = "DeepSeek-V4-Pro"
    foundry_chat_model = "gpt-5.4"


def test_version_is_pinned_compares_exact_versions():
    assert version_is_pinned("0.4.3", EXPECTED_RAGAS_VERSION) is True
    assert version_is_pinned("0.4.4", EXPECTED_RAGAS_VERSION) is False


def test_prompt_signature_is_stable_and_order_sensitive():
    assert prompt_signature("a", "b") == prompt_signature("a", "b")
    assert prompt_signature("a", "b") != prompt_signature("ab", "")
    assert len(prompt_signature("a", "b")) == 16


def test_judge_fingerprint_assembles_injected_values():
    fingerprint = judge_fingerprint(
        _S(),
        ragas_version="0.4.3",
        prompt_instructions=("nli", "statements"),
    )

    assert fingerprint == {
        "ragas_version": "0.4.3",
        "expected_ragas_version": "0.4.3",
        "ragas_version_pinned": True,
        "online_judge_model": "claude-sonnet-4-6",
        "offline_judge_model": "DeepSeek-V4-Pro",
        "generator_model": "gpt-5.4",
        "faithfulness_prompt_signature": prompt_signature("nli", "statements"),
    }


def test_judge_fingerprint_handles_missing_judge_fields():
    class _Missing:
        foundry_chat_model = "gpt-5.4"
        judge_model = None
        offline_judge_model = None

    fingerprint = judge_fingerprint(
        _Missing(),
        ragas_version="0.4.4",
        prompt_instructions=("nli", "statements"),
    )

    assert fingerprint["ragas_version_pinned"] is False
    assert fingerprint["online_judge_model"] == ""
    assert fingerprint["offline_judge_model"] == ""


def test_real_ragas_version_and_prompts_are_readable_without_network():
    assert installed_ragas_version()
    instructions = faithfulness_prompt_instructions()
    assert len(instructions) == 2
    assert all(isinstance(text, str) and text for text in instructions)
