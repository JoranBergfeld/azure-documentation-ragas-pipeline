import pytest

from ragpipe.config import Settings, TestsetMode


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://proj.services.ai.azure.com")
    monkeypatch.setenv("FOUNDRY_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://s.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "ms-docs")
    monkeypatch.setenv("GENERATOR_AGENT_NAME", "ragpipe-generator")

    s = Settings.from_env()

    assert s.foundry_chat_model == "gpt-4o"
    assert s.search_index == "ms-docs"
    assert s.faithfulness_threshold == 0.7  # default
    assert s.max_retries == 2  # default
    assert s.top_k == 5  # default
    assert s.rrf_k == 60  # default
    assert s.testset_mode is TestsetMode.HANDAUTHORED  # default


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="FOUNDRY_PROJECT_ENDPOINT"):
        Settings.from_env(load=False)
