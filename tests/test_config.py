import pytest

from ragpipe.config import Settings, TestsetMode


def test_settings_loads_from_env(monkeypatch):
    _base_env(monkeypatch)
    # Override base env with specific values for this test
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://proj.services.ai.azure.com")
    monkeypatch.setenv("FOUNDRY_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("SEARCH_INDEX", "ms-docs")

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


def _base_env(monkeypatch):
    for k, v in {
        "FOUNDRY_PROJECT_ENDPOINT": "https://acct.services.ai.azure.com/api/projects/p",
        "FOUNDRY_CHAT_MODEL": "gpt-5.4",
        "FOUNDRY_EMBEDDING_MODEL": "text-embedding-3-small",
        "SEARCH_ENDPOINT": "https://s.search.windows.net",
        "SEARCH_INDEX": "idx",
        "GENERATOR_AGENT_NAME": "gen",
    }.items():
        monkeypatch.setenv(k, v)
    # Isolate test env to ensure defaults are tested, not real environment values
    monkeypatch.delenv("FAITHFULNESS_THRESHOLD", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    monkeypatch.delenv("CANDIDATE_POOL", raising=False)
    monkeypatch.delenv("MAX_RETRIES", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OFFLINE_JUDGE_MODEL", raising=False)


def test_judge_models_parsed_from_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OFFLINE_JUDGE_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("CANDIDATE_POOL", "20")
    s = Settings.from_env(load=False)
    assert s.judge_model == "claude-sonnet-4-6"
    assert s.offline_judge_model == "DeepSeek-V4-Pro"
    assert s.candidate_pool == 20


def test_judge_models_default_to_none(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env(load=False)
    assert s.judge_model is None
    assert s.offline_judge_model is None
    assert s.candidate_pool == 15


def test_threshold_out_of_range_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("FAITHFULNESS_THRESHOLD", "7")  # typo for 0.7
    with pytest.raises(ValueError, match="FAITHFULNESS_THRESHOLD"):
        Settings.from_env(load=False)


def test_candidate_pool_smaller_than_top_k_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("CANDIDATE_POOL", "3")
    monkeypatch.setenv("TOP_K", "5")
    with pytest.raises(ValueError, match="CANDIDATE_POOL"):
        Settings.from_env(load=False)


def test_retrieval_mode_values():
    from ragpipe.config import RetrievalMode

    assert RetrievalMode.CONTEXTUAL.value == "contextual"
    assert RetrievalMode.BASELINE.value == "baseline"
    # the 8 spec modes exist as names even if not all wired yet
    assert {"baseline", "baseline_agentic", "raptor_sac", "raptor_sac_agentic",
            "graphrag", "graphrag_agentic", "combined", "combined_agentic"} <= {
        m.value for m in RetrievalMode}


def test_settings_has_no_default_mode_field():
    import dataclasses

    from ragpipe.config import Settings

    names = {f.name for f in dataclasses.fields(Settings)}
    assert "default_mode" not in names
