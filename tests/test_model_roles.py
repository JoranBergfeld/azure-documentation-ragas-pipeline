"""Every model role resolves to one endpoint, on the Foundry account or an
OpenAI-compatible host like the DGX Spark, and gets a bounded client for it.

SDK client classes are swapped for recorders that keep their constructor kwargs, so
no live call is made; only what each client was built with is asserted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragpipe import model_roles, usage
from ragpipe.config import Settings
from ragpipe.model_roles import (
    InferenceHost,
    ModelRole,
    RoleEndpoint,
    endpoints_from_env,
)

PROJECT = "https://acct.services.ai.azure.com/api/projects/p"


def _settings(**endpoints) -> Settings:
    return Settings(
        foundry_project_endpoint=PROJECT,
        foundry_chat_model="gpt-5.4",
        foundry_embedding_model="text-embedding-3-small",
        generator_agent_name="ragpipe-generator",
        generator_agent_version="1",
        judge_model=endpoints.pop("judge_model", "Kimi-K2.5"),
        offline_judge_model="DeepSeek-V4-Pro",
        endpoints={ModelRole(k): v for k, v in endpoints.items()},
    )


def _spark(role, model, key="k"):
    return RoleEndpoint(role=ModelRole(role), model=model, url="http://spark:8101", api_key=key)


# --- where a role is served -------------------------------------------------


def test_every_role_defaults_to_foundry_with_the_existing_models():
    s = _settings()
    llm_roles = [r for r in ModelRole if r is not ModelRole.RERANKER]
    assert {r: (s.endpoint(r).host, s.endpoint(r).model) for r in llm_roles} == {
        ModelRole.GENERATOR: (InferenceHost.FOUNDRY, "gpt-5.4"),
        ModelRole.PLANNER: (InferenceHost.FOUNDRY, "gpt-5.4"),
        ModelRole.INGEST_WRITER: (InferenceHost.FOUNDRY, "gpt-5.4"),
        ModelRole.EMBEDDER: (InferenceHost.FOUNDRY, "text-embedding-3-small"),
        ModelRole.ONLINE_JUDGE: (InferenceHost.FOUNDRY, "Kimi-K2.5"),
        ModelRole.OFFLINE_JUDGE: (InferenceHost.FOUNDRY, "DeepSeek-V4-Pro"),
    }
    # Foundry has no reranker: without a URL it's simply not served anywhere.
    rerank = s.endpoint(ModelRole.RERANKER)
    assert rerank.url is None and rerank.model == "nvidia/llama-nemotron-rerank-1b-v2"


def test_a_url_moves_a_role_onto_the_spark():
    env = {
        "PLANNER_URL": "http://spark:8101/",
        "PLANNER_MODEL": "nvidia/some-chat",
        "PLANNER_API_KEY": "secret",
    }
    endpoints = endpoints_from_env(env)
    assert endpoints[ModelRole.PLANNER] == RoleEndpoint(
        role=ModelRole.PLANNER, model="nvidia/some-chat", url="http://spark:8101", api_key="secret"
    )
    assert endpoints[ModelRole.PLANNER].host is InferenceHost.SPARK
    # Nothing else moves with it.
    assert set(endpoints) == {ModelRole.PLANNER}


def test_the_reranker_keeps_its_existing_env_names():
    endpoints = endpoints_from_env({"RERANKER_URL": "http://spark:8100", "RERANKER_API_KEY": "k"})
    rerank = endpoints[ModelRole.RERANKER]
    assert rerank.url == "http://spark:8100" and rerank.api_key == "k"
    assert rerank.model is None  # falls back to the pinned cross-encoder


def test_the_judges_keep_their_existing_model_names():
    endpoints = endpoints_from_env({"JUDGE_URL": "http://spark:8102", "JUDGE_MODEL": "qwen-judge"})
    assert endpoints[ModelRole.ONLINE_JUDGE].model == "qwen-judge"


def test_a_spark_role_needs_a_model_name():
    # Foundry's deployment names mean nothing on another host.
    with pytest.raises(ValueError, match="EMBEDDER_MODEL"):
        endpoints_from_env({"EMBEDDER_URL": "http://spark:8103"})


def test_settings_from_env_reads_the_role_endpoints(monkeypatch):
    for name, value in {
        "FOUNDRY_PROJECT_ENDPOINT": PROJECT,
        "FOUNDRY_CHAT_MODEL": "gpt-5.4",
        "FOUNDRY_EMBEDDING_MODEL": "text-embedding-3-small",
        "GENERATOR_AGENT_NAME": "gen",
        "GENERATOR_URL": "http://spark:8104",
        "GENERATOR_MODEL": "nvidia/gen",
    }.items():
        monkeypatch.setenv(name, value)
    s = Settings.from_env(load=False)
    assert s.endpoint(ModelRole.GENERATOR).host is InferenceHost.SPARK
    assert s.endpoint(ModelRole.PLANNER).host is InferenceHost.FOUNDRY


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-sonnet-4-6", "anthropic"),
        ("Claude-Sonnet-4-6", "anthropic"),
        ("Kimi-K2.5", "openai"),
        ("DeepSeek-V4-Pro", "openai"),
    ],
)
def test_foundry_routes_judges_by_provider(model, provider):
    assert model_roles.judge_provider(model) == provider


def test_derives_the_foundry_routes_from_the_project_endpoint():
    assert model_roles.openai_endpoint_from_project(PROJECT) == "https://acct.openai.azure.com/"
    assert model_roles.services_endpoint_from_project(PROJECT) == "https://acct.services.ai.azure.com"
    assert (
        model_roles.anthropic_endpoint_from_project(PROJECT)
        == "https://acct.services.ai.azure.com/anthropic"
    )
    with pytest.raises(ValueError):
        model_roles.services_endpoint_from_project("not-a-url")
    with pytest.raises(ValueError):
        model_roles.openai_endpoint_from_project("not-a-url")


# --- clients ---------------------------------------------------------------


@pytest.fixture
def identity(monkeypatch):
    """No real Entra token is ever fetched."""
    import azure.identity as ident

    monkeypatch.setattr(ident, "DefaultAzureCredential", lambda *a, **k: "cred")
    monkeypatch.setattr(ident, "get_bearer_token_provider", lambda cred, scope: lambda: f"tok:{scope}")
    model_roles._credential.cache_clear()


def _recorder(built, *, reply="ok"):
    class _Client:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)
            message = SimpleNamespace(content=reply)
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
            )
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: built.append(kw) or response)
            )
            block = SimpleNamespace(type="text", text=reply)
            self.messages = SimpleNamespace(
                create=lambda **kw: built.append(kw) or SimpleNamespace(content=[block])
            )
            self._client = SimpleNamespace(api_key="x")
            self._async_client = SimpleNamespace(api_key="x")

    return _Client


def test_foundry_openai_client_uses_the_account_route_with_entra(monkeypatch, identity):
    import openai

    built = []
    monkeypatch.setattr(openai, "AzureOpenAI", _recorder(built))
    model_roles.openai_client(_settings(), ModelRole.EMBEDDER)

    assert built[0]["azure_endpoint"] == "https://acct.openai.azure.com/"
    assert built[0]["azure_ad_token_provider"]() == f"tok:{model_roles.COGNITIVE_SERVICES_SCOPE}"
    assert built[0]["timeout"] and built[0]["max_retries"]


def test_spark_openai_client_talks_to_the_v1_route_with_its_key(monkeypatch):
    import openai

    built = []
    monkeypatch.setattr(openai, "OpenAI", _recorder(built))
    s = _settings(embedder=_spark("embedder", "nvidia/embed"))
    model_roles.openai_client(s, ModelRole.EMBEDDER, timeout=7.0, max_retries=2)

    assert built[0] == {
        "base_url": "http://spark:8101/v1", "api_key": "k", "timeout": 7.0, "max_retries": 2,
    }


def test_spark_client_without_a_key_still_satisfies_the_sdk(monkeypatch):
    import openai

    built = []
    monkeypatch.setattr(openai, "OpenAI", _recorder(built))
    s = _settings(planner=_spark("planner", "nvidia/chat", key=None))
    model_roles.openai_client(s, ModelRole.PLANNER)
    assert built[0]["api_key"]  # the SDK refuses an empty key; vLLM without --api-key ignores it


def test_planner_completion_is_recorded_under_the_plan_stage(monkeypatch):
    import openai

    built = []
    monkeypatch.setattr(openai, "OpenAI", _recorder(built, reply="sub one\nsub two"))
    s = _settings(planner=_spark("planner", "nvidia/chat"))

    with usage.collect_usage([]) as entries:
        text = model_roles.complete_fn(s, ModelRole.PLANNER)("split this")

    assert text == "sub one\nsub two"
    assert built[1]["model"] == "nvidia/chat"
    assert built[1]["messages"] == [{"role": "user", "content": "split this"}]
    assert [(e.stage, e.deployment, e.input_tokens) for e in entries] == [
        ("plan", "nvidia/chat", 12)
    ]


def test_ingest_writer_completion_records_nothing(monkeypatch, identity):
    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", _recorder([]))
    with usage.collect_usage([]) as entries:
        model_roles.complete_fn(_settings(), ModelRole.INGEST_WRITER)("situate this chunk")
    assert entries == []


def test_foundry_claude_judge_completion_goes_to_the_anthropic_route(monkeypatch, identity):
    import anthropic

    built = []
    monkeypatch.setattr(anthropic, "Anthropic", _recorder(built))
    s = _settings(judge_model="claude-sonnet-4-6")
    assert model_roles.complete_fn(s, ModelRole.ONLINE_JUDGE)("judge this") == "ok"

    assert built[0]["base_url"] == "https://acct.services.ai.azure.com/anthropic"
    assert built[0]["auth_token"] == f"tok:{model_roles.AI_FOUNDRY_SCOPE}"
    assert built[0]["timeout"] == model_roles.JUDGE_TIMEOUT
    assert built[0]["max_retries"] == model_roles.JUDGE_MAX_RETRIES
    assert built[1]["model"] == "claude-sonnet-4-6" and built[1]["max_tokens"] == 2048


def test_foundry_openai_judge_completion_goes_to_the_services_route(monkeypatch, identity):
    import openai

    built = []
    monkeypatch.setattr(openai, "AzureOpenAI", _recorder(built))
    model_roles.complete_fn(_settings(), ModelRole.ONLINE_JUDGE)("judge this")

    assert built[0]["azure_endpoint"] == "https://acct.services.ai.azure.com"
    assert built[0]["timeout"] == model_roles.JUDGE_TIMEOUT
    assert built[1]["model"] == "Kimi-K2.5" and built[1]["max_tokens"] == 2048


def _chat_recorder(built):
    class _Chat:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self._client = SimpleNamespace(api_key="x")
            self._async_client = SimpleNamespace(api_key="x")

    return _Chat


def test_foundry_claude_judge_chat_is_rebuilt_per_call_without_an_api_key(monkeypatch, identity):
    import langchain_anthropic

    built = []
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", _chat_recorder(built))
    new_chat = model_roles.judge_chat_factory(
        _settings(judge_model="claude-sonnet-4-6"), ModelRole.ONLINE_JUDGE
    )
    first = new_chat()
    new_chat()

    # Entra tokens expire and ChatAnthropic fixes its headers at construction.
    assert len(built) == 2
    assert built[0]["base_url"] == "https://acct.services.ai.azure.com/anthropic"
    assert built[0]["default_headers"] == {
        "Authorization": f"Bearer tok:{model_roles.AI_FOUNDRY_SCOPE}"
    }
    assert built[0]["timeout"] == model_roles.JUDGE_TIMEOUT
    assert built[0]["max_retries"] == model_roles.JUDGE_MAX_RETRIES
    # The SDK would send X-Api-Key next to the bearer header; a gateway may 401 on it.
    assert first._client.api_key is None and first._async_client.api_key is None


def test_foundry_openai_judge_chat_is_built_once_with_the_model_in_the_body(monkeypatch, identity):
    import langchain_openai

    built = []
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", _chat_recorder(built))
    new_chat = model_roles.judge_chat_factory(_settings(), ModelRole.ONLINE_JUDGE)
    assert new_chat() is new_chat()

    assert built[0]["azure_endpoint"] == "https://acct.services.ai.azure.com"
    # sglang-backed deployments 400 on a null body model.
    assert built[0]["model"] == built[0]["azure_deployment"] == "Kimi-K2.5"
    assert built[0]["timeout"] == model_roles.JUDGE_TIMEOUT
    assert built[0]["callbacks"]  # the online judge's tokens land in the run's usage


def test_offline_judge_chat_gets_the_longer_budget_and_no_usage_callback(monkeypatch, identity):
    import langchain_openai

    built = []
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", _chat_recorder(built))
    model_roles.judge_chat_factory(_settings(), ModelRole.OFFLINE_JUDGE)()

    assert built[0]["model"] == "DeepSeek-V4-Pro"
    assert built[0]["timeout"] == model_roles.OFFLINE_JUDGE_TIMEOUT
    assert built[0]["max_retries"] == model_roles.JUDGE_MAX_RETRIES
    assert "callbacks" not in built[0]


def test_spark_judge_chat_is_a_plain_openai_compatible_client(monkeypatch):
    import langchain_openai

    built = []
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _chat_recorder(built))
    s = _settings(online_judge=_spark("online_judge", "qwen-judge"))
    model_roles.judge_chat_factory(s, ModelRole.ONLINE_JUDGE)()

    assert built[0]["base_url"] == "http://spark:8101/v1"
    assert built[0]["api_key"] == "k"
    assert built[0]["model"] == "qwen-judge"
    assert built[0]["timeout"] == model_roles.JUDGE_TIMEOUT


def test_langchain_embeddings_follow_the_embedder_role(monkeypatch, identity):
    import langchain_openai

    foundry, spark = [], []
    monkeypatch.setattr(langchain_openai, "AzureOpenAIEmbeddings", _chat_recorder(foundry))
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", _chat_recorder(spark))

    model_roles.langchain_embeddings(_settings())
    model_roles.langchain_embeddings(_settings(embedder=_spark("embedder", "nvidia/embed")))

    assert foundry[0]["azure_deployment"] == "text-embedding-3-small"
    assert foundry[0]["azure_endpoint"] == "https://acct.openai.azure.com/"
    assert foundry[0]["timeout"] == model_roles.OFFLINE_JUDGE_TIMEOUT
    assert spark[0]["base_url"] == "http://spark:8101/v1" and spark[0]["model"] == "nvidia/embed"


def test_foundry_generator_is_the_registered_agent(monkeypatch, identity):
    import agent_framework.foundry as foundry

    built = []
    monkeypatch.setattr(foundry, "FoundryAgent", _chat_recorder(built))
    model_roles.generator_agent(_settings())

    assert built[0] == {
        "project_endpoint": PROJECT,
        "agent_name": "ragpipe-generator",
        "agent_version": "1",
        "credential": "cred",
    }


def test_spark_generator_is_a_local_agent_with_the_same_instructions(monkeypatch):
    import agent_framework
    import agent_framework.openai as af_openai

    from ragpipe.generate import GENERATOR_INSTRUCTIONS

    clients, agents = [], []
    monkeypatch.setattr(af_openai, "OpenAIChatCompletionClient", _chat_recorder(clients))
    monkeypatch.setattr(agent_framework, "Agent", lambda **kw: agents.append(kw) or kw)
    model_roles.generator_agent(_settings(generator=_spark("generator", "nvidia/gen")))

    assert clients[0] == {"model": "nvidia/gen", "base_url": "http://spark:8101/v1", "api_key": "k"}
    assert agents[0]["instructions"] == GENERATOR_INSTRUCTIONS
    # No Foundry tools off Foundry: Code Interpreter only exists in the Agent Service.
    assert "code interpreter" not in agents[0]["instructions"].lower()
