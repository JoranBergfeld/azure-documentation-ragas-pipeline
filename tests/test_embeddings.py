import pytest

from ragpipe.embeddings import (
    anthropic_endpoint_from_project,
    openai_endpoint_from_project,
    services_endpoint_from_project,
)


def test_derives_openai_endpoint_from_project_endpoint():
    project = "https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project"
    assert openai_endpoint_from_project(project) == "https://ragpipe-foundry.openai.azure.com/"


def test_derives_endpoint_when_no_path():
    assert (
        openai_endpoint_from_project("https://acct.services.ai.azure.com")
        == "https://acct.openai.azure.com/"
    )


def test_raises_on_unparseable_endpoint():
    with pytest.raises(ValueError):
        openai_endpoint_from_project("not-a-url")


def test_services_endpoint_strips_project_path():
    ep = services_endpoint_from_project(
        "https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project"
    )
    assert ep == "https://ragpipe-foundry.services.ai.azure.com"


def test_anthropic_endpoint_appends_route():
    ep = anthropic_endpoint_from_project(
        "https://ragpipe-foundry.services.ai.azure.com/api/projects/ragpipe-project"
    )
    assert ep == "https://ragpipe-foundry.services.ai.azure.com/anthropic"


def test_services_endpoint_rejects_garbage():
    with pytest.raises(ValueError):
        services_endpoint_from_project("not-a-url")
