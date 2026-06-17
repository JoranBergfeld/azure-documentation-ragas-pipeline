import pytest

from ragpipe.embeddings import (
    _embed_in_chunks,
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


def test_embed_in_chunks_splits_preserves_order_and_covers_all():
    calls: list[int] = []

    def embed_one(sub):
        calls.append(len(sub))
        # echo each input's first char code so order is verifiable
        return [[float(ord(s[0]))] for s in sub]

    texts = [chr(ord("a") + i) for i in range(23)]  # a..w
    out = _embed_in_chunks(embed_one, texts, max_inputs=10)

    # 23 inputs -> sub-batches of 10, 10, 3 (none exceeds the cap)
    assert calls == [10, 10, 3]
    # every input embedded exactly once, in original order
    assert out == [[float(ord(t))] for t in texts]


def test_embed_in_chunks_handles_empty_and_exact_multiples():
    assert _embed_in_chunks(lambda sub: [[0.0] for _ in sub], [], max_inputs=5) == []

    sizes: list[int] = []

    def embed_one(sub):
        sizes.append(len(sub))
        return [[1.0] for _ in sub]

    out = _embed_in_chunks(embed_one, ["x"] * 10, max_inputs=5)
    assert sizes == [5, 5]
    assert len(out) == 10
