"""The embedder role: query embeddings in a run, batches during ingest and import.

Which host serves it (the Foundry account's Azure OpenAI route, or an
OpenAI-compatible server like the DGX Spark) is ``model_roles``' business; both speak
the openai SDK, so this module only owns how inputs are clipped, batched and ordered.
(On Foundry the embedding model is an Azure OpenAI deployment on the ``/openai``
route, which is why it doesn't go through ``FoundryEmbeddingClient``: that targets
the Inference ``/models`` API and answers 200 with an empty body for it.)
"""
from __future__ import annotations

from typing import Callable

from ragpipe.config import Settings
from ragpipe.model_roles import ModelRole, openai_client
from ragpipe.usage import STAGE_QUERY_EMBEDDING, record_openai_embedding


def build_embed_fn(
    settings: Settings,
    timeout: float = 30.0,
    max_retries: int = 8,
) -> Callable[[str], list[float]]:  # pragma: no cover - live model call
    """Return a synchronous `embed(text) -> list[float]` for the embedder role.

    The underlying `openai` client is synchronous, so the returned callable is safe to
    invoke from inside a running event loop (the retrievers call it that way).

    `timeout` and `max_retries` are essential for bulk ingestion: without a request
    timeout a throttled/stalled call blocks its worker thread indefinitely (the
    openai client otherwise waits on the socket forever). With them, 429s back off
    and retry a bounded number of times, then raise instead of hanging.
    """
    client = openai_client(settings, ModelRole.EMBEDDER, timeout=timeout, max_retries=max_retries)
    model = settings.endpoint(ModelRole.EMBEDDER).model

    def embed(text: str) -> list[float]:
        result = client.embeddings.create(model=model, input=[text])
        # No-op outside a pipeline run, so ingest callers are unaffected.
        record_openai_embedding(STAGE_QUERY_EMBEDDING, model, result)
        return list(result.data[0].embedding)

    return embed


# Per-input character ceiling. text-embedding-3 rejects any input over 8192
# tokens with a 400 that fails the whole batch. Merged GraphRAG entity
# descriptions grow without bound (a hub entity like AZURE unions descriptions
# from thousands of chunks; the largest reached ~50k chars). 16k chars stays
# under 8192 tokens even at ~2 chars/token, and the leading descriptions carry
# the entity's meaning. Only the embedding input is clipped, never stored text.
EMBED_INPUT_CHAR_LIMIT = 16_000


def _embed_in_chunks(
    embed_one: Callable[[list[str]], list[list[float]]],
    texts: list[str],
    max_inputs: int,
    max_chars: int = EMBED_INPUT_CHAR_LIMIT,
) -> list[list[float]]:
    """Embed ``texts`` by splitting into ``<= max_inputs`` sub-batches, in order.

    A single embeddings request is bounded by both an input-array cap (2048) and
    the deployment's tokens-per-minute limit -- passing every entity/relationship
    description at once (tens of thousands of inputs, >1M tokens) guarantees a 429
    that no retry budget can clear. Chunking keeps each request small; results are
    concatenated in input order. Each input is clipped to ``max_chars`` so one
    oversized text cannot 400 its whole sub-batch.
    """
    clipped = [t[:max_chars] for t in texts]
    vectors: list[list[float]] = []
    for start in range(0, len(clipped), max_inputs):
        vectors.extend(embed_one(clipped[start : start + max_inputs]))
    return vectors


def build_batch_embed_fn(
    settings: Settings,
    timeout: float = 60.0,
    max_retries: int = 10,
    max_inputs_per_request: int = 512,
    on_tokens: Callable[[int], None] | None = None,
) -> Callable[[list[str]], list[list[float]]]:  # pragma: no cover - live model call
    """Return `embed_batch(texts) -> list[vector]` sending many inputs per request.

    The embeddings API accepts an array of inputs in a single request, so a batch
    of N chunks costs one request instead of N. This is essential for bulk ingest:
    it keeps total requests far under the deployment's per-10s request limit (the
    one-request-per-chunk approach thrashed on 429s). Results are returned in input
    order (the API guarantees response `index` ordering, which we sort on).

    Inputs are chunked to `max_inputs_per_request` so an arbitrarily large list
    (e.g. tens of thousands of graph entity/relationship descriptions) is split
    into several modestly-sized requests instead of one oversized request that
    would blow past the array cap and the per-minute token limit.

    ``on_tokens`` gets each request's billed prompt tokens, for callers that
    report their own spend (the Postgres import).
    """
    client = openai_client(settings, ModelRole.EMBEDDER, timeout=timeout, max_retries=max_retries)
    model = settings.endpoint(ModelRole.EMBEDDER).model

    def _embed_one(sub: list[str]) -> list[list[float]]:
        result = client.embeddings.create(model=model, input=sub)
        if on_tokens is not None:
            on_tokens(result.usage.prompt_tokens)
        ordered = sorted(result.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]

    def embed_batch(texts: list[str]) -> list[list[float]]:
        return _embed_in_chunks(_embed_one, texts, max_inputs_per_request)

    return embed_batch
