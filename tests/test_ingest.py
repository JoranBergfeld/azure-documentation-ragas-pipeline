import re

import pytest

from ragpipe.ingest import (
    _graph_extract_workers,
    _parse_args,
    _upload_batch_with_retry,
    build_documents,
    prune_stale_documents,
)


def _fake_batch_embed(texts):
    # one deterministic 2-d vector per input, in order
    return [[float(len(t) % 7), 0.1] for t in texts]


def _no_context(document: str, chunk: str) -> str:
    return ""


def test_build_documents_chunks_embeds_and_decorates():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\n" + "word " * 600}]

    docs = build_documents(
        pages,
        embed_batch_fn=_fake_batch_embed,
        context_fn=lambda doc, chunk: "generated ctx",
        max_chars=1000,
        overlap=100,
        batch_size=2,
    )

    assert len(docs) >= 2
    first = docs[0]
    assert re.fullmatch(r"[A-Za-z0-9_\-=]+", first["id"])
    again = build_documents(
        pages,
        embed_batch_fn=_fake_batch_embed,
        context_fn=lambda doc, chunk: "generated ctx",
        max_chars=1000,
        overlap=100,
        batch_size=2,
    )
    assert again[0]["id"] == first["id"]
    assert docs[0]["id"] != docs[1]["id"]
    assert first["title"] == "T"
    # ADR-0003: content stays clean; decoration lives in `context`
    assert "generated ctx" not in first["content"]
    assert first["context"] == "T > H\ngenerated ctx"
    assert len(first["content_vector"]) == 2


def test_build_documents_breadcrumb_only_on_empty_context():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\nbody text"}]
    docs = build_documents(
        pages, embed_batch_fn=_fake_batch_embed, context_fn=_no_context
    )
    assert docs[0]["context"] == "T > H"


def test_build_documents_embeds_decorated_text_in_chunk_order():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\n" + "word " * 1500}]
    seen = []

    def batch(texts):
        seen.extend(texts)
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    docs = build_documents(
        pages, embed_batch_fn=batch, context_fn=_no_context,
        max_chars=500, overlap=50, batch_size=2,
    )
    # the EMBEDDED text is context + "\n\n" + content (ADR-0003)
    assert [f"{d['context']}\n\n{d['content']}" for d in docs] == seen
    assert len(docs) >= 3


class _FakeSearchClient:
    """Minimal stand-in: returns the indexed ids, records deleted batches."""

    def __init__(self, existing_ids):
        self._existing = [{"id": i} for i in existing_ids]
        self.deleted: list[str] = []

    def search(self, *, search_text, select):
        return iter(self._existing)

    def delete_documents(self, batch):
        self.deleted.extend(doc["id"] for doc in batch)


def test_prune_removes_only_ids_absent_from_fresh_set():
    client = _FakeSearchClient(["a", "b", "c", "d"])

    pruned = prune_stale_documents(client, fresh_ids={"a", "c"}, batch_size=500)

    assert pruned == 2
    assert sorted(client.deleted) == ["b", "d"]


def test_prune_keeps_everything_when_corpus_unchanged():
    client = _FakeSearchClient(["a", "b"])

    pruned = prune_stale_documents(client, fresh_ids={"a", "b"})

    assert pruned == 0
    assert client.deleted == []


def test_prune_batches_large_stale_sets():
    existing = [str(n) for n in range(1200)]
    client = _FakeSearchClient(existing)

    # keep none -> all 1200 are stale, deleted across 500-sized batches
    pruned = prune_stale_documents(client, fresh_ids=set(), batch_size=500)

    assert pruned == 1200
    assert sorted(client.deleted, key=int) == existing


def test_parse_args_defaults_to_contextual_for_backward_compat():
    # The azure.yaml postprovision hook calls `python -m ragpipe.ingest` (no args)
    # and the README smokes with a bare integer limit; both must stay 'contextual'.
    assert _parse_args([]) == ("contextual", None)
    assert _parse_args(["3"]) == ("contextual", 3)
    assert _parse_args(["contextual", "5"]) == ("contextual", 5)


def test_parse_args_selects_substrate_builders_with_optional_limit():
    assert _parse_args(["baseline"]) == ("baseline", None)
    assert _parse_args(["raptor", "3"]) == ("raptor", 3)
    assert _parse_args(["graph", "10"]) == ("graph", 10)


def test_parse_args_rejects_unknown_command():
    with pytest.raises(SystemExit):
        _parse_args(["bogus"])


def test_graph_extract_workers_resolution(monkeypatch):
    # explicit value always wins over the env
    monkeypatch.setenv("RAGPIPE_GRAPH_EXTRACT_WORKERS", "24")
    assert _graph_extract_workers(4) == 4
    # env override applies when unset
    assert _graph_extract_workers() == 24
    # default when neither given
    monkeypatch.delenv("RAGPIPE_GRAPH_EXTRACT_WORKERS", raising=False)
    assert _graph_extract_workers() == 8
    # floored at 1
    assert _graph_extract_workers(0) == 1


class _FlakyClient:
    """upload_documents fails `fail_times` times, then succeeds; records calls."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def upload_documents(self, batch):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("('Connection aborted.', write operation timed out)")


def test_upload_batch_with_retry_recovers_from_transient_failures():
    client = _FlakyClient(fail_times=2)
    sleeps = []
    _upload_batch_with_retry(client, [{"id": "1"}], sleep_fn=sleeps.append)
    assert client.calls == 3  # 2 failures + 1 success
    assert sleeps == [1, 2]  # exponential backoff between attempts


def test_upload_batch_with_retry_raises_after_exhausting_retries():
    client = _FlakyClient(fail_times=99)
    with pytest.raises(RuntimeError):
        _upload_batch_with_retry(client, [{"id": "1"}], max_retries=3, sleep_fn=lambda _d: None)
    assert client.calls == 4  # initial attempt + 3 retries
