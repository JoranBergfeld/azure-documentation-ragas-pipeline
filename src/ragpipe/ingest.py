from __future__ import annotations

import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from bs4 import BeautifulSoup

from ragpipe.chunking import chunk_markdown
from ragpipe.extraction import html_to_markdown


def _doc_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    slug = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"{slug}_{index}_{digest}"


def build_documents(
    pages: list[dict[str, Any]],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    context_fn: Callable[[str, str], str],
    max_chars: int = 2000,
    overlap: int = 200,
    batch_size: int = 64,
    context_workers: int = 4,
) -> list[dict[str, Any]]:
    """Turn fetched pages into decorated Azure AI Search documents.

    Per chunk: `context` = breadcrumb + generated situating context (ADR-0001),
    `content` = clean chunk text (ADR-0003), `content_vector` = embedding of
    `context + "\\n\\n" + content`. Context generation runs on a small thread
    pool (the callable is cache-backed and thread-safe); embedding stays
    batched as before.
    """
    metas: list[dict[str, Any]] = []
    chunks: list[Any] = []  # MarkdownChunk
    page_md: list[str] = []
    for page in pages:
        for i, chunk in enumerate(
            chunk_markdown(page["markdown"], page["title"], max_chars=max_chars, overlap=overlap)
        ):
            metas.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                }
            )
            chunks.append(chunk)
            page_md.append(page["markdown"])

    total = len(chunks)
    print(f"Generating context for {total} chunks…", flush=True)
    with ThreadPoolExecutor(max_workers=context_workers) as pool:
        generated = list(pool.map(lambda pair: context_fn(*pair), zip(page_md, [c.text for c in chunks])))
    contexts = [
        c.breadcrumb + (f"\n{g}" if g else "") for c, g in zip(chunks, generated)
    ]

    embed_inputs = [f"{ctx}\n\n{c.text}" for ctx, c in zip(contexts, chunks)]
    print(f"Embedding {total} chunks in batches of {batch_size}…", flush=True)
    vectors: list[list[float]] = []
    for start in range(0, total, batch_size):
        vectors.extend(embed_batch_fn(embed_inputs[start : start + batch_size]))
        print(f"  embedded {min(start + batch_size, total)}/{total} chunks", flush=True)

    return [
        {**meta, "content": chunk.text, "context": ctx, "content_vector": vector}
        for meta, chunk, ctx, vector in zip(metas, chunks, contexts, vectors)
    ]


def fetch_pages(
    urls: list[str], max_workers: int = 6, max_retries: int = 4
) -> list[dict[str, Any]]:  # pragma: no cover - network
    """Fetch and extract pages concurrently; retry throttling, skip+log real failures.

    learn.microsoft.com rate-limits aggressive crawls (HTTP 429), so concurrency is
    modest and 429/5xx responses are retried with exponential backoff (honoring a
    Retry-After header when present). Genuine 404s are skipped immediately. Failures
    are logged with the actual status code, not just the exception type, so a high
    skip rate is diagnosable.
    """
    import collections

    import httpx

    client = httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": "ragpipe-ingest"}
    )
    skip_reasons: collections.Counter[str] = collections.Counter()

    def fetch_one(url: str) -> dict[str, Any] | None:
        for attempt in range(max_retries):
            try:
                resp = client.get(url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2**attempt, 30)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                title = (
                    soup.title.string.strip() if soup.title and soup.title.string else url
                )
                markdown, used_main = html_to_markdown(resp.text)
                if not used_main:
                    skip_reasons["main-content fallback (kept)"] += 1
                if not markdown.strip():
                    return None
                return {"url": url, "title": title, "markdown": markdown}
            except httpx.HTTPStatusError as exc:
                skip_reasons[f"HTTP {exc.response.status_code}"] += 1
                return None
            except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the crawl
                skip_reasons[type(exc).__name__] += 1
                return None
        skip_reasons["throttled (gave up)"] += 1
        return None

    pages: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, page in enumerate(pool.map(fetch_one, urls), 1):
            if page is not None:
                pages.append(page)
            if i % 50 == 0:
                print(f"  fetched {i}/{len(urls)} ({len(pages)} ok)", flush=True)
    client.close()
    print(f"Fetched {len(pages)}/{len(urls)} pages.", flush=True)
    if skip_reasons:
        print(f"  skipped: {dict(skip_reasons)}", flush=True)
    return pages


def _upload_in_batches(
    search_client: Any, docs: list[dict[str, Any]], batch_size: int = 500
) -> None:  # pragma: no cover - network
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        search_client.upload_documents(batch)
        print(f"  uploaded {min(start + batch_size, len(docs))}/{len(docs)} chunks", flush=True)


def prune_stale_documents(
    search_client: Any, fresh_ids: set[str], batch_size: int = 500
) -> int:
    """Delete indexed chunks whose id is not in the freshly-uploaded set.

    Uploading is an upsert keyed on id, so re-ingesting a *changed* corpus
    overwrites matching chunks but leaves orphans from a previous, larger corpus
    behind (inflating the doc count and surfacing dead content). We can't just
    recreate the index — a Foundry knowledge source binds to it — so instead we
    enumerate current ids and delete the ones no longer present.

    Safe under indexing lag: stale chunks are old and already searchable, and
    fresh chunks are excluded by membership in `fresh_ids` even if their upload
    hasn't become searchable yet.
    """
    existing = [doc["id"] for doc in search_client.search(search_text="*", select=["id"])]
    stale = [doc_id for doc_id in existing if doc_id not in fresh_ids]
    for start in range(0, len(stale), batch_size):
        batch = [{"id": doc_id} for doc_id in stale[start : start + batch_size]]
        search_client.delete_documents(batch)
    return len(stale)


def main(limit: int | None = None) -> None:  # pragma: no cover - integration entry point
    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from ragpipe.config import Settings
    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.search_index import create_index

    settings = Settings.from_env()
    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]
    if limit is not None:
        urls = urls[:limit]

    cred = DefaultAzureCredential()
    embed_batch = build_batch_embed_fn(settings)
    context_gen = ContextGenerator(
        build_context_complete_fn(settings), model=settings.foundry_chat_model
    )

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    first_vec = embed_batch([pages[0]["markdown"][:100]])[0]
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    create_index(index_client, settings.search_index, vector_dimensions=len(first_vec))

    docs = build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)
    search_client = SearchClient(settings.search_endpoint, settings.search_index, cred)
    _upload_in_batches(search_client, docs)
    if limit is not None:
        # Partial ingest (smoke run): pruning against a partial fresh set would
        # delete the rest of the index. Skip; the next full ingest reconverges.
        print(f"Uploaded {len(docs)} chunks from {len(pages)} pages (limit={limit}, prune skipped).")
        return
    pruned = prune_stale_documents(search_client, fresh_ids={d["id"] for d in docs})
    print(
        f"Uploaded {len(docs)} chunks from {len(pages)} pages "
        f"to index '{settings.search_index}' (pruned {pruned} stale chunks)."
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
