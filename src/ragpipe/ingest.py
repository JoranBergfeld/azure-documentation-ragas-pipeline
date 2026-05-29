from __future__ import annotations

import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from bs4 import BeautifulSoup

from ragpipe.chunking import chunk_text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _doc_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    slug = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"{slug}_{index}_{digest}"


def build_documents(
    pages: list[dict[str, Any]],
    embed_fn: Callable[[str], list[float]],
    max_chars: int = 2000,
    overlap: int = 200,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Turn fetched pages into Azure AI Search documents.

    Embeds chunks concurrently (`max_workers` threads) while preserving order, so
    a large corpus is not embedded one-chunk-at-a-time. `embed_fn` must be safe to
    call from multiple threads (the OpenAI client is).
    """
    metas: list[dict[str, Any]] = []
    chunks: list[str] = []
    for page in pages:
        for i, chunk in enumerate(
            chunk_text(page["text"], max_chars=max_chars, overlap=overlap)
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

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        vectors = list(pool.map(embed_fn, chunks))

    return [
        {**meta, "content": chunk, "content_vector": vector}
        for meta, chunk, vector in zip(metas, chunks, vectors)
    ]


def fetch_pages(
    urls: list[str], max_workers: int = 16
) -> list[dict[str, Any]]:  # pragma: no cover - network
    """Fetch and extract pages concurrently; skip (and log) any that fail."""
    import httpx

    client = httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": "ragpipe-ingest"}
    )

    def fetch_one(url: str) -> dict[str, Any] | None:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = html_to_text(resp.text)
            if not text.strip():
                return None
            return {"url": url, "title": title, "text": text}
        except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the crawl
            print(f"  skip {url}: {type(exc).__name__}", flush=True)
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
    return pages


def _upload_in_batches(
    search_client: Any, docs: list[dict[str, Any]], batch_size: int = 500
) -> None:  # pragma: no cover - network
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        search_client.upload_documents(batch)
        print(f"  uploaded {min(start + batch_size, len(docs))}/{len(docs)} chunks", flush=True)


def main() -> None:  # pragma: no cover - integration entry point
    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from ragpipe.config import Settings
    from ragpipe.embeddings import build_embed_fn
    from ragpipe.search_index import create_index

    settings = Settings.from_env()
    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]

    cred = DefaultAzureCredential()
    embed = build_embed_fn(settings)

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    first_vec = embed(pages[0]["text"][:100])
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    create_index(index_client, settings.search_index, vector_dimensions=len(first_vec))

    print(f"Embedding chunks from {len(pages)} pages…", flush=True)
    docs = build_documents(pages, embed_fn=embed)
    search_client = SearchClient(settings.search_endpoint, settings.search_index, cred)
    _upload_in_batches(search_client, docs)
    print(
        f"Uploaded {len(docs)} chunks from {len(pages)} pages "
        f"to index '{settings.search_index}'."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
