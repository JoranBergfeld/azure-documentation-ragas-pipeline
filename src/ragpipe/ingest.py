from __future__ import annotations

import hashlib
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
    return f"{url}#{index}-{digest}"


def build_documents(
    pages: list[dict[str, Any]],
    embed_fn: Callable[[str], list[float]],
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Turn fetched pages into Azure AI Search documents."""
    documents: list[dict[str, Any]] = []
    for page in pages:
        chunks = chunk_text(page["text"], max_chars=max_chars, overlap=overlap)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                    "content": chunk,
                    "content_vector": embed_fn(chunk),
                }
            )
    return documents


def fetch_pages(urls: list[str]) -> list[dict[str, Any]]:
    import httpx

    pages: list[dict[str, Any]] = []
    for url in urls:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        pages.append({"url": url, "title": title, "text": html_to_text(resp.text)})
    return pages


def main() -> None:  # pragma: no cover - integration entry point
    import asyncio

    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from agent_framework.foundry import FoundryEmbeddingClient

    from ragpipe.config import Settings
    from ragpipe.search_index import create_index

    settings = Settings.from_env()
    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]

    cred = DefaultAzureCredential()
    embed_client = FoundryEmbeddingClient()

    def embed(text: str) -> list[float]:
        result = asyncio.get_event_loop().run_until_complete(
            embed_client.get_embeddings([text])
        )
        return list(result[0].embedding)

    pages = fetch_pages(urls)
    first_vec = embed(pages[0]["text"][:100])
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    create_index(index_client, settings.search_index, vector_dimensions=len(first_vec))

    docs = build_documents(pages, embed_fn=embed)
    search_client = SearchClient(settings.search_endpoint, settings.search_index, cred)
    search_client.upload_documents(docs)
    print(f"Uploaded {len(docs)} chunks to index '{settings.search_index}'.")


if __name__ == "__main__":  # pragma: no cover
    main()
