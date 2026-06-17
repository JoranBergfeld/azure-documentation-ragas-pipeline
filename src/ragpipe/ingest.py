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


def build_baseline_documents(
    pages: list[dict[str, Any]],
    embed_fn: Callable[[str], list[float]],
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Build plain (no contextual decoration) documents for the baseline index.

    Mirrors build_documents field shapes but skips the ContextGenerator entirely:
    - ``context`` field is empty string
    - ``content_vector`` embeds raw ``content`` only (not context + content)
    """
    docs: list[dict[str, Any]] = []
    for page in pages:
        for i, chunk in enumerate(
            chunk_markdown(page["markdown"], page["title"], max_chars=max_chars, overlap=overlap)
        ):
            content = chunk.text
            docs.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                    "content": content,
                    "context": "",
                    "content_vector": embed_fn(content),
                }
            )
    return docs


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


def raptor_summary_documents(nodes: list[Any]) -> list[dict[str, Any]]:
    """Shape RAPTOR summary nodes into the same field layout as leaf docs, plus `level`.

    Summary nodes don't come from a URL or a discrete integer chunk, so `url` is
    empty and `chunk_id` is 0. The `level` field (>= 1) lets callers distinguish
    summaries from leaves in a mixed index.
    """
    return [
        {
            "id": node.id,
            "title": f"RAPTOR summary L{node.level}",
            "url": "",
            "chunk_id": 0,
            "content": node.text,
            "context": "",
            "content_vector": node.embedding,
            "level": node.level,
        }
        for node in nodes
    ]


def build_raptor(settings: Any, limit: int | None = None) -> None:  # pragma: no cover
    """Ingest the RAPTOR SAC index: leaf docs with level=0 plus RAPTOR summary nodes.

    Mirrors build_baseline but builds the full RAPTOR tree on top of the SAC
    (situating-context + embedding) leaf docs and uploads both leaves and summaries
    to settings.raptor_sac_index with include_level=True.
    """
    import sys

    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.raptor import RaptorNode, build_raptor_tree
    from ragpipe.search_index import build_index

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

    # 1. Build SAC leaf docs (same decoration as main())
    leaf_docs = build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)
    for d in leaf_docs:
        d["level"] = 0

    # 2. Wrap leaf docs as RaptorNode leaves
    leaves = [
        RaptorNode(id=d["id"], text=d["content"], level=0, embedding=d["content_vector"])
        for d in leaf_docs
    ]

    # 3. Summarizer: bounded chat-completion, same client factory as context_gen.
    #    Not content-address-cached (ADR-0005 style); would need a separate cache
    #    keyed on cluster membership which changes every ingest. Acceptable for now.
    complete_fn = build_context_complete_fn(settings)

    SUMMARY_PROMPT = (
        "You are a technical writer. Write a concise abstractive summary of the "
        "following related Azure documentation passages. Capture the key concepts "
        "and relationships. Reply with only the summary, no preamble.\n\n{passages}"
    )

    def summarize_fn(texts: list[str]) -> str:
        joined = "\n\n---\n\n".join(texts)
        prompt = SUMMARY_PROMPT.format(passages=joined)
        for attempt in range(settings.max_retries + 1):
            try:
                return complete_fn(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"summarize_fn attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return " ".join(texts[:3])[:500]  # last-resort truncated concat

    # 4. Build RAPTOR tree
    print(f"Building RAPTOR tree from {len(leaves)} leaves (max_levels={settings.raptor_max_levels})…", flush=True)
    summaries = build_raptor_tree(
        leaves,
        summarize_fn=summarize_fn,
        embed_batch_fn=embed_batch,
        max_levels=settings.raptor_max_levels,
    )
    print(f"  RAPTOR produced {len(summaries)} summary nodes.", flush=True)

    # 5. Create/update the RAPTOR SAC index with include_level=True
    dims = len(leaf_docs[0]["content_vector"])
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    index_client.create_or_update_index(
        build_index(settings.raptor_sac_index, dims, include_context=True, include_level=True)
    )

    # 6. Upload leaves + summaries; prune stale (skip on partial ingest)
    docs = leaf_docs + raptor_summary_documents(summaries)
    search_client = SearchClient(settings.search_endpoint, settings.raptor_sac_index, cred)
    _upload_in_batches(search_client, docs)

    if limit is not None:
        print(
            f"Uploaded {len(leaf_docs)} leaves + {len(summaries)} summary nodes "
            f"to index '{settings.raptor_sac_index}' (limit={limit}, prune skipped).",
            flush=True,
        )
        return

    pruned = prune_stale_documents(search_client, fresh_ids={d["id"] for d in docs})
    print(
        f"Uploaded {len(leaf_docs)} leaves + {len(summaries)} summary nodes "
        f"to index '{settings.raptor_sac_index}' (pruned {pruned} stale).",
        flush=True,
    )


def build_baseline(settings: Any, limit: int | None = None) -> None:  # pragma: no cover
    """Ingest the plain-chunk baseline index (no contextual decoration).

    Mirrors main() but uses build_baseline_documents and targets
    settings.baseline_index with include_context=False.
    """
    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from ragpipe.embeddings import build_batch_embed_fn

    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]
    if limit is not None:
        urls = urls[:limit]

    cred = DefaultAzureCredential()
    embed_batch = build_batch_embed_fn(settings)

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    first_vec = embed_batch([pages[0]["markdown"][:100]])[0]
    dims = len(first_vec)

    index_client = SearchIndexClient(settings.search_endpoint, cred)
    # Reuse create_index (which calls build_index internally) but we need
    # include_context=False. create_index is a thin wrapper; call build_index
    # directly and use create_or_update_index.
    from ragpipe.search_index import build_index

    index_client.create_or_update_index(
        build_index(settings.baseline_index, dims, include_context=False)
    )

    # Build docs: embed_fn takes a single string; wrap embed_batch for this.
    def _embed_one(text: str) -> list[float]:
        return embed_batch([text])[0]

    docs = build_baseline_documents(pages, embed_fn=_embed_one)

    search_client = SearchClient(settings.search_endpoint, settings.baseline_index, cred)
    _upload_in_batches(search_client, docs)
    if limit is not None:
        print(f"Uploaded {len(docs)} chunks from {len(pages)} pages (limit={limit}, prune skipped).")
        return
    pruned = prune_stale_documents(search_client, fresh_ids={d["id"] for d in docs})
    print(
        f"Uploaded {len(docs)} chunks from {len(pages)} pages "
        f"to index '{settings.baseline_index}' (pruned {pruned} stale chunks)."
    )


def build_graph(settings: Any, limit: int | None = None) -> None:  # pragma: no cover
    """Ingest the GraphRAG indexes: entities, relationships, and communities.

    Steps:
    1. Fetch pages; chunk into SAC leaf docs (same decoration as main).
    2. For each leaf doc, call an LLM to extract entities + relationships in the
       delimited format parse_extraction expects.
    3. Merge entities, detect communities (Louvain), produce Community objects
       with an LLM-written title + summary per community.
    4. Embed and upload to three separate Azure AI Search indexes.

    Community reports: LLM-generated (title + summary) per community -- NOT a
    shortcut. A separate chat completion writes a short report over the member
    entities' descriptions for each community group.
    """
    import sys

    import yaml
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.graphrag import (
        Community,
        community_documents,
        detect_communities,
        entity_documents,
        merge_entities,
        parse_extraction,
        relationship_documents,
    )
    from ragpipe.search_index import (
        build_communities_index,
        build_entities_index,
        build_relationships_index,
    )

    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]
    if limit is not None:
        urls = urls[:limit]

    cred = DefaultAzureCredential()
    embed_batch = build_batch_embed_fn(settings)
    complete_fn = build_context_complete_fn(settings)
    context_gen = ContextGenerator(complete_fn, model=settings.foundry_chat_model)

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    # 1. Build SAC leaf docs for chunk text + ids + urls.
    leaf_docs = build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)

    EXTRACT_PROMPT = (
        "Extract entities and relationships from the following text. "
        "Output records separated by '##'. "
        "Entity format: (\"entity\"<|>NAME<|>type<|>description). "
        "Relationship format: (\"relationship\"<|>SOURCE<|>TARGET<|>description<|>weight). "
        "Use a numeric weight 1-10 for relationships. "
        "Output only the records, no preamble.\n\n{text}"
    )

    def extract_fn(chunk_text: str) -> str:
        prompt = EXTRACT_PROMPT.format(text=chunk_text)
        for attempt in range(settings.max_retries + 1):
            try:
                return complete_fn(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"extract_fn attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return ""

    # 2. Extract entities + relationships from each leaf doc.
    print(f"Extracting entities/relationships from {len(leaf_docs)} leaf docs…", flush=True)
    all_entities = []
    all_relationships = []
    for i, doc in enumerate(leaf_docs):
        raw = extract_fn(doc["content"])
        if raw:
            ents, rels = parse_extraction(raw, source_chunk_id=doc["id"], source_url=doc["url"])
            all_entities.extend(ents)
            all_relationships.extend(rels)
        if (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(leaf_docs)} docs ({len(all_entities)} entities so far)", flush=True)

    print(f"  raw: {len(all_entities)} entities, {len(all_relationships)} relationships", flush=True)

    # 3. Merge and detect communities.
    entities = merge_entities(all_entities)
    community_map = detect_communities([e.name for e in entities], all_relationships)
    print(f"  merged: {len(entities)} entities, {len(community_map)} entity->community mappings", flush=True)

    # Group entities by community id for report generation.
    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for e in entities:
        cid = community_map.get(e.name, -1)
        groups[cid].append(e)

    # Group relationships by community (both endpoints must be in community).
    rel_groups: dict[int, list] = defaultdict(list)
    for r in all_relationships:
        src_cid = community_map.get(r.source, -1)
        tgt_cid = community_map.get(r.target, -1)
        if src_cid == tgt_cid and src_cid != -1:
            rel_groups[src_cid].append(r)

    COMMUNITY_REPORT_PROMPT = (
        "You are a technical writer. Write a short report for a cluster of related entities. "
        "First line: a concise title (no label prefix). "
        "Second line: a 2-4 sentence summary of what this cluster covers and why it matters. "
        "Reply with only title and summary, separated by a newline.\n\n"
        "Entities:\n{entities}\n\nRelationships:\n{relationships}"
    )

    print(f"Generating community reports for {len(groups)} communities…", flush=True)
    communities = []
    for cid, members in sorted(groups.items()):
        ent_text = "\n".join(f"- {e.name} ({e.type}): {e.description[:120]}" for e in members[:20])
        rel_text = "\n".join(
            f"- {r.source} -> {r.target}: {r.description[:80]}"
            for r in rel_groups.get(cid, [])[:10]
        )
        prompt = COMMUNITY_REPORT_PROMPT.format(entities=ent_text, relationships=rel_text or "(none)")
        report = ""
        for attempt in range(settings.max_retries + 1):
            try:
                report = complete_fn(prompt).strip()
                break
            except Exception as exc:  # noqa: BLE001
                print(
                    f"community report cid={cid} attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        lines = report.splitlines()
        title = lines[0].strip() if lines else f"Community {cid}"
        summary = "\n".join(lines[1:]).strip() if len(lines) > 1 else "; ".join(e.description[:80] for e in members[:5])
        communities.append(Community(id=cid, level=0, title=title, summary=summary))

    print(f"  generated {len(communities)} community reports.", flush=True)

    # 4. Build docs for all three indexes.
    ent_docs = entity_documents(entities, community=community_map, embed_batch_fn=embed_batch)
    rel_docs = relationship_documents(all_relationships, embed_batch_fn=embed_batch)
    com_docs = community_documents(communities, embed_batch_fn=embed_batch)

    # 5. Create the three indexes.
    dims = len(ent_docs[0]["description_vector"])
    index_client = SearchIndexClient(settings.search_endpoint, cred)
    index_client.create_or_update_index(build_entities_index(settings.graph_entities_index, dims))
    index_client.create_or_update_index(build_relationships_index(settings.graph_relationships_index, dims))
    index_client.create_or_update_index(build_communities_index(settings.graph_communities_index, dims))

    # 6. Upload each doc set.
    ent_client = SearchClient(settings.search_endpoint, settings.graph_entities_index, cred)
    rel_client = SearchClient(settings.search_endpoint, settings.graph_relationships_index, cred)
    com_client = SearchClient(settings.search_endpoint, settings.graph_communities_index, cred)

    print(f"Uploading {len(ent_docs)} entity docs to '{settings.graph_entities_index}'…", flush=True)
    _upload_in_batches(ent_client, ent_docs)
    print(f"Uploading {len(rel_docs)} relationship docs to '{settings.graph_relationships_index}'…", flush=True)
    _upload_in_batches(rel_client, rel_docs)
    print(f"Uploading {len(com_docs)} community docs to '{settings.graph_communities_index}'…", flush=True)
    _upload_in_batches(com_client, com_docs)

    print(
        f"build_graph done: {len(entities)} entities, {len(all_relationships)} relationships, "
        f"{len(communities)} communities → indexes "
        f"'{settings.graph_entities_index}', '{settings.graph_relationships_index}', "
        f"'{settings.graph_communities_index}'.",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
