from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ragpipe.config import TestsetMode


@dataclass(frozen=True)
class TestItem:
    question: str
    ground_truth: str
    ground_truth_context: str
    # Optional difficulty/category tags (ADR-0006): 'original', 'paraphrase',
    # 'lookalike', 'synthetic'. Empty means 'original'.
    tags: tuple[str, ...] = ()


def _load_jsonl(path: str) -> list[TestItem]:
    items: list[TestItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                TestItem(
                    question=row["question"],
                    ground_truth=row["ground_truth"],
                    ground_truth_context=row["ground_truth_context"],
                    tags=tuple(row.get("tags", ())),
                )
            )
    return items


def load_testset(
    mode: TestsetMode,
    handauthored_path: str = "data/testset.jsonl",
    synthetic_fn: Callable[[], list[TestItem]] | None = None,
) -> list[TestItem]:
    if mode is TestsetMode.HANDAUTHORED:
        return _load_jsonl(handauthored_path)
    if synthetic_fn is None:
        raise ValueError("synthetic mode requires a synthetic_fn generator")
    return synthetic_fn()


def rows_to_items(rows: list[dict], docs: list[dict]) -> list[TestItem]:
    """Map RAGAS testset rows to TestItems with provenance gold URLs.

    The gold label MUST be a page URL (ADR-0002) — the harness passes it to
    URL-match metrics, so chunk text here silently scores hit_rate/mrr = 0.
    RAGAS only returns the source chunk text; recover the URL by matching the
    chunk back to the seed docs. Unrecoverable rows are dropped: a wrong gold
    label is worse than a smaller testset.
    """
    items: list[TestItem] = []
    for row in rows:
        probe = (row.get("reference_contexts") or [""])[0][:200]
        url = next((d["url"] for d in docs if probe and probe in d["content"]), "")
        if not url or not row.get("reference"):
            continue
        items.append(
            TestItem(
                question=row["user_input"],
                ground_truth=row["reference"],
                ground_truth_context=url,
                tags=("synthetic",),
            )
        )
    return items


def build_synthetic_generator(
    settings, corpus_docs, testset_size: int = 15
):  # pragma: no cover - live Azure call
    """Return a synthetic_fn that builds a test set from corpus docs via RAGAS.

    `corpus_docs` is a list of {"content": str, "url": str}. Uses the same
    OpenAI-route + Entra auth as the rest of the eval (the embedding model is an
    Azure OpenAI deployment, served on /openai — not FoundryEmbeddingClient).
    """
    def synthetic_fn() -> list[TestItem]:
        from ragpipe.eval.harness import _build_ragas_clients
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from langchain_core.documents import Document
        from ragas.testset import TestsetGenerator

        llm, emb = _build_ragas_clients(settings)
        docs = [
            Document(page_content=d["content"], metadata={"url": d.get("url", "")})
            for d in corpus_docs
        ]
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        return rows_to_items(dataset.to_list(), corpus_docs)

    return synthetic_fn
