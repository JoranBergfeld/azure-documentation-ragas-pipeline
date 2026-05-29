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


def build_synthetic_generator(settings, corpus_docs):  # pragma: no cover
    """Return a synthetic_fn that builds a test set from corpus docs via RAGAS."""
    def synthetic_fn() -> list[TestItem]:
        from ragpipe.guardrail import _ensure_ragas_importable

        _ensure_ragas_importable()

        from langchain_core.documents import Document
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.testset import TestsetGenerator

        llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_chat_model,
                api_version="2024-10-21",
            )
        )
        emb = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                azure_endpoint=settings.foundry_project_endpoint,
                azure_deployment=settings.foundry_embedding_model,
                api_version="2024-10-21",
            )
        )
        docs = [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in corpus_docs
        ]
        generator = TestsetGenerator(llm=llm, embedding_model=emb)
        dataset = generator.generate_with_langchain_docs(docs, testset_size=15)
        items: list[TestItem] = []
        for row in dataset.to_list():
            items.append(
                TestItem(
                    question=row["user_input"],
                    ground_truth=row.get("reference", ""),
                    ground_truth_context=(row.get("reference_contexts") or [""])[0],
                )
            )
        return items

    return synthetic_fn
