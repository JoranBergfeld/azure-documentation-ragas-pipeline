"""Claude-authored synthetic test-item candidates (ADR-0010).

Family separation: the question author (Claude) is a different model family
from the generator (gpt) and the embeddings (OpenAI), so the testset is not
phrased in the system-under-test's own idiom. Gold URLs are provenance — the
caller names the page — never recovered by substring matching. Candidates that
lexically echo the source document are screened out: verbatim questions are
trivially easy for BM25/embeddings and would understate exactly the
vocabulary-mismatch failures the hard subsets exist to measure.
"""
from __future__ import annotations

import json
import re
from typing import Callable

CANDIDATE_PROMPT = """You are writing evaluation questions for a documentation \
search system that indexes Microsoft Learn.

The document below is the page {url}.

<document>
{document}
</document>

Write {n} question-and-answer pairs about this document.
Rules:
- Phrase each question as a real user who has NOT read this page: everyday \
wording, no reuse of the page's phrasing, headings, or distinctive terms.
- Each answer must be fully supported by the document alone.
- Return ONLY a JSON array: [{{"question": "...", "ground_truth": "..."}}, ...]"""

# Small closed-class list; enough to keep overlap about content words.
_STOPWORDS = frozenset(
    "the a an and or but if then else when what which how why where who whom "
    "this that these those is are was were be been being have has had do does "
    "did can could should would may might must will shall with without within "
    "into onto from for of to in on at by as it its they them their there "
    "here you your our we us not no nor so than too very just about over "
    "under again further once more most other some such only own same".split()
)


def content_word_overlap(question: str, document: str) -> float:
    """Fraction of the question's content words that appear in the document.

    1.0 = every content word is lifted from the page (verbatim echo);
    low values = user phrasing. Words shorter than 4 chars are ignored.
    """
    words = {
        w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOPWORDS
    }
    if not words:
        return 1.0
    doc = document.lower()
    return sum(1 for w in words if w in doc) / len(words)


def parse_candidates(raw: str) -> list[dict]:
    """Extract the JSON array from a model response (tolerates code fences)."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except ValueError:
        return []
    return [
        {"question": d["question"], "ground_truth": d["ground_truth"]}
        for d in data
        if isinstance(d, dict) and d.get("question") and d.get("ground_truth")
    ]


def make_candidates(
    complete_fn: Callable[[str], str],
    url: str,
    document: str,
    n: int = 5,
    max_overlap: float = 0.6,
) -> list[dict]:
    """Screened candidate rows for one page, gold URL stamped by provenance."""
    raw = complete_fn(CANDIDATE_PROMPT.format(url=url, document=document[:30000], n=n))
    rows = []
    for cand in parse_candidates(raw):
        if content_word_overlap(cand["question"], document) > max_overlap:
            continue  # lexical echo — exactly what we must not reward
        rows.append(
            {
                "question": cand["question"],
                "ground_truth": cand["ground_truth"],
                "ground_truth_context": url,
                "tags": ["synthetic"],
            }
        )
    return rows


def page_text_from_index(search_client, url: str) -> str:  # pragma: no cover - live Azure
    """Reassemble a page's text from its indexed chunks (clean content only)."""
    safe = url.replace("'", "''")
    results = search_client.search(
        search_text="*",
        filter=f"url eq '{safe}'",
        select=["content", "chunk_id"],
        top=200,
    )
    chunks = sorted(results, key=lambda d: d.get("chunk_id", 0))
    return "\n\n".join(c["content"] for c in chunks)
