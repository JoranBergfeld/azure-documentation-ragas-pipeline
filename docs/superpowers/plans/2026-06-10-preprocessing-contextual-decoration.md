# Preprocessing: Contextual Chunk Decoration + Deterministic Retrieval Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic retrieval metrics + a harder tagged test set, capture a baseline, then ship structure-preserving extraction, heading-aware chunking, and per-chunk contextual decoration — and measure the difference.

**Architecture:** Phase 1 (Tasks 1–6) lands measurement only — no pipeline changes — and commits a baseline (`eval_baseline.json`). Phase 2 (Tasks 7–14) lands extraction→chunking→decoration→index changes, re-ingests, and re-measures. Order is mandatory (ADR-0006). Decoration lives in a separate `context` index field visible to retrieval but never to the generator or faithfulness judge (ADR-0003).

**Tech Stack:** Python 3.11, uv, BeautifulSoup + markdownify, Azure AI Search SDK, Azure OpenAI (gpt-4o via Entra), RAGAS, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md` · **ADRs:** `docs/adr/0001`–`0007`

**Conventions for all tasks:** run tests with `uv run pytest <path> -q`. Run `uv run ruff check src tests` before each commit. Tasks marked **LIVE** need the Azure `.env` and are executed by the orchestrator directly, not a subagent.

---

## Phase 1 — Measurement first

### Task 1: Deterministic retrieval metrics module

**Files:**
- Create: `src/ragpipe/eval/retrieval_metrics.py`
- Test: `tests/test_retrieval_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval_metrics.py
from ragpipe.eval.retrieval_metrics import (
    hit_rate,
    mrr,
    normalize_url,
    stage_retrieval_metrics,
)


def test_normalize_url_strips_locale_query_fragment_and_case():
    assert (
        normalize_url("https://learn.microsoft.com/en-US/azure/search/overview/?x=1#y")
        == "https://learn.microsoft.com/azure/search/overview"
    )


def test_normalize_url_leaves_locale_free_urls_alone():
    url = "https://learn.microsoft.com/azure/go-to/concepts"
    assert normalize_url(url) == url


def test_hit_rate_one_when_url_present():
    assert hit_rate(["http://a", "http://b"], "http://b") == 1.0
    assert hit_rate(["http://a"], "http://b") == 0.0
    assert hit_rate([], "http://b") == 0.0


def test_mrr_is_reciprocal_rank_of_first_match():
    assert mrr(["http://a", "http://b", "http://b"], "http://b") == 0.5
    assert mrr(["http://b"], "http://b") == 1.0
    assert mrr(["http://a"], "http://b") == 0.0


def test_stage_retrieval_metrics_keys_and_normalization():
    stage_urls = {
        "dense": ["https://learn.microsoft.com/en-us/azure/x"],
        "bm25": ["https://learn.microsoft.com/azure/y"],
    }
    got = stage_retrieval_metrics(stage_urls, "https://learn.microsoft.com/azure/x")
    assert got == {
        "hit_rate@dense": 1.0,
        "mrr@dense": 1.0,
        "hit_rate@bm25": 0.0,
        "mrr@bm25": 0.0,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragpipe.eval.retrieval_metrics'`

- [ ] **Step 3: Implement the module**

```python
# src/ragpipe/eval/retrieval_metrics.py
"""Deterministic, LLM-free retrieval metrics (ADR-0002).

Judged by URL match: every test item carries a ground-truth source URL and every
indexed chunk carries its page URL. Exact, reproducible, and free, unlike the
LLM-judged context metrics (which remain as a complement).
"""
from __future__ import annotations

import re

# Strip a learn.microsoft.com-style locale segment directly after the host
# (e.g. /en-us/). Only the first path segment is considered, so service names
# that happen to look locale-ish deeper in the path are untouched.
_LOCALE_RE = re.compile(r"^(https?://[^/]+)/[a-z]{2}-[a-z]{2}(/|$)")


def normalize_url(url: str) -> str:
    """Canonical form for URL equality: lowercase, no locale segment, no
    query/fragment, no trailing slash."""
    url = url.strip().lower()
    url = url.split("#", 1)[0].split("?", 1)[0]
    url = _LOCALE_RE.sub(r"\1\2", url, count=1)
    return url.rstrip("/")


def hit_rate(urls: list[str], ground_truth_url: str) -> float:
    """1.0 if the ground-truth URL appears anywhere in the list, else 0.0."""
    return 1.0 if ground_truth_url in urls else 0.0


def mrr(urls: list[str], ground_truth_url: str) -> float:
    """Reciprocal rank (1-based) of the first matching URL; 0.0 if absent."""
    for i, url in enumerate(urls):
        if url == ground_truth_url:
            return 1.0 / (i + 1)
    return 0.0


def stage_retrieval_metrics(
    stage_urls: dict[str, list[str]], ground_truth_url: str
) -> dict[str, float]:
    """hit_rate/mrr per stage, keyed 'hit_rate@<stage>' / 'mrr@<stage>'."""
    gt = normalize_url(ground_truth_url)
    out: dict[str, float] = {}
    for stage, urls in stage_urls.items():
        normalized = [normalize_url(u) for u in urls]
        out[f"hit_rate@{stage}"] = hit_rate(normalized, gt)
        out[f"mrr@{stage}"] = mrr(normalized, gt)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_metrics.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/retrieval_metrics.py tests/test_retrieval_metrics.py
git commit -m "feat(eval): deterministic URL-match retrieval metrics (ADR-0002)"
```

---

### Task 2: TestItem tags

**Files:**
- Modify: `src/ragpipe/eval/testset.py`
- Test: create `tests/test_testset.py` (no dedicated testset test file exists yet)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_testset.py
import json

from ragpipe.config import TestsetMode
from ragpipe.eval.testset import TestItem, load_testset


def test_load_testset_parses_tags(tmp_path):
    p = tmp_path / "ts.jsonl"
    rows = [
        {"question": "q1", "ground_truth": "a1", "ground_truth_context": "http://u1",
         "tags": ["paraphrase"]},
        {"question": "q2", "ground_truth": "a2", "ground_truth_context": "http://u2"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    items = load_testset(TestsetMode.HANDAUTHORED, handauthored_path=str(p))
    assert items[0].tags == ("paraphrase",)
    assert items[1].tags == ()  # absent tags -> empty tuple (treated as 'original')


def test_testitem_tags_default_empty():
    item = TestItem(question="q", ground_truth="a", ground_truth_context="u")
    assert item.tags == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_testset.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tags'` / attribute error

- [ ] **Step 3: Implement**

In `src/ragpipe/eval/testset.py`, change `TestItem` and `_load_jsonl`:

```python
@dataclass(frozen=True)
class TestItem:
    question: str
    ground_truth: str
    ground_truth_context: str
    # Optional difficulty/category tags (ADR-0006): 'original', 'paraphrase',
    # 'lookalike', 'synthetic'. Empty means 'original'.
    tags: tuple[str, ...] = ()
```

and in `_load_jsonl`, build items as:

```python
            items.append(
                TestItem(
                    question=row["question"],
                    ground_truth=row["ground_truth"],
                    ground_truth_context=row["ground_truth_context"],
                    tags=tuple(row.get("tags", ())),
                )
            )
```

- [ ] **Step 4: Run the full suite (loader is shared)**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/eval/testset.py tests/test_testset.py
git commit -m "feat(eval): optional tags on test items (ADR-0006)"
```

---

### Task 3: Wire deterministic metrics + per-tag aggregation into the harness

**Files:**
- Modify: `src/ragpipe/eval/harness.py` (EvalRecord, run_harness, new aggregate_by_tag)
- Modify: `src/ragpipe/eval/run.py` (payload)
- Test: create `tests/test_harness_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_harness_metrics.py
import asyncio

from ragpipe.eval.harness import EvalRecord, aggregate, aggregate_by_tag, run_harness
from ragpipe.eval.testset import TestItem
from ragpipe.models import Chunk, PipelineState


def _chunk(cid: str, url: str) -> Chunk:
    return Chunk(id=cid, title=cid, url=url, content=f"content {cid}")


def _state(query: str) -> PipelineState:
    s = PipelineState(query=query)
    s.dense = [_chunk("d1", "https://learn.microsoft.com/en-us/azure/right")]
    s.bm25 = [_chunk("b1", "https://learn.microsoft.com/azure/wrong")]
    s.fused = s.dense + s.bm25
    s.reranked = [s.bm25[0], s.dense[0]]  # right page at rank 2
    s.answer = "an answer"
    return s


def test_run_harness_computes_deterministic_metrics_without_evaluator():
    items = [
        TestItem(
            question="q",
            ground_truth="a",
            ground_truth_context="https://learn.microsoft.com/azure/right",
            tags=("lookalike",),
        )
    ]

    async def pipeline_fn(q):
        return _state(q)

    async def evaluator_fn(records):
        return records  # no LLM metrics

    records = asyncio.run(run_harness(items, pipeline_fn, evaluator_fn))
    r = records[0]
    assert r.tags == ("lookalike",)
    assert r.metrics["hit_rate@dense"] == 1.0
    assert r.metrics["hit_rate@bm25"] == 0.0
    assert r.metrics["hit_rate@reranked"] == 1.0
    assert r.metrics["mrr@reranked"] == 0.5
    assert r.metrics["hit_rate@fused"] == 1.0


def test_aggregate_by_tag_groups_untagged_as_original():
    r1 = EvalRecord(question="q1", answer="a", contexts=[], ground_truth="g",
                    metrics={"m": 1.0}, tags=("paraphrase",))
    r2 = EvalRecord(question="q2", answer="a", contexts=[], ground_truth="g",
                    metrics={"m": 0.0})
    by_tag = aggregate_by_tag([r1, r2])
    assert by_tag["paraphrase"]["m"] == 1.0
    assert by_tag["original"]["m"] == 0.0


def test_aggregate_unchanged_for_plain_metrics():
    r = EvalRecord(question="q", answer="a", contexts=[], ground_truth="g",
                   metrics={"m": 0.5})
    assert aggregate([r]) == {"m": 0.5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness_metrics.py -q`
Expected: FAIL — `EvalRecord.__init__() got an unexpected keyword argument 'tags'` / ImportError for aggregate_by_tag

- [ ] **Step 3: Implement in `harness.py`**

Add to imports: `from ragpipe.eval.retrieval_metrics import stage_retrieval_metrics`.

Extend `EvalRecord` (add after `stage_contexts`):

```python
    # Per-stage page URLs captured during the run, for deterministic URL-match
    # metrics (ADR-0002).
    stage_urls: dict[str, list[str]] = field(default_factory=dict)
    # Test-item tags (ADR-0006); empty means 'original'.
    tags: tuple[str, ...] = ()
```

Replace the record construction inside `run_harness` with:

```python
    for item in items:
        state = await pipeline_fn(item.question)
        by_stage = {
            "dense": state.dense,
            "bm25": state.bm25,
            "fused": state.fused,
            "reranked": state.reranked,
        }
        record = EvalRecord(
            question=item.question,
            answer=state.answer,
            contexts=[c.content for c in state.reranked],
            ground_truth=item.ground_truth,
            stage_contexts={s: [c.content for c in cs] for s, cs in by_stage.items()},
            stage_urls={s: [c.url for c in cs] for s, cs in by_stage.items()},
            tags=item.tags,
        )
        # Deterministic metrics are free — always computed, no toggle.
        record.metrics.update(
            stage_retrieval_metrics(record.stage_urls, item.ground_truth_context)
        )
        records.append(record)
```

Add after `aggregate`:

```python
def aggregate_by_tag(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    """aggregate() per tag group; records without tags count as 'original'.

    A record with several tags contributes to each of its groups.
    """
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        for tag in r.tags or ("original",):
            groups.setdefault(tag, []).append(r)
    return {tag: aggregate(rs) for tag, rs in sorted(groups.items())}
```

- [ ] **Step 4: Wire into `run.py`**

In `main()`, after `means = aggregate(records)` add:

```python
    means_by_tag = aggregate_by_tag(records)
```

(import `aggregate_by_tag` alongside `aggregate`), add `"means_by_tag": means_by_tag,` to the payload dict (after `"means"`), and include it in the final print:

```python
    print(json.dumps({"means": means, "means_by_tag": means_by_tag, "coverage": cov}, indent=2))
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/eval/harness.py src/ragpipe/eval/run.py tests/test_harness_metrics.py
git commit -m "feat(eval): always-on per-stage hit_rate/mrr + per-tag aggregation"
```

---

### Task 4: Test-set expansion — hard items + corpus-membership guard

**Files:**
- Modify: `data/testset.jsonl`
- Test: append to `tests/test_testset.py`

- [ ] **Step 1: Write the failing corpus-membership test**

```python
# append to tests/test_testset.py
import yaml

from ragpipe.eval.retrieval_metrics import normalize_url
from ragpipe.eval.testset import _load_jsonl


def _corpus_urls() -> set[str]:
    with open("data/corpus_sources.yaml") as f:
        return {normalize_url(u) for u in yaml.safe_load(f)["sources"]}


def test_every_testset_url_is_in_the_corpus():
    """hit_rate/mrr are meaningless if the gold URL was never ingested."""
    corpus = _corpus_urls()
    items = _load_jsonl("data/testset.jsonl")
    missing = sorted(
        {
            it.ground_truth_context
            for it in items
            if normalize_url(it.ground_truth_context) not in corpus
        }
    )
    assert not missing, f"testset gold URLs not in data/corpus_sources.yaml: {missing}"


def test_testset_has_hard_subsets():
    items = _load_jsonl("data/testset.jsonl")
    tags = [t for it in items for t in it.tags]
    assert tags.count("paraphrase") >= 6
    assert tags.count("lookalike") >= 6
    assert len(items) >= 28
```

- [ ] **Step 2: Run to see what fails**

Run: `uv run pytest tests/test_testset.py -q`
Expected: `test_testset_has_hard_subsets` FAILS (only 16 untagged items). `test_every_testset_url_is_in_the_corpus` may also fail — that tells you which of the original 16 items reference pages outside the corpus.

- [ ] **Step 3: Repair original items + author the hard items**

1. Tag all 16 existing rows with `"tags": ["original"]`.
2. For each original item flagged by the membership test: find the in-corpus page (grep `data/corpus_sources.yaml` for the service) that actually contains the answer and update `ground_truth_context` to it. If no in-corpus page can answer the question, rewrite the question to one the corpus can answer (keep it on-topic for the same service).
3. Author **≥6 `paraphrase`** items: questions about in-corpus pages phrased with low lexical overlap (no shared keywords with the page title/URL). Example shape:

```json
{"question": "If my chatbot's search keeps surfacing text that merely shares words with the user's wording rather than meaning, which Azure capability re-orders results by intent?", "ground_truth": "Semantic ranker in Azure AI Search re-ranks results using language understanding so semantically relevant results rise to the top.", "ground_truth_context": "https://learn.microsoft.com/en-us/azure/search/semantic-search-overview", "tags": ["paraphrase"]}
```

4. Author **≥6 `lookalike`** items: the answer lives on one specific service page among several near-identical candidates in the corpus (e.g. consistency/scaling/pricing questions that exist for Cosmos DB *and* other data services; "what is X" across sibling services). The question must name the *scenario*, not the service's page title. Example shape:

```json
{"question": "Which Azure database offers five well-defined consistency choices ranging from strong to eventual for globally distributed reads?", "ground_truth": "Azure Cosmos DB offers five consistency levels: strong, bounded staleness, session, consistent prefix, and eventual.", "ground_truth_context": "https://learn.microsoft.com/en-us/azure/cosmos-db/consistency-levels", "tags": ["lookalike"]}
```

   Every `ground_truth_context` MUST be a URL present in `data/corpus_sources.yaml` (the membership test enforces this — pick pages from that file, do not invent URLs).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_testset.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add data/testset.jsonl tests/test_testset.py
git commit -m "feat(eval): tag testset, repair gold URLs, add paraphrase/lookalike hard items"
```

---

### Task 5 (LIVE): Synthetic test items, screened

**Files:**
- Create: `scripts/generate_synthetic_testset.py`
- Modify: `data/testset.jsonl`

- [ ] **Step 1: Write the generator script**

```python
# scripts/generate_synthetic_testset.py
"""Generate synthetic test-item CANDIDATES from the indexed corpus (spec §6).

Prints candidate rows as JSON to stdout for manual screening — nothing is
written to data/testset.jsonl by this script. Items whose source URL cannot be
recovered are dropped (URL-match metrics need a gold URL, ADR-0002).
"""
import json
import sys

from ragpipe.config import Settings
from ragpipe.eval.run import _sample_corpus_docs
from ragpipe.eval.testset import build_synthetic_generator


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    settings = Settings.from_env()
    docs = _sample_corpus_docs(settings)
    items = build_synthetic_generator(settings, docs, testset_size=n)()
    rows = []
    for it in items:
        probe = (it.ground_truth_context or "")[:200]
        url = next((d["url"] for d in docs if probe and probe in d["content"]), "")
        if not url or not it.ground_truth:
            continue
        rows.append(
            {
                "question": it.question,
                "ground_truth": it.ground_truth,
                "ground_truth_context": url,
                "tags": ["synthetic"],
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 (LIVE): Generate candidates**

Run: `uv run python scripts/generate_synthetic_testset.py 16`
Expected: JSON array printed (some candidates dropped for unrecoverable URLs is normal).

- [ ] **Step 3 (LIVE): Screen and append**

Screening criteria (orchestrator applies; spec §6): the question is answerable from its page, is not a near-duplicate of an existing item, is self-contained (no "according to the document…"), and the gold URL passes the corpus-membership test. Append the ~12 best as JSONL lines to `data/testset.jsonl`.

- [ ] **Step 4: Run the testset tests**

Run: `uv run pytest tests/test_testset.py -q`
Expected: all pass (membership test validates the appended URLs)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_synthetic_testset.py data/testset.jsonl
git commit -m "feat(eval): screened synthetic test items"
```

**Fallback:** if the live generator fails (RAGAS/Azure issues), skip synthetic items — the hard subsets from Task 4 satisfy the success-criteria measurement — and note the deviation in the final report.

---

### Task 6 (LIVE): Baseline run

**Files:**
- Create: `eval_baseline.json` (committed)

- [ ] **Step 1 (LIVE): Run the harness against the CURRENT index**

Run: `uv run python -m ragpipe.eval.run` (PER_STAGE_METRICS unset/false; the deterministic per-stage metrics are always on and free)
Expected: prints means incl. `hit_rate@*`/`mrr@*` and `means_by_tag`; writes `eval_results.json`. Hard-subset hit rates SHOULD be < 1.0 — that's the headroom.

- [ ] **Step 2: Freeze as baseline**

```bash
cp eval_results.json eval_baseline.json
git add -f eval_baseline.json
git commit -m "chore(eval): freeze pre-decoration baseline (ADR-0006)"
```

**Gate:** Phase 2 must not start until this commit exists. `data/corpus_sources.yaml` is frozen from here until the post-change run (treatment must be the only variable).

---

## Phase 2 — Treatment

### Task 7: Structure-preserving extraction

**Files:**
- Create: `src/ragpipe/extraction.py`
- Create: `tests/fixtures/mslearn_sample.html`
- Test: create `tests/test_extraction.py`; modify `tests/test_ingest.py` (html_to_text test removed in Task 11)
- Modify: `pyproject.toml` (markdownify)

- [ ] **Step 1: Add the dependency**

Run: `uv add markdownify`
Expected: `pyproject.toml` + `uv.lock` updated.

- [ ] **Step 2: Create the fixture** — `tests/fixtures/mslearn_sample.html`:

```html
<html><head><title>Consistency levels - Azure Cosmos DB | Microsoft Learn</title>
<script>telemetry();</script></head>
<body>
<nav>Skip to main content Documentation Learn Sign in</nav>
<main>
<h1>Consistency levels in Azure Cosmos DB</h1>
<p>Distributed databases trade off consistency, availability, and latency.</p>
<h2>Configure the default consistency level</h2>
<p>Use the CLI to set it:</p>
<pre><code>az cosmosdb update --name mydb --default-consistency-level Session</code></pre>
<h2>Consistency levels overview</h2>
<table><tr><th>Level</th><th>Staleness</th></tr>
<tr><td>Strong</td><td>None</td></tr>
<tr><td>Eventual</td><td>Unbounded</td></tr></table>
</main>
<footer>Previous Next Feedback</footer>
</body></html>
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_extraction.py
from pathlib import Path

from ragpipe.extraction import html_to_markdown

FIXTURE = Path(__file__).parent / "fixtures" / "mslearn_sample.html"


def test_extracts_main_content_only():
    md, used_main = html_to_markdown(FIXTURE.read_text())
    assert used_main is True
    assert "Skip to main content" not in md     # nav dropped
    assert "Feedback" not in md                 # footer dropped
    assert "telemetry" not in md                # script dropped


def test_preserves_structure():
    md, _ = html_to_markdown(FIXTURE.read_text())
    assert "# Consistency levels in Azure Cosmos DB" in md
    assert "## Configure the default consistency level" in md
    assert "az cosmosdb update" in md
    assert "```" in md                          # fenced code survives
    assert "|" in md and "Strong" in md         # table survives


def test_falls_back_to_whole_page_without_main():
    md, used_main = html_to_markdown("<html><body><p>plain body</p></body></html>")
    assert used_main is False
    assert "plain body" in md


def test_empty_input():
    md, used_main = html_to_markdown("")
    assert md == ""
    assert used_main is False
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragpipe.extraction'`

- [ ] **Step 5: Implement**

```python
# src/ragpipe/extraction.py
"""HTML → markdown extraction that preserves document structure (ADR-0004).

MS Learn pages carry their content in a <main> element; code fences, tables,
and the heading hierarchy must survive into chunking (the old whitespace-
flattening path destroyed exactly the content that answers most questions).
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as _markdownify

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def html_to_markdown(html: str) -> tuple[str, bool]:
    """Convert page HTML to markdown.

    Returns (markdown, used_main): used_main is False when no <main>/<article>
    container was found and the whole (noise-stripped) page was converted —
    callers count these fallbacks so a drift in MS Learn's layout is visible
    in ingest output (spec §8).
    """
    if not html.strip():
        return "", False
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    container = soup.find("main") or soup.find("article")
    used_main = container is not None
    target = container if used_main else soup
    md = _markdownify(str(target), heading_style="ATX")
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return _BLANK_RUN_RE.sub("\n\n", md).strip(), used_main
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction.py -q`
Expected: 4 passed. (If the code-fence assertion fails, markdownify may emit indented code blocks for `<pre><code>`; in that case pass `code_language=""` and check for 4-space indented `az cosmosdb` lines instead — adjust the *test* to assert `"az cosmosdb update" in md` only and drop the triple-backtick assertion. Keep behavior, not formatting, under test.)

- [ ] **Step 7: Commit**

```bash
git add src/ragpipe/extraction.py tests/test_extraction.py tests/fixtures/mslearn_sample.html pyproject.toml uv.lock
git commit -m "feat(ingest): structure-preserving HTML->markdown extraction (ADR-0004)"
```

---

### Task 8: Heading-aware chunking with breadcrumbs

**Files:**
- Modify: `src/ragpipe/chunking.py` (keep `chunk_text` as-is, add `chunk_markdown`)
- Test: append to `tests/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_chunking.py
from ragpipe.chunking import MarkdownChunk, chunk_markdown


MD = """# Consistency levels

Intro paragraph.

## Configure

Some text about configuring.

```
# this hash is code, not a heading
az cosmosdb update
```

## Overview

| Level | Staleness |
| --- | --- |
| Strong | None |
"""


def test_chunk_markdown_splits_on_headings_with_breadcrumbs():
    chunks = chunk_markdown(MD, page_title="Cosmos DB docs", max_chars=2000, overlap=200)
    crumbs = [c.breadcrumb for c in chunks]
    assert "Cosmos DB docs > Consistency levels" in crumbs
    assert "Cosmos DB docs > Consistency levels > Configure" in crumbs
    assert "Cosmos DB docs > Consistency levels > Overview" in crumbs


def test_chunk_markdown_ignores_headings_inside_code_fences():
    chunks = chunk_markdown(MD, page_title="T", max_chars=2000, overlap=200)
    configure = next(c for c in chunks if c.breadcrumb.endswith("Configure"))
    assert "az cosmosdb update" in configure.text
    assert not any("this hash is code" in c.breadcrumb for c in chunks)


def test_chunk_markdown_size_bounds_long_sections():
    md = "# Title\n\n" + ("word " * 1000)
    chunks = chunk_markdown(md, page_title="P", max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.breadcrumb == "P > Title" for c in chunks)


def test_chunk_markdown_empty():
    assert chunk_markdown("", page_title="P") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunking.py -q`
Expected: FAIL — ImportError (`MarkdownChunk`, `chunk_markdown`)

- [ ] **Step 3: Implement (append to `src/ragpipe/chunking.py`)**

```python
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownChunk:
    text: str
    breadcrumb: str  # "page title > h1 > h2 > h3" path of this chunk's section


def chunk_markdown(
    markdown: str, page_title: str, max_chars: int = 2000, overlap: int = 200
) -> list[MarkdownChunk]:
    """Split markdown into chunks on heading boundaries (ADR-0004).

    Sections longer than max_chars are size-bounded with chunk_text. Headings
    inside fenced code blocks are not section boundaries. Every chunk carries
    its heading-path breadcrumb, the deterministic decoration floor (ADR-0001).
    """
    heading_stack: dict[int, str] = {}
    sections: list[tuple[str, list[str]]] = []  # (breadcrumb, lines)

    def crumb() -> str:
        path = [page_title] + [heading_stack[lvl] for lvl in sorted(heading_stack)]
        return " > ".join(p for p in path if p)

    current: list[str] = []
    in_fence = False
    sections.append((crumb(), current))
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            heading_stack[level] = title
            for deeper in [lvl for lvl in heading_stack if lvl > level]:
                del heading_stack[deeper]
            current = [line]
            sections.append((crumb(), current))
        else:
            current.append(line)

    chunks: list[MarkdownChunk] = []
    for breadcrumb, lines in sections:
        for piece in chunk_text("\n".join(lines), max_chars=max_chars, overlap=overlap):
            chunks.append(MarkdownChunk(text=piece, breadcrumb=breadcrumb))
    return chunks
```

(Move the `import re` / `from dataclasses import dataclass` lines to the top of the file with the existing imports.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -q`
Expected: all pass (existing `chunk_text` tests too)

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/chunking.py tests/test_chunking.py
git commit -m "feat(ingest): heading-aware markdown chunking with breadcrumbs (ADR-0004)"
```

---

### Task 9: Context generator with cache + fallback

**Files:**
- Create: `src/ragpipe/context_gen.py`
- Test: create `tests/test_context_gen.py`
- Modify: `.gitignore` (add `.context_cache.json`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_context_gen.py
import json

import pytest

from ragpipe.context_gen import ContextGenerator


def _gen(tmp_path, fn, **kw):
    return ContextGenerator(fn, cache_path=tmp_path / "cache.json", **kw)


def test_generates_and_caches(tmp_path):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return "  situating context  "

    g = _gen(tmp_path, fake)
    assert g.generate("DOC", "CHUNK") == "situating context"
    assert g.generate("DOC", "CHUNK") == "situating context"  # cache hit
    assert len(calls) == 1
    assert "DOC" in calls[0] and "CHUNK" in calls[0]


def test_cache_persists_across_instances(tmp_path):
    g1 = _gen(tmp_path, lambda p: "ctx")
    g1.generate("D", "C")
    g2 = _gen(tmp_path, lambda p: pytest.fail("should hit cache"))
    assert g2.generate("D", "C") == "ctx"


def test_distinct_chunks_get_distinct_keys(tmp_path):
    g = _gen(tmp_path, lambda p: f"ctx-{len(p)}")
    assert g.generate("D", "C1") != g.generate("D", "C2 longer")


def test_fallback_after_retries(tmp_path):
    def boom(prompt):
        raise RuntimeError("429")

    g = _gen(tmp_path, boom, max_retries=2)
    assert g.generate("D", "C") == ""
    assert g.fallback_count == 1


def test_corrupt_cache_treated_as_empty(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    g = ContextGenerator(lambda p: "ctx", cache_path=path)
    assert g.generate("D", "C") == "ctx"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_gen.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragpipe.context_gen'`

- [ ] **Step 3: Implement**

```python
# src/ragpipe/context_gen.py
"""Per-chunk situating-context generation with deterministic controls (ADR-0001/0005).

One LLM call per (document, chunk) pair, content-address-cached so re-ingests
only pay for changed chunks. Failures fall back to "" — the caller then
decorates with the breadcrumb only, so ingest never blocks on the LLM.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

# Bump to deliberately invalidate every cached context (ADR-0005).
PROMPT_VERSION = "v1"

# Anthropic's published situating prompt (Introducing Contextual Retrieval, 2024).
SITUATE_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk. Answer only \
with the succinct context and nothing else."""


class ContextGenerator:
    """Wraps a `complete(prompt) -> str` callable with cache, retries, fallback."""

    def __init__(
        self,
        complete_fn: Callable[[str], str],
        cache_path: str | Path = ".context_cache.json",
        max_retries: int = 2,
    ) -> None:
        self._complete = complete_fn
        self._cache_path = Path(cache_path)
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        self.fallback_count = 0

    def _load_cache(self) -> dict[str, str]:
        try:
            return json.loads(self._cache_path.read_text())
        except (OSError, ValueError):
            return {}  # missing or corrupt -> empty, never an error (ADR-0005)

    def _save_cache(self) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._cache_path)

    @staticmethod
    def _key(document: str, chunk: str) -> str:
        payload = "\x00".join((PROMPT_VERSION, document, chunk))
        return hashlib.sha256(payload.encode()).hexdigest()

    def generate(self, document: str, chunk: str) -> str:
        """Situating context for `chunk`, or "" after retries are exhausted."""
        key = self._key(document, chunk)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        prompt = SITUATE_PROMPT.format(document=document, chunk=chunk)
        for _ in range(self._max_retries):
            try:
                context = self._complete(prompt).strip()
            except Exception:  # noqa: BLE001 - one bad chunk must not abort ingest
                continue
            with self._lock:
                self._cache[key] = context
                self._save_cache()
            return context
        with self._lock:
            self.fallback_count += 1
        return ""


def build_context_complete_fn(
    settings, api_version: str = "2024-10-21", timeout: float = 60.0, max_retries: int = 5
) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """`complete(prompt) -> str` over the account's /openai endpoint, temperature=0."""
    from ragpipe.embeddings import _build_client

    client = _build_client(settings, api_version, timeout, max_retries)

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=settings.foundry_chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
```

- [ ] **Step 4: Add cache file to `.gitignore`**

Append a line: `.context_cache.json` (and `.context_cache.tmp`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_gen.py -q`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/ragpipe/context_gen.py tests/test_context_gen.py .gitignore
git commit -m "feat(ingest): cached per-chunk context generation with fallback (ADR-0001/0005)"
```

---

### Task 10: Index schema — `context` field

**Files:**
- Modify: `src/ragpipe/search_index.py`
- Test: `tests/test_search_index.py`

- [ ] **Step 1: Write the failing tests (append to existing file)**

```python
# append to tests/test_search_index.py
from ragpipe.search_index import build_index


def test_index_has_searchable_context_field():
    index = build_index("idx", vector_dimensions=2)
    ctx = next(f for f in index.fields if f.name == "context")
    assert ctx.searchable is True


def test_semantic_config_includes_context_after_content():
    index = build_index("idx", vector_dimensions=2)
    config = index.semantic_search.configurations[0]
    content_fields = [f.field_name for f in config.prioritized_fields.content_fields]
    assert content_fields == ["content", "context"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search_index.py -q`
Expected: FAIL — StopIteration (no `context` field)

- [ ] **Step 3: Implement in `build_index`**

After the `content` SearchableField add:

```python
        # Retrieval-only decoration: breadcrumb + generated situating context
        # (ADR-0003). Searchable for BM25 and in the semantic config, but never
        # returned into generator prompts or the faithfulness judge.
        SearchableField(name="context", type=SearchFieldDataType.String),
```

and change the semantic configuration's content fields to:

```python
                    content_fields=[
                        SemanticField(field_name="content"),
                        SemanticField(field_name="context"),
                    ],
```

- [ ] **Step 4: Run tests + full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/search_index.py tests/test_search_index.py
git commit -m "feat(index): additive searchable context field in schema + semantic config (ADR-0003/0007)"
```

**Note:** retrieval code (`bm25.py`, `dense.py`, `rerank.py`) keeps `select=["id", "title", "url", "content"]` — the `context` field must NOT be selected into chunks; that is the ADR-0003 isolation. Do not change those files.

---

### Task 11: Rewire ingest — markdown pages, decorated documents

**Files:**
- Modify: `src/ragpipe/ingest.py`
- Test: modify `tests/test_ingest.py`

- [ ] **Step 1: Update the tests**

In `tests/test_ingest.py`:
1. DELETE `test_html_to_text_strips_tags_and_scripts` (function is being removed; extraction is covered by `tests/test_extraction.py`).
2. Update the two `build_documents` tests: pages now carry `"markdown"` instead of `"text"`, and `build_documents` takes a `context_fn`. Replace them with:

```python
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
```

(The prune tests are untouched.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -q`
Expected: FAIL — `build_documents() got an unexpected keyword argument 'context_fn'`

- [ ] **Step 3: Implement in `src/ragpipe/ingest.py`**

1. Remove `html_to_text` and the BeautifulSoup import if now unused; add imports:

```python
from ragpipe.chunking import chunk_markdown
from ragpipe.extraction import html_to_markdown
```

(`chunk_text` import becomes unused — remove it.)

2. In `fetch_pages.fetch_one`, replace the `text = html_to_text(resp.text)` block with:

```python
                markdown, used_main = html_to_markdown(resp.text)
                if not used_main:
                    skip_reasons["main-content fallback (kept)"] += 1
                if not markdown.strip():
                    return None
                return {"url": url, "title": title, "markdown": markdown}
```

(`skip_reasons` is already printed at the end — fallback pages are *kept* but counted, spec §8.)

3. Replace `build_documents` with:

```python
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
```

4. Update `main()`:

```python
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
    context_gen = ContextGenerator(build_context_complete_fn(settings))

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
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/ragpipe/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): markdown pages, breadcrumb+context decoration, embed decorated text"
```

---

### Task 12 (LIVE): Smoke ingest — 3 pages

- [ ] **Step 1 (LIVE):** `uv run python -m ragpipe.ingest 3`
Expected: fetch 3 pages, context generation runs (first run pays LLM calls), upload succeeds, "prune skipped" printed.

- [ ] **Step 2 (LIVE): Verify decoration in the index** — run a quick check that a fresh chunk has a populated `context` field and clean `content`:

```bash
uv run python - <<'EOF'
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from ragpipe.config import Settings

s = Settings.from_env()
client = SearchClient(s.search_endpoint, s.search_index, DefaultAzureCredential())
doc = next(iter(client.search(search_text="*", top=1, select=["content", "context"], order_by=None)))
assert doc.get("context"), "context field empty"
assert doc["context"] not in doc["content"], "content contaminated with decoration"
print("context sample:", doc["context"][:200])
EOF
```

Expected: a breadcrumb + situating sentence printed. (Search returns arbitrary order; if the sampled doc is an old undecorated chunk, query for one of the 3 smoke URLs instead via `filter="url eq '<smoke url>'"`.)

- [ ] **Step 3 (LIVE):** Re-run `uv run python -m ragpipe.ingest 3` — expected: near-instant context phase (all cache hits), confirming ADR-0005 cache behavior.

---

### Task 13 (LIVE): Full re-ingest

- [ ] **Step 1 (LIVE):** `uv run python -m ragpipe.ingest` (no limit). ~2,700 context calls on first run; expect 30–90 min. Watch for a high `main-content fallback` or `context fallbacks` count — investigate before proceeding if either exceeds ~5% of pages/chunks.
- [ ] **Step 2 (LIVE):** Confirm the final line reports pruning of the entire old chunk set (old ids all replaced — expect pruned ≈ old index count) and the new total ≈ chunk count.

---

### Task 14 (LIVE): Post-change eval + comparison

- [ ] **Step 1 (LIVE):** `uv run python -m ragpipe.eval.run`
- [ ] **Step 2:** `cp eval_results.json eval_post_decoration.json && git add -f eval_post_decoration.json`
- [ ] **Step 3: Compare against `eval_baseline.json`** (spec §7 success criteria, pre-registered in ADR-0006):
  - `hit_rate@*`/`mrr@*` improve on `paraphrase` and `lookalike` tag groups,
  - no regression on `original`,
  - `faithfulness` mean does not degrade.

  Write the comparison table (per tag group, baseline vs post) into the final report to the user. If hard-subset gains are absent, say so plainly and flag ADR-0001's revisit clause (SAC fallback) — do not spin the numbers.
- [ ] **Step 4: Commit**

```bash
git commit -m "chore(eval): post-decoration eval results vs baseline (ADR-0006)"
```

---

## Out of scope (later specs, from the 2026-06-10 review)

Hybrid-rerank candidate drop (I1), refusal/judge-error routing (D3/I5), judge-model independence (D1), candidate-pool widening (D4), retry-loop feedback (D2).
