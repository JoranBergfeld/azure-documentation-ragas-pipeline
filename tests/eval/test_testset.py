import json

import pytest
import yaml

from ragpipe.config import TestsetMode
from ragpipe.eval.retrieval_metrics import normalize_url
from ragpipe.eval.testset import TestItem, _load_jsonl, load_testset, rows_to_items


def test_load_handauthored_reads_jsonl(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        json.dumps(
            {"question": "q1", "ground_truth": "a1", "ground_truth_context": "c1"}
        )
        + "\n"
    )

    items = load_testset(TestsetMode.HANDAUTHORED, handauthored_path=str(p))

    assert items == [TestItem(question="q1", ground_truth="a1", ground_truth_context="c1")]


def test_load_synthetic_calls_generator(tmp_path):
    sentinel = [TestItem(question="gen-q", ground_truth="gen-a", ground_truth_context="gen-c")]

    items = load_testset(
        TestsetMode.SYNTHETIC,
        handauthored_path="unused",
        synthetic_fn=lambda: sentinel,
    )

    assert items == sentinel


def test_load_synthetic_without_generator_raises():
    with pytest.raises(ValueError, match="synthetic"):
        load_testset(TestsetMode.SYNTHETIC, handauthored_path="x", synthetic_fn=None)


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


def test_gold_urls_prefers_multi_url_gold_then_single_context_then_empty():
    multi = TestItem(
        question="q",
        ground_truth="g",
        ground_truth_context="http://single",
        ground_truth_urls=("http://a", "http://b"),
    )
    single = TestItem(question="q", ground_truth="g", ground_truth_context="http://single")
    global_item = TestItem(question="q", ground_truth="g")

    assert multi.gold_urls() == ("http://a", "http://b")
    assert single.gold_urls() == ("http://single",)
    assert global_item.gold_urls() == ()


def test_load_testset_parses_multi_url_gold_and_missing_gold(tmp_path):
    p = tmp_path / "ts.jsonl"
    rows = [
        {
            "question": "q1",
            "ground_truth": "a1",
            "ground_truth_urls": ["http://u1", "http://u2"],
            "tags": ["multihop"],
        },
        {"question": "q2", "ground_truth": "a2", "tags": ["global"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))

    items = load_testset(TestsetMode.HANDAUTHORED, handauthored_path=str(p))

    assert items[0].ground_truth_context == ""
    assert items[0].ground_truth_urls == ("http://u1", "http://u2")
    assert items[0].gold_urls() == ("http://u1", "http://u2")
    assert items[1].ground_truth_context == ""
    assert items[1].ground_truth_urls == ()
    assert items[1].gold_urls() == ()


def _corpus_urls() -> set[str]:
    with open("data/corpus_sources.yaml") as f:
        return {normalize_url(u) for u in yaml.safe_load(f)["sources"]}


def test_every_testset_url_is_in_the_corpus():
    """hit_rate/mrr are meaningless if the gold URL was never ingested."""
    corpus = _corpus_urls()
    items = _load_jsonl("data/testset.jsonl")
    missing = sorted(
        {
            url
            for it in items
            for url in it.gold_urls()
            if normalize_url(url) not in corpus
        }
    )
    assert not missing, f"testset gold URLs not in data/corpus_sources.yaml: {missing}"


def test_handauthored_testset_loads_legacy_multihop_and_global_gold_shapes():
    items = load_testset(TestsetMode.HANDAUTHORED)

    global_items = [item for item in items if "global" in item.tags]
    multihop_items = [item for item in items if "multihop" in item.tags]
    legacy_items = [
        item for item in items if "global" not in item.tags and "multihop" not in item.tags
    ]

    assert len(items) == 44
    assert all(item.gold_urls() == () for item in global_items)
    assert all(len(item.gold_urls()) >= 2 for item in multihop_items)
    assert len(legacy_items) == 33
    assert all(len(item.gold_urls()) == 1 for item in legacy_items)


def test_testset_has_hard_subsets():
    items = _load_jsonl("data/testset.jsonl")
    tags = [t for it in items for t in it.tags]
    assert tags.count("paraphrase") >= 6
    assert tags.count("lookalike") >= 6
    assert len(items) >= 28


def test_rows_to_items_recovers_provenance_url_and_tags():
    docs = [
        {"content": "Azure AI Search supports hybrid retrieval over indexes.", "url": "http://learn/a"},
        {"content": "Cosmos DB bulk executor moves documents fast.", "url": "http://learn/b"},
    ]
    rows = [
        {
            "user_input": "How fast is bulk import?",
            "reference": "Fast.",
            "reference_contexts": ["Cosmos DB bulk executor moves documents fast."],
        }
    ]
    items = rows_to_items(rows, docs)
    assert len(items) == 1
    assert items[0].ground_truth_context == "http://learn/b"  # a URL, never chunk text
    assert items[0].tags == ("synthetic",)


def test_rows_to_items_drops_unrecoverable_rows():
    rows = [
        {"user_input": "q", "reference": "a", "reference_contexts": ["no such text"]},
        {"user_input": "q2", "reference": "", "reference_contexts": ["irrelevant"]},
    ]
    assert rows_to_items(rows, docs=[{"content": "other", "url": "http://x"}]) == []
