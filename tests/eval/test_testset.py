import json

import pytest
import yaml

from ragpipe.config import TestsetMode
from ragpipe.eval.retrieval_metrics import gold_set, normalize_url
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


def _corpus_urls() -> set[str]:
    with open("data/corpus_sources.yaml") as f:
        return {normalize_url(u) for u in yaml.safe_load(f)["sources"]}


def test_every_testset_url_is_in_the_corpus():
    """hit_rate/mrr are meaningless if a gold URL was never ingested. Items may
    carry one gold URL or several (multi-hop / global, ADR-0019) -- check each."""
    corpus = _corpus_urls()
    items = _load_jsonl("data/testset.jsonl")
    missing = sorted(
        {u for it in items for u in gold_set(it.ground_truth_context) if u not in corpus}
    )
    assert not missing, f"testset gold URLs not in data/corpus_sources.yaml: {missing}"


def test_load_testset_parses_list_ground_truth_context(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        json.dumps(
            {
                "question": "q",
                "ground_truth": "a",
                "ground_truth_context": ["http://u1", "http://u2"],
                "tags": ["multihop"],
            }
        )
    )
    items = load_testset(TestsetMode.HANDAUTHORED, handauthored_path=str(p))
    # A list gold becomes a hashable tuple; a bare string stays a string.
    assert items[0].ground_truth_context == ("http://u1", "http://u2")
    assert items[0].tags == ("multihop",)


def test_testset_has_synthesis_cohorts():
    """The multi-hop and global/sensemaking cohorts (ADR-0019) exist and each of
    their items carries a list of >=2 gold URLs."""
    items = _load_jsonl("data/testset.jsonl")
    tags = [t for it in items for t in it.tags]
    assert tags.count("multihop") >= 5
    assert tags.count("global") >= 5
    for it in items:
        if {"multihop", "global"} & set(it.tags):
            assert isinstance(it.ground_truth_context, tuple)
            assert len(it.ground_truth_context) >= 2


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
