import json

import pytest
import yaml

from ragpipe.config import TestsetMode
from ragpipe.eval.retrieval_metrics import normalize_url
from ragpipe.eval.testset import TestItem, _load_jsonl, load_testset


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
