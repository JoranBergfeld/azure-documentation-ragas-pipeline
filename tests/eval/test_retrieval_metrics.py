from ragpipe.eval.retrieval_metrics import (
    gold_set,
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
# --- multi-gold / global support (ADR-0019) ---


def test_gold_set_normalizes_dedupes_and_drops_blanks():
    assert gold_set("https://H.com/EN-US/x/") == {"https://h.com/x"}
    assert gold_set(["", "   "]) == set()
    assert gold_set(["https://h.com/a", "https://h.com/a"]) == {"https://h.com/a"}
    assert gold_set("") == set()


def test_hit_rate_is_recall_over_a_multi_url_gold_set():
    gold = ["a", "b", "c"]
    assert hit_rate(["a", "b", "x"], gold) == 2 / 3
    assert hit_rate(["a", "b", "c"], gold) == 1.0
    assert hit_rate(["x"], gold) == 0.0


def test_mrr_uses_first_chunk_matching_any_gold_url():
    assert mrr(["x", "b", "a"], ["a", "b"]) == 0.5  # b matches at rank 2
    assert mrr(["a", "b"], ["a", "b"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_stage_metrics_skip_entirely_when_no_gold_url():
    # Gold-less (e.g. open sensemaking) items get no deterministic keys at all;
    # they are scored by RAGAS only (ADR-0016).
    assert stage_retrieval_metrics({"reranked": ["x"]}, "") == {}
    assert stage_retrieval_metrics({"reranked": ["x"]}, []) == {}


def test_stage_metrics_normalize_both_sides_with_multi_gold():
    stage_urls = {
        "reranked": [
            "https://learn.microsoft.com/en-us/azure/a",
            "https://learn.microsoft.com/azure/b",
        ]
    }
    gold = [
        "https://learn.microsoft.com/azure/a",
        "https://learn.microsoft.com/en-us/azure/c",
    ]
    m = stage_retrieval_metrics(stage_urls, gold)
    assert m["hit_rate@reranked"] == 0.5  # a retrieved, c missing
    assert m["mrr@reranked"] == 1.0  # a is at rank 1 after locale-normalization
