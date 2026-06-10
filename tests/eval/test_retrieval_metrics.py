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
