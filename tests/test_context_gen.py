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
