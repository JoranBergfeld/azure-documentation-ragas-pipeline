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


def test_cache_key_includes_model(tmp_path):
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return "ctx"

    g1 = _gen(tmp_path, complete, model="gpt-4o")
    g1.generate("doc", "chunk")
    g2 = _gen(tmp_path, complete, model="gpt-5.4")
    g2.generate("doc", "chunk")
    # model change must miss the cache and re-call the LLM
    assert len(calls) == 2


def test_generation_failure_is_logged(tmp_path, capsys):
    def boom(prompt):
        raise RuntimeError("model rejected temperature")

    g = _gen(tmp_path, boom, max_retries=1)
    assert g.generate("doc", "chunk") == ""
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "temperature" in err


def test_save_cache_atomic_writes_and_leaves_no_temp(tmp_path):
    import json as _json

    path = tmp_path / "cache.json"
    g = ContextGenerator(lambda p: "ctx", cache_path=path)
    g.generate("D", "C")
    # cache content round-trips and no temp file is left behind (a fixed-name
    # ".tmp" would race across concurrent ingest processes).
    assert "ctx" in _json.loads(path.read_text()).values()
    assert list(tmp_path.glob("*.tmp")) == []
