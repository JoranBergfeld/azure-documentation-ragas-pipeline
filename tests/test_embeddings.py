from ragpipe.embeddings import _embed_in_chunks


def test_embed_in_chunks_splits_preserves_order_and_covers_all():
    calls: list[int] = []

    def embed_one(sub):
        calls.append(len(sub))
        # echo each input's first char code so order is verifiable
        return [[float(ord(s[0]))] for s in sub]

    texts = [chr(ord("a") + i) for i in range(23)]  # a..w
    out = _embed_in_chunks(embed_one, texts, max_inputs=10)

    # 23 inputs -> sub-batches of 10, 10, 3 (none exceeds the cap)
    assert calls == [10, 10, 3]
    # every input embedded exactly once, in original order
    assert out == [[float(ord(t))] for t in texts]


def test_embed_in_chunks_handles_empty_and_exact_multiples():
    assert _embed_in_chunks(lambda sub: [[0.0] for _ in sub], [], max_inputs=5) == []

    sizes: list[int] = []

    def embed_one(sub):
        sizes.append(len(sub))
        return [[1.0] for _ in sub]

    out = _embed_in_chunks(embed_one, ["x"] * 10, max_inputs=5)
    assert sizes == [5, 5]
    assert len(out) == 10


def test_embed_in_chunks_clips_oversized_inputs_only():
    seen: list[str] = []

    def embed_one(sub):
        seen.extend(sub)
        return [[0.0] for _ in sub]

    _embed_in_chunks(embed_one, ["x" * 50, "short"], max_inputs=5, max_chars=10)
    assert seen == ["x" * 10, "short"]
