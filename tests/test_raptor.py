from __future__ import annotations

import numpy as np

from ragpipe.raptor import RaptorNode, build_raptor_tree, cluster_embeddings


def test_cluster_separates_two_obvious_groups():
    va = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    vb = [[10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]
    labels = cluster_embeddings(va + vb, max_clusters=4, random_state=0)
    # BIC may sub-split a tight blob, so we don't require each blob to be a single
    # cluster — only that no cluster mixes a point from va with one from vb.
    assert set(labels[:3]).isdisjoint(set(labels[3:]))


def test_cluster_handles_tiny_input():
    labels = cluster_embeddings([[1.0, 2.0]], max_clusters=4, random_state=0)
    assert labels == [0]


def test_cluster_separates_high_dim_groups():
    # Real embeddings are ~1536-dim; full-covariance GMM/BIC directly on that is
    # intractable (ADR-0013 calls for GMM/UMAP). The PCA reduction must keep
    # clustering both correct and tractable on high-dim input.
    rng = np.random.default_rng(0)
    dim = 256
    a = rng.standard_normal((20, dim)).tolist()
    b = (rng.standard_normal((20, dim)) + 8.0).tolist()
    labels = cluster_embeddings(a + b, max_clusters=8, random_state=0)
    assert set(labels[:20]).isdisjoint(set(labels[20:]))


def test_build_tree_produces_higher_level_nodes_and_terminates():
    leaves = [
        RaptorNode(id=f"L{i}", text=t, level=0, embedding=e)
        for i, (t, e) in enumerate([
            ("alpha", [0.0, 0.0]), ("alpha2", [0.1, 0.0]), ("alpha3", [0.0, 0.1]),
            ("beta", [9.0, 9.0]), ("beta2", [9.1, 9.0]), ("beta3", [9.0, 9.1]),
        ])
    ]

    def fake_summarize(texts: list[str]) -> str:
        return "summary:" + "|".join(texts)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]

    summaries = build_raptor_tree(
        leaves, summarize_fn=fake_summarize, embed_batch_fn=fake_embed,
        max_levels=3, max_clusters=4, random_state=0,
    )
    assert summaries, "expected at least one summary node"
    assert all(s.level >= 1 for s in summaries)
    assert len(summaries) < len(leaves) * 3
    assert any(s.text.startswith("summary:") for s in summaries)
