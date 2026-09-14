from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Character budget for one summary prompt's passages. A cluster over a
# full-corpus ingest can hold hundreds of chunks: joined whole, the two largest
# came to ~172k tokens against gpt-4o's 128k window, failed every attempt and
# fell back to a 500-char slice. 250k chars stays under 128k tokens even at a
# code-heavy ~2.5 chars/token, with room left for the prompt and the reply.
SUMMARY_PASSAGE_CHAR_BUDGET = 250_000


def pack_passages(texts: list[str], max_chars: int = SUMMARY_PASSAGE_CHAR_BUDGET) -> list[str]:
    """Longest prefix of ``texts`` that fits ``max_chars`` (separators counted).

    Always returns at least one passage — truncated if it alone is over budget —
    so a summary is never asked of nothing. Cluster members are already
    semantically grouped, so a prefix is a representative sample of the cluster.
    """
    sep = len("\n\n---\n\n")
    packed: list[str] = []
    used = 0
    for text in texts:
        cost = len(text) + (sep if packed else 0)
        if used + cost > max_chars:
            if not packed:
                packed.append(text[:max_chars])
            break
        packed.append(text)
        used += cost
    return packed


def cluster_embeddings(
    vectors: list[list[float]],
    *,
    max_clusters: int = 50,
    random_state: int = 0,
    pca_dims: int = 50,
) -> list[int]:
    """Hard-assign each vector to a cluster. Component count chosen by BIC over
    1..min(max_clusters, n-1) GaussianMixtures (RAPTOR uses GMM soft clustering;
    we take the argmax responsibility as a hard label). Degenerate inputs (<=2
    vectors) return a single cluster.

    Embeddings are first reduced with PCA (ADR-0013: "GMM/UMAP"). Full-covariance
    GMM/BIC directly on raw embedding dimensionality (e.g. 1536) is intractable --
    the per-component covariance is d*d, so the BIC sweep blows up at every tree
    level. PCA to a small dense space keeps clustering tractable, dependency-free
    (scikit-learn is already a dep), and deterministic via ``random_state``."""
    n = len(vectors)
    if n <= 2:
        return [0] * n
    x = np.asarray(vectors, dtype=float)
    n_components = min(pca_dims, n - 1, x.shape[1])
    if n_components < x.shape[1]:
        x = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    upper = min(max_clusters, n - 1)
    best_bic = float("inf")
    best_labels = [0] * n
    for k in range(1, upper + 1):
        gm = GaussianMixture(n_components=k, random_state=random_state)
        gm.fit(x)
        bic = gm.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_labels = gm.predict(x).tolist()
    return best_labels


@dataclass
class RaptorNode:
    id: str
    text: str
    level: int
    embedding: list[float]


def build_raptor_tree(
    leaves: list[RaptorNode],
    *,
    summarize_fn: Callable[[list[str]], str],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    max_levels: int = 3,
    max_clusters: int = 50,
    random_state: int = 0,
) -> list[RaptorNode]:
    """Build RAPTOR summary nodes above the leaves. Returns ONLY the summary
    nodes (levels >= 1); the caller already has the leaves. Stops when a level
    yields a single cluster or max_levels is reached."""
    summaries: list[RaptorNode] = []
    current = leaves
    level = 1
    while level <= max_levels and len(current) > 1:
        labels = cluster_embeddings(
            [n.embedding for n in current], max_clusters=max_clusters, random_state=random_state
        )
        groups: dict[int, list[RaptorNode]] = {}
        for node, lab in zip(current, labels):
            groups.setdefault(lab, []).append(node)
        if len(groups) <= 1 and level > 1:
            break
        texts = [summarize_fn([n.text for n in group]) for group in groups.values()]
        embeddings = embed_batch_fn(texts)
        new_nodes = [
            RaptorNode(id=f"raptor-L{level}-{i}", text=t, level=level, embedding=e)
            for i, (t, e) in enumerate(zip(texts, embeddings))
        ]
        summaries.extend(new_nodes)
        current = new_nodes
        level += 1
    return summaries
