"""Top-K content-based document recommendation (with an optional cluster-aware hybrid).

Content-based score  = cosine(query, document) on unit-length TF-IDF vectors.
Hybrid score         = (1 - w) * cosine + w * [document is in the query's K-Means cluster]
"""
from __future__ import annotations

import numpy as np


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in part[np.argsort(-scores[part])]]


def score_documents(X: np.ndarray, query_vec: np.ndarray, clusters: np.ndarray | None = None,
                    query_cluster: int | None = None, cluster_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (final scores, plain cosine scores) for every document."""
    cosine = (X @ query_vec).astype(np.float64)
    if clusters is None or query_cluster is None or cluster_weight <= 0:
        return cosine, cosine
    bonus = (clusters == query_cluster).astype(np.float64)
    return (1 - cluster_weight) * cosine + cluster_weight * bonus, cosine


def recommend(X: np.ndarray, query_vec: np.ndarray, k: int, exclude: set[int] | None = None,
              clusters: np.ndarray | None = None, query_cluster: int | None = None,
              cluster_weight: float = 0.0) -> list[tuple[int, float, float]]:
    """Top-k (doc position, final score, cosine) excluding positions in `exclude`."""
    final, cosine = score_documents(X, query_vec, clusters, query_cluster, cluster_weight)
    final = final.copy()
    for i in exclude or ():
        final[i] = -np.inf
    return [(i, float(final[i]), float(cosine[i])) for i in _top_k(final, k)]


def shared_terms(query_vec: np.ndarray, doc_vec: np.ndarray, vocab: list[str], n: int = 5) -> list[str]:
    """Terms contributing most to the cosine score (largest products of weights)."""
    contrib = query_vec * doc_vec
    idx = np.argsort(-contrib)[:n]
    return [vocab[i] for i in idx if contrib[i] > 0]


def precision_at_k(X: np.ndarray, labels: list[str], k: int, clusters: np.ndarray | None = None,
                   cluster_weight: float = 0.0, dup_threshold: float = 0.98) -> dict:
    """Use every document as a query; relevant = same category label.

    Also reports how often the top-k list contains a (near-)duplicate of the query,
    which shows how duplicates crowd out genuinely related documents.
    """
    sims = (X @ X.T).astype(np.float64)
    np.fill_diagonal(sims, -np.inf)
    scores = sims
    if clusters is not None and cluster_weight > 0:
        same = (clusters[:, None] == clusters[None, :]).astype(np.float64)
        scores = (1 - cluster_weight) * sims + cluster_weight * same
        np.fill_diagonal(scores, -np.inf)
    lab = np.array(labels)
    k = min(k, len(labels) - 1)
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    relevant = (lab[top] == lab[:, None]).mean(axis=1)
    has_dup = (np.take_along_axis(sims, top, axis=1) >= dup_threshold).any(axis=1)
    return {"k": k, "precision_at_k": float(relevant.mean()),
            "queries_with_duplicate_in_top_k": float(has_dup.mean())}
