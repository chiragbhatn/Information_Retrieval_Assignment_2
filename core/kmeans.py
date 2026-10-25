"""K-Means clustering written from scratch, recording every step for the UI.

1. Initialisation : pick k documents as starting centroids (random or k-means++)
2. Assignment     : each document goes to the nearest centroid (squared Euclidean)
3. Update         : each centroid becomes the mean of its documents
4. Stop           : when no document changes cluster, centroids move less than
                    `tol`, or `max_iter` is reached

K-Means only finds a local optimum, so it is restarted `n_init` times with
different seeds and the run with the lowest SSE is kept.

On unit-length TF-IDF vectors, Euclidean distance ranks the same as cosine
distance: ||x - c||^2 = 2 - 2 cos(x, c) when ||c|| = 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    sse: float
    n_iter: int
    stop_reason: str
    init_indices: list[int]
    history: list[dict] = field(default_factory=list)
    seed: int = 0
    all_runs: list[dict] = field(default_factory=list)


def _sq_distances(X: np.ndarray, x_sq: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance of every row of X to every row of C.

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 with ||x||^2 precomputed.
    """
    d = x_sq[:, None] - 2.0 * (X @ C.T) + (C ** 2).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)


def _init_centroids(X: np.ndarray, x_sq: np.ndarray, k: int, method: str,
                    rng: np.random.Generator) -> list[int]:
    n = X.shape[0]
    if method == "random":
        return [int(i) for i in rng.choice(n, size=k, replace=False)]
    # k-means++: next centroid chosen with probability proportional to D(x)^2
    chosen = [int(rng.integers(n))]
    closest = _sq_distances(X, x_sq, X[chosen])[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:
            remaining = [i for i in range(n) if i not in chosen]
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(n, p=closest / total))
        chosen.append(nxt)
        closest = np.minimum(closest, _sq_distances(X, x_sq, X[[nxt]])[:, 0])
    return chosen


def _single_run(X: np.ndarray, x_sq: np.ndarray, k: int, init: str, max_iter: int, tol: float,
                seed: int) -> KMeansResult:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    init_idx = _init_centroids(X, x_sq, k, init, rng)
    centroids = X[init_idx].copy()
    labels = np.full(n, -1)
    history: list[dict] = []
    stop_reason = f"reached max_iter ({max_iter})"
    n_iter = 0
    rows = np.arange(n)
    for it in range(1, max_iter + 1):
        n_iter = it
        # Assignment step
        dist = _sq_distances(X, x_sq, centroids)
        new_labels = dist.argmin(axis=1)
        point_dist = dist[rows, new_labels]
        sse = float(point_dist.sum())
        changed = int((new_labels != labels).sum())
        labels = new_labels
        # Update step: centroid = mean of member documents (sum via one-hot matrix product)
        sizes = np.bincount(labels, minlength=k)
        one_hot = np.zeros((k, n))
        one_hot[labels, rows] = 1.0
        sums = one_hot @ X
        new_centroids = centroids.copy()
        for j in range(k):
            if sizes[j] > 0:
                new_centroids[j] = sums[j] / sizes[j]
            else:
                # Empty cluster: restart it at the document farthest from its centroid.
                new_centroids[j] = X[int(point_dist.argmax())]
        shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
        centroids = new_centroids
        history.append({"iteration": it, "sse": round(sse, 4), "reassigned": changed,
                        "max_centroid_shift": round(shift, 6), "cluster_sizes": sizes.tolist()})
        if it > 1 and changed == 0:
            stop_reason = f"no document changed cluster (iteration {it})"
            break
        if shift < tol:
            stop_reason = f"centroid shift {shift:.2e} < tol {tol:g} (iteration {it})"
            break
    final_sse = float(_sq_distances(X, x_sq, centroids)[rows, labels].sum())
    return KMeansResult(labels, centroids, final_sse, n_iter, stop_reason, init_idx, history, seed)


def kmeans(X: np.ndarray, k: int, init: str = "k-means++", max_iter: int = 100,
           tol: float = 1e-4, seed: int = 42, n_init: int = 10) -> KMeansResult:
    """Run K-Means n_init times with seeds seed, seed+1, ... and keep the lowest-SSE run."""
    if not 1 <= k <= X.shape[0]:
        raise ValueError("k must be between 1 and the number of documents")
    Xd = np.asarray(X, dtype=np.float64)
    x_sq = (Xd ** 2).sum(axis=1)
    runs = [_single_run(Xd, x_sq, k, init, max_iter, tol, seed + r) for r in range(n_init)]
    best = min(runs, key=lambda r: r.sse)
    best.all_runs = [{"seed": r.seed, "sse": round(r.sse, 4), "iterations": r.n_iter,
                      "selected": r is best} for r in runs]
    return best
