"""2-D projection for plotting only (PCA via randomized SVD, numpy).

Used solely to draw the cluster scatter plot; the clustering itself runs on the
full TF-IDF vectors.
"""
from __future__ import annotations

import numpy as np


def pca_2d(X: np.ndarray, seed: int = 0, oversample: int = 10, power_iters: int = 4) -> tuple[np.ndarray, list[float]]:
    Xc = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    Y = Xc @ rng.standard_normal((Xc.shape[1], 2 + oversample))
    for _ in range(power_iters):
        Y, _ = np.linalg.qr(Y)
        Y = Xc @ (Xc.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, s, vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    coords = Xc @ vt[:2].T
    total_var = float((Xc ** 2).sum()) or 1.0
    explained = [float(v ** 2 / total_var) for v in s[:2]]
    return coords, explained
