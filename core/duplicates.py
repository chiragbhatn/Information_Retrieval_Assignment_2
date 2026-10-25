"""Exact and near-duplicate document detection."""
from __future__ import annotations

import hashlib
import re

import numpy as np

_NON_WORD = re.compile(r"\W+")


def text_fingerprint(text: str) -> str:
    """Hash of the text after lowercasing and dropping punctuation/whitespace differences."""
    norm = _NON_WORD.sub(" ", text.replace("�", " ").lower()).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def exact_duplicate_groups(texts: list[str]) -> list[list[int]]:
    """Groups (size >= 2) of positions whose texts are identical after normalisation."""
    buckets: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        buckets.setdefault(text_fingerprint(t), []).append(i)
    return [g for g in buckets.values() if len(g) > 1]


def near_duplicate_pairs(X: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """All pairs i < j whose cosine similarity is >= threshold (rows of X are unit length)."""
    sims = X @ X.T
    iu, ju = np.where(np.triu(sims, k=1) >= threshold)
    return [(int(i), int(j), float(sims[i, j])) for i, j in zip(iu, ju)]


def group_ids(n: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Union-find: documents linked by any pair share one group id."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]
