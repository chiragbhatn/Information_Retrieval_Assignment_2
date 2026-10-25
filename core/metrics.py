"""Evaluation metrics written from scratch.

Classification: accuracy, precision, recall, F1 (per class, macro, weighted), confusion matrix.
Clustering:     Purity and Rand Index (pair counting).
"""
from __future__ import annotations

from collections import Counter
from math import comb

import pandas as pd


# ---------------------------------------------------------------- classification
def confusion_matrix(y_true: list, y_pred: list, labels: list) -> pd.DataFrame:
    pos = {l: i for i, l in enumerate(labels)}
    grid = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        grid[pos[t]][pos[p]] += 1
    return pd.DataFrame(grid, index=[f"actual: {l}" for l in labels],
                        columns=[f"pred: {l}" for l in labels])


def accuracy(y_true: list, y_pred: list) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def classification_report(y_true: list, y_pred: list, labels: list) -> pd.DataFrame:
    rows = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": label, "precision": precision, "recall": recall, "f1": f1,
                     "support": tp + fn})
    df = pd.DataFrame(rows)
    total = df["support"].sum()
    macro = {"class": "macro avg", **{m: df[m].mean() for m in ("precision", "recall", "f1")},
             "support": total}
    weighted = {"class": "weighted avg",
                **{m: (df[m] * df["support"]).sum() / total if total else 0.0
                   for m in ("precision", "recall", "f1")}, "support": total}
    return pd.concat([df, pd.DataFrame([macro, weighted])], ignore_index=True)


def summary_scores(y_true: list, y_pred: list, labels: list) -> dict:
    rep = classification_report(y_true, y_pred, labels).set_index("class")
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": rep.loc["macro avg", "precision"],
        "macro_recall": rep.loc["macro avg", "recall"],
        "macro_f1": rep.loc["macro avg", "f1"],
        "weighted_f1": rep.loc["weighted avg", "f1"],
    }


# ---------------------------------------------------------------- clustering
def contingency_table(true_labels: list, clusters: list) -> pd.DataFrame:
    """Rows = clusters, columns = true classes, cells = number of documents."""
    counts = Counter(zip(clusters, true_labels))
    cl = sorted(set(clusters))
    cls = sorted(set(true_labels))
    return pd.DataFrame([[counts.get((c, t), 0) for t in cls] for c in cl],
                        index=[f"cluster {c}" for c in cl], columns=cls)


def purity(true_labels: list, clusters: list) -> float:
    """(1/N) * sum over clusters of the size of the most common true class in it."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    per_cluster: dict = {}
    for c, t in zip(clusters, true_labels):
        per_cluster.setdefault(c, Counter())[t] += 1
    return sum(cnt.most_common(1)[0][1] for cnt in per_cluster.values()) / n


def rand_index(true_labels: list, clusters: list) -> dict:
    """Rand Index = (TP + TN) / total pairs.

    TP: pair in same cluster and same class     FP: same cluster, different class
    FN: different cluster, same class           TN: different cluster, different class
    Counted from the contingency table, which gives exactly the same numbers as
    checking all N(N-1)/2 pairs one by one, but in O(N) time.
    """
    n = len(true_labels)
    total = comb(n, 2)
    pair_counts = Counter(zip(clusters, true_labels))
    tp = sum(comb(v, 2) for v in pair_counts.values())
    same_cluster = sum(comb(v, 2) for v in Counter(clusters).values())
    same_class = sum(comb(v, 2) for v in Counter(true_labels).values())
    fp = same_cluster - tp
    fn = same_class - tp
    tn = total - tp - fp - fn
    ri = (tp + tn) / total if total else 0.0
    return {"rand_index": ri, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "total_pairs": total}


def rand_index_bruteforce(true_labels: list, clusters: list) -> float:
    """Direct pair-by-pair definition; used in tests to check rand_index()."""
    n = len(true_labels)
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            agree += (clusters[i] == clusters[j]) == (true_labels[i] == true_labels[j])
    return agree / comb(n, 2) if n > 1 else 0.0


def majority_mapping(true_labels: list, clusters: list) -> dict:
    """Name each cluster after its most common true class."""
    per_cluster: dict = {}
    for c, t in zip(clusters, true_labels):
        per_cluster.setdefault(c, Counter())[t] += 1
    return {c: cnt.most_common(1)[0][0] for c, cnt in per_cluster.items()}
