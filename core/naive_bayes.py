"""Multinomial Naive Bayes written from scratch.

    P(c | d)  is proportional to  P(c) * prod_t P(t | c) ^ count(t, d)
    P(c)      = N_c / N
    P(t | c)  = (count(t, c) + alpha) / (sum_t' count(t', c) + alpha * |V|)     (Laplace smoothing)

Computed in log space to avoid underflow.
"""
from __future__ import annotations

import math
import random

import numpy as np

SparseDoc = dict[int, float]  # feature column -> value (term count or tf-idf weight)


def stratified_split(labels: list[str], test_size: float = 0.2, seed: int = 42,
                     groups: list[int] | None = None) -> tuple[list[int], list[int]]:
    """Split positions into train/test keeping class proportions.

    If `groups` is given, documents in the same group (e.g. duplicates) always
    land on the same side, which prevents train/test leakage.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {}
    group_of = groups if groups is not None else list(range(len(labels)))
    seen_groups: dict[int, list[int]] = {}
    for pos, g in enumerate(group_of):
        seen_groups.setdefault(g, []).append(pos)
    for g, members in seen_groups.items():
        by_class.setdefault(labels[members[0]], []).append(g)
    train, test = [], []
    for label in sorted(by_class):
        grp_ids = by_class[label]
        rng.shuffle(grp_ids)
        # At least one test and one training group per class; a class with a
        # single group stays in training so the model can still learn it.
        n = len(grp_ids)
        n_test = 0 if n == 1 else min(max(1, round(n * test_size)), n - 1)
        for i, g in enumerate(grp_ids):
            (test if i < n_test else train).extend(seen_groups[g])
    return sorted(train), sorted(test)


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, docs: list[SparseDoc], labels: list[str], n_features: int) -> "MultinomialNaiveBayes":
        self.classes_ = sorted(set(labels))
        cls_pos = {c: i for i, c in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        self.class_count_ = np.zeros(n_classes)
        self.feature_count_ = np.zeros((n_classes, n_features))
        for doc, label in zip(docs, labels):
            c = cls_pos[label]
            self.class_count_[c] += 1
            for col, value in doc.items():
                self.feature_count_[c, col] += value
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        smoothed = self.feature_count_ + self.alpha
        denom = self.feature_count_.sum(axis=1, keepdims=True) + self.alpha * n_features
        self.feature_log_prob_ = np.log(smoothed / denom)
        self.n_features_ = n_features
        return self

    def joint_log_likelihood(self, docs: list[SparseDoc]) -> np.ndarray:
        scores = np.tile(self.class_log_prior_, (len(docs), 1))
        for i, doc in enumerate(docs):
            if not doc:
                continue
            cols = np.fromiter(doc.keys(), dtype=np.int64)
            vals = np.fromiter(doc.values(), dtype=np.float64)
            scores[i] += self.feature_log_prob_[:, cols] @ vals
        return scores

    def predict_proba(self, docs: list[SparseDoc]) -> np.ndarray:
        jll = self.joint_log_likelihood(docs)
        # log-sum-exp trick for a stable softmax
        top = jll.max(axis=1, keepdims=True)
        probs = np.exp(jll - top)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, docs: list[SparseDoc]) -> list[str]:
        jll = self.joint_log_likelihood(docs)
        return [self.classes_[i] for i in jll.argmax(axis=1)]

    def top_terms(self, vocab: list[str], n: int = 10) -> dict[str, list[tuple[str, float]]]:
        """Most indicative terms per class: log P(t|c) minus mean log P(t|other classes)."""
        out = {}
        for c, name in enumerate(self.classes_):
            others = np.delete(self.feature_log_prob_, c, axis=0).mean(axis=0) if len(self.classes_) > 1 else 0
            score = self.feature_log_prob_[c] - others
            idx = np.argsort(-score)[:n]
            out[name] = [(vocab[i], float(score[i])) for i in idx]
        return out

    def describe(self) -> dict:
        return {
            "classes": self.classes_,
            "priors": {c: round(float(math.exp(p)), 4) for c, p in zip(self.classes_, self.class_log_prior_)},
            "alpha": self.alpha,
            "vocabulary_size": self.n_features_,
        }
