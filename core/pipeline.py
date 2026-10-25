"""End-to-end pipeline shared by the Streamlit app and the experiments script."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data_loader import Corpus
from .duplicates import exact_duplicate_groups, group_ids, near_duplicate_pairs, text_fingerprint
from .kmeans import KMeansResult, kmeans
from .metrics import (classification_report, confusion_matrix, contingency_table, majority_mapping,
                      purity, rand_index, summary_scores)
from .naive_bayes import MultinomialNaiveBayes, stratified_split
from .preprocessing import PreprocessConfig, preprocess_corpus
from .tfidf import TfidfConfig, TfidfModel


# ---------------------------------------------------------------- configs
@dataclass(frozen=True)
class DedupConfig:
    remove_exact: bool = True
    remove_near: bool = False
    near_threshold: float = 0.90


@dataclass(frozen=True)
class NBConfig:
    test_size: float = 0.2
    seed: int = 42
    alpha: float = 1.0
    features: str = "counts"          # "counts" (classic multinomial) or "tfidf"
    group_duplicates: bool = True     # keep duplicate copies on the same side of the split


@dataclass(frozen=True)
class KMConfig:
    k: int = 5
    init: str = "k-means++"           # or "random"
    max_iter: int = 100
    tol: float = 1e-4
    seed: int = 42
    n_init: int = 10


# ---------------------------------------------------------------- features
@dataclass
class FeatureSet:
    doc_ids: list[str]
    titles: list[str]
    labels: list[str]
    bodies: list[str]
    tokens: list[list[str]]
    model: TfidfModel
    prep: PreprocessConfig
    tfidf_cfg: TfidfConfig
    dedup: DedupConfig
    input_size: int
    exact_groups: list[list[str]]          # doc_id groups found in the input corpus
    near_pairs: pd.DataFrame               # near-duplicate pairs among kept documents
    removed: dict[str, list[str]]
    dup_group: list[int]                   # group id per kept doc (duplicates share one)
    timings: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.model.matrix

    def __len__(self) -> int:
        return len(self.doc_ids)


def _near_pairs_frame(pairs, ids, titles, labels) -> pd.DataFrame:
    rows = [{"doc_a": ids[i], "doc_b": ids[j], "similarity": round(s, 4),
             "title_a": titles[i], "title_b": titles[j], "label_a": labels[i], "label_b": labels[j],
             "same_label": labels[i] == labels[j]} for i, j, s in pairs]
    cols = ["doc_a", "doc_b", "similarity", "title_a", "title_b", "label_a", "label_b", "same_label"]
    return pd.DataFrame(rows, columns=cols).sort_values("similarity", ascending=False, ignore_index=True)


def build_features(corpus: Corpus, prep: PreprocessConfig, tfidf_cfg: TfidfConfig,
                   dedup: DedupConfig) -> FeatureSet:
    timings = {}
    t0 = time.perf_counter()
    ids, titles, labels = corpus.doc_ids, corpus.titles, corpus.labels
    bodies = corpus.texts.tolist()

    # 1. Exact duplicates on the raw text
    groups = exact_duplicate_groups(bodies)
    exact_groups = [[ids[i] for i in g] for g in groups]
    drop = {i for g in groups for i in g[1:]} if dedup.remove_exact else set()
    keep = [i for i in range(len(ids)) if i not in drop]
    removed = {"exact": [ids[i] for i in sorted(drop)], "near": []}
    timings["deduplicate_exact"] = time.perf_counter() - t0

    # 2. Preprocess + TF-IDF
    t1 = time.perf_counter()
    k_titles = [titles[i] for i in keep]
    k_bodies = [bodies[i] for i in keep]
    tokens = preprocess_corpus(k_titles, k_bodies, prep)
    timings["preprocess"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    model = TfidfModel(tfidf_cfg).fit(tokens)
    timings["tfidf"] = time.perf_counter() - t2
    if not model.vocab:
        raise ValueError(f"No terms are left after preprocessing and the document-frequency filter "
                         f"({len(tokens)} documents, min_df={tfidf_cfg.min_df}, "
                         f"max_df={tfidf_cfg.max_df_ratio}). Lower min_df or load a larger corpus.")

    # 3. Near duplicates on TF-IDF cosine similarity
    t3 = time.perf_counter()
    k_ids = [ids[i] for i in keep]
    k_labels = [labels[i] for i in keep]
    pairs = near_duplicate_pairs(model.matrix, dedup.near_threshold)
    if dedup.remove_near and pairs:
        grp = group_ids(len(keep), [(a, b) for a, b, _ in pairs])
        first_of_group: dict[int, int] = {}
        survivors = []
        for pos, g in enumerate(grp):
            if g in first_of_group:
                removed["near"].append(k_ids[pos])
            else:
                first_of_group[g] = pos
                survivors.append(pos)
        keep = [keep[p] for p in survivors]
        tokens = [tokens[p] for p in survivors]
        k_titles = [k_titles[p] for p in survivors]
        k_bodies = [k_bodies[p] for p in survivors]
        k_ids = [k_ids[p] for p in survivors]
        k_labels = [k_labels[p] for p in survivors]
        model = TfidfModel(tfidf_cfg).fit(tokens)
        pairs = near_duplicate_pairs(model.matrix, dedup.near_threshold)
    near_df = _near_pairs_frame(pairs, k_ids, k_titles, k_labels)
    # Duplicate copies that are still present share a group, so the train/test split
    # can keep them together.
    fp_group: dict[str, int] = {}
    dup_group = [fp_group.setdefault(text_fingerprint(b), len(fp_group)) for b in k_bodies]
    timings["deduplicate_near"] = time.perf_counter() - t3

    return FeatureSet(k_ids, k_titles, k_labels, k_bodies, tokens, model, prep, tfidf_cfg, dedup,
                      len(ids), exact_groups, near_df, removed, dup_group, timings)


# ---------------------------------------------------------------- classification
@dataclass
class ClassificationResult:
    config: NBConfig
    train_idx: list[int]
    test_idx: list[int]
    y_true: list[str]
    y_pred: list[str]
    proba: np.ndarray
    classes: list[str]
    report: pd.DataFrame
    confusion: pd.DataFrame
    scores: dict
    model: MultinomialNaiveBayes
    leakage: int
    top_terms: dict
    seconds: float


def _tfidf_rows(X: np.ndarray, rows: list[int]) -> list[dict[int, float]]:
    out = []
    for r in rows:
        nz = np.flatnonzero(X[r])
        out.append({int(j): float(X[r, j]) for j in nz})
    return out


def run_classification(fs: FeatureSet, cfg: NBConfig) -> ClassificationResult:
    t0 = time.perf_counter()
    groups = fs.dup_group if cfg.group_duplicates else None
    train, test = stratified_split(fs.labels, cfg.test_size, cfg.seed, groups)
    if cfg.features == "tfidf":
        feats_train, feats_test = _tfidf_rows(fs.X, train), _tfidf_rows(fs.X, test)
    else:
        feats_train = [fs.model.doc_counts[i] for i in train]
        feats_test = [fs.model.doc_counts[i] for i in test]
    y_train = [fs.labels[i] for i in train]
    y_true = [fs.labels[i] for i in test]
    nb = MultinomialNaiveBayes(cfg.alpha).fit(feats_train, y_train, len(fs.model.vocab))
    proba = nb.predict_proba(feats_test)
    y_pred = [nb.classes_[i] for i in proba.argmax(axis=1)]
    classes = sorted(set(fs.labels))
    train_prints = {text_fingerprint(fs.bodies[i]) for i in train}
    leakage = sum(text_fingerprint(fs.bodies[i]) in train_prints for i in test)
    return ClassificationResult(
        cfg, train, test, y_true, y_pred, proba, nb.classes_,
        classification_report(y_true, y_pred, classes), confusion_matrix(y_true, y_pred, classes),
        summary_scores(y_true, y_pred, classes), nb, leakage, nb.top_terms(fs.model.vocab, 12),
        time.perf_counter() - t0)


# ---------------------------------------------------------------- clustering
@dataclass
class ClusteringResult:
    config: KMConfig
    km: KMeansResult
    purity: float
    rand: dict
    contingency: pd.DataFrame
    mapping: dict
    mapped_accuracy: float
    top_terms: dict
    seconds: float


def cluster_top_terms(centroids: np.ndarray, vocab: list[str], n: int = 12) -> dict[int, list[tuple[str, float]]]:
    out = {}
    for c, row in enumerate(centroids):
        idx = np.argsort(-row)[:n]
        out[c] = [(vocab[i], float(row[i])) for i in idx]
    return out


def run_clustering(fs: FeatureSet, cfg: KMConfig) -> ClusteringResult:
    t0 = time.perf_counter()
    km = kmeans(fs.X, cfg.k, cfg.init, cfg.max_iter, cfg.tol, cfg.seed, cfg.n_init)
    clusters = km.labels.tolist()
    mapping = majority_mapping(fs.labels, clusters)
    mapped_acc = float(np.mean([mapping[c] == t for c, t in zip(clusters, fs.labels)]))
    return ClusteringResult(cfg, km, purity(fs.labels, clusters), rand_index(fs.labels, clusters),
                            contingency_table(fs.labels, clusters), mapping, mapped_acc,
                            cluster_top_terms(km.centroids, fs.model.vocab), time.perf_counter() - t0)


def config_snapshot(**cfgs) -> dict:
    return {name: asdict(c) for name, c in cfgs.items()}
