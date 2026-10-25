"""Experiments that produce the evidence for the report's inference questions.

Each function returns a DataFrame; the Streamlit Performance page and
scripts/run_experiments.py both call these.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .data_loader import Corpus
from .kmeans import kmeans
from .metrics import purity, rand_index
from .pipeline import (DedupConfig, FeatureSet, KMConfig, NBConfig, build_features, run_classification,
                       run_clustering)
from .preprocessing import PreprocessConfig
from .recommender import precision_at_k
from .tfidf import TfidfConfig

PREPROCESSING_VARIANTS: list[tuple[str, PreprocessConfig]] = [
    ("A. Minimal (lowercase + split only)",
     PreprocessConfig(fix_encoding=False, strip_markup=False, remove_numbers=False, remove_punctuation=False,
                      remove_stopwords=False, normalizer="none", min_token_len=1)),
    ("B. No lowercasing (cleaned, case kept)",
     PreprocessConfig(lowercase=False, remove_stopwords=False, normalizer="none")),
    ("C. Cleaned + lowercase", PreprocessConfig(remove_stopwords=False, normalizer="none")),
    ("D. + stop-word removal", PreprocessConfig(normalizer="none")),
    ("E. + Porter stemming", PreprocessConfig(normalizer="stem")),
    ("F. + WordNet lemmatization", PreprocessConfig(normalizer="lemmatize")),
    ("G. + lemmatization + bigrams", PreprocessConfig(normalizer="lemmatize", ngram_max=2)),
    ("H. + lemmatization + news stop-words", PreprocessConfig(normalizer="lemmatize", news_stopwords=True)),
]


def _row_metrics(fs: FeatureSet, nb_cfg: NBConfig, km_cfg: KMConfig, rec_k: int) -> dict:
    cr = run_classification(fs, nb_cfg)
    kr = run_clustering(fs, km_cfg)
    rec = precision_at_k(fs.X, fs.labels, rec_k)
    return {
        "nb_accuracy": cr.scores["accuracy"], "nb_macro_f1": cr.scores["macro_f1"],
        "test_docs_with_copy_in_train": cr.leakage,
        "km_purity": kr.purity, "km_rand_index": kr.rand["rand_index"], "km_sse": kr.km.sse,
        f"rec_precision_at_{rec_k}": rec["precision_at_k"],
        f"rec_dup_in_top_{rec_k}": rec["queries_with_duplicate_in_top_k"],
    }


def preprocessing_ablation(corpus: Corpus, tfidf_cfg: TfidfConfig = TfidfConfig(),
                           dedup: DedupConfig = DedupConfig(), nb_cfg: NBConfig = NBConfig(),
                           km_cfg: KMConfig = KMConfig(), rec_k: int = 5, progress=None) -> pd.DataFrame:
    rows = []
    for n, (name, prep) in enumerate(PREPROCESSING_VARIANTS):
        if progress:
            progress(n / len(PREPROCESSING_VARIANTS), name)
        t0 = time.perf_counter()
        fs = build_features(corpus, prep, tfidf_cfg, dedup)
        tokens_per_doc = float(np.mean([len(t) for t in fs.tokens]))
        row = {"config": name, "full_vocabulary": fs.model.full_vocab_size,
               "vocab_after_df_filter": fs.model.vocab_after_df_filter,
               "features_used": len(fs.model.vocab), "avg_tokens_per_doc": round(tokens_per_doc, 1)}
        row.update(_row_metrics(fs, nb_cfg, km_cfg, rec_k))
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_impact(corpus: Corpus, prep: PreprocessConfig = PreprocessConfig(),
                     tfidf_cfg: TfidfConfig = TfidfConfig(), nb_cfg: NBConfig = NBConfig(),
                     km_cfg: KMConfig = KMConfig(), rec_k: int = 5, near_threshold: float = 0.9,
                     progress=None) -> pd.DataFrame:
    scenarios = [
        ("1. Keep duplicates, random split", DedupConfig(remove_exact=False, near_threshold=near_threshold), False),
        ("2. Keep duplicates, duplicate-aware split", DedupConfig(remove_exact=False, near_threshold=near_threshold), True),
        ("3. Remove exact duplicates", DedupConfig(near_threshold=near_threshold), True),
        (f"4. Remove exact + near (cos >= {near_threshold})",
         DedupConfig(remove_near=True, near_threshold=near_threshold), True),
    ]
    rows = []
    for n, (name, dedup, grouped) in enumerate(scenarios):
        if progress:
            progress(n / len(scenarios), name)
        fs = build_features(corpus, prep, tfidf_cfg, dedup)
        row = {"scenario": name, "documents": len(fs),
               "removed_exact": len(fs.removed["exact"]), "removed_near": len(fs.removed["near"]),
               "near_dup_pairs_left": len(fs.near_pairs)}
        row.update(_row_metrics(fs, replace(nb_cfg, group_duplicates=grouped), km_cfg, rec_k))
        rows.append(row)
    return pd.DataFrame(rows)


def k_sweep(fs: FeatureSet, ks=range(2, 11), init: str = "k-means++", seed: int = 42, n_init: int = 10,
            progress=None) -> pd.DataFrame:
    rows = []
    ks = list(ks)
    for n, k in enumerate(ks):
        if progress:
            progress(n / len(ks), f"k = {k}")
        res = kmeans(fs.X, k, init=init, seed=seed, n_init=n_init)
        clusters = res.labels.tolist()
        rows.append({"k": k, "sse": res.sse, "purity": purity(fs.labels, clusters),
                     "rand_index": rand_index(fs.labels, clusters)["rand_index"], "iterations": res.n_iter})
    return pd.DataFrame(rows)


def init_comparison(fs: FeatureSet, k: int = 5, seeds=range(10)) -> pd.DataFrame:
    rows = []
    for init in ("random", "k-means++"):
        for s in seeds:
            res = kmeans(fs.X, k, init=init, seed=s, n_init=1)
            clusters = res.labels.tolist()
            rows.append({"init": init, "seed": s, "sse": res.sse, "iterations": res.n_iter,
                         "purity": purity(fs.labels, clusters),
                         "rand_index": rand_index(fs.labels, clusters)["rand_index"]})
    return pd.DataFrame(rows)


def recommender_eval(fs: FeatureSet, clusters: np.ndarray | None, ks=(1, 3, 5, 10),
                     weights=(0.0, 0.2, 0.5)) -> pd.DataFrame:
    rows = []
    for w in weights:
        if w > 0 and clusters is None:
            continue
        for k in ks:
            r = precision_at_k(fs.X, fs.labels, k, clusters, w)
            rows.append({"method": "content-based" if w == 0 else f"hybrid (cluster weight {w})", "k": k,
                         "precision_at_k": r["precision_at_k"],
                         "queries_with_duplicate_in_top_k": r["queries_with_duplicate_in_top_k"]})
    return pd.DataFrame(rows)


def nb_variants(fs: FeatureSet, base: NBConfig = NBConfig(), alphas=(0.01, 0.1, 0.5, 1.0, 2.0)) -> pd.DataFrame:
    rows = []
    for feats in ("counts", "tfidf"):
        for a in alphas:
            cr = run_classification(fs, replace(base, alpha=a, features=feats))
            rows.append({"features": feats, "alpha": a, "accuracy": cr.scores["accuracy"],
                         "macro_f1": cr.scores["macro_f1"]})
    return pd.DataFrame(rows)
