"""Session state + cached pipeline steps shared by all pages.

Every result is cached on (corpus, configuration), so changing a setting on
one page automatically recomputes the dependent steps on the others.
"""
from __future__ import annotations

import os

import numpy as np
import streamlit as st

from core.data_loader import Corpus, load_bundled
from core.pipeline import (ClassificationResult, ClusteringResult, DedupConfig, FeatureSet, KMConfig,
                           NBConfig, build_features, run_classification, run_clustering)
from core.preprocessing import PreprocessConfig
from core.recommender import precision_at_k
from core.reduction import pca_2d
from core.tfidf import TfidfConfig

DEFAULT_CONFIGS = {
    "prep": PreprocessConfig(),
    "tfidf": TfidfConfig(),
    "dedup": DedupConfig(),
    "nb": NBConfig(),
    "km": KMConfig(),
}


@st.cache_resource(show_spinner="Loading the BBC News corpus...")
def bundled_corpus() -> Corpus:
    return load_bundled()


def init_state() -> None:
    ss = st.session_state
    for name, value in DEFAULT_CONFIGS.items():
        ss.setdefault(f"cfg_{name}", value)
    if "corpus" not in ss:
        base = bundled_corpus()
        # IR_SAMPLE lets a laptop run on a smaller stratified sample, e.g. IR_SAMPLE=600
        n = int(os.environ.get("IR_SAMPLE", "0") or 0)
        ss.corpus = base.stratified_sample(n) if 0 < n < len(base) else base
        ss.corpus_source = "Bundled BBC News dataset (Hugging Face: gopalkalpande/bbc-news-summary)"


def corpus() -> Corpus:
    init_state()
    return st.session_state.corpus


def set_corpus(c: Corpus, source: str) -> None:
    st.session_state.corpus = c
    st.session_state.corpus_source = source


def cfg(name: str):
    init_state()
    return st.session_state[f"cfg_{name}"]


def set_cfg(name: str, value) -> None:
    st.session_state[f"cfg_{name}"] = value


# ---------------------------------------------------------------- cached computations
@st.cache_resource(show_spinner="Preprocessing text and building the TF-IDF index...", max_entries=8)
def _features(key: str, _corpus: Corpus, prep: PreprocessConfig, tfidf: TfidfConfig,
              dedup: DedupConfig) -> FeatureSet:
    return build_features(_corpus, prep, tfidf, dedup)


def features() -> FeatureSet:
    c = corpus()
    return _features(c.fingerprint, c, cfg("prep"), cfg("tfidf"), cfg("dedup"))


def _feature_key() -> tuple:
    return (corpus().fingerprint, cfg("prep"), cfg("tfidf"), cfg("dedup"))


@st.cache_resource(show_spinner="Training and testing Naive Bayes...", max_entries=8)
def _classification(key: tuple, nb: NBConfig, _fs: FeatureSet) -> ClassificationResult:
    return run_classification(_fs, nb)


def classification() -> ClassificationResult:
    res = _classification(_feature_key(), cfg("nb"), features())
    st.session_state.summary_nb = res.scores
    return res


@st.cache_resource(show_spinner="Running K-Means...", max_entries=8)
def _clustering(key: tuple, km: KMConfig, _fs: FeatureSet) -> ClusteringResult:
    return run_clustering(_fs, km)


def clustering() -> ClusteringResult:
    res = _clustering(_feature_key(), cfg("km"), features())
    st.session_state.summary_km = {"purity": res.purity, "rand_index": res.rand["rand_index"]}
    return res


@st.cache_resource(show_spinner="Projecting documents to 2-D...", max_entries=8)
def _projection(key: tuple, _fs: FeatureSet) -> tuple[np.ndarray, list[float]]:
    return pca_2d(_fs.X)


def projection() -> tuple[np.ndarray, list[float]]:
    return _projection(_feature_key(), features())


@st.cache_resource(show_spinner="Evaluating recommendations...", max_entries=16)
def _rec_eval(key: tuple, k: int, km: KMConfig | None, weight: float, _fs: FeatureSet, _clusters) -> dict:
    return precision_at_k(_fs.X, _fs.labels, k, _clusters, weight)


def recommendation_quality(k: int = 5, weight: float = 0.0) -> dict:
    fs = features()
    clusters = clustering().km.labels if weight > 0 else None
    return _rec_eval(_feature_key(), k, cfg("km") if weight > 0 else None, weight, fs, clusters)
