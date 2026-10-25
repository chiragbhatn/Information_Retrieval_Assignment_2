"""Inverted index, TF-IDF weighting with length normalization, cosine similarity.

Written from scratch (no TfidfVectorizer). Formulas:
    tf weight   w_tf  = 1 + log10(tf)   (sublinear)   or   tf   (raw)
    idf         idf_t = log10(N / df_t)
    weight      w_t,d = w_tf * idf_t
    length norm w_t,d / sqrt(sum_t w_t,d^2)      (unit-length / cosine normalization)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

InvertedIndex = dict[str, dict[int, int]]  # term -> {doc position: term frequency}


def build_inverted_index(token_lists: list[list[str]]) -> InvertedIndex:
    index: InvertedIndex = {}
    for doc, tokens in enumerate(token_lists):
        for term in tokens:
            postings = index.get(term)
            if postings is None:
                postings = index[term] = {}
            postings[doc] = postings.get(doc, 0) + 1
    return index


@dataclass(frozen=True)
class TfidfConfig:
    min_df: int = 2               # drop terms in fewer than min_df documents
    max_df_ratio: float = 0.9     # drop terms in more than this share of documents
    max_features: int = 5000      # keep the N terms with highest document frequency (0 = no cap)
    sublinear_tf: bool = True     # 1 + log10(tf) instead of raw tf

    def as_dict(self) -> dict:
        return asdict(self)


class TfidfModel:
    """Fits vocabulary + idf on a corpus and produces unit-length document vectors."""

    def __init__(self, config: TfidfConfig = TfidfConfig()):
        self.config = config

    # ------------------------------------------------------------ fit
    def fit(self, token_lists: list[list[str]]) -> "TfidfModel":
        cfg = self.config
        n_docs = len(token_lists)
        full_index = build_inverted_index(token_lists)
        self.n_docs = n_docs
        self.full_vocab_size = len(full_index)
        self.full_df = {t: len(p) for t, p in full_index.items()}

        max_df = cfg.max_df_ratio * n_docs
        kept = [(t, df) for t, df in self.full_df.items() if cfg.min_df <= df <= max_df]
        kept.sort(key=lambda x: (-x[1], x[0]))
        self.vocab_after_df_filter = len(kept)
        if cfg.max_features and len(kept) > cfg.max_features:
            kept = kept[: cfg.max_features]
        self.vocab: list[str] = sorted(t for t, _ in kept)
        self.term_to_col = {t: i for i, t in enumerate(self.vocab)}
        self.index: InvertedIndex = {t: full_index[t] for t in self.vocab}
        self.df = np.array([len(self.index[t]) for t in self.vocab], dtype=np.int32)
        self.idf = np.array([math.log10(n_docs / df) for df in self.df], dtype=np.float64)

        # Weights are accumulated term-by-term from the postings lists.
        weights = np.zeros((n_docs, len(self.vocab)), dtype=np.float32)
        sq_len = [0.0] * n_docs
        for col, term in enumerate(self.vocab):
            idf = self.idf[col]
            for doc, tf in self.index[term].items():
                w = self.tf_weight(tf) * idf
                weights[doc, col] = w
                sq_len[doc] += w * w
        self.doc_norms = np.sqrt(np.array(sq_len, dtype=np.float64))
        safe = np.where(self.doc_norms > 0, self.doc_norms, 1.0)
        self.matrix = (weights / safe[:, None]).astype(np.float32)
        self.empty_docs = int((self.doc_norms == 0).sum())
        # Term counts per document (restricted to vocabulary) for Naive Bayes.
        self.doc_counts: list[dict[int, int]] = [dict() for _ in range(n_docs)]
        for col, term in enumerate(self.vocab):
            for doc, tf in self.index[term].items():
                self.doc_counts[doc][col] = tf
        return self

    def tf_weight(self, tf: float) -> float:
        if tf <= 0:
            return 0.0
        return 1.0 + math.log10(tf) if self.config.sublinear_tf else float(tf)

    # ------------------------------------------------------------ transform new text
    def transform(self, token_lists: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(token_lists), len(self.vocab)), dtype=np.float32)
        for row, tokens in enumerate(token_lists):
            counts: dict[int, int] = {}
            for t in tokens:
                col = self.term_to_col.get(t)
                if col is not None:
                    counts[col] = counts.get(col, 0) + 1
            for col, tf in counts.items():
                out[row, col] = self.tf_weight(tf) * self.idf[col]
            norm = math.sqrt(float((out[row] ** 2).sum()))
            if norm > 0:
                out[row] /= norm
        return out

    # ------------------------------------------------------------ inspection helpers
    def postings_table(self, term: str, doc_ids: list[str], titles: list[str]) -> pd.DataFrame:
        col = self.term_to_col.get(term)
        if col is None:
            return pd.DataFrame()
        rows = []
        for doc, tf in sorted(self.index[term].items(), key=lambda x: -x[1]):
            raw_w = self.tf_weight(tf) * self.idf[col]
            rows.append({
                "doc_id": doc_ids[doc], "title": titles[doc], "tf": tf,
                "tf_weight": round(self.tf_weight(tf), 4), "idf": round(float(self.idf[col]), 4),
                "tf_idf": round(raw_w, 4), "doc_length_norm": round(float(self.doc_norms[doc]), 4),
                "normalized_weight": round(float(self.matrix[doc, col]), 4),
            })
        return pd.DataFrame(rows)

    def top_terms_for_doc(self, doc: int, n: int = 10) -> list[tuple[str, float]]:
        row = self.matrix[doc]
        idx = np.argsort(-row)[:n]
        return [(self.vocab[i], float(row[i])) for i in idx if row[i] > 0]

    def summary(self) -> dict:
        nnz = int(sum(len(p) for p in self.index.values()))
        cells = self.n_docs * max(1, len(self.vocab))
        return {
            "documents": self.n_docs,
            "full_vocabulary": self.full_vocab_size,
            "after_df_filter": self.vocab_after_df_filter,
            "feature_vocabulary": len(self.vocab),
            "non_zero_weights": nnz,
            "sparsity_pct": round(100 * (1 - nnz / cells), 2),
            "empty_documents": self.empty_docs,
        }


# ---------------------------------------------------------------- cosine similarity
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    nu = math.sqrt(float(np.dot(u, u)))
    nv = math.sqrt(float(np.dot(v, v)))
    return dot / (nu * nv) if nu > 0 and nv > 0 else 0.0


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine: normalise rows to unit length, then take dot products."""
    def unit(m: np.ndarray) -> np.ndarray:
        norms = np.sqrt((m.astype(np.float64) ** 2).sum(axis=1))
        norms[norms == 0] = 1.0
        return m / norms[:, None]
    ua = unit(a)
    ub = ua if b is None else unit(b)
    return (ua @ ub.T).astype(np.float32)
